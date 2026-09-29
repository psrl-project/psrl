"""E2B data plane with provider-specific AgentEnv and Cube control planes."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol

import aiohttp

from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    SandboxBackend,
    SandboxCapabilities,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.metrics import SandboxMetrics, SandboxMetricsSnapshot


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _close_client(client: Any) -> None:
    """Best-effort close of provider transport state without killing the VM."""
    close = getattr(client, "close", None)
    if close is not None:
        try:
            await _await(close())
        except Exception:
            pass


def _resource_payload(spec: SandboxSpec) -> dict[str, int]:
    """Map portable resources to E2B-compatible control-plane fields."""
    payload: dict[str, int] = {}
    if spec.resources.cpu_count is not None:
        if not float(spec.resources.cpu_count).is_integer():
            raise ValueError("MicroVM backends require an integral cpu_count.")
        payload["cpuCount"] = int(spec.resources.cpu_count)
    if spec.resources.memory_mb is not None:
        payload["memoryMB"] = spec.resources.memory_mb
    if spec.resources.disk_mb is not None:
        payload["diskSizeMB"] = spec.resources.disk_mb
    return payload


def _metadata(spec: SandboxSpec) -> dict[str, str]:
    metadata = dict(spec.metadata)
    if spec.idempotency_key:
        metadata["psrl.idempotency_key"] = spec.idempotency_key
    return metadata


def _validate_provider_spec(spec: SandboxSpec) -> None:
    if spec.policy_profile is not None:
        raise RuntimeError("Provider backends do not accept Docker policy profiles.")


class E2BClientFactory(Protocol):
    """Minimal factory boundary around an E2B-compatible SDK."""

    async def create(self, spec: SandboxSpec) -> Any: ...

    async def connect(self, sandbox_id: str) -> Any: ...

    async def delete_snapshot(self, snapshot_id: str) -> None: ...

    async def close(self) -> None: ...


class E2BStateDriver(Protocol):
    """Provider extension point for native state operations."""

    @property
    def capabilities(self) -> SandboxCapabilities: ...

    async def pause(self, session: E2BSession, mode: PauseMode) -> None: ...

    async def resume(self, session: E2BSession) -> None: ...

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef: ...

    async def fork(self, session: E2BSession) -> SandboxSession: ...

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession: ...

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None: ...

    async def close(self) -> None: ...


class ProviderControlClient:
    """Persistent HTTP client for E2B-compatible provider control planes."""

    def __init__(
        self,
        api_url: str,
        *,
        api_key: str | None = None,
        request_timeout_s: float = 300.0,
        connection_limit: int = 128,
    ) -> None:
        if not api_url:
            raise ValueError("Provider api_url is required.")
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout_s = request_timeout_s
        self.connection_limit = connection_limit
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-API-Key": self.api_key} if self.api_key else {}
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_s),
                connector=aiohttp.TCPConnector(limit=self.connection_limit),
            )
        return self._session

    async def request(
        self,
        method: str,
        path: str,
        *,
        expected: tuple[int, ...],
        payload: Mapping[str, Any] | None = None,
    ) -> Any:
        session = await self._get_session()
        async with session.request(method, f"{self.api_url}{path}", json=payload) as response:
            if response.status not in expected:
                body = await response.text()
                raise RuntimeError(f"Provider returned HTTP {response.status} for {path!r}: {body.strip()}.")
            if response.status == 204:
                return None
            return await response.json()

    async def delete_idempotent(self, path: str) -> None:
        session = await self._get_session()
        async with session.delete(f"{self.api_url}{path}") as response:
            if response.status not in (204, 404):
                body = await response.text()
                raise RuntimeError(f"Provider returned HTTP {response.status} for {path!r}: {body.strip()}.")

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class E2BSDKClientFactory:
    """Lazy wrapper for the official `e2b` Python package."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        create_kwargs: Mapping[str, Any] | None = None,
        connect_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.create_kwargs = dict(create_kwargs or {})
        self.connect_kwargs = dict(connect_kwargs or {})

    @staticmethod
    def _sandbox_class():
        try:
            from e2b import AsyncSandbox
        except ImportError as exc:
            raise RuntimeError(
                "The E2B backend requires the optional e2b package. Install PSRL with sandbox-e2b support."
            ) from exc
        return AsyncSandbox

    def _sdk_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.api_key:
            options["api_key"] = self.api_key
        if self.api_url:
            options["api_url"] = self.api_url
        return options

    async def create(self, spec: SandboxSpec) -> Any:
        _validate_provider_spec(spec)
        if spec.source.kind != SandboxSourceKind.TEMPLATE:
            raise RuntimeError("E2B requires a template source; direct OCI image creation is unsupported.")
        if spec.resources != type(spec.resources)():
            raise RuntimeError(
                "E2B runtime resources are fixed by the template and cannot be overridden at create time."
            )
        sandbox_class = self._sandbox_class()
        kwargs = {**self.create_kwargs, **self._sdk_options()}
        kwargs["template"] = spec.source.reference
        metadata = _metadata(spec)
        if metadata:
            kwargs["metadata"] = metadata
        if spec.env:
            kwargs["envs"] = dict(spec.env)
        if spec.idle_timeout_s is not None:
            kwargs["timeout"] = max(1, math.ceil(spec.idle_timeout_s))
        return await _await(sandbox_class.create(**kwargs))

    async def connect(self, sandbox_id: str) -> Any:
        sandbox_class = self._sandbox_class()
        kwargs = {**self.connect_kwargs, **self._sdk_options()}
        return await _await(sandbox_class.connect(sandbox_id, **kwargs))

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete an E2B snapshot idempotently through the SDK control plane."""
        sandbox_class = self._sandbox_class()
        delete_snapshot = getattr(sandbox_class, "delete_snapshot", None)
        if delete_snapshot is None:
            raise RuntimeError("The installed e2b SDK does not support snapshot deletion; upgrade e2b.")
        await _await(delete_snapshot(snapshot_id, **self._sdk_options()))

    async def close(self) -> None:
        return None


class ProviderSDKClientFactory(E2BSDKClientFactory):
    """Create via a provider API, then attach the E2B data-plane SDK."""

    def __init__(self, *, control: ProviderControlClient, **kwargs: Any) -> None:
        super().__init__(api_key=control.api_key, api_url=control.api_url, **kwargs)
        self.control = control

    async def _connect_created(self, response: Mapping[str, Any]) -> Any:
        sandbox_id = response.get("sandboxID") or response.get("sandbox_id")
        if not sandbox_id:
            raise RuntimeError("Provider create response did not contain sandboxID.")
        return await self.connect(str(sandbox_id))

    async def close(self) -> None:
        await self.control.close()


class AgentEnvClientFactory(ProviderSDKClientFactory):
    """AgentEnv control-plane mapping for templates and cold OCI images."""

    async def create(self, spec: SandboxSpec) -> Any:
        _validate_provider_spec(spec)
        payload: dict[str, Any] = {}
        if spec.source.kind == SandboxSourceKind.IMAGE:
            path = "/sandboxes-cold"
            payload["image"] = spec.source.reference
            payload.update(_resource_payload(spec))
        elif spec.source.kind == SandboxSourceKind.TEMPLATE:
            if _resource_payload(spec):
                raise RuntimeError("AgentEnv template resources are fixed; use an image cold start to override them.")
            path = "/sandboxes"
            payload["templateID"] = spec.source.reference
        else:
            raise RuntimeError(f"AgentEnv does not support source kind {spec.source.kind.value!r}.")
        if spec.idle_timeout_s is not None:
            payload["timeout"] = max(1, math.ceil(spec.idle_timeout_s))
        if spec.env:
            payload["envVars"] = dict(spec.env)
        metadata = _metadata(spec)
        if metadata:
            payload["metadata"] = metadata
        payload.update(self.create_kwargs)
        response = await self.control.request("POST", path, expected=(201,), payload=payload)
        return await self._connect_created(response)


class CubeSandboxClientFactory(ProviderSDKClientFactory):
    """CubeSandbox control-plane mapping for template-backed microVMs."""

    async def create(self, spec: SandboxSpec) -> Any:
        _validate_provider_spec(spec)
        if spec.source.kind != SandboxSourceKind.TEMPLATE:
            raise RuntimeError("CubeSandbox requires a built template; direct OCI image creation is unsupported.")
        payload: dict[str, Any] = {
            "templateID": spec.source.reference,
            **_resource_payload(spec),
            **self.create_kwargs,
        }
        if spec.idle_timeout_s is not None:
            payload["timeout"] = max(1, math.ceil(spec.idle_timeout_s))
        if spec.env:
            payload["envVars"] = dict(spec.env)
        metadata = _metadata(spec)
        if metadata:
            payload["metadata"] = metadata
        response = await self.control.request("POST", "/sandboxes", expected=(201,), payload=payload)
        return await self._connect_created(response)


class E2BHibernateDriver:
    """State driver for the standard E2B pause/connect lifecycle."""

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(frozenset({SandboxFeature.HIBERNATE}))

    async def pause(self, session: E2BSession, mode: PauseMode) -> None:
        if mode != PauseMode.HIBERNATE:
            raise RuntimeError("E2B pause has hibernation semantics, not process freeze semantics.")
        await _await(session.client.pause(keep_memory=True))

    async def resume(self, session: E2BSession) -> None:
        await session.refresh_transport()

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef:
        raise NotImplementedError("No snapshot state driver is configured for this E2B-compatible backend.")

    async def fork(self, session: E2BSession) -> SandboxSession:
        raise NotImplementedError("No native fork state driver is configured for this E2B-compatible backend.")

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        raise NotImplementedError("No restore state driver is configured for this E2B-compatible backend.")

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        raise NotImplementedError("No snapshot state driver is configured for this E2B-compatible backend.")

    async def close(self) -> None:
        return None


class E2BNativeStateDriver(E2BHibernateDriver):
    """State operations provided by the official E2B SDK."""

    def __init__(self, client_factory: E2BClientFactory) -> None:
        self.client_factory = client_factory

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities(
            frozenset(
                {
                    SandboxFeature.HIBERNATE,
                    SandboxFeature.FULL_STATE_SNAPSHOT,
                    SandboxFeature.RESTORE,
                    SandboxFeature.NATIVE_FORK,
                }
            )
        )

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef:
        if kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("E2B create_snapshot captures full execution state, not filesystem-only state.")
        snapshot = await _await(session.client.create_snapshot())
        return SnapshotRef(
            backend=session.backend.name,
            snapshot_id=str(snapshot.snapshot_id),
            kind=SnapshotKind.FULL_STATE,
            metadata={"names": list(snapshot.names)},
        )

    async def fork(self, session: E2BSession) -> SandboxSession:
        children = await _await(session.client.fork(count=1))
        if not children:
            raise RuntimeError("E2B native fork returned no child sandbox.")
        child = children[0]
        if isinstance(child, BaseException):
            raise RuntimeError(f"E2B native fork failed: {child}.") from child
        return E2BSession(session.backend, child, spec=session.spec)

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        if snapshot.kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("E2B native restore requires a full-state snapshot.")
        restore_spec = spec or SandboxSpec(source=SandboxSource.template(snapshot.snapshot_id))
        restore_spec = replace(restore_spec, source=SandboxSource.template(snapshot.snapshot_id))
        client = await backend.client_factory.create(restore_spec)
        return E2BSession(backend, client, spec=restore_spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        await self.client_factory.delete_snapshot(snapshot.snapshot_id)


class ProviderStateDriver(E2BHibernateDriver):
    """Full-state provider APIs shared by AgentEnv and CubeSandbox."""

    def __init__(self, control: ProviderControlClient, *, native_fork: bool) -> None:
        self.control = control
        self.native_fork = native_fork

    @property
    def capabilities(self) -> SandboxCapabilities:
        features = {
            SandboxFeature.HIBERNATE,
            SandboxFeature.FULL_STATE_SNAPSHOT,
            SandboxFeature.RESTORE,
        }
        if self.native_fork:
            features.add(SandboxFeature.NATIVE_FORK)
        return SandboxCapabilities(frozenset(features))

    async def pause(self, session: E2BSession, mode: PauseMode) -> None:
        if mode != PauseMode.HIBERNATE:
            raise RuntimeError("Provider pause has hibernation semantics, not process freeze semantics.")
        await self.control.request(
            "POST",
            f"/sandboxes/{session.sandbox_id}/pause",
            expected=(204,),
        )

    async def snapshot(self, session: E2BSession, kind: SnapshotKind) -> SnapshotRef:
        if kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("Provider snapshots contain filesystem, process, and memory state.")
        response = await self.control.request(
            "POST",
            f"/sandboxes/{session.sandbox_id}/snapshots",
            expected=(201,),
            payload={},
        )
        return SnapshotRef(
            backend=session.backend.name,
            snapshot_id=str(response["snapshotID"]),
            kind=SnapshotKind.FULL_STATE,
            metadata={"names": list(response.get("names", []))},
        )

    async def fork(self, session: E2BSession) -> SandboxSession:
        if not self.native_fork:
            raise NotImplementedError("This provider uses snapshot and restore rather than native fork.")
        response = await self.control.request(
            "POST",
            f"/sandboxes/{session.sandbox_id}/fork",
            expected=(201,),
            payload={"count": 1},
        )
        first = response[0]
        if first.get("error"):
            raise RuntimeError(f"Provider native fork failed: {first['error']}.")
        sandbox_id = str(first["sandbox"]["sandboxID"])
        client = await session.backend.client_factory.connect(sandbox_id)
        return E2BSession(session.backend, client, sandbox_id=sandbox_id, spec=session.spec)

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        if snapshot.kind != SnapshotKind.FULL_STATE:
            raise RuntimeError("Provider restore requires a full-state snapshot.")
        restore_spec = spec or SandboxSpec(source=SandboxSource.template(snapshot.snapshot_id))
        restore_spec = replace(restore_spec, source=SandboxSource.template(snapshot.snapshot_id))
        client = await backend.client_factory.create(restore_spec)
        return E2BSession(backend, client, spec=restore_spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        await self.control.delete_idempotent(f"/templates/{snapshot.snapshot_id}")

    async def close(self) -> None:
        await self.control.close()


class AgentEnvStateDriver(ProviderStateDriver):
    """AgentEnv full-state snapshot, restore, and native fork semantics."""

    def __init__(self, control: ProviderControlClient) -> None:
        super().__init__(control, native_fork=True)

    async def restore(
        self,
        backend: E2BBackend,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None,
    ) -> SandboxSession:
        """Restore with resources inherited from AgentEnv's snapshot template."""
        restore_spec = spec or SandboxSpec(source=SandboxSource.template(snapshot.snapshot_id))
        restore_spec = replace(
            restore_spec,
            source=SandboxSource.template(snapshot.snapshot_id),
            resources=type(restore_spec.resources)(),
        )
        client = await backend.client_factory.create(restore_spec)
        return E2BSession(backend, client, spec=restore_spec)


class CubeSandboxStateDriver(ProviderStateDriver):
    """CubeSandbox snapshot and restore semantics."""

    def __init__(self, control: ProviderControlClient) -> None:
        super().__init__(control, native_fork=False)


class E2BBackend(SandboxBackend):
    """Backend for E2B-compatible SDK endpoints."""

    def __init__(
        self,
        *,
        name: str = "e2b",
        api_key: str | None = None,
        api_url: str | None = None,
        create_kwargs: Mapping[str, Any] | None = None,
        connect_kwargs: Mapping[str, Any] | None = None,
        client_factory: E2BClientFactory | None = None,
        state_driver: E2BStateDriver | None = None,
    ) -> None:
        self._name = name
        self.client_factory = client_factory or E2BSDKClientFactory(
            api_key=api_key,
            api_url=api_url,
            create_kwargs=create_kwargs,
            connect_kwargs=connect_kwargs,
        )
        self.state_driver = state_driver or E2BNativeStateDriver(self.client_factory)
        self.metrics = SandboxMetrics()

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.state_driver.capabilities

    def metrics_snapshot(self) -> SandboxMetricsSnapshot:
        return self.metrics.snapshot()

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        try:
            with self.metrics.measure("create"):
                client = await self.client_factory.create(spec)
        except Exception as exc:
            raise RuntimeError(f"Could not create sandbox {spec!r}: {exc}.") from exc
        self.metrics.session_started()
        return E2BSession(self, client, spec=spec)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        try:
            with self.metrics.measure("connect"):
                client = await self.client_factory.connect(sandbox_id)
        except Exception as exc:
            raise RuntimeError(f"Could not connect to sandbox {sandbox_id!r}: {exc}.") from exc
        self.metrics.session_started()
        return E2BSession(self, client, sandbox_id=sandbox_id)

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        self.capabilities.require(SandboxFeature.RESTORE)
        with self.metrics.measure("restore"):
            session = await self.state_driver.restore(self, snapshot, spec)
        self.metrics.session_started()
        return session

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        with self.metrics.measure("delete_snapshot"):
            await self.state_driver.delete_snapshot(snapshot)

    async def shutdown(self) -> None:
        await self.client_factory.close()
        await self.state_driver.close()


class AgentEnvBackend(E2BBackend):
    """AgentEnv backend with cold-image resource mapping and native fork."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        client_factory: E2BClientFactory | None = None,
        state_driver: E2BStateDriver | None = None,
        control_client: ProviderControlClient | None = None,
        **kwargs: Any,
    ) -> None:
        if client_factory is None:
            control = control_client or ProviderControlClient(str(api_url or ""), api_key=api_key)
            client_factory = AgentEnvClientFactory(control=control, **kwargs)
            state_driver = state_driver or AgentEnvStateDriver(control)
            kwargs = {}
        else:
            state_driver = state_driver or E2BHibernateDriver()
        super().__init__(name="agentenv", client_factory=client_factory, state_driver=state_driver, **kwargs)


class CubeSandboxBackend(E2BBackend):
    """CubeSandbox backend with provider-native resources and snapshots."""

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        client_factory: E2BClientFactory | None = None,
        state_driver: E2BStateDriver | None = None,
        control_client: ProviderControlClient | None = None,
        **kwargs: Any,
    ) -> None:
        if client_factory is None:
            control = control_client or ProviderControlClient(str(api_url or ""), api_key=api_key)
            client_factory = CubeSandboxClientFactory(control=control, **kwargs)
            state_driver = state_driver or CubeSandboxStateDriver(control)
            kwargs = {}
        else:
            state_driver = state_driver or E2BHibernateDriver()
        super().__init__(name="cubesandbox", client_factory=client_factory, state_driver=state_driver, **kwargs)


class E2BSession(SandboxSession):
    """One E2B-compatible SDK session."""

    def __init__(
        self,
        backend: E2BBackend,
        client: Any,
        *,
        sandbox_id: str | None = None,
        spec: SandboxSpec | None = None,
    ) -> None:
        self.backend = backend
        self.client = client
        self.sandbox_id = sandbox_id or str(client.sandbox_id)
        self._spec = spec
        self._command_count = 0
        self._status = SandboxStatus.RUNNING
        self._terminate_lock = asyncio.Lock()

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(self.backend.name, self.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.backend.capabilities

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    @property
    def command_count(self) -> int:
        return self._command_count

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        self._command_count += 1
        kwargs: dict[str, Any] = {}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if env:
            kwargs["envs"] = dict(env)
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        with self.backend.metrics.measure("exec"):
            try:
                result = await _await(self.client.commands.run(command, **kwargs))
            except Exception as exc:
                if all(hasattr(exc, attribute) for attribute in ("exit_code", "stdout", "stderr")):
                    result = exc
                elif isinstance(exc, TimeoutError) or type(exc).__name__ in {
                    "TimeoutException",
                    "TimeoutError",
                }:
                    raise TimeoutError(f"E2B command exceeded {timeout_s} seconds.") from exc
                else:
                    raise
        return ExecResult(
            exit_code=int(getattr(result, "exit_code", 0)),
            stdout=str(getattr(result, "stdout", "")),
            stderr=str(getattr(result, "stderr", "")),
        )

    async def read_bytes(self, path: str) -> bytes:
        with self.backend.metrics.measure("read_bytes"):
            data = await _await(self.client.files.read(path, format="bytes"))
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        return str(data).encode()

    async def write_bytes(self, path: str, data: bytes) -> None:
        with self.backend.metrics.measure("write_bytes"):
            await _await(self.client.files.write(path, data))

    async def status(self) -> SandboxStatus:
        if self._status in {SandboxStatus.PAUSED, SandboxStatus.TERMINATED}:
            return self._status
        with self.backend.metrics.measure("status"):
            is_running = await _await(self.client.is_running())
        return SandboxStatus.RUNNING if is_running else SandboxStatus.UNKNOWN

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._status == SandboxStatus.TERMINATED:
                return
            with self.backend.metrics.measure("terminate"):
                try:
                    await _await(self.client.kill())
                except Exception as exc:
                    if "not found" not in str(exc).lower():
                        raise
                await _close_client(self.client)
            self._status = SandboxStatus.TERMINATED
            self.backend.metrics.session_stopped()

    async def pause(self, mode: PauseMode) -> None:
        with self.backend.metrics.measure("pause"):
            await self.backend.state_driver.pause(self, mode)
        self._status = SandboxStatus.PAUSED

    async def resume(self) -> None:
        with self.backend.metrics.measure("resume"):
            await self.backend.state_driver.resume(self)
        self._status = SandboxStatus.RUNNING

    async def refresh_transport(self) -> None:
        """Reconnect after VM restore so stale TCP pools are never reused."""
        old_client = self.client
        with self.backend.metrics.measure("refresh_transport"):
            self.client = await self.backend.client_factory.connect(self.sandbox_id)
            await _close_client(old_client)
        self._status = SandboxStatus.RUNNING

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        with self.backend.metrics.measure("snapshot"):
            return await self.backend.state_driver.snapshot(self, kind)

    async def fork(self) -> SandboxSession:
        with self.backend.metrics.measure("fork"):
            child = await self.backend.state_driver.fork(self)
        self.backend.metrics.session_started()
        return child
