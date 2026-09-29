"""Docker Engine API sandbox backend."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from psrl.sandbox.backends.docker_engine import (
    DockerEngine,
    DockerEngineClient,
    DockerEngineError,
    DockerExecOutputLimitError,
)
from psrl.sandbox.backends.docker_lifecycle import DockerLifecycle, DockerLifecycleConfig
from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
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
from psrl.sandbox.utils.docker_utils import lease_store_id

psrl_logger = logging.getLogger(__file__)

# FILESYSTEM_SNAPSHOT + RESTORE: docker commit turns a container's writable layer into a reusable
# image, and restore creates one from it: a cheap, self-contained grader snapshot with no cold start.
_DOCKER_CAPABILITIES = SandboxCapabilities(
    frozenset(
        {
            SandboxFeature.FREEZE,
            SandboxFeature.HOST_MOUNT,
            SandboxFeature.FILESYSTEM_SNAPSHOT,
            SandboxFeature.RESTORE,
        }
    )
)
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_PROXY_URL_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


def _rewrite_loopback_proxy(value: str, host_alias: str) -> str:
    """Replace a proxy URL's loopback host with the Docker host gateway alias."""
    has_scheme = "://" in value
    parsed = urlsplit(value if has_scheme else f"//{value}")
    if parsed.hostname not in _LOOPBACK_HOSTS:
        return value
    try:
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return value
    userinfo, separator, _ = parsed.netloc.rpartition("@")
    authority = f"{userinfo}{separator}{host_alias}{port}"
    rewritten = urlunsplit((parsed.scheme, authority, parsed.path, parsed.query, parsed.fragment))
    return rewritten if has_scheme else rewritten.removeprefix("//")


def _append_no_proxy_alias(value: str, host_alias: str) -> str:
    entries = [entry.strip() for entry in value.split(",") if entry.strip()]
    return ",".join(dict.fromkeys([*entries, host_alias]))


@dataclass(frozen=True)
class DockerSecurityConfig:
    """Security controls applied to every Docker sandbox."""

    require_rootless: bool = False
    pids_limit: int = 4096
    cap_drop: tuple[str, ...] = ("ALL",)
    cap_add: tuple[str, ...] = ()
    no_new_privileges: bool = True
    read_only_rootfs: bool = False
    seccomp_profile: str | None = None
    user: str | None = None
    tmpfs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pids_limit <= 0:
            raise ValueError("Docker pids_limit must be greater than zero.")


@dataclass(frozen=True)
class DockerPolicyProfile:
    """Typed per-workload Docker policy overrides."""

    network_mode: str | None = None
    extra_hosts: tuple[str, ...] = ()
    host_gateway_alias: str | None = None
    rewrite_loopback_proxies: bool = False
    pids_limit: int | None = None
    cap_drop: tuple[str, ...] | None = None
    cap_add: tuple[str, ...] | None = None
    no_new_privileges: bool | None = None
    read_only_rootfs: bool | None = None
    seccomp_profile: str | None = None
    user: str | None = None
    tmpfs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pids_limit is not None and self.pids_limit <= 0:
            raise ValueError("Docker policy pids_limit must be greater than zero.")
        if self.rewrite_loopback_proxies and not self.host_gateway_alias:
            raise ValueError("Docker proxy rewriting requires host_gateway_alias.")

    @classmethod
    def from_value(cls, value: DockerPolicyProfile | Mapping[str, Any]) -> DockerPolicyProfile:
        """Normalize Hydra mappings into an immutable policy."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Docker policy profiles must use typed mappings, not raw CLI arguments.")
        normalized = dict(value)
        for key in ("extra_hosts", "cap_drop", "cap_add"):
            if normalized.get(key) is not None:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)


@dataclass(frozen=True)
class DockerDiskAdmissionConfig:
    """Host disk headroom required before a sandbox can be created."""

    path: str | None = None
    min_free_mb: int = 0
    wait_timeout_s: float = 300.0
    poll_interval_s: float = 5.0

    def __post_init__(self) -> None:
        if self.min_free_mb < 0 or self.wait_timeout_s < 0 or self.poll_interval_s <= 0:
            raise ValueError(
                "Docker disk min_free_mb and wait_timeout_s must not be negative, "
                "and poll_interval_s must be positive."
            )
        if self.min_free_mb > 0 and not self.path:
            raise ValueError("Docker disk path is required when min_free_mb is positive.")

    @classmethod
    def from_value(
        cls,
        value: DockerDiskAdmissionConfig | Mapping[str, Any] | None,
    ) -> DockerDiskAdmissionConfig:
        """Normalize a Hydra mapping into immutable disk admission policy."""
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))


class DockerBackend(SandboxBackend):
    """Local Docker backend using one persistent Engine API connection pool."""

    def __init__(
        self,
        name: str = "docker",
        docker_host: str | None = None,
        policy_profiles: Mapping[str, DockerPolicyProfile | Mapping[str, Any]] | None = None,
        security: DockerSecurityConfig | Mapping[str, Any] | None = None,
        keepalive_command: Sequence[str] = ("tail", "-f", "/dev/null"),
        command_interpreter: Sequence[str] = ("bash", "-lc"),
        request_timeout_s: float = 180.0,
        connection_limit: int = 128,
        image_pull_concurrency: int = 2,
        max_exec_output_bytes: int = 16 * 1024 * 1024,
        auto_pull: bool = True,
        registry_auth: Mapping[str, str] | None = None,
        owner_id_env: str = "PSRL_ACTOR_ID",
        lifecycle: DockerLifecycleConfig | Mapping[str, Any] | None = None,
        disk_admission: DockerDiskAdmissionConfig | Mapping[str, Any] | None = None,
        cgroup_parent: str | None = None,
        engine: DockerEngine | None = None,
    ) -> None:
        self._name = name
        self.docker_host = docker_host
        self.policy_profiles = {
            key: DockerPolicyProfile.from_value(value) for key, value in (policy_profiles or {}).items()
        }
        self.security = (
            security if isinstance(security, DockerSecurityConfig) else DockerSecurityConfig(**dict(security or {}))
        )
        self.keepalive_command = tuple(keepalive_command)
        self.command_interpreter = tuple(command_interpreter)
        if not self.keepalive_command or not self.command_interpreter:
            raise ValueError("Docker keepalive_command and command_interpreter cannot be empty.")
        self.owner_id = os.getenv(owner_id_env, "")
        self.lifecycle = DockerLifecycle(
            self.owner_id,
            DockerLifecycleConfig.from_value(lifecycle, docker_host=docker_host),
        )
        self.disk_admission = DockerDiskAdmissionConfig.from_value(disk_admission)
        self.cgroup_parent = cgroup_parent
        self.auto_pull = auto_pull
        self.registry_auth = dict(registry_auth or {})
        self.engine = engine or DockerEngineClient(
            docker_host,
            request_timeout_s=request_timeout_s,
            connection_limit=connection_limit,
            max_exec_output_bytes=max_exec_output_bytes,
        )
        self.metrics = SandboxMetrics()
        self._rootless_checked = False
        self._rootless_lock = asyncio.Lock()
        if image_pull_concurrency < 1:
            raise ValueError("Docker image_pull_concurrency must be positive.")
        self._image_pull_slots = asyncio.Semaphore(image_pull_concurrency)
        self._image_tasks: dict[str, asyncio.Task[None]] = {}
        self._closed = False

    async def prepare(self, spec: SandboxSpec) -> None:
        """
        Warm an image with bounded, shared downloads on this backend instance.
        """
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        if spec.source.kind != SandboxSourceKind.IMAGE:
            raise RuntimeError("DockerBackend requires an image source.")
        reference = spec.source.reference
        task = self._image_tasks.get(reference)
        if task is None:
            task = asyncio.create_task(self._prepare_image(reference))
            self._image_tasks[reference] = task
            task.add_done_callback(lambda done: self._finish_image_prepare(reference, done))
        await asyncio.shield(task)

    def _finish_image_prepare(self, reference: str, task: asyncio.Task[None]) -> None:
        """
        Forget completed downloads and observe failures after waiter cancellation.
        """
        self._image_tasks.pop(reference, None)
        if not task.cancelled():
            task.exception()

    async def _prepare_image(self, reference: str) -> None:
        """
        Recheck the daemon cache after admission before downloading an image.
        """
        async with self._image_pull_slots:
            if await self.engine.image_exists(reference):
                return
            if not self.auto_pull:
                raise RuntimeError(f"Docker image {reference!r} is unavailable and auto_pull is disabled.")
            await self._await_disk_headroom()
            with self.metrics.measure("pull_image"):
                await self.engine.pull_image(reference, self.registry_auth or None)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> SandboxCapabilities:
        return _DOCKER_CAPABILITIES

    @property
    def uses_node_capacity(self) -> bool:
        """Return that Docker sessions share the worker node's resources."""
        return True

    def metrics_snapshot(self) -> SandboxMetricsSnapshot:
        """Return Docker lifecycle, latency, and memory metrics."""
        return self.metrics.snapshot()

    async def _await_disk_headroom(self) -> None:
        """Block until the Docker data path has enough free space, then give up."""
        policy = self.disk_admission
        if not policy.path or policy.min_free_mb <= 0:
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + policy.wait_timeout_s
        while True:
            try:
                usage = await asyncio.to_thread(shutil.disk_usage, policy.path)
                free_mb = usage.free // (1024 * 1024)
            except OSError as exc:
                raise RuntimeError(f"Could not inspect Docker data path {policy.path!r}.") from exc
            if free_mb >= policy.min_free_mb:
                return
            if loop.time() >= deadline:
                with self.metrics.measure("disk_admission_failure"):
                    raise RuntimeError(
                        f"Docker data path {policy.path!r} has {free_mb} MiB free, below the "
                        f"{policy.min_free_mb} MiB admission threshold, after waiting "
                        f"{policy.wait_timeout_s:.0f}s."
                    )
            with self.metrics.measure("disk_admission_wait"):
                await asyncio.sleep(min(policy.poll_interval_s, max(0.0, deadline - loop.time())))

    async def _check_rootless(self) -> None:
        if not self.security.require_rootless or self._rootless_checked:
            return
        async with self._rootless_lock:
            if self._rootless_checked:
                return
            info = await self.engine.info()
            security_options = [str(option).lower() for option in info.get("SecurityOptions", [])]
            if not any("rootless" in option for option in security_options):
                raise RuntimeError(
                    "Docker backend requires a rootless daemon, but Docker did not report rootless mode."
                )
            self._rootless_checked = True

    def _container_name(self, spec: SandboxSpec) -> str:
        if spec.idempotency_key:
            digest = hashlib.sha256(f"{self.name}\0{spec.idempotency_key}".encode()).hexdigest()[:20]
            return f"psrl-sandbox-{digest}"
        return f"psrl-sandbox-{uuid.uuid4().hex[:20]}"

    def _resolve_policy(self, spec: SandboxSpec) -> DockerPolicyProfile:
        if spec.policy_profile is None:
            return DockerPolicyProfile()
        try:
            return self.policy_profiles[spec.policy_profile]
        except KeyError as exc:
            raise RuntimeError(f"Unknown Docker policy profile {spec.policy_profile!r}.") from exc

    def _build_container_config(self, spec: SandboxSpec, policy: DockerPolicyProfile) -> dict[str, Any]:
        labels = {**dict(spec.metadata), "psrl.sandbox": "true"}
        labels["psrl.lease_store"] = lease_store_id(self.lifecycle.config.heartbeat_dir)
        if self.owner_id:
            labels["psrl.actor_id"] = self.owner_id
        if spec.idempotency_key:
            labels["psrl.idempotency_key"] = spec.idempotency_key

        mounts = []
        for mount in spec.mounts:
            if not Path(mount.source).is_absolute() or not Path(mount.target).is_absolute():
                raise ValueError("Docker bind-mount source and target must be absolute paths.")
            mounts.append(
                {
                    "Type": "bind",
                    "Source": mount.source,
                    "Target": mount.target,
                    "ReadOnly": mount.read_only,
                }
            )

        cap_drop = self.security.cap_drop if policy.cap_drop is None else policy.cap_drop
        cap_add = self.security.cap_add if policy.cap_add is None else policy.cap_add
        no_new_privileges = (
            self.security.no_new_privileges if policy.no_new_privileges is None else policy.no_new_privileges
        )
        read_only = self.security.read_only_rootfs if policy.read_only_rootfs is None else policy.read_only_rootfs
        pids_limit = self.security.pids_limit if policy.pids_limit is None else policy.pids_limit
        seccomp_profile = policy.seccomp_profile or self.security.seccomp_profile
        if seccomp_profile == "unconfined":
            raise ValueError("Docker seccomp cannot be disabled for PSRL sandboxes.")
        security_options = ["no-new-privileges"] if no_new_privileges else []
        if seccomp_profile:
            security_options.append(f"seccomp={seccomp_profile}")
        tmpfs = {**dict(self.security.tmpfs), **dict(policy.tmpfs)}

        host_config: dict[str, Any] = {
            "AutoRemove": True,
            "Init": True,
            "CapDrop": list(cap_drop),
            "CapAdd": list(cap_add),
            "ReadonlyRootfs": read_only,
            "PidsLimit": pids_limit,
            "SecurityOpt": security_options,
            "Mounts": mounts,
        }
        if policy.network_mode:
            host_config["NetworkMode"] = policy.network_mode
        extra_hosts = list(policy.extra_hosts)
        if policy.host_gateway_alias and not any(
            entry.partition(":")[0] == policy.host_gateway_alias for entry in extra_hosts
        ):
            extra_hosts.append(f"{policy.host_gateway_alias}:host-gateway")
        if extra_hosts:
            host_config["ExtraHosts"] = extra_hosts
        if tmpfs:
            host_config["Tmpfs"] = tmpfs
        if self.cgroup_parent:
            host_config["CgroupParent"] = self.cgroup_parent
        if spec.resources.cpu_count is not None:
            host_config["NanoCpus"] = int(spec.resources.cpu_count * 1_000_000_000)
        if spec.resources.memory_mb is not None:
            host_config["Memory"] = spec.resources.memory_mb * 1024 * 1024

        environment = dict(spec.env)
        if policy.host_gateway_alias:
            if policy.rewrite_loopback_proxies:
                for key in _PROXY_URL_ENV_KEYS:
                    if key in environment:
                        environment[key] = _rewrite_loopback_proxy(environment[key], policy.host_gateway_alias)
            no_proxy = _append_no_proxy_alias(
                ",".join(filter(None, (environment.get("no_proxy"), environment.get("NO_PROXY")))),
                policy.host_gateway_alias,
            )
            environment["no_proxy"] = no_proxy
            environment["NO_PROXY"] = no_proxy

        config: dict[str, Any] = {
            "Image": spec.source.reference,
            "Cmd": list(self.keepalive_command),
            "Env": [f"{key}={value}" for key, value in sorted(environment.items())],
            "Labels": labels,
            "HostConfig": host_config,
        }
        if spec.workdir:
            config["WorkingDir"] = spec.workdir
        user = policy.user or self.security.user
        if user:
            config["User"] = user
        identity = {
            "container": config,
            "idle_timeout_s": spec.idle_timeout_s,
        }
        labels["psrl.spec_hash"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return config

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        if self._closed:
            raise RuntimeError("Docker backend is closed.")
        if spec.source.kind != SandboxSourceKind.IMAGE:
            raise RuntimeError("DockerBackend requires an image source.")
        if spec.resources.disk_mb is not None:
            raise RuntimeError("DockerBackend does not implement a portable disk size limit.")
        await self._await_disk_headroom()
        await self._check_rootless()
        await asyncio.to_thread(self.lifecycle.start)
        policy = self._resolve_policy(spec)
        name = self._container_name(spec)
        config = self._build_container_config(spec, policy)
        with self.metrics.measure("create"):
            container_id = await self._create_or_recover(name, config, spec)
        self.metrics.session_started()
        return DockerSession(
            self,
            container_id,
            spec=spec,
            lifetime_timeout_s=spec.idle_timeout_s,
        )

    async def _create_or_recover(
        self,
        name: str,
        config: Mapping[str, Any],
        spec: SandboxSpec,
    ) -> str:
        """
        Create once, pull a missing image once, or join an exact retry.
        """
        try:
            return await self._create_and_start(name, config)
        except DockerEngineError as exc:
            create_error = exc
        if create_error.status == 404 and self.auto_pull:
            await self.prepare(spec)
            try:
                return await self._create_and_start(name, config)
            except DockerEngineError as exc:
                create_error = exc
        if create_error.status == 409 and spec.idempotency_key:
            return await self._recover_idempotent_conflict(name, config, spec, create_error)
        raise create_error

    async def _create_and_start(self, name: str, config: Mapping[str, Any]) -> str:
        """Create and start a container without leaking a failed start."""
        container_id = await self.engine.create_container(name, config)
        try:
            await self.engine.start_container(container_id)
        except BaseException:
            try:
                await self.engine.remove_container(container_id)
            except BaseException as cleanup_error:
                psrl_logger.warning(f"Failed to remove Docker container {container_id!r}: {cleanup_error!r}.")
            raise
        return container_id

    async def _recover_idempotent_conflict(
        self,
        name: str,
        config: Mapping[str, Any],
        spec: SandboxSpec,
        conflict: DockerEngineError,
    ) -> str:
        """Reuse an exact retry without racing another process's start call."""
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            existing = await self.engine.inspect_container(name)
            if existing is None:
                return await self._create_and_start(name, config)
            labels = (existing.get("Config") or {}).get("Labels") or {}
            if labels.get("psrl.idempotency_key") != spec.idempotency_key:
                raise RuntimeError(f"Docker container name {name!r} has an unrelated owner.") from conflict
            expected_hash = config["Labels"]["psrl.spec_hash"]
            if labels.get("psrl.spec_hash") != expected_hash:
                raise RuntimeError(
                    f"Docker idempotency key {spec.idempotency_key!r} was reused with a different spec."
                ) from conflict
            container_id = str(existing["Id"])
            status = str((existing.get("State") or {}).get("Status", "unknown"))
            if status == "running":
                return container_id
            if status == "paused":
                await self.engine.unpause_container(container_id)
                return container_id
            if status in {"created", "restarting"} and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.05)
                continue
            await self.engine.remove_container(container_id)
            return await self._create_and_start(name, config)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        session = DockerSession(self, sandbox_id)
        status = await session.status()
        if status == SandboxStatus.PAUSED:
            await session.resume()
        elif status != SandboxStatus.RUNNING:
            raise RuntimeError(f"Docker sandbox {sandbox_id!r} is not running (status={status.value}).")
        self.metrics.session_started()
        return session

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        """Create a session from a filesystem snapshot (a committed image)."""
        if snapshot.kind != SnapshotKind.FILESYSTEM:
            raise NotImplementedError(f"DockerBackend only restores FILESYSTEM snapshots, got {snapshot.kind.value}.")
        image = snapshot.metadata.get("psrl.docker.image") or snapshot.snapshot_id
        if not image:
            raise ValueError(f"Snapshot {snapshot.snapshot_id!r} has no docker image reference.")
        if spec is None:
            raise ValueError("DockerBackend.restore requires a SandboxSpec.")
        restored_spec = replace(spec, source=SandboxSource.image(str(image)))
        return await self.create(restored_spec)

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Remove the committed image backing a filesystem snapshot."""
        if snapshot.kind != SnapshotKind.FILESYSTEM:
            return
        image = snapshot.metadata.get("psrl.docker.image") or snapshot.snapshot_id
        if image:
            await self.engine.remove_image(str(image))

    async def shutdown(self) -> None:
        """Stop crash recovery and close the persistent Engine connection pool."""
        self._closed = True
        tasks = list(self._image_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.to_thread(self.lifecycle.close)
        finally:
            await self.engine.close()


class DockerSession(SandboxSession):
    """One Docker container session."""

    def __init__(
        self,
        backend: DockerBackend,
        sandbox_id: str,
        *,
        spec: SandboxSpec | None = None,
        lifetime_timeout_s: float | None = None,
    ) -> None:
        self.backend = backend
        self.sandbox_id = sandbox_id
        self._spec = spec
        self._command_count = 0
        self._terminate_lock = asyncio.Lock()
        self._terminated = False
        self._timeout_task = (
            asyncio.create_task(self._terminate_at_deadline(lifetime_timeout_s))
            if lifetime_timeout_s is not None
            else None
        )

    async def _terminate_at_deadline(self, timeout_s: float) -> None:
        """Enforce Docker lifetime locally because Engine has no native TTL."""
        try:
            await asyncio.sleep(timeout_s)
            with self.backend.metrics.measure("lifetime_timeout"):
                await self.terminate()
        except asyncio.CancelledError:
            return

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(self.backend.name, self.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        return self.backend.capabilities

    async def snapshot(self, kind: SnapshotKind) -> SnapshotRef:
        """Capture a filesystem snapshot by committing the container's writable layer."""
        if kind != SnapshotKind.FILESYSTEM:
            raise NotImplementedError(f"DockerSession only supports FILESYSTEM snapshots, got {kind.value}.")
        repo = f"psrl/snapshot/{self.sandbox_id}"
        tag = uuid.uuid4().hex
        image_id = await self.backend.engine.commit_container(self.sandbox_id, repo, tag)
        image_tag = f"{repo}:{tag}"
        return SnapshotRef(
            backend=self.backend.name,
            snapshot_id=image_tag,
            kind=SnapshotKind.FILESYSTEM,
            metadata={"psrl.docker.image": image_tag, "psrl.docker.image_id": image_id},
        )

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    @property
    def command_count(self) -> int:
        return self._command_count

    def resolve_callback_url(self, url: str) -> str:
        """Rewrite worker-loopback URLs through the configured host gateway."""
        if self._spec is None or self._spec.policy_profile is None:
            return url
        policy = self.backend.policy_profiles.get(self._spec.policy_profile)
        if policy is None or not policy.host_gateway_alias:
            return url
        return _rewrite_loopback_proxy(url, policy.host_gateway_alias)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        self._command_count += 1
        try:
            with self.backend.metrics.measure("exec"):
                exit_code, stdout, stderr = await asyncio.wait_for(
                    self.backend.engine.exec(
                        self.sandbox_id,
                        [*self.backend.command_interpreter, command],
                        cwd=cwd,
                        env=env,
                        timeout_s=timeout_s,
                    ),
                    timeout=timeout_s,
                )
        except (
            asyncio.CancelledError,
            TimeoutError,
            asyncio.TimeoutError,
            aiohttp.ClientError,
            DockerEngineError,
            DockerExecOutputLimitError,
        ) as exc:
            # Losing the exec stream does not stop the process in Docker.
            try:
                await self.terminate()
            except Exception:
                psrl_logger.warning("Failed to terminate a Docker sandbox after command failure.", exc_info=True)
            if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
                raise TimeoutError(f"Docker command timed out (requested timeout={timeout_s!r}).") from exc
            raise
        return ExecResult(
            exit_code=exit_code,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    async def read_bytes(self, path: str) -> bytes:
        with self.backend.metrics.measure("read_bytes"):
            return await self.backend.engine.read_file(self.sandbox_id, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        with self.backend.metrics.measure("write_bytes"):
            await self.backend.engine.write_file(self.sandbox_id, path, data)

    async def status(self) -> SandboxStatus:
        if self._terminated:
            return SandboxStatus.TERMINATED
        with self.backend.metrics.measure("status"):
            inspection = await self.backend.engine.inspect_container(self.sandbox_id)
        if inspection is None:
            return SandboxStatus.TERMINATED
        state = str((inspection.get("State") or {}).get("Status", "unknown"))
        return {
            "running": SandboxStatus.RUNNING,
            "paused": SandboxStatus.PAUSED,
            "exited": SandboxStatus.EXITED,
            "dead": SandboxStatus.EXITED,
        }.get(state, SandboxStatus.UNKNOWN)

    async def stats(self) -> ResourceUsage:
        with self.backend.metrics.measure("stats"):
            stats = await self.backend.engine.stats(self.sandbox_id)
        memory = stats.get("memory_stats") or {}
        current = int(memory.get("usage", 0) or 0)
        peak = int(memory.get("max_usage", 0) or (memory.get("stats") or {}).get("peak", 0) or current)
        cpu_total = int(((stats.get("cpu_stats") or {}).get("cpu_usage") or {}).get("total_usage", 0) or 0)
        self.backend.metrics.observe_memory(current, peak)
        return ResourceUsage(memory_bytes=current, peak_memory_bytes=peak, cpu_total_ns=cpu_total)

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self._terminated:
                return
            with self.backend.metrics.measure("terminate"):
                try:
                    await self.backend.engine.remove_container(self.sandbox_id)
                except Exception as exc:
                    psrl_logger.warning(
                        f"Could not remove Docker sandbox {self.sandbox_id!r}: {exc!r}. "
                        "Leaving the container to backend shutdown and the node sandbox GC."
                    )
                    pass
            self._terminated = True
            current_task = asyncio.current_task()
            if self._timeout_task is not None and self._timeout_task is not current_task:
                self._timeout_task.cancel()
            self.backend.metrics.session_stopped()

    async def pause(self, mode: PauseMode) -> None:
        if mode != PauseMode.FREEZE:
            raise RuntimeError("DockerBackend supports freeze, not hibernation.")
        with self.backend.metrics.measure("pause"):
            await self.backend.engine.pause_container(self.sandbox_id)

    async def resume(self) -> None:
        with self.backend.metrics.measure("resume"):
            await self.backend.engine.unpause_container(self.sandbox_id)
