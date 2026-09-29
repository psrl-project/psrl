from __future__ import annotations

import asyncio
import logging
import secrets
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from psrl.sandbox.capacity import ResourceQuantity
from psrl.sandbox.core import (
    ResourceSpec,
    SandboxBackend,
    SandboxFeature,
    SandboxRef,
    SandboxSession,
    SandboxSpec,
    SandboxStatePolicy,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.metrics import SandboxMetricsSnapshot

if TYPE_CHECKING:
    from psrl.sandbox.sync import SyncSandboxManager

psrl_logger = logging.getLogger(__file__)


class SandboxLease:
    """Single-owner lifecycle guard for a sandbox session."""

    def __init__(
        self,
        session: SandboxSession,
        capacity_lease_id: str | None = None,
        release_capacity: Callable[[str], Awaitable[None]] | None = None,
        on_released: Callable[[SandboxLease], None] | None = None,
    ) -> None:
        self.session = session
        self._capacity_lease_id = capacity_lease_id
        self._release_capacity = release_capacity
        self._released = False
        self._lock = asyncio.Lock()
        self._on_released = on_released

    @property
    def ref(self) -> SandboxRef:
        """Return the leased sandbox reference."""
        return self.session.ref

    @property
    def released(self) -> bool:
        """Return whether the exit disposition has already been applied."""
        return self._released

    async def release(self) -> None:
        """Terminate the sandbox exactly once."""
        async with self._lock:
            if self._released:
                return
            await self.session.terminate()
            if self._capacity_lease_id is not None and self._release_capacity is not None:
                await self._release_capacity(self._capacity_lease_id)
            self._released = True
            if self._on_released is not None:
                self._on_released(self)

    async def __aenter__(self) -> SandboxSession:
        return self.session

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.release()


@dataclass(frozen=True)
class SandboxTask:
    """One shared create task and the spec bound to its idempotency key."""

    spec: SandboxSpec
    task: asyncio.Task[SandboxLease]


class SandboxManager:
    """Resolve backends and own active sandbox leases for one worker."""

    def __init__(
        self,
        backends: Mapping[str, SandboxBackend],
        default_backend: str,
        capacity_coordinator=None,
        capacity_owner_id: str | None = None,
        capacity_heartbeat_interval_s: float | None = None,
    ) -> None:
        self._backends = dict(backends)
        self.default_backend = default_backend
        if default_backend not in self._backends:
            raise ValueError(f"Default sandbox backend {default_backend!r} is not configured.")
        for name, backend in self._backends.items():
            if name != backend.name:
                raise ValueError(f"Sandbox backend key {name!r} does not match backend.name={backend.name!r}.")
        self._leases: set[SandboxLease] = set()
        self._idempotent_leases: dict[tuple[str, str], SandboxLease] = {}
        self._create_tasks: dict[tuple[str, str], SandboxTask] = {}
        self._anonymous_create_tasks: set[asyncio.Task[SandboxLease]] = set()
        self._create_lock = asyncio.Lock()
        self._closed = False
        self._capacity_coordinator = capacity_coordinator
        self._capacity_owner_id = capacity_owner_id
        self._capacity_heartbeat_interval_s = capacity_heartbeat_interval_s
        self._capacity_heartbeat_task: asyncio.Task[None] | None = None
        if capacity_coordinator is not None and (not capacity_owner_id or capacity_heartbeat_interval_s is None):
            raise ValueError("Sandbox capacity coordination requires an owner id and heartbeat interval.")

    def backend(self, name: str | None = None) -> SandboxBackend:
        """Resolve a configured backend by name or use the default backend."""
        backend_name = name or self.default_backend
        return self._backends[backend_name]

    def sync(
        self,
        backend: str | None = None,
    ) -> SyncSandboxManager:
        """Create a facade for synchronous code running outside the current event loop."""
        from psrl.sandbox.sync import SyncSandboxManager

        self._require_open()
        self.backend(backend)
        return SyncSandboxManager(self, asyncio.get_running_loop(), backend=backend)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Sandbox manager is closed.")

    async def _acquire_capacity(
        self,
        backend: SandboxBackend,
        resources: ResourceSpec | None,
    ) -> str | None:
        """
        Admit one node-local resource request through the shared coordinator.
        """
        if not backend.uses_node_capacity or self._capacity_coordinator is None:
            return None
        if resources is None:
            raise ValueError("Node-capacity admission requires a ResourceSpec.")
        request = ResourceQuantity.from_spec(resources)
        if self._capacity_heartbeat_task is None:
            self._capacity_heartbeat_task = asyncio.create_task(self._heartbeat_capacity_owner())
        lease_id = uuid.uuid4().hex
        try:
            await self._capacity_coordinator.acquire.remote(
                lease_id,
                self._capacity_owner_id,
                request.memory_mb,
                request.cpu_millis / 1000,
            )
        except asyncio.CancelledError:
            await asyncio.shield(self._capacity_coordinator.cancel.remote(lease_id))
            raise
        return lease_id

    async def _release_capacity(self, lease_id: str) -> None:
        await self._capacity_coordinator.release.remote(lease_id)

    async def _heartbeat_capacity_owner(self) -> None:
        while True:
            await asyncio.sleep(self._capacity_heartbeat_interval_s)
            try:
                await self._capacity_coordinator.renew_owner.remote(self._capacity_owner_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                psrl_logger.warning(
                    f"Sandbox capacity heartbeat failed for owner {self._capacity_owner_id!r}.",
                    exc_info=True,
                )

    async def _close_capacity(self) -> None:
        if self._capacity_heartbeat_task is not None:
            self._capacity_heartbeat_task.cancel()
            await asyncio.gather(self._capacity_heartbeat_task, return_exceptions=True)
        await self._capacity_coordinator.release_owner.remote(self._capacity_owner_id)

    async def prepare(
        self,
        spec: SandboxSpec,
        backend: str | None = None,
    ) -> None:
        """
        Warm backend artifacts without reserving container CPU or memory.

        Callers may schedule this alongside rollout and must join or cancel
        their preparation task before releasing task ownership.
        """
        self._require_open()
        await self.backend(backend).prepare(spec)

    async def acquire(
        self,
        spec: SandboxSpec,
        backend: str | None = None,
    ) -> SandboxLease:
        """Provision a sandbox and return its explicit ownership guard."""
        self._require_open()
        selected = self.backend(backend)
        idempotency_key = spec.idempotency_key
        if idempotency_key is None:
            async with self._create_lock:
                self._require_open()
                task = asyncio.create_task(self._acquire_new(spec, selected))
                self._anonymous_create_tasks.add(task)
                task.add_done_callback(self._anonymous_create_tasks.discard)
            return await asyncio.shield(task)

        task_key = (selected.name, idempotency_key)
        async with self._create_lock:
            self._require_open()
            existing = self._idempotent_leases.get(task_key)
            if existing is not None and not existing.released:
                if existing.session.spec != spec:
                    raise RuntimeError("A sandbox idempotency key cannot be reused with a different spec.")
                return existing
            create_task = self._create_tasks.get(task_key)
            if create_task is not None and create_task.task.done():
                self._create_tasks.pop(task_key, None)
                create_task = None
            if create_task is None:
                task = asyncio.create_task(
                    self._acquire_new(spec, selected, task_key),
                )
                self._create_tasks[task_key] = SandboxTask(spec, task)
            elif create_task.spec != spec:
                raise RuntimeError("A sandbox idempotency key cannot be reused with a different spec.")
            else:
                task = create_task.task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                async with self._create_lock:
                    create_task = self._create_tasks.get(task_key)
                    if create_task is not None and create_task.task is task:
                        self._create_tasks.pop(task_key, None)

    async def _acquire_new(
        self,
        spec: SandboxSpec,
        selected: SandboxBackend,
        task_key: tuple[str, str] | None = None,
    ) -> SandboxLease:
        """Create and register one lease after admission checks."""
        required = set(spec.required_features)
        if spec.mounts:
            required.add(SandboxFeature.HOST_MOUNT)
        selected.capabilities.require(*required)
        capacity_lease_id = await self._acquire_capacity(selected, spec.resources)
        try:
            session = await selected.create(spec)
        except BaseException:
            if capacity_lease_id is not None:
                await self._release_capacity(capacity_lease_id)
            raise
        try:
            session.capabilities.require(*required)
        except BaseException:
            await session.terminate()
            if capacity_lease_id is not None:
                await self._release_capacity(capacity_lease_id)
            raise
        lease = SandboxLease(
            session,
            capacity_lease_id,
            self._release_capacity,
            on_released=lambda released: self._forget_lease(released, task_key),
        )
        self._leases.add(lease)
        if task_key is not None:
            self._idempotent_leases[task_key] = lease
        return lease

    def _forget_lease(
        self,
        lease: SandboxLease,
        task_key: tuple[str, str] | None = None,
    ) -> None:
        """Forget a released lease and its process-local idempotency key."""
        self._leases.discard(lease)
        if task_key is not None and self._idempotent_leases.get(task_key) is lease:
            self._idempotent_leases.pop(task_key, None)

    async def connect(
        self,
        ref: SandboxRef,
        resources: ResourceSpec | None = None,
    ) -> SandboxLease:
        """Reconnect to a backend-owned sandbox and assume lifecycle ownership."""
        self._require_open()
        backend = self.backend(ref.backend)
        capacity_lease_id = await self._acquire_capacity(backend, resources)
        try:
            session = await backend.connect(ref.sandbox_id)
        except BaseException:
            if capacity_lease_id is not None:
                await self._release_capacity(capacity_lease_id)
            raise
        lease = SandboxLease(session, capacity_lease_id, self._release_capacity, on_released=self._leases.discard)
        self._leases.add(lease)
        return lease

    async def restore(
        self,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None = None,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SandboxLease:
        """Restore a snapshot through its owning backend."""
        self._require_open()
        backend = self.backend(snapshot.backend)
        backend.capabilities.require(SandboxFeature.RESTORE)
        policy = state_policy or (spec.state_policy if spec is not None else SandboxStatePolicy())
        if not policy.enabled:
            raise RuntimeError("Sandbox restore requires an explicitly enabled SandboxStatePolicy.")
        resources = spec.resources if spec is not None else self._resources_from_snapshot(snapshot)
        capacity_lease_id = await self._acquire_capacity(backend, resources)
        try:
            session = await backend.restore(snapshot, spec)
        except BaseException:
            if capacity_lease_id is not None:
                await self._release_capacity(capacity_lease_id)
            raise
        try:
            await self._sanitize_restored_session(session, policy)
        except BaseException:
            if capacity_lease_id is not None:
                await self._release_capacity(capacity_lease_id)
            raise
        lease = SandboxLease(session, capacity_lease_id, self._release_capacity, on_released=self._leases.discard)
        self._leases.add(lease)
        return lease

    async def branch(
        self,
        session: SandboxSession,
        snapshot_kind: SnapshotKind = SnapshotKind.FULL_STATE,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SandboxLease:
        """Branch state without silently weakening snapshot semantics."""
        self._require_open()
        policy = state_policy or (session.spec.state_policy if session.spec is not None else SandboxStatePolicy())
        self._validate_state_policy(session, policy)
        if session.capabilities.supports(SandboxFeature.NATIVE_FORK):
            backend = self.backend(session.ref.backend)
            capacity_lease_id = await self._acquire_capacity(backend, session.spec.resources)
            try:
                child = await session.fork()
            except BaseException:
                if capacity_lease_id is not None:
                    await self._release_capacity(capacity_lease_id)
                raise
            try:
                await self._sanitize_restored_session(child, policy)
            except BaseException:
                if capacity_lease_id is not None:
                    await self._release_capacity(capacity_lease_id)
                raise
            lease = SandboxLease(child, capacity_lease_id, self._release_capacity, on_released=self._leases.discard)
            self._leases.add(lease)
            return lease

        snapshot_feature = {
            SnapshotKind.FILESYSTEM: SandboxFeature.FILESYSTEM_SNAPSHOT,
            SnapshotKind.FULL_STATE: SandboxFeature.FULL_STATE_SNAPSHOT,
        }[snapshot_kind]
        session.capabilities.require(snapshot_feature, SandboxFeature.RESTORE)
        snapshot = await self.checkpoint(session, snapshot_kind, policy)
        try:
            child = await self.restore(snapshot, state_policy=policy)
        except BaseException as restore_error:
            try:
                await self.delete_snapshot(snapshot)
            except BaseException as delete_error:
                restore_error.add_note(f"Temporary snapshot cleanup also failed: {delete_error!r}")
            raise
        try:
            await self.delete_snapshot(snapshot)
        except BaseException:
            await child.release()
            raise
        return child

    async def checkpoint(
        self,
        session: SandboxSession,
        kind: SnapshotKind,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SnapshotRef:
        """Create a capability-gated snapshot after enforcing RL safety policy."""
        policy = state_policy or (session.spec.state_policy if session.spec is not None else SandboxStatePolicy())
        self._validate_state_policy(session, policy)
        feature = {
            SnapshotKind.FILESYSTEM: SandboxFeature.FILESYSTEM_SNAPSHOT,
            SnapshotKind.FULL_STATE: SandboxFeature.FULL_STATE_SNAPSHOT,
        }[kind]
        session.capabilities.require(feature, SandboxFeature.RESTORE)
        try:
            snapshot = await session.snapshot(kind)
        except BaseException as snapshot_error:
            # Snapshotting invalidates command streams on E2B-compatible
            # providers. Reconnect the live parent before its next command.
            try:
                await session.refresh_transport()
            except BaseException as refresh_error:
                snapshot_error.add_note(f"Parent transport refresh also failed: {refresh_error!r}")
            raise
        try:
            await session.refresh_transport()
        except BaseException as refresh_error:
            try:
                await self.delete_snapshot(snapshot)
            except BaseException as delete_error:
                refresh_error.add_note(f"Snapshot cleanup also failed: {delete_error!r}")
            raise
        spec = session.spec
        metadata = dict(snapshot.metadata)
        metadata.update(
            {
                "psrl.network_connections_restored": False,
                "psrl.external_side_effects_restored": False,
            }
        )
        if spec is not None:
            metadata.update(
                {
                    "psrl.source_kind": spec.source.kind.value,
                    "psrl.source_reference": spec.source.reference,
                    "psrl.cpu_count": spec.resources.cpu_count,
                    "psrl.memory_mb": spec.resources.memory_mb,
                    "psrl.disk_mb": spec.resources.disk_mb,
                }
            )
        return replace(snapshot, metadata=metadata)

    @staticmethod
    def _validate_state_policy(session: SandboxSession, policy: SandboxStatePolicy) -> None:
        if not policy.enabled:
            raise RuntimeError("Sandbox state operations require an explicitly enabled SandboxStatePolicy.")
        spec = session.spec
        if spec is None:
            raise RuntimeError("Sandbox state safety cannot be verified for a session without its creation spec.")
        if not policy.allow_external_side_effects and session.command_count:
            raise RuntimeError(
                "Sandbox state capture after user commands requires allow_external_side_effects=True because external "
                "effects cannot be rolled back."
            )
        if policy.allow_secret_capture:
            return
        secret_markers = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "PRIVATE_KEY", "CREDENTIAL")
        sensitive_keys = [key for key in spec.env if any(marker in key.upper() for marker in secret_markers)]
        proxy_secrets = [
            key for key, value in spec.env.items() if "PROXY" in key.upper() and urlsplit(value).username is not None
        ]
        if sensitive_keys or proxy_secrets:
            names = ", ".join(sorted(set(sensitive_keys + proxy_secrets)))
            raise RuntimeError(f"Sandbox snapshot would capture secret-bearing environment variables: {names}.")

    @staticmethod
    async def _sanitize_restored_session(session: SandboxSession, policy: SandboxStatePolicy) -> None:
        """Reset stale transports and mix host entropy into a restored microVM."""
        try:
            await session.refresh_transport()
            if not policy.reseed_after_restore:
                return
            seed_path = f"/tmp/psrl-branch-seed-{secrets.token_hex(8)}"
            await session.write_bytes(seed_path, secrets.token_bytes(64))
            result = await session.exec(
                f"cat {seed_path} > /dev/urandom && rm -f {seed_path}",
                timeout_s=10,
            )
            if result.exit_code != 0:
                raise RuntimeError(f"Restored sandbox entropy reseed failed: {result.stderr.strip()}.")
        except BaseException:
            try:
                await session.terminate()
            except BaseException:
                pass
            raise

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Delete a snapshot through its owning backend."""
        await self.backend(snapshot.backend).delete_snapshot(snapshot)

    @staticmethod
    def _resources_from_snapshot(snapshot: SnapshotRef) -> ResourceSpec:
        """Recover the original resource request recorded by checkpoint()."""
        return ResourceSpec(
            cpu_count=snapshot.metadata.get("psrl.cpu_count"),
            memory_mb=snapshot.metadata.get("psrl.memory_mb"),
            disk_mb=snapshot.metadata.get("psrl.disk_mb"),
        )

    async def release(self, lease: SandboxLease) -> None:
        """Release one lease and remove it from worker ownership."""
        await lease.release()

    def metrics_snapshot(self) -> dict[str, SandboxMetricsSnapshot]:
        """Return one metrics snapshot per configured backend."""
        return {name: backend.metrics_snapshot() for name, backend in self._backends.items()}

    async def shutdown(self) -> None:
        """Terminate unreleased leases and close all backends."""
        if self._closed:
            return
        self._closed = True
        async with self._create_lock:
            create_tasks = [
                *(create_task.task for create_task in self._create_tasks.values()),
                *self._anonymous_create_tasks,
            ]
        leases = list(self._leases)
        results = await asyncio.gather(*(lease.release() for lease in leases), return_exceptions=True)
        # A create may wait for capacity held by an existing lease. Free those leases before
        # joining creates, then reclaim anything admitted while shutdown was in progress.
        create_results = await asyncio.gather(*create_tasks, return_exceptions=True)
        new_leases = self._leases.difference(leases)
        results.extend(await asyncio.gather(*(lease.release() for lease in new_leases), return_exceptions=True))
        self._leases.clear()
        self._idempotent_leases.clear()
        self._create_tasks.clear()
        self._anonymous_create_tasks.clear()
        backend_results = await asyncio.gather(
            *(backend.shutdown() for backend in self._backends.values()),
            return_exceptions=True,
        )
        shutdown_results = [*create_results, *results, *backend_results]
        if self._capacity_coordinator is not None:
            shutdown_results.extend(await asyncio.gather(self._close_capacity(), return_exceptions=True))
        errors = [result for result in shutdown_results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError(f"Sandbox manager shutdown had {len(errors)} failure(s).") from errors[0]
