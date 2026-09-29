"""Synchronous facade for thread-based sandbox consumers."""

import asyncio
import logging
import threading
from collections.abc import Callable, Mapping

from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
    SandboxCapabilities,
    SandboxFeature,
    SandboxRef,
    SandboxSpec,
    SandboxStatePolicy,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)
from psrl.sandbox.manager import SandboxLease, SandboxManager
from psrl.sandbox.metrics import SandboxMetricsSnapshot
from psrl.utils.common.async_utils import run_coroutine_on_loop

psrl_logger = logging.getLogger(__name__)


class SyncSandboxSession:
    """Synchronous view of one worker-owned sandbox session."""

    def __init__(
        self,
        lease: SandboxLease,
        manager: SandboxManager,
        event_loop: asyncio.AbstractEventLoop,
        on_close: Callable[[SandboxLease], None] | None = None,
    ) -> None:
        self._lease = lease
        self._manager = manager
        self._event_loop = event_loop
        self._on_close = on_close
        self._close_lock = threading.Lock()
        self._closed = False

    @property
    def ref(self) -> SandboxRef:
        """Return the stable backend reference."""
        return self._lease.ref

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Return semantic capabilities implemented by the session."""
        return self._lease.session.capabilities

    def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        """Execute a command on the event loop that owns the session."""
        return run_coroutine_on_loop(
            self._event_loop,
            self._lease.session.exec(command, cwd=cwd, env=env, timeout_s=timeout_s),
            timeout_s=None if timeout_s is None else timeout_s + 10,
        )

    def read_bytes(self, path: str) -> bytes:
        """Read a file through the owning event loop."""
        return run_coroutine_on_loop(self._event_loop, self._lease.session.read_bytes(path))

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write a file through the owning event loop."""
        run_coroutine_on_loop(self._event_loop, self._lease.session.write_bytes(path, data))

    def status(self) -> SandboxStatus:
        """Return current session status."""
        return run_coroutine_on_loop(self._event_loop, self._lease.session.status())

    def stats(self) -> ResourceUsage:
        """Return current resource usage."""
        return run_coroutine_on_loop(self._event_loop, self._lease.session.stats())

    def resolve_callback_url(self, url: str) -> str:
        """Translate a worker URL into one reachable from the sandbox."""
        return self._lease.session.resolve_callback_url(url)

    def pause(self, mode: PauseMode = PauseMode.HIBERNATE) -> None:
        """Pause with explicit freeze or hibernate semantics."""
        feature = {
            PauseMode.FREEZE: SandboxFeature.FREEZE,
            PauseMode.HIBERNATE: SandboxFeature.HIBERNATE,
        }[mode]
        self.capabilities.require(feature)
        run_coroutine_on_loop(self._event_loop, self._lease.session.pause(mode))

    def resume(self) -> None:
        """Resume a paused session."""
        run_coroutine_on_loop(self._event_loop, self._lease.session.resume())

    def snapshot(
        self,
        kind: SnapshotKind = SnapshotKind.FULL_STATE,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SnapshotRef:
        """Create a capability-gated snapshot with RL safety checks."""
        return run_coroutine_on_loop(
            self._event_loop,
            self._manager.checkpoint(self._lease.session, kind, state_policy),
        )

    def close(self) -> None:
        """Release the session exactly once through the owning event loop."""
        with self._close_lock:
            if self._closed:
                return
            run_coroutine_on_loop(self._event_loop, self._lease.release())
            self._closed = True
            if self._on_close is not None:
                self._on_close(self._lease)


class SyncSandboxManager:
    """Create sandbox sessions for synchronous code running in worker threads.

    The facade retains the worker's asynchronous manager and event loop instead
    of creating a per-task loop or backend client. It also owns sessions until
    they close, allowing the asynchronous caller to reclaim abandoned work
    after a thread timeout.
    """

    def __init__(
        self,
        manager: SandboxManager,
        event_loop: asyncio.AbstractEventLoop,
        *,
        backend: str | None = None,
    ) -> None:
        self._manager = manager
        self._event_loop = event_loop
        self._backend = backend
        self._leases: set[SandboxLease] = set()
        self._lock = threading.Lock()
        self._closed = False

    def create(self, spec: SandboxSpec) -> SyncSandboxSession:
        """Create a sandbox from synchronous worker-thread code."""
        self._require_open()
        lease = run_coroutine_on_loop(self._event_loop, self._manager.acquire(spec, backend=self._backend))
        return self._adopt(lease)

    def restore(
        self,
        snapshot: SnapshotRef,
        spec: SandboxSpec | None = None,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SyncSandboxSession:
        """Restore a microVM snapshot and own the returned lease."""
        self._require_open()
        lease = run_coroutine_on_loop(
            self._event_loop,
            self._manager.restore(snapshot, spec, state_policy=state_policy),
        )
        return self._adopt(lease)

    def branch(
        self,
        session: SyncSandboxSession,
        state_policy: SandboxStatePolicy | None = None,
    ) -> SyncSandboxSession:
        """Fork or snapshot/restore according to backend capabilities."""
        self._require_open()
        lease = run_coroutine_on_loop(
            self._event_loop,
            self._manager.branch(session._lease.session, state_policy=state_policy),
        )
        return self._adopt(lease)

    def _require_open(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Synchronous sandbox manager is closed.")

    def _adopt(self, lease: SandboxLease) -> SyncSandboxSession:
        """Register a lease or release it when close won the create race."""
        with self._lock:
            if not self._closed:
                self._leases.add(lease)
                return SyncSandboxSession(lease, self._manager, self._event_loop, self._discard)
        run_coroutine_on_loop(self._event_loop, lease.release())
        raise RuntimeError("Synchronous sandbox manager closed while creating a session.")

    def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Delete a provider-owned snapshot."""
        run_coroutine_on_loop(self._event_loop, self._manager.delete_snapshot(snapshot))

    def _discard(self, lease: SandboxLease) -> None:
        with self._lock:
            self._leases.discard(lease)

    async def aclose(self) -> None:
        """Reclaim sessions abandoned by failed or timed-out synchronous code."""
        with self._lock:
            self._closed = True
            leases = list(self._leases)
            self._leases.clear()
        results = await asyncio.gather(*(self._manager.release(lease) for lease in leases), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            psrl_logger.warning(f"Synchronous sandbox cleanup had {len(failures)} failure(s).")

    def metrics_snapshot(self) -> dict[str, SandboxMetricsSnapshot]:
        """Return backend metrics without crossing the event loop."""
        return self._manager.metrics_snapshot()
