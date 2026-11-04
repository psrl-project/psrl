"""The p3b sandbox service as a PSRL backend.

PSRL has two sandbox paths. The in-process path (`docker`, `agentenv`,
`opensandbox`) runs sandbox logic inside the worker and charges the worker's
node envelope. This path does neither: it delegates every placement and
lifetime decision to the p3b service, and PSRL becomes a pure client.

The separation is explicit by design: p3b is an independently deployable
service that PSRL calls rather than a library it hosts. Nothing in this file
may import from other `psrl.sandbox` modules — it imports only from the
standalone `sandboxd` SDK and from `psrl.sandbox.core`, which holds the
abstract interfaces. The rest of `psrl.sandbox` is the in-process path and
must not bleed through here.

What this backend is responsible for:
- Translating between PSRL's SandboxSpec and the service's, refusing anything
  the service cannot represent rather than silently dropping it
- Wrapping the service's sandbox handle in a SandboxSession so the rest of
  PSRL never knows which path it is on
- Reporting fleet and quota state through the service's own reports, so PSRL's
  metric hooks see a consistent view of a fleet the service manages

What this backend is not responsible for:
- Placement, admission, or capacity — the service owns its ledger
- Reclamation — the service's sweep handles it; a worker exiting does not
  strand sandboxes
- Node capacity charging — uses_node_capacity is False; the service's ledger
  is the only admission that applies
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping
from typing import Any

from sandboxd import (
    ExecMode as ServiceExecMode,
)
from sandboxd import (
    Feature as ServiceFeature,
)
from sandboxd import (
    Mount as ServiceMount,
)
from sandboxd import (
    Resources as ServiceResources,
)
from sandboxd import (
    ResumeLevel as ServiceResumeLevel,
)
from sandboxd import Sandbox as ServiceSandbox
from sandboxd import SandboxClient
from sandboxd import SandboxSpec as ServiceSpec
from sandboxd import (
    SnapshotRef as ServiceSnapshotRef,
)
from sandboxd import (
    Source as ServiceSource,
)
from sandboxd import errors as service_errors

from psrl.sandbox.core import (
    ExecResult,
    PauseMode,
    ResourceUsage,
    ResumeLevel,
    SandboxBackend,
    SandboxCapabilities,
    SandboxCapabilityError,
    SandboxCapacityTimeout,
    SandboxCommandTimeout,
    SandboxExitReason,
    SandboxFeature,
    SandboxOomError,
    SandboxPathNotFoundError,
    SandboxRef,
    SandboxSession,
    SandboxSessionLostError,
    SandboxSetupError,
    SandboxSourceKind,
    SandboxSpec,
    SandboxStatus,
    SandboxTransportError,
    SnapshotKind,
    SnapshotRef,
)

psrl_logger = logging.getLogger(__name__)

# The service's default socket. A service per node uses this; a shared control
# plane is reached by host:port.
DEFAULT_ENDPOINT = "unix:///run/sandboxd.sock"


def _exit_reason(raw: str) -> SandboxExitReason:
    """Map the service's exit label onto PSRL's vocabulary, defaulting to UNKNOWN."""
    if not raw:
        return SandboxExitReason.UNKNOWN
    try:
        return SandboxExitReason(raw)
    except ValueError:
        return SandboxExitReason.UNKNOWN


def _translate(exc: service_errors.SandboxError) -> BaseException:
    """Map a service-side failure onto PSRL's neutral error vocabulary.

    The SDK raises the service's own types; PSRL's callers catch PSRL's. Without
    this translation the two sets never meet: a lost sandbox would surface as an
    unclassified exception instead of a lost session, and its attribution would be
    dropped, so a rollout could not be retried as `container_lost` nor a post
    mortem tell why it went.
    """
    if isinstance(exc, service_errors.SandboxSessionLost):
        return SandboxSessionLostError(
            str(exc),
            sandbox_id=exc.sandbox_id,
            exit_reason=_exit_reason(exc.exit_reason),
            exit_code=exc.exit_code,
            finished_at=exc.finished_at,
            state_error=exc.state_error,
        )
    if isinstance(exc, service_errors.SandboxCapacityTimeout):
        return SandboxCapacityTimeout(str(exc))
    if isinstance(exc, service_errors.SandboxCapabilityError):
        return SandboxCapabilityError(str(exc))
    if isinstance(exc, service_errors.SandboxOomError):
        return SandboxOomError(str(exc))
    if isinstance(exc, service_errors.SandboxCommandTimeout):
        return SandboxCommandTimeout(str(exc), sandbox_preserved=getattr(exc, "sandbox_preserved", False))
    if isinstance(exc, service_errors.SandboxPathNotFound):
        # A path-level fact, not a lost sandbox: the sandbox stays usable and a
        # caller that reads this as infrastructure loss replaces a healthy one.
        return SandboxPathNotFoundError(str(exc))
    if isinstance(exc, service_errors.SandboxTransportError):
        return SandboxTransportError(str(exc))
    return SandboxSetupError(str(exc))


async def _service_call(awaitable):
    """Await one SDK call, re-raising a service failure in PSRL's vocabulary."""
    try:
        return await awaitable
    except service_errors.SandboxError as exc:
        raise _translate(exc) from exc


class P3bSession(SandboxSession):
    """One sandbox held by the p3b service.

    Commands go to the sandbox's agent endpoint, which `create` already
    resolved. Lifecycle goes back to the service. This class holds no
    admission state of its own — the service is the source of truth.
    """

    def __init__(
        self,
        backend: P3bBackend,
        sandbox: ServiceSandbox,
        spec: SandboxSpec | None,
    ) -> None:
        self._backend = backend
        self._sandbox = sandbox
        self._spec = spec
        self._command_count = 0
        self._busy = False
        self._last_activity_at: float | None = None
        self._terminated = False
        self._exit_reason = SandboxExitReason.UNKNOWN

    # -- SandboxSession identity -----------------------------------------------

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef(backend=self._backend.name, sandbox_id=self._sandbox.sandbox_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        # The backend declares the fleet-level capabilities; per-sandbox
        # capabilities are not reported by the service yet.
        return self._backend.capabilities

    @property
    def spec(self) -> SandboxSpec | None:
        return self._spec

    # -- SandboxSession activity -----------------------------------------------

    @property
    def command_count(self) -> int:
        return self._command_count

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def last_activity_at(self) -> float | None:
        return self._last_activity_at

    @property
    def exit_reason(self) -> SandboxExitReason:
        return self._exit_reason

    # -- Node identity (not part of the SandboxSession contract) ---------------

    @property
    def node_id(self) -> str:
        """Return the node the service placed this sandbox on."""
        return self._sandbox.node_id

    # -- Data plane: straight to the sandbox -----------------------------------

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        silence_timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command straight to the sandbox agent endpoint.

        A service-side failure is re-raised in PSRL's vocabulary — a lost session
        carries its exit attribution, a command timeout keeps its
        `sandbox_preserved` flag — so the agent loop classifies the fault by type
        instead of seeing an unclassified exception.
        """
        self._busy = True
        try:
            result = await _service_call(
                self._sandbox.exec(
                    command,
                    cwd=cwd,
                    env=env,
                    timeout_s=timeout_s,
                    silence_timeout_s=silence_timeout_s,
                )
            )
        finally:
            self._busy = False
            self._command_count += 1
            self._last_activity_at = asyncio.get_running_loop().time()
        return ExecResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            truncated=result.truncated,
        )

    async def read_bytes(self, path: str) -> bytes:
        return await _service_call(self._sandbox.read_bytes(path))

    async def write_bytes(self, path: str, data: bytes) -> None:
        await _service_call(self._sandbox.write_bytes(path, data))

    # -- Control plane: back to the service ------------------------------------

    async def status(self) -> SandboxStatus:
        if self._terminated:
            return SandboxStatus.TERMINATED
        raw = await _service_call(self._sandbox.status())
        # The SDK returns its own `SandboxStatus` enum. `str()` on a `str, Enum`
        # member yields "SandboxStatus.RUNNING" in Python 3.12, not "running",
        # so converting it back through the constructor raises and made every
        # status call read as unreadable. Take the wire value instead.
        return SandboxStatus(getattr(raw, "value", raw))

    async def terminate(self) -> None:
        """Release the sandbox back to the service.

        Idempotent: a lease teardown and the service's reclamation sweep can
        both reach a session, and the second must not raise.
        """
        if self._terminated:
            return
        self._terminated = True
        self._exit_reason = SandboxExitReason.RELEASED
        await _service_call(self._sandbox.release())

    async def stats(self) -> ResourceUsage:
        """Return unknown usage.

        The service's fleet reports describe a node and a class, not one
        sandbox. Returning None fields rather than zeros: a metric that reads
        zero is indistinguishable from a sandbox that used nothing.
        """
        return ResourceUsage()

    async def snapshot(self, kind: SnapshotKind = SnapshotKind.FILESYSTEM) -> SnapshotRef:
        ref = await _service_call(self._sandbox.snapshot(kind))
        return SnapshotRef(
            backend=self._backend.name,
            snapshot_id=ref.snapshot_id,
            kind=SnapshotKind(str(ref.kind)),
            resume_level=ResumeLevel(str(ref.resume_level)) if ref.resume_level else None,
        )

    async def pause(self, mode: PauseMode = PauseMode.HIBERNATE) -> None:
        await _service_call(self._sandbox.pause(mode))

    async def resume(self) -> None:
        await _service_call(self._sandbox.resume())

    def resolve_callback_url(self, url: str) -> str:
        """Return the URL unchanged.

        A worker-loopback rewrite is a property of where the sandbox runs
        relative to the worker. The service places the sandbox — possibly on
        another node — so no alias this worker names is guaranteed to resolve
        there. A deployment that needs the worker reachable from the sandbox
        gives the worker a routable address, not a loopback.
        """
        return url


# How often this worker tells the service its owner is still alive.
#
# The node reclaims an owner's sandboxes once it has been silent for three of
# these, so the interval is the unit that budget is counted in: a shorter one
# reclaims a killed worker's sandboxes sooner, a longer one tolerates a stalled
# process before doing so. It is stated here rather than derived because the two
# sides are separate deployments, and it must stay below a third of the node's
# silence budget (`timing.OwnerSilenceTTL` in p3b).
OWNER_HEARTBEAT_INTERVAL_S = 30.0

# The node reclaims an owner's sandboxes after this many missed beats, so the interval
# above is a third of the silence budget by construction. Named here because the renewal
# loop has to be able to say how close it is to losing everything.
_OWNER_SILENCE_BEATS = 3

# A renewal that has not come back in this long is abandoned rather than waited on. It has
# to be well under one beat: a call that is allowed to run for the whole interval cannot
# be distinguished from a loop that has stopped running.
OWNER_RENEW_TIMEOUT_S = 10.0

# How many consecutive misses before the renewal stops being a warning. Two, because the
# budget is three: at two the next failure is the one that takes the sandboxes, and there
# is still one beat left in which something can be done about it.
_RENEWAL_MISSES_BEFORE_ALARM = 2


class P3bBackend(SandboxBackend):
    """The p3b service as a PSRL backend.

    Selecting this backend switches PSRL from in-process sandbox management
    to the p3b service. The service holds the quota ledger, placement, and
    reclamation; PSRL is a client.

    Configuration in psrl_rollout.yaml:

        sandbox:
          default_backend: p3b
          backends:
            p3b:
              _target_: psrl.sandbox.backends.P3bBackend
              endpoint: unix:///run/sandboxd.sock
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        name: str = "p3b",
        resource_class: str = "rollout",
    ) -> None:
        self._name = name
        self._endpoint = endpoint
        self._resource_class = resource_class
        # No client deadline on a create: provisioning may pull a cold image, and
        # the service bounds its own work. The client keeps only its internal
        # liveness for calls expected to answer promptly.
        self._client = SandboxClient(endpoint)
        # The renewal gets its own connection pool. A create holds a control
        # connection for its whole duration (it has no client deadline), so a
        # burst of creates can occupy every slot in the shared pool; a heartbeat
        # that then waits on a slot reports a missed beat while the service is
        # answering every request promptly.
        self._renew_client: SandboxClient | None = None
        self._owner_id = ""
        self._closed = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._contract_loaded = False
        self._heartbeat_interval_s = OWNER_HEARTBEAT_INTERVAL_S
        self._silence_ttl_s = OWNER_HEARTBEAT_INTERVAL_S * _OWNER_SILENCE_BEATS
        self._renew_timeout_s = OWNER_RENEW_TIMEOUT_S
        # Renewal health, exposed as metrics beside the fleet report: the
        # caller's own timeouts are the only signal that a renewal is not
        # reaching a node, and they must not exist only as log lines.
        self._renew_attempts = 0
        self._renew_failures = 0
        self._renew_consecutive_misses = 0
        self._renew_last_duration_s = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def uses_node_capacity(self) -> bool:
        """Return False: the service's ledger is the only admission that applies.

        Charging the worker's node envelope as well would refuse a sandbox
        twice for the same memory.
        """
        return False

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare the union of what a fleet running all four backends can serve.

        A union rather than an intersection, because the service routes a spec
        to whichever backend can meet it. A spec no backend can serve is
        refused at admission, and that refusal names what was missing — so a
        capability claimed here and absent from every deployed backend fails
        with a usable message rather than silently degrading.

        Operators running a Docker-only fleet should not reduce this set here:
        the service's own admission is the gate. Reducing this set would only
        cause PSRL to refuse before asking the service.
        """
        return SandboxCapabilities(
            features=frozenset(
                {
                    SandboxFeature.FREEZE,
                    SandboxFeature.HIBERNATE,
                    SandboxFeature.FILESYSTEM_SNAPSHOT,
                    SandboxFeature.FULL_STATE_SNAPSHOT,
                    SandboxFeature.RESTORE,
                    SandboxFeature.NATIVE_FORK,
                    SandboxFeature.RESUME_ANYWHERE,
                    SandboxFeature.WARM_POOL,
                    SandboxFeature.IMAGE_ON_DEMAND,
                    # The service answers a mount per node: it shares the host trees its
                    # operator listed and refuses the rest, so a deployment that allows
                    # none advertises none and the caller is refused before it asks.
                    SandboxFeature.HOST_MOUNT,
                }
            ),
            resume_level=ResumeLevel.FULL_STATE,
        )

    def adopt_owner_id(self, owner_id: str) -> None:
        """Record the owner for this worker's sandboxes.

        Carried as spec metadata for correlation, and named by the heartbeat
        that follows. This runs while the manager is built — which can be before
        an event loop exists — so it only records the id; the renewal loop
        starts with the first sandbox, in `create`, where a loop is running.
        """
        if not self._owner_id:
            self._owner_id = owner_id

    async def _load_contract(self) -> None:
        """Read the service's timing spans once, so this client stops guessing.

        A failure here is not a create failure: the client keeps a conservative
        default and says once that it did. The values it would have read govern
        a liveness probe, not the sandbox the caller is asking for.
        """
        if self._contract_loaded:
            return
        self._contract_loaded = True
        try:
            contract = await self._client.contract()
        except Exception as exc:  # noqa: BLE001 - a liveness detail must not fail a create
            psrl_logger.warning(
                "Could not read the sandbox service's timing contract; using the client defaults: %r",
                exc,
            )
            return
        heartbeat = float(contract.get("owner_heartbeat_interval_s") or self._heartbeat_interval_s)
        silence = float(contract.get("owner_silence_ttl_s") or self._silence_ttl_s)
        if heartbeat > 0 and silence > 0:
            self._heartbeat_interval_s = heartbeat
            self._silence_ttl_s = silence
            # Never wait longer than a third of the silence budget: a renewal
            # allowed to run that long cannot be told apart from a loop that has
            # stopped, and the next beat would carry the loss.
            self._renew_timeout_s = min(OWNER_RENEW_TIMEOUT_S, max(1.0, silence / 3.0))

    def _ensure_owner_heartbeat(self) -> None:
        """Start renewing this worker's ownership of the sandboxes it holds.

        The node reclaims an owner's sandboxes once it has stopped hearing from
        the owner, and this renewal is the only evidence it has: a sandbox's own
        lifetime is a backstop measured in hours, which is far longer than a
        killed worker should be allowed to strand its containers. Idempotent, so
        every sandbox this worker starts keeps the same single loop.
        """
        if self._owner_id and (self._heartbeat_task is None or self._heartbeat_task.done()):
            if self._renew_client is None:
                self._renew_client = SandboxClient(self._endpoint, pool_size=4)
            task = asyncio.create_task(self._renew_owner_loop())
            task.add_done_callback(self._heartbeat_done)
            self._heartbeat_task = task

    def _heartbeat_done(self, task: asyncio.Task[None]) -> None:
        """Report a renewal loop that stopped without being asked to.

        A loop that dies silently is indistinguishable from a control plane
        that is not answering: both end with the node reclaiming every sandbox
        this worker holds, and neither leaves a line saying which one it was.
        """
        if self._closed or task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            psrl_logger.error(
                "Sandbox owner renewal loop stopped unexpectedly; the node will "
                "reclaim every sandbox this worker holds after %.0fs: %r",
                self._silence_ttl_s,
                exc,
            )

    async def _renew_owner_loop(self) -> None:
        # Three properties this loop has to have, each of which it did not:
        #
        # 1. **A hung renewal must not eat the next beat.** The sleep came first and the
        #    call was awaited inline with no deadline, so one slow renewal pushed every
        #    later one out by however long it took. Under a loaded control plane that is
        #    exactly the shape that lets a node's silence budget expire.
        # 2. **The cadence is a period, not a gap.** Sleep the remainder of the interval
        #    rather than the whole interval after the call, so the beats are evenly spaced
        #    however long the call itself took.
        # 3. **Failures have to be visible before they become reclamations.** A single
        #    miss is harmless -- the node's budget is three beats wide -- but a run of
        #    them means this worker is about to lose every sandbox it holds, and that was
        #    previously only discoverable from the 409 the next exec happened to hit.
        missed = 0
        while not self._closed:
            started = time.monotonic()
            client = self._renew_client or self._client
            self._renew_attempts += 1
            try:
                async with asyncio.timeout(self._renew_timeout_s):
                    await client.renew_owner(self._owner_id)
                if missed:
                    psrl_logger.info(
                        "Sandbox owner renewal recovered for owner %r after %d missed beat(s).",
                        self._owner_id,
                        missed,
                    )
                missed = 0
                self._renew_consecutive_misses = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                missed += 1
                self._renew_failures += 1
                self._renew_consecutive_misses = missed
                # One missed beat costs nothing; the second means one more will cost every
                # sandbox this worker holds. Say so at the level that gets read.
                log = psrl_logger.error if missed >= _RENEWAL_MISSES_BEFORE_ALARM else psrl_logger.warning
                beats = max(1, int(round(self._silence_ttl_s / max(self._heartbeat_interval_s, 0.001))))
                log(
                    "Sandbox owner renewal failed for owner %r (%d consecutive miss(es), "
                    "the node reclaims after %d): %r. Every sandbox this worker holds is at risk.",
                    self._owner_id,
                    missed,
                    beats,
                    exc,
                )
            finally:
                self._renew_last_duration_s = time.monotonic() - started
            remaining = self._heartbeat_interval_s - (time.monotonic() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        self._require_open()
        await self._load_contract()
        self._ensure_owner_heartbeat()
        service_spec = _to_service_spec(spec, self._owner_id, self._resource_class)
        if service_spec.backend:
            # The service's backend names are not PSRL's manager keys. A pin that
            # reaches this line names a runtime the service may not host.
            psrl_logger.warning(
                "P3bBackend forwards a backend pin %r with idempotency key %r.",
                service_spec.backend,
                spec.idempotency_key,
            )
        sandbox = await _service_call(self._client.create(service_spec))
        return P3bSession(self, sandbox, spec)

    async def connect(self, sandbox_id: str) -> SandboxSession:
        """Not supported: the agent endpoint only travels with a create reply.

        A worker that lost its session asks for a new sandbox. The service's
        reclamation sweep owns the lifetime of orphaned sandboxes.
        """
        raise NotImplementedError(
            "P3bBackend does not support connect(): the agent endpoint is only "
            "available in the create reply. Request a new sandbox instead."
        )

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> SandboxSession:
        self._require_open()
        await self._load_contract()
        self._ensure_owner_heartbeat()
        service_spec = _to_service_spec(spec, self._owner_id, self._resource_class) if spec is not None else None
        sandbox = await _service_call(
            self._client.restore(
                ServiceSnapshotRef(
                    backend=snapshot.backend,
                    snapshot_id=snapshot.snapshot_id,
                    kind=str(snapshot.kind.value if hasattr(snapshot.kind, "value") else snapshot.kind),
                ),
                service_spec,
            )
        )
        return P3bSession(self, sandbox, spec)

    async def prepare(self, spec: SandboxSpec) -> None:
        """No-op: the service manages its own warm pools and image pulls.

        A caller that prepares is asking to pay the cost early. There is no
        early to pay from here — the service does it on demand.
        """
        return None

    async def shutdown(self) -> None:
        """Stop renewing this worker's ownership and close the connection.

        Sandboxes outlive the connection in the sense that the service holds
        their lifetime, but not in the sense of staying charged to a worker that
        is gone: stopping the renewals here is what tells the node this owner has
        left, so it reclaims them once the silence budget runs out rather than
        holding them until their own lifetime expires.
        """
        if self._closed:
            return
        self._closed = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        if self._renew_client is not None:
            await self._renew_client.close()
            self._renew_client = None
        await self._client.close()

    # -- Metric passthroughs ---------------------------------------------------

    async def fleet_report(self) -> dict[str, Any]:
        """Return the service's fleet view for PSRL's metric hooks.

        Not part of the SandboxBackend contract, but callable by any code that
        knows it is talking to a P3bBackend and wants to report fleet state.
        Delegates directly to the SDK's FleetReport.as_metrics() for a flat
        metrics dict, and includes per-node state for debugging.
        """
        self._require_open()
        report = await _service_call(self._client.fleet())
        result = report.as_metrics()
        result["nodes"] = [
            {
                "node_id": n.node_id,
                # NodeReport carries every runtime the node hosts, not one: a
                # machine can serve a spec through its container daemon and
                # another through a microVM, and naming one would hide the other.
                "backends": list(n.backends),
                "live_sandboxes": n.live_sandboxes,
                "cpu_used_pct": n.cpu_used_pct,
                "mem_used_pct": n.mem_used_pct,
                "draining": n.draining,
            }
            for n in (report.nodes or [])
        ]
        return result

    async def diagnostics(self) -> dict[str, Any]:
        """Return the service's own health counters, for PSRL's metric hooks."""
        self._require_open()
        return await _service_call(self._client.diagnostics())

    def heartbeat_report(self) -> dict[str, float]:
        """Return this backend's renewal health as flat metrics.

        Renewal failures are otherwise only visible as repeated log lines on the
        worker; a metric makes "this run's heartbeats never reached a node" a
        number beside the fleet's, instead of a paragraph someone has to notice.
        """
        return {
            "sandbox/p3b/renew/attempts": float(self._renew_attempts),
            "sandbox/p3b/renew/failures": float(self._renew_failures),
            "sandbox/p3b/renew/consecutive_misses": float(self._renew_consecutive_misses),
            "sandbox/p3b/renew/last_duration_s": self._renew_last_duration_s,
            "sandbox/p3b/renew/interval_s": self._heartbeat_interval_s,
            "sandbox/p3b/renew/silence_ttl_s": self._silence_ttl_s,
        }

    async def quota_report(self) -> dict[str, Any]:
        """Return the service's quota view for PSRL's metric hooks.

        Delegates to QuotaReport.as_metrics() for a flat metrics dict.
        """
        self._require_open()
        report = await _service_call(self._client.quota())
        return report.as_metrics()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Sandbox backend {self._name!r} is closed.")


# -- Module-level helpers (no PSRL sandbox state, pure conversion) -------------


# Which of a PSRL spec's fields this conversion carries, and which it does not.
# They make the boundary auditable, and the exhaustiveness test below enforces it.
_FORWARDED_FIELDS = frozenset(
    {
        "source",
        "resources",
        "resource_class",
        "workflow_id",
        "idempotency_key",
        "env",
        "metadata",
        "workdir",
        "exec_mode",
        "required_features",
        "required_resume_level",
        "backend",
        "lifetime_timeout_s",
    }
)

# Refused, because a sandbox that quietly lost one would run and produce a wrong result.
_REFUSED_FIELDS = frozenset(
    {
        "mounts",
        "volumes",
        "egress",
        "credentials",
        "credential_bindings",
        "assigned_gpus",
    }
)

# Dropped, with the reason. None of them changes what the service does.
_NOT_FORWARDED_FIELDS = {
    "idle_timeout_s": (
        "the service bounds a sandbox by its lifetime and its owner's silence and has no "
        "idle concept, and the in-process Docker backend ignores this field too"
    ),
    "policy_profile": "a backend-specific profile, consumed only by the e2b backend",
    "state_policy": "PSRL's session layer decides what a snapshot means, not the service",
}

_OUT_OF_SCOPE_FIELDS = _REFUSED_FIELDS | frozenset(_NOT_FORWARDED_FIELDS)


def _to_service_spec(
    spec: SandboxSpec,
    owner_id: str,
    default_resource_class: str,
) -> ServiceSpec:
    """Convert a PSRL SandboxSpec into the service's.

    Every field of a PSRL spec is forwarded, refused, or named as deliberately
    not forwarded; `_FORWARDED_FIELDS`, `_REFUSED_FIELDS` and
    `_NOT_FORWARDED_FIELDS` say which, and the exhaustiveness test fails when
    PSRL gains a field that none of them accounts for.

    Refused rather than dropped, because a sandbox that quietly lost a bind
    mount, an egress policy, or injected credentials would run and produce a
    wrong result, while one that was refused costs a configuration fix. The same
    reasoning is why the exec mode and the required features are forwarded: this
    backend silently served one-shot commands and unchecked features for a
    workload that had declared it needed otherwise.
    """
    if spec.volumes:
        raise ValueError(
            "P3bBackend does not forward provider volumes to the service. "
            "Declare the volume in the service configuration instead."
        )
    if spec.egress is not None:
        raise ValueError(
            "P3bBackend does not forward an egress policy to the service. "
            "Declare the policy in the service configuration instead."
        )
    if spec.credentials or spec.credential_bindings:
        raise ValueError(
            "P3bBackend does not forward injected credentials to the service. "
            "Declare the credentials in the service configuration instead."
        )
    if spec.assigned_gpus:
        raise ValueError(
            "A p3b sandbox is placed by the service, which assigns its own "
            "GPU indices on the node it chose. A worker cannot pin host GPU "
            "indices for a remotely placed sandbox."
        )

    metadata = dict(spec.metadata)
    if owner_id:
        # The name the node reclaims this sandbox by: it holds that owner's
        # sandboxes only while the owner keeps renewing, and the renewal names
        # the same string. It travels as metadata because it is an opaque caller
        # identity the service only compares for equality.
        metadata.setdefault("owner_id", owner_id)

    return ServiceSpec(
        source=ServiceSource(
            kind=_source_kind_str(spec.source.kind),
            reference=spec.source.reference,
        ),
        resources=ServiceResources(
            cpu_count=spec.resources.cpu_count,
            memory_mb=spec.resources.memory_mb,
            disk_mb=spec.resources.disk_mb,
            gpu_count=spec.resources.gpu_count,
        ),
        resource_class=spec.resource_class or default_resource_class,
        workflow_id=spec.workflow_id,
        idempotency_key=spec.idempotency_key,
        env=dict(spec.env),
        metadata=metadata,
        # The container's starting directory, so a command that names none works where
        # the workload said it works.
        workdir=spec.workdir,
        # Fixed at creation and not selectable per command, so a dropped mode serves
        # every persistent-shell workload one-shot instead.
        exec_mode=ServiceExecMode(spec.exec_mode.value) if spec.exec_mode is not None else None,
        # The service refuses a backend that cannot provide one, which is the point of
        # declaring them. Passing them on is what makes that refusal possible.
        required_features=frozenset(ServiceFeature(feature.value) for feature in spec.required_features),
        required_resume_level=(
            ServiceResumeLevel(spec.required_resume_level.value) if spec.required_resume_level is not None else None
        ),
        # A recipe's pin, forwarded because the service validates the requirements
        # against the pinned backend rather than trusting it.
        backend=spec.backend,
        # How long this sandbox is meant to work, which the caller knows and the node
        # cannot infer.
        lifetime_s=int(spec.lifetime_timeout_s) if spec.lifetime_timeout_s else None,
        # Host paths this sandbox needs, forwarded so the node decides: it shares the
        # trees its operator listed and refuses everything else. A path only exists on
        # the machine that holds it, so this is the one request that depends on where
        # the sandbox lands -- which is exactly why the node answers it rather than
        # this client guessing. A fleet whose nodes share no storage refuses here, and
        # that refusal names the path, which is the answer a caller can act on.
        mounts=tuple(
            ServiceMount(source=mount.source, target=mount.target, read_only=mount.read_only) for mount in spec.mounts
        ),
    )


def _source_kind_str(kind: SandboxSourceKind) -> str:
    return str(kind.value if hasattr(kind, "value") else kind)
