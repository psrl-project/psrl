"""The client and the sandbox handle a caller actually uses.

Two channels, and the split is the whole performance argument.

- **Control plane**, over the service socket: create, release, snapshot, pause,
  and the reports. Roughly two calls per episode.
- **Data plane**, straight to the sandbox: commands and files. Dozens of calls
  per episode, and none of them needs a decision from the fleet, so none of them
  pays for a hop through it.

`create` returns the resolved agent endpoint with the handle, so the data plane
costs no extra round trip to discover.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections import deque
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

from sandboxd.errors import (
    SandboxCapabilityError,
    SandboxCapacityTimeout,
    SandboxCommandTimeout,
    SandboxError,
    SandboxPathNotFound,
    SandboxSessionLost,
    SandboxTransportError,
)
from sandboxd.types import (
    AgentEndpoint,
    Capabilities,
    ClassQuota,
    ExecResult,
    Feature,
    FleetReport,
    NodeReport,
    PauseMode,
    ProvisionStatus,
    QuotaReport,
    ResumeLevel,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
    SnapshotKind,
    SnapshotRef,
)

DEFAULT_ENDPOINT = "unix:///run/sandboxd.sock"

# The budget for a control call that is expected to answer promptly: a question, a
# report, or a reservation move. It is liveness, not a policy -- it is here so a
# dead service is reported rather than waited on, and it is not a knob a caller
# tunes. A create is deliberately not in this class: it may pull a cold image or
# try several nodes, so it carries no client deadline at all. The service bounds
# its own work; a client deadline shorter than that both cuts a working service
# and leaves a sandbox provisioning behind the client that gave up.
CALL_LIVENESS_S = 300.0

# How long an endpoint has to accept a connection. A socket that never opens is a
# dead service, which is worth reporting rather than waiting on forever.
CONNECT_LIVENESS_S = 30.0

# How long past a command's own deadline the caller keeps waiting for its reply.
# The deadline is enforced where the command runs, and its typed timeout reply is
# the result the caller acts on. Waiting exactly as long as the command may run
# races that reply: the client would give up at the same instant the sandbox stops
# the command, report a transport fault, and leave the caller unable to tell a
# stopped command from a broken service.
EXEC_REPLY_GRACE_S = 30.0

# How many connections one client keeps to one endpoint. The wire protocol
# matches replies to requests by order, so a connection serves one call at a
# time and concurrency is the pool's width. Sized for the concurrent episodes one
# worker runs; a burst wider than this waits for a connection rather than opening
# an unbounded number of sockets.
DEFAULT_POOL_SIZE = 32


class Transport:
    """A pool of connections to one service endpoint, speaking length-prefixed JSON.

    JSON rather than protobuf on the Python side: the SDK's job is to serialize
    and forward, and a generated stub would make the SDK's install depend on a
    protobuf runtime for no measurable gain. The wire shape is still the one the
    proto defines.

    A pool rather than one connection, because the wire protocol has no request
    ids: a reply is matched to a request by arrival order, so two callers sharing
    a connection would have to take turns. That used to be a lock around every
    call, which made the whole SDK serial -- one slow create (a cold image pull
    is minutes) held every other caller behind it until they hit their own
    deadline. Giving each in-flight call its own connection removes the queue
    without adding request ids to the protocol.

    Connections are reused rather than opened per call, so a steady workload pays
    the handshake once per pooled connection rather than once per command.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        self.endpoint = endpoint
        if pool_size < 1:
            raise ValueError(f"A transport pool needs at least one connection, got {pool_size}.")
        # Idle connections, available for reuse. A call takes one, uses it
        # exclusively, and returns it; the pool is a free-list, never a set of
        # shared connections.
        self._idle: deque[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = deque()
        # Bounds concurrent connections. A burst wider than the pool waits here
        # rather than opening an unbounded number of sockets against the service.
        self._slots = asyncio.Semaphore(pool_size)
        self._closed = False

    async def _open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if self.endpoint.startswith("unix://"):
            path = self.endpoint[len("unix://") :]
            return await asyncio.open_unix_connection(path)
        parsed = urlsplit(self.endpoint if "://" in self.endpoint else f"//{self.endpoint}")
        if parsed.hostname is None or parsed.port is None:
            raise SandboxTransportError(f"A service endpoint must be unix:// or host:port, got {self.endpoint!r}.")
        return await asyncio.open_connection(parsed.hostname, parsed.port)

    async def _acquire(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        """Take an idle connection, or open one when the pool has a free slot."""
        while self._idle:
            reader, writer = self._idle.popleft()
            if not writer.is_closing():
                return reader, writer
            # The peer closed it while it sat idle. Drop it and look further.
            writer.close()
        return await self._open()

    def _release(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Return a healthy connection to the pool."""
        if self._closed or writer.is_closing():
            writer.close()
            return
        self._idle.append((reader, writer))

    async def call(
        self, method: str, payload: Mapping[str, Any], *, timeout_s: float | None = CALL_LIVENESS_S
    ) -> dict[str, Any]:
        """Send one request on its own connection and return its reply.

        `timeout_s` bounds the wait for a reply and is liveness, not a policy:
        ``None`` waits however long the service takes, which is what a create
        needs. Connecting is always bounded, because an endpoint that never
        accepts a connection is a service that is not working.
        """
        async with self._slots:
            # Acquiring is inside the try: opening a fresh connection can fail the
            # same ways a live one can (the socket path is gone, the host refused
            # it), and a caller does not care which side of that line the failure
            # was on. Left outside, a bare OSError -- a FileNotFoundError for a
            # missing unix socket, say -- reached the caller unconverted, naming
            # neither the service nor the method that was ever reached for.
            try:
                reader, writer = await asyncio.wait_for(self._acquire(), timeout=CONNECT_LIVENESS_S)
            except asyncio.TimeoutError as exc:
                raise SandboxTransportError(
                    f"The sandbox service at {self.endpoint!r} did not accept a connection "
                    f"for {method!r} within {CONNECT_LIVENESS_S:g}s."
                ) from exc
            except (ConnectionError, OSError) as exc:
                raise SandboxTransportError(
                    f"The sandbox service at {self.endpoint!r} could not be reached for {method!r}: {exc}"
                ) from exc
            # A connection is reusable only when its reply was read in full. Every
            # other exit -- a timeout, a transport fault, or the caller cancelling
            # the task (which is a BaseException and matches no except below) --
            # leaves an unread reply or a dead socket, so the finally closes it
            # instead of pooling it. Without the finally, a cancelled heartbeat
            # leaked its connection and the service kept the mirror of it open.
            reusable = False
            try:
                roundtrip = self._roundtrip(reader, writer, method, payload)
                if timeout_s is None:
                    result = await roundtrip
                else:
                    result = await asyncio.wait_for(roundtrip, timeout=timeout_s)
                reusable = True
                return result
            except SandboxError:
                # A typed service error leaves the connection clean: the reply was
                # read in full, so the connection is still reusable. It is caught
                # before asyncio.TimeoutError on purpose: a command timeout is both
                # a typed service error and a TimeoutError, and the caller acts on
                # the type. Read as a transport fault, the sandbox survives a
                # command nobody can tell was stopped, and work it could finish is
                # abandoned.
                reusable = True
                raise
            except asyncio.TimeoutError as exc:
                # The service may still be acting on it, so this connection is
                # dropped rather than pooled with an unread reply in the pipe.
                raise SandboxTransportError(
                    f"The sandbox service did not answer {method!r} within {timeout_s:g}s."
                ) from exc
            except (ConnectionError, OSError, asyncio.IncompleteReadError) as exc:
                raise SandboxTransportError(f"The sandbox service connection failed during {method!r}: {exc}") from exc
            finally:
                if reusable:
                    self._release(reader, writer)
                else:
                    writer.close()

    async def _roundtrip(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        method: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        request = json.dumps({"method": method, "payload": dict(payload)}).encode()
        writer.write(len(request).to_bytes(4, "big") + request)
        await writer.drain()
        header = await reader.readexactly(4)
        body = await reader.readexactly(int.from_bytes(header, "big"))
        reply = json.loads(body)
        if error := reply.get("error"):
            raise _error_for(
                error.get("code", ""),
                error.get("message", ""),
                exit_reason=error.get("exit_reason") or "",
                sandbox_id=str(payload.get("sandbox_id") or ""),
                exit_code=error.get("exit_code"),
                finished_at=error.get("finished_at") or "",
                state_error=error.get("state_error") or "",
                preserved=bool(error.get("preserved", False)),
            )
        return reply.get("result") or {}

    async def close(self) -> None:
        """Drop every pooled connection."""
        self._closed = True
        idle, self._idle = self._idle, deque()
        for _, writer in idle:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                # The peer is already gone, which is the state close wanted.
                pass


def _error_for(
    code: str,
    message: str,
    *,
    exit_reason: str = "",
    sandbox_id: str = "",
    exit_code: int | None = None,
    finished_at: str = "",
    state_error: str = "",
    preserved: bool = False,
) -> SandboxError:
    """Rebuild a typed failure from the service's reply.

    The type is the service's classification, not a guess from the message: the
    caller's response differs per type, and an infrastructure fault read as a
    task result corrupts a reward. A lost session also carries the service's
    attribution -- reason, exit code, and finish time -- so a post mortem does
    not have to infer why it went.
    """
    if code == "session_lost":
        return SandboxSessionLost(
            message,
            sandbox_id=sandbox_id,
            exit_reason=exit_reason,
            exit_code=exit_code,
            finished_at=finished_at,
            state_error=state_error,
        )
    return {
        "capacity_timeout": SandboxCapacityTimeout,
        "cannot_admit": SandboxCapacityTimeout,
        "capability": SandboxCapabilityError,
        "command_timeout": lambda text: SandboxCommandTimeout(text, sandbox_preserved=preserved),
        "path_not_found": SandboxPathNotFound,
        "transport": SandboxTransportError,
    }.get(code, SandboxError)(message)


class Sandbox:
    """One sandbox. Commands go straight to it; lifecycle goes to the service."""

    def __init__(
        self,
        client: SandboxClient,
        handle: SandboxHandle,
        capabilities: Capabilities,
        agent: AgentEndpoint,
        *,
        warm_start: bool = False,
    ) -> None:
        self.handle = handle
        self.capabilities = capabilities
        self.agent = agent
        self.warm_start = warm_start
        self._client = client
        self._released = False

    @property
    def sandbox_id(self) -> str:
        """Return the backend's id for this sandbox."""
        return self.handle.sandbox_id

    @property
    def node_id(self) -> str:
        """Return the node holding this sandbox."""
        return self.handle.node_id

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
        """Run one command and collect its result.

        `silence_timeout_s` exists beside `timeout_s` because a command that
        prints nothing for a long stretch is stuck rather than slow, and the two
        need different answers.

        Raises:
            SandboxCommandTimeout: When the deadline expires. Its
                `sandbox_preserved` says whether the sandbox survived, which
                decides between continuing with a fresh shell and replacing it.
            SandboxOomError: When the kernel killed the sandbox mid-command.
        """
        return await self._client._exec(
            self.handle,
            self.agent,
            command,
            cwd=cwd,
            env=env,
            timeout_s=timeout_s,
            silence_timeout_s=silence_timeout_s,
        )

    async def read_bytes(self, path: str) -> bytes:
        """Read one file, without assuming a text encoding."""
        return await self._client._read_bytes(self.handle, self.agent, path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write one complete file."""
        await self._client._write_bytes(self.handle, self.agent, path, data)

    # -- Control plane ---------------------------------------------------------

    async def status(self) -> SandboxStatus:
        """Return the sandbox's portable state."""
        reply = await self._client._call("status", _handle_payload(self.handle))
        return SandboxStatus(reply.get("status", "unknown"))

    async def snapshot(self, kind: SnapshotKind = SnapshotKind.FILESYSTEM) -> SnapshotRef:
        """Capture this sandbox's state."""
        reply = await self._client._call("snapshot", {**_handle_payload(self.handle), "kind": kind.value})
        return SnapshotRef(
            backend=reply.get("backend", self.handle.backend),
            snapshot_id=reply["snapshot_id"],
            kind=SnapshotKind(reply.get("kind", kind.value)),
            resume_level=_resume_level(reply.get("resume_level")),
        )

    async def pause(self, mode: PauseMode = PauseMode.HIBERNATE) -> None:
        """Pause this sandbox with explicit semantics."""
        await self._client._call("pause", {**_handle_payload(self.handle), "mode": mode.value})

    async def resume(self) -> None:
        """Resume a paused sandbox."""
        await self._client._call("resume", _handle_payload(self.handle))

    async def release(self) -> None:
        """Destroy this sandbox and return everything it held.

        Idempotent: a retried release is normal, because the first may have timed
        out after the service already destroyed the sandbox.
        """
        if self._released:
            return
        await self._client._call("release", _handle_payload(self.handle))
        self._released = True

    async def __aenter__(self) -> Sandbox:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.release()


def node_plane_agent_factory(
    *,
    pool_size: int = DEFAULT_POOL_SIZE,
):
    """Return a factory that reaches each sandbox's owning node directly.

    This is what keeps command and file traffic off the control plane for a
    backend whose sandboxes run no agent of their own. A bare container is that
    case: there is no process inside it listening for commands, so something has
    to drive the runtime on its behalf, and the node that owns the daemon is the
    nearest thing that can. Routing those calls through the control plane instead
    costs a second hop and a second serialization on every command, which an
    episode pays dozens of times for no decision.

    The node plane speaks the same length-prefixed JSON as the control plane and
    already implements exec, read_bytes, and write_bytes, so the dial target is
    the only difference. One pool per node address, shared by every sandbox on
    that node: a burst on one node is a burst of calls to one host, and a pool per
    sandbox would spend a handshake on each.

    A sandbox whose create reply names no data-plane address falls back to the
    control plane by returning None, which is also what a provider backend's own
    agent protocol would do if this factory were replaced.
    """
    pools: dict[str, Transport] = {}

    def factory(agent: AgentEndpoint):
        address = getattr(agent, "data_plane", "") or ""
        if not address:
            return None
        pool = pools.get(address)
        if pool is None:
            pool = Transport(address, pool_size=pool_size)
            pools[address] = pool

        async def run(method: str, payload: Mapping[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
            return await pool.call(method, payload, timeout_s=timeout_s)

        return run

    return factory


class SandboxClient:
    """A connection to one sandbox service."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        pool_size: int = DEFAULT_POOL_SIZE,
        agent_factory: Any = None,
    ) -> None:
        self._transport = Transport(endpoint, pool_size=pool_size)
        # Returns a runner for a sandbox whose data plane is reachable directly,
        # or None to fall back to the control plane. The default reaches the
        # owning node's own plane, which is what keeps commands off the control
        # plane for a backend whose sandboxes run no agent of their own.
        #
        # Injected so a test can drive the data plane, and so a deployment whose
        # agents speak another protocol (execd, say) can supply its own.
        self._agent_factory = agent_factory or node_plane_agent_factory(pool_size=pool_size)
        # One pooled transport per node address, used only to report activity for
        # a sandbox whose commands bypass its node. Kept off the control-plane
        # transport because the two endpoints are different hosts.
        self._node_pool_size = pool_size
        self._node_pools: dict[str, Transport] = {}

    async def create(self, spec: SandboxSpec) -> Sandbox:
        """Provision one sandbox.

        Carries no client deadline: a create may pull a cold image or try several
        nodes, and the service bounds its own work. A client deadline here would
        both cut a working service and leave a sandbox provisioning behind it.

        Raises:
            SandboxCapabilityError: When no backend can serve the spec. Waiting
                will not add a capability, so this is a deployment fault.
            SandboxCapacityTimeout: When the fleet had no room in time. This
                clears as sandboxes are released.
        """
        reply = await self._call("create", {"spec": _spec_payload(spec)}, timeout_s=None)
        return self._sandbox_from(reply)

    async def create_group(self, specs: Sequence[SandboxSpec]) -> list[Sandbox]:
        """Provision one group.

        Members are admitted independently, because a group is a completion unit
        rather than a scheduling unit: the algorithm needs its trajectories to
        finish together, not to start together. A failed group leaves nothing
        charged.
        """
        if not specs:
            raise ValueError("A sandbox group needs at least one member spec.")
        reply = await self._call(
            "create_group", {"specs": [_spec_payload(spec) for spec in specs]}, timeout_s=None
        )
        return [self._sandbox_from(member) for member in reply.get("members", [])]

    async def restore(self, snapshot: SnapshotRef, spec: SandboxSpec | None = None) -> Sandbox:
        """Create a sandbox from a capture."""
        payload: dict[str, Any] = {"snapshot": _snapshot_payload(snapshot)}
        if spec is not None:
            payload["spec"] = _spec_payload(spec)
        return self._sandbox_from(await self._call("restore", payload, timeout_s=None))

    async def delete_snapshot(self, snapshot: SnapshotRef) -> None:
        """Delete a capture."""
        await self._call("delete_snapshot", _snapshot_payload(snapshot))

    async def fleet(self) -> FleetReport:
        """Return what the monitor sees, for a metric hook."""
        reply = await self._call("fleet", {})
        return FleetReport(
            nodes=tuple(
                NodeReport(
                    node_id=node.get("node_id", ""),
                    backends=tuple(node.get("backends", ())),
                    live_sandboxes=int(node.get("live_sandboxes", 0)),
                    cpu_used_pct=float(node.get("cpu_used_pct", 0.0)),
                    mem_used_pct=float(node.get("mem_used_pct", 0.0)),
                    gpu_free=int(node.get("gpu_free", 0)),
                    draining=bool(node.get("draining", False)),
                )
                for node in reply.get("nodes", [])
            ),
            drained_nodes=int(reply.get("drained_nodes", 0)),
            reservations_open=int(reply.get("reservations_open", 0)),
            capacity_exhausted=int(reply.get("capacity_exhausted", 0)),
            no_candidate=int(reply.get("no_candidate", 0)),
            locality_hit_ratio=float(reply.get("locality_hit_ratio", 0.0)),
        )

    async def fleet_detail(self) -> dict:
        """Return the fleet view exactly as the control plane reports it.

        `fleet()` is the metric hook: it names the handful of numbers a caller acts on.
        This is the operator's view -- every field the nodes publish, including each
        class's headroom and the envelope each node is admitting against. It is a raw
        mapping rather than a dataclass precisely because its job is to be complete: a
        telemetry consumer wants whatever the node publishes today, without this SDK
        deciding which of those fields are worth carrying.
        """
        return await self._call("fleet", {})

    async def diagnostics(self) -> dict:
        """Return the service's own health counters.

        Renewal failures and fan-out latency are facts only the control plane
        has: a node that stopped answering is invisible from the node side, and
        the heartbeat's caller only sees its own timeout. Report them beside the
        fleet so an operator can tell a quiet fleet from a broken liveness path.
        """
        return await self._call("diagnostics", {})

    async def contract(self) -> dict:
        """Return the mechanism's own timing spans.

        A client derives its heartbeat cadence and timeout from this instead of
        restating values that live in the service; the two drifting apart is how
        a client reported "the node reclaims after three beats" while the node
        used a different clock.
        """
        return await self._call("contract", {})

    async def provision_status(self, ticket: str) -> ProvisionStatus:
        """Return how a create with this ticket is progressing.

        A create carries no client deadline, so a caller that wants to know its
        request is being worked on rather than lost asks here. An unknown ticket
        reads as state ``unknown``, which is not the same as a failure.
        """
        reply = await self._call("provision_status", {"ticket": ticket})
        return ProvisionStatus(
            state=reply.get("state", "unknown"),
            elapsed_s=float(reply.get("elapsed_s", 0.0)),
            since_progress_s=float(reply.get("since_progress_s", 0.0)),
        )

    async def quota(self) -> QuotaReport:
        """Return the cross-backend ledger, for a metric hook."""
        reply = await self._call("quota", {})
        return QuotaReport(
            classes={
                name: ClassQuota(
                    guaranteed_share=float(entry.get("guaranteed_share", 0.0)),
                    max_share=float(entry.get("max_share", 0.0)),
                    granted_memory_mb=int(entry.get("granted_memory_mb", 0)),
                    headroom_memory_mb=int(entry.get("headroom_memory_mb", 0)),
                    queued=int(entry.get("queued", 0)),
                )
                for name, entry in (reply.get("classes") or {}).items()
            }
        )

    async def renew_owner(self, owner: str) -> None:
        """Tell the fleet this owner is still alive.

        Called on a timer while a run holds sandboxes. The control plane refreshes
        the owner's quota and forwards the renewal to every live node, so a node
        reclaims that owner's sandboxes only once the caller has really stopped
        renewing. A failed renewal is retried by the next heartbeat, so this
        reports nothing to the caller.
        """
        await self._call("renew_owner", {"owner": owner})

    async def close(self) -> None:
        """Close the connection to the service."""
        node_pools, self._node_pools = self._node_pools, {}
        for pool in node_pools.values():
            await pool.close()
        await self._transport.close()

    async def __aenter__(self) -> SandboxClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    # -- Internals -------------------------------------------------------------

    async def _call(
        self, method: str, payload: Mapping[str, Any], *, timeout_s: float | None = CALL_LIVENESS_S
    ) -> dict[str, Any]:
        return await self._transport.call(method, payload, timeout_s=timeout_s)

    def _sandbox_from(self, reply: Mapping[str, Any]) -> Sandbox:
        handle = reply.get("handle") or {}
        agent = reply.get("agent") or {}
        capabilities = reply.get("capabilities") or {}
        return Sandbox(
            self,
            SandboxHandle(
                backend=handle.get("backend", ""),
                sandbox_id=handle.get("sandbox_id", ""),
                node_id=handle.get("node_id", ""),
            ),
            Capabilities(
                features=frozenset(Feature(name) for name in capabilities.get("features", [])),
                resume_level=_resume_level(capabilities.get("resume_level")),
                pause_modes=frozenset(PauseMode(name) for name in capabilities.get("pause_modes", [])),
            ),
            AgentEndpoint(
                address=agent.get("address", ""),
                data_plane=agent.get("data_plane", ""),
                node_plane=agent.get("node_plane", ""),
                headers=agent.get("headers") or {},
                callback_host_alias=agent.get("callback_host_alias"),
                callback_port=agent.get("callback_port"),
            ),
            warm_start=bool(reply.get("warm_start", False)),
        )

    async def _exec(
        self,
        handle: SandboxHandle,
        agent: AgentEndpoint,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str] | None,
        timeout_s: float | None,
        silence_timeout_s: float | None,
    ) -> ExecResult:
        """Run a command, direct to the sandbox where it has its own agent.

        A backend whose sandboxes run no agent of their own -- a bare container
        is the case -- reports an empty address, and its node proxies instead.
        Either way the traffic stays node-local and off the control plane.
        """
        payload = {
            **_handle_payload(handle),
            "command": command,
            "cwd": cwd,
            "env": dict(env or {}),
            "timeout_s": timeout_s,
            "silence_timeout_s": silence_timeout_s,
        }
        reply = await self._run(agent, "exec", payload, timeout_s=_exec_liveness(timeout_s))
        # An agent-backed sandbox runs its own commands, so the node that owns it
        # never observed this one. Tell it the sandbox is still working; an
        # agentless sandbox's node proxied the command and already knows.
        if agent.node_plane and not agent.data_plane:
            await self._touch(agent, handle)
        return ExecResult(
            exit_code=int(reply["exit_code"]),
            stdout=reply.get("stdout", ""),
            stderr=reply.get("stderr", ""),
            truncated=bool(reply.get("truncated", False)),
        )

    async def _read_bytes(self, handle: SandboxHandle, agent: AgentEndpoint, path: str) -> bytes:
        reply = await self._run(
            agent, "read_bytes", {**_handle_payload(handle), "path": path}, timeout_s=CALL_LIVENESS_S
        )
        return base64.b64decode(reply["data"])

    async def _write_bytes(self, handle: SandboxHandle, agent: AgentEndpoint, path: str, data: bytes) -> None:
        await self._run(
            agent,
            "write_bytes",
            {**_handle_payload(handle), "path": path, "data": base64.b64encode(data).decode()},
            timeout_s=CALL_LIVENESS_S,
        )

    async def _run(
        self,
        agent: AgentEndpoint,
        method: str,
        payload: Mapping[str, Any],
        *,
        timeout_s: float | None,
    ) -> dict[str, Any]:
        """Dispatch one data-plane call to the sandbox, or to the node proxying it.

        A provider sandbox runs its own agent, so the call goes straight there. A
        bare container runs none, so the owning node proxies -- still one hop and
        still node-local, with the control plane out of the path either way.

        A command's deadline is its own budget and is enforced where the command
        runs, so it travels with the call rather than being capped here.
        """
        runner = self._agent_factory(agent)
        if runner is not None:
            return await runner(method, payload, timeout_s=timeout_s)
        return await self._transport.call(method, payload, timeout_s=timeout_s)

    async def _touch(self, agent: AgentEndpoint, handle: SandboxHandle) -> None:
        """Report activity to the node holding the sandbox, best effort.

        Only needed when the command did not pass through the node: a node that
        proxied it already saw the activity. A node that is briefly unreachable
        costs at most one idle window, so a failure here must not fail the
        command it accompanies -- the note is for the reclaimer, not for the
        caller's result.
        """
        address = agent.node_plane
        if not address:
            return
        try:
            pool = self._node_pools.get(address)
            if pool is None:
                pool = Transport(address, pool_size=self._node_pool_size)
                self._node_pools[address] = pool
            await pool.call("touch", _handle_payload(handle), timeout_s=CALL_LIVENESS_S)
        except Exception:
            # Best-effort by contract: see the docstring.
            pass


def _exec_liveness(timeout_s: float | None) -> float | None:
    """The liveness bound for one exec: the command's deadline plus a reply grace.

    A command with no deadline has no client bound either: it runs until it finishes or
    the caller cancels. See `EXEC_REPLY_GRACE_S` for why a bounded command's wait is
    not the command's own budget.
    """
    return None if timeout_s is None else timeout_s + EXEC_REPLY_GRACE_S


def _handle_payload(handle: SandboxHandle) -> dict[str, Any]:
    return {"backend": handle.backend, "sandbox_id": handle.sandbox_id, "node_id": handle.node_id}


def _snapshot_payload(snapshot: SnapshotRef) -> dict[str, Any]:
    return {"backend": snapshot.backend, "snapshot_id": snapshot.snapshot_id, "kind": snapshot.kind.value}


def _spec_payload(spec: SandboxSpec) -> dict[str, Any]:
    """Render a spec for the wire, omitting what the caller did not state.

    An omitted resource stays omitted rather than becoming zero: a dimension
    nobody named must not be compared against a limit.
    """
    payload: dict[str, Any] = {
        "source": {"kind": spec.source.kind, "reference": spec.source.reference},
        "resource_class": spec.resource_class,
    }
    resources = {
        name: value
        for name, value in (
            ("cpu_count", spec.resources.cpu_count),
            ("memory_mb", spec.resources.memory_mb),
            ("disk_mb", spec.resources.disk_mb),
            ("gpu_count", spec.resources.gpu_count),
        )
        if value is not None
    }
    if resources:
        payload["resources"] = resources
    for name, value in (
        ("workflow_id", spec.workflow_id),
        ("idempotency_key", spec.idempotency_key),
        ("workdir", spec.workdir),
        ("backend", spec.backend),
    ):
        if value:
            payload[name] = value
    if spec.env:
        payload["env"] = dict(spec.env)
    if spec.metadata:
        payload["metadata"] = dict(spec.metadata)
    if spec.exec_mode is not None:
        payload["exec_mode"] = spec.exec_mode.value
    if spec.required_features:
        payload["required_features"] = sorted(feature.value for feature in spec.required_features)
    if spec.required_resume_level is not None:
        payload["required_resume_level"] = spec.required_resume_level.value
    if spec.lifetime_s is not None:
        payload["lifetime_s"] = spec.lifetime_s
    if spec.backend_options:
        payload["backend_options"] = {name: dict(values) for name, values in spec.backend_options.items()}
    # A mount is part of the environment the sandbox is for, so it travels with the
    # request rather than being left for the caller to arrange afterwards: the service
    # decides which host paths this node will share, and refuses the rest.
    if spec.mounts:
        payload["mounts"] = [
            {"source": mount.source, "target": mount.target, "read_only": mount.read_only} for mount in spec.mounts
        ]
    return payload


def _resume_level(value: Any) -> ResumeLevel | None:
    if not value:
        return None
    return ResumeLevel(value)
