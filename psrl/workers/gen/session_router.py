import asyncio
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import quote

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from psrl.utils.common.http_utils import (
    SESSION_TURN_ACCOUNTING_KEY,
    SESSION_TURN_BUDGET_ERROR_CODE,
    SESSION_TURN_BUDGET_HEADER,
    HttpResponse,
    create_aiohttp_client,
    filter_http_headers,
    request_raw,
)
from psrl.utils.rollout.turn_output_writer import TurnOutputWriter
from psrl.utils.sampler_distribution import SamplerCapture, configure_capture
from psrl.utils.subagent_training import SubagentTrainingPolicy
from psrl.workers.gen.harness.protocol import (
    anthropic_error_body,
    anthropic_error_response,
    anthropic_stream_response,
    openai_stream_response,
    responses_stream_response,
)
from psrl.workers.gen.smg_adapter import TITO_SESSIONS_PATH, TRAJECTORY_ID_STRATEGIES

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Hang-state values (see SessionState.hang_state).
SESSION_RUNNING = "running"
SESSION_HUNG = "hung"

# Session status values, derived from inflight: a trajectory is either inferring
# on vLLM/SMG (generate) or between turns / calling the environment (env).
STATUS_GENERATE = "generate"
STATUS_ENV = "env"

SESSION_ID_HEADER = "x-smg-tito-session-id"
TRAJECTORY_ID_HEADER = "x-smg-tito-trajectory-id"

# Codex marks every model call with its own turn metadata. `thread_source` separates a
# subagent's calls from the parent's, which is what lets a session be held to a
# parent-agent budget without charging a subagent's work to it.
CODEX_TURN_METADATA_HEADER = "x-codex-turn-metadata"
CODEX_SUBAGENT_THREAD_SOURCE = "subagent"
# Claude Code labels each request with its request class and agent identity when
# CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 is set.
CLAUDE_REQUEST_CLASS_HEADER = "x-claude-code-request-class"
CLAUDE_AGENT_ID_HEADER = "x-claude-code-agent-id"
CLAUDE_PARENT_AGENT_ID_HEADER = "x-claude-code-parent-agent-id"
CLAUDE_MAIN_REQUEST_CLASSES = frozenset({"main"})
CLAUDE_SUBAGENT_REQUEST_CLASSES = frozenset({"subagent", "workflow"})
CLAUDE_AUXILIARY_REQUEST_CLASSES = frozenset({"auxiliary", "side_reply"})

# Normalized agent headers the router owns. A harness may send its native labels, but
# only the router decides what the gateway sees.
PSRL_AGENT_ROLE_HEADER = "x-psrl-agent-role"
PSRL_AGENT_ID_HEADER = "x-psrl-agent-id"
PSRL_PARENT_AGENT_ID_HEADER = "x-psrl-parent-agent-id"
PSRL_TITO_CAPTURE_HEADER = "x-psrl-tito-capture"
PSRL_AGENT_HEADERS = (
    PSRL_AGENT_ROLE_HEADER,
    PSRL_AGENT_ID_HEADER,
    PSRL_PARENT_AGENT_ID_HEADER,
    PSRL_TITO_CAPTURE_HEADER,
)
AGENT_ROLES = frozenset({"main", "subagent", "auxiliary", "compaction"})


def _first_metadata_value(metadata: Mapping[str, object], *keys: str) -> str | None:
    """Return the first non-empty metadata value among ``keys`` as a string."""
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


@dataclass(slots=True)
class SessionState:
    """Local concurrency state for one TITO session."""

    headers: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    closing: bool = False
    inflight: int = 0
    # Per-trajectory turn tracking: trajectory_id → current_turn
    trajectory_turns: dict[int, int] = field(default_factory=dict)
    base_worker_id: str | None = None
    target_dp_rank: str | None = None
    # Subsequent turns require a version at least as fresh as the first serving
    # instance.
    version_tag: str | None = None
    # --- Hang/continue scheduling (ThunderAgent port) ---
    # Accumulated token footprint of the session (max prompt+completion observed).
    total_tokens: int = 0
    # "running" | "hung": whether the coordinator has hung this session.
    hang_state: str = SESSION_RUNNING
    # In-flight hang requests take effect at the next turn boundary.
    marked_for_hang: bool = False
    # Set means "may proceed". A hung session blocks at the next turn entry until
    # the coordinator continues it (sets the event). Initialized set in __post_init__.
    continue_event: asyncio.Event = field(default_factory=asyncio.Event)
    # A continue target pins only the next turn. `None` leaves routing to SMG.
    pin_once_instance: tuple[str, str] | None = None
    # The turn budget this session declared when it was created, and what it has spent.
    # `main_turns` is the parent agent's model calls, which is what the budget bounds: a
    # harness that counts its own turns counts the parent's, so charging a subagent's
    # calls to the same budget would make one configuration mean two different numbers.
    max_turns: int | None = None
    main_turns: int = 0
    subagent_turns: int = 0
    # How agent trajectories in this session are selected and captured. Immutable after
    # creation; the gateway snapshots the same policy into the session view.
    subagent_training: SubagentTrainingPolicy = field(default_factory=SubagentTrainingPolicy)
    # Identity of the parent agent, used to attribute compaction calls that do not carry
    # their own agent id to the parent.
    main_agent_id: str = "main"
    # Whether a main call has been seen. A later main-labeled call with a different
    # native id is a second thread, not a second artifact.
    main_agent_seen: bool = False
    # Fallback identity for one unidentified subagent call. Without a native id,
    # concurrent calls cannot be grouped into one agent.
    synthetic_agent_index: int = 0
    # Whether the main episode has closed. Later subagent turns are background.
    main_closed: bool = False
    # In-flight model calls per (role, agent id).
    agent_inflight: dict[tuple[str, str], int] = field(default_factory=dict)
    # Live background subagents observed at the main-close boundary.
    background_live_count: int = 0
    # Set when no main-path request is in flight; background work is ignored.
    main_drained: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        if self.inflight == 0:
            self.drained.set()
        else:
            self.drained.clear()
        # A fresh session is running: allow it to proceed.
        self.continue_event.set()
        self.main_drained.set()

    @property
    def status(self) -> str:
        """generate while a turn is in flight, else env (between turns)."""
        return STATUS_GENERATE if self.inflight > 0 else STATUS_ENV

    def get_trajectory_turn(self, trajectory_id: int) -> int:
        """Get the current turn for a trajectory, initializing to 0 if new."""
        return self.trajectory_turns.get(trajectory_id, 0)

    def advance_trajectory_turn(self, trajectory_id: int) -> None:
        """Advance the turn counter for a specific trajectory."""
        current = self.trajectory_turns.get(trajectory_id, 0)
        self.trajectory_turns[trajectory_id] = current + 1

    @property
    def turn_budget_exhausted(self) -> bool:
        """Whether this session has spent the turn budget it declared."""
        return self.max_turns is not None and self.main_turns >= self.max_turns

    def charge_turn(self, *, is_subagent: bool) -> None:
        """Charge one model call to this session."""
        if is_subagent:
            self.subagent_turns += 1
        else:
            self.main_turns += 1

    def is_background(self, role: str, agent_id: str | None) -> bool:
        """Whether this in-flight call is background work after main close."""
        if not self.main_closed:
            return False
        if role == "subagent":
            return True
        if role == "compaction":
            return agent_id not in (None, "", "main", self.main_agent_id)
        return False

    @property
    def main_inflight(self) -> int:
        """Model calls on the main path; background work is excluded."""
        background = sum(
            count
            for (role, agent_id), count in self.agent_inflight.items()
            if count > 0 and self.is_background(role, agent_id)
        )
        return max(0, self.inflight - background)

    def refresh_main_drained(self) -> None:
        """Update the event the delete path waits on."""
        if self.main_inflight == 0:
            self.main_drained.set()
        else:
            self.main_drained.clear()


class SessionRouter:
    def __init__(
        self,
        smg_url: str,
        client_concurrency: int = 1024,
        trajectory_id_strategy: str = "manual",
        turn_output_writer: TurnOutputWriter | None = None,
        sampler_capture: SamplerCapture | None = None,
    ):
        trajectory_id_strategy = trajectory_id_strategy.lower()
        if trajectory_id_strategy not in TRAJECTORY_ID_STRATEGIES:
            choices = ", ".join(sorted(TRAJECTORY_ID_STRATEGIES))
            raise ValueError(f"Invalid trajectory_id_strategy {trajectory_id_strategy!r}; expected one of: {choices}.")
        self.smg_url = smg_url.rstrip("/")
        self.app = FastAPI()
        self.client: aiohttp.ClientSession | None = None
        self.client_concurrency = client_concurrency
        self.trajectory_id_strategy = trajectory_id_strategy
        self.turn_output_writer = turn_output_writer
        # Resolve eagerly so a bad mode or budget fails at construction, not on the first request.
        self.sampler_capture = sampler_capture or SamplerCapture()
        _ = self.sampler_capture.width
        self.states: dict[str, SessionState] = {}
        self.states_lock = asyncio.Lock()
        self.setup_routes()
        self.app.router.on_shutdown.append(self.aclose)

    def setup_routes(self):
        self.app.post("/sessions")(self.create_session)
        self.app.get("/sessions/{sid}")(self.get_session)
        self.app.post("/sessions/{sid}/main-close")(self.main_close)
        self.app.delete("/sessions/{sid}")(self.delete_session)
        self.app.post("/sessions/{sid}/v1/chat/completions")(self.session_chat_completions)
        self.app.post("/sessions/{sid}/v1/responses")(self.session_responses)
        self.app.post("/sessions/{sid}/v1/messages/count_tokens")(self.session_anthropic_count_tokens)
        self.app.post("/sessions/{sid}/v1/messages")(self.session_messages)
        # Coordinator-facing hang/continue control plane.
        self.app.get("/control/sessions")(self.control_list_sessions)
        self.app.post("/control/hang")(self.control_hang)
        self.app.post("/control/continue")(self.control_continue)
        self.app.api_route(
            "/sessions/{sid}/{path:path}",
            methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        )(self.session_proxy)

    async def aclose(self):
        """Close the router-owned aiohttp client."""
        if self.client is not None and not self.client.closed:
            await self.client.close()

    async def create_session(self, request: Request) -> Response:
        raw_body = await request.body()
        try:
            body = json.loads(raw_body) if raw_body else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            return JSONResponse(status_code=400, content={"error": {"message": "request body must be valid JSON"}})
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "request body must be an object"}})
        try:
            subagent_training = SubagentTrainingPolicy.from_mapping(body.get("subagent_training"))
        except (TypeError, ValueError) as error:
            return JSONResponse(status_code=400, content={"error": {"message": str(error)}})

        result = await self._request_upstream("POST", TITO_SESSIONS_PATH)
        if result.status < 400:
            session_id = self.extract_session_id(result)
            state = await self._ensure_state(session_id)
            async with state.lock:
                state.headers = self.session_headers(request.headers, session_id)
                state.max_turns = self.parse_turn_budget(request.headers)
                state.subagent_training = subagent_training
        return self.build_response(result)

    @staticmethod
    def parse_turn_budget(headers: Mapping[str, str]) -> int | None:
        """Read the model-call budget a session declared, or None when it declared none.

        Only a harness whose CLI takes no cap of its own declares one, so a session that
        declares nothing is unbounded here and bounded by its CLI if it has one.
        """
        raw = headers.get(SESSION_TURN_BUDGET_HEADER)
        if raw is None:
            return None
        try:
            budget = int(str(raw).strip())
        except ValueError:
            psrl_logger.warning(f"Ignoring an unreadable {SESSION_TURN_BUDGET_HEADER} value: {raw!r}.")
            return None
        return budget if budget > 0 else None

    @staticmethod
    def classify_request(headers: Mapping[str, str]) -> tuple[str, str | None, str | None]:
        """Normalize a harness's native agent markers into one role and identity.

        Codex labels every call with turn metadata and Claude Code labels calls with a
        request class. A caller that sends neither is the parent because an
        unattributable call must still be bounded by the parent's budget.

        Args:
            headers (Mapping[str, str]): Request headers.

        Returns:
            tuple[str, str | None, str | None]: Role, native agent id, native parent id.
        """
        raw_metadata = headers.get(CODEX_TURN_METADATA_HEADER)
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except (json.JSONDecodeError, UnicodeDecodeError):
                metadata = None
            if isinstance(metadata, dict):
                source = str(metadata.get("thread_source") or "").strip().lower()
                if source == CODEX_SUBAGENT_THREAD_SOURCE:
                    role = "subagent"
                elif source == "compaction":
                    role = "compaction"
                elif source in ("auxiliary", "side_reply"):
                    role = "auxiliary"
                else:
                    role = "main"
                agent_id = _first_metadata_value(
                    metadata,
                    "agent_id",
                    "agentId",
                    "thread_id",
                    "threadId",
                )
                parent_id = _first_metadata_value(
                    metadata,
                    "parent_agent_id",
                    "parentAgentId",
                    "parent_thread_id",
                    "parentThreadId",
                )
                return role, agent_id, parent_id

        request_class = headers.get(CLAUDE_REQUEST_CLASS_HEADER)
        if request_class:
            value = request_class.strip().lower()
            if value in CLAUDE_SUBAGENT_REQUEST_CLASSES:
                role = "subagent"
            elif value == "compaction":
                role = "compaction"
            elif value in CLAUDE_AUXILIARY_REQUEST_CLASSES:
                role = "auxiliary"
            elif value in CLAUDE_MAIN_REQUEST_CLASSES:
                role = "main"
            else:
                role = "main"
            return (
                role,
                headers.get(CLAUDE_AGENT_ID_HEADER),
                headers.get(CLAUDE_PARENT_AGENT_ID_HEADER),
            )

        # An older Claude Code runtime emits the agent ids without the request class.
        # A main call has no agent id; an auxiliary call has an id but no parent.
        claude_agent_id = headers.get(CLAUDE_AGENT_ID_HEADER)
        if claude_agent_id:
            claude_parent_id = headers.get(CLAUDE_PARENT_AGENT_ID_HEADER)
            return ("subagent" if claude_parent_id else "auxiliary"), claude_agent_id, claude_parent_id

        role = str(headers.get(PSRL_AGENT_ROLE_HEADER) or "main").strip().lower()
        if role not in AGENT_ROLES:
            role = "main"
        return role, headers.get(PSRL_AGENT_ID_HEADER), headers.get(PSRL_PARENT_AGENT_ID_HEADER)

    def is_subagent_request(self, request: Request) -> bool:
        """Whether this model call is a subagent's rather than the parent agent's."""
        role, _, _ = self.classify_request(request.headers)
        return role == "subagent"

    def resolve_agent_identity(
        self,
        state: SessionState,
        headers: Mapping[str, str],
    ) -> tuple[str, str | None, str | None, str]:
        """Resolve the role, identity, parent, and capture mode for one model call.

        This runs once per call, before the request is buffered, so the sampler capture
        and the capture header sent to the gateway are decided from the same value. The
        caller must hold ``state.lock`` because the first main call and an unidentified
        subagent update session state.

        Args:
            state (SessionState): The session state.
            headers (Mapping[str, str]): Request headers.

        Returns:
            tuple[str, str | None, str | None, str]: Role, agent id, parent agent id,
                and capture mode (``full`` or ``tokens``).
        """
        role, agent_id, parent_agent_id = self.classify_request(headers)
        if role == "main":
            second_main_thread = (
                state.main_agent_seen and agent_id is not None and agent_id != state.main_agent_id
            )
            if second_main_thread:
                # A second non-subagent thread is not a second primary. A declared
                # parent makes it a subagent; without one it is auxiliary.
                role = "subagent" if parent_agent_id else "auxiliary"
            else:
                if agent_id:
                    state.main_agent_id = agent_id
                state.main_agent_seen = True
                state.main_closed = False
                agent_id = state.main_agent_id
                parent_agent_id = None
        if role == "compaction" and agent_id in (None, "", state.main_agent_id):
            # A compaction of the parent has no parent of its own; attribute it even
            # when the harness omits the agent id.
            agent_id = state.main_agent_id
            parent_agent_id = None
        elif role != "main":
            if role == "subagent" and not agent_id:
                state.synthetic_agent_index += 1
                agent_id = f"subagent:{state.synthetic_agent_index}"
            if parent_agent_id is None:
                parent_agent_id = state.main_agent_id

        is_validate = state.headers.get("x-is-validate") == "true"
        capture = state.subagent_training.capture_for(
            role,
            agent_id,
            validate=is_validate,
            main_agent_id=state.main_agent_id,
        )
        if is_validate:
            # Validation never trains; token ids are enough to score the session.
            capture = "tokens"
        return role, agent_id, parent_agent_id, capture

    async def charge_session_turn(self, state: SessionState, request: Request) -> HttpResponse | None:
        """Charge this model call to its session, refusing it once the budget is spent.

        Charged on the attempt rather than on success: the budget bounds what a session
        may ask for, and a call that fails and is retried is another ask. Once the parent
        agent's budget is spent nothing more runs, so a subagent's call is refused too.

        Returns the response to send instead of running the turn, or None to run it.
        """
        is_subagent = self.is_subagent_request(request)
        async with state.lock:
            if state.turn_budget_exhausted:
                psrl_logger.debug(
                    f"Session {state.headers.get(SESSION_ID_HEADER, '?')!r} spent its turn budget "
                    f"(main_turns={state.main_turns}, subagent_turns={state.subagent_turns}); refusing this turn."
                )
                return self.turn_budget_response(state)
            state.charge_turn(is_subagent=is_subagent)
        return None

    @staticmethod
    def turn_budget_response(state: SessionState) -> HttpResponse:
        """Build the refusal a session gets when its turn budget is spent.

        It carries the code in the body because a harness CLI echoes an error body
        verbatim into its own output, and that is where the framework reads an episode's
        stop reason from. The header repeats it for callers that classify on it, the way
        the gateway's own sentinels are read.
        """
        return HttpResponse(
            status=400,
            body=json.dumps(
                {
                    "error": {
                        "type": "invalid_request_error",
                        "code": SESSION_TURN_BUDGET_ERROR_CODE,
                        "message": (
                            f"this session's turn budget ({state.max_turns}) is spent; "
                            "stop working and report what you have"
                        ),
                    }
                },
                ensure_ascii=False,
            ).encode(),
            headers={"content-type": "application/json", "x-smg-error-code": SESSION_TURN_BUDGET_ERROR_CODE},
        )

    async def get_session(self, sid: str, request: Request) -> Response:
        path = f"{TITO_SESSIONS_PATH}/{sid}"
        snapshot_format = request.query_params.get("format")
        if snapshot_format:
            path = f"{path}?format={quote(snapshot_format, safe='')}"
        result = await self._request_upstream("GET", path)
        state = self.states.get(sid)
        if state is not None:
            async with state.lock:
                if state.base_worker_id is not None:
                    result.headers["x-base-worker-id"] = state.base_worker_id
                if state.target_dp_rank is not None:
                    result.headers["x-target-dp-rank"] = state.target_dp_rank
                result = self.with_session_state(result, state)
        return self.build_response(result)

    async def main_close(self, sid: str, request: Request) -> Response:
        """Close the main episode and count live background subagents.

        This is an observation boundary, not a drain: it never waits for a
        subagent. The count is a lower bound from in-flight calls unless the
        caller reports a harness lifecycle count in the body.
        """
        state = await self._ensure_state(sid)
        reported: int | None = None
        raw_body = await request.body()
        if raw_body:
            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = None
            if isinstance(body, dict) and body.get("background_subagents") is not None:
                try:
                    reported = max(0, int(body["background_subagents"]))
                except (TypeError, ValueError):
                    reported = None
        async with state.lock:
            state.main_closed = True
            observed = sum(
                1
                for (role, agent_id), count in state.agent_inflight.items()
                if count > 0 and state.is_background(role, agent_id)
            )
            state.background_live_count = max(
                observed, reported if reported is not None else 0
            )
            state.refresh_main_drained()
            count = state.background_live_count
        psrl_logger.info(
            "TITO session %r main closed with %d live background subagent(s).", sid, count
        )
        return JSONResponse(content={"background_live_count": count})

    @staticmethod
    def with_session_state(result: HttpResponse, state: SessionState) -> HttpResponse:
        """Add this router's turn accounting and agent policy to a session view.

        A body that will not parse is passed through untouched: the session view is the
        gateway's, and its shape is not this router's to guarantee.
        """
        try:
            body = result.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return result
        if not isinstance(body, dict):
            return result
        body[SESSION_TURN_ACCOUNTING_KEY] = {
            "max_turns": state.max_turns,
            "main_turns": state.main_turns,
            "subagent_turns": state.subagent_turns,
            "exhausted": state.turn_budget_exhausted,
        }
        # The policy is resolved once at session creation; extraction reads it from the
        # snapshot rather than from the trainer's current config. `main_agent_id` is
        # session state, not policy: it names the parent a compaction call belongs to.
        body["subagent_training"] = state.subagent_training.to_dict()
        body["main_agent_id"] = state.main_agent_id
        return HttpResponse(
            status=result.status,
            body=json.dumps(body, ensure_ascii=False).encode(),
            headers=result.headers,
        )

    async def delete_session(self, sid: str) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            state.closing = True
            # Unblock any turn hung at the hang point so the session can drain
            # and be torn down instead of deadlocking on continue_event.
            state.hang_state = SESSION_RUNNING
            state.marked_for_hang = False
            state.continue_event.set()

        # Wait only for the main path. Background work cannot contribute a
        # training row, and waiting for it would hold the session open.
        while True:
            async with state.lock:
                if state.main_inflight == 0:
                    break
                main_drained = state.main_drained
            await main_drained.wait()

        result = await self._request_upstream("DELETE", f"{TITO_SESSIONS_PATH}/{sid}")

        async with self.states_lock:
            if self.states.get(sid) is state:
                self.states.pop(sid, None)
        return self.build_response(result)

    async def session_chat_completions(self, sid: str, request: Request) -> Response:
        """Handle Chat Completions, synthesizing SSE after a completed turn."""
        prepared = await self.prepare_protocol_request(request, "chat", sid)
        if isinstance(prepared, Response):
            return prepared
        content, stream = prepared

        result, response_body = await self.run_protocol_turn(
            sid,
            request,
            content,
            upstream_path="v1/chat/completions",
        )
        if stream and result.status < 400:
            if response_body is None:
                return JSONResponse(
                    status_code=502,
                    content={"error": {"message": "backend returned an invalid chat completion response"}},
                )
            return openai_stream_response(response_body, headers=result.headers)
        return self.build_response(result)

    async def session_responses(self, sid: str, request: Request) -> Response:
        """Proxy Codex Responses natively, buffering one complete training turn."""
        prepared = await self.prepare_protocol_request(request, "responses", sid)
        if isinstance(prepared, Response):
            return prepared
        content, stream = prepared
        result, response_body = await self.run_protocol_turn(
            sid,
            request,
            content,
            upstream_path="v1/responses",
        )
        if result.status >= 400:
            return self.build_response(result)

        if stream:
            if response_body is None:
                return JSONResponse(
                    status_code=502,
                    content={
                        "error": {"type": "api_error", "message": "backend returned an invalid Responses response"}
                    },
                )
            return responses_stream_response(response_body, headers=result.headers)
        return self.build_response(result)

    async def session_messages(self, sid: str, request: Request) -> Response:
        """Proxy Anthropic Messages natively, buffering one complete training turn."""
        prepared = await self.prepare_protocol_request(request, "messages", sid)
        if isinstance(prepared, Response):
            return prepared
        content, stream = prepared
        result, response_body = await self.run_protocol_turn(
            sid,
            request,
            content,
            upstream_path="v1/messages",
        )
        if result.status >= 400:
            return anthropic_error_response(result)

        if stream:
            if response_body is None:
                return JSONResponse(
                    status_code=502,
                    content=anthropic_error_body(502, "backend returned an invalid Messages response"),
                )
            return anthropic_stream_response(response_body, headers=result.headers)
        return self.build_response(result)

    async def prepare_protocol_request(
        self,
        request: Request,
        protocol: str,
        sid: str,
    ) -> tuple[bytes, bool] | Response:
        """Buffer one JSON request and force a completed upstream turn."""
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            message = "request body must be valid JSON"
            error = (
                anthropic_error_body(400, message)
                if protocol == "messages"
                else {"error": {"type": "invalid_request_error", "message": message}}
            )
            return JSONResponse(status_code=400, content=error)

        if not isinstance(body, dict):
            message = "request body must be an object"
            error = (
                anthropic_error_body(400, message)
                if protocol == "messages"
                else {"error": {"type": "invalid_request_error", "message": message}}
            )
            return JSONResponse(status_code=400, content=error)

        stream = body.get("stream") is True
        body["stream"] = False
        state = await self._ensure_state(sid)
        async with state.lock:
            identity = self.resolve_agent_identity(state, request.headers)
        request.state.psrl_agent = identity
        capture_mode = identity[3]
        # An untrained trajectory needs only token ids, so the engine does no sampler work.
        capture = self.sampler_capture if capture_mode == "full" else SamplerCapture()
        try:
            # Always apply the capture contract: a disabled capture still sets the OpenAI logprob
            # fields TITO needs, while Anthropic Messages only carries sampler_top_logprobs.
            configure_capture(body, capture, protocol=protocol)
        except (ValueError, TypeError) as error:
            return JSONResponse(status_code=400, content={"error": str(error)})
        content = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        return content, stream

    async def session_anthropic_count_tokens(self, sid: str, request: Request) -> Response:
        """Return the optional Claude Code token-count hint.

        Counting must use the backend tokenizer to be exact. The harness treats
        this endpoint as an advisory hint, so returning zero avoids a second
        untracked tokenization path and matches the other training adapters.
        """
        await self._ensure_state(sid)
        await request.body()
        return JSONResponse(content={"input_tokens": 0})

    async def run_protocol_turn(
        self,
        sid: str,
        request: Request,
        content: bytes,
        *,
        upstream_path: str,
    ) -> tuple[HttpResponse, Mapping[str, object] | None]:
        """Run one completed protocol turn with shared TITO bookkeeping."""
        state = await self._ensure_state(sid)

        # This is where a model call is made, so it is where a declared turn budget is
        # enforced. Charging happens before the hang wait: a session that is out of turns
        # must not be held open waiting to be continued.
        refusal = await self.charge_session_turn(state, request)
        if refusal is not None:
            return refusal, None

        # SMG owns trajectory resolution in auto mode, so the SessionRouter only
        # tracks caller-selected IDs in manual mode.
        trajectory_id = (
            int(request.headers.get(TRAJECTORY_ID_HEADER, "0")) if self.trajectory_id_strategy == "manual" else None
        )

        # Wait outside the lock so the coordinator can resume a hung session.
        while True:
            async with state.lock:
                if state.closing:
                    return self._json_result(409, {"error": "session is closing"}), None
                if state.hang_state != SESSION_HUNG:
                    break
                continue_event = state.continue_event
            psrl_logger.debug(f"Session blocked at hang point: session_id={sid!r}, trajectory_id={trajectory_id!r}.")
            await continue_event.wait()

        async with state.lock:
            if state.closing:
                return self._json_result(409, {"error": "session is closing"}), None
            session_headers = state.headers.copy()
            base_worker_id = state.base_worker_id
            target_dp_rank = state.target_dp_rank
            version_tag = state.version_tag
            # Consume the one-shot pin (if any) so only this turn is force-pinned.
            pin_once_instance = state.pin_once_instance
            state.pin_once_instance = None
            agent_identity = getattr(request.state, "psrl_agent", None)
            if agent_identity is None:
                agent_identity = self.resolve_agent_identity(state, request.headers)
            role, agent_id, parent_agent_id, capture = agent_identity
            agent_key = (role, agent_id or "")
            state.agent_inflight[agent_key] = state.agent_inflight.get(agent_key, 0) + 1
            state.inflight += 1
            state.drained.clear()
            state.refresh_main_drained()

        headers = self.add_session_headers(request, session_headers)
        headers["accept"] = "application/json"
        headers["content-type"] = "application/json"
        # The router owns the normalized agent headers. A client may send them, but
        # only the native harness markers are trusted and only the router decides
        # whether a call is trainable.
        for header_name in PSRL_AGENT_HEADERS:
            headers.pop(header_name, None)
        headers[PSRL_AGENT_ROLE_HEADER] = role
        if agent_id:
            headers[PSRL_AGENT_ID_HEADER] = agent_id
        if parent_agent_id:
            headers[PSRL_PARENT_AGENT_ID_HEADER] = parent_agent_id
        headers[PSRL_TITO_CAPTURE_HEADER] = capture
        if base_worker_id is not None:
            headers["x-base-worker-id"] = base_worker_id
        if target_dp_rank is not None:
            headers["x-target-dp-rank"] = target_dp_rank
        if version_tag is not None:
            headers["x-version-tag"] = version_tag
        # The one-shot pin expires before any partial-rollout redispatch.
        if pin_once_instance is not None:
            headers["x-base-worker-id"] = pin_once_instance[0]
            headers["x-target-dp-rank"] = pin_once_instance[1]
            headers["x-force-pin-once"] = "true"
            psrl_logger.debug(f"Force-pinned session turn: session_id={sid!r}, instance={pin_once_instance!r}.")
        turn_index = state.get_trajectory_turn(trajectory_id or 0)

        result: HttpResponse | None = None
        response_body: Mapping[str, object] | None = None
        try:
            result = await self._request_upstream(
                "POST",
                upstream_path,
                content=content,
                headers=headers,
            )
            if result.status < 400:
                response_body = result.json()
        finally:
            # Record the turn before bookkeeping so an upstream error (e.g. a
            # context-overflow 400) is captured alongside the request that caused it.
            if self.turn_output_writer is not None:
                self.turn_output_writer.write_turn(
                    session_id=sid,
                    turn=turn_index,
                    request_body=content,
                    response_body=result.body if result is not None else None,
                    trajectory_id=trajectory_id,
                    status=result.status if result is not None else None,
                )
            # Single combined critical section: close out inflight bookkeeping
            # and, on success, advance the trajectory's turn counter.
            async with state.lock:
                state.inflight = max(0, state.inflight - 1)
                remaining = state.agent_inflight.get(agent_key, 0) - 1
                if remaining > 0:
                    state.agent_inflight[agent_key] = remaining
                else:
                    state.agent_inflight.pop(agent_key, None)
                if state.inflight == 0:
                    state.drained.set()
                state.refresh_main_drained()
                if result is not None and result.status < 400:
                    base_worker_id = result.headers.get("x-base-worker-id")
                    target_dp_rank = result.headers.get("x-target-dp-rank")
                    if base_worker_id is not None and target_dp_rank is not None:
                        state.base_worker_id = base_worker_id
                        state.target_dp_rank = target_dp_rank
                    version_tag = result.headers.get("x-version-tag")
                    if version_tag is not None:
                        state.version_tag = version_tag

                    # `usage.prompt_tokens` already includes the accumulated TITO context, so
                    # prompt plus completion is the live session footprint.
                    self._update_total_tokens(state, response_body)

                    if trajectory_id is not None:
                        state.advance_trajectory_turn(trajectory_id)

                # Deferred-hang conversion: a hang requested mid-turn takes effect
                # now that this trajectory has returned and the session is idle.
                if state.marked_for_hang and state.inflight == 0:
                    state.marked_for_hang = False
                    state.hang_state = SESSION_HUNG
                    state.continue_event.clear()
                    psrl_logger.debug(
                        f"Applied deferred session hang: session_id={sid!r}, trajectory_id={trajectory_id!r}."
                    )

        return result, response_body

    @staticmethod
    def _json_result(status: int, body: dict) -> HttpResponse:
        return HttpResponse(
            status=status,
            body=json.dumps(body, ensure_ascii=False).encode(),
            headers={"content-type": "application/json"},
        )

    async def session_proxy(self, sid: str, path: str, request: Request) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            session_headers = state.headers.copy()
        headers = self.add_session_headers(request, session_headers)
        result = await self._request_upstream(
            request.method,
            path,
            content=await request.body(),
            headers=headers,
        )
        return self.build_response(result)

    # --- Hang and Continue Control Plane ---

    async def control_list_sessions(self) -> Response:
        """Return a snapshot of every live session for the coordinator scheduler."""
        # Snapshot the states dict first so we don't hold states_lock while
        # acquiring per-session locks.
        items = list(self.states.items())
        sessions = []
        for sid, state in items:
            async with state.lock:
                sessions.append(
                    {
                        "session_id": sid,
                        "base_worker_id": state.base_worker_id,
                        "target_dp_rank": state.target_dp_rank,
                        "status": state.status,
                        "hang_state": state.hang_state,
                        "inflight": state.inflight,
                        "total_tokens": state.total_tokens,
                        "marked_for_hang": state.marked_for_hang,
                        "closing": state.closing,
                        "main_closed": state.main_closed,
                        "background_live_count": state.background_live_count,
                    }
                )
        return JSONResponse(content={"sessions": sessions})

    async def control_hang(self, request: Request) -> Response:
        """
        Hang the requested sessions.

        Idle sessions hang immediately. In-flight sessions hang at the next turn
        boundary.
        """
        payload = await self._read_control_ids(request)
        applied, deferred, missing = [], [], []
        for sid in payload:
            state = self.states.get(sid)
            if state is None:
                missing.append(sid)
                continue
            async with state.lock:
                if state.closing:
                    missing.append(sid)
                    continue
                if state.inflight == 0:
                    state.hang_state = SESSION_HUNG
                    state.marked_for_hang = False
                    state.continue_event.clear()
                    applied.append(sid)
                else:
                    state.marked_for_hang = True
                    deferred.append(sid)
        if applied or deferred or missing:
            psrl_logger.info(f"control_hang: hung={applied} deferred={deferred} missing={missing}.")
        return JSONResponse(content={"hung": applied, "deferred": deferred, "missing": missing})

    async def control_continue(self, request: Request) -> Response:
        """
        Continue the requested sessions.

        Optional worker IDs pin only the next turn. Without them, SMG selects the
        next worker.
        """
        pins = await self._read_control_pins(request)
        applied, missing = [], []
        for sid, instance in pins.items():
            state = self.states.get(sid)
            if state is None:
                missing.append(sid)
                continue
            async with state.lock:
                state.marked_for_hang = False
                state.hang_state = SESSION_RUNNING
                if instance is not None:
                    # Update the routing hint and arm the one-shot pin for the
                    # next turn (consumed in session_chat_completions).
                    state.base_worker_id = instance[0]
                    state.target_dp_rank = instance[1]
                    state.pin_once_instance = instance
                state.continue_event.set()
                applied.append(sid)
        if applied or missing:
            pinned = {sid: inst for sid, inst in pins.items() if inst is not None}
            psrl_logger.info(f"control_continue: continued={applied} missing={missing} pinned={pinned}.")
        return JSONResponse(content={"continued": applied, "missing": missing})

    @staticmethod
    async def _read_control_ids(request: Request) -> list[str]:
        """Parse a control request body into a list of session ids.

        Accepts ``[{"session_id": ...}, ...]``, ``["sid", ...]``, or
        ``{"sessions": [...]}``.
        """
        try:
            body = await request.json()
        except Exception:
            return []
        if isinstance(body, dict):
            body = body.get("sessions", [])
        ids: list[str] = []
        for item in body or []:
            if isinstance(item, dict):
                sid = item.get("session_id")
            else:
                sid = item
            if sid is not None:
                ids.append(str(sid))
        return ids

    @staticmethod
    async def _read_control_pins(request: Request) -> dict[str, tuple[str, str] | None]:
        """Parse a continue request body into ``{session_id: instance | None}``.

        Accepts ``[{"session_id": ..., "base_worker_id": ..., "target_dp_rank": ...}, ...]``
        or ``{"sessions": [...]}``. ``instance`` is ``(base_worker_id, target_dp_rank)``
        when both are supplied, else ``None`` (route normally). Bare-string items
        (``["sid", ...]``) are accepted and map to ``None``.
        """
        try:
            body = await request.json()
        except Exception:
            return {}
        if isinstance(body, dict):
            body = body.get("sessions", [])
        pins: dict[str, tuple[str, str] | None] = {}
        for item in body or []:
            if isinstance(item, dict):
                sid = item.get("session_id")
                base_worker_id = item.get("base_worker_id")
                target_dp_rank = item.get("target_dp_rank")
                instance = None
                if base_worker_id is not None and target_dp_rank is not None:
                    instance = (str(base_worker_id), str(target_dp_rank))
            else:
                sid = item
                instance = None
            if sid is not None:
                pins[str(sid)] = instance
        return pins

    @staticmethod
    def _update_total_tokens(
        state: SessionState,
        response_body: Mapping[str, object] | None,
    ) -> None:
        """Update state.total_tokens from Chat, Responses, or Messages usage."""
        if response_body is None:
            return
        usage = response_body.get("usage")
        if not isinstance(usage, Mapping):
            return
        prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
        footprint = prompt + completion
        if footprint > state.total_tokens:
            state.total_tokens = footprint

    async def _ensure_state(self, sid: str) -> SessionState:
        state = self.states.get(sid)
        if state is not None:
            return state
        async with self.states_lock:
            state = self.states.get(sid)
            if state is None:
                state = SessionState(headers={SESSION_ID_HEADER: sid})
                self.states[sid] = state
            return state

    async def _request_upstream(
        self,
        method: str,
        path: str,
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> HttpResponse:
        url = f"{self.smg_url}/{path.lstrip('/')}"
        try:
            return await request_raw(
                method,
                url,
                content=content,
                headers=headers,
                client=await self._ensure_client(),
                max_retries=1,
            )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            psrl_logger.warning(
                "SessionRouter upstream transport error for %s %s: %s.",
                method,
                path,
                exc,
            )
            body = json.dumps(
                {
                    "error": f"backend transport error: {type(exc).__name__}: {exc}",
                }
            ).encode()
            return HttpResponse(
                status=502,
                body=body,
                headers={"content-type": "application/json"},
            )

    async def _ensure_client(self) -> aiohttp.ClientSession:
        existing = self.client
        if existing is None:
            self.client = create_aiohttp_client(concurrency=self.client_concurrency)
            return self.client
        is_closed = getattr(existing, "closed", None)
        if is_closed is None:
            is_closed = getattr(existing, "is_closed", False)
        if is_closed:
            self.client = create_aiohttp_client(concurrency=self.client_concurrency)
        return self.client

    @staticmethod
    def build_response(result: HttpResponse) -> Response:
        content_type = result.headers.get("content-type", "")
        return Response(
            content=result.body,
            status_code=result.status,
            headers=result.headers,
            media_type=content_type or None,
        )

    @staticmethod
    def extract_session_id(result: HttpResponse) -> str:
        session_id = result.json()["session_id"]
        return session_id

    @staticmethod
    def session_headers(headers, sid: str) -> dict[str, str]:
        """Capture session defaults and bind them to the session identity."""
        allowed = {
            "x-prompt-id",
            "x-request-id",
            "x-is-validate",
            "x-is-sticky",
            "x-version-tag",
            "x-base-worker-id",
            "x-target-dp-rank",
            # Session-scoped prompt-too-long budget. SMG returns an Anthropic `prompt_too_long`
            # error at this value, which drives Claude Code's reactive compact.
            "x-smg-prompt-too-long-limit",
        }
        session_headers = {key.lower(): value for key, value in headers.items() if key.lower() in allowed}
        session_headers[SESSION_ID_HEADER] = sid
        return session_headers

    def add_session_headers(
        self,
        request: Request,
        session_headers: dict[str, str],
    ) -> dict[str, str]:
        # Session headers are defaults and request-scoped values take precedence.
        # The session identity is reserved and can only come from SessionState.
        request_headers = filter_http_headers(request.headers)
        request_headers.pop(SESSION_ID_HEADER, None)
        if self.trajectory_id_strategy == "auto":
            request_headers.pop(TRAJECTORY_ID_HEADER, None)
        headers = dict(session_headers)
        headers.update(request_headers)
        if self.trajectory_id_strategy == "manual":
            headers.setdefault(TRAJECTORY_ID_HEADER, "0")
        return headers
