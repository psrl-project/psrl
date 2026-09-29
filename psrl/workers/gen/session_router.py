import asyncio
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import aiohttp
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from psrl.utils.common.http_utils import (
    HttpResponse,
    create_aiohttp_client,
    filter_http_headers,
    request_raw,
)
from psrl.utils.rollout.turn_output_writer import TurnOutputWriter
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

    def __post_init__(self) -> None:
        if self.inflight == 0:
            self.drained.set()
        else:
            self.drained.clear()
        # A fresh session is running: allow it to proceed.
        self.continue_event.set()

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


class SessionRouter:
    def __init__(
        self,
        smg_url: str,
        client_concurrency: int = 1024,
        trajectory_id_strategy: str = "manual",
        turn_output_writer: TurnOutputWriter | None = None,
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
        self.states: dict[str, SessionState] = {}
        self.states_lock = asyncio.Lock()
        self.setup_routes()
        self.app.router.on_shutdown.append(self.aclose)

    def setup_routes(self):
        self.app.post("/sessions")(self.create_session)
        self.app.get("/sessions/{sid}")(self.get_session)
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
        result = await self._request_upstream("POST", TITO_SESSIONS_PATH)
        if result.status < 400:
            session_id = self.extract_session_id(result)
            state = await self._ensure_state(session_id)
            async with state.lock:
                state.headers = self.session_headers(request.headers, session_id)
        return self.build_response(result)

    async def get_session(self, sid: str) -> Response:
        result = await self._request_upstream("GET", f"{TITO_SESSIONS_PATH}/{sid}")
        state = self.states.get(sid)
        if state is not None:
            async with state.lock:
                if state.base_worker_id is not None:
                    result.headers["x-base-worker-id"] = state.base_worker_id
                if state.target_dp_rank is not None:
                    result.headers["x-target-dp-rank"] = state.target_dp_rank
        return self.build_response(result)

    async def delete_session(self, sid: str) -> Response:
        state = await self._ensure_state(sid)
        async with state.lock:
            state.closing = True
            # Unblock any turn hung at the hang point so the session can drain
            # and be torn down instead of deadlocking on continue_event.
            state.hang_state = SESSION_RUNNING
            state.marked_for_hang = False
            state.continue_event.set()

        await state.drained.wait()

        result = await self._request_upstream("DELETE", f"{TITO_SESSIONS_PATH}/{sid}")

        async with self.states_lock:
            if self.states.get(sid) is state:
                self.states.pop(sid, None)
        return self.build_response(result)

    async def session_chat_completions(self, sid: str, request: Request) -> Response:
        """Handle Chat Completions, synthesizing SSE after a completed turn."""
        prepared = await self.prepare_protocol_request(request, "chat")
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
        prepared = await self.prepare_protocol_request(request, "responses")
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
        prepared = await self.prepare_protocol_request(request, "messages")
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

    @staticmethod
    async def prepare_protocol_request(
        request: Request,
        protocol: str,
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
        if protocol == "chat":
            body["logprobs"] = True
            body["top_logprobs"] = 1
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
            state.inflight += 1
            state.drained.clear()

        headers = self.add_session_headers(request, session_headers)
        headers["accept"] = "application/json"
        headers["content-type"] = "application/json"
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
                if state.inflight == 0:
                    state.drained.set()
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
