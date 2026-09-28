"""A session-scoped OpenAI proxy over an endpoint that has no session layer.

`SessionAgentLoop` hands an external agent a per-episode session URL and later
reads that session back to learn what happened. SMG provides this via TITO, which
captures the token stream the engine actually processed. An external API returns
text and nothing else, so this proxy provides the same session surface over it.

What it deliberately does NOT do is re-tokenize. Re-encoding assistant text with a
local tokenizer produces ids that merely resemble what the server processed, with
different boundaries at turn seams and no log probabilities, and writing them into
the same `response_ids` field that carries real TITO output would give fabricated
tokens the provenance of measured ones. The trajectories here are text-native:
`response_ids` stays empty and the transcript, with the per-turn token counts the
API itself reported, is written alongside.

Consequently a session served here supports collection, not training. Use
`serving=smg_local` when the token stream matters.
"""

import asyncio
import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from psrl.utils.common.http_utils import find_available_port

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Mirrors SessionRouter so an agent cannot tell the two apart.
SESSION_ID_HEADER = "x-smg-tito-session-id"
TRAJECTORY_ID_HEADER = "x-smg-tito-trajectory-id"
REQUEST_ID_HEADER = "x-request-id"


def _drop_empty_messages(messages):
    """Remove assistant messages carrying neither content nor tool calls.

    An agent harness emits one whenever the model returns an empty completion, and
    strict providers reject the whole request rather than the offending message.
    That turns a single bad turn into a dead episode. A message with no content and
    no tool call conveys nothing, so dropping it is lossless.

    Args:
        messages: The `messages` array from a chat-completion payload.

    Returns:
        The array with empty assistant turns removed, or the input unchanged when
        it is not a list.
    """
    if not isinstance(messages, list):
        return messages
    kept = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            kept.append(message)
            continue
        content = message.get("content")
        has_text = bool(content) if not isinstance(content, str) else bool(content.strip())
        if has_text or message.get("tool_calls"):
            kept.append(message)
        else:
            psrl_logger.debug("Dropped an empty assistant message before forwarding upstream.")
    return kept


@dataclass
class _Turn:
    """One recorded chat completion."""

    turn: int
    request: dict
    response: dict | None
    status: int
    finish_reason: str | None
    prompt_tokens: int
    completion_tokens: int


@dataclass
class _Session:
    """Everything the proxy remembers about one episode."""

    session_id: str
    uid: str | None = None
    turns: list[_Turn] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class OpenAISessionAdapter:
    """Serve the SessionRouter API surface, forwarding to any OpenAI endpoint.

    Implements the three routes `SessionAgentLoop` uses (`POST /sessions`,
    `GET /sessions/{sid}`, `DELETE /sessions/{sid}`) plus the session-scoped
    `/sessions/{sid}/v1/chat/completions` an external agent calls.
    """

    def __init__(
        self,
        upstream_url: str,
        transcript_dir: str,
        api_key: str = "EMPTY",
        model_name: str = "",
        client_concurrency: int = 256,
        request_timeout_s: float = 1800.0,
        trust_env: bool = True,
    ) -> None:
        """
        Args:
            upstream_url (str): OpenAI-compatible base URL, ending in `/v1`.
            transcript_dir (str): Directory for per-episode transcripts.
            api_key (str): Bearer token forwarded upstream.
            model_name (str): Model id to substitute into each request. Empty
                forwards whatever the agent loop sent.
            client_concurrency (int): Upstream connection pool size.
            request_timeout_s (float): Per-request upstream timeout.
            trust_env (bool): Whether to honor `HTTP(S)_PROXY` and `NO_PROXY`.
                On by default, because a cluster behind a corporate proxy has no
                direct route to a hosted API, and aiohttp ignores those variables
                unless asked. `NO_PROXY` still exempts a local endpoint.
        """
        self.upstream_url = upstream_url.rstrip("/")
        self.transcript_dir = os.path.abspath(os.path.expanduser(transcript_dir))
        os.makedirs(self.transcript_dir, exist_ok=True)
        self.api_key = api_key
        self.model_name = model_name
        self.client_concurrency = client_concurrency
        self.request_timeout_s = request_timeout_s
        self.trust_env = trust_env

        self.sessions: dict[str, _Session] = {}
        self.sessions_lock = asyncio.Lock()
        self.client: aiohttp.ClientSession | None = None

        self.app = FastAPI()
        self._setup_routes()
        self.app.router.on_shutdown.append(self.aclose)

    def _setup_routes(self) -> None:
        self.app.post("/sessions")(self.create_session)
        self.app.get("/sessions/{sid}")(self.get_session)
        self.app.delete("/sessions/{sid}")(self.delete_session)
        self.app.post("/sessions/{sid}/v1/chat/completions")(self.session_chat_completions)
        self.app.get("/health")(self.health)

    async def health(self) -> Response:
        return JSONResponse(content={"status": "ok", "sessions": len(self.sessions)})

    async def _ensure_client(self) -> aiohttp.ClientSession:
        if self.client is None or self.client.closed:
            connector = aiohttp.TCPConnector(
                limit=self.client_concurrency,
                limit_per_host=self.client_concurrency,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            self.client = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout_s),
                trust_env=self.trust_env,
            )
        return self.client

    async def aclose(self) -> None:
        """Close the upstream client."""
        if self.client is not None and not self.client.closed:
            await self.client.close()

    async def create_session(self, request: Request) -> Response:
        """Open a session and bind it to the caller's request id."""
        session_id = uuid.uuid4().hex
        session = _Session(session_id=session_id, uid=request.headers.get(REQUEST_ID_HEADER))
        async with self.sessions_lock:
            self.sessions[session_id] = session
        psrl_logger.debug("Opened proxy session %r for uid=%s.", session_id, session.uid)
        return JSONResponse(content={"session_id": session_id})

    async def get_session(self, sid: str) -> Response:
        """Return the session snapshot in the shape `SessionAgentLoop` expects.

        `accumulated_token_ids` is empty and every record reports no log
        probabilities, which is what makes the resulting trajectory text-native:
        `build_training_data` derives a correct `num_turns` and an empty
        `response_ids` rather than a fabricated token stream.
        """
        session = self.sessions.get(sid)
        if session is None:
            return JSONResponse(status_code=404, content={"error": f"unknown session {sid}"})

        async with session.lock:
            records = [
                {
                    "prompt_token_count": 0,
                    "output_logprobs": None,
                    "finish_reason": turn.finish_reason,
                }
                for turn in session.turns
            ]
            usage = {
                "prompt_tokens": sum(t.prompt_tokens for t in session.turns),
                "completion_tokens": sum(t.completion_tokens for t in session.turns),
                "turns": len(session.turns),
            }

        return JSONResponse(
            content={
                "session_id": sid,
                "max_trim_tokens": 0,
                "header_info": {},
                "usage": usage,
                "trajectories": [
                    {
                        "trajectory_id": 0,
                        "accumulated_token_ids": [],
                        "records": records,
                    }
                ],
            }
        )

    async def delete_session(self, sid: str) -> Response:
        """Close a session and flush its transcript."""
        async with self.sessions_lock:
            session = self.sessions.pop(sid, None)
        if session is None:
            return JSONResponse(content={"deleted": False})
        self._write_transcript(session)
        return JSONResponse(content={"deleted": True})

    async def session_chat_completions(self, sid: str, request: Request) -> Response:
        """Forward one chat completion upstream and record it."""
        session = self.sessions.get(sid)
        if session is None:
            return JSONResponse(status_code=404, content={"error": f"unknown session {sid}"})

        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "request body is not JSON"})

        # NOTE(claude): Loops send `model_config.path`, which a local vLLM serves
        # under but a hosted API rejects. Empty `model_name` forwards it unchanged.
        if self.model_name:
            payload["model"] = self.model_name

        payload["messages"] = _drop_empty_messages(payload.get("messages"))

        client = await self._ensure_client()
        url = f"{self.upstream_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        status = 500
        response_json: dict | None = None
        try:
            async with client.post(url, json=payload, headers=headers) as resp:
                status = resp.status
                text = await resp.text()
                try:
                    response_json = json.loads(text)
                except json.JSONDecodeError:
                    response_json = {"error": text[:2000]}
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            psrl_logger.warning("Upstream call failed for session %r: %s.", sid, exc)
            response_json = {"error": f"{type(exc).__name__}: {exc}"}

        await self._record_turn(session, payload, response_json, status)

        if status >= 400 or response_json is None:
            return JSONResponse(status_code=status or 502, content=response_json or {"error": "upstream failure"})
        return JSONResponse(status_code=status, content=response_json)

    async def _record_turn(self, session: _Session, payload: dict, response: dict | None, status: int) -> None:
        """Append one turn, reading token counts from the upstream usage block."""
        usage = (response or {}).get("usage") or {}
        choices = (response or {}).get("choices") or []
        finish_reason = choices[0].get("finish_reason") if choices else None

        async with session.lock:
            session.turns.append(
                _Turn(
                    turn=len(session.turns),
                    request=payload,
                    response=response,
                    status=status,
                    finish_reason=finish_reason,
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                )
            )

    def _write_transcript(self, session: _Session) -> None:
        """Write the full transcript, named by uid so it joins to `rollout.jsonl`.

        The uid comes from the `x-request-id` header `SessionAgentLoop` sets when
        opening the session, which is the same uid the rollout record carries.
        """
        stem = session.uid or session.session_id
        path = os.path.join(self.transcript_dir, f"{stem}.json")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "uid": session.uid,
                        "session_id": session.session_id,
                        "turns": [
                            {
                                "turn": turn.turn,
                                "status": turn.status,
                                "finish_reason": turn.finish_reason,
                                "prompt_tokens": turn.prompt_tokens,
                                "completion_tokens": turn.completion_tokens,
                                "messages": turn.request.get("messages"),
                                "response": turn.response,
                            }
                            for turn in session.turns
                        ],
                    },
                    fh,
                    ensure_ascii=False,
                )
        except OSError as exc:
            psrl_logger.warning("Failed to write transcript %s: %s.", path, exc)


def launch_session_adapter(
    upstream_url: str,
    transcript_dir: str,
    *,
    host: str,
    api_key: str = "EMPTY",
    model_name: str = "",
    client_concurrency: int = 256,
    request_timeout_s: float = 1800.0,
    trust_env: bool = True,
    base_port: int = 8400,
) -> tuple[str, threading.Thread]:
    """Start the proxy on a background uvicorn thread.

    A thread rather than a process, because the proxy holds only per-session state
    and must outlive nothing. The Ray actor that owns it exits with the run.

    Args:
        upstream_url (str): OpenAI-compatible base URL to forward to.
        transcript_dir (str): Directory for per-episode transcripts.
        host (str): Address to bind. Must be reachable from the agent containers.
        api_key (str): Bearer token forwarded upstream.
        model_name (str): Model id to substitute into each request.
        client_concurrency (int): Upstream connection pool size.
        request_timeout_s (float): Per-request upstream timeout.
        trust_env (bool): Whether to honor `HTTP(S)_PROXY` and `NO_PROXY`.
        base_port (int): First port to try.

    Returns:
        tuple[str, threading.Thread]: The proxy base URL and its serving thread.
    """
    port = find_available_port(base_port=base_port)
    proxy = OpenAISessionAdapter(
        upstream_url=upstream_url,
        transcript_dir=transcript_dir,
        api_key=api_key,
        model_name=model_name,
        client_concurrency=client_concurrency,
        request_timeout_s=request_timeout_s,
        trust_env=trust_env,
    )
    config = uvicorn.Config(
        proxy.app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="batch-rollout-session-proxy")
    thread.start()

    url = f"http://{host}:{port}"
    psrl_logger.info("Session proxy serving %s at %s.", upstream_url, url)
    return url, thread
