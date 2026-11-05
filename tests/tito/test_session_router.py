import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse
from httpx import ASGITransport
from psrl.utils.common.http_utils import SESSION_TURN_ACCOUNTING_KEY, SESSION_TURN_BUDGET_ERROR_CODE
from psrl.workers.gen.session_router import SessionRouter

# Mock SMG server
mock_smg = FastAPI()
captured: dict = {}


@mock_smg.post("/tito/sessions")
async def mock_create():
    return {"session_id": "test-sid-123"}


@mock_smg.get("/tito/sessions/{sid}")
async def mock_get(sid: str, request: Request):
    captured["get_query"] = dict(request.query_params)
    return {
        "session_id": sid,
        "max_trim_tokens": 0,
        "trajectories": [{"trajectory_id": 0, "accumulated_token_ids": [1, 2, 3], "records": []}],
    }


@mock_smg.delete("/tito/sessions/{sid}")
async def mock_delete(sid: str):
    return FastAPIResponse(status_code=204)


@mock_smg.post("/v1/chat/completions")
async def mock_chat(request: Request):
    raw_body = await request.body()
    body = json.loads(raw_body)
    headers = dict(request.headers)
    captured["chat_raw_body"] = raw_body
    captured["chat_body"] = body
    captured["chat_headers"] = headers
    response_body = json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "my-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    return FastAPIResponse(
        content=response_body,
        media_type="application/json",
        headers={"x-base-worker-id": "worker-a", "x-target-dp-rank": "2", "x-version-tag": "3"},
    )


@mock_smg.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def mock_catchall(path: str, request: Request):
    captured["proxy_path"] = path
    captured["proxy_headers"] = dict(request.headers)
    captured["proxy_body"] = await request.body()
    return {"proxied": True}


# Fixtures
@pytest.fixture(autouse=True)
def _clear_captured():
    captured.clear()


@pytest.fixture()
def router():
    """Build a SessionRouter whose internal httpx client talks to mock_smg."""
    sr = SessionRouter(smg_url="http://mock-smg")
    # Replace the real client with one backed by the mock ASGI app
    sr.client = httpx.AsyncClient(transport=ASGITransport(app=mock_smg), base_url="http://mock-smg")
    return sr


@pytest.fixture()
def client(router):
    """httpx client that talks to the SessionRouter's FastAPI app."""
    return httpx.AsyncClient(transport=ASGITransport(app=router.app), base_url="http://testserver")


@pytest.fixture()
def auto_router():
    sr = SessionRouter(smg_url="http://mock-smg", trajectory_id_strategy="auto")
    sr.client = httpx.AsyncClient(transport=ASGITransport(app=mock_smg), base_url="http://mock-smg")
    return sr


@pytest.fixture()
def auto_client(auto_router):
    return httpx.AsyncClient(transport=ASGITransport(app=auto_router.app), base_url="http://testserver")


# Tests
@pytest.mark.asyncio
async def test_create_session(client, router):
    resp = await client.post(
        "/sessions",
        headers={
            "x-request-id": "request-1",
            "x-smg-tito-session-id": "spoofed-sid",
            "x-unrelated": "ignored",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-sid-123"
    assert router.states["test-sid-123"].headers == {
        "x-request-id": "request-1",
        "x-smg-tito-session-id": "test-sid-123",
    }


@pytest.mark.asyncio
async def test_get_session(client):
    resp = await client.get("/sessions/my-session")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "my-session"
    assert data["trajectories"][0]["accumulated_token_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_delete_session(client):
    resp = await client.delete("/sessions/my-session")
    assert resp.status_code == 204


@pytest.mark.asyncio
async def test_chat_completions_buffers_body_with_logprobs(client):
    raw_body = b'{"model":"my-model","messages":[],"logprobs":false,"stream":true}'
    resp = await client.post(
        "/sessions/sid-abc/v1/chat/completions",
        content=raw_body,
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200
    # The router buffers the turn: it forces a non-streaming upstream call and
    # requests logprobs so TITO can reconstruct token-level trajectories.
    assert captured["chat_body"] == {
        "model": "my-model",
        "messages": [],
        "logprobs": True,
        "stream": False,
        "top_logprobs": 0,
        "sampler_top_logprobs": 0,
    }


@pytest.mark.asyncio
async def test_chat_completions_injects_session_metadata(client):
    await client.post(
        "/sessions",
        headers={
            "x-is-sticky": "true",
            "x-request-id": "request-1",
        },
    )
    await client.post(
        "/sessions/test-sid-123/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"x-request-id": "request-2"},
    )
    assert captured["chat_headers"]["x-is-sticky"] == "true"
    assert captured["chat_headers"]["x-request-id"] == "request-2"
    assert captured["chat_headers"]["x-smg-tito-trajectory-id"] == "0"


@pytest.mark.asyncio
async def test_unbound_session_preserves_request_trajectory_id(client):
    await client.post(
        "/sessions",
        headers={
            "x-is-sticky": "true",
            "x-request-id": "request-1",
        },
    )
    await client.post(
        "/sessions/test-sid-123/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"x-smg-tito-trajectory-id": "4"},
    )
    assert captured["chat_headers"]["x-smg-tito-trajectory-id"] == "4"


@pytest.mark.asyncio
async def test_auto_strategy_removes_request_trajectory_id(auto_client, auto_router):
    await auto_client.post(
        "/sessions/sid-auto/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers={"x-smg-tito-trajectory-id": "4"},
    )
    assert "x-smg-tito-trajectory-id" not in captured["chat_headers"]
    assert auto_router.states["sid-auto"].trajectory_turns == {}


def test_rejects_invalid_trajectory_id_strategy():
    with pytest.raises(ValueError, match="trajectory_id_strategy"):
        SessionRouter(smg_url="http://mock-smg", trajectory_id_strategy="invalid")


@pytest.mark.asyncio
async def test_chat_completions_injects_session_header(client, router):
    payload = {"model": "m", "messages": []}
    await client.post(
        "/sessions/sid-xyz/v1/chat/completions",
        json=payload,
        headers={"x-smg-tito-session-id": "spoofed-sid"},
    )
    assert captured["chat_headers"]["x-smg-tito-session-id"] == "sid-xyz"
    assert captured["chat_headers"]["x-smg-tito-trajectory-id"] == "0"
    assert router.states["sid-xyz"].headers["x-smg-tito-session-id"] == "sid-xyz"


@pytest.mark.asyncio
async def test_session_proxy_uses_request_overrides_and_fixed_session_id(client):
    await client.post("/sessions", headers={"x-request-id": "request-1"})
    resp = await client.post(
        "/sessions/test-sid-123/v1/some/other/endpoint",
        content=b"hello",
        headers={
            "x-request-id": "request-2",
            "x-smg-tito-session-id": "spoofed-sid",
        },
    )
    assert resp.status_code == 200
    assert captured["proxy_headers"]["x-request-id"] == "request-2"
    assert captured["proxy_headers"]["x-smg-tito-session-id"] == "test-sid-123"
    assert captured["proxy_path"] == "v1/some/other/endpoint"


@pytest.mark.asyncio
async def test_chat_completions_pins_worker_for_session(client):
    payload = {"model": "m", "messages": []}
    await client.post("/sessions/sid-sticky/v1/chat/completions", json=payload)
    await client.post("/sessions/sid-sticky/v1/chat/completions", json=payload)
    assert captured["chat_headers"]["x-base-worker-id"] == "worker-a"
    assert captured["chat_headers"]["x-target-dp-rank"] == "2"

    resp = await client.get("/sessions/sid-sticky")
    assert resp.headers["x-base-worker-id"] == "worker-a"
    assert resp.headers["x-target-dp-rank"] == "2"


@pytest.mark.asyncio
async def test_chat_completions_pins_version_tag_for_session(client):
    payload = {"model": "m", "messages": []}
    # The first turn arrives unversioned. SMG echoes the pinned version in the
    # response headers, which the SessionRouter records on the session.
    await client.post(
        "/sessions/sid-ver/v1/chat/completions",
        json=payload,
        headers={"x-version-tag": "-1"},
    )
    # Second turn should carry the pinned version instead of the original -1.
    await client.post(
        "/sessions/sid-ver/v1/chat/completions",
        json=payload,
        headers={"x-version-tag": "-1"},
    )
    assert captured["chat_headers"]["x-version-tag"] == "3"


@pytest.mark.asyncio
async def test_control_continue_force_pins_next_turn(client, router):
    """continue with an instance force-pins ONLY the next turn (one-shot)."""
    payload = {"model": "m", "messages": []}
    # Establish the session, then hang it.
    await client.post("/sessions/sid-pin/v1/chat/completions", json=payload)
    await client.post("/control/hang", json=[{"session_id": "sid-pin"}])
    assert router.states["sid-pin"].hang_state == "hung"

    # Continue onto a specific instance.
    resp = await client.post(
        "/control/continue",
        json=[{"session_id": "sid-pin", "base_worker_id": "worker-z", "target_dp_rank": "5"}],
    )
    assert resp.json()["continued"] == ["sid-pin"]
    assert router.states["sid-pin"].pin_once_instance == ("worker-z", "5")

    # Next turn carries the force-pin header + hinted instance...
    await client.post("/sessions/sid-pin/v1/chat/completions", json=payload)
    assert captured["chat_headers"]["x-force-pin-once"] == "true"
    assert captured["chat_headers"]["x-base-worker-id"] == "worker-z"
    assert captured["chat_headers"]["x-target-dp-rank"] == "5"
    # ...and the one-shot pin is consumed.
    assert router.states["sid-pin"].pin_once_instance is None

    # The turn after that no longer force-pins (SMG response re-set worker-a).
    await client.post("/sessions/sid-pin/v1/chat/completions", json=payload)
    assert "x-force-pin-once" not in captured["chat_headers"]


@pytest.mark.asyncio
async def test_control_continue_without_instance_does_not_pin(client, router):
    """continue without an instance leaves routing to SMG (no force-pin)."""
    payload = {"model": "m", "messages": []}
    await client.post("/sessions/sid-nopin/v1/chat/completions", json=payload)
    await client.post("/control/hang", json=[{"session_id": "sid-nopin"}])
    await client.post("/control/continue", json=[{"session_id": "sid-nopin"}])
    assert router.states["sid-nopin"].pin_once_instance is None

    await client.post("/sessions/sid-nopin/v1/chat/completions", json=payload)
    assert "x-force-pin-once" not in captured["chat_headers"]


@pytest.mark.asyncio
async def test_turn_counter_increments_on_chat_completion(router, client):
    """Turn counter increments exactly once per successful chat completion."""
    payload = {"model": "m", "messages": []}
    state = await router._ensure_state("sid-abc")
    assert state.get_trajectory_turn(0) == 0

    await client.post("/sessions/sid-abc/v1/chat/completions", json=payload)
    assert state.get_trajectory_turn(0) == 1

    await client.post("/sessions/sid-abc/v1/chat/completions", json=payload)
    assert state.get_trajectory_turn(0) == 2


@pytest.mark.asyncio
async def test_delete_during_new_request_returns_409(router, client):
    """After session is closed, new chat completion requests get 409."""
    state = await router._ensure_state("sid-close")
    async with state.lock:
        state.closing = True

    payload = {"model": "m", "messages": []}
    resp = await client.post("/sessions/sid-close/v1/chat/completions", json=payload)
    assert resp.status_code == 409
    assert "session is closing" in resp.text


# --- Declared turn budgets ------------------------------------------------------------
#
# A harness whose CLI takes no turn cap declares one here instead, and this router is what
# holds the session to it. Only the parent agent's model calls are charged: a subagent's
# calls are counted separately, so the same configured cap means the same thing whether a
# CLI enforces it or this does.

BUDGET_HEADER = "x-psrl-max-turns"
METADATA_HEADER = "x-codex-turn-metadata"


def _metadata(thread_source: str) -> dict[str, str]:
    return {METADATA_HEADER: json.dumps({"thread_source": thread_source, "turn_id": "t1"})}


async def _create_with_budget(client, budget: int | None) -> None:
    headers = {"x-request-id": "r1"}
    if budget is not None:
        headers[BUDGET_HEADER] = str(budget)
    resp = await client.post("/sessions", headers=headers)
    assert resp.status_code == 200


async def _turn(client, sid: str, *, thread_source: str | None = None, stream: bool = False):
    headers = _metadata(thread_source) if thread_source else {}
    return await client.post(
        f"/sessions/{sid}/v1/responses",
        json={"model": "m", "input": [], "stream": stream},
        headers=headers,
    )


@pytest.mark.asyncio
async def test_a_declared_budget_is_captured_at_session_creation(client, router):
    await _create_with_budget(client, 3)

    assert router.states["test-sid-123"].max_turns == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["abc", "0", "-1"])
async def test_an_unusable_budget_header_leaves_the_session_unbounded(client, router, raw):
    resp = await client.post("/sessions", headers={BUDGET_HEADER: raw})
    assert resp.status_code == 200

    assert router.states["test-sid-123"].max_turns is None


@pytest.mark.asyncio
async def test_a_session_without_a_declared_budget_is_never_refused(client, router):
    await _create_with_budget(client, None)

    for _ in range(5):
        await _turn(client, "test-sid-123")

    assert router.states["test-sid-123"].main_turns == 5


@pytest.mark.asyncio
async def test_the_turn_after_the_budget_is_spent_is_refused(client, router):
    await _create_with_budget(client, 2)

    assert (await _turn(client, "test-sid-123")).status_code == 200
    assert (await _turn(client, "test-sid-123")).status_code == 200

    refused = await _turn(client, "test-sid-123")

    assert refused.status_code == 400
    assert refused.headers["x-smg-error-code"] == SESSION_TURN_BUDGET_ERROR_CODE
    # The code travels in the body as well: a harness CLI echoes an error body verbatim
    # into its own output, which is where the framework reads the stop reason from.
    assert refused.json()["error"]["code"] == SESSION_TURN_BUDGET_ERROR_CODE
    assert router.states["test-sid-123"].main_turns == 2, "a refused turn is not a served one"


@pytest.mark.asyncio
async def test_a_subagent_turn_is_charged_separately_from_the_parent(client, router):
    await _create_with_budget(client, 2)
    state = router.states["test-sid-123"]

    await _turn(client, "test-sid-123", thread_source="user")
    await _turn(client, "test-sid-123", thread_source="subagent")

    assert (state.main_turns, state.subagent_turns) == (1, 1)
    # The parent still has its second turn: a subagent's work is not the parent's.
    assert (await _turn(client, "test-sid-123", thread_source="user")).status_code == 200


@pytest.mark.asyncio
async def test_a_subagent_turn_is_refused_once_the_parent_budget_is_spent(client, router):
    """Nothing more runs once the parent's turns are gone, a subagent included."""
    await _create_with_budget(client, 1)
    await _turn(client, "test-sid-123", thread_source="user")

    refused = await _turn(client, "test-sid-123", thread_source="subagent")

    assert refused.status_code == 400
    assert router.states["test-sid-123"].subagent_turns == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["not json", "[]", "{}"])
async def test_a_turn_that_cannot_be_attributed_is_charged_to_the_parent(client, router, raw):
    """A caller that sends no usable label spends the parent's budget.

    Reading an unattributable call as a subagent's would let anything unbounded slip past
    the budget.
    """
    await _create_with_budget(client, 1)

    resp = await client.post(
        "/sessions/test-sid-123/v1/responses",
        json={"model": "m", "input": [], "stream": False},
        headers={METADATA_HEADER: raw},
    )

    assert resp.status_code == 200
    assert router.states["test-sid-123"].main_turns == 1
    assert (await _turn(client, "test-sid-123")).status_code == 400


@pytest.mark.asyncio
async def test_the_session_view_reports_the_turn_accounting(client, router):
    await _create_with_budget(client, 4)
    await _turn(client, "test-sid-123")
    await _turn(client, "test-sid-123", thread_source="subagent")

    body = (await client.get("/sessions/test-sid-123")).json()

    assert body[SESSION_TURN_ACCOUNTING_KEY] == {
        "max_turns": 4,
        "main_turns": 1,
        "subagent_turns": 1,
        "exhausted": False,
    }
    # The gateway's own view is untouched: this router adds one key of its own.
    assert body["trajectories"][0]["accumulated_token_ids"] == [1, 2, 3]


# --- Subagent policy and normalized agent headers -------------------------------------

CLAUDE_CLASS_HEADER = "x-claude-code-request-class"
CLAUDE_AGENT_HEADER = "x-claude-code-agent-id"
CLAUDE_PARENT_HEADER = "x-claude-code-parent-agent-id"


async def _create_with_policy(client, policy: dict, *, validate: bool = False):
    headers = {"x-is-validate": "true"} if validate else {}
    resp = await client.post("/sessions", json={"subagent_training": policy}, headers=headers)
    assert resp.status_code == 200
    return resp


async def _response_turn(client, headers: dict | None = None):
    return await client.post(
        "/sessions/test-sid-123/v1/responses",
        json={"model": "m", "input": [], "stream": False},
        headers=headers or {},
    )


@pytest.mark.asyncio
async def test_create_session_stores_the_subagent_policy(client, router):
    await _create_with_policy(
        client,
        {"mode": "selected", "train_agents": ["x"], "val_mode": "selected"},
    )

    state = router.states["test-sid-123"]
    assert state.subagent_training.mode == "selected"
    assert state.subagent_training.train_agents == ("x",)


@pytest.mark.asyncio
async def test_create_session_rejects_an_invalid_policy(client, router):
    resp = await client.post("/sessions", json={"subagent_training": {"mode": "unknown"}})

    assert resp.status_code == 400
    assert "mode" in resp.json()["error"]["message"]
    assert "test-sid-123" not in router.states


@pytest.mark.asyncio
async def test_codex_metadata_is_normalized_and_cannot_spoof_capture(client, router):
    await _create_with_policy(client, {"mode": "main_only"})
    await _response_turn(
        client,
        headers={
            "x-codex-turn-metadata": json.dumps(
                {"thread_source": "subagent", "thread_id": "sub-1", "parent_thread_id": "main-1"}
            ),
            "x-psrl-agent-role": "main",
            "x-psrl-tito-capture": "full",
        },
    )

    headers = captured["proxy_headers"]
    assert headers["x-psrl-agent-role"] == "subagent"
    assert headers["x-psrl-agent-id"] == "sub-1"
    assert headers["x-psrl-parent-agent-id"] == "main-1"
    assert headers["x-psrl-tito-capture"] == "tokens"


@pytest.mark.asyncio
async def test_main_call_gets_the_main_identity_and_full_capture(client, router):
    await _create_with_policy(client, {"mode": "all"})
    await _response_turn(client, headers={"x-codex-turn-metadata": json.dumps({"thread_source": "user"})})

    headers = captured["proxy_headers"]
    assert headers["x-psrl-agent-role"] == "main"
    assert headers["x-psrl-agent-id"] == "main"
    assert "x-psrl-parent-agent-id" not in headers
    assert headers["x-psrl-tito-capture"] == "full"


@pytest.mark.asyncio
async def test_claude_code_request_class_is_normalized(client, router):
    await _create_with_policy(client, {"mode": "all"})
    await _response_turn(
        client,
        headers={
            CLAUDE_CLASS_HEADER: "subagent",
            CLAUDE_AGENT_HEADER: "claude-sub",
            CLAUDE_PARENT_HEADER: "claude-main",
        },
    )

    headers = captured["proxy_headers"]
    assert headers["x-psrl-agent-role"] == "subagent"
    assert headers["x-psrl-agent-id"] == "claude-sub"
    assert headers["x-psrl-parent-agent-id"] == "claude-main"
    assert headers["x-psrl-tito-capture"] == "full"


@pytest.mark.asyncio
async def test_claude_code_auxiliary_is_never_trained(client, router):
    await _create_with_policy(client, {"mode": "all"})
    await _response_turn(
        client,
        headers={CLAUDE_CLASS_HEADER: "auxiliary", CLAUDE_AGENT_HEADER: "claude-aux"},
    )

    headers = captured["proxy_headers"]
    assert headers["x-psrl-agent-role"] == "auxiliary"
    assert headers["x-psrl-tito-capture"] == "tokens"


@pytest.mark.asyncio
async def test_compaction_follows_the_main_agent(client, router):
    await _create_with_policy(client, {"mode": "main_only"})
    await _response_turn(client, headers={CLAUDE_CLASS_HEADER: "main"})
    await _response_turn(
        client,
        headers={
            CLAUDE_CLASS_HEADER: "compaction",
            CLAUDE_AGENT_HEADER: "main",
            CLAUDE_PARENT_HEADER: "unused",
        },
    )

    headers = captured["proxy_headers"]
    assert headers["x-psrl-agent-role"] == "compaction"
    assert headers["x-psrl-agent-id"] == "main"
    assert "x-psrl-parent-agent-id" not in headers
    assert headers["x-psrl-tito-capture"] == "full"


@pytest.mark.asyncio
async def test_validation_captures_tokens_only(client, router):
    await _create_with_policy(client, {"mode": "all", "val_mode": "all"}, validate=True)
    await _response_turn(client)

    assert captured["proxy_headers"]["x-psrl-tito-capture"] == "tokens"


@pytest.mark.asyncio
async def test_session_view_persists_policy_and_main_agent_id(client, router):
    await _create_with_policy(client, {"mode": "all", "val_mode": "all"})
    await _response_turn(
        client,
        headers={"x-codex-turn-metadata": json.dumps({"thread_source": "user", "thread_id": "main-7"})},
    )

    body = (await client.get("/sessions/test-sid-123")).json()

    assert body["subagent_training"]["mode"] == "all"
    assert body["main_agent_id"] == "main-7"


@pytest.mark.asyncio
async def test_responses_requests_ask_for_output_logprobs(client):
    await _response_turn(client)

    body = json.loads(captured["proxy_body"])
    assert body["logprobs"] is True
    assert "message.output_text.logprobs" in body["include"]


@pytest.mark.asyncio
async def test_a_second_non_subagent_thread_is_not_a_second_primary(client, router):
    await _create_with_policy(client, {"mode": "all"})
    await _response_turn(
        client,
        headers={"x-codex-turn-metadata": json.dumps({"thread_source": "user", "thread_id": "main-1"})},
    )
    await _response_turn(
        client,
        headers={"x-codex-turn-metadata": json.dumps({"thread_source": "user", "thread_id": "side-2"})},
    )

    headers = captured["proxy_headers"]
    assert headers["x-psrl-agent-role"] == "auxiliary"
    assert headers["x-psrl-agent-id"] == "side-2"
    assert headers["x-psrl-parent-agent-id"] == "main-1"
    assert headers["x-psrl-tito-capture"] == "tokens"


@pytest.mark.asyncio
async def test_a_second_thread_with_a_parent_is_a_subagent(client, router):
    await _create_with_policy(client, {"mode": "all"})
    await _response_turn(
        client,
        headers={"x-codex-turn-metadata": json.dumps({"thread_source": "user", "thread_id": "main-1"})},
    )
    await _response_turn(
        client,
        headers={
            "x-codex-turn-metadata": json.dumps(
                {"thread_source": "user", "thread_id": "child-2", "parent_thread_id": "main-1"}
            )
        },
    )

    headers = captured["proxy_headers"]
    assert headers["x-psrl-agent-role"] == "subagent"
    assert headers["x-psrl-agent-id"] == "child-2"
    assert headers["x-psrl-parent-agent-id"] == "main-1"
    assert headers["x-psrl-tito-capture"] == "full"


@pytest.mark.asyncio
async def test_get_session_forwards_the_snapshot_format(client):
    resp = await client.get("/sessions/my-session?format=compact")

    assert resp.status_code == 200
    assert captured["get_query"] == {"format": "compact"}


@pytest.mark.asyncio
async def test_main_close_counts_distinct_inflight_background_agents(client, router):
    state = await router._ensure_state("sid-bg")
    async with state.lock:
        state.inflight = 3
        state.agent_inflight = {
            ("main", "main"): 1,
            ("subagent", "sub-a"): 1,
            ("subagent", "sub-b"): 1,
        }

    resp = await client.post("/sessions/sid-bg/main-close")

    assert resp.status_code == 200
    assert resp.json() == {"background_live_count": 2}
    assert state.background_live_count == 2


@pytest.mark.asyncio
async def test_delete_does_not_wait_for_background_inflight(client, router):
    state = await router._ensure_state("sid-bg-delete")
    async with state.lock:
        state.main_closed = True
        state.inflight = 1
        state.agent_inflight = {("subagent", "sub-a"): 1}

    resp = await client.delete("/sessions/sid-bg-delete")

    assert resp.status_code == 204
    assert "sid-bg-delete" not in router.states


@pytest.mark.asyncio
async def test_main_close_uses_a_harness_reported_count(client, router):
    state = await router._ensure_state("sid-reported")
    async with state.lock:
        state.inflight = 0

    resp = await client.post(
        "/sessions/sid-reported/main-close",
        json={"background_subagents": 3},
    )

    assert resp.json() == {"background_live_count": 3}
    assert state.background_live_count == 3
