import json

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import Response as FastAPIResponse
from httpx import ASGITransport
from psrl.workers.gen.session_router import SessionRouter

# Mock SMG server
mock_smg = FastAPI()
captured: dict = {}


@mock_smg.post("/tito/sessions")
async def mock_create():
    return {"session_id": "test-sid-123"}


@mock_smg.get("/tito/sessions/{sid}")
async def mock_get(sid: str):
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
        "top_logprobs": 1,
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
