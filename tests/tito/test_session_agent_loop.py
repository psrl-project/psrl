from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from examples.mini_swe.runner import _build_model_config
from psrl.utils.rollout.vision_utils import normalize_messages
from psrl.workers.agent_loop.loops import session_agent_loop as module
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop


def _bare_loop(trajectory_id_strategy: str = "manual") -> SessionAgentLoop:
    loop = SessionAgentLoop.__new__(SessionAgentLoop)
    loop.session_router_url = "http://session-router"
    loop.trajectory_id_strategy = trajectory_id_strategy
    loop.config = SimpleNamespace(
        psrl=SimpleNamespace(
            rollout_coordination=SimpleNamespace(routing_strategy=SimpleNamespace(enable_trajectory_sticky=True)),
            rollout_gateway=SimpleNamespace(tito_debug=False),
        )
    )
    loop.timer = SimpleNamespace(generation=nullcontext)
    loop.model_config = SimpleNamespace(path="model")

    return loop


def test_session_urls_share_one_protocol_neutral_root() -> None:
    loop = _bare_loop()
    loop.session_router_url = "http://router:8080"

    assert loop.session_root_url("session-1") == "http://router:8080/sessions/session-1"
    assert loop.session_api_url("session-1") == "http://router:8080/sessions/session-1/v1"
    assert loop.session_root_url("session-1", "https://public-router/") == ("https://public-router/sessions/session-1")


def test_mini_swe_model_headers_follow_trajectory_strategy():
    payload = {
        "runtime_config": {"model": {}, "sandbox_config": {"rollout_turn_timeout": 30}},
        "sampling_params": {"top_k": -1},
        "base_url": "http://session-router/sessions/sid/v1",
        "model": "openai/model",
        "trajectory_id_strategy": "manual",
    }
    manual = _build_model_config(payload)
    assert manual["model_kwargs"]["extra_headers"] == {"x-smg-tito-trajectory-id": "0"}

    payload["trajectory_id_strategy"] = "auto"
    auto = _build_model_config(payload)
    assert "extra_headers" not in auto["model_kwargs"]


@pytest.mark.asyncio
async def test_create_session_does_not_bind_a_trajectory(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del payload, max_retries, kwargs
        captured["url"] = url
        captured["headers"] = headers
        return {"session_id": "sid"}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop()
    session_id = await loop.create_session({"uid": 3, "parent_id": 2, "version_tag": 7, "trajectory_id": 99})

    assert session_id == "sid"
    assert "x-smg-tito-trajectory-id" not in captured["headers"]
    assert captured["headers"]["x-request-id"] == "3"


@pytest.mark.asyncio
async def test_chat_completion_sends_request_trajectory_header(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del max_retries, kwargs
        captured.update(url=url, payload=payload, headers=headers)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop()
    await loop.chat_completion("sid", [{"role": "user", "content": "hello"}], {}, trajectory_id=4)

    assert captured["headers"] == {"x-smg-tito-trajectory-id": "4"}
    assert captured["payload"]["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_chat_completion_omits_trajectory_header_in_auto_mode(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del url, payload, max_retries, kwargs
        captured["headers"] = headers
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop("auto")
    await loop.chat_completion("sid", [{"role": "user", "content": "hello"}], {}, trajectory_id=4)

    assert captured["headers"] is None


@pytest.mark.asyncio
async def test_session_scope_cleans_up_after_external_agent_error():
    loop = _bare_loop()
    deleted = []

    async def create_session(request):
        assert request["uid"] == 1
        return "sid"

    async def delete_session(session_id):
        deleted.append(session_id)

    loop.create_session = create_session
    loop.delete_session = delete_session
    with pytest.raises(RuntimeError, match="agent failed"):
        async with loop.session_scope({"uid": 1}):
            raise RuntimeError("agent failed")
    assert deleted == ["sid"]


@pytest.mark.asyncio
async def test_get_training_data_always_returns_a_trajectory_list():
    loop = _bare_loop()
    session_data = {
        "header_info": {"base_worker_id": "worker", "target_dp_rank": "1"},
        "max_trim_tokens": 0,
        "trajectories": [
            {
                "trajectory_id": 7,
                "accumulated_token_ids": [1, 2, 10],
                "records": [
                    {
                        "prompt_token_count": 2,
                        "output_logprobs": [[-0.1, 10]],
                        "finish_reason": "stop",
                    }
                ],
            }
        ],
    }

    get_calls = 0

    async def get_session_data(session_id):
        nonlocal get_calls
        get_calls += 1
        assert session_id == "sid"
        return session_data

    loop.get_session_data = get_session_data
    training_data = await loop.get_training_data("sid")

    assert len(training_data) == 1
    primary = training_data[0]
    assert primary["trajectory_id"] == 7
    assert primary["prompt_ids"] == [1, 2]
    assert primary["response_ids"] == [10]
    assert primary["rollout_instance_id"] == ("worker", 1)
    assert primary["finish_reason"] == "stop"
    assert primary["num_turns"] == 1
    assert get_calls == 1


@pytest.mark.asyncio
async def test_get_training_data_collects_all_trajectories():
    loop = _bare_loop("auto")
    session_data = {
        "max_trim_tokens": 0,
        "trajectories": [
            {
                "trajectory_id": 0,
                "accumulated_token_ids": [1, 10],
                "records": [{"prompt_token_count": 1, "output_logprobs": [[-0.1, 10]]}],
            },
            {
                "trajectory_id": 1,
                "accumulated_token_ids": [2, 20],
                "records": [{"prompt_token_count": 1, "output_logprobs": [[-0.2, 20]]}],
            },
        ],
    }

    async def get_session_data(session_id):
        assert session_id == "sid"
        return session_data

    loop.get_session_data = get_session_data
    training_data = await loop.get_training_data("sid")

    assert [item["trajectory_id"] for item in training_data] == [0, 1]
    assert [item["prompt_ids"] for item in training_data] == [[1], [2]]
    assert [item["response_ids"] for item in training_data] == [[10], [20]]


@pytest.mark.asyncio
async def test_primary_training_data_returns_the_only_trajectory():
    loop = _bare_loop()
    expected = {"trajectory_id": 0, "response_ids": [1]}

    async def get_training_data(session_id):
        assert session_id == "sid"
        return [expected]

    loop.get_training_data = get_training_data
    assert await loop.get_primary_training_data("sid") is expected


@pytest.mark.asyncio
async def test_primary_training_data_rejects_multiple_trajectories():
    loop = _bare_loop("auto")

    async def get_training_data(session_id):
        assert session_id == "sid"
        return [{"response_ids": [1]}, {"response_ids": [2]}]

    loop.get_training_data = get_training_data
    with pytest.raises(RuntimeError, match="produced 2 trajectories"):
        await loop.get_primary_training_data("sid")


@pytest.mark.asyncio
async def test_get_training_data_rejects_legacy_flat_snapshot():
    loop = _bare_loop()

    async def get_session_data(session_id):
        assert session_id == "sid"
        return {"session_id": session_id, "accumulated_token_ids": [], "records": []}

    loop.get_session_data = get_session_data
    with pytest.raises(RuntimeError, match="expected a trajectories list"):
        await loop.get_training_data("sid")


@pytest.mark.asyncio
async def test_normalize_messages_returns_canonical_urls_without_copying():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                {"type": "text", "text": "describe it"},
            ],
        }
    ]

    normalized = await normalize_messages(messages)

    assert normalized is messages


@pytest.mark.asyncio
async def test_normalize_messages_aligns_bare_placeholder_by_prompt_ordinal():
    original_url_part = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/original.png", "detail": "high"},
    }
    bare_part = {"type": "image", "detail": "low"}
    messages = [{"role": "user", "content": [original_url_part, bare_part]}]

    normalized = await normalize_messages(
        messages,
        mm_data={"images": ["decoded-first-image", "https://example.com/fallback-second.png"]},
    )

    assert normalized is not messages
    assert normalized[0] is not messages[0]
    assert normalized[0]["content"] is not messages[0]["content"]
    assert normalized[0]["content"][0] is original_url_part
    assert normalized[0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/fallback-second.png", "detail": "low"},
    }
    assert messages[0]["content"][1] is bare_part


@pytest.mark.asyncio
async def test_normalize_messages_rejects_ambiguous_fallback_count():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "input_image"},
            ],
        }
    ]

    with pytest.raises(ValueError, match="Cannot align image placeholders"):
        await normalize_messages(messages, mm_data={"images": ["only-one"]})
