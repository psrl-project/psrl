from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from examples.mini_swe.runner import _build_model_config
from psrl.utils.rollout.vision_utils import normalize_messages
from psrl.utils.sampler_distribution import SamplerCapture
from psrl.utils.subagent_training import SubagentTrainingPolicy
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
    loop.rollout_config = {}
    loop._capture_sessions = set()
    # The base constructor normally sets this from the rollout config, so a __new__-built loop must too.
    loop.sampler_capture = SamplerCapture()

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
async def test_get_training_data_rejects_a_capture_without_distributions():
    loop = _bare_loop()
    loop._capture_sessions = {"sid"}
    session_data = {
        "max_trim_tokens": 0,
        "trajectories": [
            {
                "trajectory_id": 7,
                "accumulated_token_ids": [1, 2, 10],
                "records": [{"prompt_token_count": 2, "output_logprobs": [[-0.1, 10]]}],
            }
        ],
    }

    async def get_session_data(session_id):
        assert session_id == "sid"
        return session_data

    loop.get_session_data = get_session_data
    with pytest.raises(ValueError, match="Sampler capture was requested"):
        await loop.get_training_data("sid")


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
async def test_primary_training_data_selects_the_marked_row():
    loop = _bare_loop("auto")
    expected = {"response_ids": [1], "session_primary": True}

    async def get_training_data(session_id):
        assert session_id == "sid"
        return [{"response_ids": [0], "session_primary": False}, expected]

    loop.get_training_data = get_training_data
    assert await loop.get_primary_training_data("sid") is expected


@pytest.mark.asyncio
async def test_primary_training_data_rejects_untagged_multiple_rows():
    loop = _bare_loop("auto")

    async def get_training_data(session_id):
        assert session_id == "sid"
        return [{"response_ids": [1]}, {"response_ids": [2]}]

    loop.get_training_data = get_training_data
    with pytest.raises(RuntimeError, match="did not mark a primary"):
        await loop.get_primary_training_data("sid")


@pytest.mark.asyncio
async def test_get_training_data_rejects_legacy_flat_snapshot():
    loop = _bare_loop()

    async def get_session_data(session_id):
        assert session_id == "sid"
        return {"session_id": session_id, "accumulated_token_ids": [], "records": []}

    loop.get_session_data = get_session_data
    with pytest.raises(ValueError, match="trajectories list"):
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


_BUDGET = 65536


def _loop_with_budget(rollout_budget: int) -> SessionAgentLoop:
    loop = _bare_loop()
    loop.rollout_budget = rollout_budget
    return loop


def _training_data(prompt_tokens: int, response_tokens: int, routed_experts_rows: int = 0) -> dict:
    return {
        "trajectory_id": 3,
        "prompt_ids": list(range(prompt_tokens)),
        "response_ids": list(range(response_tokens)),
        "response_mask": [1] * response_tokens,
        "logprobs": [-0.5] * response_tokens,
        "routed_experts": None if routed_experts_rows == 0 else [[0]] * routed_experts_rows,
        "num_turns": 7,
        "finish_reason": "stop",
    }


def test_build_token_output_keeps_a_response_longer_than_response_length() -> None:
    # A multi-turn trajectory is one sample whose limit is the total context, so a
    # 40k response must survive while prompt+response fits the trainable budget.
    loop = _loop_with_budget(_BUDGET)
    training_data = _training_data(prompt_tokens=1000, response_tokens=40000)

    output = loop.build_token_output(training_data)

    assert len(output.response_ids) == 40000
    assert len(output.response_mask) == 40000
    assert len(output.response_log_probs) == 40000


def test_build_token_output_clamps_only_at_the_shared_budget() -> None:
    loop = _loop_with_budget(_BUDGET)
    training_data = _training_data(prompt_tokens=30000, response_tokens=40000)

    output = loop.build_token_output(training_data)

    assert len(output.response_ids) == _BUDGET - 30000
    assert len(output.response_mask) == len(output.response_ids)
    assert len(output.response_log_probs) == len(output.response_ids)


def test_build_token_output_trims_routed_experts_to_the_kept_tokens() -> None:
    loop = _loop_with_budget(_BUDGET)
    training_data = _training_data(prompt_tokens=30000, response_tokens=40000, routed_experts_rows=200000)

    output = loop.build_token_output(training_data)

    assert len(output.routed_experts) == 30000 + (_BUDGET - 30000)


def _subagent_session(*, mode: str = "main_only") -> dict:
    return {
        "session_id": "sid",
        "max_trim_tokens": 0,
        "main_agent_id": "main",
        "subagent_training": {
            "mode": mode,
            "train_agents": [],
            "shared_prefix": "owner",
            "reward": "session",
            "loss_weight": {"main": 1.0, "subagent": 2.0},
            "require_primary_capture": True,
            "val_mode": "main_only",
        },
        "trajectories": [
            {
                "trajectory_id": 0,
                "thread_source": "main",
                "agent_id": "main",
                "parent_trajectory_id": None,
                "capture": "full",
                "accumulated_token_ids": [1, 2, 3],
                "records": [
                    {
                        "prompt_token_count": 2,
                        "output_token_ids": [3],
                        "output_logprobs": [[-0.1, 3]],
                        "owner_trajectory_id": 0,
                        "capture": "full",
                    }
                ],
            },
            {
                "trajectory_id": 1,
                "thread_source": "subagent",
                "agent_id": "sub-1",
                "parent_trajectory_id": 0,
                "capture": "full",
                "accumulated_token_ids": [1, 2, 3, 4, 5],
                "records": [
                    {
                        "prompt_token_count": 2,
                        "output_token_ids": [3],
                        "output_logprobs": [[-0.1, 3]],
                        "owner_trajectory_id": 0,
                        "capture": "full",
                    },
                    {
                        "prompt_token_count": 4,
                        "output_token_ids": [5],
                        "output_logprobs": [[-0.2, 5]],
                        "owner_trajectory_id": 1,
                        "capture": "full",
                    },
                ],
            },
        ],
    }


@pytest.mark.asyncio
async def test_get_training_data_main_only_returns_only_the_main_row():
    loop = _bare_loop("auto")
    session = _subagent_session()

    async def get_session_data(session_id):
        return session

    loop.get_session_data = get_session_data
    training_data = await loop.get_training_data("sid")

    assert len(training_data) == 1
    primary = training_data[0]
    assert primary["trajectory_id"] == 0
    assert primary["session_primary"] is True
    assert primary["thread_source"] == "main"
    assert primary["loss_weight"] == 1.0
    assert primary["response_ids"] == [3]
    assert primary["response_mask"] == [1]


@pytest.mark.asyncio
async def test_get_training_data_all_returns_subagent_row_with_session_primary_on_main():
    loop = _bare_loop("auto")
    session = _subagent_session(mode="all")

    async def get_session_data(session_id):
        return session

    loop.get_session_data = get_session_data
    training_data = await loop.get_training_data("sid")

    assert [item["trajectory_id"] for item in training_data] == [0, 1]
    assert [item["session_primary"] for item in training_data] == [True, False]
    subagent = training_data[1]
    assert subagent["agent_id"] == "sub-1"
    assert subagent["parent_trajectory_id"] == 0
    assert subagent["prompt_ids"] == [1, 2, 3, 4]
    assert subagent["response_ids"] == [5]
    assert subagent["response_mask"] == [1]
    assert subagent["loss_weight"] == 2.0


@pytest.mark.asyncio
async def test_get_training_data_uses_the_snapshot_policy_not_the_loop_config():
    loop = _bare_loop("auto")
    session = _subagent_session(mode="all")

    async def get_session_data(session_id):
        return session

    loop.get_session_data = get_session_data
    training_data = await loop.get_training_data("sid")

    assert len(training_data) == 2


@pytest.mark.asyncio
async def test_get_primary_training_data_returns_the_main_row():
    loop = _bare_loop("auto")
    expected = {"trajectory_id": 0, "session_primary": True}

    async def get_training_data(session_id, *, validate=None):
        assert validate is None
        return [{"trajectory_id": 1, "session_primary": False}, expected]

    loop.get_training_data = get_training_data
    assert await loop.get_primary_training_data("sid") is expected


@pytest.mark.asyncio
async def test_create_session_sends_the_resolved_subagent_policy(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del url, max_retries, kwargs
        captured["payload"] = payload
        return {"session_id": "sid"}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop()
    loop.subagent_training = SubagentTrainingPolicy.from_mapping(
        {"mode": "selected", "val_mode": "selected", "train_agents": ["x"]}
    )
    await loop.create_session({"uid": 1})

    assert captured["payload"]["subagent_training"]["mode"] == "selected"
    assert captured["payload"]["subagent_training"]["val_mode"] == "selected"
    assert captured["payload"]["subagent_training"]["train_agents"] == ["x"]


@pytest.mark.asyncio
async def test_create_session_applies_a_row_level_policy_override(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del url, max_retries, kwargs
        captured["payload"] = payload
        return {"session_id": "sid"}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop()
    loop.subagent_training = SubagentTrainingPolicy.from_mapping({"mode": "main_only"})
    await loop.create_session({"uid": 1, "extra_info": {"subagent_training": {"mode": "all"}}})

    assert captured["payload"]["subagent_training"]["mode"] == "all"


def test_shared_context_annotation_uses_the_tree():
    loop = _bare_loop("auto")
    trajectory = {
        "trajectory_id": 1,
        "accumulated_token_ids": [1, 2, 3, 4, 5],
        "records": [
            {"prompt_token_count": 2, "output_token_ids": [3], "output_logprobs": [[-0.1, 3]]},
            {"prompt_token_count": 4, "output_token_ids": [5], "output_logprobs": [[-0.2, 5]]},
        ],
    }
    session = {
        "tree": {
            "nodes": [
                {"id": 10, "stored": True, "trajectory_ids": [0, 1]},
                {"id": 11, "stored": True, "trajectory_ids": [1]},
            ],
            "leaves": [{"trajectory_id": 1, "path_node_ids": [10, 11]}],
        }
    }

    loop._annotate_shared_records(trajectory, session)

    assert [record["shared"] for record in trajectory["records"]] == [True, False]


def test_context_mode_masks_a_shared_owned_turn():
    loop = _bare_loop("auto")
    trajectory = {
        "trajectory_id": 0,
        "accumulated_token_ids": [1, 2, 3],
        "records": [
            {
                "prompt_token_count": 2,
                "output_token_ids": [3],
                "output_logprobs": [[-0.1, 3]],
                "owner_trajectory_id": 0,
            }
        ],
        "shared": True,
    }
    trajectory["records"][0]["shared"] = True
    session = {"max_trim_tokens": 0}

    built = loop.build_training_data_from_trajectory(
        trajectory,
        session,
        trainable_trajectory_id=0,
        shared_prefix="context",
    )

    assert built["response_mask"] == [0]


@pytest.mark.asyncio
async def test_mark_main_close_records_the_router_count(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del max_retries, headers, kwargs
        captured["url"] = url
        captured["payload"] = payload
        return {"background_live_count": 2}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop()
    loop._created_sessions = {"sid"}
    loop._main_closed_sessions = set()

    assert await loop.mark_main_close("sid") == 2
    assert loop._background_subagent_counts == {"sid": 2}
    assert captured["url"].endswith("/sessions/sid/main-close")


@pytest.mark.asyncio
async def test_mark_main_close_keeps_counts_per_session(monkeypatch):
    counts = {"a": 2, "b": 0}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del payload, max_retries, headers, kwargs
        return {"background_live_count": counts[url.rsplit("/", 2)[-2]]}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop()
    loop._created_sessions = {"a", "b"}
    loop._main_closed_sessions = set()

    assert await loop.mark_main_close("a") == 2
    assert await loop.mark_main_close("b") == 0
    assert await loop.mark_main_close("a") == 2


@pytest.mark.asyncio
async def test_mark_main_close_uses_the_harness_lifecycle_count(monkeypatch):
    captured = {}

    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del url, max_retries, headers, kwargs
        captured["payload"] = payload
        return {"background_live_count": 1}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop()
    loop._created_sessions = {"sid"}
    loop._main_closed_sessions = set()
    loop._episode_summary = SimpleNamespace(background_subagents=1)

    assert await loop.mark_main_close("sid") == 1
    assert captured["payload"] == {"background_subagents": 1}


@pytest.mark.asyncio
async def test_get_training_data_attaches_the_background_count(monkeypatch):
    async def fake_post(url, payload, max_retries=5, headers=None, **kwargs):
        del url, payload, max_retries, headers, kwargs
        return {"background_live_count": 2}

    monkeypatch.setattr(module, "post", fake_post)
    loop = _bare_loop("auto")
    session = _subagent_session()
    loop._created_sessions = {"sid"}
    loop._main_closed_sessions = set()

    async def get_session_data(session_id, *, format=None):
        assert session_id == "sid"
        return session

    loop.get_session_data = get_session_data
    training_data = await loop.get_training_data("sid")

    assert training_data
    assert all(item["background_subagent_count"] == 2 for item in training_data)
