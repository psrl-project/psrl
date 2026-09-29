from contextlib import asynccontextmanager

import pytest
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop, SessionAgentResult
from psrl.workers.agent_loop.loops.utils import TerminateReason


class ExampleSessionAgentLoop(SessionAgentLoop):
    async def run_session(self, request, api_base_url):
        assert request["raw_prompt"]
        assert api_base_url == "http://session-router/sessions/sid/v1"
        return SessionAgentResult(
            extra_fields={"answer": "done"},
        )


@pytest.mark.asyncio
async def test_session_agent_loop_owns_external_agent_lifecycle():
    loop = ExampleSessionAgentLoop.__new__(ExampleSessionAgentLoop)
    loop.session_router_url = "http://session-router"
    loop.trajectory_id_strategy = "manual"

    @asynccontextmanager
    async def session_scope(request):
        assert request["uid"] == 7
        yield "sid"

    async def get_training_data(session_id):
        assert session_id == "sid"
        return [
            {
                "trajectory_id": 0,
                "prompt_ids": [1],
                "response_ids": [2],
                "response_mask": [1],
                "logprobs": [-0.1],
                "routed_experts": None,
                "finish_reason": "stop",
                "num_turns": 3,
            }
        ]

    async def compute_reward_score(output, **request):
        assert request["uid"] == 7
        output.reward_score = 1.0
        return output

    loop.session_scope = session_scope
    loop.get_training_data = get_training_data
    loop.compute_reward_score = compute_reward_score
    output, reason = await loop.run(
        {
            "uid": 7,
            "trajectory_id": 99,
            "raw_prompt": [{"role": "user", "content": "question"}],
        }
    )

    assert reason is TerminateReason.FINISHED
    assert output.extra_fields["answer"] == "done"
    assert output.num_turns == 3
    assert output.reward_score == 1.0


@pytest.mark.asyncio
async def test_auto_session_agent_loop_returns_all_resolved_trajectories():
    loop = ExampleSessionAgentLoop.__new__(ExampleSessionAgentLoop)
    loop.session_router_url = "http://session-router"
    loop.trajectory_id_strategy = "auto"

    @asynccontextmanager
    async def session_scope(request):
        del request
        yield "sid"

    async def get_training_data(session_id):
        assert session_id == "sid"
        return [
            {
                "trajectory_id": 0,
                "prompt_ids": [1],
                "response_ids": [2],
                "response_mask": [1],
                "logprobs": [-0.1],
                "routed_experts": None,
                "finish_reason": "stop",
                "num_turns": 1,
            },
            {
                "trajectory_id": 1,
                "prompt_ids": [3],
                "response_ids": [4],
                "response_mask": [1],
                "logprobs": [-0.2],
                "routed_experts": None,
                "finish_reason": "stop",
                "num_turns": 1,
            },
        ]

    async def compute_reward_score(outputs, **request):
        assert request["uid"] == 8
        return outputs

    loop.session_scope = session_scope
    loop.get_training_data = get_training_data
    loop.compute_reward_score = compute_reward_score
    outputs, reason = await loop.run({"uid": 8, "raw_prompt": [{"role": "user", "content": "question"}]})

    assert reason is TerminateReason.FINISHED
    assert isinstance(outputs, list)
    assert [output.response_ids for output in outputs] == [[2], [4]]
