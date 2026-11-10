"""
Verify reward RPC backpressure, cancellation, and result acknowledgement.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from psrl.workers.agent_loop.loops.generate_agent_loop import GenerateAgentLoop
from psrl.workers.reward.client import (
    RewardManagerClient,
    reward_manager_concurrency,
    reward_pool_concurrency,
    reward_producer_concurrency,
)
from psrl.workers.reward.reward_worker import RewardLoopWorker
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


def _worker_config():
    return SimpleNamespace(num_workers=8, max_concurrency_per_worker=256)


async def _wait_until(
    condition,
) -> None:
    async def wait():
        while not condition():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1.0)


@pytest.mark.parametrize("producer_count", [1, 3, 8, 17])
def test_producer_budgets_reserve_result_and_control_capacity(producer_count):
    worker_config = _worker_config()
    capacities = [reward_producer_concurrency(worker_config, producer_count, index) for index in range(producer_count)]
    request_capacity = sum(capacities)
    result_capacity = reward_pool_concurrency(worker_config)
    assert request_capacity == result_capacity == 2048
    assert max(capacities) - min(capacities) <= 1
    assert reward_manager_concurrency(worker_config) > request_capacity + result_capacity
    assert reward_manager_concurrency(worker_config) < 5000


def test_submissions_are_bounded_until_remote_completion():
    async def scenario():
        responses = []

        def submit(payload):
            response = asyncio.get_running_loop().create_future()
            responses.append((payload, response))
            return response

        manager = SimpleNamespace(compute_score=SimpleNamespace(remote=Mock(side_effect=submit)))
        client = RewardManagerClient(manager, max_pending_requests=4)
        requests = [asyncio.create_task(client.compute_score(index)) for index in range(64)]
        for offset in range(0, len(requests), 4):
            await _wait_until(lambda offset=offset: len(responses) == offset + 4)
            await asyncio.sleep(0)
            assert len(responses) == offset + 4
            for payload, response in responses[offset : offset + 4]:
                response.set_result({"reward_score": payload})
        results = await asyncio.gather(*requests)
        assert [result["reward_score"] for result in results] == list(range(64))
        assert not client._pending

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["compute_score", "put_reward_result"])
def test_cancelled_caller_keeps_capacity_until_rpc_settles(method):
    async def scenario():
        responses = []

        def submit(payload):
            response = asyncio.get_running_loop().create_future()
            responses.append(response)
            return response

        manager = SimpleNamespace(**{method: SimpleNamespace(remote=Mock(side_effect=submit))})
        client = RewardManagerClient(manager, max_pending_requests=1)
        call = getattr(client, method)
        first = asyncio.create_task(call({}))
        await _wait_until(lambda: len(responses) == 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(call({}))
        await asyncio.sleep(0)
        assert len(responses) == 1
        assert not responses[0].cancelled()
        responses[0].set_exception(RuntimeError("Remote request failed after cancellation."))
        await _wait_until(lambda: len(responses) == 2)
        responses[1].set_result(None)
        assert await second is None
        assert not client._pending

    asyncio.run(scenario())


@pytest.mark.parametrize("submission_error", [True, False])
def test_rpc_failure_releases_capacity_and_reaches_caller(submission_error):
    async def scenario():
        def fail(payload):
            error = RuntimeError("Reward manager died.")
            if submission_error:
                raise error
            future = asyncio.get_running_loop().create_future()
            future.set_exception(error)
            return future

        method = Mock(side_effect=fail)
        client = RewardManagerClient(SimpleNamespace(compute_score=SimpleNamespace(remote=method)), 1)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="Reward manager died"):
                await asyncio.wait_for(client.compute_score({}), timeout=1.0)
        assert method.call_count == 2
        assert not client._pending

    asyncio.run(scenario())


def test_cancelling_queued_request_does_not_submit_an_rpc():
    async def scenario():
        response = asyncio.get_running_loop().create_future()
        method = Mock(return_value=response)
        client = RewardManagerClient(SimpleNamespace(compute_score=SimpleNamespace(remote=method)), 1)
        first = asyncio.create_task(client.compute_score({}))
        await _wait_until(lambda: method.call_count == 1)
        second = asyncio.create_task(client.compute_score({}))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        response.set_result(None)
        await first
        method.assert_called_once()

    asyncio.run(scenario())


@pytest.mark.parametrize("callback_failure", [True, False])
def test_worker_keeps_execution_slot_until_result_is_acknowledged(callback_failure, monkeypatch):
    async def scenario():
        cls = RewardLoopWorker.__ray_metadata__.modified_class.__ray_actor_class__
        worker = object.__new__(cls)
        worker.worker_id = 0
        worker.completed = 0
        worker.failed = 0
        worker.pre_process = Mock(return_value={"responses": torch.ones(1, 2)})
        event_loop = asyncio.get_running_loop()

        def preprocess(executor, function, request):
            future = event_loop.create_future()
            future.set_result(function(request))
            return future

        monkeypatch.setattr(event_loop, "run_in_executor", preprocess)
        worker._compute_score = AsyncMock(return_value={"reward_score": 1.0})
        acknowledgement = asyncio.get_running_loop().create_future()
        callback = Mock(return_value=acknowledgement)
        worker.max_concurrency = 1
        worker.set_reward_manager(SimpleNamespace(put_reward_result=SimpleNamespace(remote=callback)))
        worker._semaphore = asyncio.Semaphore(1)
        request = tu.get_tensordict(
            tensor_dict={
                "uid": np.array(["request"]),
                "request_keys": np.array([["request"]], dtype=object),
                "attempt_id": np.array([1]),
            },
            non_tensor_dict={"validate": False},
        )
        await worker._semaphore.acquire()
        task = asyncio.create_task(worker._run_reward_loop_with_release(request))
        try:
            await _wait_until(lambda: callback.called)
            assert worker._semaphore.locked()
            assert not task.done()
            result = callback.call_args.args[0]
            assert result["uid"] == "request"
            assert result["result"]["reward_score"] == 1.0
            if callback_failure:
                acknowledgement.set_exception(RuntimeError("Reward manager died."))
                with pytest.raises(RuntimeError, match="Reward manager died"):
                    await task
            else:
                acknowledgement.set_result(None)
                await task
            assert not worker._semaphore.locked()
        finally:
            if not acknowledgement.done():
                acknowledgement.set_result(None)
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("launch_async", [True, False])
def test_agent_loop_uses_shared_reward_client(launch_async):
    async def scenario():
        result = {"reward_score": 1.0, "reward_extra_info": {"answer": "correct"}}
        manager = SimpleNamespace(compute_score=SimpleNamespace(remote=AsyncMock(return_value=result)))
        loop = object.__new__(GenerateAgentLoop)
        loop.reward_manager = RewardManagerClient(manager, 1)
        loop.config = SimpleNamespace(reward=SimpleNamespace(launch_reward_fn_async=launch_async))
        output = SimpleNamespace(
            prompt_ids=[1],
            response_ids=[2],
            multi_modal_data=None,
            num_turns=2,
            extra_fields={},
        )
        assert await loop.compute_reward_score(output, uid="request") is output
        manager.compute_score.remote.assert_awaited_once()
        if not launch_async:
            assert output.reward_score == 1.0
            assert output.extra_fields["reward_extra_info"] == {"answer": "correct"}

    asyncio.run(scenario())
