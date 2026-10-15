"""
Verify validation progresses while training reward submission and collection are blocked.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from psrl.workers.reward import reward_manager as manager_module
from psrl.workers.reward import reward_worker as worker_module
from psrl.workers.reward.client import RewardManagerClient, RewardSubmissionClient
from psrl.workers.reward.reward_manager import RewardLoopManager
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


def _request(is_validate):
    return tu.get_tensordict(
        tensor_dict={
            "uid": np.array(["shared-uid"]),
            "n_trajectory": np.array([1]),
            "data_source": np.array(["math"]),
        },
        non_tensor_dict={"validate": is_validate},
    )


def _config(logging_path, launch_async):
    def pool():
        return SimpleNamespace(result_queue_size=1, result_drain_batch_size=1)

    return SimpleNamespace(
        psrl=SimpleNamespace(
            logging_path=str(logging_path),
            rollout_coordination=SimpleNamespace(redundant_rollout=SimpleNamespace(enable=False)),
        ),
        reward=SimpleNamespace(
            reward_normalization=None,
            launch_reward_fn_async=launch_async,
            reward_loop_worker=pool(),
            validation_reward_loop_worker=pool(),
        ),
        gen_actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(n=1)),
        train_actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(val_kwargs=SimpleNamespace(n=1))),
    )


@pytest.mark.parametrize("cancel_training", [False, True])
def test_validation_submission_does_not_wait_for_training_rpc_capacity(cancel_training):
    async def scenario():
        training_response = asyncio.get_running_loop().create_future()
        submitted = asyncio.Event()

        def submit(request):
            submitted.set()
            return training_response

        training = SimpleNamespace(compute_score=SimpleNamespace(remote=Mock(side_effect=submit)))
        validation = SimpleNamespace(
            compute_score=SimpleNamespace(remote=AsyncMock(return_value={"reward_score": 1.0}))
        )
        client = RewardSubmissionClient(RewardManagerClient(training, 1), RewardManagerClient(validation, 1))
        first = asyncio.create_task(client.compute_score(_request(False)))
        second = None
        try:
            await asyncio.wait_for(submitted.wait(), timeout=1.0)
            if cancel_training:
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            second = asyncio.create_task(client.compute_score(_request(False)))
            result = await asyncio.wait_for(client.compute_score(_request(True)), timeout=1.0)
            assert result["reward_score"] == 1.0, "Validation must complete with training capacity occupied."
            assert not second.done(), "The second training request must still respect training backpressure."
            training.compute_score.remote.assert_called_once()
            validation.compute_score.remote.assert_awaited_once()
        finally:
            training_response.set_result(None)
            await asyncio.gather(first, *([second] if second else []), return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("launch_async", [True, False])
@pytest.mark.parametrize("blocked_stage", ["dispatch", "collection", "result_queue"])
def test_validation_completes_while_training_pipeline_is_blocked(
    tmp_path,
    monkeypatch,
    launch_async,
    blocked_stage,
):
    async def scenario():
        gate = asyncio.Event()
        blocked = asyncio.Event()
        training_results = []
        writes = []

        async def update_status(uids, status, *, is_validate):
            if (
                not is_validate
                and status == manager_module.PSRL_RequestStatus.REWARD_COMPLETED
                and blocked_stage != "dispatch"
            ):
                blocked.set()
                await gate.wait()
            return [True for _ in uids]

        ps_manager = SimpleNamespace(
            update_request_status=SimpleNamespace(remote=AsyncMock(side_effect=update_status))
        )
        config = _config(tmp_path, launch_async)
        managers = []
        for is_validate in (False, True):
            worker = SimpleNamespace(
                start_busy_loop=SimpleNamespace(remote=Mock()),
                stop_busy_loop=SimpleNamespace(remote=AsyncMock()),
            )
            manager = RewardLoopManager(config, ps_manager, [worker], is_validate=is_validate)

            async def add_requests(requests, *, manager=manager):
                if not manager.is_validate and blocked_stage == "dispatch":
                    blocked.set()
                    await gate.wait()
                for request in requests:
                    result = {
                        "uid": tu.get(request, "uid")[0],
                        "request_keys": tu.get(request, "request_keys")[0],
                        "attempt_id": tu.get(request, "attempt_id")[0],
                        "is_validate": manager.is_validate,
                        "worker_id": 0,
                        "response_len": 2,
                        "result": {"reward_score": 1.0 if manager.is_validate else -1.0},
                        "error": None,
                    }
                    if not manager.is_validate:
                        training_results.append(result)
                    await manager.put_reward_result(result)

            worker.add_reward_requests = SimpleNamespace(remote=AsyncMock(side_effect=add_requests))
            managers.append(manager)

        async def put(keys, partition_id, fields):
            writes.append((keys, partition_id, fields))

        monkeypatch.setattr(manager_module.tq, "async_kv_batch_put", put)
        training, validation = managers
        assert training.dispatch_queue is not validation.dispatch_queue, "Dispatch queues must be independent."
        assert training.result_queue is not validation.result_queue, "Result queues must be independent."
        for manager in managers:
            manager.start_busy_loop()

        training_task = asyncio.create_task(training.compute_score(_request(False)))
        blocked_result = None
        try:
            await asyncio.wait_for(blocked.wait(), timeout=1.0)
            if blocked_stage == "result_queue":
                await training.put_reward_result(training_results[0])
                blocked_result = asyncio.create_task(training.put_reward_result(training_results[0]))
                await asyncio.sleep(0)
                assert not blocked_result.done(), "Training result submission must wait when its queue is full."
            validation_result = await asyncio.wait_for(validation.compute_score(_request(True)), timeout=1.0)
            if launch_async:
                batch = KVBatchMeta(keys=["shared-uid"], tags=[{}], partition_id="val")
                assert await asyncio.wait_for(validation.wait_for_reward_of_requests(batch), timeout=1.0) is batch, (
                    "Validation must write rewards while training is blocked."
                )
                assert writes[0][:2] == (["shared-uid"], "val"), (
                    "Validation rewards must use the validation partition."
                )
                assert tu.get(writes[0][2], "reward_score")[0] == 1.0, "Validation must retain its own result."
            else:
                assert validation_result["reward_score"] == 1.0, "Synchronous validation must return its own result."
            assert not training.request_key_to_future["shared-uid"].done(), "Training must still be blocked."
            if blocked_result is not None:
                assert not blocked_result.done(), "Validation must not consume the training result queue."
        finally:
            gate.set()
            await asyncio.wait_for(training_task, timeout=1.0)
            if blocked_result is not None:
                await asyncio.wait_for(blocked_result, timeout=1.0)
            await asyncio.wait_for(asyncio.gather(*(manager.stop_busy_loop() for manager in managers)), timeout=1.0)

    asyncio.run(scenario())


def test_reward_manager_rejects_requests_results_and_batches_for_the_other_pool(tmp_path):
    async def scenario():
        ps_manager = SimpleNamespace(update_request_status=SimpleNamespace(remote=AsyncMock()))
        manager = RewardLoopManager(_config(tmp_path, True), ps_manager, [Mock()], is_validate=True)
        with pytest.raises(ValueError, match="Reward request"):
            await manager.compute_score(_request(False))
        with pytest.raises(ValueError, match="Reward result"):
            await manager.put_reward_result({"is_validate": False})
        batch = KVBatchMeta(keys=["shared-uid"], tags=[{}], partition_id="train")
        with pytest.raises(ValueError, match="Reward batch"):
            await manager.wait_for_reward_of_requests(batch)
        ps_manager.update_request_status.remote.assert_not_awaited()

    asyncio.run(scenario())


def test_validation_worker_accepts_requests_while_training_worker_queue_is_full(tmp_path, monkeypatch):
    async def scenario():
        cls = worker_module.RewardLoopWorker.__ray_metadata__.modified_class.__ray_actor_class__
        monkeypatch.setattr(worker_module.tq, "init", Mock())
        monkeypatch.setattr(cls, "_init_reward_fn", Mock())
        config = _config(tmp_path, True)
        config.reward.reward_loop_worker.max_concurrency_per_worker = 1
        config.reward.reward_loop_worker.queue_size_per_worker = 1
        config.reward.validation_reward_loop_worker.max_concurrency_per_worker = 2
        config.reward.validation_reward_loop_worker.queue_size_per_worker = 3
        training = cls(config, None, None, [], is_validate=False)
        validation = cls(config, None, None, [], is_validate=True)
        assert training.max_concurrency == 1, "Training worker must use its own concurrency."
        assert validation.max_concurrency == 2, "Validation worker must use its own concurrency."
        assert validation.queue_size == 3, "Validation worker must use its own queue capacity."
        await training.add_reward_requests([_request(False)])
        blocked = asyncio.create_task(training.add_reward_requests([_request(False)]))
        try:
            await asyncio.sleep(0)
            await asyncio.wait_for(validation.add_reward_requests([_request(True)]), timeout=1.0)
            assert len(validation.pending_reward_queue) == 1, "Validation worker must accept its request immediately."
            assert not blocked.done(), "Training worker must still apply queue backpressure."
            with pytest.raises(ValueError, match="worker's pool"):
                await validation.add_reward_requests([_request(False)])
            assert len(validation.pending_reward_queue) == 1, "Validation worker must reject training requests."
        finally:
            training.pending_reward_queue.popleft()
            await asyncio.wait_for(blocked, timeout=1.0)

    asyncio.run(scenario())
