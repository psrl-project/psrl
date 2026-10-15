"""
Check that inference failures reach pending and future trainer consumers.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager

pytestmark = pytest.mark.cpu_test


def _manager():
    manager = object.__new__(PSRL_AgentLoopManager)
    manager._pipeline_failure = None
    manager._train_buffer_waiters = {}
    manager._train_chunk_waiters = {}
    manager._val_buffer_waiters = {}
    manager._resolved_train_chunks = {}
    manager.train_data_buffers = {}
    manager.train_accumulated_buffers = {}
    manager.val_data_buffers = {}
    manager._val_round_all_failed = False
    manager.ps_manager_handle = SimpleNamespace(
        ensure_train_buffer_exists=SimpleNamespace(remote=AsyncMock()),
        ensure_validate_buffer_exists=SimpleNamespace(remote=AsyncMock()),
        acquire=SimpleNamespace(remote=AsyncMock(return_value=True)),
        release=SimpleNamespace(remote=AsyncMock()),
    )
    return manager


def _wait(manager, consumer):
    if consumer == "train":
        return manager.wait_for_training_batch(7)
    if consumer == "chunk":
        return manager.wait_for_training_chunk(7, 0)
    if consumer == "validation":
        return manager.wait_for_validation_batch(7)
    return manager.wait_for_training_readiness(7, timeout_s=60)


def test_inference_failure_releases_all_pending_consumers_and_preserves_first_cause():
    async def scenario():
        manager = _manager()
        tasks = [asyncio.create_task(_wait(manager, consumer)) for consumer in ["train", "chunk", "validation"]]
        await asyncio.sleep(0)
        assert all(not task.done() for task in tasks), "Consumers must be waiting when inference fails."
        manager.fail_pipeline("Rollout replica 2 exited during wake up.")
        manager.fail_pipeline("Later status polling also failed.")
        for task in tasks:
            with pytest.raises(RuntimeError, match="replica 2 exited"):
                await asyncio.wait_for(task, timeout=1)
        assert not manager._train_buffer_waiters, "Failed consumers must release their waiter records."
        assert not manager._train_chunk_waiters, "Failed chunk consumers must release their waiter records."
        assert not manager._val_buffer_waiters, "Failed validation consumers must release their waiter records."
        assert "Later status" not in manager._pipeline_failure, "Repeated failures must preserve the first cause."

    asyncio.run(scenario())


@pytest.mark.parametrize("consumer", ["train", "chunk", "validation", "readiness"])
def test_late_consumer_reports_inference_failure_even_when_data_is_ready(consumer):
    async def scenario():
        manager = _manager()
        manager.train_data_buffers[7] = ["ready"]
        manager.val_data_buffers[7] = ["ready"]
        manager._resolved_train_chunks[(7, 0)] = (["ready"], True)
        manager.fail_pipeline("Reward model engine exited.")
        with pytest.raises(RuntimeError, match="Reward model engine exited"):
            await asyncio.wait_for(_wait(manager, consumer), timeout=1)

    asyncio.run(scenario())


@pytest.mark.parametrize("consumer", ["train", "validation"])
def test_failure_during_buffer_preparation_does_not_register_a_stranded_waiter(consumer):
    async def scenario():
        manager = _manager()

        async def prepare(*args):
            manager.fail_pipeline("Engine failed during buffer preparation.")

        method = "ensure_train_buffer_exists" if consumer == "train" else "ensure_validate_buffer_exists"
        getattr(manager.ps_manager_handle, method).remote = prepare
        with pytest.raises(RuntimeError, match="during buffer preparation"):
            await asyncio.wait_for(_wait(manager, consumer), timeout=1)
        assert not manager._train_buffer_waiters, "Failed preparation must not register another training waiter."
        assert not manager._val_buffer_waiters, "Failed preparation must not register another validation waiter."

    asyncio.run(scenario())
