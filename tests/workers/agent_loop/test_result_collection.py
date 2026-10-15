"""
Test result collection with batched validation and scalar training requests.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
async def test_collection_batches_validation_before_training():
    manager = object.__new__(PSRL_AgentLoopManager)
    manager.result_queue = asyncio.Queue()
    manager.stop_collect_task = False
    manager.notify_group_failed = AsyncMock()
    calls = []

    async def occupy(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("is_validate"):
            manager.stop_collect_task = True
        return [SimpleNamespace(value="accepted")]

    manager.occupy_requests = occupy
    for uid, validate in [(1, False), (2, True), (3, True)]:
        await manager.put_result(
            {
                "request_id": uid,
                "prompt_id": uid,
                "rollout_instance_id": ("worker", 0),
                "version_tag": 0,
                "is_validate": validate,
            }
        )
    await asyncio.wait_for(manager._collect_results(), timeout=1)
    assert calls[0]["request_id"] == [2, 3], "Validation results must be batched."
    assert calls[0]["is_validate"] is True, "Validation must retain its mode."
    assert calls[1]["request_id"] == 1, "Training results must be scalar."
    manager.notify_group_failed.assert_not_awaited()


@pytest.mark.asyncio
async def test_ready_validation_buffer_wakes_waiter_and_deletes_buffer():
    manager = object.__new__(PSRL_AgentLoopManager)
    batch = object()
    manager.val_data_buffers = {5: batch}
    waiter = asyncio.get_running_loop().create_future()
    manager._val_buffer_waiters = {5: [waiter]}
    manager.log_ready_buffer = Mock()
    manager.consume_buffer = Mock(return_value=batch)
    manager.ps_manager_handle = SimpleNamespace(maybe_delete_buffer=SimpleNamespace(remote=AsyncMock()))
    await manager.handle_ready_buffer(5, is_validate=True)
    assert waiter.result() is batch, "The validation waiter must receive the ready batch."
    assert not manager._val_buffer_waiters, "The completed waiter must be removed."
    manager.ps_manager_handle.maybe_delete_buffer.remote.assert_awaited_once_with(5, True)
