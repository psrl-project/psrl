"""
Check trainer readiness observation without consuming data or claiming GPUs.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager

pytestmark = pytest.mark.cpu_test


def _manager():
    manager = object.__new__(PSRL_AgentLoopManager)
    manager.train_data_buffers = {}
    manager.train_accumulated_buffers = {7: {0: ["entry-1"]}}
    manager.train_accumulated_buffer_size = {7: 1}
    manager.ready_entries_per_buffer = 2
    manager._resolved_train_chunks = {}
    manager._train_buffer_waiters = {}
    manager._refill_breaker_diagnosis = None
    manager.ps_manager_handle = SimpleNamespace(ensure_train_buffer_exists=SimpleNamespace(remote=AsyncMock()))
    return manager


def test_wait_for_training_readiness_observes_buffer_without_consuming():
    async def scenario():
        manager = _manager()

        async def mark_ready():
            await asyncio.sleep(0.01)
            manager.train_data_buffers[7] = ["ready"]

        marker = asyncio.create_task(mark_ready())
        ready = await manager.wait_for_training_readiness(7, timeout_s=0.2)
        await marker
        assert ready is True
        assert manager.train_data_buffers[7] == ["ready"]
        assert manager._train_buffer_waiters == {}
        manager.ps_manager_handle.ensure_train_buffer_exists.remote.assert_not_awaited()

    asyncio.run(scenario())


def test_first_training_chunk_is_ready_without_full_batch():
    manager = _manager()
    manager._resolved_train_chunks[(7, 0)] = (["first-chunk"], False)
    assert asyncio.run(manager.wait_for_training_readiness(7, timeout_s=0)) is True
    assert manager._resolved_train_chunks[(7, 0)] == (["first-chunk"], False)
    assert 7 not in manager.train_data_buffers


def test_later_training_chunk_does_not_make_first_chunk_ready():
    manager = _manager()
    manager._resolved_train_chunks[(7, 1)] = (["second-chunk"], True)
    assert asyncio.run(manager.wait_for_training_readiness(7, timeout_s=0)) is False


def test_wait_for_training_readiness_times_out_without_consuming_accumulator():
    manager = _manager()
    assert asyncio.run(manager.wait_for_training_readiness(7, timeout_s=0.01)) is False
    assert manager.train_accumulated_buffers == {7: {0: ["entry-1"]}}
    assert manager.train_accumulated_buffer_size == {7: 1}
    assert manager._train_buffer_waiters == {}
    assert manager.train_data_buffers == {}


def test_readiness_reports_refill_failure_even_if_data_is_present():
    manager = _manager()
    manager._refill_breaker_diagnosis = "Rollouts cannot refill the buffer."
    manager.train_data_buffers[7] = ["ready"]
    with pytest.raises(RuntimeError, match="Rollouts cannot refill"):
        asyncio.run(manager.wait_for_training_readiness(7, timeout_s=0))


def test_waiting_hint_reports_accumulation_for_requested_buffer():
    manager = _manager()
    assert manager.get_trainer_waiting_hint(7) == {"pending_total": 1, "waiting_on": "rollout"}
    manager.train_data_buffers[7] = ["ready"]
    assert manager.get_trainer_waiting_hint(7) == {"pending_total": 0, "waiting_on": "none"}


def test_waiting_hint_selects_earliest_consumer_buffer():
    manager = _manager()
    manager._train_buffer_waiters = {8: [], 7: []}
    assert manager.get_trainer_waiting_hint() == {"pending_total": 1, "waiting_on": "rollout"}


def test_waiting_hint_has_no_pending_work_without_a_consumer():
    assert _manager().get_trainer_waiting_hint() == {"pending_total": 0, "waiting_on": "none"}


def test_wait_for_ready_batch_consumes_through_current_buffer_api():
    manager = _manager()
    manager.train_data_buffers[7] = ["ready"]
    manager.consume_buffer = Mock(side_effect=manager.train_data_buffers.pop)
    assert asyncio.run(manager.wait_for_training_batch(7)) == ["ready"]
    manager.consume_buffer.assert_called_once_with(7)
    assert 7 not in manager.train_data_buffers
