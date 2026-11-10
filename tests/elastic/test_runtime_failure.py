"""
Check that elastic failures end inference and notify the trainer's data owner.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tests.elastic.runtime_helpers import _executor

pytestmark = pytest.mark.cpu_test


def test_engine_rpc_failure_reaches_trainer_and_stops_monitor(modules):
    async def scenario():
        coord = SimpleNamespace(
            get_instance_runtime_snapshot=SimpleNamespace(remote=AsyncMock(side_effect=RuntimeError("engine exited")))
        )
        executor = _executor(modules, coord)
        executor.agent_loop_manager = SimpleNamespace(fail_pipeline=SimpleNamespace(remote=AsyncMock()))
        executor.stop_monitor = False
        executor._sync_router_backlog_from_coordinators = AsyncMock(return_value=True)
        executor.scaling_policy.decide = Mock()
        await asyncio.wait_for(executor._monitor_loop(), timeout=0.5)
        diagnosis = executor.agent_loop_manager.fail_pipeline.remote.call_args.args[0]
        assert "engine exited" in diagnosis, "The trainer must receive the engine failure."
        executor.scaling_policy.decide.assert_not_called()
        coord.get_instance_runtime_snapshot.remote.assert_awaited_once()
        assert executor.stop_scale_up and executor.stop_scale_down, "A failed runtime cannot accept new transitions."

    asyncio.run(scenario())


def test_engine_rpc_timeout_remains_recoverable(modules):
    async def scenario():
        coord = SimpleNamespace(
            get_instance_runtime_snapshot=SimpleNamespace(remote=AsyncMock(side_effect=TimeoutError("late snapshot")))
        )
        executor = _executor(modules, coord)
        executor.agent_loop_manager = SimpleNamespace(fail_pipeline=SimpleNamespace(remote=AsyncMock()))
        assert await executor._sync_engine_status_from_coordinators() is False, (
            "A late observation must reject the current decision tick."
        )
        executor.agent_loop_manager.fail_pipeline.remote.assert_not_awaited()

    asyncio.run(scenario())


def test_timeout_for_one_role_cannot_hide_engine_failure_in_another_role(modules):
    async def scenario():
        coord = SimpleNamespace(
            get_instance_runtime_snapshot=SimpleNamespace(remote=AsyncMock(side_effect=TimeoutError("late snapshot")))
        )
        executor = _executor(modules, coord)
        role = modules.role.RewardModel
        executor.roles.append((role, "reward"))
        executor.coordinators[role] = {
            "reward": SimpleNamespace(
                get_instance_runtime_snapshot=SimpleNamespace(
                    remote=AsyncMock(side_effect=RuntimeError("reward engine exited"))
                )
            )
        }
        with pytest.raises(RuntimeError, match="reward engine exited"):
            await executor._sync_engine_status_from_coordinators()

    asyncio.run(scenario())


def test_failure_keeps_inflight_transitions_reserved_until_they_settle(modules):
    async def scenario():
        executor = _executor(modules, None)
        executor._decision_pending_action_counts = {1: 2}
        executor._decision_settled.clear()
        executor._decision_execution_in_progress = True
        await executor._fail_runtime(RuntimeError("engine exited"), context="Scale failed")
        assert not executor._decision_settled.is_set(), "Failure reporting must not abandon inflight transitions."
        executor._mark_decision_action_finished(1)
        assert not executor._decision_settled.is_set(), "A sibling transition still owns its GPU reservation."
        executor._mark_decision_action_finished(1)
        assert executor._decision_settled.is_set(), "Settled transitions must release pending reclaim operations."
        with pytest.raises(RuntimeError, match="engine exited"):
            await executor._scale_up_instances([])

    asyncio.run(scenario())


def test_failure_cancels_queued_actions_and_forbids_releasing_the_pool(modules):
    async def scenario():
        executor = _executor(modules, None)
        executor.agent_loop_manager = SimpleNamespace(fail_pipeline=SimpleNamespace(remote=AsyncMock()))
        executor._decision_execution_in_progress = True
        executor._decision_pending_action_counts = {1: 2}
        executor._decision_settled.clear()
        executor.scale_up_task_queue = asyncio.Queue()
        executor.scale_down_task_queue = asyncio.Queue()
        executor.scale_up_task_queue.put_nowait({"decision_id": 1})
        executor.scale_down_task_queue.put_nowait({"decision_id": 1})
        await executor._fail_runtime(RuntimeError("worker segfault"), context="Scale up failed")
        assert executor._decision_settled.is_set(), "Cancelled actions must release decision waiters."
        executor._sleep_validation_workers = AsyncMock()
        executor._sleep_trainer = AsyncMock()
        with pytest.raises(RuntimeError, match="worker segfault"):
            await executor.release_trainer_pool()
        executor._sleep_trainer.assert_not_awaited()
        executor._sleep_validation_workers.assert_not_awaited()
        await executor._fail_runtime(RuntimeError("second failure"), context="Status polling failed")
        executor.agent_loop_manager.fail_pipeline.remote.assert_awaited_once()

    asyncio.run(scenario())
