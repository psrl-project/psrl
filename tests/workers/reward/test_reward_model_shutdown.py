"""
Exercise shutdown of the SMG based reward coordinator's shared runtime.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from psrl.workers.reward.reward_model.coordinator import RewardModelCoordinator

pytestmark = pytest.mark.cpu_test


def _coordinator():
    coordinator = object.__new__(RewardModelCoordinator)
    for flag in (
        "stop_command_handler",
        "stop_sync_and_migrate",
        "stop_sync_status_to_router",
        "stop_process_status_queue",
        "stop_stats_recorder",
        "stop_thunder_agent",
    ):
        setattr(coordinator, flag, False)
    for task in (
        "command_handler_task",
        "process_status_queue_task",
        "sync_status_to_router_task",
        "sync_task",
        "stats_recorder_task",
        "thunder_agent_task",
    ):
        setattr(coordinator, task, None)
    coordinator.model_sync_tasks = set()
    coordinator._stats_recorder = Mock()
    coordinator.status_queue = Mock()
    coordinator.gateway_client = SimpleNamespace(closed=False, close=AsyncMock())
    coordinator._session_client = SimpleNamespace(closed=False, close=AsyncMock())
    return coordinator


def _start_status_tasks(coordinator, completed):
    async def run(flag, name):
        while not getattr(coordinator, flag):
            await asyncio.sleep(0)
        completed.append(name)

    coordinator.process_status_queue_task = asyncio.create_task(run("stop_process_status_queue", "status"))
    coordinator.sync_status_to_router_task = asyncio.create_task(run("stop_sync_status_to_router", "router"))
    return run


def _assert_resources_closed(coordinator):
    coordinator._stats_recorder.close.assert_called_once_with()
    coordinator.status_queue.close.assert_called_once_with()
    coordinator.gateway_client.close.assert_awaited_once_with()
    coordinator._session_client.close.assert_awaited_once_with()


def test_reward_coordinator_stop_settles_tasks_and_closes_resources():
    async def scenario():
        coordinator = _coordinator()
        completed = []
        run = _start_status_tasks(coordinator, completed)
        coordinator.command_handler_task = asyncio.create_task(run("stop_command_handler", "command"))
        await asyncio.wait_for(coordinator.stop_busy_loop(), timeout=1.0)
        assert set(completed) == {"command", "status", "router"}
        assert coordinator.stop_command_handler is True
        assert coordinator.stop_process_status_queue is True
        assert coordinator.stop_sync_status_to_router is True
        _assert_resources_closed(coordinator)

    asyncio.run(scenario())


@pytest.mark.parametrize("command_started", [False, True])
def test_reward_coordinator_stop_closes_resources_when_command_loop_has_ended(command_started):
    async def scenario():
        coordinator = _coordinator()
        completed = []
        _start_status_tasks(coordinator, completed)
        if command_started:
            coordinator.command_handler_task = asyncio.create_task(asyncio.sleep(0))
            await coordinator.command_handler_task
        await asyncio.wait_for(coordinator.stop_busy_loop(), timeout=1.0)
        assert set(completed) == {"status", "router"}
        _assert_resources_closed(coordinator)

    asyncio.run(scenario())


def test_reward_coordinator_failed_task_still_settles_siblings_and_closes_resources():
    async def scenario():
        coordinator = _coordinator()
        completed = []
        _start_status_tasks(coordinator, completed)

        async def fail():
            raise RuntimeError("Command loop failed.")

        coordinator.command_handler_task = asyncio.create_task(fail())
        with pytest.raises(RuntimeError, match="Command loop failed"):
            await asyncio.wait_for(coordinator.stop_busy_loop(), timeout=1.0)
        assert set(completed) == {"status", "router"}
        _assert_resources_closed(coordinator)

    asyncio.run(scenario())
