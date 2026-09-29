"""CPU scheduling tests with the unrelated Ray/Torch base replaced."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.sandbox import SandboxManager, SandboxSource, SandboxSpec
from psrl.sandbox.backends.docker import DockerBackend
from psrl.workers.agent_loop.harness import HarnessTaskContext
from tests.sandbox.test_docker_backend import FakeDockerEngine

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def harness_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "psrl.workers.agent_loop.loops.session_agent_loop",
        SimpleNamespace(SessionAgentLoop=object),
    )
    monkeypatch.setitem(sys.modules, "psrl.workers.agent_loop.context", SimpleNamespace(AgentLoopContext=object))
    monkeypatch.setitem(sys.modules, "psrl.workers.agent_loop.loops.utils", SimpleNamespace(TerminateReason=object))
    path = Path(__file__).parents[2] / "psrl/workers/agent_loop/loops/harness_agent_loop.py"
    spec = importlib.util.spec_from_file_location("harness_preparation_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_grader_preparation_overlaps_rollout_without_allocating_a_grader(harness_module, monkeypatch) -> None:
    engine = FakeDockerEngine()
    engine.pulled.append("rollout")
    backend = DockerBackend(engine=engine)
    manager = SandboxManager({"docker": backend}, "docker")
    started = asyncio.Event()
    finish = asyncio.Event()

    async def pull(reference, auth):
        started.set()
        await finish.wait()
        engine.pulled.append(reference)

    monkeypatch.setattr(engine, "pull_image", pull)
    task = HarnessTaskContext(
        state=None,
        prompt="fix",
        sandbox_spec=SandboxSpec(SandboxSource.image("rollout")),
        clean_sandbox_spec=SandboxSpec(SandboxSource.image("grader")),
    )

    async def run_harness(*args):
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not finish.is_set(), "Rollout must start before grader preparation finishes."
        assert len(manager._leases) == 1, "Speculative preparation must not reserve grader capacity."
        assert engine.config["Image"] == "rollout", "Only the rollout container should exist."
        finish.set()
        return SimpleNamespace()

    harness = SimpleNamespace(prepare=AsyncMock(), run=run_harness, abort=AsyncMock())
    monkeypatch.setattr(harness_module, "create_harness", lambda *args: harness)
    loop = harness_module.HarnessAgentLoop.__new__(harness_module.HarnessAgentLoop)
    output = SimpleNamespace()
    loop.sandbox_manager = manager
    loop.resolve_request_settings = lambda request: None
    loop.prepare_harness_task = AsyncMock(return_value=task)
    loop.attach_runtime_mount = lambda value: value
    loop.create_session = AsyncMock(return_value="session")
    loop.session_root_url = lambda *args: "http://router"
    loop.harness_config = SimpleNamespace(callback_base_url=None)
    loop.model_config = SimpleNamespace(path="model")
    loop.compaction_budget = None
    loop.max_turns = 1
    loop.collect_harness_artifact = AsyncMock(return_value="patch")
    loop.get_training_data = AsyncMock(return_value=[{"num_turns": 1, "response_ids": [1]}])
    loop.delete_session = AsyncMock()
    loop.finalize_harness_task = AsyncMock(return_value={})
    loop._build_reward_info = lambda *args: {}
    loop._build_capped_output = lambda *args: output
    loop.attach_tito_tree_metadata = lambda *args: None
    loop.compute_reward_score = AsyncMock(return_value=output)
    loop.get_harness_terminate_reason = lambda *args: "finished"
    loop.close_harness_task = AsyncMock()
    try:
        result, reason = await loop.run({})
        assert result is output and reason == "finished", "Preparation must preserve the rollout result."
        assert not manager._leases, "The rollout lease must be released before finalization."
        assert engine.pulled == ["rollout", "grader"], "Grader preparation must finish during rollout."
        assert loop.finalize_harness_task.await_args.args[2] is None, "Docker must not commit an unprepared sandbox."
        loop.close_harness_task.assert_awaited_once()
    finally:
        await manager.shutdown()


async def test_cancelled_session_setup_cancels_preparation_and_cleans_task(harness_module) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def prepare(*args, **kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    loop = harness_module.HarnessAgentLoop.__new__(harness_module.HarnessAgentLoop)
    task = HarnessTaskContext(state=None, prompt="fix", sandbox_spec=SandboxSpec(SandboxSource.image("rollout")))
    loop.sandbox_manager = SimpleNamespace(prepare=prepare)
    loop.resolve_request_settings = lambda request: None
    loop.prepare_harness_task = AsyncMock(return_value=task)
    loop.attach_runtime_mount = lambda value: value

    async def create_session(request):
        await asyncio.Future()

    loop.create_session = create_session
    loop.close_harness_task = AsyncMock()
    running = asyncio.create_task(loop.run({}))
    await asyncio.wait_for(started.wait(), timeout=1)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    assert cancelled.is_set(), "Task cancellation must cancel the preparation waiter."
    loop.close_harness_task.assert_awaited_once_with(task)
