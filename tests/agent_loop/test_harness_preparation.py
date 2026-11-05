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
from psrl.workers.agent_loop.harness import HarnessResult, HarnessTaskContext
from psrl.workers.agent_loop.loops.budget import EpisodeBudget
from psrl.workers.agent_loop.timeouts import resolve_rollout_timeouts
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

    async def run_harness(*args, **kwargs):
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not finish.is_set(), "Rollout must start before grader preparation finishes."
        assert len(manager._leases) == 1, "Speculative preparation must not reserve grader capacity."
        assert engine.config["Image"] == "rollout", "Only the rollout container should exist."
        finish.set()
        return HarnessResult(exit_code=0)

    harness = SimpleNamespace(
        prepare=AsyncMock(),
        run=run_harness,
        abort=AsyncMock(),
        # The lifecycle reads the CLI's own episode report from the harness.
        episode_summary=lambda log_bytes: None,
    )
    monkeypatch.setattr(harness_module, "create_harness", lambda *args: harness)
    loop = harness_module.HarnessAgentLoop.__new__(harness_module.HarnessAgentLoop)
    # Mirrors `AgentLoopBase.__init__`, which `__new__` skipped.
    loop.episode_budget = EpisodeBudget(None, starts_after_provisioning=True)
    loop.timeouts = resolve_rollout_timeouts(None, None)
    # Mirrors `SessionAgentLoop.__init__`: the session view this loop counts turns from.
    loop.session_snapshot = None
    # The base class is stubbed out for this module, so borrow the real arming hook.
    loop.arm_episode_budget = loop.episode_budget.arm
    output = SimpleNamespace()
    loop.sandbox_manager = manager
    loop.resolve_request_settings = lambda request: None
    loop.prepare_harness_task = AsyncMock(return_value=task)
    loop.attach_runtime_mount = lambda value: value
    loop.create_session = AsyncMock(return_value="session")
    loop.session_root_url = lambda *args: "http://router"
    # Isolation is a prompt convention; this scheduling test only needs a no-op.
    loop.harness_config = SimpleNamespace(
        callback_base_url=None,
        task_prompt=lambda prompt, workdir: prompt,
    )
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
    # Mirrors `AgentLoopBase.__init__`, which `__new__` skipped.
    loop.episode_budget = EpisodeBudget(None, starts_after_provisioning=True)
    loop.timeouts = resolve_rollout_timeouts(None, None)
    # Mirrors `SessionAgentLoop.__init__`: the session view this loop counts turns from.
    loop.session_snapshot = None
    # The base class is stubbed out for this module, so borrow the real arming hook.
    loop.arm_episode_budget = loop.episode_budget.arm
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
    loop.close_harness_task.assert_awaited_once()
    cleaned = loop.close_harness_task.await_args.args[0]
    assert cleaned.sandbox_spec.lifetime_timeout_s == loop.timeouts.sandbox_lifetime_s, (
        "The task the loop cleans up carries the sandbox lifetime it derived."
    )


async def test_the_rollout_and_grader_specs_get_the_sandbox_lifetime(harness_module) -> None:
    """A sandbox is reclaimed by a lifetime the caller states, so the loop must state it."""
    from psrl.workers.agent_loop.timeouts import LIFETIME_SLACK_S

    loop = harness_module.HarnessAgentLoop.__new__(harness_module.HarnessAgentLoop)
    loop.timeouts = resolve_rollout_timeouts(3600, 900)
    task = HarnessTaskContext(
        state=None,
        prompt="fix",
        sandbox_spec=SandboxSpec(SandboxSource.image("rollout")),
        clean_sandbox_spec=SandboxSpec(SandboxSource.image("grader")),
    )

    out = loop.apply_sandbox_lifetime(task)

    expected = 3600 + 900 + LIFETIME_SLACK_S
    assert out.sandbox_spec.lifetime_timeout_s == expected
    assert out.clean_sandbox_spec.lifetime_timeout_s == expected
    # The original is left alone; the loop works on the copy.
    assert task.sandbox_spec.lifetime_timeout_s is None


async def test_grading_operation_past_its_budget_releases_the_sandbox(harness_module) -> None:
    """A grading phase past its budget must abort rather than hang.

    The worker thread cannot be killed, so expiry closes the sandbox -- which is what
    unblocks the thread -- and surfaces a ``TimeoutError`` for the task to report the
    episode as ungraded.
    """
    import time

    class SlowSandbox:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class Manager:
        def __init__(self) -> None:
            self.sandbox = SlowSandbox()

        def sync(self, backend=None):  # noqa: ANN001
            return self.sandbox

    loop = harness_module.HarnessAgentLoop.__new__(harness_module.HarnessAgentLoop)
    loop.sandbox_manager = Manager()

    def operation(sync_sandbox):
        # A blocked worker thread that only ends once the sandbox is closed under it.
        deadline = time.monotonic() + 5
        while not sync_sandbox.closed and time.monotonic() < deadline:
            time.sleep(0.01)
        return "late"

    with pytest.raises(TimeoutError):
        await loop._run_sync_sandbox_operation(None, operation, timeout_s=0.05)

    assert loop.sandbox_manager.sandbox.closed, "Expiry must close the sandbox to unblock the worker."
    await asyncio.sleep(0.1)  # let the detached thread drain before the loop closes
