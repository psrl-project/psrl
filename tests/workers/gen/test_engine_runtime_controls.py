"""
Exercise the engine utility boundary without allocating a GPU model.
"""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest
from psrl.workers.gen.rollout_scheduler import RolloutScheduler
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.core_client import AsyncMPClient
from vllm_patches.patches.executor import TMSExecutorPatch

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def engine_controls():
    root = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "psrl_engine_runtime_controls_test",
        root / "patch/vllm/vllm_plugin/vllm_patches/patches/engine_core.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_engine_core_patches()
    return module


def test_runtime_stats_include_skipped_waiting_without_consuming_events(engine_controls):
    scheduler = object.__new__(RolloutScheduler)
    scheduler.log_stats = False
    scheduler.requests = {
        uid: SimpleNamespace(request_id=uid, num_prompt_tokens=10, num_output_tokens=index)
        for index, uid in enumerate(("running", "waiting", "skipped", "finished"))
    }
    scheduler.running = [scheduler.requests["running"]]
    scheduler.waiting = [scheduler.requests["waiting"]]
    scheduler.skipped_waiting = [scheduler.requests["skipped"]]
    scheduler.kv_cache_manager = SimpleNamespace(usage=0.25, make_prefix_cache_stats=Mock())
    scheduler.preemption_req_ids = ["preempted"]
    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    core.model_executor = SimpleNamespace(is_sleeping=False)

    snapshot = core.psrl_get_runtime_snapshot()

    assert snapshot["scheduler_stats"]["num_running_reqs"] == 1
    assert snapshot["scheduler_stats"]["num_waiting_reqs"] == 2
    assert snapshot["scheduler_stats"]["req_id_in_waiting"] == ["waiting", "skipped"]
    assert snapshot["scheduler_stats"]["req_id_to_response_token_num"] == {
        "running": 0,
        "waiting": 1,
        "skipped": 2,
    }
    assert snapshot["is_sleeping"] is False
    assert snapshot["timestamp"].endswith("+00:00")
    assert scheduler.preemption_req_ids == ["preempted"]
    scheduler.kv_cache_manager.make_prefix_cache_stats.assert_not_called()


def test_memory_restore_never_resumes_pending_scheduler_requests(engine_controls):
    core = object.__new__(EngineCore)
    core.is_scheduler_paused = Mock(return_value=True)
    core.resume_scheduler = Mock(side_effect=AssertionError("Pending requests must remain paused."))
    core.model_executor = SimpleNamespace(wake_up=Mock())
    core.psrl_wake_up_memory(["weights"])
    core.psrl_wake_up_memory(["kv_cache"])
    assert core.model_executor.wake_up.call_args_list == [
        call(["weights"]),
        call(["kv_cache"]),
    ]
    core.resume_scheduler.assert_not_called()


@pytest.mark.parametrize("scheduler_paused,memory_sleeping", [(True, False), (True, True), (False, False)])
def test_runtime_sleep_state_tracks_gpu_memory_while_scheduler_is_paused(
    engine_controls,
    scheduler_paused,
    memory_sleeping,
):
    core = object.__new__(EngineCore)
    core.scheduler = SimpleNamespace(psrl_get_runtime_stats=Mock(return_value={}))
    core.is_scheduler_paused = Mock(return_value=scheduler_paused)
    core.model_executor = SimpleNamespace(is_sleeping=memory_sleeping)

    snapshot = core.psrl_get_runtime_snapshot()

    assert snapshot["is_sleeping"] is memory_sleeping, "Paused scheduling must not advertise released GPU memory."
    assert core.psrl_is_memory_sleeping() is memory_sleeping, "Lifecycle queries must agree with runtime snapshots."


def test_memory_restore_rejects_a_running_scheduler(engine_controls):
    core = object.__new__(EngineCore)
    core.is_scheduler_paused = Mock(return_value=False)
    core.model_executor = SimpleNamespace(wake_up=Mock())
    with pytest.raises(RuntimeError, match="paused scheduler"):
        core.psrl_wake_up_memory(["weights"])
    core.model_executor.wake_up.assert_not_called()


def test_runtime_client_probes_every_dp_engine_concurrently(engine_controls):
    async def scenario():
        client = object.__new__(AsyncMPClient)
        client.core_engines = [rank.to_bytes(2, "little") for rank in (2, 3)]
        started = set()
        ready = asyncio.Event()

        async def utility(method, *, engine):
            assert method == "psrl_get_runtime_snapshot"
            started.add(engine)
            if len(started) == 2:
                ready.set()
            await asyncio.wait_for(ready.wait(), 1.0)
            return {"engine": int.from_bytes(engine, "little")}

        client._call_utility_async = AsyncMock(side_effect=utility)
        assert await client.psrl_get_runtime_snapshots_async() == {2: {"engine": 2}, 3: {"engine": 3}}

    asyncio.run(scenario())


@pytest.mark.parametrize("graph_enabled", [False, True])
def test_tms_sleep_tracks_only_the_memory_tags_that_were_released(monkeypatch, graph_enabled):
    monkeypatch.setenv("PSRL_VLLM_PATCHES", "TMS:GRAPH" if graph_enabled else "TMS")
    executor = SimpleNamespace(is_sleeping=False, collective_rpc=Mock())
    TMSExecutorPatch.sleep(executor, level=2)
    assert executor.sleeping_tags == ({"weights", "kv_cache", "graph"} if graph_enabled else {"weights", "kv_cache"})
    assert executor.is_sleeping is True
