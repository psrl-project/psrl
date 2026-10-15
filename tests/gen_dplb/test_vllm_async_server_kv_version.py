import asyncio
from contextlib import asynccontextmanager, nullcontext
from datetime import datetime, timezone
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import psrl.utils.common.tms_env as tms_env
import psrl.workers.gen.vllm_async_server as server_module
import pytest
from psrl.workers.gen.smg_adapter import build_worker_stats_update
from psrl.workers.gen.stats_collector import DPLBStatCollector, EngineStats
from psrl.workers.gen.vllm_async_server import PSRL_vLLMHttpServer
from vllm.v1.metrics.stats import SchedulerStats

pytestmark = pytest.mark.cpu_test


@asynccontextmanager
async def _pull_context(_):
    yield


def _make_server(current_version: int, actual_version: int):
    server = object.__new__(PSRL_vLLMHttpServer)
    server.curr_rollout_instance_model_version = [current_version, current_version]
    server.get_instance_num = Mock(return_value=2)
    server.base_worker_id = "replica-0"
    server.pull_model = AsyncMock()
    server.kv_cache_manager = Mock()
    server.stat_collector = None
    server.gen_interface = SimpleNamespace(
        ps_manager_handle=SimpleNamespace(
            get_rollout_instance_model_version=SimpleNamespace(
                remote=AsyncMock(return_value=actual_version),
            ),
        ),
    )
    return server


@pytest.mark.parametrize("model_id", ["GLM-Z1-9B-0414", "models/Qwen2.5-1.5B"])
def test_gateway_registration_uses_explicit_served_model_id(model_id):
    server = object.__new__(PSRL_vLLMHttpServer)
    server.node_rank = 0
    server._wait_for_grpc_servicer_ready = AsyncMock()
    server.get_total_kv_cache_tokens = AsyncMock(return_value=40960)
    server.model_config = SimpleNamespace(path="models/Checkpoint")
    server.config = SimpleNamespace(
        data_parallel_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    server.gen_interface = SimpleNamespace(rollout_replica_idx=0, role="validate")
    server.psrl_config = SimpleNamespace(fuse_rollout_with_validate=False)
    server.kv_cache_manager = None
    server._server_address = "localhost"
    server._server_port = 9000
    response = SimpleNamespace(raise_for_status=Mock(), json=AsyncMock(return_value={"worker_id": "0"}))
    registration = {}

    @asynccontextmanager
    async def post(url, json):
        registration.update(json)
        yield response

    server._gateway_client = SimpleNamespace(closed=False, post=post)
    worker_id = asyncio.run(server.register_server_to_gateway("http://localhost:8300", model_id=model_id))

    assert worker_id == "0", "Registration must return the gateway worker ID."
    assert registration["models"] == [{"id": model_id}], "Worker model ID must match request model ID."
    assert registration["labels"]["psrl_request_scope"] == "validate", "Validation must register dedicated scope."


@pytest.mark.unit
def test_pull_model_commits_actual_version_to_local_kv_manager(monkeypatch):
    monkeypatch.setattr(server_module, "shared_pull_model_context_async", _pull_context)
    monkeypatch.setattr(server_module, "log_dual_events", lambda *args, **kwargs: nullcontext())
    server = _make_server(current_version=1, actual_version=3)

    version = asyncio.run(PSRL_vLLMHttpServer.pull_model_for_sync(server, ps_version=2))

    assert version == 3
    assert server.curr_rollout_instance_model_version == [3, 3]
    server.kv_cache_manager.set_current_version.assert_called_once_with(3)


@pytest.mark.unit
def test_pull_model_recommits_current_version_when_pull_is_not_needed():
    server = _make_server(current_version=4, actual_version=5)

    version = asyncio.run(PSRL_vLLMHttpServer.pull_model_for_sync(server, ps_version=3))

    assert version == 4
    server.pull_model.assert_not_awaited()
    server.kv_cache_manager.set_current_version.assert_called_once_with(4)


@pytest.mark.unit
def test_pull_model_does_not_advance_kv_version_when_actual_version_is_too_old(monkeypatch):
    monkeypatch.setattr(server_module, "shared_pull_model_context_async", _pull_context)
    monkeypatch.setattr(server_module, "log_dual_events", lambda *args, **kwargs: nullcontext())
    server = _make_server(current_version=1, actual_version=1)

    with pytest.raises(AssertionError, match="Pulled model version is stale"):
        asyncio.run(PSRL_vLLMHttpServer.pull_model_for_sync(server, ps_version=2))

    server.kv_cache_manager.set_current_version.assert_not_called()


@pytest.mark.parametrize("cuda_graph", [False, True])
def test_reward_wake_resumes_scheduling_only_after_memory_is_restored(cuda_graph):
    server = object.__new__(PSRL_vLLMHttpServer)
    events = []

    async def utility(method, tags):
        assert method == "psrl_wake_up_memory"
        events.append(("wake_memory", tags))

    async def load_weights():
        events.append(("load_weights", None))

    server.gen_interface = SimpleNamespace(ps_manager_handle=None)
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(enable_cuda_graph=cuda_graph))
    server.engine = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=utility),
        resume_generation=AsyncMock(side_effect=lambda: events.append(("resume", None))),
    )
    server.grpc_servicer = SimpleNamespace(
        close_generate_admission=AsyncMock(side_effect=lambda: events.append(("close_admission", None))),
        open_generate_admission=AsyncMock(side_effect=lambda: events.append(("open_admission", None))),
    )
    server._load_reward_weights_from_cpu_cache = load_weights

    asyncio.run(server.wake_up())

    assert events == [
        ("close_admission", None),
        ("wake_memory", ["weights"]),
        ("load_weights", None),
        ("wake_memory", ["kv_cache", "graph"] if cuda_graph else ["kv_cache"]),
        ("resume", None),
        ("open_admission", None),
    ]


@pytest.mark.parametrize("failure_stage", ["weights", "load_weights", "kv_cache"])
def test_reward_wake_failure_keeps_scheduling_and_admission_closed(failure_stage):
    server = object.__new__(PSRL_vLLMHttpServer)
    server.gen_interface = SimpleNamespace(ps_manager_handle=None)
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(enable_cuda_graph=False))

    async def utility(method, tags):
        if failure_stage in tags:
            raise RuntimeError("Memory restore failed.")

    server.engine = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=AsyncMock(side_effect=utility)),
        resume_generation=AsyncMock(),
    )
    server.grpc_servicer = SimpleNamespace(close_generate_admission=AsyncMock(), open_generate_admission=AsyncMock())
    server._load_reward_weights_from_cpu_cache = AsyncMock(
        side_effect=RuntimeError("Memory restore failed.") if failure_stage == "load_weights" else None,
    )
    with pytest.raises(RuntimeError, match="Memory restore failed"):
        asyncio.run(server.wake_up())
    server.grpc_servicer.close_generate_admission.assert_awaited_once()
    server.engine.resume_generation.assert_not_awaited()
    server.grpc_servicer.open_generate_admission.assert_not_awaited()


def test_reward_sleep_closes_admission_before_draining_and_releasing_memory():
    server = object.__new__(PSRL_vLLMHttpServer)
    server.gen_interface = SimpleNamespace(ps_manager_handle=None)
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(range=None))
    events = []
    server.grpc_servicer = SimpleNamespace(
        close_generate_admission=AsyncMock(side_effect=lambda: events.append("close_admission")),
    )

    async def sleep(level, mode):
        events.append(("sleep", level, mode))

    server.engine = SimpleNamespace(sleep=sleep)
    asyncio.run(server.sleep(level=2))
    assert events == ["close_admission", ("sleep", 2, "abort")]


def test_rollout_wake_restores_weights_cache_and_graph_together():
    server = object.__new__(PSRL_vLLMHttpServer)
    wake_calls = []

    async def wake_up(tags):
        wake_calls.append(tags)

    server.gen_interface = SimpleNamespace(ps_manager_handle=object())
    server.engine = SimpleNamespace(wake_up=wake_up)
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(enable_cuda_graph=True))

    asyncio.run(server.wake_up())

    assert wake_calls == [["weights", "kv_cache", "graph"]]


def test_nixl_sleep_deregisters_before_releasing_rollout_memory():
    server = object.__new__(PSRL_vLLMHttpServer)
    events = []
    server._is_init_nixl_client = asyncio.Event()
    server._is_init_nixl_client.set()
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(range=None))
    server.gen_interface = SimpleNamespace(ps_manager_handle=object())
    server.get_instance_num = Mock(return_value=2)
    server.curr_rollout_instance_model_version = [4, 4]

    async def collective_rpc(**kwargs):
        events.append(kwargs["method"])

    async def sleep(level, mode):
        events.append(("sleep", level, mode))

    server.engine = SimpleNamespace(collective_rpc=collective_rpc, sleep=sleep)

    asyncio.run(server.nixl_sleep(level=2))

    assert events == ["nixl_deregister", ("sleep", 2, "abort")]
    assert server.curr_rollout_instance_model_version == [-1, -1]


def test_level_two_sleep_forces_pull_even_when_ps_version_has_not_advanced(monkeypatch):
    monkeypatch.setattr(server_module, "shared_pull_model_context_async", _pull_context)
    monkeypatch.setattr(server_module, "log_dual_events", lambda *args, **kwargs: nullcontext())
    server = _make_server(current_version=4, actual_version=4)
    server.engine = SimpleNamespace(sleep=AsyncMock())
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(range=None))

    async def scenario():
        await server.sleep(level=2)
        return await server.pull_model_for_sync(ps_version=4)

    assert asyncio.run(scenario()) == 4
    server.pull_model.assert_awaited_once()
    assert server.curr_rollout_instance_model_version == [4, 4]


def test_scheduler_snapshot_preserves_waiting_ids_and_utc_time():
    collector = object.__new__(DPLBStatCollector)
    collector.output_queue = Queue()
    collector.start_time = 0.0
    collector.last_record_time = 0.0
    collector.last_push_to_queue_time = 0.0
    collector.replica_idx = 1
    collector.model_version = 3
    collector.psrl_config = SimpleNamespace(
        status_collection=SimpleNamespace(
            dump_logging_to_file_level="none",
            engine_sync_interval_in_ms=100,
        )
    )
    collector.record(
        SchedulerStats(num_waiting_reqs=1, req_id_in_waiting=["cmpl-request-0"]),
        None,
    )
    snapshot = collector.output_queue.get_nowait().snapshot
    assert snapshot["scheduler_stats"]["req_id_in_waiting"] == ["cmpl-request-0"]
    recorded = datetime.fromisoformat(snapshot["timestamp"])
    assert recorded.tzinfo == timezone.utc
    worker_update = build_worker_stats_update("replica-1", 0, snapshot)
    advertised = datetime.fromisoformat(worker_update["timestamp"])
    assert advertised == recorded, "Router timestamps must preserve the engine observation time."
    assert EngineStats.get_default_snapshot()["timestamp"] is None


def test_reward_sleep_aborts_nonresumable_requests():
    server = object.__new__(PSRL_vLLMHttpServer)
    server.gen_interface = SimpleNamespace(ps_manager_handle=None)
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(range=None))
    server.engine = SimpleNamespace(sleep=AsyncMock())
    server.grpc_servicer = SimpleNamespace(close_generate_admission=AsyncMock())

    asyncio.run(server.sleep(level=2))

    server.engine.sleep.assert_awaited_once_with(2, mode="abort")
    server.grpc_servicer.close_generate_admission.assert_awaited_once()


def test_weight_arena_worker_env_matches_role_and_requires_nixl_tms():
    reward_config = {
        "nixl": {"weight_arena": {"reward_enabled": True}},
        "ps_mode": "cpu_ref",
        "tms": {"range": "all"},
    }
    assert server_module._weight_arena_worker_env(reward_config, "reward") == {
        "PSRL_VLLM_WEIGHT_ARENA": "1",
        "VLLM_PSRL_WEIGHT_ARENA": "1",
    }

    rollout_config = {
        "nixl": {"weight_arena": {"rollout_enabled": True}},
        "ps_mode": "nixl_gpu",
        "tms": {"range": "all"},
    }
    assert server_module._weight_arena_worker_env(rollout_config, "validate") == {
        "PSRL_VLLM_WEIGHT_ARENA": "1",
        "VLLM_PSRL_WEIGHT_ARENA": "1",
    }
    rollout_config["ps_mode"] = "cpu_ref"
    with pytest.raises(RuntimeError, match="requires psrl.ps_mode"):
        server_module._weight_arena_worker_env(rollout_config, "rollout")


def test_reward_weight_arena_uses_level2_without_tms_cpu_backup():
    server = object.__new__(PSRL_vLLMHttpServer)
    server.gen_interface = SimpleNamespace(ps_manager_handle=None)
    server.psrl_config = {
        "nixl": {"weight_arena": {"reward_enabled": True}},
    }
    server.config = SimpleNamespace(enable_weights_cpu_backup=True)

    assert server._tms_weights_cpu_backup_enabled() is False


def test_rollout_preserves_explicit_tms_cpu_backup_setting():
    server = object.__new__(PSRL_vLLMHttpServer)
    server.gen_interface = SimpleNamespace(ps_manager_handle=object())
    server.psrl_config = {"nixl": {"weight_arena": {"rollout_enabled": True}}}
    server.config = SimpleNamespace(enable_weights_cpu_backup=True)

    assert server._tms_weights_cpu_backup_enabled() is True


def test_prepend_python_library_path_prioritizes_active_environment(monkeypatch):
    monkeypatch.setattr(tms_env.sys, "prefix", "/opt/psrl-env")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/lib64:/opt/psrl-env/lib")

    assert server_module.prepend_python_library_path() == "/opt/psrl-env/lib:/usr/local/cuda/lib64"
