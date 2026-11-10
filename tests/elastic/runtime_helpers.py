"""
Load elastic runtime components with isolated external service dependencies.
"""

import asyncio
import importlib.util
import logging
import sys
from contextlib import nullcontext
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(monkeypatch, name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    class Role(Enum):
        Rollout = "Rollout"
        RewardModel = "RewardModel"

    class EngineStats(SimpleNamespace):
        @staticmethod
        def get_default_snapshot():
            return {
                "timestamp": None,
                "generation_throughput": 0.0,
                "scheduler_stats": {
                    "num_running_reqs": 0,
                    "num_waiting_reqs": 0,
                    "req_id_to_prompt_token_num": {},
                    "req_id_to_response_token_num": {},
                    "req_id_in_waiting": [],
                    "kv_cache_usage": 0.0,
                },
            }

    stubs = {
        "ray": {"remote": lambda cls: cls, "actor": SimpleNamespace(ActorHandle=object)},
        "psrl.trainer.ppo.utils": {"PSRL_Role": Role},
        "psrl.utils.logger": {
            "DualOutputHandler": logging.NullHandler,
            "FileOnlyHandler": logging.NullHandler,
            "EventType": SimpleNamespace(OTHER="OTHER"),
            "log_dual_events": lambda *args, **kwargs: nullcontext(),
        },
        "psrl.workers.gen.utils": {
            "RolloutInstanceId": tuple[str, int],
            "DEFAULT_MAX_CONNECTIONS": 10000,
            "DEFAULT_TIMEOUT": 3600,
        },
        "psrl.workers.gen.stats_collector": {
            "EngineStats": EngineStats,
        },
    }
    for name, attrs in stubs.items():
        stub = ModuleType(name)
        stub.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, stub)
    paths = {
        "command": ("psrl.utils.server.command", "psrl/utils/server/command.py"),
        "adapter": ("psrl.workers.gen.smg_adapter", "psrl/workers/gen/smg_adapter.py"),
        "topology": ("psrl.workers.elastic.cluster_topology", "psrl/workers/elastic/cluster_topology.py"),
        "policy": ("psrl.workers.elastic.scaling_policy.base", "psrl/workers/elastic/scaling_policy/base.py"),
        "executor": ("elastic_executor_runtime_test", "psrl/workers/elastic/executor.py"),
        "model_state": (
            "psrl.workers.global_coordination.model_state",
            "psrl/workers/global_coordination/model_state.py",
        ),
        "lifecycle": ("psrl.workers.global_coordination.lifecycle", "psrl/workers/global_coordination/lifecycle.py"),
        "commands": ("psrl.workers.global_coordination.commands", "psrl/workers/global_coordination/commands.py"),
    }
    loaded = {key: _load(monkeypatch, name, path) for key, (name, path) in paths.items()}
    return SimpleNamespace(role=Role, **loaded)


def _coordinator(modules):
    return SimpleNamespace()


def _model_services(modules):
    model = modules.model_state.ModelState(("rollout", ""), "http://rollout", True, "", None)
    gateway = SimpleNamespace(
        pause_replicas=AsyncMock(),
        eligibility=AsyncMock(),
        publish_versions=AsyncMock(),
    )
    lifecycle = modules.lifecycle.ReplicaLifecycle(gateway)
    commands = modules.commands.CommandService(SimpleNamespace(), gateway, lifecycle)
    return model, gateway, commands


def _executor(modules, coord):
    executor = object.__new__(modules.executor.ElasticExecutor)
    role = modules.role.Rollout
    instance_id = ("replica-1", 0)
    status = modules.topology.InstanceStatus.AWAKEN
    executor.roles = [(role, "model")]
    executor.coordinators = {role: {"model": coord}}
    executor.instances_status_flags = {role: {"model": {instance_id: status}}}
    executor.instances_engine_stats = {
        role: {"model": {instance_id: {"elastic_model_version": 3}}},
    }
    executor.role_model_versions = {(role, "model"): 3}
    executor.topology = modules.topology.ClusterTopology()
    executor.topology.register(role, "model", instance_id, status=status)
    executor.scaling_policy = SimpleNamespace(min_awake_per_role=0, monitor_interval_ms=1)
    executor._coordinator_command_timeout_s = 0.001
    executor._coordinator_sync_timeout_s = 0.1
    executor._decision_execution_in_progress = False
    executor._runtime_failure = None
    executor.agent_loop_manager = None
    executor.scale_up_task_queue = asyncio.Queue()
    executor.scale_down_task_queue = asyncio.Queue()
    executor._instance_state_epoch = 0
    executor._decision_settled = asyncio.Event()
    executor._decision_settled.set()
    executor._trainer_pool_transition_lock = asyncio.Lock()
    executor._trainer_pool_transition = False
    executor._trainer_pool_available = False
    executor._trainer_state = modules.executor.TrainerState.AWAKE
    executor._deferred_actor_ckpt_path = None
    executor._validation_active = False
    executor._validation_instance_ids = []
    executor.actor_wg = SimpleNamespace(world_size=1, execute_all_async=Mock(return_value=[]))
    executor._trainer_pool_epoch = 0
    executor._trainer_waiting_buffer_id = None
    executor.instance_pool_ids = {(role, "model", instance_id): "shared_rollout_pool"}
    executor._enable_monitor_instance_log = True
    executor._monitor_instance_log_interval_ms = 5000
    executor._last_monitor_instance_log_ms = 0.0
    executor.router_backlog_by_role = {}
    executor.config = SimpleNamespace(
        psrl=SimpleNamespace(
            rollout_coordination=SimpleNamespace(partial_rollout=SimpleNamespace(enable=True)),
            deployment=SimpleNamespace(mode="elastic_rl"),
        ),
    )
    executor._post_scale_up_abort_waiting_ratio = 1.0
    return executor


def _trainer_pool_executor(modules):
    executor = _executor(modules, _coordinator(modules))
    events = []

    async def execute(method):
        events.append(method)

    executor.actor_wg.execute_all_async.side_effect = lambda method, *args, **kwargs: [execute(method)]
    executor.ps_manager_handle = SimpleNamespace()
    executor._broadcast_trainer_client_infos = AsyncMock(side_effect=lambda: events.append("broadcast"))
    executor._reclaim_trainer_pool = AsyncMock(side_effect=lambda: events.append("reclaim"))
    return executor, events
