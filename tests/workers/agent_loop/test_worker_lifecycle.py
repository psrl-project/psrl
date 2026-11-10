"""
Test configured agent construction and fatal initialization failures.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import psrl.workers.agent_loop.worker as worker_module
import pytest
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
from psrl.workers.agent_loop.loops import utils as loop_utils
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.gen.utils import TokenOutput
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


def test_worker_initializes_with_teacher_reward_service(tmp_path, monkeypatch):
    config = OmegaConf.create(
        {
            "distillation": {"enabled": True},
            "data": {},
            "reward": {"reward_loop_worker": {}, "validation_reward_loop_worker": {}},
            "psrl": {
                "logging_path": str(tmp_path),
                "deployment": {"n_rollout_instances": 1},
                "colocate_validate_and_train": False,
                "rollout_gateway": {"server_max_concurrency": 4},
            },
            "gen_actor_rollout_ref": {
                "model": {},
                "rollout": {"agent": {"agent_loop_config_path": None, "default_agent_loop": "single_turn_agent"}},
            },
            "trainer": {"project_name": "test", "experiment_name": "opd"},
        }
    )
    model = SimpleNamespace(tokenizer=object(), processor=None)
    monkeypatch.setattr(worker_module, "omega_conf_to_dataclass", Mock(return_value=model))
    monkeypatch.setattr(worker_module, "spawn_actor_reaper", Mock(return_value=SimpleNamespace(pid=0)))
    monkeypatch.setattr(worker_module.atexit, "register", Mock())
    monkeypatch.setenv("PSRL_ACTOR_ID", "test")
    monkeypatch.setattr(worker_module.tq, "init", Mock())
    monkeypatch.setattr(worker_module, "get_dataset_class", Mock(return_value=object))
    monkeypatch.setattr(worker_module, "reward_producer_concurrency", Mock(return_value=1))
    monkeypatch.setattr(worker_module, "init_http_client", Mock())
    monkeypatch.setattr(worker_module, "init_http_io_thread", Mock())
    monkeypatch.setattr(worker_module, "get_agent_loop_config", Mock(return_value={}))
    monkeypatch.setattr(worker_module.RolloutTraceConfig, "init", Mock())
    handler = worker_module.logging.NullHandler()
    monkeypatch.setattr(worker_module, "DualOutputHandler", Mock(return_value=handler))
    cls = worker_module.PSRL_AgentLoopWorker.__ray_metadata__.modified_class
    worker = cls(config, None, "http://rollout", "http://session")
    assert worker.tokenizer is model.tokenizer, "OPD must reach normal agent worker initialization."
    assert not worker.pending_program_queue, "An initialized worker must be ready for rollout requests."
    worker_module.logging.getLogger("psrl").removeHandler(handler)
    worker_module.psrl_logger.removeHandler(handler)


@pytest.fixture
def worker(monkeypatch):
    cls = worker_module.PSRL_AgentLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.config = OmegaConf.create(
        {
            "data": {},
            "gen_actor_rollout_ref": {
                "rollout": {
                    "agent": {"retry_limit": 2, "raise_on_error": True, "traj_reward_mode": "traj"},
                    "multi_turn": {"enable": True},
                }
            },
        }
    )
    worker.agent_loop_configs = {}
    worker.rollout_gateway_url = "http://rollout"
    worker.session_router_url = "http://session"
    worker.reward_manager = None
    worker.inference_orchestrator = None
    worker.tokenizer = object()
    worker.processor = None
    worker.dataset_cls = object
    worker.agent_loop_manager = SimpleNamespace(
        fail_pipeline=SimpleNamespace(remote=AsyncMock()),
        notify_group_failed=SimpleNamespace(remote=AsyncMock()),
    )
    worker.ps_manager_handle = SimpleNamespace(
        abort_requests=SimpleNamespace(remote=AsyncMock()),
        update_request_status=SimpleNamespace(remote=AsyncMock(return_value=True)),
    )
    worker.postprocess_output = AsyncMock()
    monkeypatch.setattr(SessionAgentLoop, "__init__", lambda self, context: None)
    return worker


def _mini_swe_config():
    path = Path(__file__).resolve().parents[3] / "examples/mini_swe/config/simple_agent_config.yaml"
    return OmegaConf.load(path)[0]


def _batch(validate=False):
    return tu.get_tensordict({"uid": [7], "parent_id": [3], "validate": [validate]})


@pytest.mark.asyncio
async def test_mini_swe_yaml_survives_class_registration_and_repeated_episodes(worker, monkeypatch):
    configured = _mini_swe_config()
    expected = OmegaConf.to_container(configured)
    worker.agent_loop_configs[configured.name] = configured
    runtime_configs = []
    output = TokenOutput(prompt_ids=[1], response_ids=[2], response_mask=[1])

    async def run(loop, batch, raise_on_error):
        runtime_configs.append(loop.runtime_config)
        return output, loop_utils.TerminateReason.FINISHED

    monkeypatch.setattr(AgentLoopBase, "run_with_termination_handling", run)

    await worker._run_agent_loop("mini_swe_agent", _batch())
    from psrl.workers.agent_loop.loops.mini_swe_agent_loop_v1 import MiniSWEAgentLoopV1

    loop_utils.register("mini_swe_agent")(MiniSWEAgentLoopV1)
    await worker._run_agent_loop("mini_swe_agent", _batch())

    assert len(runtime_configs) == 2, "Both episodes must reach generation."
    for runtime in runtime_configs:
        assert runtime.agent.system_template == configured.agent.system_template, "The system template must survive."
        assert runtime.agent.problem_template == configured.agent.problem_template, (
            "The problem template must survive."
        )
        assert runtime.model.action_regex == configured.model.action_regex, "Model options must survive."
        assert runtime.sandbox_config.max_parallel_tasks_per_worker == 8, "Sandbox options must survive."
    assert OmegaConf.to_container(configured) == expected, "Construction must preserve all worker configuration."
    assert set(loop_utils.AGENT_LOOP_REGISTRY["mini_swe_agent"]) == {"_target_"}, "Class defaults must stay separate."
    assert worker.postprocess_output.await_count == 2, "Both episodes must publish their output."
    worker.agent_loop_manager.fail_pipeline.remote.assert_not_awaited()
    worker.agent_loop_manager.notify_group_failed.remote.assert_not_awaited()
    worker.ps_manager_handle.abort_requests.remote.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("failure_stage", ["lookup", "import", "constructor"])
async def test_initialization_failure_stops_pipeline_and_aborts_request(worker, monkeypatch, validate, failure_stage):
    configured = _mini_swe_config()
    agent_name = "unknown_agent" if failure_stage == "lookup" else "mini_swe_agent"
    if failure_stage == "import":
        configured._target_ = "missing_psrl_agent_module.AgentLoop"
        expected_error = InstantiationException
        message = "Error locating target"
    elif failure_stage == "constructor":
        del configured["agent"]
        expected_error = ValueError
        message = "agent.system_template and agent.problem_template must be provided"
    else:
        expected_error = ValueError
        message = "Unregistered agent loop"
    if failure_stage != "lookup":
        worker.agent_loop_configs[agent_name] = configured
    run = AsyncMock()
    monkeypatch.setattr(AgentLoopBase, "run_with_termination_handling", run)

    with pytest.raises(expected_error, match=message) as raised:
        await worker._run_agent_loop(agent_name, _batch(validate))

    worker.agent_loop_manager.fail_pipeline.remote.assert_awaited_once()
    diagnosis = worker.agent_loop_manager.fail_pipeline.remote.call_args.args[0]
    assert agent_name in diagnosis, "The fatal diagnosis must name the agent."
    assert "request_id=7" in diagnosis, "The fatal diagnosis must identify the request."
    assert "prompt_id=3" in diagnosis, "The fatal diagnosis must identify the prompt group."
    assert str(raised.value) in diagnosis, "The fatal diagnosis must preserve the original cause."
    worker.ps_manager_handle.abort_requests.remote.assert_awaited_once_with([7])
    worker.agent_loop_manager.notify_group_failed.remote.assert_not_awaited()
    worker.postprocess_output.assert_not_awaited()
    run.assert_not_awaited()


def test_worker_configs_are_isolated_from_other_workers_and_class_defaults(worker):
    registered = dict(loop_utils.AGENT_LOOP_REGISTRY)
    configured = _mini_swe_config()
    worker.agent_loop_configs[configured.name] = configured
    other = object.__new__(type(worker))
    other.agent_loop_configs = {configured.name: OmegaConf.create(OmegaConf.to_container(configured))}
    other.agent_loop_configs[configured.name].agent.system_template = "A different worker template."

    assert worker._get_agent_loop_config(configured.name).agent.system_template == configured.agent.system_template, (
        "Another worker must not replace this worker's options."
    )
    assert other._get_agent_loop_config(configured.name).agent.system_template == "A different worker template.", (
        "Each worker must resolve its own options."
    )
    assert loop_utils.AGENT_LOOP_REGISTRY == registered, "YAML lookup must not change class registration."


def test_worker_without_yaml_uses_registered_class_defaults(worker):
    resolved = worker._get_agent_loop_config("generate_only_agent")

    assert "GenerateAgentLoop" in resolved["_target_"], "Unconfigured agents must resolve registered classes."
    assert set(resolved) == {"_target_"}, "Class defaults must not contain another agent's YAML options."
