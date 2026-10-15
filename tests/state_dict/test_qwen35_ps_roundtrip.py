import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from psrl.utils.converter.param_sync import (
    DTypeCastSync,
    ParamSyncPlan,
    ZeroCenteredGammaSync,
)
from psrl.utils.nixl.fingerprint import precision_sensitive_parameter_stats

pytestmark = pytest.mark.cpu_test


def _load_workspace_module(module_name: str, relative_path: str):
    path = Path(__file__).parents[2] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


vLLMWorkerExtension = _load_workspace_module(
    "_qwen35_test_vllm_extension",
    "psrl/workers/gen/vllm_extension.py",
).vLLMWorkerExtension
PSRL_BaseTrainWorker = _load_workspace_module(
    "_qwen35_test_base_train_worker",
    "psrl/workers/train/base_train_worker.py",
).PSRL_BaseTrainWorker

GAMMA_KEY = "model.language_model.layers.0.linear_attn.norm.weight"
A_LOG_KEY = "model.language_model.layers.0.linear_attn.A_log"


def _make_train_worker(state_dict, sync_plan):
    worker = object.__new__(PSRL_BaseTrainWorker)
    worker.worker_rank = 0
    worker.pull_times = 0
    worker.psrl_config = OmegaConf.create({"ps_mode": "nixl_cpu"})
    worker.train_interface = SimpleNamespace(ps_manager_handle=None)
    worker._cached_ps_nixl_agent_names = ["ps_agent"]
    worker._cached_ps_nixl_train_storage_client_names = ["ps_client"]
    worker.unified_state_dict = state_dict
    worker.param_sync_plan = sync_plan
    return worker


class _MemoryPullClient:
    def __init__(self, ps_state, target_state):
        self.ps_state = ps_state
        self.target_state = target_state

    def client_read(self, _agent, _client, key, _tag):
        self.target_state[key].copy_(self.ps_state[key].to(self.target_state[key].dtype))
        return [(0,)]

    def wait(self, *_args, **_kwargs):
        return None

    def merge_and_finish_cached_xfer(self):
        return None

    def clear_intermediate_cached_data(self):
        return None


def test_fsdp_ps_push_pull_and_vllm_pull_preserve_qwen35_weights():
    ps_state = {
        GAMMA_KEY: torch.tensor([0.533203125, 0.8818359375, 1.16796875], dtype=torch.float32),
        A_LOG_KEY: torch.tensor([-4.71875, -2.375, 3.703125], dtype=torch.float32),
    }
    actor_state = {key: torch.empty_like(value) for key, value in ps_state.items()}
    actor = _make_train_worker(actor_state, ParamSyncPlan())

    def actor_pull(_agents, _clients):
        for key in actor_state:
            actor_state[key].copy_(ps_state[key])

    actor.nixl_pull_model_core = actor_pull
    actor.nixl_pull_model()
    for key in ps_state:
        torch.testing.assert_close(actor_state[key], ps_state[key], rtol=0, atol=0)

    actor_state[GAMMA_KEY].add_(0.03125)
    actor_state[A_LOG_KEY].add_(0.0078125)
    expected_after_train = {key: value.clone() for key, value in actor_state.items()}

    def actor_push():
        for key in actor_state:
            ps_state[key].copy_(actor_state[key])

    actor.nixl_push_model = actor_push
    actor.wait_for_nixl_push_completion = lambda: True
    actor.push_model()
    for key in ps_state:
        torch.testing.assert_close(ps_state[key], expected_after_train[key], rtol=0, atol=0)
        torch.testing.assert_close(actor_state[key], expected_after_train[key], rtol=0, atol=0)

    rollout_state = {
        GAMMA_KEY: torch.empty(ps_state[GAMMA_KEY].shape, dtype=torch.bfloat16),
        A_LOG_KEY: torch.empty_like(ps_state[A_LOG_KEY]),
    }
    rollout = object.__new__(vLLMWorkerExtension)
    rollout.unified_state_dict = rollout_state
    rollout.param_sync_plan = ParamSyncPlan()
    rollout.nixl_storage_client = _MemoryPullClient(ps_state, rollout_state)
    rollout.cuda_synchronize = lambda: None
    rollout.get_instance_local_rank = lambda: 0
    rollout.get_instance_local_tp_rank = lambda: 0
    rollout.nixl_pull_model_core(["ps_agent"], ["ps_client"])

    torch.testing.assert_close(
        rollout_state[GAMMA_KEY],
        ps_state[GAMMA_KEY].to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(rollout_state[A_LOG_KEY], ps_state[A_LOG_KEY], rtol=0, atol=0)

    actor_stats = precision_sensitive_parameter_stats(actor_state)
    rollout_stats = precision_sensitive_parameter_stats(rollout_state)
    assert actor_stats["linear_attn.norm.weight"]["mean"] == pytest.approx(
        expected_after_train[GAMMA_KEY].mean().item()
    )
    assert rollout_stats["linear_attn.norm.weight"]["mean"] == pytest.approx(
        expected_after_train[GAMMA_KEY].to(torch.bfloat16).float().mean().item()
    )


def test_megatron_zero_centered_gamma_is_converted_only_at_ps_boundary():
    standard_gamma = torch.tensor([0.533203125, 0.8818359375, 1.16796875])
    local_gamma = standard_gamma - 1
    state_dict = {GAMMA_KEY: local_gamma.clone()}
    ps_state = {GAMMA_KEY: torch.empty_like(standard_gamma)}
    plan = ParamSyncPlan([ZeroCenteredGammaSync(key=GAMMA_KEY)])
    actor = _make_train_worker(state_dict, plan)

    actor.nixl_push_model = lambda: ps_state[GAMMA_KEY].copy_(state_dict[GAMMA_KEY])
    actor.wait_for_nixl_push_completion = lambda: True
    actor.push_model()

    torch.testing.assert_close(ps_state[GAMMA_KEY], standard_gamma, rtol=0, atol=0)
    torch.testing.assert_close(state_dict[GAMMA_KEY], local_gamma, rtol=0, atol=0)

    updated_standard_gamma = standard_gamma + 0.0625
    ps_state[GAMMA_KEY].copy_(updated_standard_gamma)
    actor.nixl_pull_model_core = lambda _agents, _clients: state_dict[GAMMA_KEY].copy_(ps_state[GAMMA_KEY])
    actor.nixl_pull_model()

    torch.testing.assert_close(state_dict[GAMMA_KEY], updated_standard_gamma - 1, rtol=0, atol=0)


def _make_megatron_converter(sync_plan=None, fp32_patterns=("A_log",)):
    from psrl.utils.converter.megatron_converter import MegatronConverter

    converter = object.__new__(MegatronConverter)
    converter.sync_plan = ParamSyncPlan() if sync_plan is None else sync_plan
    converter.parameter_mapping = SimpleNamespace(get_external_fp32_param_patterns=lambda: fp32_patterns)
    return converter


def test_megatron_converter_registers_qwen35_sync_actions():
    source_a_log = torch.tensor([-2.5, 1.25], dtype=torch.bfloat16)
    state_dict = {
        GAMMA_KEY: torch.tensor([-0.4, -0.1, 0.2]),
        A_LOG_KEY: source_a_log,
    }
    converter = _make_megatron_converter()
    converter._register_param_sync_actions(state_dict)

    assert [type(action) for action in converter.sync_plan.actions] == [
        ZeroCenteredGammaSync,
        DTypeCastSync,
    ]
    assert state_dict[A_LOG_KEY].dtype == torch.float32


def test_megatron_converter_does_not_double_register_zero_centered_gamma():
    converter = _make_megatron_converter(
        sync_plan=ParamSyncPlan([ZeroCenteredGammaSync(key=GAMMA_KEY)]),
        fp32_patterns=(),
    )
    converter._register_param_sync_actions({GAMMA_KEY: torch.tensor([-0.4])})
    assert [type(action) for action in converter.sync_plan.actions] == [ZeroCenteredGammaSync]


def test_fsdp_first_pull_rejects_accidental_zero_centered_gamma():
    state_dict = {GAMMA_KEY: torch.empty(3)}
    actor = _make_train_worker(state_dict, ParamSyncPlan())

    def corrupted_pull(_agents, _clients):
        actor.pull_times = 1
        state_dict[GAMMA_KEY].copy_(torch.tensor([-0.47, -0.12, 0.17]))

    actor.nixl_pull_model_core = corrupted_pull
    with pytest.raises(RuntimeError, match="must not apply Megatron zero-centered-gamma"):
        actor.nixl_pull_model()
