"""
Validate PSRL rollout backends for policy, validation and reward inference.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
from psrl.workers.config import RolloutConfig
from verl.utils.config import omega_conf_to_dataclass

pytestmark = pytest.mark.cpu_test

_ROLLOUT_CONFIG_PATHS = (
    "gen_actor_rollout_ref.rollout",
    "train_actor_rollout_ref.rollout",
    "reward.managers.gen_qwen3_8b.rollout",
    "reward.managers.gen_skywork.rollout",
)


def test_rollout_defaults_to_vllm():
    assert RolloutConfig().name == "vllm", "PSRL rollout must default to vLLM."


@pytest.mark.parametrize("name", ["transformers", "transformer", "hf", "sglang", "trtllm"])
def test_unsupported_backend_fails_explicitly(name):
    with pytest.raises(ValueError, match="only the vLLM rollout backend"):
        RolloutConfig(name=name)


@pytest.mark.parametrize("config_path", _ROLLOUT_CONFIG_PATHS)
@pytest.mark.parametrize("backend", ["vllm", "transformers"])
def test_composed_rollout_backend_validation(config_path, backend):
    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer", overrides=[f"{config_path}.name={backend}"])
    rollout = OmegaConf.select(config, config_path)
    if backend == "vllm":
        assert isinstance(omega_conf_to_dataclass(rollout), RolloutConfig), (
            "Every inference role must use the PSRL rollout configuration."
        )
    else:
        with pytest.raises(InstantiationException, match="only the vLLM rollout backend") as error:
            omega_conf_to_dataclass(rollout)
        assert isinstance(error.value.__cause__, ValueError), "Unsupported backends must fail config validation."


def test_vllm_retains_parallelism_validation():
    with pytest.raises(AssertionError, match="expert_parallel_size"):
        RolloutConfig(expert_parallel_size=2, tensor_model_parallel_size=1)
