from pathlib import Path

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


def test_rollout_log_probs_are_enabled_by_default():
    config_path = Path(__file__).resolve().parents[2] / "psrl/trainer/config/psrl/log_prob.yaml"
    config = OmegaConf.load(config_path)
    assert config.enable_rollout_engine_log_prob is True, "Rollout engines must return log probabilities."
