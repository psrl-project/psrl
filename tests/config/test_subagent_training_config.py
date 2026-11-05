"""Validation of the subagent training policy against batch aggregation."""

import pytest
from omegaconf import OmegaConf
from psrl.utils.config import validate_subagent_training_config


def _config(policy: dict):
    return OmegaConf.create({"psrl": {"agentic_rl": {"subagent_training": policy}}})


def test_main_only_is_valid_under_trajectory_aggregation():
    policy = validate_subagent_training_config(_config({"mode": "main_only"}), "trajectory")

    assert policy.mode == "main_only"


@pytest.mark.parametrize("mode", ["all", "selected"])
def test_multi_row_modes_require_request_aggregation(mode):
    mapping = {"mode": mode}
    if mode == "selected":
        mapping["train_agents"] = ["worker-*"]

    with pytest.raises(AssertionError, match="batch_agg_mode must be 'request'"):
        validate_subagent_training_config(_config(mapping), "trajectory")

    assert validate_subagent_training_config(_config(mapping), "request").mode == mode


def test_val_mode_also_requires_request_aggregation():
    with pytest.raises(AssertionError, match="batch_agg_mode must be 'request'"):
        validate_subagent_training_config(
            _config({"mode": "main_only", "val_mode": "all"}),
            "trajectory",
        )
