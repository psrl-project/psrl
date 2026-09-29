"""CPU tests for RewardModelCoordinator."""

import asyncio
from unittest.mock import MagicMock

import pytest
from psrl.workers.gen.rollout_coordination import RolloutCoordinator
from psrl.workers.reward.reward_model.coordinator import RewardModelCoordinator

pytestmark = pytest.mark.cpu_test


def _make_config():
    cfg = MagicMock()
    cfg.psrl.logging_path = "/tmp"
    return cfg


def _make_rm_config():
    rm_cfg = MagicMock()
    rm_cfg.reward_model_name = "TestRM"
    return rm_cfg


def test_coordinator_is_subclass():
    """RewardModelCoordinator must be a subclass of RolloutCoordinator."""
    assert issubclass(RewardModelCoordinator, RolloutCoordinator)


def test_sync_model_is_noop():
    """sync_model() must be a coroutine returning None (no PS interaction)."""
    coord = object.__new__(RewardModelCoordinator)
    coord.config = _make_config()
    coord.rm_config = _make_rm_config()
    coord.reward_model_name = "TestRM"
    assert asyncio.run(coord.sync_model()) is None


def test_update_model_version_is_noop():
    """update_model_version() must be a coroutine returning None."""
    coord = object.__new__(RewardModelCoordinator)
    coord.config = _make_config()
    coord.rm_config = _make_rm_config()
    coord.reward_model_name = "TestRM"
    assert asyncio.run(coord.update_model_version(model_version=5)) is None
