"""CPU tests for RewardModelManager interface."""

import inspect

import pytest
from psrl.workers.reward.reward_model.manager import RewardModelManager

pytestmark = pytest.mark.cpu_test


def test_manager_requires_gateway_url():
    """RewardModelManager constructor must accept gateway_url parameter."""
    sig = inspect.signature(RewardModelManager.__init__)
    assert "gateway_url" in sig.parameters, "RewardModelManager.__init__ must have a 'gateway_url' parameter"
    assert "status_queues" not in sig.parameters, "RewardModelManager.__init__ must NOT have 'status_queues' (old API)"
