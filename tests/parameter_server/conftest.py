# tests/parameter_server/conftest.py
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def ps_config():
    return OmegaConf.create(
        {
            "staleness": 2,
            "staleness_buffer_entries": 8,
        }
    )


@pytest.fixture
def tracker_config(tmp_path):
    """Full config for RequestStatusTracker.__init__. tmp_path is pytest's built-in temp dir fixture.

    Note: logging_path is used as a directory by DualOutputHandler (not a file path).
    The actual log file will be created at logging_path/RequestStatusTracker.log.
    """
    return OmegaConf.create(
        {
            "rollout_n": 4,
            "val_rollout_n": 1,
            "rollout_coordination": {
                "redundant_rollout": {
                    "enable": False,
                    "redundant_rollout_n": 4,
                    "alg_rollout_n": 4,
                },
            },
            "logging_path": str(tmp_path / "psrl_test.log"),
        }
    )


@pytest.fixture
def mock_model_store():
    store = MagicMock()
    store.get_version.return_value = 5
    store.get_weights.return_value = {"layer.weight": MagicMock()}
    return store


@pytest.fixture
def mock_ps_manager(ps_config, mock_model_store):
    manager = MagicMock()
    manager.config = ps_config
    manager.model_store = mock_model_store
    manager.can_reserve_request = MagicMock(return_value=True)
    manager.reserve_request = MagicMock(return_value={"version": 5})
    manager.complete_request = MagicMock()
    return manager


@pytest.fixture
def tracker(tracker_config):
    """A real RequestStatusTracker with no live Ray dependencies.

    Uses yield + teardown to remove the log handler added during __init__,
    preventing handler accumulation across the test session.
    """
    from psrl.utils.logger import get_ps_logger
    from psrl.workers.ps.request_status_tracker import RequestStatusTracker

    psrl_logger = get_ps_logger()
    handler_count_before = len(psrl_logger.handlers)

    t = RequestStatusTracker(psrl_config=tracker_config)
    t.rollout_coordinator = None
    t.reward_manager = None
    yield t

    # Remove any handlers added during __init__ to avoid log pollution
    while len(psrl_logger.handlers) > handler_count_before:
        h = psrl_logger.handlers[-1]
        h.close()
        psrl_logger.removeHandler(h)
