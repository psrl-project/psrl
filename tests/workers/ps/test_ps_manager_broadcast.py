"""
Test `PSManager` broadcast initialization integration.

Tests cover:
1. `TestCoordinateBroadcastInit` builds a plan internally and drives the broadcast loop.
2. `TestBindPsWorkerGroupBroadcast` registers PS agent names when broadcast initialization is enabled.
"""

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


def _make_psrl_config(enabled: bool = True, algorithm: str = "binary_tree") -> object:
    """Build a minimal psrl DictConfig for testing."""
    return OmegaConf.create(
        {
            "ps_mode": "nixl_cpu",
            "broadcast_init": {
                "enabled": enabled,
                "algorithm": algorithm,
            },
            "staleness": 1,
            "staleness_buffer_entries": 4,
            "redundant_rollout": {"enable": False},
            "logging_path": "/tmp/test_ps_manager.log",
        }
    )


def _make_ps_manager(psrl_config=None):
    """
    Construct a PSManager-like object with only the attributes needed by the methods
    under test, bypassing __init__ entirely.
    """
    from psrl.workers.ps.ps_manager import PSManager

    # Instantiate without calling __init__ to avoid Ray / heavy dependencies.
    manager = object.__new__(PSManager)

    # Attributes referenced by _coordinate_broadcast_init and bind_ps_worker_group.
    manager.psrl_config = psrl_config or _make_psrl_config()
    manager.ps_worker_group = None
    manager.ps_nixl_agent_names = None
    manager.ps_worker_handles_by_rank = []
    manager.nixl_meta_server = MagicMock()
    return manager


class TestCoordinateBroadcastInit:
    """Tests for PSManager._coordinate_broadcast_init (no-arg version)."""

    def test_single_worker_no_rounds(self):
        """Verify one worker receives the transfer call without broadcast rounds."""
        manager = _make_ps_manager()
        worker_mock = MagicMock()
        manager.ps_worker_handles_by_rank = [worker_mock]

        with patch("ray.get"):
            manager._coordinate_broadcast_init()

        # do_transfer_train_to_gen_after_broadcast must be called exactly once.
        worker_mock.do_transfer_train_to_gen_after_broadcast.remote.assert_called_once()
        # broadcast_send_to_children should NOT be called (no rounds).
        worker_mock.broadcast_send_to_children.remote.assert_not_called()

    def test_two_workers_one_round(self):
        """Verify rank zero sends to rank one in the only broadcast round."""
        manager = _make_ps_manager()
        worker0 = MagicMock()
        worker1 = MagicMock()
        manager.ps_worker_handles_by_rank = [worker0, worker1]

        with patch("ray.get"):
            manager._coordinate_broadcast_init()

        # Round 0: rank 0 is the sender.
        worker0.broadcast_send_to_children.remote.assert_called_once()
        worker1.broadcast_send_to_children.remote.assert_not_called()

        # After rounds: both workers receive the transfer call.
        worker0.do_transfer_train_to_gen_after_broadcast.remote.assert_called_once()
        worker1.do_transfer_train_to_gen_after_broadcast.remote.assert_called_once()

    def test_plan_built_from_config_algorithm(self):
        """_coordinate_broadcast_init uses psrl_config.broadcast_init.algorithm."""
        manager = _make_ps_manager(_make_psrl_config(algorithm="binary_tree"))
        workers = [MagicMock() for _ in range(4)]
        manager.ps_worker_handles_by_rank = workers

        with patch("psrl.workers.ps.ps_manager.build_broadcast_plan", wraps=None) as mock_build, patch("ray.get"):
            # Make the mock return a plan whose num_rounds() == 0 so we don't need real ray handles.
            fake_plan = MagicMock()
            fake_plan.num_rounds.return_value = 0
            fake_plan.senders_in_round.return_value = []
            mock_build.return_value = fake_plan

            manager._coordinate_broadcast_init()

        mock_build.assert_called_once_with(world_size=4, algorithm="binary_tree")

    def test_ray_get_called_per_round_and_final(self):
        """ray.get is called once per non-empty round (barrier) plus once for the final transfer."""
        manager = _make_ps_manager()
        worker0 = MagicMock()
        worker1 = MagicMock()
        worker2 = MagicMock()
        manager.ps_worker_handles_by_rank = [worker0, worker1, worker2]

        ray_get_calls = []
        with patch("ray.get", side_effect=lambda x: ray_get_calls.append(x)):
            manager._coordinate_broadcast_init()

        # The binary tree has senders only in round zero for three workers.
        # One barrier plus the final transfer produces two `ray.get` calls.
        assert len(ray_get_calls) == 2


class TestBindPsWorkerGroupBroadcast:
    """Tests that bind_ps_worker_group configures broadcast_init MetaServer when enabled."""

    def _make_worker_group_mock(self, world_size: int = 3):
        """Return a mock PSWorkerGroup with world_size workers."""
        wg = MagicMock()
        wg._workers = [MagicMock() for _ in range(world_size)]
        # execute_all_async returns a list of mock futures.
        wg.execute_all_async.return_value = [MagicMock() for _ in range(world_size)]
        return wg

    def test_broadcast_enabled_calls_enable_on_server(self):
        """When broadcast_init.enabled=True, enable_broadcast_init_on_server is called."""
        manager = _make_ps_manager(_make_psrl_config(enabled=True))
        wg = self._make_worker_group_mock(world_size=2)
        wg.execute_all_async.return_value = [MagicMock()]

        # Pre-populate ps_nixl_agent_names so enable_broadcast_init_on_server won't raise.
        manager.ps_nixl_agent_names = ["agent_0", "agent_1"]

        with patch.object(manager, "enable_broadcast_init_on_server") as mock_enable, patch("ray.get"):
            # Override execute_all_async so agent name futures resolve correctly.
            wg.execute_all_async.return_value = [MagicMock()]
            manager.bind_ps_worker_group(wg)

        mock_enable.assert_called_once()

    def test_broadcast_disabled_no_extra_calls(self):
        """When broadcast_init.enabled=False, enable_broadcast_init_on_server is NOT called."""
        manager = _make_ps_manager(_make_psrl_config(enabled=False))
        wg = self._make_worker_group_mock(world_size=2)
        wg.execute_all_async.return_value = [MagicMock()]

        with patch("ray.get"), patch.object(manager, "enable_broadcast_init_on_server") as mock_enable:
            manager.bind_ps_worker_group(wg)

        mock_enable.assert_not_called()
