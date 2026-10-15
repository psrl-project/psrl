"""
Check reward router initialization and optional strategy bookkeeping.
"""

import logging

import psrl.workers.reward.reward_model.router as router_module
import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "policy_config",
    [{}, {"exclusive_rebalance_migration_queue": False}, {"exclusive_rebalance_migration_queue": True}],
)
def test_router_reads_exclusive_migration_queue_from_throughput_config(monkeypatch, policy_config):
    monkeypatch.setattr(router_module, "DualOutputHandler", lambda *args: logging.NullHandler())
    config = OmegaConf.create(
        {
            "data": {},
            "psrl": {
                "logging_path": ".",
                "deployment": {
                    "elastic": {
                        "scaling_policy_variant": "rule-based",
                        "throughput_optimal": policy_config,
                    },
                },
            },
        }
    )
    cls = router_module.PSRL_RewardModelRouter.__ray_metadata__.modified_class
    existing_handlers = list(router_module.psrl_logger.handlers)
    try:
        router = cls(
            worker_handles=[object()],
            worker_groups=None,
            config=config,
            route_strategy_config={"method": "round_robin"},
        )
        assert router._exclusive_rebalance_migration_queue_enabled is policy_config.get(
            "exclusive_rebalance_migration_queue", False
        ), "The router must use its throughput policy configuration for exclusive migration."
        strategy = router.route_strategy
        assert strategy.instance_to_engine_status[0].replica_idx == 0, "The snapshot must identify its replica."
        assert strategy.instance_to_engine_status[0].data_parallel_rank == 0, "The snapshot must identify its DP rank."
        strategy.update_instance_loads({0: 1})
        strategy.push_request(None, 0)
        strategy.pop_request(None, 0)
        assert strategy.route(None) == 0, "Stateless strategies must support optional bookkeeping calls."
    finally:
        for handler in list(router_module.psrl_logger.handlers):
            if handler not in existing_handlers:
                router_module.psrl_logger.removeHandler(handler)
                handler.close()
