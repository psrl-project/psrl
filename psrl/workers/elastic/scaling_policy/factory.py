"""
Construct the configured scaling policy.
"""

from importlib import import_module


def create_scaling_policy(config, policy_config):
    variant = str(policy_config.get("scaling_policy_variant", "throughput-optimal")).strip().lower()
    implementations = {
        "none": ("dummy", "DummyScalingPolicy"),
        "rule-based": ("rule_based", "RuleBasedScalingPolicy"),
        "throughput-optimal": ("thoughput_optimal", "ThroughputOptimalScalingPolicy"),
    }
    if variant not in implementations:
        raise ValueError(
            f"Unknown elastic scaling policy {variant!r}. Expected none, rule-based, or throughput-optimal."
        )
    module_name, class_name = implementations[variant]
    policy_cls = getattr(import_module(f"psrl.workers.elastic.scaling_policy.{module_name}"), class_name)
    return policy_cls(config=config, policy_config=policy_config)
