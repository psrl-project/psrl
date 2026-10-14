"""Unit tests for lightweight deployment-mode resolution and validation.

Verifies that each `psrl.deployment.mode` value resolves to the expected
`elastic` flag combination and that unset `mode` still infers from existing flags.
"""

import pytest
from omegaconf import OmegaConf
from psrl.utils.deployment_mode import (
    expand_ngpus_per_node,
    resolve_deployment_mode,
    validate_trainer_sleep_optimizer_offload,
)

pytestmark = pytest.mark.cpu_test


def _make_config(
    mode,
    elastic_enable=False,
    enable_trainer_pool=False,
    enable_policy=True,
    colocate=False,
    colocate_validate_and_train=None,
    optimizer_offload=False,
    strategy="fsdp2",
):
    if colocate_validate_and_train is None:
        colocate_validate_and_train = elastic_enable or mode in ("elastic_rl", "colocated", "rollout_rm_colocated")
    cfg = OmegaConf.create(
        {
            "psrl": {
                "ps_mode": "nixl_cpu",
                "colocate": colocate,
                "colocate_validate_and_train": colocate_validate_and_train,
                "fuse_rollout_with_validate": not colocate_validate_and_train,
                "deployment": {
                    "mode": mode,
                    "n_validate_instances": 1,
                    "trainer_pool_idle_rollout_instances": 0,
                    "trainer_pool_idle_rm_instances": 0,
                    "elastic": {
                        "enable": elastic_enable,
                        "shared_nnodes": 1,
                        "shared_ngpus_per_node": 8,
                        "enable_trainer_pool": enable_trainer_pool,
                        "enable_policy": enable_policy,
                    },
                },
            },
            "train_actor_rollout_ref": {
                "actor": {
                    "strategy": strategy,
                    "fsdp_config": {"optimizer_offload": optimizer_offload},
                    "megatron": {"optimizer_offload": optimizer_offload},
                }
            },
        }
    )
    return cfg


def _resolved(cfg):
    return resolve_deployment_mode(cfg), cfg.psrl.deployment.elastic


def test_disaggregated():
    cfg = _make_config("disaggregated")
    mode, erm = _resolved(cfg)
    assert mode == "disaggregated"
    assert erm.enable is False
    assert erm.enable_trainer_pool is False


def test_elastic_rl():
    cfg = _make_config("elastic_rl", elastic_enable=True, enable_policy=True)
    mode, erm = _resolved(cfg)
    assert mode == "elastic_rl"
    assert erm.enable is True


def test_trainer_pool_only():
    cfg = _make_config("trainer_pool_only")
    mode, erm = _resolved(cfg)
    assert mode == "trainer_pool_only"
    assert erm.enable is False
    assert erm.enable_trainer_pool is False
    assert erm.enable_policy is False


def test_colocated():
    cfg = _make_config("colocated")
    mode, erm = _resolved(cfg)
    assert mode == "colocated"
    assert erm.enable is True
    assert erm.enable_trainer_pool is True
    assert erm.enable_policy is False


def test_rollout_rm_colocated():
    cfg = _make_config("rollout_rm_colocated")
    mode, erm = _resolved(cfg)
    assert mode == "rollout_rm_colocated"
    assert erm.enable is True
    assert erm.enable_trainer_pool is False
    assert erm.enable_policy is False


def test_null_infers_elastic():
    # mode=null + elastic.enable=True infers elastic_rl.
    cfg = _make_config(None, elastic_enable=True)
    mode, erm = _resolved(cfg)
    assert mode == "elastic_rl"
    assert erm.enable is True


def test_null_infers_disaggregated():
    cfg = _make_config(None)
    mode, _ = _resolved(cfg)
    assert mode == "disaggregated"


@pytest.mark.parametrize("mode", ["elastic_rl", "colocated", "rollout_rm_colocated", None])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("colocate_validate_and_train", False, "colocate_validate_and_train=True"),
        ("fuse_rollout_with_validate", True, "fuse_rollout_with_validate=False"),
        ("deployment.n_validate_instances", 0, "n_validate_instances > 0"),
        ("deployment.n_validate_instances", -1, "n_validate_instances > 0"),
        ("ps_mode", "cpu", "ps_mode=nixl_cpu or nixl_gpu"),
        ("ps_mode", "cpu_ref", "ps_mode=nixl_cpu or nixl_gpu"),
    ],
)
def test_elastic_modes_require_dedicated_validation_on_trainer_gpus(mode, field, value, message):
    config = _make_config(mode, elastic_enable=True)
    OmegaConf.update(config.psrl, field, value)
    with pytest.raises(ValueError, match=message):
        resolve_deployment_mode(config)


def test_invalid_mode_raises():
    cfg = _make_config("bogus_mode")
    try:
        resolve_deployment_mode(cfg)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for invalid deployment mode")


def test_expand_ngpus_per_node_scalar():
    assert expand_ngpus_per_node(8, 2) == [8, 8]


def test_expand_ngpus_per_node_list_config():
    cfg = OmegaConf.create({"ngpus": [4, 8]})
    assert expand_ngpus_per_node(cfg.ngpus, 2) == [4, 8]


def test_expand_ngpus_per_node_rejects_wrong_length():
    cfg = OmegaConf.create({"ngpus": [4, 8]})
    try:
        expand_ngpus_per_node(cfg.ngpus, 3)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for mismatched per-node GPU list length")


def test_elastic_trainer_pool_requires_optimizer_offload():
    cfg = _make_config(
        "elastic_rl",
        elastic_enable=True,
        enable_trainer_pool=True,
        optimizer_offload=False,
    )
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match=r"fsdp_config\.optimizer_offload=True"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_elastic_trainer_pool_accepts_optimizer_offload():
    cfg = _make_config(
        "elastic_rl",
        elastic_enable=True,
        enable_trainer_pool=True,
        optimizer_offload=True,
    )
    mode, _ = _resolved(cfg)

    validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_trainer_pool_only_also_requires_optimizer_offload():
    cfg = _make_config("trainer_pool_only", optimizer_offload=False)
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match="trainer_pool_only"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_colocated_validation_requires_optimizer_offload_without_elastic_pool():
    cfg = _make_config(
        "disaggregated",
        colocate_validate_and_train=True,
        optimizer_offload=False,
    )
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match="colocate_validate_and_train=True"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


def test_megatron_sleep_uses_megatron_optimizer_offload_setting():
    cfg = _make_config(
        "elastic_rl",
        elastic_enable=True,
        enable_trainer_pool=True,
        optimizer_offload=False,
        strategy="megatron",
    )
    mode, _ = _resolved(cfg)

    with pytest.raises(ValueError, match=r"actor\.megatron\.optimizer_offload=True"):
        validate_trainer_sleep_optimizer_offload(cfg, mode)


if __name__ == "__main__":
    for fn in [
        test_disaggregated,
        test_elastic_rl,
        test_trainer_pool_only,
        test_colocated,
        test_rollout_rm_colocated,
        test_null_infers_elastic,
        test_null_infers_disaggregated,
        test_invalid_mode_raises,
        test_expand_ngpus_per_node_scalar,
        test_expand_ngpus_per_node_list_config,
        test_expand_ngpus_per_node_rejects_wrong_length,
    ]:
        fn()
        print(f"PASS: {fn.__name__}")
    print("All deployment-mode resolution tests passed.")
