from argparse import Namespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from psrl.tools.trainer_actor_ps_timing_probe import (
    _assert_fingerprints_match,
    _compose_config,
    _summarize_worker_wake,
)

pytestmark = pytest.mark.cpu_test


def test_weight_arena_is_enabled_by_default() -> None:
    config_path = Path(__file__).parents[2] / "psrl/trainer/config/psrl/psrl.yaml"
    config = OmegaConf.load(config_path)

    assert config.nixl.weight_arena.actor_enabled is True
    assert config.nixl.weight_arena.rollout_enabled is True
    assert config.nixl.weight_arena.actor_materialization == "direct"
    assert config.nixl.weight_arena.rollout_materialization == "direct"


def test_reward_node_shared_cache_is_enabled_by_default() -> None:
    config_path = Path(__file__).parents[2] / "psrl/trainer/config/psrl/psrl.yaml"
    config = OmegaConf.load(config_path)

    assert config.nixl.weight_arena.reward_cpu_cache_mode == "node_shared"
    assert config.nixl.weight_arena.reward_node_cache_dir == "/dev/shm/psrl-rm-weight-cache"
    assert config.nixl.weight_arena.reward_node_cache_backend == "auto"
    assert config.nixl.weight_arena.reward_node_cache_shm_reserve_gb == 256
    assert config.nixl.weight_arena.reward_node_cache_memfd_reserve_gb == 256
    assert config.nixl.weight_arena.max_chunk_gb == 4
    assert config.nixl.weight_arena.reward_node_cache_wait_timeout_s == 1800


def test_summarize_worker_wake_preserves_rank_and_node_scaling_data() -> None:
    results = [
        {
            "rank": 0,
            "node_id": "node-a",
            "resume_s": 0.2,
            "reregister_s": 4.0,
            "total_s": 4.2,
            "nixl_client": {
                "register_memory": 3.0,
                "build_shard_index": 0.1,
                "group_contig_descs": 0.1,
                "contig_get_xfer_descs": 0.2,
                "contig_serialize_descs": 0.3,
                "group_temp_descs": 0.1,
                "temp_get_xfer_descs": 0.1,
                "temp_serialize_descs": 0.1,
                "total": 4.0,
                "registered_regions": 2,
                "registered_bytes": 4096,
                "logical_slices": 1155,
                "contig_descs": 643,
                "temp_descs": 512,
            },
            "arena_virtual_addresses": (1000, 2000),
        },
        {
            "rank": 1,
            "node_id": "node-a",
            "resume_s": 0.4,
            "reregister_s": 6.0,
            "total_s": 6.4,
            "nixl_client": {
                "register_memory": 5.0,
                "build_shard_index": 0.1,
                "group_contig_descs": 0.1,
                "contig_get_xfer_descs": 0.2,
                "contig_serialize_descs": 0.3,
                "group_temp_descs": 0.1,
                "temp_get_xfer_descs": 0.1,
                "temp_serialize_descs": 0.1,
                "total": 6.0,
                "registered_regions": 2,
                "registered_bytes": 4096,
                "logical_slices": 1155,
                "contig_descs": 643,
                "temp_descs": 512,
            },
            "arena_virtual_addresses": (3000, 4000),
        },
    ]

    summary = _summarize_worker_wake(results)

    assert summary["resume_s"] == pytest.approx({"min": 0.2, "mean": 0.3, "max": 0.4})
    assert summary["reregister_s"] == {"min": 4.0, "mean": 5.0, "max": 6.0}
    assert summary["nixl_client"]["register_memory"] == {
        "min": 3.0,
        "mean": 4.0,
        "max": 5.0,
    }
    assert summary["counts_per_rank"] == {
        "registered_regions": [2],
        "registered_bytes": [4096],
        "logical_slices": [1155],
        "contig_descs": [643],
        "temp_descs": [512],
    }
    assert summary["arena_counts_per_rank"] == [2]
    assert summary["nodes"]["node-a"] == {
        "ranks": [0, 1],
        "max_reregister_s": 6.0,
    }


def test_compose_config_enables_weight_arena(tmp_path) -> None:
    model_path = tmp_path / "model"
    args = Namespace(
        output=tmp_path / "report.json",
        model_path=model_path,
        world_size=8,
        weight_arena=True,
        weight_arena_max_chunk_gb=2,
        overrides=[],
    )

    config = _compose_config(args, "127.0.0.1", 12345)

    assert config.psrl.nixl.weight_arena.actor_enabled is True
    assert config.psrl.nixl.weight_arena.max_chunk_gb == 2
    assert config.train_actor_rollout_ref.actor.fsdp_config.optimizer_offload is True


def test_compose_config_explicitly_disables_weight_arena_for_baseline(tmp_path) -> None:
    args = Namespace(
        output=tmp_path / "report.json",
        model_path=tmp_path / "model",
        world_size=8,
        weight_arena=False,
        weight_arena_max_chunk_gb=2,
        overrides=[],
    )

    config = _compose_config(args, "127.0.0.1", 12345)

    assert config.psrl.nixl.weight_arena.actor_enabled is False


def test_exact_fingerprint_comparison_reports_changed_tensor() -> None:
    baseline = [
        {
            "rank": 0,
            "digest": "baseline",
            "tensor_digests": {"layer.weight|(0,)": "a", "layer.bias|(0,)": "b"},
        }
    ]
    _assert_fingerprints_match(baseline, baseline, label="same")

    changed = [
        {
            "rank": 0,
            "digest": "changed",
            "tensor_digests": {"layer.weight|(0,)": "different", "layer.bias|(0,)": "b"},
        }
    ]
    with pytest.raises(RuntimeError, match=r"layer\.weight"):
        _assert_fingerprints_match(baseline, changed, label="cycle 1")
