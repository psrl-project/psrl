from types import SimpleNamespace

import pytest
from psrl.workers.gen.smg_adapter import (
    CACHE_AWARE_METHODS,
    _cache_aware_cfg,
    build_rollout_router_args,
    build_worker_registration_payload,
    get_trajectory_id_strategy,
    is_cache_aware_method,
)


def _make_config(**routing_strategy_overrides):
    routing_strategy = {
        "method": "cache_aware",
        "request_budget": 512,
        "enable_group_sticky": True,
        "check_interval_in_ms": 100,
        "request_sort_indicator": "small_id",
        "enable_multi_priority_queue": False,
        "candidate_sort_indicator": "version",
        "max_concurrent_seqs_per_instance": 128,
        "max_num_waiting_reqs_after_preemption": 50,
        "delta_throughput_threshold": 0.3,
        "cost_model_path": None,
        "kv_transfer": {"enable": False, "transfer_mode": "async", "transfer_timeout_ms": 5000},
        **routing_strategy_overrides,
    }
    return SimpleNamespace(
        psrl=SimpleNamespace(
            rollout_coordination=SimpleNamespace(
                routing_strategy=SimpleNamespace(**routing_strategy),
            ),
            rollout_gateway=SimpleNamespace(tito_debug=False, tito_gc_threshold=None),
            logging_path=None,
        ),
        data=SimpleNamespace(max_prompt_length=4096),
        rollout=SimpleNamespace(prompt_length=4096),
    )


@pytest.mark.unit
def test_is_cache_aware_method():
    assert is_cache_aware_method("cache_aware")
    assert is_cache_aware_method("cache_aware_v1")
    assert not is_cache_aware_method("request_num_balance")
    assert CACHE_AWARE_METHODS == frozenset({"cache_aware", "cache_aware_v1"})


@pytest.mark.unit
def test_trajectory_id_strategy_defaults_validates_and_forwards():
    config = _make_config()
    assert get_trajectory_id_strategy(config) == "manual"

    config.psrl.rollout_gateway.trajectory_id_strategy = "auto"
    assert get_trajectory_id_strategy(config) == "auto"
    assert build_rollout_router_args(config, "127.0.0.1", 30000, "127.0.0.1:8000").trajectory_id_strategy == "auto"

    config.psrl.rollout_gateway.trajectory_id_strategy = "invalid"
    with pytest.raises(ValueError, match="trajectory_id_strategy"):
        get_trajectory_id_strategy(config)


@pytest.mark.unit
def test_multimodal_transport_defaults_to_safe_shm_auto_and_forwards_overrides():
    config = _make_config()
    router_args = build_rollout_router_args(config, "127.0.0.1", 30000, "127.0.0.1:8000")
    assert router_args.multimodal_tensor_transport == "auto"
    assert router_args.multimodal_shm_min_bytes == 64 * 1024

    config.psrl.rollout_gateway.multimodal_tensor_transport = "inline"
    config.psrl.rollout_gateway.multimodal_shm_min_bytes = 128 * 1024
    router_args = build_rollout_router_args(config, "127.0.0.1", 30000, "127.0.0.1:8000")
    assert router_args.multimodal_tensor_transport == "inline"
    assert router_args.multimodal_shm_min_bytes == 128 * 1024


@pytest.mark.unit
def test_cache_aware_cfg_reads_nested_block():
    config = _make_config(
        cache_aware_policy=SimpleNamespace(
            cache_threshold=0.5,
            gpu_overlap_weight=2.0,
            lmcache_overlap_weight=0.25,
            balance_abs_threshold=32,
            balance_rel_threshold=2.0,
            balance_token_usage_threshold=0.8,
            overload_token_usage_threshold=0.9,
            eviction_interval_secs=120,
            max_tree_size=1024,
            block_size=32,
        ),
    )
    assert _cache_aware_cfg(config, "cache_threshold") == 0.5
    assert _cache_aware_cfg(config, "gpu_overlap_weight") == 2.0
    assert _cache_aware_cfg(config, "block_size") == 32


@pytest.mark.unit
def test_build_rollout_router_args_cache_aware_nested():
    config = _make_config(
        method="cache_aware",
        cache_aware_policy=SimpleNamespace(
            cache_threshold=0.4,
            gpu_overlap_weight=1.2,
            lmcache_overlap_weight=0.6,
            balance_abs_threshold=48,
            balance_rel_threshold=1.8,
            balance_token_usage_threshold=0.7,
            overload_token_usage_threshold=0.85,
            eviction_interval_secs=90,
            max_tree_size=2048,
            block_size=8,
        ),
    )
    router_args = build_rollout_router_args(config, "127.0.0.1", 30000, "127.0.0.1:8000")

    assert router_args.policy == "cache_aware"
    assert router_args.cache_threshold == 0.4
    assert router_args.gpu_overlap_weight == 1.2
    assert router_args.lmcache_overlap_weight == 0.6
    assert router_args.balance_abs_threshold == 48
    assert router_args.balance_rel_threshold == 1.8
    assert router_args.balance_token_usage_threshold == 0.7
    assert router_args.overload_token_usage_threshold == 0.85
    assert router_args.eviction_interval_secs == 90
    assert router_args.max_tree_size == 2048
    assert router_args.block_size == 8


@pytest.mark.unit
def test_build_rollout_router_args_cache_aware_v1():
    config = _make_config(
        method="cache_aware_v1",
        cache_aware_policy=SimpleNamespace(
            cache_threshold=0.3,
            gpu_overlap_weight=1.0,
            lmcache_overlap_weight=0.5,
            balance_abs_threshold=64,
            balance_rel_threshold=1.5,
            balance_token_usage_threshold=1.0,
            overload_token_usage_threshold=1.0,
            eviction_interval_secs=60,
            max_tree_size=67108864,
            block_size=16,
        ),
    )
    router_args = build_rollout_router_args(config, "127.0.0.1", 30000, "127.0.0.1:8000")
    assert router_args.policy == "cache_aware_v1"


@pytest.mark.unit
def test_worker_registration_payload_includes_worker_id():
    payload = build_worker_registration_payload(
        url="grpc://127.0.0.1:30000",
        model_id="my-model",
        max_model_len=4096,
        dp_size=1,
        tp_size=1,
        pp_size=1,
        worker_id="2",
    )
    assert payload["id"] == "2"


@pytest.mark.unit
def test_worker_registration_payload_omits_worker_id_when_absent():
    payload = build_worker_registration_payload(
        url="grpc://127.0.0.1:30000",
        model_id="my-model",
        max_model_len=4096,
        dp_size=1,
        tp_size=1,
        pp_size=1,
    )
    assert "id" not in payload


@pytest.mark.unit
def test_policy_from_str_cache_aware_v1():
    from smg.router import policy_from_str
    from smg.smg_rs import PolicyType

    assert policy_from_str("cache_aware_v1") == PolicyType.CacheAwareV1
    assert policy_from_str("cache_aware") == PolicyType.CacheAware
