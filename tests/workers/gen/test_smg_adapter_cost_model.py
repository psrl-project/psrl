from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from psrl.workers.gen.smg_adapter import build_reward_router_args, build_rollout_router_args

pytestmark = pytest.mark.cpu_test

COST_MODEL_DIR = Path("psrl/trainer/config/cost_model")


def _rollout_config(cost_model_path, model_path="Qwen/Qwen2.5-7B"):
    return SimpleNamespace(
        psrl=SimpleNamespace(
            rollout_coordination=SimpleNamespace(
                routing_strategy=SimpleNamespace(
                    method="throughput_optimal",
                    request_budget=512,
                    enable_group_sticky=True,
                    check_interval_in_ms=100,
                    request_sort_indicator="small_id",
                    enable_multi_priority_queue=False,
                    candidate_sort_indicator="version",
                    max_concurrent_seqs_per_instance=128,
                    max_num_waiting_reqs_after_preemption=50,
                    delta_throughput_threshold=0.3,
                    cost_model_path=cost_model_path,
                    kv_transfer=SimpleNamespace(enable=False, transfer_mode="async", transfer_timeout_ms=5000),
                ),
            ),
            rollout_gateway=SimpleNamespace(tito_debug=False, tito_gc_threshold=None),
            logging_path=None,
        ),
        data=SimpleNamespace(max_prompt_length=4096),
        rollout=SimpleNamespace(prompt_length=4096),
        gen_actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path=model_path)),
        train_actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path=model_path)),
    )


def _reward_config(cost_model_path, model_path="Qwen/Qwen3-8B", reward_model_name="Qwen3-8B"):
    return OmegaConf.create(
        {
            "reward_model_name": reward_model_name,
            "model": {"path": model_path},
            "rollout": {"max_model_len": 40960},
            "max_concurrent_requests_per_instance": 128,
            "routing_strategy": {
                "method": "itl",
                "cost_model_path": cost_model_path,
                "max_concurrent_seqs_per_instance": 512,
                "max_num_waiting_reqs_after_preemption": 3,
                "request_budget": 1024,
            },
        }
    )


def test_rollout_router_resolves_cost_model_directory():
    args = build_rollout_router_args(
        _rollout_config(str(COST_MODEL_DIR)),
        "127.0.0.1",
        30000,
        "127.0.0.1:8000",
    )
    assert args.cost_model_path == str(COST_MODEL_DIR / "qwen2.5_7b.json")


def test_rollout_router_keeps_explicit_cost_model_json():
    json_path = str(COST_MODEL_DIR / "qwen3_8b.json")
    args = build_rollout_router_args(
        _rollout_config(json_path, model_path="Qwen/Qwen2.5-7B"),
        "127.0.0.1",
        30000,
        "127.0.0.1:8000",
    )
    assert args.cost_model_path == json_path


def test_rollout_router_raises_when_directory_has_no_matching_model():
    with pytest.raises(ValueError, match="does not resolve"):
        build_rollout_router_args(
            _rollout_config(str(COST_MODEL_DIR), model_path="Unknown-Model-99B"),
            "127.0.0.1",
            30000,
            "127.0.0.1:8000",
        )


def test_reward_router_resolves_cost_model_directory():
    args = build_reward_router_args(
        OmegaConf.create({}),
        "127.0.0.1",
        8300,
        None,
        rm_config=_reward_config(str(COST_MODEL_DIR)),
    )
    assert args.cost_model_path == str(COST_MODEL_DIR / "qwen3_8b.json")


def test_reward_router_keeps_explicit_cost_model_json():
    json_path = str(COST_MODEL_DIR / "glm_z1_9b_0414.json")
    args = build_reward_router_args(
        OmegaConf.create({}),
        "127.0.0.1",
        8300,
        None,
        rm_config=_reward_config(json_path),
    )
    assert args.cost_model_path == json_path
