from types import SimpleNamespace

import pytest
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.elastic.cost_model import (
    compute_throughput,
    estimate_instance_throughput,
    resolve_cost_model_params,
)

pytestmark = pytest.mark.cpu_test


def test_resolve_cost_model_params_reads_tp_pp_cost_model_json():
    cost_path = "psrl/trainer/config/cost_model"
    rollout_params = resolve_cost_model_params(
        role_name=PSRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        cost_config={"cost_model_path": cost_path},
        tp_pp="TP1_PP1",
    )
    assert rollout_params.A == pytest.approx(3.71e-08)
    assert rollout_params.B == pytest.approx(0.0051426)
    assert rollout_params.C == pytest.approx(0.00008)
    assert rollout_params.D == pytest.approx(0.0005060241)

    rm_params = resolve_cost_model_params(
        role_name=PSRL_Role.RewardModel,
        model_name="Qwen3-8B",
        cost_config={"cost_model_path": cost_path},
        tp_pp="TP1_PP1",
    )
    assert rm_params.A == pytest.approx(8.57e-08)
    assert rm_params.B == pytest.approx(0.0057469)
    assert rm_params.C == pytest.approx(7.59e-05)
    assert rm_params.D == pytest.approx(0.0010808344)


def test_resolve_cost_model_params_reads_explicit_json_file():
    cost_path = "psrl/trainer/config/cost_model/qwen2.5_7b.json"
    params = resolve_cost_model_params(
        role_name=PSRL_Role.Rollout,
        model_name="SomeOtherModel",
        cost_config={"cost_model_path": cost_path},
        tp_pp="TP1_PP1",
    )
    assert params.A == pytest.approx(3.71e-08)
    assert params.B == pytest.approx(0.0051426)


def test_resolve_cost_model_params_auto_selects_single_tp_pp_bucket():
    cost_path = "psrl/trainer/config/cost_model"
    params = resolve_cost_model_params(
        role_name=PSRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        cost_config={"cost_model_path": cost_path},
    )
    assert params.B == pytest.approx(0.0051426)


def test_resolve_cost_model_params_uses_config_parallelism_for_tp_pp():
    cost_path = "psrl/trainer/config/cost_model"
    config = SimpleNamespace(
        gen_actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        ),
        reward=SimpleNamespace(
            active_managers=["default"],
            managers={
                "default": SimpleNamespace(
                    reward_model_name="Qwen3-8B",
                    rollout=SimpleNamespace(tensor_model_parallel_size=1, pipeline_model_parallel_size=1),
                )
            },
        ),
    )
    rollout_params = resolve_cost_model_params(
        role_name=PSRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        cost_config={"cost_model_path": cost_path},
        config=config,
    )
    rm_params = resolve_cost_model_params(
        role_name=PSRL_Role.RewardModel,
        model_name="Qwen3-8B",
        cost_config={"cost_model_path": cost_path},
        config=config,
    )
    assert rollout_params.B == pytest.approx(0.0051426)
    assert rm_params.B == pytest.approx(0.0057469)


def test_estimate_instance_throughput_uses_cost_model_directory():
    mu = estimate_instance_throughput(
        role_name=PSRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        running_queue_num=10,
        total_token_num=1000,
        cost_config={"cost_model_path": "psrl/trainer/config/cost_model"},
        tp_pp="TP1_PP1",
    )
    params = resolve_cost_model_params(
        role_name=PSRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        cost_config={"cost_model_path": "psrl/trainer/config/cost_model"},
        tp_pp="TP1_PP1",
    )
    assert mu == pytest.approx(compute_throughput(params, 1000, 10))


def test_estimate_instance_throughput_falls_back_without_cost_model():
    mu = estimate_instance_throughput(
        role_name=PSRL_Role.Rollout,
        model_name="Qwen2.5-7B",
        running_queue_num=10,
        total_token_num=1000,
        cost_config={},
        fallback=12.5,
    )
    assert mu == pytest.approx(12.5)
