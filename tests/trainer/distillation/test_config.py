"""Validate explicit OPD reward service and dataset selections."""

import copy

import pytest
from omegaconf import OmegaConf
from psrl.utils.config.validation import validate_opd_config
from psrl.workers.config import DistillationLossConfig

pytestmark = pytest.mark.cpu_test


def config():
    teacher = {
        "reward_loop_type": "opd",
        "reward_manager": {"name": "opd"},
        "reward_model_name": "math",
        "reward_fn": [{"name": "default"}],
        "rollout": {"name": "vllm"},
    }
    return OmegaConf.create(
        {
            "distillation": {"enabled": True},
            "reward": {"active_managers": ["math"], "managers": {"math": teacher}},
            "gen_actor_rollout_ref": {"rollout": {"agent": {"traj_reward_mode": "traj"}}},
            "data": {
                "train_datas": [
                    {
                        "reward_model_dicts": [
                            {
                                "reward_loop_type": "opd",
                                "reward_fn": "default",
                                "reward_model_name": "math",
                            }
                        ]
                    }
                ]
            },
        }
    )


def test_multi_teacher_dataset_routes():
    cfg = config()
    teacher = copy.deepcopy(cfg.reward.managers.math)
    teacher.reward_model_name = "code"
    cfg.reward.managers.code = teacher
    cfg.reward.active_managers.append("code")
    cfg.data.train_datas.append(
        {
            "reward_model_dicts": [
                {
                    "reward_loop_type": "opd",
                    "reward_fn": "default",
                    "reward_model_name": "code",
                    "teacher_weight": 2.0,
                }
            ]
        }
    )
    validate_opd_config(cfg)


@pytest.mark.parametrize("damage", ["no_service", "duplicate_service", "missing", "duplicate", "unknown", "step"])
def test_invalid_teacher_config(damage):
    cfg = config()
    if damage == "no_service":
        cfg.reward.active_managers = []
    elif damage == "duplicate_service":
        cfg.reward.active_managers.append("math")
    elif damage == "missing":
        cfg.data.train_datas[0].reward_model_dicts = []
    elif damage == "duplicate":
        cfg.data.train_datas[0].reward_model_dicts.append(cfg.data.train_datas[0].reward_model_dicts[0])
    elif damage == "unknown":
        cfg.data.train_datas[0].reward_model_dicts[0].reward_model_name = "unknown"
    else:
        cfg.gen_actor_rollout_ref.rollout.agent.traj_reward_mode = "step"
    with pytest.raises(ValueError):
        validate_opd_config(cfg)


@pytest.mark.parametrize("topk", [0, -1, 1.5, True, None])
def test_invalid_topk(topk):
    with pytest.raises(ValueError, match="topk"):
        DistillationLossConfig(loss_mode="forward_kl_topk", use_policy_gradient=False, topk=topk)
