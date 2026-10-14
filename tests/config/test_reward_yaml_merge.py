# psrl/tests/config/test_reward_yaml_merge.py
import pytest

pytestmark = pytest.mark.cpu_test


def test_reward_yaml_merges_psrl_fields():
    """Verify that ppo_trainer config composes PSRL reward fields correctly."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_trainer")

    # PSRL reward fields (from psrl's reward/reward.yaml)
    assert "managers" in cfg.reward
    assert isinstance(dict(cfg.reward.managers), dict)
    assert "active_managers" in cfg.reward
    assert isinstance(list(cfg.reward.active_managers), list)
    assert "naive" in cfg.reward.active_managers
    assert "naive" in cfg.reward.managers
    assert cfg.reward.launch_reward_fn_async is True

    naive_cfg = cfg.reward.managers.naive
    assert naive_cfg.reward_loop_type == "naive"
    assert naive_cfg.reward_manager.name == "naive"
    assert naive_cfg.reward_manager.source == "register"


def test_gen_reward_model_uses_hf_model_schema():
    """Gen reward `model` is a HuggingFace config with `_target_`."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward.managers.gen_qwen3_8b.model.path=/models/rm",
                "reward.managers.gen_qwen3_8b.model.use_shm=False",
            ],
        )

    model = cfg.reward.managers.gen_qwen3_8b.model
    train_model = cfg.train_actor_rollout_ref.model
    assert model._target_ == "verl.workers.config.HFModelConfig"
    assert model._target_ == train_model._target_
    assert model.path == "/models/rm"
    assert model.use_shm is False
    assert "override_config" in model
    assert "lora" in model

    skywork = cfg.reward.managers.gen_skywork.model
    assert skywork._target_ == "verl.workers.config.HFModelConfig"
    assert "input_tokenizer" not in skywork


_SERVER_ADAPTER_ROLLOUT_KEYS = (
    "layered_summon",
    "prometheus",
    "mtp",
    "scheduling_policy",
    "enable_rollout_routing_replay",
    "reward_kwargs",
    "engine_kwargs",
    "val_kwargs",
    "checkpoint_engine",
    "skip_tokenizer_init",
    "load_format",
    "disable_attn",
)


def test_gen_reward_rollout_uses_psrl_rollout_schema():
    """Gen reward `rollout` is the same schema `ServerAdapter` reads for policy rollout."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward.managers.gen_qwen3_8b.rollout.layered_summon=False",
            ],
        )

    gen_rollout = cfg.gen_actor_rollout_ref.rollout
    rm_rollout = cfg.reward.managers.gen_qwen3_8b.rollout
    skywork_rollout = cfg.reward.managers.gen_skywork.rollout

    missing = set(gen_rollout.keys()) - set(rm_rollout.keys())
    assert not missing, f"Gen reward rollout is missing policy rollout keys {sorted(missing)}."
    assert rm_rollout._target_ == gen_rollout._target_
    assert rm_rollout._target_ == "psrl.workers.config.RolloutConfig"
    for key in _SERVER_ADAPTER_ROLLOUT_KEYS:
        assert key in rm_rollout, f"Gen reward rollout is missing {key}."
        assert key in skywork_rollout, f"Skywork reward rollout is missing {key}."

    assert rm_rollout.load_format == "auto"
    assert rm_rollout.disable_kv_cache is True
    assert rm_rollout.skip_tokenizer_init is False
    assert rm_rollout.n == 1
    assert rm_rollout.layered_summon is False
    assert "vllm" in rm_rollout.engine_kwargs

    assert skywork_rollout.runner == "pooling"
    assert skywork_rollout.task == "classify"
    assert skywork_rollout.load_format == "auto"


def test_mixed_gen_reward_selects_default_function_without_skywork():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward.active_managers=[naive,dapo,gen_qwen3_8b]",
                "reward.managers.gen_qwen3_8b.reward_model_name=GLM-Z1-9B-0414",
            ],
        )

    active = [cfg.reward.managers[name] for name in cfg.reward.active_managers]
    generative = [manager for manager in active if manager.reward_loop_type == "gen"]
    assert len(generative) == 1
    assert generative[0].reward_model_name == "GLM-Z1-9B-0414"
    assert [function.name for function in generative[0].reward_fn] == ["default"]
    assert cfg.reward.managers.gen_skywork.reward_manager.name == "gen"
    assert cfg.reward.managers.gen_skywork.reward_fn[0].name == "skywork"
