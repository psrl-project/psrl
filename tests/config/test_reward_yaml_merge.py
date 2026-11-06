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
    assert "reward_loops" in cfg.reward
    assert isinstance(dict(cfg.reward.reward_loops), dict)
    assert "active_loops" in cfg.reward
    assert isinstance(list(cfg.reward.active_loops), list)
    assert "naive" in cfg.reward.active_loops
    assert "naive" in cfg.reward.reward_loops
    assert cfg.reward.launch_reward_fn_async is True

    naive_cfg = cfg.reward.reward_loops.naive
    assert naive_cfg.reward_loop_type == "naive"
    assert naive_cfg.reward_loop.name == "naive"
    assert naive_cfg.reward_loop.source == "register"


def test_validation_reward_pool_can_be_configured_independently():
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward.validation_reward_loop_worker.num_workers=3",
                "reward.validation_reward_loop_worker.max_concurrency_per_worker=17",
                "reward.validation_reward_loop_worker.queue_size_per_worker=64",
            ],
        )

    validation = cfg.reward.validation_reward_loop_worker
    assert validation.num_workers == 3, "Validation worker count must support Hydra overrides."
    assert validation.max_concurrency_per_worker == 17, "Validation concurrency must support Hydra overrides."
    assert validation.queue_size_per_worker == 64, "Validation queue size must support Hydra overrides."
    assert cfg.reward.reward_loop_worker.num_workers == 8, "Training worker count must remain independently set."
    assert cfg.reward.reward_loop_worker.max_concurrency_per_worker == 256, "Training must retain its own capacity."


def test_gen_reward_model_uses_hf_model_schema():
    """Gen reward `model` is a HuggingFace config with `_target_`."""
    import os

    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward.reward_loops.gen_qwen3_8b.model.path=/models/rm",
                "reward.reward_loops.gen_qwen3_8b.model.use_shm=False",
            ],
        )

    model = cfg.reward.reward_loops.gen_qwen3_8b.model
    train_model = cfg.train_actor_rollout_ref.model
    assert model._target_ == "verl.workers.config.HFModelConfig"
    assert model._target_ == train_model._target_
    assert model.path == "/models/rm"
    assert model.use_shm is False
    assert "override_config" in model
    assert "lora" in model

    skywork = cfg.reward.reward_loops.gen_skywork.model
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
                "reward.reward_loops.gen_qwen3_8b.rollout.layered_summon=False",
            ],
        )

    gen_rollout = cfg.gen_actor_rollout_ref.rollout
    rm_rollout = cfg.reward.reward_loops.gen_qwen3_8b.rollout
    skywork_rollout = cfg.reward.reward_loops.gen_skywork.rollout

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
                "reward.active_loops=[naive,dapo,llm_as_a_judge]",
                "reward.reward_loops.llm_as_a_judge.reward_model_name=GLM-Z1-9B-0414",
                "reward.reward_loops.llm_as_a_judge.model.path=models/GLM-Z1-9B-0414",
            ],
        )

    active = [cfg.reward.reward_loops[name] for name in cfg.reward.active_loops]
    generative = [manager for manager in active if manager.reward_loop_type == "gen"]
    assert len(generative) == 1
    assert generative[0].reward_model_name == "GLM-Z1-9B-0414"
    assert [function.name for function in generative[0].reward_fn] == ["default"]
    assert cfg.reward.reward_loops.gen_skywork.reward_loop.name == "gen"
    assert cfg.reward.reward_loops.gen_skywork.reward_fn[0].name == "skywork"


@pytest.mark.parametrize("preset", ["gen_qwen3_8b", "gen_skywork", "llm_as_a_judge", "opd"])
def test_reward_preset_supports_independent_named_instances(preset):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from psrl.workers.config.reward_model import resolve_active_loops

    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                f"+reward/loops@reward.reward_loops.judge_a={preset}",
                f"+reward/loops@reward.reward_loops.judge_b={preset}",
                "reward.active_loops=[judge_a,judge_b]",
                "reward.reward_loops.judge_a.reward_model_name=judge_a",
                "reward.reward_loops.judge_b.reward_model_name=judge_b",
                "reward.reward_loops.judge_a.model.path=models/judge_a",
                "reward.reward_loops.judge_b.model.path=models/judge_b",
                "reward.reward_loops.judge_a.rollout.tensor_model_parallel_size=2",
            ],
        )

    judge_a, judge_b = resolve_active_loops(cfg.reward)
    assert judge_a.model.path == "models/judge_a", "The first instance must accept a model override."
    assert judge_b.model.path == "models/judge_b", "The second instance must keep its own model override."
    assert judge_a.rollout.tensor_model_parallel_size == 2, "New instances must inherit overridable rollout fields."
    assert judge_b.rollout.tensor_model_parallel_size == 1, "Instances must not share rollout mutations."
    original = cfg.reward.reward_loops[preset]
    assert OmegaConf.is_missing(original, "reward_model_name") or original.reward_model_name not in (
        "judge_a",
        "judge_b",
    ), "Instance overrides must not modify the preset."
    for instance in (judge_a, judge_b):
        assert instance.model._target_ == "verl.workers.config.HFModelConfig", (
            "Instances must inherit the model schema."
        )
        assert instance.rollout._target_ == "psrl.workers.config.RolloutConfig", (
            "Instances must inherit rollout schema."
        )
        assert instance.reward_fn == original.reward_fn, "Instances must inherit the preset's scoring functions."
        assert instance.rollout.runner == original.rollout.runner, "Instances must inherit the preset's model runner."


def test_extra_yaml_can_add_a_reward_instance(tmp_path):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    (experiment_dir / "extra_reward.yaml").write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - /reward/loops@reward.reward_loops.judge: llm_as_a_judge\n"
        "  - _self_\n"
        "reward:\n"
        "  active_loops: [naive, judge]\n"
        "  reward_loops:\n"
        "    judge:\n"
        "      reward_model_name: judge\n"
        "      model:\n"
        "        path: models/judge\n",
        encoding="utf-8",
    )
    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[f"hydra.searchpath=[{tmp_path.as_uri()}]", "+experiment=extra_reward"],
        )

    assert list(cfg.reward.active_loops) == ["naive", "judge"], "Extra YAML must control activation."
    judge = cfg.reward.reward_loops.judge
    assert judge.model.path == "models/judge", "Extra YAML must override the model after composing its schema."
    assert judge.model.use_shm == cfg.reward.reward_loops.llm_as_a_judge.model.use_shm, (
        "Extra YAML instances must inherit model defaults."
    )
    assert judge.rollout.load_format == "auto", "Extra YAML instances must inherit reward rollout defaults."
    assert OmegaConf.is_missing(cfg.reward.reward_loops.llm_as_a_judge, "reward_model_name"), (
        "Instance overrides must leave the generic preset unconfigured."
    )


@pytest.mark.parametrize("preset,loop_type", [("llm_as_a_judge", "gen"), ("opd", "opd")])
def test_generic_reward_loop_requires_explicit_model_without_qwen_preset(preset, loop_type):
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from psrl.utils.config.validation import validate_reward_loops

    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=["~reward/loops@reward.reward_loops.gen_qwen3_8b"],
        )

    loop = cfg.reward.reward_loops[preset]
    assert loop.reward_loop_type == loop_type, "The preset must select its own reward loop type."
    assert loop.reward_loop.name == loop_type, "The preset must select its registered implementation."
    assert OmegaConf.is_missing(loop, "reward_model_name"), "Generic reward model names must be explicit."
    assert loop.model.path is None, "Generic reward checkpoints must be explicit."
    assert loop.model._target_ == "verl.workers.config.HFModelConfig", "Model schema must be self contained."
    assert loop.rollout._target_ == "psrl.workers.config.RolloutConfig", "Rollout schema must be self contained."
    validate_reward_loops(cfg)
    cfg.reward.active_loops = [preset]
    with pytest.raises(ValueError, match="reward_model_name"):
        validate_reward_loops(cfg)
    loop.reward_model_name = "teacher"
    with pytest.raises(ValueError, match="model.path"):
        validate_reward_loops(cfg)
    loop.model.path = "models/teacher"
    validate_reward_loops(cfg)


def test_opd_preset_validates_teacher_routing_without_type_overrides():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from psrl.utils.config.validation import validate_opd_config, validate_reward_loops

    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "reward.active_loops=[naive,opd]",
                "reward.reward_loops.opd.reward_model_name=teacher",
                "reward.reward_loops.opd.model.path=models/teacher",
                "distillation.enabled=True",
                "gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj",
                "data.train_datas.0.reward_model_dicts=[{reward_loop_type:opd,reward_fn:default,"
                "reward_model_name:teacher,teacher_weight:1.0}]",
            ],
        )

    validate_reward_loops(cfg)
    validate_opd_config(cfg)
    teacher = cfg.reward.reward_loops.opd
    assert teacher.rollout.response_length == 1, "Teacher scoring must reserve one output token."
    assert teacher.rollout.max_model_len == teacher.rollout.prompt_length + 1, (
        "Teacher context must cover the prompt and one output token."
    )
