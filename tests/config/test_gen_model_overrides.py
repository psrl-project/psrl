# tests/config/test_gen_model_overrides.py
import os

import pytest
from hydra import compose, initialize_config_dir

pytestmark = pytest.mark.cpu_test


_CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../psrl/trainer/config"))


def test_gen_model_shares_train_path():
    """Gen `path` is the train HuggingFace model path."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=["train_actor_rollout_ref.model.path=/models/shared"],
        )
    assert cfg.train_actor_rollout_ref.model.path == "/models/shared"
    assert cfg.gen_actor_rollout_ref.model.path == "/models/shared"


def test_gen_model_shares_train_override_config():
    """Nested train `override_config` is visible on gen."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=["+train_actor_rollout_ref.model.override_config.max_position_embeddings=32768"],
        )
    assert cfg.train_actor_rollout_ref.model.override_config.max_position_embeddings == 32768
    assert cfg.gen_actor_rollout_ref.model.override_config.max_position_embeddings == 32768
