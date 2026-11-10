# tests/dataset/test_reward_model_dicts.py
import numpy as np
import pytest
from omegaconf import OmegaConf
from psrl.utils.dataset.utils import as_object_sample, normalize_reward_model_dicts

pytestmark = pytest.mark.cpu_test


def test_normalize_list_passthrough():
    raw = [{"reward_fn": "compute_score", "reward_loop_type": "dapo"}]
    assert normalize_reward_model_dicts(raw) == raw


def test_normalize_hydra_index_dict():
    raw = {"0": {"reward_fn": "compute_score", "reward_loop_type": "dapo"}}
    assert normalize_reward_model_dicts(raw) == [
        {"reward_fn": "compute_score", "reward_loop_type": "dapo"},
    ]


def test_normalize_omega_conf_index_dict():
    cfg = OmegaConf.create({"0": {"reward_fn": "compute_score", "reward_loop_type": "dapo"}})
    specs = normalize_reward_model_dicts(cfg)
    assert specs == [{"reward_fn": "compute_score", "reward_loop_type": "dapo"}]


def test_normalize_index_dict_sorts_numeric_keys():
    raw = {
        "1": {"reward_loop_type": "gen"},
        "0": {"reward_loop_type": "dapo"},
    }
    specs = normalize_reward_model_dicts(raw)
    assert [spec["reward_loop_type"] for spec in specs] == ["dapo", "gen"]


def test_normalize_single_spec_dict():
    raw = {"reward_fn": "compute_score", "reward_loop_type": "dapo"}
    assert normalize_reward_model_dicts(raw) == [raw]


def test_normalize_none_and_empty():
    assert normalize_reward_model_dicts(None) == []
    assert normalize_reward_model_dicts({}) == []
    assert normalize_reward_model_dicts([]) == []


def test_normalize_rejects_non_dict_entry():
    with pytest.raises(TypeError, match="entry must be a dict"):
        normalize_reward_model_dicts([{"reward_loop_type": "dapo"}, "dapo"])


def test_normalize_rejects_string():
    with pytest.raises(TypeError, match="list of dicts"):
        normalize_reward_model_dicts("dapo")
    with pytest.raises(TypeError, match="list of dicts"):
        normalize_reward_model_dicts(["dapo"])


def test_normalize_object_array_of_dicts():
    specs = [{"reward_fn": "compute_score", "reward_loop_type": "dapo"}]
    raw = np.array(specs, dtype=object)
    assert normalize_reward_model_dicts(raw) == specs


def test_normalize_numpy_wrapped_list():
    specs = [{"reward_fn": "compute_score", "reward_loop_type": "dapo"}]
    raw = np.array([specs], dtype=object)
    assert normalize_reward_model_dicts(raw) == specs
    assert normalize_reward_model_dicts(raw[0]) == specs


def test_as_object_sample_preserves_list():
    specs = [{"reward_fn": "compute_score", "reward_loop_type": "dapo"}]
    sample = as_object_sample(specs)
    assert sample.shape == (1,)
    assert sample[0] is specs
    assert normalize_reward_model_dicts(sample[0]) == specs
