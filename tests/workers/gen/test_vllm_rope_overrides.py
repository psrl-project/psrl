from types import SimpleNamespace

import pytest
from psrl.workers.gen.rope import (
    _ensure_vllm_hf_rope_overrides,
    _normalize_rope_override_dict,
    effective_max_position_embeddings,
)

pytestmark = pytest.mark.cpu_test


def test_legacy_type_key_becomes_rope_type():
    normalized = _normalize_rope_override_dict(
        {
            "type": "yarn",
            "factor": 1.25,
            "original_max_position_embeddings": 32768,
        }
    )
    assert normalized["rope_type"] == "yarn"
    assert normalized["type"] == "yarn"
    assert normalized["factor"] == 1.25


def test_hf_overrides_copy_yarn_onto_rope_parameters():
    engine_kwargs = {
        "hf_overrides": {
            "rope_scaling": {
                "type": "yarn",
                "factor": 1.25,
                "original_max_position_embeddings": 32768,
            }
        }
    }
    model_hf_config = SimpleNamespace(rope_parameters={"rope_type": "default", "rope_theta": 10000.0})

    _ensure_vllm_hf_rope_overrides(engine_kwargs, model_hf_config)

    rope_parameters = engine_kwargs["hf_overrides"]["rope_parameters"]
    assert rope_parameters["rope_type"] == "yarn"
    assert rope_parameters["factor"] == 1.25
    assert rope_parameters["original_max_position_embeddings"] == 32768
    assert rope_parameters["rope_theta"] == 10000.0
    assert engine_kwargs["hf_overrides"]["rope_scaling"]["rope_type"] == "yarn"


def test_default_rope_parameters_are_not_treated_as_yarn():
    engine_kwargs = {"hf_overrides": {}}
    model_hf_config = SimpleNamespace(rope_parameters={"rope_type": "default", "rope_theta": 10000.0})

    _ensure_vllm_hf_rope_overrides(engine_kwargs, model_hf_config)

    assert engine_kwargs["hf_overrides"] == {}


def test_yarn_extends_max_position_embeddings():
    engine_kwargs = {
        "hf_overrides": {
            "rope_scaling": {
                "type": "yarn",
                "factor": 1.25,
                "original_max_position_embeddings": 32768,
            }
        }
    }
    model_hf_config = SimpleNamespace(
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    assert effective_max_position_embeddings(model_hf_config, engine_kwargs, 32768) == 40960


def test_default_rope_keeps_native_max_position():
    engine_kwargs = {"hf_overrides": {}}
    model_hf_config = SimpleNamespace(
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    assert effective_max_position_embeddings(model_hf_config, engine_kwargs, 32768) == 32768
