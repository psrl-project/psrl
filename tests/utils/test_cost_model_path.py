from pathlib import Path

import pytest
from psrl.utils.cost_model_path import (
    model_name_to_cost_model_stems,
    require_cost_model_json_path,
    resolve_cost_model_json_path,
)

pytestmark = pytest.mark.cpu_test


def test_qwen2_5_32b_uses_qwen_32b_cost_model():
    assert model_name_to_cost_model_stems("Qwen2.5-32B")[0] == "qwen_32b"
    assert model_name_to_cost_model_stems("Qwen/Qwen2.5-32B")[0] == "qwen_32b"

    cost_model_dir = Path("psrl/trainer/config/cost_model")
    resolved = resolve_cost_model_json_path(str(cost_model_dir), "Qwen2.5-32B")

    assert resolved == str(cost_model_dir / "qwen_32b.json")


def test_other_qwen2_5_sizes_keep_their_family():
    assert model_name_to_cost_model_stems("Qwen2.5-7B")[0] == "qwen2.5_7b"

    cost_model_dir = Path("psrl/trainer/config/cost_model")
    resolved = resolve_cost_model_json_path(str(cost_model_dir), "Qwen/Qwen2.5-72B")

    assert resolved == str(cost_model_dir / "qwen2.5_72b.json")


def test_qwen3_5_122b_a10b_uses_model_specific_cost_model():
    assert model_name_to_cost_model_stems("Qwen3.5-122B-A10B")[0] == "qwen3.5_122b_a10b"
    assert model_name_to_cost_model_stems("Qwen/Qwen3.5-122B-A10B")[0] == "qwen3.5_122b_a10b"

    cost_model_dir = Path("psrl/trainer/config/cost_model")
    resolved = resolve_cost_model_json_path(str(cost_model_dir), "Qwen/Qwen3.5-122B-A10B")

    assert resolved == str(cost_model_dir / "qwen3.5_122b_a10b.json")


def test_qwen3_30b_thinking_2507_uses_its_model_specific_cost_model():
    cost_model_dir = Path("psrl/trainer/config/cost_model")
    resolved = resolve_cost_model_json_path(str(cost_model_dir), "Qwen3-30B-A3B-Thinking-2507")

    assert resolved == str(cost_model_dir / "qwen3_30b_a3b_thinking_2507.json")


def test_json_file_path_is_returned_as_is():
    json_path = Path("psrl/trainer/config/cost_model/qwen2.5_7b.json")
    resolved = resolve_cost_model_json_path(str(json_path), "Qwen3-8B")

    assert resolved == str(json_path)


def test_missing_json_file_path_is_returned_as_is(tmp_path):
    json_path = tmp_path / "missing_cost_model.json"
    resolved = resolve_cost_model_json_path(str(json_path), "Qwen2.5-7B")

    assert resolved == str(json_path)


def test_directory_without_matching_model_returns_none(tmp_path):
    assert resolve_cost_model_json_path(str(tmp_path), "Unknown-Model-99B") is None


def test_require_directory_without_matching_model_raises(tmp_path):
    with pytest.raises(ValueError, match="does not resolve"):
        require_cost_model_json_path(str(tmp_path), "Unknown-Model-99B")


def test_require_directory_returns_model_named_json():
    cost_model_dir = Path("psrl/trainer/config/cost_model")
    resolved = require_cost_model_json_path(str(cost_model_dir), "Qwen2.5-7B")

    assert resolved == str(cost_model_dir / "qwen2.5_7b.json")
