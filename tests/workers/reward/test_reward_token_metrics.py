import numpy as np
import pytest
from psrl.utils.reward_token_metrics import (
    collect_reward_model_token_lengths,
    extract_reward_model_token_counts,
)

pytestmark = pytest.mark.cpu_test


def test_extract_reward_model_token_counts_recursively():
    extra_info = {
        "rule": {"score": 1.0},
        "gen/default/model_a": {"rm_input_len": 100, "rm_output_len": 200},
        "ensemble": [
            {"gen/default/model_b": {"rm_input_len": np.int64(300), "rm_output_len": np.int64(400)}},
            "ignored",
        ],
    }

    assert extract_reward_model_token_counts(extra_info) == (400, 600)


def test_collect_reward_model_token_lengths_groups_by_model_name():
    extra_info = {
        "rule": {"score": 1.0},
        "gen/default/model_a": {"rm_input_len": 100, "rm_output_len": 200},
        "ensemble": [
            {"gen/default/model_b": {"rm_input_len": np.int64(300), "rm_output_len": np.int64(400)}},
            "ignored",
        ],
    }

    lengths = collect_reward_model_token_lengths(extra_info)

    assert sorted(lengths.overall_input) == [100, 300]
    assert sorted(lengths.overall_output) == [200, 400]
    assert lengths.model_input == {"model_a": [100], "model_b": [300]}
    assert lengths.model_output == {"model_a": [200], "model_b": [400]}


def test_collect_reward_model_token_lengths_keeps_model_name_after_spec_prefix():
    extra_info = {"gen/default/Qwen/Qwen3-8B": {"rm_input_len": 3, "rm_output_len": 4}}

    lengths = collect_reward_model_token_lengths(extra_info)

    assert lengths.model_input == {"Qwen/Qwen3-8B": [3]}
    assert lengths.model_output == {"Qwen/Qwen3-8B": [4]}


def test_extract_reward_model_token_counts_ignores_non_numeric_values():
    extra_info = {
        "rm_input_len": "100",
        "rm_output_len": None,
        "nested": {"rm_input_len": True, "rm_output_len": False},
    }

    assert extract_reward_model_token_counts(extra_info) == (0, 0)
