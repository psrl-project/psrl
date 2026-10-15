import numpy as np
import pytest
from psrl.utils.reward_token_metrics import extract_reward_model_token_counts

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


def test_extract_reward_model_token_counts_ignores_non_numeric_values():
    extra_info = {
        "rm_input_len": "100",
        "rm_output_len": None,
        "nested": {"rm_input_len": True, "rm_output_len": False},
    }

    assert extract_reward_model_token_counts(extra_info) == (0, 0)
