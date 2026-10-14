"""Tests for TITO training data builder."""

import pytest
from psrl.utils.tito.training_data import build_training_data

pytestmark = pytest.mark.cpu_test


def test_single_turn_no_trim():
    """Single turn: no env tokens, no trim needed."""
    accumulated = [1, 2, 3, 10, 11, 12]
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.1, 12]],
            "finish_reason": "stop",
        }
    ]
    result = build_training_data(accumulated, records)
    assert result["prompt_ids"] == [1, 2, 3]
    assert result["response_ids"] == [10, 11, 12]
    assert result["response_mask"] == [1, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, -0.1]


def test_two_turns_with_env_tokens():
    """Two turns with environment response between them."""
    accumulated = [1, 2, 3, 10, 11, 20, 21, 30, 31]
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "tool_calls",
        },
        {
            "prompt_token_count": 7,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
        },
    ]
    result = build_training_data(accumulated, records)
    assert result["prompt_ids"] == [1, 2, 3]
    assert result["response_ids"] == [10, 11, 20, 21, 30, 31]
    assert result["response_mask"] == [1, 1, 0, 0, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, 0.0, 0.0, -0.2, -0.1]


def test_trailing_trim():
    """GLM47-style trailing stop token gets trimmed on non-last turn (max_trim_tokens=1)."""
    accumulated = [1, 2, 10, 11, 20, 30, 31]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.9, 99]],
            "finish_reason": "tool_calls",
        },
        {
            "prompt_token_count": 5,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
        },
    ]
    result = build_training_data(accumulated, records, max_trim_tokens=1)
    assert result["response_ids"] == [10, 11, 20, 30, 31]
    assert result["response_mask"] == [1, 1, 0, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, 0.0, -0.2, -0.1]


def test_empty_records():
    result = build_training_data([], [])
    assert result["prompt_ids"] == []
    assert result["response_ids"] == []
    assert result["num_turns"] == 0


def test_no_logprobs():
    """Records without logprobs recover token IDs and use neutral logprobs."""
    accumulated = [1, 2, 3]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": None,
            "finish_reason": "stop",
        }
    ]
    result = build_training_data(accumulated, records)
    assert result["prompt_ids"] == [1, 2]
    assert result["response_ids"] == [3]
    assert result["response_mask"] == [1]
    assert result["logprobs"] == [0.0]


def test_trailing_trim_within_max_allowed():
    """trim_count=1 with max_trim_tokens=1 should succeed without ValueError."""
    accumulated = [1, 2, 10, 11, 20, 30, 31]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.9, 99]],
            "finish_reason": "tool_calls",
        },
        {
            "prompt_token_count": 5,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
        },
    ]
    result = build_training_data(accumulated, records, max_trim_tokens=1)
    assert result["response_ids"] == [10, 11, 20, 30, 31]
    assert result["response_mask"] == [1, 1, 0, 1, 1]


def test_trailing_trim_exceeds_max_raises():
    """trim_count=1 with max_trim_tokens=0 should raise ValueError."""
    accumulated = [1, 2, 10, 11, 20, 30, 31]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.9, 99]],
            "finish_reason": "tool_calls",
        },
        {
            "prompt_token_count": 5,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
        },
    ]
    with pytest.raises(ValueError, match="trailing trim overflow"):
        build_training_data(accumulated, records, max_trim_tokens=0)


def test_last_turn_trim_never_occurs():
    """Last turn output is never trimmed even if it doesn't match accumulated tail."""
    accumulated = [1, 2, 10, 11]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "stop",
        }
    ]
    result = build_training_data(accumulated, records, max_trim_tokens=0)
    assert result["response_ids"] == [10, 11]
    assert result["response_mask"] == [1, 1]
