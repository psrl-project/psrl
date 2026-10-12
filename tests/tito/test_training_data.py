"""Tests for TITO training data builder."""

import base64
import io

import numpy as np
import pytest
from psrl.utils.tito.training_data import build_training_data


def _npy_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode()


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
    """GLM47-style trailing stop token gets trimmed on non-last turn."""
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


def test_turn_without_logprobs_raises():
    """A turn that owns accumulated tokens but reports no logprobs is a gateway bug.

    Recovering the tokens and zero-filling their logprobs would keep the positions
    trainable with fabricated values, corrupting the rollout log-prob correction.
    """
    accumulated = [1, 2, 3]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": None,
            "finish_reason": "stop",
        }
    ]
    with pytest.raises(ValueError, match="carries no output_logprobs"):
        build_training_data(accumulated, records)


def test_turn_without_tokens_needs_no_logprobs():
    """A turn that produced no assistant tokens owns nothing to log-prob."""
    accumulated = [1, 2, 10, 11, 12]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [],
            "finish_reason": "stop",
        },
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.1, 12]],
            "finish_reason": "stop",
        },
    ]
    result = build_training_data(accumulated, records)
    assert result["response_ids"] == [10, 11, 12]
    assert result["response_mask"] == [1, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, -0.1]


def test_routed_experts_none_when_absent():
    """No routed_experts in records → None (backward compatible)."""
    accumulated = [1, 2, 3, 10, 11, 12]
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.1, 12]],
            "finish_reason": "stop",
        }
    ]
    assert build_training_data(accumulated, records)["routed_experts"] is None


def test_routed_experts_cross_turn_assembly():
    """Place per-turn routed-expert data at absolute positions without gaps."""
    accumulated = [1, 2, 3, 10, 11, 20, 21, 30, 31]
    num_layers, top_k = 2, 3
    t1 = np.arange(5 * num_layers * top_k, dtype=np.uint8).reshape(5, num_layers, top_k)
    t2 = np.arange(100, 100 + 3 * num_layers * top_k, dtype=np.uint8).reshape(3, num_layers, top_k)
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "tool_calls",
            "routed_experts": {
                "data": _npy_b64(t1),
                "num_layers": num_layers,
                "top_k": top_k,
                "dtype": "uint8",
                "prompt_start": 0,
            },
        },
        {
            "prompt_token_count": 7,
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(t2),
                "num_layers": num_layers,
                "top_k": top_k,
                "dtype": "uint8",
                "prompt_start": 5,
            },
        },
    ]
    re = build_training_data(accumulated, records)["routed_experts"]
    # 9 tokens assembled, last sampled token has no RE → 8 rows.
    assert re.shape == (8, num_layers, top_k)
    assert re.dtype == np.uint8
    np.testing.assert_array_equal(re[0:5], t1)
    np.testing.assert_array_equal(re[5:8], t2)


def test_trailing_trim_overflow_is_clamped():
    """trim_count=1 with max_trim_tokens=0 clamps the trim (keeps the divergent token) instead of raising."""
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
    result = build_training_data(accumulated, records, max_trim_tokens=0)
    # The non-matching token 99 is retained because the trim is clamped to 0.
    assert result["response_ids"] == [10, 11, 99, 30, 31]
    assert result["response_mask"] == [1, 1, 1, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, -0.9, -0.2, -0.1]


def test_last_turn_trim_never_occurs():
    """Last turn output is never trimmed even if it doesn't match accumulated tail."""
    # accumulated ends before the last turn's output, so trim is skipped for is_last
    accumulated = [1, 2, 10, 11]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "stop",
        }
    ]
    # Single-turn (is_last=True from the start): no trim attempted, no ValueError
    result = build_training_data(accumulated, records, max_trim_tokens=0)
    assert result["response_ids"] == [10, 11]
    assert result["response_mask"] == [1, 1]
