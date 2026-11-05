"""Tests for TITO training data builder."""

import base64
import io

import numpy as np
import pytest
import torch
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


def test_no_logprobs_is_rejected():
    """A trainable turn with captured token ids but no logprobs is rejected."""
    accumulated = [1, 2, 3]
    records = [
        {
            "prompt_token_count": 2,
            "output_token_ids": [3],
            "output_logprobs": None,
            "finish_reason": "stop",
        }
    ]
    with pytest.raises(ValueError, match="Missing output_logprobs"):
        build_training_data(accumulated, records)


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
    """Concatenate per-turn routed-expert segments into the canonical int16 tensor."""
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
    # Canonical replay data covers all 9 tokens, and the final never-forwarded row is filler.
    assert re.shape == (9, num_layers, top_k)
    assert re.dtype == torch.int16
    torch.testing.assert_close(re[0:5], torch.from_numpy(t1).to(torch.int16))
    torch.testing.assert_close(re[5:8], torch.from_numpy(t2).to(torch.int16))
    assert torch.count_nonzero(re[-1]) == 0


def test_routed_experts_places_segment_at_prompt_start():
    """A segment is written at its absolute prompt_start; earlier rows stay zero."""
    accumulated = [1, 2, 3, 10, 11]
    segment = np.ones((4, 1, 1), dtype=np.uint8)
    records = [
        {
            "prompt_token_count": 3,
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(segment),
                "num_layers": 1,
                "top_k": 1,
                "dtype": "uint8",
                "prompt_start": 1,
            },
        }
    ]

    routed_experts = build_training_data(accumulated, records)["routed_experts"]
    assert torch.count_nonzero(routed_experts[0]) == 0
    torch.testing.assert_close(routed_experts[1:5], torch.from_numpy(segment).to(torch.int16))


def test_routed_experts_rejects_overlap():
    """Overlapping segments describe the same positions twice, so they are rejected."""
    accumulated = [1, 2, 10, 20, 30]
    first = np.ones((3, 1, 1), dtype=np.uint16)
    second = np.full((1, 1, 1), 2, dtype=np.uint16)
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10]],
            "finish_reason": "tool_calls",
            "routed_experts": {
                "data": _npy_b64(first),
                "num_layers": 1,
                "top_k": 1,
                "dtype": "uint16",
                "prompt_start": 0,
            },
        },
        {
            "prompt_token_count": 4,
            "output_logprobs": [[-0.2, 30]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(second),
                "num_layers": 1,
                "top_k": 1,
                "dtype": "uint16",
                "prompt_start": 2,
            },
        },
    ]

    with pytest.raises(ValueError, match="overlap"):
        build_training_data(accumulated, records)


def test_routed_experts_zero_fills_a_token_only_turn():
    """A turn without a payload leaves its positions zero; the assembler does not reject it."""
    accumulated = [1, 2, 10, 20, 30]
    segment = np.ones((3, 1, 1), dtype=np.uint8)
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10]],
            "finish_reason": "tool_calls",
            "routed_experts": {
                "data": _npy_b64(segment),
                "num_layers": 1,
                "top_k": 1,
                "dtype": "uint8",
                "prompt_start": 0,
            },
        },
        {
            "prompt_token_count": 4,
            "output_logprobs": [[-0.2, 30]],
            "finish_reason": "stop",
            "routed_experts": None,
        },
    ]

    routed_experts = build_training_data(accumulated, records)["routed_experts"]
    assert routed_experts.shape == (5, 1, 1)
    torch.testing.assert_close(routed_experts[:3], torch.from_numpy(segment).to(torch.int16))
    assert torch.count_nonzero(routed_experts[3:]) == 0


def test_routed_experts_rejects_metadata_mismatch():
    """Metadata that disagrees with the payload shape is rejected."""
    accumulated = [1, 2, 10]
    segment = np.ones((2, 1, 1), dtype=np.uint8)
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10]],
            "finish_reason": "stop",
            "routed_experts": {
                "data": _npy_b64(segment),
                "num_layers": 2,
                "top_k": 1,
                "dtype": "uint8",
                "prompt_start": 0,
            },
        }
    ]

    with pytest.raises(ValueError, match="metadata mismatch"):
        build_training_data(accumulated, records)


def test_trailing_trim_overflow_is_rejected():
    """A divergence beyond max_trim_tokens is rejected instead of clamped."""
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
    with pytest.raises(ValueError, match="diverge at turn 0"):
        build_training_data(accumulated, records, max_trim_tokens=0)


def test_boundary_token_within_ceiling_is_trimmed():
    """One divergent trailing boundary token is trimmed when the ceiling allows it."""
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
    # The divergent token 99 is trimmed, and 20 is the environment token between turns.
    assert result["response_ids"] == [10, 11, 20, 30, 31]
    assert result["response_mask"] == [1, 1, 0, 1, 1]
    assert result["logprobs"] == [-0.5, -0.3, 0.0, -0.2, -0.1]


def test_unconsumed_accumulated_tokens_are_rejected():
    """Trailing accumulated tokens that no turn accounts for are rejected."""
    accumulated = [1, 2, 3, 4]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.1, 3]],
            "finish_reason": "stop",
        }
    ]
    with pytest.raises(ValueError, match="did not consume"):
        build_training_data(accumulated, records)


def test_last_turn_divergence_is_rejected():
    """The final turn must align exactly, so a divergent trailing token is rejected."""
    accumulated = [1, 2, 10, 11]
    records = [
        {
            "prompt_token_count": 2,
            "output_logprobs": [[-0.5, 10], [-0.3, 11], [-0.9, 99]],
            "finish_reason": "stop",
        }
    ]
    with pytest.raises(ValueError, match="diverge at turn 0"):
        build_training_data(accumulated, records, max_trim_tokens=1)


def test_single_turn_exact_alignment():
    """A single turn that matches the accumulated stream exactly is accepted."""
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


def test_owner_aware_prompt_starts_at_first_owned_turn():
    """Shared main-prefix turns are the subagent's prompt, not its response."""
    accumulated = [1, 2, 10, 11, 20, 30, 31]
    records = [
        {
            "prompt_token_count": 2,
            "output_token_ids": [10, 11],
            "output_logprobs": [[-0.5, 10], [-0.3, 11]],
            "owner_trajectory_id": 0,
            "capture": "full",
        },
        {
            "prompt_token_count": 5,
            "output_token_ids": [30, 31],
            "output_logprobs": [[-0.2, 30], [-0.1, 31]],
            "owner_trajectory_id": 1,
            "capture": "full",
        },
    ]

    result = build_training_data(
        accumulated,
        records,
        trainable_trajectory_id=1,
    )

    assert result["prompt_ids"] == [1, 2, 10, 11, 20]
    assert result["response_ids"] == [30, 31]
    assert result["response_mask"] == [1, 1]
    assert result["num_turns"] == 1


def test_owner_aware_main_masks_a_subagent_context_turn():
    """A main trajectory trains its own turn and masks another owner's output."""
    accumulated = [1, 2, 10, 20, 30]
    records = [
        {
            "prompt_token_count": 2,
            "output_token_ids": [10],
            "output_logprobs": [[-0.5, 10]],
            "owner_trajectory_id": 0,
            "capture": "full",
        },
        {
            "prompt_token_count": 4,
            "output_token_ids": [30],
            "output_logprobs": [[-0.2, 30]],
            "owner_trajectory_id": 1,
            "capture": "full",
        },
    ]

    result = build_training_data(
        accumulated,
        records,
        trainable_trajectory_id=0,
    )

    assert result["prompt_ids"] == [1, 2]
    assert result["response_ids"] == [10, 20, 30]
    assert result["response_mask"] == [1, 0, 0]
    assert result["logprobs"] == [-0.5, 0.0, 0.0]
    assert result["num_turns"] == 1


def test_token_only_owned_turn_is_masked():
    """A token-only turn contributes replay text but no training payload."""
    accumulated = [1, 2, 10, 11]
    records = [
        {
            "prompt_token_count": 2,
            "output_token_ids": [10],
            "owner_trajectory_id": 0,
            "capture": "full",
            "output_logprobs": [[-0.5, 10]],
        },
        {
            "prompt_token_count": 3,
            "output_token_ids": [11],
            "owner_trajectory_id": 0,
            "capture": "tokens",
        },
    ]

    result = build_training_data(
        accumulated,
        records,
        trainable_trajectory_id=0,
    )

    assert result["response_ids"] == [10, 11]
    assert result["response_mask"] == [1, 0]
    assert result["logprobs"] == [-0.5, 0.0]


def test_shared_prefix_context_masks_a_shared_owned_turn():
    """`context` trains only nodes unique to the trajectory."""
    accumulated = [1, 2, 10]
    records = [
        {
            "prompt_token_count": 2,
            "output_token_ids": [10],
            "output_logprobs": [[-0.5, 10]],
            "owner_trajectory_id": 0,
            "capture": "full",
            "shared": True,
        },
    ]

    result = build_training_data(
        accumulated,
        records,
        trainable_trajectory_id=0,
        shared_prefix="context",
    )

    assert result["response_mask"] == [0]
