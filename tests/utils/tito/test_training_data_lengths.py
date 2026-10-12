"""Length invariants across the TITO training-data flow."""

import pytest
import torch
from psrl.utils.tito.training_data import build_training_data


def _record(prompt_token_count: int, output_token_ids: list[int], finish_reason: str = "stop") -> dict:
    """Build one SMG turn record."""
    return {
        "prompt_token_count": prompt_token_count,
        "output_logprobs": [[-0.5, tid] for tid in output_token_ids],
        "finish_reason": finish_reason,
    }


class TestTitoLengthInvariant:
    """build_training_data must keep response_ids, response_mask and logprobs in lockstep."""

    def test_single_turn(self):
        prompt = [1, 2, 3]
        output = [10, 11, 12, 13]
        data = build_training_data(
            accumulated_token_ids=prompt + output,
            records=[_record(len(prompt), output)],
        )
        assert len(data["response_ids"]) == len(data["response_mask"]), (
            f"Length mismatch. ids={len(data['response_ids'])!r}, mask={len(data['response_mask'])!r}."
        )
        assert len(data["logprobs"]) == len(data["response_ids"])
        # A single turn is all model output, so every position is trainable.
        assert data["response_mask"] == [1] * len(output)

    def test_multi_turn_with_environment_tokens(self):
        """The env tokens injected between turns must be masked 0 but still counted."""
        prompt = [1, 2, 3]
        turn1_out = [10, 11]
        env_tokens = [90, 91, 92]  # environment/user text appended after turn 1
        turn2_out = [20, 21, 22]
        accumulated = prompt + turn1_out + env_tokens + turn2_out
        records = [
            _record(len(prompt), turn1_out),
            # turn 2's prompt covers prompt + turn1 output + env tokens
            _record(len(prompt) + len(turn1_out) + len(env_tokens), turn2_out),
        ]
        data = build_training_data(accumulated_token_ids=accumulated, records=records)

        assert len(data["response_ids"]) == len(data["response_mask"]), (
            f"Length mismatch. ids={len(data['response_ids'])!r}, mask={len(data['response_mask'])!r}."
        )
        assert len(data["logprobs"]) == len(data["response_ids"])
        assert data["response_mask"] == [1, 1] + [0, 0, 0] + [1, 1, 1], (
            "env tokens must be masked 0 while model output stays 1"
        )
        assert data["num_turns"] == 2

    def test_missing_logprobs_fails_instead_of_zero_filling(self):
        """A turn with tokens but no logprobs must not be trained on fabricated logprobs."""
        prompt = [1, 2]
        output = [10, 11, 12]
        records = [{"prompt_token_count": len(prompt), "output_logprobs": None, "finish_reason": "stop"}]

        with pytest.raises(ValueError, match="carries no output_logprobs"):
            build_training_data(accumulated_token_ids=prompt + output, records=records)

    def test_twenty_five_turns_stays_aligned(self):
        """The real configuration runs 25 turns, where a per-turn drift would compound."""
        prompt = list(range(50))
        accumulated = list(prompt)
        records = []
        for turn in range(25):
            out = [1000 + turn * 10 + k for k in range(7)]
            records.append(_record(len(accumulated), out))
            accumulated.extend(out)
            if turn < 24:
                env = [5000 + turn]
                accumulated.extend(env)

        data = build_training_data(accumulated_token_ids=accumulated, records=records)
        assert len(data["response_ids"]) == len(data["response_mask"]), (
            f"Length mismatch after 25 turns. ids={len(data['response_ids'])!r}, mask={len(data['response_mask'])!r}."
        )
        assert len(data["logprobs"]) == len(data["response_ids"])
        assert data["num_turns"] == 25


class TestResponseLengthTruncation:
    """session_agent_loop slices ids, mask and logprobs with the same bound."""

    def test_truncation_keeps_lengths_equal(self):
        prompt = [1, 2]
        output = list(range(100, 140))
        data = build_training_data(
            accumulated_token_ids=prompt + output,
            records=[_record(len(prompt), output)],
        )

        # Mirror session_agent_loop.py: all three are sliced by the same response_length.
        response_length = 17
        response_ids = data["response_ids"][:response_length]
        response_mask = data["response_mask"][:response_length]
        logprobs = data["logprobs"][:response_length]

        assert len(response_ids) == len(response_mask) == len(logprobs) == response_length


class TestPpoLossWidthSources:
    """Require response and mask widths to agree before PPO loss padding."""

    def test_matching_fields_give_matching_widths(self):
        lens = [273, 150, 200]
        responses = torch.nested.as_nested_tensor(
            [torch.ones(n, dtype=torch.int64) for n in lens], layout=torch.jagged
        )
        response_mask = torch.nested.as_nested_tensor(
            [torch.ones(n, dtype=torch.int64) for n in lens], layout=torch.jagged
        )

        width_from_responses = int(responses.offsets().diff().max())
        width_from_mask = response_mask.to_padded_tensor(0).shape[1]
        assert width_from_responses == width_from_mask == 273

    def test_diverging_fields_reproduce_the_crash_shape(self):
        """A per-row disagreement is exactly what produced 273 vs 337."""
        response_lens = [273, 150]
        # `responses` carrying one longer row is enough to shift the derived width.
        responses = torch.nested.as_nested_tensor(
            [torch.ones(n, dtype=torch.int64) for n in [337, 150]], layout=torch.jagged
        )
        response_mask = torch.nested.as_nested_tensor(
            [torch.ones(n, dtype=torch.int64) for n in response_lens], layout=torch.jagged
        )

        width_from_responses = int(responses.offsets().diff().max())
        width_from_mask = response_mask.to_padded_tensor(0).shape[1]
        assert width_from_responses == 337
        assert width_from_mask == 273
        assert width_from_responses != width_from_mask, (
            "this is the divergence that reaches compute_policy_loss_vanilla"
        )
