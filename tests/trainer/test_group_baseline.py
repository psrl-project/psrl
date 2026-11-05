"""Which samples a group-relative advantage is measured against.

A row that trains no tokens holds a placeholder reward: no verifier scored it, the rollout
masked it as budget-truncated, or it exists only to keep the batch shape. Those tests pin
that such a row no longer moves the baseline its siblings are measured against, and that
nothing else about the advantage changes.
"""

import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")
TensorDict = pytest.importorskip("tensordict").TensorDict

from psrl.trainer.ppo.utils import (  # noqa: E402
    PSRL_compute_advantage,
    compute_advantage_for_multi_trajectories,
    valid_sample_mask,
)
from verl import DataProto  # noqa: E402
from verl.trainer.ppo.core_algos import AdvantageEstimator, compute_grpo_outcome_advantage  # noqa: E402

_SWITCH = "group_baseline_valid_only"
_EPS = 1e-6
# A row's reward is the sum of its token rewards, so one entry per row is enough to place it.
_LENGTH = 3


def _batch(
    rewards: list[float],
    masks: list[float],
    groups: list[int],
    *,
    values: bool = False,
    mask_width: int | None = None,
) -> DataProto:
    """A batch whose fields may be padded to different widths.

    Batch fields do not share a trailing width: each is padded to its own maximum, so a
    test can hand the estimator a `response_mask` wider than `token_level_rewards`.
    """
    size = len(rewards)
    token_rewards = torch.tensor([[reward / _LENGTH] * _LENGTH for reward in rewards], dtype=torch.float32)
    width = mask_width or _LENGTH
    tensors = {
        "token_level_rewards": token_rewards.clone(),
        "token_level_scores": token_rewards.clone(),
        "response_mask": torch.tensor([[mask] * width for mask in masks], dtype=torch.float32),
    }
    if values:
        tensors["values"] = torch.zeros(size, _LENGTH)
    return DataProto(
        batch=TensorDict(tensors, batch_size=[size]),
        non_tensor_batch={"parent_id": np.array(groups, dtype=object)},
    )


def _advantages(data: DataProto) -> list[float]:
    return [round(float(row[0]), 6) for row in data.batch["advantages"]]


def _baseline(scores: list[float], index: int) -> float:
    """A measured sample's advantage, as the group statistic defines it."""
    measured = torch.tensor(scores, dtype=torch.float32)
    return float((measured[index] - measured.mean()) / (measured.std() + _EPS))


def test_the_switch_off_keeps_the_reference_behaviour() -> None:
    """With the switch off the estimator is the one it always was."""
    data = _batch([1.0, 0.5, 0.25, 0.0], [1.0, 1.0, 0.0, 1.0], [7, 7, 7, 7])
    reference, _ = compute_grpo_outcome_advantage(
        token_level_rewards=data.batch["token_level_rewards"],
        response_mask=data.batch["response_mask"],
        index=data.non_tensor_batch["parent_id"],
    )

    PSRL_compute_advantage(data, AdvantageEstimator.GRPO, config={_SWITCH: False})

    assert torch.equal(data.batch["advantages"], reference)


def test_the_switch_is_a_no_op_when_every_row_carries_a_measurement() -> None:
    data = _batch([1.0, 0.5, 0.25, 0.0], [1.0, 1.0, 1.0, 1.0], [7, 7, 7, 7])
    expected = _advantages(
        PSRL_compute_advantage(
            _batch([1.0, 0.5, 0.25, 0.0], [1.0, 1.0, 1.0, 1.0], [7, 7, 7, 7]),
            AdvantageEstimator.GRPO,
            config={_SWITCH: False},
        )
    )

    PSRL_compute_advantage(data, AdvantageEstimator.GRPO, config={_SWITCH: True})

    assert _advantages(data) == expected


def test_a_row_without_a_measurement_leaves_its_group_baseline() -> None:
    """The defect this switch exists for: a placeholder reward moved its siblings' baseline."""
    rewards = [1.0, 0.5, 0.5, 0.25, 0.0]
    masks = [1.0, 1.0, 1.0, 1.0, 0.0]  # the last row trains nothing
    groups = [7] * 5

    excluded = PSRL_compute_advantage(_batch(rewards, masks, groups), AdvantageEstimator.GRPO, config={_SWITCH: True})
    kept = PSRL_compute_advantage(_batch(rewards, masks, groups), AdvantageEstimator.GRPO, config={_SWITCH: False})

    measured = rewards[:4]
    assert _advantages(excluded)[:4] == [round(_baseline(measured, i), 6) for i in range(4)]
    # And it is not the baseline the placeholder used to produce.
    assert _advantages(excluded)[:4] != _advantages(kept)[:4]
    # A row that trains no tokens gets no advantage, whatever its placeholder reward was.
    assert _advantages(excluded)[4] == 0.0


def test_each_group_is_measured_against_its_own_measured_rows() -> None:
    # Group 7 has two measured rows and keeps measuring them against each other. Group 8
    # has one, which leaves it without a relative baseline.
    data = _batch([1.0, 0.5, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [7, 7, 8, 8])

    PSRL_compute_advantage(data, AdvantageEstimator.GRPO, config={_SWITCH: True})

    assert _advantages(data) == [
        round(_baseline([1.0, 0.5], 0), 6),
        round(_baseline([1.0, 0.5], 1), 6),
        0.0,
        0.0,
    ]


def test_the_advantage_follows_the_response_mask_not_the_reward_width() -> None:
    """A batch pads each field to its own maximum, so the two widths can differ.

    The estimator reads `token_level_rewards` but shapes its output like `response_mask`.
    Sizing the scatter buffer from the rewards would make it fail whenever a row's reward
    is padded narrower than its mask, and the trainer un-pads against `response_mask`.
    """
    rewards = [1.0, 0.5, 0.5, 0.25, 0.0]
    masks = [1.0, 1.0, 1.0, 1.0, 0.0]
    data = _batch(rewards, masks, [7] * 5, mask_width=5)
    assert data.batch["token_level_rewards"].shape[1] == _LENGTH
    assert data.batch["response_mask"].shape[1] == 5

    PSRL_compute_advantage(data, AdvantageEstimator.GRPO, config={_SWITCH: True})

    advantages = data.batch["advantages"]
    assert tuple(advantages.shape) == (5, 5)
    assert [round(float(row[0]), 6) for row in advantages[:4]] == [
        round(_baseline([1.0, 0.5, 0.5, 0.25], i), 6) for i in range(4)
    ]
    assert float(advantages[4].abs().sum()) == 0.0


def test_a_group_left_with_one_measurement_gets_no_advantage(caplog) -> None:
    """One sample has no deviation from its own mean, so its group carries no signal."""
    data = _batch([1.0, 0.0], [1.0, 0.0], [7, 7])

    with caplog.at_level(logging.WARNING):
        PSRL_compute_advantage(data, AdvantageEstimator.GRPO, config={_SWITCH: True})

    assert _advantages(data) == [0.0, 0.0]
    assert "without a relative baseline" in caplog.text


def test_a_masked_session_does_not_move_its_groups_baseline() -> None:
    """The same rule through the session-level wrapper the trainer calls.

    There one session is one sample: its final trajectory is what the group statistic
    measures, and every earlier trajectory of it is trained on that same number. A session
    whose episode was masked holds a placeholder instead of a measurement, so it leaves the
    statistic and its trajectories get no advantage.
    """
    # Session 11 has two trajectories, of which the final one is measured. Session 12 is
    # measured, and session 13's episode was masked as budget-truncated.
    rewards = [1.0, 1.0, 0.5, 0.0]
    masks = [1.0, 1.0, 1.0, 0.0]
    groups = [7, 7, 7, 7]
    keys = ["11_0", "11_1", "12_0", "13_0"]

    excluded = compute_advantage_for_multi_trajectories(
        _batch(rewards, masks, groups),
        batch_keys=keys,
        adv_estimator=AdvantageEstimator.GRPO,
        config={_SWITCH: True},
    )
    kept = compute_advantage_for_multi_trajectories(
        _batch(rewards, masks, groups),
        batch_keys=keys,
        adv_estimator=AdvantageEstimator.GRPO,
        config={_SWITCH: False},
    )

    assert _advantages(excluded) == [
        round(_baseline([1.0, 0.5], 0), 6),  # session 11's final trajectory, broadcast back
        round(_baseline([1.0, 0.5], 0), 6),
        round(_baseline([1.0, 0.5], 1), 6),
        0.0,
    ]
    assert _advantages(excluded) != _advantages(kept)


def test_a_value_based_advantage_is_unchanged_by_the_switch() -> None:
    """GAE measures a row against its own values, so removing rows changes nothing else."""
    rewards = [1.0, 0.5, 0.25]
    masks = [1.0, 1.0, 0.0]
    groups = [7, 7, 7]

    excluded = PSRL_compute_advantage(
        _batch(rewards, masks, groups, values=True), AdvantageEstimator.GAE, config={_SWITCH: True}
    )
    kept = PSRL_compute_advantage(
        _batch(rewards, masks, groups, values=True), AdvantageEstimator.GAE, config={_SWITCH: False}
    )

    assert _advantages(excluded)[:2] == _advantages(kept)[:2]
    assert _advantages(excluded)[2] == 0.0


def test_the_switch_defaults_to_off() -> None:
    data = _batch([1.0, 0.0], [1.0, 0.0], [7, 7])
    expected = _advantages(PSRL_compute_advantage(_batch([1.0, 0.0], [1.0, 0.0], [7, 7]), AdvantageEstimator.GRPO))

    PSRL_compute_advantage(data, AdvantageEstimator.GRPO, config={})

    assert _advantages(data) == expected


def test_the_group_of_a_row_must_be_known() -> None:
    """The restriction needs the group a row belongs to, and says so rather than guessing."""
    data = _batch([1.0, 0.5], [1.0, 0.0], [7, 7])
    data.non_tensor_batch.pop("parent_id")

    with pytest.raises(ValueError, match="parent_id or uid"):
        PSRL_compute_advantage(data, AdvantageEstimator.GRPO, config={_SWITCH: True})


def test_valid_sample_mask_reads_the_mask() -> None:
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0]])

    assert valid_sample_mask(mask).tolist() == [True, False, True]


def test_the_session_primary_is_measured_even_when_it_is_not_the_last_row() -> None:
    """A subagent row can sort after the main row; the advantage still reads the main."""
    rewards = [1.0, 0.0, 0.0]
    masks = [1.0, 1.0, 1.0]
    groups = [7, 7, 7]
    data = _batch(rewards, masks, groups)
    data.non_tensor_batch["session_primary"] = np.array([True, False, True])

    result = compute_advantage_for_multi_trajectories(
        data,
        batch_keys=["11_0", "11_1", "12_0"],
        adv_estimator=AdvantageEstimator.GRPO,
        config={_SWITCH: True},
    )

    expected = round(_baseline([1.0, 0.0], 0), 6)
    assert _advantages(result) == [expected, expected, round(_baseline([1.0, 0.0], 1), 6)]
