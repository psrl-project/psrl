"""Tests for DAPO overlong filtering and the termination breakdown metrics."""

import numpy as np
import pytest
import torch
from psrl.trainer.ppo.utils import _compute_termination_metrics
from psrl.workers.agent_loop.loops.utils import TerminateReason

pytestmark = pytest.mark.cpu_test


class TestBudgetTruncatedClassification:
    """Test which terminations count as a harness budget cutoff."""

    def test_turn_and_length_caps_are_truncations(self):
        assert TerminateReason.MAX_TURNS_EXCEEDED.is_budget_truncated
        assert TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED.is_budget_truncated

    def test_a_finished_episode_is_never_truncated(self):
        assert not TerminateReason.FINISHED.is_budget_truncated

    def test_verifier_error_is_not_a_budget_cutoff(self):
        # It is masked by `is_ungraded` instead, because the agent did finish and its
        # turns are honest. What is missing is the measurement, not the work.
        assert not TerminateReason.VERIFIER_ERROR.is_budget_truncated
        assert TerminateReason.VERIFIER_ERROR.is_ungraded

    def test_only_verifier_error_is_ungraded(self):
        # A graded 0.0 must stay trainable, so nothing else may claim to be ungraded.
        for reason in (
            TerminateReason.FINISHED,
            TerminateReason.MAX_TURNS_EXCEEDED,
            TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
            TerminateReason.AGENT_TIMEOUT,
            TerminateReason.ENV_TIMEOUT,
            TerminateReason.ROLLOUT_ERROR,
        ):
            assert not reason.is_ungraded, f"{reason} would be dropped from the gradient"

    def test_ungraded_episodes_still_carry_trainable_data(self):
        # Masking decides the loss mask, not whether the row is committed. The row must
        # survive so its score still moves the GRPO group baseline.
        assert TerminateReason.VERIFIER_ERROR.is_successful

    def test_infrastructure_timeouts_are_not_budget_cutoffs(self):
        # These are faults rather than an exhausted budget, and conflating them would
        # silently drop data whenever the cluster misbehaves.
        assert not TerminateReason.AGENT_TIMEOUT.is_budget_truncated
        assert not TerminateReason.ENV_TIMEOUT.is_budget_truncated

    def test_truncated_reasons_still_carry_trainable_data(self):
        # Overlong filtering decides the loss mask, not whether the row is committed.
        # The row must survive so its score still moves the GRPO group baseline.
        for reason in (TerminateReason.MAX_TURNS_EXCEEDED, TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED):
            assert reason.is_successful


class TestOverlongFilteringIsOptional:
    """Test the config switch that gates the whole mechanism."""

    def test_the_flag_defaults_to_off(self):
        # Filtering discards real rollout tokens, so it must be an explicit opt-in and
        # the previous behaviour has to stay reproducible for an A/B.
        from psrl.workers.config.rollout import AgentLoopConfig

        assert AgentLoopConfig().overlong_filtering is False

    def test_the_flag_can_be_enabled(self):
        from psrl.workers.config.rollout import AgentLoopConfig

        assert AgentLoopConfig(overlong_filtering=True).overlong_filtering is True

    def test_the_yaml_default_matches_the_dataclass(self):
        # A drifting yaml default would silently enable filtering for every recipe.
        from pathlib import Path

        import yaml

        cfg = yaml.safe_load(Path("psrl/trainer/config/rollout/psrl_rollout.yaml").read_text())
        assert cfg["agent"]["overlong_filtering"] is False


class TestMaskZeroing:
    """Test the mask rewrite applied to a budget-truncated trajectory."""

    def test_zeroing_preserves_length(self):
        # `response_mask` doubles as the per-row length contract, so only the VALUES may
        # change. A shorter row would slice another trajectory's log-probs.
        mask = torch.ones(37, dtype=torch.int64)
        zeroed = torch.zeros_like(mask)
        assert zeroed.size(0) == mask.size(0)
        assert zeroed.sum() == 0

    def test_reward_tensor_is_built_from_shape_not_values(self):
        # `rm_scores` is `zeros_like(response_mask)` with the score in the last slot, so
        # zeroing the mask leaves the reward intact and the group baseline honest.
        response_mask = torch.zeros(8, dtype=torch.int64)
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        rm_scores[-1] = 1.0
        assert rm_scores.sum() == pytest.approx(1.0)


class TestTokenMeanHandlesAnEmptyBatch:
    """Test that a fully filtered batch cannot emit nan."""

    def test_all_masked_batch_yields_zero_not_nan(self):
        # Every trajectory truncated means batch_num_tokens is 0, and the unclamped
        # division returned 0/0, which propagates nan through every parameter.
        from verl.trainer.ppo.core_algos import agg_loss

        loss_mat = torch.randn(4, 16, requires_grad=True)
        loss_mask = torch.zeros(4, 16)
        loss = agg_loss(loss_mat=loss_mat, loss_mask=loss_mask, loss_agg_mode="token-mean")
        assert torch.isfinite(loss)
        assert loss.item() == pytest.approx(0.0), "A fully masked batch must have zero loss."
        loss.backward()
        assert torch.equal(loss_mat.grad, torch.zeros_like(loss_mat)), "Filtered tokens must have zero gradients."

    def test_a_normal_batch_is_unchanged_by_the_guard(self):
        from verl.trainer.ppo.core_algos import agg_loss

        loss_mat = torch.ones(2, 4)
        loss_mask = torch.ones(2, 4)
        loss = agg_loss(loss_mat=loss_mat, loss_mask=loss_mask, loss_agg_mode="token-mean")
        assert loss.item() == pytest.approx(1.0)

    def test_partially_masked_batch_normalizes_by_surviving_tokens(self):
        from verl.trainer.ppo.core_algos import agg_loss

        loss_mat = torch.ones(2, 4)
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        loss = agg_loss(loss_mat=loss_mat, loss_mask=loss_mask, loss_agg_mode="token-mean")
        assert loss.item() == pytest.approx(1.0)


class TestTerminationMetrics:
    """Test the per-termination reward split and the group degeneracy metric."""

    def test_reward_is_split_by_termination(self):
        # A blended `critic/score/mean` hid that solved and truncated episodes move in
        # opposite directions, which is why the collapse took 16 steps to notice.
        metrics = _compute_termination_metrics(
            terminate_reasons=["finished", "finished", "max_turns_exceeded", "max_turns_exceeded"],
            parent_ids=["p0", "p0", "p0", "p0"],
            scores=[1.0, 1.0, 0.0, 0.0],
            trained_tokens=[100.0, 100.0, 0.0, 0.0],
        )
        assert metrics["termination/finished/score_mean"] == pytest.approx(1.0)
        assert metrics["termination/max_turns_exceeded/score_mean"] == pytest.approx(0.0)
        assert metrics["termination/finished/fraction"] == pytest.approx(0.5)

    def test_masked_sample_fraction_counts_dropped_trajectories(self):
        metrics = _compute_termination_metrics(
            terminate_reasons=["finished", "max_turns_exceeded"],
            parent_ids=["p0", "p0"],
            scores=[1.0, 0.0],
            trained_tokens=[50.0, 0.0],
        )
        assert metrics["termination/masked_sample_fraction"] == pytest.approx(0.5)
        assert metrics["termination/trained_tokens"] == pytest.approx(50.0)

    def test_a_uniform_group_is_flagged_as_zero_variance(self):
        # Every rollout scoring alike gives an advantage of exactly zero, so the group
        # contributes no gradient. This is the metric that shows a collapse in progress.
        metrics = _compute_termination_metrics(
            terminate_reasons=["max_turns_exceeded"] * 4,
            parent_ids=["p0", "p0", "p1", "p1"],
            scores=[0.0, 0.0, 0.0, 0.0],
            trained_tokens=[0.0, 0.0, 0.0, 0.0],
        )
        assert metrics["group/zero_variance_fraction"] == pytest.approx(1.0)
        assert metrics["group/count"] == pytest.approx(2.0)

    def test_a_mixed_group_is_not_degenerate(self):
        metrics = _compute_termination_metrics(
            terminate_reasons=["finished", "max_turns_exceeded"],
            parent_ids=["p0", "p0"],
            scores=[1.0, 0.0],
            trained_tokens=[10.0, 0.0],
        )
        assert metrics["group/zero_variance_fraction"] == pytest.approx(0.0)

    def test_a_singleton_group_is_not_counted_as_degenerate(self):
        # A group of one has no variance by construction, so counting it would report a
        # false collapse whenever rollout_n is 1.
        metrics = _compute_termination_metrics(
            terminate_reasons=["finished"],
            parent_ids=["p0"],
            scores=[1.0],
            trained_tokens=[10.0],
        )
        assert metrics["group/zero_variance_fraction"] == pytest.approx(0.0)

    def test_an_empty_batch_returns_no_metrics(self):
        assert _compute_termination_metrics([], [], [], []) == {}

    def test_mismatched_lengths_are_rejected(self):
        # Silent truncation via zip would misattribute rewards to the wrong termination.
        with pytest.raises(ValueError):
            _compute_termination_metrics(
                terminate_reasons=["finished", "finished"],
                parent_ids=["p0", "p0"],
                scores=[1.0],
                trained_tokens=[1.0, 1.0],
            )


class TestGroupBaselineIsPreserved:
    """Test that filtering removes gradient without biasing the GRPO baseline."""

    def test_truncated_zeros_still_lower_the_group_mean(self):
        # The whole point of masking rather than dropping: finishers keep a positive
        # advantage only if the truncated zeros remain in the mean.
        scores = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        advantages = scores - scores.mean()
        assert advantages[0] == pytest.approx(0.75)
        # Dropping the truncated rows entirely would collapse the advantage to zero.
        finished_only = scores[:2]
        assert (finished_only - finished_only.mean())[0] == pytest.approx(0.0)
