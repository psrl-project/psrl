"""`session-mean-token-mean` aggregates without patching verl's `agg_loss`.

The mode is psrl's; verl's `agg_loss` has never accepted `session_loss_weights`, in
either checkout. psrl passed it anyway -- and unconditionally, so the call raised
`TypeError` even for plain `token-mean` runs where the weights were `None` and the
argument carried no information at all.

The property that matters beyond "it runs": the weights encode `1 / (S * T_s)` over the
*full* training window, so the aggregate must not be renormalized per micro batch. If it
were, the loss would depend on how rows happened to be split across workers.
"""

from __future__ import annotations

import pytest
import torch
from psrl.trainer.ppo.session_loss import agg_session_loss, compute_session_loss_weights

pytestmark = pytest.mark.cpu_test


class TestItDelegatesWhenNoWeightsAreGiven:
    def test_token_mean_matches_verls_own_result(self):
        from verl.trainer.ppo.core_algos import agg_loss

        loss_mat = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

        ours = agg_session_loss(loss_mat, loss_mask, "token-mean")
        theirs = agg_loss(loss_mat=loss_mat, loss_mask=loss_mask, loss_agg_mode="token-mean")

        assert torch.allclose(ours, theirs)

    def test_extra_kwargs_reach_verl(self):
        """`loss_scale_factor` and friends must still work; this is a passthrough."""
        loss_mat = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        loss_mask = torch.ones_like(loss_mat)

        scaled = agg_session_loss(
            loss_mat, loss_mask, "seq-mean-token-sum-norm", loss_scale_factor=4
        )
        unscaled = agg_session_loss(
            loss_mat, loss_mask, "seq-mean-token-sum-norm", loss_scale_factor=1
        )

        assert torch.allclose(scaled * 4, unscaled)


class TestSessionWeightingIsAppliedAsComputed:
    def test_two_equal_sessions_each_contribute_half(self):
        """The defining property of session-mean: a session's weight is independent of
        how many tokens it has."""
        loss_mat = torch.tensor([[2.0, 2.0, 2.0, 2.0], [6.0, 0.0, 0.0, 0.0]])
        loss_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
        weights = compute_session_loss_weights(loss_mask, ["a", "b"])

        aggregated = agg_session_loss(loss_mat, loss_mask, "session-mean-token-mean", weights)

        # session a: mean 2.0, session b: mean 6.0 -> (2 + 6) / 2
        assert aggregated.item() == pytest.approx(4.0)

    def test_rows_of_one_session_are_pooled_not_averaged_twice(self):
        """Two rows of the same session count once, weighted by their shared token total."""
        loss_mat = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
        loss_mask = torch.ones_like(loss_mat)
        weights = compute_session_loss_weights(loss_mask, ["s", "s"])

        aggregated = agg_session_loss(loss_mat, loss_mask, "session-mean-token-mean", weights)

        # one session, 4 tokens, total 8 -> 8/4
        assert aggregated.item() == pytest.approx(2.0)

    def test_a_split_batch_sums_to_the_whole_batch_result(self):
        """Micro-batch invariance: the reason weights are precomputed over the full window.

        Renormalizing inside the aggregate would break this, and nothing would report it
        -- training would simply optimize a slightly different objective depending on
        worker count.
        """
        loss_mat = torch.tensor([[2.0, 2.0], [6.0, 0.0], [1.0, 3.0]])
        loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        weights = compute_session_loss_weights(loss_mask, ["a", "b", "c"])

        whole = agg_session_loss(loss_mat, loss_mask, "session-mean-token-mean", weights)
        first = agg_session_loss(
            loss_mat[:2], loss_mask[:2], "session-mean-token-mean", weights[:2]
        )
        second = agg_session_loss(
            loss_mat[2:], loss_mask[2:], "session-mean-token-mean", weights[2:]
        )

        assert torch.allclose(whole, first + second)

    def test_fully_masked_rows_contribute_nothing(self):
        loss_mat = torch.tensor([[5.0, 5.0], [9.0, 9.0]])
        loss_mask = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        weights = compute_session_loss_weights(loss_mask, ["a", "b"])

        aggregated = agg_session_loss(loss_mat, loss_mask, "session-mean-token-mean", weights)

        assert aggregated.item() == pytest.approx(5.0)


class TestTheCallSiteNoLongerPassesTheKwargToVerl:
    def test_the_trainer_uses_the_brianparker325085_aggregator(self):
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[2] / "psrl/trainer/ppo/ray_trainer.py"
        ).read_text()
        at = source.index("session_loss_weights=session_loss_weights")
        preceding = source[max(0, at - 400) : at]

        assert "agg_session_loss(" in preceding


class TestPerRowLossWeights:
    def test_role_weights_scale_each_rows_session_mean(self):
        loss_mat = torch.tensor([[2.0, 2.0], [6.0, 6.0]])
        loss_mask = torch.ones_like(loss_mat)
        weights = compute_session_loss_weights(loss_mask, ["main", "sub"], row_weights=[1.0, 0.5])

        aggregated = agg_session_loss(loss_mat, loss_mask, "session-mean-token-mean", weights)

        # two sessions, so each session's share is 1/2 of its token mean:
        # 1.0 * (2.0 / 2) + 0.5 * (6.0 / 2)
        assert aggregated.item() == pytest.approx(2.5)

    def test_a_weight_count_mismatch_is_rejected(self):
        loss_mask = torch.ones(2, 2)

        with pytest.raises(ValueError, match="one loss weight per row"):
            compute_session_loss_weights(loss_mask, ["a", "a"], row_weights=[1.0])
