"""Session normalization prepared before training rows are distributed to workers."""

from collections.abc import Hashable, Sequence

import torch


def compute_session_loss_weights(
    response_mask: torch.Tensor,
    session_ids: Sequence[Hashable],
    row_weights: Sequence[float] | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Return one FP32 weight per row for the full training window.

    Each valid token in session `s` receives weight `w_r / (S * T_s)`, where
    `T_s` counts unmasked tokens across all rows of that session, `S` counts
    nonempty sessions, and `w_r` is the row's configured loss weight (1.0 by
    default). Compute after rejection masking and before worker splitting.
    Never renormalize these weights inside an optimizer or micro batch.

    Args:
        response_mask (torch.Tensor): Two-dimensional response mask.
        session_ids (Sequence[Hashable]): One session id per row.
        row_weights (Sequence[float] | torch.Tensor | None): Optional per-row loss
            weights, one per row.
    """
    if response_mask.ndim != 2 or len(session_ids) != response_mask.shape[0]:
        raise ValueError("Expected a two-dimensional response mask and one session ID per row.")
    if row_weights is None:
        weights = [1.0] * response_mask.shape[0]
    else:
        weights = [float(weight) for weight in torch.as_tensor(row_weights).reshape(-1).tolist()]
        if len(weights) != response_mask.shape[0]:
            raise ValueError(
                f"Expected one loss weight per row ({response_mask.shape[0]}), got {len(weights)}."
            )
    counts = response_mask.sum(dim=-1).tolist()
    totals: dict[Hashable, float] = {}
    for session_id, count in zip(session_ids, counts, strict=True):
        if count > 0:
            totals[session_id] = totals.get(session_id, 0) + count
    num_sessions = len(totals)
    return torch.tensor(
        [
            weight / (num_sessions * totals[sid]) if count > 0 else 0.0
            for sid, count, weight in zip(session_ids, counts, weights, strict=True)
        ],
        dtype=torch.float32,
        device=response_mask.device,
    )


def agg_session_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    session_loss_weights: torch.Tensor | None = None,
    **agg_loss_kwargs,
) -> torch.Tensor:
    """Aggregate a loss matrix, honouring psrl's `session-mean-token-mean` mode.

    `session-mean-token-mean` is a psrl loss mode: verl's `agg_loss` has never accepted a
    `session_loss_weights` argument, in either checkout, so passing one through raised
    `TypeError: agg_loss() got an unexpected keyword argument` -- and it did so on *every*
    call, because the keyword was passed unconditionally even when the weights were
    `None` and the mode was plain `token-mean`.

    Keeping the dispatch here rather than patching verl: the mode is psrl's, the weights
    come from `compute_session_loss_weights` above, and verl is free to keep evolving
    `agg_loss` without this having to be re-applied.

    Args:
        session_loss_weights: Per-row FP32 weights from `compute_session_loss_weights`,
            or None for any mode that does not use them.
    """
    from verl.trainer.ppo.core_algos import agg_loss

    if session_loss_weights is None:
        return agg_loss(
            loss_mat=loss_mat,
            loss_mask=loss_mask,
            loss_agg_mode=loss_agg_mode,
            **agg_loss_kwargs,
        )

    # Each row already carries `1 / (S * T_s)`, so the weighted token sum *is* the mean
    # over sessions of the per-session token mean. Deliberately no further division:
    # renormalizing inside a micro batch would make the result depend on how rows were
    # split across workers, which is exactly what the precomputed weights prevent.
    row_sums = torch.sum(loss_mat * loss_mask, dim=-1)
    return torch.sum(row_sums * session_loss_weights.to(row_sums.dtype))
