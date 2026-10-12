"""Session normalization prepared before training rows are distributed to workers."""

from collections.abc import Hashable, Sequence

import torch


def compute_session_loss_weights(
    response_mask: torch.Tensor,
    session_ids: Sequence[Hashable],
) -> torch.Tensor:
    """
    Return one FP32 weight per row for the full training window.

    Each valid token in session `s` receives weight `1 / (S * T_s)`, where
    `T_s` counts unmasked tokens across all rows of that session and `S` counts
    nonempty sessions. Compute after rejection masking and before worker splitting.
    Never renormalize these weights inside an optimizer or micro batch.
    """
    if response_mask.ndim != 2 or len(session_ids) != response_mask.shape[0]:
        raise ValueError("Expected a two-dimensional response mask and one session ID per row.")
    counts = response_mask.sum(dim=-1).tolist()
    totals: dict[Hashable, float] = {}
    for session_id, count in zip(session_ids, counts, strict=True):
        if count > 0:
            totals[session_id] = totals.get(session_id, 0) + count
    num_sessions = len(totals)
    return torch.tensor(
        [
            1.0 / (num_sessions * totals[sid]) if count > 0 else 0.0
            for sid, count in zip(session_ids, counts, strict=True)
        ],
        dtype=torch.float32,
        device=response_mask.device,
    )
