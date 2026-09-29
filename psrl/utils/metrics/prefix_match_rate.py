"""
Compute prefix sharing metrics without materializing a token trie.

Global compression is total sequence length divided by the number of distinct
nonempty prefixes. Sharing is the fraction of tokens saved. Within compression
builds an independent tree per prompt group. Cross compression deduplicates only
prefixes also present outside that group, keeping private tokens per occurrence.
Both group scopes report unweighted mean, population variance, maximum and minimum.
Empty trees have compression 1.0 and sharing 0.0.

Byte sorting fixed-width token encodings keeps every token prefix contiguous.
The tree size is total length minus the sum of adjacent longest common prefixes
(LCPs). One global sort also supplies each group's order and the adjacent LCPs
needed for two linear scans to find prefixes shared outside each group.
"""

from collections.abc import Sequence

import numpy as np


def _token_lcp(
    a: np.ndarray,
    b: np.ndarray,
) -> int:
    """
    Return the longest common prefix length in tokens.
    """
    n = min(len(a), len(b))
    if not n:
        return 0
    equal = a[:n] == b[:n]
    first = int(np.argmin(equal))
    return n if equal[first] else first


def _cross_lcp(
    adjacent: list[int],
    group_ids: list[int],
) -> list[int]:
    """
    Find each sorted sample's longest prefix shared outside its group.

    The LCP with a nonadjacent sample is the minimum adjacent LCP between them.
    Each scan propagates the nearest different group's LCP through a same-group
    run. More distant groups cannot improve it.
    """
    cross = [0] * len(group_ids)
    best = 0
    for i in range(1, len(group_ids)):
        best = adjacent[i - 1] if group_ids[i] != group_ids[i - 1] else min(best, adjacent[i - 1])
        cross[i] = best
    best = 0
    for i in range(len(group_ids) - 2, -1, -1):
        best = adjacent[i] if group_ids[i] != group_ids[i + 1] else min(best, adjacent[i])
        cross[i] = max(cross[i], best)
    return cross


def compute_pmr_metrics(
    sequences: Sequence[np.ndarray],
    group_ids: Sequence[int] | None = None,
) -> dict[str, float | int | None]:
    """
    Compute prefix sharing metrics over the full training batch.

    Args:
        sequences: Per-sample one-dimensional token ID sequences.
        group_ids: Per-sample prompt keys. Missing or mismatched keys disable
            group metrics.

    Returns:
        Global `pmr/compression_ratio`, `pmr/sharing_ratio`,
        `pmr/common_prefix_len` and `pmr/num_samples`, plus
        `pmr/{within,cross}_compression_ratio/{mean,var,max,min}`.
        Group metrics are None when groups are unavailable or the batch is empty.
    """
    if group_ids is not None and len(group_ids) != len(sequences):
        group_ids = None
    seqs = [np.asarray(sequence, dtype=np.uint32) for sequence in sequences]
    if any(s.ndim != 1 for s in seqs):
        raise ValueError("Expected one-dimensional token sequences.")
    keys = [s.tobytes() for s in seqs]
    order = sorted(range(len(seqs)), key=keys.__getitem__)
    seqs = [seqs[i] for i in order]
    adjacent = [_token_lcp(a, b) for a, b in zip(seqs, seqs[1:])]
    total = sum(map(len, seqs))
    saved = sum(adjacent)
    metrics = {
        "pmr/compression_ratio": total / (total - saved) if total else 1.0,
        "pmr/sharing_ratio": saved / total if total else 0.0,
        "pmr/common_prefix_len": min(adjacent) if adjacent else len(seqs[0]) if seqs else 0,
        "pmr/num_samples": len(seqs),
        **{
            f"pmr/{scope}_compression_ratio/{stat}": None
            for scope in ("within", "cross")
            for stat in ("mean", "var", "max", "min")
        },
    }
    if group_ids is None or not seqs:
        return metrics

    sorted_groups = [int(group_ids[i]) for i in order]
    cross = _cross_lcp(adjacent, sorted_groups)
    groups: dict[int, list[int]] = {}
    for i, group in enumerate(sorted_groups):
        groups.setdefault(group, []).append(i)

    within_ratios, cross_ratios = [], []
    for positions in groups.values():
        total = sum(len(seqs[i]) for i in positions)
        within_saved = cross_saved = 0
        for left, right in zip(positions, positions[1:]):
            shared = _token_lcp(seqs[left], seqs[right])
            within_saved += shared
            # Outside-group prefixes form a prefix-closed subtree: truncating
            # to it preserves prefix contiguity in the group's existing order.
            cross_saved += min(shared, cross[left], cross[right])
        within_ratios.append(total / (total - within_saved) if total else 1.0)
        cross_ratios.append(total / (total - cross_saved) if total else 1.0)

    for scope, values in (("within", within_ratios), ("cross", cross_ratios)):
        for stat, reduce in (("mean", np.mean), ("var", np.var), ("max", np.max), ("min", np.min)):
            metrics[f"pmr/{scope}_compression_ratio/{stat}"] = float(reduce(values))
    return metrics


def sequences_from_batch(data) -> tuple[list[np.ndarray], list[int] | None]:
    """Extract per-sample token sequences and group ids from a batch.

    ``data`` is the ``TensorDict`` returned by ``transfer_queue.kv_batch_get``
    for a training batch. ``input_ids`` is a jagged nested tensor (one 1-D
    tensor per sample). ``parent_id`` groups samples sharing a prompt.
    """
    import torch

    def unbind(tensor) -> list[np.ndarray]:
        if isinstance(tensor, (list, tuple)):
            return [np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t) for t in tensor]
        if isinstance(tensor, np.ndarray):
            if tensor.ndim == 1:
                return [tensor]
            raise ValueError(f"Expected 1-D per-sample sequences, got shape {tensor.shape}")
        if hasattr(tensor, "offsets"):
            # Jagged NestedTensor: split the flat values by offsets.
            offsets = tensor.offsets().detach().cpu().numpy()
            flat = tensor.values().detach().cpu().numpy()
            return [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 1:
                return [tensor.detach().cpu().numpy()]
            raise ValueError(f"Expected a jagged nested tensor or 1-D tensor, got shape {tuple(tensor.shape)}")
        raise ValueError(f"Unsupported input representation: {type(tensor)}")

    seqs = unbind(data["input_ids"])

    group_ids: list[int] | None = None
    if data.get("parent_id") is not None:
        parent = data["parent_id"]
        if hasattr(parent, "tolist"):
            group_ids = [int(x) for x in parent.tolist()]
        else:
            group_ids = [int(x) for x in parent]
        if len(group_ids) != len(seqs):
            group_ids = None

    return seqs, group_ids
