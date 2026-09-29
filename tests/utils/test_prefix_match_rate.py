# tests/utils/test_prefix_match_rate.py
"""Unit tests for psrl.utils.metrics.prefix_match_rate (AReaL-DTA C/S metrics)."""

import numpy as np
import pytest
from psrl.utils.metrics.prefix_match_rate import compute_pmr_metrics, sequences_from_batch

pytestmark = pytest.mark.cpu_test


def _distinct_prefixes(seqs):
    """Brute force: set of all non-empty prefixes over the sequences."""
    out = set()
    for s in seqs:
        for k in range(1, len(s) + 1):
            out.add(tuple(int(x) for x in s[:k]))
    return out


class TestGlobalCompression:
    def test_no_sharing(self):
        seqs = [np.array([1, 2]), np.array([3, 4])]
        m = compute_pmr_metrics(seqs)
        assert m["pmr/compression_ratio"] == pytest.approx(1.0)
        assert m["pmr/sharing_ratio"] == pytest.approx(0.0)

    def test_identical(self):
        seqs = [np.array([1, 2])] * 3
        m = compute_pmr_metrics(seqs)
        assert m["pmr/compression_ratio"] == pytest.approx(3.0)
        assert m["pmr/sharing_ratio"] == pytest.approx(2 / 3)

    def test_partial_sharing(self):
        # [1,2,3],[1,2,4],[5,6] -> tree keeps [1,2,3,4,5,6] => 6 tree tokens / 8 raw
        seqs = [np.array([1, 2, 3]), np.array([1, 2, 4]), np.array([5, 6])]
        m = compute_pmr_metrics(seqs)
        assert m["pmr/compression_ratio"] == pytest.approx(8 / 6)
        assert m["pmr/sharing_ratio"] == pytest.approx(1 - 6 / 8)
        assert m["pmr/common_prefix_len"] == 0

    def test_common_prefix(self):
        seqs = [np.array([1, 2, 3]), np.array([1, 2, 4]), np.array([1, 9])]
        m = compute_pmr_metrics(seqs)
        assert m["pmr/common_prefix_len"] == 1

    def test_tree_tokens_matches_bruteforce(self):
        rng = np.random.default_rng(5)
        for _ in range(30):
            n = int(rng.integers(1, 12))
            seqs = [rng.integers(0, 6, size=int(rng.integers(1, 12))).tolist() for _ in range(n)]
            m = compute_pmr_metrics(seqs)
            assert m["pmr/compression_ratio"] == pytest.approx(sum(map(len, seqs)) / len(_distinct_prefixes(seqs)))
            assert m["pmr/sharing_ratio"] == pytest.approx(1 - 1 / m["pmr/compression_ratio"])


class TestWithinCross:
    def test_within_cross_sharing(self):
        # g0: [7,7,1,2], [7,7,1,3], g1: [7,7,9,9] (SP=[7,7] shared across groups)
        seqs = [np.array([7, 7, 1, 2]), np.array([7, 7, 1, 3]), np.array([7, 7, 9, 9])]
        groups = [0, 0, 1]
        m = compute_pmr_metrics(seqs, group_ids=groups)
        # global: raw 12, sorted LCPs 3+2 -> tree 7
        assert m["pmr/compression_ratio"] == pytest.approx(12 / 7)
        assert m["pmr/common_prefix_len"] == 2
        # within: g0 -> 8 raw, LCP 3 -> tree 5 (C=1.6, S=0.375), g1 -> C=1
        assert m["pmr/within_compression_ratio/mean"] == pytest.approx((8 / 5 + 1) / 2)
        assert m["pmr/within_compression_ratio/min"] == pytest.approx(1.0)
        assert m["pmr/within_compression_ratio/max"] == pytest.approx(8 / 5)
        # cross: g0 tails [1,2],[1,3] (adj LCP 1) -> n_tree_cross = 5 + 1 = 6 -> C=8/6
        assert m["pmr/cross_compression_ratio/mean"] == pytest.approx((8 / 6 + 1) / 2)

    def test_no_cross_sharing(self):
        # distinct prompts with no common prefix -> cross C/S = 1 / 0 for every group
        seqs = [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5]), np.array([9, 9, 9])]
        groups = [0, 0, 1]
        m = compute_pmr_metrics(seqs, group_ids=groups)
        assert m["pmr/cross_compression_ratio/max"] == pytest.approx(1.0)

    def test_single_sample_groups(self):
        # each group has one sample -> within C = 1 everywhere
        seqs = [np.array([1, 2, 3]), np.array([1, 2, 9])]
        groups = [0, 1]
        m = compute_pmr_metrics(seqs, group_ids=groups)
        assert m["pmr/within_compression_ratio/max"] == pytest.approx(1.0)
        assert m["pmr/cross_compression_ratio/max"] == pytest.approx(1.0)


class TestGroupInvariants:
    def test_within_tree_matches_bruteforce(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            n_groups = int(rng.integers(2, 6))
            seqs, groups = [], []
            for g in range(n_groups):
                for _ in range(int(rng.integers(1, 4))):
                    seqs.append(rng.integers(0, 8, size=int(rng.integers(1, 10))).tolist())
                    groups.append(g)
            m = compute_pmr_metrics(seqs, group_ids=groups)
            # aggregate within tree tokens per group == distinct prefixes per group
            pref_by_group = {}
            for s, g in zip(seqs, groups):
                for k in range(1, len(s) + 1):
                    pref_by_group.setdefault(g, set()).add(tuple(int(x) for x in s[:k]))
            raw_by_group = {}
            for s, g in zip(seqs, groups):
                raw_by_group[g] = raw_by_group.get(g, 0) + len(s)
            # recompute per-group C from brute force and compare aggregates
            c_vals = [raw_by_group[g] / len(p) for g, p in pref_by_group.items()]
            assert m["pmr/within_compression_ratio/mean"] == pytest.approx(float(np.mean(c_vals)))
            assert m["pmr/within_compression_ratio/min"] == pytest.approx(float(np.min(c_vals)))
            # global C / S in range and consistent
            assert m["pmr/compression_ratio"] >= 1.0
            assert 0.0 <= m["pmr/sharing_ratio"] < 1.0
            for scope in ("within", "cross"):
                assert m[f"pmr/{scope}_compression_ratio/mean"] >= 1.0


def _brute_cross_c(seqs, groups):
    """Reference per-group cross C from first principles (small inputs)."""
    n = len(seqs)
    cm = []
    for i in range(n):
        best = 0
        for j in range(n):
            if groups[j] == groups[i]:
                continue
            k = 0
            while k < len(seqs[i]) and k < len(seqs[j]) and seqs[i][k] == seqs[j][k]:
                k += 1
            best = max(best, k)
        cm.append(best)
    group_ids = sorted(set(groups))
    out = {}
    for g in group_ids:
        idxs = [i for i in range(n) if groups[i] == g]
        n_token_g = sum(len(seqs[i]) for i in idxs)
        # distinct cross-shared prefixes of g
        cross_prefixes = set()
        for i in idxs:
            for k in range(1, cm[i] + 1):
                cross_prefixes.add(tuple(int(x) for x in seqs[i][:k]))
        n_tree_cross = len(cross_prefixes) + sum(len(seqs[i]) - cm[i] for i in idxs)
        out[g] = n_token_g / n_tree_cross if n_token_g else 1.0
    return out


class TestCrossBruteforce:
    def test_cross_matches_bruteforce(self):
        rng = np.random.default_rng(23)
        for _ in range(40):
            n_groups = int(rng.integers(2, 6))
            seqs, groups = [], []
            for g in range(n_groups):
                for _ in range(int(rng.integers(1, 4))):
                    seqs.append(rng.integers(0, 8, size=int(rng.integers(1, 10))).tolist())
                    groups.append(g)
            m = compute_pmr_metrics(seqs, group_ids=groups)
            brute = _brute_cross_c(seqs, groups)
            vals = list(brute.values())
            assert m["pmr/cross_compression_ratio/mean"] == pytest.approx(float(np.mean(vals))), (
                f"cross mean mismatch on {seqs} groups={groups}"
            )
            assert m["pmr/cross_compression_ratio/max"] == pytest.approx(float(np.max(vals)))
            assert m["pmr/cross_compression_ratio/min"] == pytest.approx(float(np.min(vals)))


class TestFullBatch:
    def test_all_samples_contribute(self):
        seqs = [[1, 2, 3]] * 49 + [[4, 5, 6]]
        metrics = compute_pmr_metrics(seqs, group_ids=[0] * 49 + [1])
        assert metrics["pmr/num_samples"] == 50, "All batch samples must be counted."
        assert metrics["pmr/compression_ratio"] == pytest.approx(150 / 6), "All prefixes must be included."
        assert metrics["pmr/common_prefix_len"] == 0, "The final sample breaks the common prefix."
        assert metrics["pmr/within_compression_ratio/mean"] == 25, "Both groups must be included."
        assert metrics["pmr/cross_compression_ratio/mean"] == 1, "The groups share no prefixes."


class TestDegenerate:
    def test_empty(self):
        m = compute_pmr_metrics([])
        assert m["pmr/num_samples"] == 0
        assert m["pmr/compression_ratio"] == 1.0
        assert m["pmr/sharing_ratio"] == 0.0

    def test_single(self):
        m = compute_pmr_metrics([np.array([1, 2, 3])])
        assert m["pmr/compression_ratio"] == pytest.approx(1.0)
        assert m["pmr/sharing_ratio"] == pytest.approx(0.0)
        assert m["pmr/common_prefix_len"] == 3


class TestSequencesFromBatch:
    def test_jagged_nested_tensor(self):
        import torch

        input_ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
        data = {"input_ids": input_ids, "parent_id": [10, 10]}
        seqs, groups = sequences_from_batch(data)
        assert [s.tolist() for s in seqs] == [[1, 2, 3], [4, 5]]
        assert groups == [10, 10]

    def test_list_of_tensors_no_group(self):
        import torch

        data = {"input_ids": [torch.tensor([1, 2]), torch.tensor([3])], "parent_id": None}
        seqs, groups = sequences_from_batch(data)
        assert [s.tolist() for s in seqs] == [[1, 2], [3]]
        assert groups is None


@pytest.mark.parametrize("groups", [None, [], [0, 1]])
def test_metric_keys(groups):
    sequences = [[], []] if groups else []
    metrics = compute_pmr_metrics(sequences, groups)
    expected = {f"pmr/{name}" for name in ("compression_ratio", "sharing_ratio", "common_prefix_len", "num_samples")}
    expected.update(
        f"pmr/{scope}_compression_ratio/{stat}"
        for scope in ("within", "cross")
        for stat in ("mean", "var", "max", "min")
    )
    assert set(metrics) == expected, "Only the requested metrics should be returned."


@pytest.mark.parametrize("seed", range(8))
def test_all_metrics_against_prefix_sets(seed):
    rng = np.random.default_rng(seed)
    for _ in range(100):
        # Include empty, duplicate, nested and byte-order-sensitive sequences.
        pool = [0, 1, 256, 65536, 16777216, 2**32 - 1]
        sequences = [rng.choice(pool, size=int(rng.integers(0, 9))).tolist() for _ in range(16)]
        sequences[1] = sequences[0][:]
        sequences[2] = sequences[0][:2]
        groups = rng.integers(-2, 3, size=len(sequences)).tolist()
        metrics = compute_pmr_metrics(sequences, groups)
        total = sum(map(len, sequences))
        tree = len(_distinct_prefixes(sequences))
        assert metrics["pmr/compression_ratio"] == pytest.approx(total / tree if total else 1.0), (
            "Global compression differs."
        )
        assert metrics["pmr/sharing_ratio"] == pytest.approx(1 - tree / total if total else 0.0), (
            "Global sharing differs."
        )
        common = 0
        for tokens in zip(*sequences):
            if len(set(tokens)) != 1:
                break
            common += 1
        assert metrics["pmr/common_prefix_len"] == common, "Common prefix must match all samples."
        within = []
        for group in set(groups):
            selected = [s for s, g in zip(sequences, groups) if g == group]
            raw = sum(map(len, selected))
            within.append(raw / len(_distinct_prefixes(selected)) if raw else 1.0)
        cross = list(_brute_cross_c(sequences, groups).values())
        for scope, values in (("within", within), ("cross", cross)):
            for stat, reduce in (("mean", np.mean), ("var", np.var), ("max", np.max), ("min", np.min)):
                assert metrics[f"pmr/{scope}_compression_ratio/{stat}"] == pytest.approx(reduce(values)), (
                    f"Incorrect {scope} compression {stat}."
                )


@pytest.mark.parametrize("sequences,groups", [([[], []], [0, 1]), ([[], [1, 2]], [0, 1]), ([[1], [1]], [0, 0])])
def test_empty_and_single_group(sequences, groups):
    metrics = compute_pmr_metrics(sequences, groups)
    assert metrics["pmr/cross_compression_ratio/mean"] == 1.0, "No outside sharing should give unit compression."
    assert all(value is not None and np.isfinite(value) for value in metrics.values()), "Metrics must be finite."


def test_mismatched_groups_disable_group_metrics():
    actual = compute_pmr_metrics([[1], [2], [3]], [0])
    expected = compute_pmr_metrics([[1], [2], [3]])
    assert actual == expected, "Invalid group lengths must disable group metrics."


def test_reject_multidimensional_sequence():
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_pmr_metrics([np.array([[1, 2], [3, 4]])])
