"""Tests for the episode reward source and the graded-ratio gate."""

from __future__ import annotations

import pytest
from psrl.trainer.ppo.utils import graded_ratio
from psrl.workers.agent_loop.loops.base_agent_loop import reward_source_index


def test_the_primary_row_is_the_reward_source():
    assert reward_source_index(3, 1) == 1


def test_the_legacy_call_still_reads_the_last_row():
    assert reward_source_index(3, None) == 2


def test_an_out_of_range_primary_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        reward_source_index(2, 2)
    with pytest.raises(ValueError, match="at least one trajectory"):
        reward_source_index(0, None)


def test_graded_ratio_counts_only_measured_rows():
    reasons = ["finished", "verifier_error", "grader_capacity_timeout", "agent_timeout"]
    assert graded_ratio(reasons) == 0.5


def test_graded_ratio_of_an_empty_batch_is_zero():
    assert graded_ratio([]) == 0.0


def test_an_unknown_reason_is_treated_as_measured():
    assert graded_ratio(["finished", "not_a_reason"]) == 1.0
