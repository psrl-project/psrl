"""Tests for the reward mapping."""

from __future__ import annotations

from examples.math_rl.reward import compute_score


def test_a_graded_verdict_carries_its_mapped_score():
    result = compute_score(
        "math_rl",
        "",
        "",
        {"verifier_verdict": "6", "verifier_graded": True, "reward_score": 0.8},
    )
    assert result["score"] == 0.8
    assert result["acc"] == 0.8
    assert result["verifier_verdict"] == "6"
    assert result["verifier_error"] == ""
    assert result["grading_failed"] == 0.0
    assert result["gold_ceiling"] == 1.0


def test_a_full_verdict_is_one():
    result = compute_score(
        "math_rl",
        "",
        "",
        {"verifier_verdict": "7", "verifier_graded": True, "reward_score": 1.0},
    )
    assert result["score"] == 1.0


def test_a_verifier_error_is_reported_without_a_fake_zero():
    result = compute_score(
        "math_rl",
        "",
        "",
        {"verifier_error": "timeout", "verifier_graded": False, "reward_score": 0.0},
    )
    assert result["score"] == 0.0
    assert result["verifier_verdict"] == "error"
    assert result["verifier_error"] == "timeout"
    assert result["grading_failed"] == 1.0


def test_an_ungraded_episode_does_not_train_its_mapped_score():
    result = compute_score(
        "math_rl",
        "",
        "",
        {"verifier_verdict": "1", "verifier_graded": False, "reward_score": 0.3},
    )
    assert result["score"] == 0.0
    assert result["verifier_verdict"] == "1"
    assert result["verifier_error"] == "ungraded"


def test_score_is_clamped():
    result = compute_score(
        "math_rl",
        "",
        "",
        {"verifier_verdict": "7", "verifier_graded": True, "reward_score": 3.0},
    )
    assert result["score"] == 1.0
