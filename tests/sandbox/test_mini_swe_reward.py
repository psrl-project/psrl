"""Reward compatibility tests for Mini-SWE recipes."""

from examples.mini_swe.reward import compute_score


def _extra_info(*, resolved: bool) -> dict:
    return {
        "num_turns": 3,
        "patch": "diff --git a/a.py b/a.py\n",
        "grader_result": {
            "resolved": resolved,
            "apply_ok": True,
            "f2p_pass": int(resolved),
            "f2p_total": 1,
        },
    }


def test_legacy_binary_remains_signed_for_existing_recipes() -> None:
    failure = compute_score("swe_gym", "", {}, _extra_info(resolved=False), reward_mode="binary")

    assert failure == {"score": -1.0, "acc": 0.0}
