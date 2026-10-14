import pytest
from examples.airs_bench.scripted_episode import SCRIPTED_ACTIONS


@pytest.mark.cpu_test
def test_scripted_actions_end_with_a_submission():
    joined = " ".join(SCRIPTED_ACTIONS)

    assert "submission.csv" in joined, "The scripted sequence must produce a submission."
    assert any(action.startswith("cd ") for action in SCRIPTED_ACTIONS), (
        "The sequence must exercise shell state persistence with a cd."
    )


@pytest.mark.cpu_test
def test_scripted_actions_exercise_state_persistence():
    """A later action relies on the cwd set by an earlier one, which docker exec cannot do."""
    cd_index = next(i for i, action in enumerate(SCRIPTED_ACTIONS) if action.startswith("cd "))
    relative_index = next(i for i, action in enumerate(SCRIPTED_ACTIONS) if action == "ls -la submission.csv")

    assert relative_index > cd_index, (
        "A relative-path action must come after the cd so persistence is actually tested."
    )
