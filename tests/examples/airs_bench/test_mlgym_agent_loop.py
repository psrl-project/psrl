import pytest
from psrl.workers.agent_loop.loops.mlgym_agent_loop import classify_exit_status
from psrl.workers.agent_loop.loops.utils import TerminateReason


@pytest.mark.cpu_test
def test_submitted_maps_to_finished():
    assert classify_exit_status("submitted") is TerminateReason.FINISHED


@pytest.mark.cpu_test
def test_context_overflow_maps_to_max_response_length():
    assert classify_exit_status("exit_context") is TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED


@pytest.mark.cpu_test
def test_max_steps_maps_to_max_turns():
    assert classify_exit_status("max_steps") is TerminateReason.MAX_TURNS_EXCEEDED


@pytest.mark.cpu_test
def test_error_statuses_map_to_rollout_error():
    for status in ("exit_error", "exit_api", "exit_format"):
        assert classify_exit_status(status) is TerminateReason.ROLLOUT_ERROR, (
            f"Unexpected classification for status={status!r}. Expected an infrastructure error."
        )


@pytest.mark.cpu_test
def test_unknown_status_maps_to_unknown():
    assert classify_exit_status("something_new") is TerminateReason.UNKNOWN


@pytest.mark.cpu_test
def test_agent_loop_is_registered_under_mlgym_agent():
    from psrl.workers.agent_loop.loops.utils import AGENT_LOOP_REGISTRY

    assert "mlgym_agent" in AGENT_LOOP_REGISTRY, "The loop must be registered for config lookup."
    assert "MLGymAgentLoop" in AGENT_LOOP_REGISTRY["mlgym_agent"]["_target_"]
