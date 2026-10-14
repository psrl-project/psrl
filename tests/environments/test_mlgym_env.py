import pytest
import yaml
from examples.airs_bench.config import build_runtime_config
from psrl.environments.mlgym_env import (
    assert_default_history_processor,
    build_sandbox_spec,
    sync_exec,
)


class FakeHandle:
    """Record exec calls without a real container."""

    def __init__(self, responses=None):
        self.calls: list[tuple[str, float]] = []
        self.responses = responses or {}
        self.destroyed = False

    async def exec(self, command, timeout_s, no_output_timeout_s=None):
        from psrl.workers.env_worker.sandbox import ExecResult

        self.calls.append((command, timeout_s))
        payload = self.responses.get(command, "")
        return ExecResult(stdout=payload, exit_code=0, timed_out=False, duration_s=0.01)

    async def destroy(self):
        self.destroyed = True


@pytest.mark.cpu_test
def test_runtime_config_defaults_are_sane():
    config = build_runtime_config(None)

    assert config.image, "A default sandbox image must be set."
    assert config.per_action_timeout_s >= 3600.0, "Per-action timeout must cover MLGym's 3600s training_timeout."
    assert config.max_observation_chars > 0, "Observation truncation must be on by default."


@pytest.mark.cpu_test
def test_runtime_config_merges_overrides():
    config = build_runtime_config({"sandbox_memory": "64g", "sandbox_cpus": 16.0})

    assert config.sandbox_memory == "64g"
    assert config.sandbox_cpus == 16.0


@pytest.mark.cpu_test
def test_build_sandbox_spec_mounts_prepared_task_data_read_only():
    config = build_runtime_config({"sandbox_memory": "32g"})
    spec = build_sandbox_spec(
        config,
        dataset_data_path="/shared/airs_prepared/GraphRegressionZincMae",
        labels={"psrl.airs_task_id": "t1"},
    )

    assert spec.memory == "32g"
    assert spec.gpus == 0, "AIRS-Bench tasks are CPU graded."
    mount_modes = {mode for _, _, mode in spec.mounts}
    assert "ro" in mount_modes, "Task data must be mounted read only so labels cannot be edited."
    assert any("airs_prepared" in host for host, _, _ in spec.mounts)
    assert spec.labels["psrl.airs_task_id"] == "t1"


@pytest.mark.cpu_test
def test_assert_default_history_processor_accepts_the_correct_value(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"history_processor": "DefaultHistoryProcessor"}))

    assert_default_history_processor(str(path))  # must not raise


@pytest.mark.cpu_test
def test_assert_default_history_processor_rejects_last5(tmp_path):
    """Last5Observations rewrites history in place, which corrupts TITO data."""
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"history_processor": "Last5Observations"}))

    with pytest.raises(ValueError, match="DefaultHistoryProcessor"):
        assert_default_history_processor(str(path))


@pytest.mark.cpu_test
def test_assert_default_history_processor_rejects_a_missing_key(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({"system_template": "hello"}))

    with pytest.raises(ValueError, match="history_processor"):
        assert_default_history_processor(str(path))


@pytest.mark.cpu_test
def test_shipped_agent_config_pins_default_history_processor():
    """Guard the real config file, not just the checker."""
    assert_default_history_processor("examples/airs_bench/agent_config.yaml")


@pytest.mark.cpu_test
def test_sync_exec_bridges_async_handle_to_blocking_caller():
    import asyncio

    handle = FakeHandle(responses={"echo hi": "hi\n"})
    loop = asyncio.new_event_loop()
    thread_loop = _start_loop_in_thread(loop)
    try:
        result = sync_exec(handle, loop, "echo hi", timeout_s=5.0)
        assert result.stdout == "hi\n"
        assert handle.calls == [("echo hi", 5.0)]
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread_loop.join(timeout=5)


def _start_loop_in_thread(loop):
    import threading

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    return thread
