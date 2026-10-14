import asyncio

import pytest
from psrl.workers.env_worker.sandbox import SandboxSpec
from psrl.workers.env_worker.worker import SANDBOX_LABEL_KEY, EnvWorker


class FakeShell:
    """Stand in for a container shell so exec logic is testable without Docker."""

    def __init__(self, scripted_outputs: list[str]):
        self.scripted_outputs = scripted_outputs
        self.written: list[str] = []
        self.drain_calls = 0
        self.closed = False

    def write(self, text: str) -> None:
        self.written.append(text)

    def drain(self) -> str:
        self.drain_calls += 1
        return ""

    async def read_until(self, deadline: float, no_output_timeout_s: float) -> str:
        if not self.scripted_outputs:
            await asyncio.sleep(0.01)
            return ""
        return self.scripted_outputs.pop(0)

    def close(self) -> None:
        self.closed = True


@pytest.mark.cpu_test
def test_gpu_slot_indices_come_from_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,6,7")
    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=4, exec_default_timeout_s=60.0)

    assert worker.available_gpu_indices == (2, 3, 6, 7), (
        f"Worker must only ever expose Ray-assigned GPUs, got {worker.available_gpu_indices!r}."
    )


@pytest.mark.cpu_test
def test_worker_without_cuda_visible_devices_has_no_gpus(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=0, exec_default_timeout_s=60.0)

    assert worker.available_gpu_indices == ()


@pytest.mark.cpu_test
def test_allocate_gpu_indices_is_exclusive(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=2, exec_default_timeout_s=60.0)

    first = worker._allocate_gpu_indices(1)
    second = worker._allocate_gpu_indices(1)

    assert set(first).isdisjoint(set(second)), "Two sandboxes must never share a GPU index."


@pytest.mark.cpu_test
def test_allocate_gpu_indices_raises_when_exhausted(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=1, exec_default_timeout_s=60.0)

    worker._allocate_gpu_indices(1)
    with pytest.raises(RuntimeError, match="no free GPU"):
        worker._allocate_gpu_indices(1)


@pytest.mark.cpu_test
def test_sandbox_labels_always_include_the_cleanup_key(monkeypatch):
    monkeypatch.setenv("PSRL_ACTOR_ID", "actor-7")
    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=0, exec_default_timeout_s=60.0)

    labels = worker._build_labels(SandboxSpec(image="x"), sandbox_id="s-1")

    assert labels[SANDBOX_LABEL_KEY] == "s-1", "Sandbox id label is required for reaping."
    assert labels["psrl.actor_id"] == "actor-7", "Actor id label is required for reaper cleanup."


@pytest.mark.cpu_test
def test_exec_on_unknown_sandbox_raises():
    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=0, exec_default_timeout_s=60.0)

    with pytest.raises(KeyError, match="unknown-id"):
        asyncio.run(worker.exec("unknown-id", "ls", timeout_s=5.0))


@pytest.mark.cpu_test
def test_exec_parses_sentinel_and_returns_exit_code():
    from psrl.workers.env_worker.shell import PROCESS_DONE_MARKER_END, PROCESS_DONE_MARKER_START

    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=0, exec_default_timeout_s=60.0)
    output = f"hi\n{PROCESS_DONE_MARKER_START}0{PROCESS_DONE_MARKER_END}\n"
    worker._shells["s-1"] = FakeShell([output])

    result = asyncio.run(worker.exec("s-1", "echo hi", timeout_s=5.0))

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.strip() == "hi"
    assert result.duration_s >= 0.0


@pytest.mark.cpu_test
def test_exec_reports_timeout_without_killing_the_sandbox():
    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=0, exec_default_timeout_s=60.0)
    worker._shells["s-1"] = FakeShell([])  # never emits a sentinel

    result = asyncio.run(worker.exec("s-1", "sleep 100", timeout_s=0.2))

    assert result.timed_out is True, "A command that never completes must report a timeout."
    assert "s-1" in worker._shells, "Timeout must leave the sandbox alive, matching MLGym semantics."


@pytest.mark.cpu_test
def test_exec_drains_stale_output_before_issuing_a_command():
    """
    A timed-out command keeps running and later emits output plus a stale sentinel.
    Without a drain, that leaks into the next observation and its exit code.
    """
    from psrl.workers.env_worker.shell import PROCESS_DONE_MARKER_END, PROCESS_DONE_MARKER_START

    worker = EnvWorker(worker_id=0, cpu_slots=4, gpu_slots=0, exec_default_timeout_s=60.0)
    shell = FakeShell([f"second\n{PROCESS_DONE_MARKER_START}0{PROCESS_DONE_MARKER_END}\n"])
    worker._shells["s-1"] = shell

    asyncio.run(worker.exec("s-1", "echo second", timeout_s=5.0))

    assert shell.drain_calls == 1, "Every exec must drain stale output first."


@pytest.mark.cpu_test
def test_exec_truncates_oversized_observations():
    from psrl.workers.env_worker.shell import PROCESS_DONE_MARKER_END, PROCESS_DONE_MARKER_START

    worker = EnvWorker(
        worker_id=0,
        cpu_slots=4,
        gpu_slots=0,
        exec_default_timeout_s=60.0,
        max_observation_chars=50,
    )
    flood = "y" * 10000
    output = f"{flood}\n{PROCESS_DONE_MARKER_START}0{PROCESS_DONE_MARKER_END}\n"
    worker._shells["s-1"] = FakeShell([output])

    result = asyncio.run(worker.exec("s-1", "cat big", timeout_s=5.0))

    assert len(result.stdout) < 500, (
        f"Observation length after truncation={len(result.stdout)}. Expected fewer than 500 characters."
    )
    assert "omitted" in result.stdout
