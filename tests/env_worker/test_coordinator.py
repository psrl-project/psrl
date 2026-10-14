import pytest
from psrl.workers.env_worker.coordinator import WorkerSlot, select_by_method
from psrl.workers.env_worker.sandbox import SandboxSpec


def _slot(worker_id: int, cpu_slots: int = 4, gpu_slots: int = 0, used_cpu: int = 0, used_gpu: int = 0):
    return WorkerSlot(
        worker_id=worker_id,
        handle=object(),
        cpu_slots=cpu_slots,
        gpu_slots=gpu_slots,
        node_ip=f"10.0.0.{worker_id}",
        used_cpu=used_cpu,
        used_gpu=used_gpu,
    )


@pytest.mark.cpu_test
def test_least_loaded_picks_the_emptiest_worker():
    candidates = [_slot(0, used_cpu=3), _slot(1, used_cpu=1), _slot(2, used_cpu=2)]

    chosen = select_by_method("least_loaded", candidates, SandboxSpec(image="x"))

    assert chosen is not None
    assert chosen.worker_id == 1, "least_loaded must pick the worker with the fewest live sandboxes."


@pytest.mark.cpu_test
def test_round_robin_cycles_across_calls():
    candidates = [_slot(0), _slot(1), _slot(2)]

    picked = [select_by_method("round_robin", candidates, SandboxSpec(image="x")).worker_id for _ in range(6)]

    assert len(set(picked)) == 3, f"round_robin must spread across all workers, got {picked!r}."


@pytest.mark.cpu_test
def test_full_workers_are_never_selected():
    candidates = [_slot(0, cpu_slots=2, used_cpu=2), _slot(1, cpu_slots=2, used_cpu=2)]

    assert select_by_method("least_loaded", candidates, SandboxSpec(image="x")) is None, (
        "A saturated pool must yield no candidate so the caller queues."
    )


@pytest.mark.cpu_test
def test_gpu_request_only_matches_workers_with_free_gpus():
    candidates = [_slot(0, gpu_slots=0), _slot(1, gpu_slots=2)]

    chosen = select_by_method("least_loaded", candidates, SandboxSpec(image="x", gpus=1))

    assert chosen is not None
    assert chosen.worker_id == 1, "A GPU sandbox must land on a worker that owns GPUs."


@pytest.mark.cpu_test
def test_gpu_request_rejected_when_all_gpus_busy():
    candidates = [_slot(0, gpu_slots=1, used_gpu=1)]

    assert select_by_method("least_loaded", candidates, SandboxSpec(image="x", gpus=1)) is None


@pytest.mark.cpu_test
def test_unknown_routing_method_raises():
    with pytest.raises(ValueError, match="Unknown routing method"):
        select_by_method("magic", [_slot(0)], SandboxSpec(image="x"))
