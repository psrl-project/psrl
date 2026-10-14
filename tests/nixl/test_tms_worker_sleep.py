import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_test


_PLUGIN_ROOT = Path(__file__).resolve().parents[2] / "patch" / "vllm" / "vllm_plugin"
sys.path.insert(0, str(_PLUGIN_ROOT))

pytest.importorskip("vllm")

gpu_worker = importlib.import_module("vllm_patches.patches.gpu_worker")


def test_release_inactive_cuda_cache_runs_in_current_process(monkeypatch) -> None:
    events = []
    reserved = iter([10_000, 4_000])
    allocated = iter([3_000, 3_000])

    monkeypatch.setattr(gpu_worker.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(gpu_worker.torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(gpu_worker.torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(gpu_worker.torch.cuda, "memory_reserved", lambda *args: next(reserved))
    monkeypatch.setattr(gpu_worker.torch.cuda, "memory_allocated", lambda *args: next(allocated))

    stats = gpu_worker._release_inactive_cuda_cache()

    assert events == ["gc", "empty_cache", "synchronize"]
    assert stats == {
        "reserved_before": 10_000,
        "reserved_after": 4_000,
        "reserved_freed": 6_000,
        "allocated_before": 3_000,
        "allocated_after": 3_000,
    }


@pytest.mark.parametrize(
    ("patches", "expected_pauses"),
    [
        ("TMS", ["weights", "kv_cache"]),
        ("TMS:GRAPH", ["weights", "kv_cache", "graph"]),
    ],
)
def test_sleep_clears_default_allocator_after_tms_pools(
    monkeypatch,
    patches: str,
    expected_pauses: list[str],
) -> None:
    events = []

    class _MemorySaver:
        @staticmethod
        def pause(tag: str) -> None:
            events.append(("pause", tag))

    monkeypatch.setitem(
        sys.modules,
        "torch_memory_saver",
        SimpleNamespace(torch_memory_saver=_MemorySaver),
    )
    monkeypatch.setenv("PSRL_VLLM_PATCHES", patches)
    monkeypatch.setattr(
        gpu_worker,
        "_release_inactive_cuda_cache",
        lambda: (
            events.append(("empty_cache", "default_allocator"))
            or {
                "reserved_before": 8_000,
                "reserved_after": 2_000,
                "reserved_freed": 6_000,
                "allocated_before": 1_000,
                "allocated_after": 1_000,
            }
        ),
    )

    memory_info = iter([(10_000, 100_000), (20_000, 100_000)])
    monkeypatch.setattr(gpu_worker.torch.cuda, "mem_get_info", lambda: next(memory_info))
    monkeypatch.setattr(gpu_worker.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_worker.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(uuid="gpu-0", name="test-gpu"),
    )
    monkeypatch.setattr(gpu_worker.torch.cuda, "memory_allocated", lambda *args: 1_000)
    monkeypatch.setattr(gpu_worker.torch.cuda, "memory_reserved", lambda *args: 2_000)

    timings = []
    worker = SimpleNamespace(
        rank=0,
        local_rank=0,
        model_runner=SimpleNamespace(model=SimpleNamespace(named_buffers=lambda: [])),
        log_tms_timing=lambda operation, stage, elapsed_s, tag=None: timings.append((operation, stage, tag)),
    )

    gpu_worker.TMSWorkerPatch.sleep(worker, level=2)

    assert events == [
        *(("pause", tag) for tag in expected_pauses),
        ("empty_cache", "default_allocator"),
    ]
    assert ("sleep", "empty_cache", "default_allocator") in timings
    assert timings[-1] == ("sleep", "total", None)
