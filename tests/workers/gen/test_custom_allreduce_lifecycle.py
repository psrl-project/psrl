import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

VLLM_ROOT = Path(__file__).resolve().parents[3] / "third_party/vllm/vllm"


def load_function(relative_path, name, namespace, class_name=None):
    tree = ast.parse((VLLM_ROOT / relative_path).read_text())
    if class_name is not None:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(relative_path), "exec"), namespace)
    return namespace[name]


@pytest.mark.cpu_test
def test_reset_preserves_shared_buffers_and_releases_peer_mappings():
    ops = Mock()
    ops.init_custom_ar.side_effect = range(2, 66)
    dist = Mock()
    events = Mock()
    events.attach_mock(dist.barrier, "barrier")
    events.attach_mock(ops.dispose, "dispose")
    events.attach_mock(ops.init_custom_ar, "init")
    events.attach_mock(ops.register_buffer, "register")
    reset = load_function(
        "distributed/device_communicators/custom_all_reduce.py",
        "reset_graph_buffers",
        {"ops": ops, "dist": dist},
        "CustomAllreduce",
    )
    communicator = SimpleNamespace(
        disabled=False,
        _IS_CAPTURING=False,
        _ptr=1,
        group=object(),
        meta_ptrs=[11, 12],
        rank_data=object(),
        rank=0,
        fully_connected=True,
        buffer_ptrs=[21, 22],
    )
    expected = []
    for pointer in range(1, 65):
        reset(communicator)
        expected.extend(
            [
                call.barrier(group=communicator.group),
                call.dispose(pointer),
                call.init(communicator.meta_ptrs, communicator.rank_data, 0, True),
                call.register(pointer + 1, communicator.buffer_ptrs),
            ]
        )
    assert events.mock_calls == expected, "Reset must reclaim IPC mappings before reusing registration storage."

    events.reset_mock()
    communicator.disabled = True
    reset(communicator)
    assert not events.mock_calls, "Disabled communicators must not allocate or enter a barrier."
    communicator.disabled = False
    communicator._IS_CAPTURING = True
    with pytest.raises(AssertionError, match="during graph capture"):
        reset(communicator)
    assert not events.mock_calls, "Active captures must retain their registration storage."


@pytest.mark.cpu_test
def test_reset_visits_live_communicators_once_in_group_order():
    first, second = Mock(), Mock()
    events = Mock()
    events.attach_mock(first.reset_graph_buffers, "first")
    events.attach_mock(second.reset_graph_buffers, "second")
    groups = [
        None,
        SimpleNamespace(device_communicator=None),
        SimpleNamespace(device_communicator=SimpleNamespace()),
        SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=first)),
        SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=first)),
        SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=second)),
    ]
    reset = load_function(
        "distributed/parallel_state.py",
        "reset_cudagraph_communication_buffers",
        {"_groups": {str(index): lambda group=group: group for index, group in enumerate(groups)}},
    )
    reset()
    assert events.mock_calls == [call.first(), call.second()], (
        "Collective reset order must match group creation order."
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("released", [True, False])
def test_sleep_resets_registrations_after_graph_release(monkeypatch, released):
    events = Mock()
    events.release.return_value = released
    allocator = SimpleNamespace(sleep=events.sleep)
    monkeypatch.setitem(
        sys.modules,
        "vllm.device_allocator.cumem",
        SimpleNamespace(CuMemAllocator=SimpleNamespace(get_instance=lambda: allocator)),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators",
        SimpleNamespace(nccl_suspend=SimpleNamespace(suspend_nccl_comms=events.suspend)),
    )
    memory_info = iter([(0, 100), (50, 100)])
    sleep = load_function(
        "v1/worker/gpu_worker.py",
        "sleep",
        {
            "torch": SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: next(memory_info))),
            "reset_cudagraph_communication_buffers": events.reset,
            "logger": Mock(),
            "format_gib": str,
        },
        "Worker",
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(release_cudagraphs=events.release), _cudagraphs_released=True
    )
    sleep(worker)
    expected = [call.release()]
    if released:
        expected.append(call.reset())
    expected.extend([call.sleep(offload_tags=("weights",)), call.suspend()])
    assert events.mock_calls == expected, (
        "Graph registrations must be reset after release and before memory unmapping."
    )
    assert worker._cudagraphs_released, "A repeated sleep must preserve pending graph recapture."


def run_graph_registration_cycles(rank, init_method):
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    communicator = CustomAllreduce(dist.group.WORLD, rank)
    assert not communicator.disabled, "GPU regression requires custom allreduce support."
    # Two records fit one eager buffer and one graph, making accumulation fail on the second capture.
    original_rank_data = communicator.rank_data
    communicator.rank_data = torch.empty(128, dtype=torch.uint8, device=rank)
    communicator.reset_graph_buffers()
    del original_rank_data
    source = torch.full((4096,), rank + 1.0, device=rank)
    expected = torch.full_like(source, 3.0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def capture_once():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), communicator.capture():
            communicator.custom_all_reduce(source)
            with torch.cuda.graph(graph, stream=stream):
                output = communicator.custom_all_reduce(source)
        return graph, output

    try:
        graph, output = capture_once()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected)
        graph.reset()
        del graph, output
        with pytest.raises(RuntimeError, match="Rank data buffer is overflowed"):
            capture_once()
        communicator.reset_graph_buffers()
        for _ in range(64):
            graph, output = capture_once()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected)
            graph.reset()
            del graph, output
            communicator.reset_graph_buffers()
            torch.testing.assert_close(communicator.custom_all_reduce(source), expected)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        communicator.close()
        dist.destroy_process_group()


@pytest.mark.gpu
def test_graph_registration_storage_is_bounded(tmp_path):
    import torch
    import torch.multiprocessing as mp

    if torch.cuda.device_count() < 2:
        pytest.skip("Two GPUs are required.")
    mp.spawn(run_graph_registration_cycles, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
