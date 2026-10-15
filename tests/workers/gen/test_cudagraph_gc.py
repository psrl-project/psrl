import ast
import gc
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest

pytestmark = pytest.mark.cpu_test


@pytest.fixture(
    params=[
        ("third_party/vllm/vllm/compilation/cuda_graph.py", "CUDAGraphWrapper"),
        ("patch/vllm/vllm_plugin/vllm_patches/patches/cuda_graph.py", "TMSCUDAGraphWrapperPatch"),
    ],
    ids=["vllm", "tms"],
)
def capture_wrapper(request):
    """
    Execute the production capture method with GPU operations replaced by CPU stubs.
    """
    relative_path, class_name = request.param
    path = Path(__file__).resolve().parents[3] / relative_path
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    mode = "piecewise"
    namespace = {
        "Any": Any,
        "ExitStack": ExitStack,
        "patch": patch,
        "is_forward_context_available": lambda: True,
        "get_forward_context": lambda: SimpleNamespace(batch_descriptor="batch", cudagraph_runtime_mode=mode),
        "CUDAGraphMode": SimpleNamespace(NONE="none"),
        "CUDAGraphEntry": lambda **kwargs: SimpleNamespace(cudagraph=None, **kwargs),
        "validate_cudagraph_capturing_enabled": lambda: None,
        "set_graph_pool_id": lambda pool: None,
        "get_offloader": Mock,
        "current_stream": lambda: None,
        "weak_ref_tensors": lambda value: value,
        "compilation_counter": SimpleNamespace(num_cudagraph_captured=0),
        "torch": SimpleNamespace(
            Tensor=type("Tensor", (), {}),
            cuda=SimpleNamespace(CUDAGraph=Mock, graph=lambda *args, **kwargs: nullcontext()),
        ),
        "torch_memory_saver": SimpleNamespace(cuda_graph=lambda *args, **kwargs: nullcontext()),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    capture = namespace[method.name]

    wrapper = SimpleNamespace(
        runtime_mode=mode,
        concrete_cudagraph_entries={},
        cudagraph_options=SimpleNamespace(gc_disable=True, debug_log_enable=False, weak_ref_output=False),
        graph_pool=object(),
        runnable=Mock(return_value=object()),
        is_debugging_mode=False,
    )
    return capture, wrapper


def test_capture_preserves_gc_count_for_background_callbacks(capture_wrapper):
    capture, wrapper = capture_wrapper
    original_collect = gc.collect

    def background_gc():
        for generation in (None, 0, 1, 2):
            freed = gc.collect() if generation is None else gc.collect(generation=generation)
            assert freed >= 0, "GC callbacks require a nonnegative collection count."
        return freed

    with ThreadPoolExecutor(max_workers=1) as executor:
        wrapper.runnable = lambda: executor.submit(background_gc).result(timeout=5)
        assert capture(wrapper) == 0, "Suppressed collection must report zero freed objects."

    assert gc.collect is original_collect, "Capture must restore the original collector."


def test_capture_restores_gc_when_model_forward_raises(capture_wrapper):
    capture, wrapper = capture_wrapper
    original_collect = gc.collect
    wrapper.runnable = Mock(side_effect=RuntimeError("Model forward failed."))

    with pytest.raises(RuntimeError, match="Model forward failed"):
        capture(wrapper)

    assert gc.collect is original_collect, "Failed capture must restore the original collector."


def test_capture_retains_collection_when_gc_is_enabled(capture_wrapper, monkeypatch):
    capture, wrapper = capture_wrapper
    collect = Mock(return_value=7)
    monkeypatch.setattr(gc, "collect", collect)
    wrapper.cudagraph_options.gc_disable = False
    wrapper.runnable = lambda: gc.collect()

    assert capture(wrapper) == 7, "Capture with GC enabled must retain the collector result."
    collect.assert_called_once_with()
