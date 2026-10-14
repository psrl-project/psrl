import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_test


def _load_vllm_extension_module():
    path = Path(__file__).parents[3] / "psrl/workers/gen/vllm_extension.py"
    spec = importlib.util.spec_from_file_location("_test_vllm_extension_weight_loading", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rank_local_checkpoint_weights_stay_on_cpu_and_initialize_ep_filter() -> None:
    module = _load_vllm_extension_module()
    model_config = object()
    model = object()
    tensor = torch.arange(8)
    calls = []

    class Loader:
        def _init_ep_weight_filter(self, value) -> None:
            calls.append(("filter", value))

        def get_all_weights(self, config, target):
            calls.append(("iterate", config, target))
            yield "weight", tensor

    weights = list(module._iter_rank_local_checkpoint_weights(Loader(), model_config, model))

    assert calls == [("filter", model_config), ("iterate", model_config, model)]
    assert weights == [("weight", tensor)]
    assert weights[0][1] is tensor
    assert weights[0][1].device.type == "cpu"


def test_rank_local_checkpoint_weights_reject_non_cpu_tensor() -> None:
    module = _load_vllm_extension_module()

    class Loader:
        def get_all_weights(self, model_config, model):
            yield "weight", torch.empty(1, device="meta")

    with pytest.raises(RuntimeError, match="must yield CPU tensors.*weight.*meta"):
        list(module._iter_rank_local_checkpoint_weights(Loader(), object(), object()))


def test_model_weight_reload_keeps_vllm_config_context_active(monkeypatch) -> None:
    module = _load_vllm_extension_module()
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(), device_config=SimpleNamespace(device="cpu"))
    context = {"active": False}

    @contextmanager
    def set_current_config(value):
        assert value is vllm_config
        assert not context["active"]
        context["active"] = True
        try:
            yield
        finally:
            context["active"] = False

    class Model:
        def load_weights(self, *, weights):
            assert context["active"]
            assert list(weights) == [("weight", 1)]
            return {"weight"}

    monkeypatch.setattr(module, "set_current_vllm_config", set_current_config)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: None)

    loaded = module._load_model_weights_with_vllm_config(Model(), iter([("weight", 1)]), vllm_config)

    assert loaded == {"weight"}
    assert not context["active"]


def test_node_shared_preload_uses_lazy_safetensors(monkeypatch, tmp_path: Path) -> None:
    module = _load_vllm_extension_module()
    observed_load_configs = []

    class Loader:
        def get_all_weights(self, model_config, model):
            yield from ()

    def get_model_loader(load_config):
        observed_load_configs.append(load_config)
        return Loader()

    from psrl.utils import node_shared_weight_cache

    monkeypatch.setattr(module, "get_model_loader", get_model_loader)
    monkeypatch.setattr(
        node_shared_weight_cache,
        "validate_node_cache_directory",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        node_shared_weight_cache,
        "fingerprint_safetensors_checkpoint",
        lambda path: "checkpoint-fingerprint",
    )

    worker = object.__new__(module.vLLMWorkerExtension)
    worker.model_config = SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.bfloat16,
        quantization=None,
    )
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Module(),
        load_config=SimpleNamespace(
            load_format="dummy",
            safetensors_load_strategy="eager",
            model_loader_extra_config={"enable_multithread_load": True},
        ),
        vllm_config=SimpleNamespace(
            additional_config={
                "psrl_nixl_weight_arena": {
                    "reward_enabled": True,
                    "reward_cpu_cache_mode": "node_shared",
                    "reward_cpu_cache_pin_memory": False,
                    "reward_node_cache_dir": str(tmp_path),
                    "reward_node_cache_backend": "shm",
                }
            }
        ),
        _psrl_weight_arena_handle=object(),
    )

    assert worker.preload_weights_to_cpu_cache(load_format="auto") == 0
    assert len(observed_load_configs) == 1
    load_config = observed_load_configs[0]
    assert load_config.load_format == "safetensors"
    assert load_config.safetensors_load_strategy == "lazy"
    assert load_config.model_loader_extra_config == {}
    assert worker._psrl_reward_weight_cache_state == {
        "ready": True,
        "source": "node_shared_checkpoint",
        "checkpoint_fingerprint": "checkpoint-fingerprint",
    }


def test_preload_without_reward_arena_uses_process_local_cache(monkeypatch, tmp_path: Path) -> None:
    module = _load_vllm_extension_module()
    checkpoint_tensor = torch.arange(4)

    class Loader:
        def get_all_weights(self, model_config, model):
            yield "weight", checkpoint_tensor

    monkeypatch.setattr(module, "copy_to_local", lambda path: path)
    monkeypatch.setattr(module, "get_model_loader", lambda _load_config: Loader())

    worker = object.__new__(module.vLLMWorkerExtension)
    worker.model_config = SimpleNamespace(model=str(tmp_path))
    worker.model_runner = SimpleNamespace(
        model=torch.nn.Module(),
        load_config=SimpleNamespace(load_format="dummy"),
        vllm_config=SimpleNamespace(
            additional_config={
                "psrl_nixl_weight_arena": {
                    "reward_enabled": False,
                    "reward_cpu_cache_mode": "node_shared",
                }
            }
        ),
    )

    assert worker.preload_weights_to_cpu_cache(load_format="auto") == 1
    cached_tensor = worker._psrl_cpu_weight_cache[0][1]
    assert torch.equal(cached_tensor, checkpoint_tensor)
    assert cached_tensor.data_ptr() != checkpoint_tensor.data_ptr()
    assert worker._psrl_reward_weight_cache_state == {
        "ready": True,
        "source": "checkpoint",
        "checkpoint_tensor_count": 1,
    }


def test_nixl_pull_returns_engine_core_fingerprints_to_caller() -> None:
    module = _load_vllm_extension_module()
    worker = object.__new__(module.vLLMWorkerExtension)
    worker.unified_state_dict = {}
    worker.nixl_storage_client = SimpleNamespace(
        merge_and_finish_cached_xfer=lambda: None,
        clear_intermediate_cached_data=lambda: None,
    )
    worker.cuda_synchronize = lambda: None
    worker.param_sync_plan = SimpleNamespace(after_pull=lambda _state_dict: None)
    worker.get_instance_local_rank = lambda: 0
    worker.get_instance_local_tp_rank = lambda: 0

    captured_stages = []

    def capture(*, stage, model_version, options):
        captured_stages.append((stage, model_version, options))
        return {"stage": stage, "model_version": model_version}

    worker._capture_weight_fingerprint = capture
    options = {"mode": "sampled"}

    result = worker.nixl_pull_model_core([], [], model_version=7, fingerprint_options=options)

    assert captured_stages == [
        ("rollout_after_raw_pull", 7, options),
        ("rollout_after_param_sync", 7, options),
    ]
    assert result == {
        "model_version": 7,
        "raw_fingerprint": {"stage": "rollout_after_raw_pull", "model_version": 7},
        "final_fingerprint": {"stage": "rollout_after_param_sync", "model_version": 7},
    }
