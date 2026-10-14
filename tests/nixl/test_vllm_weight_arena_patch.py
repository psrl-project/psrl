from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm_patches.patches import weight_arena
from vllm_patches.patches.gpu_worker import _should_restore_sleep_buffers
from vllm_patches.patches.weight_arena import (
    _assert_direct_model_supported,
    _assert_weight_offload_disabled,
    _initialize_direct_runtime_buffers,
    _materialize_ep_metadata_on_cpu,
    _role_materialization,
    _weight_arena_enabled,
    finalize_pending_weight_arena,
)

pytestmark = pytest.mark.cpu_test


class _RuntimeBufferProbe(torch.nn.Module):
    def __init__(self, *, supported: bool = True) -> None:
        super().__init__()
        name = "cos_sin_cache" if supported else "unknown_cache"
        self.register_buffer(name, torch.empty(4, 3, device="meta"), persistent=False)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        return torch.arange(12, dtype=torch.float32).view(4, 3)


def _model_runner(*, uva_gb=0, prefetch_group_size=0, cache_config=None):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=cache_config or SimpleNamespace(),
            offload_config=SimpleNamespace(
                uva=SimpleNamespace(cpu_offload_gb=uva_gb),
                prefetch=SimpleNamespace(offload_group_size=prefetch_group_size),
            ),
        )
    )


def test_accepts_vllm_018_default_offload_config() -> None:
    _assert_weight_offload_disabled(_model_runner())


@pytest.mark.parametrize(
    "runner",
    [
        _model_runner(uva_gb=1),
        _model_runner(prefetch_group_size=2),
        _model_runner(cache_config=SimpleNamespace(cpu_offload_gb=1)),
    ],
)
def test_rejects_enabled_weight_offload(runner) -> None:
    with pytest.raises(RuntimeError, match="does not support vLLM weight offloading"):
        _assert_weight_offload_disabled(runner)


@pytest.mark.parametrize(
    "offload_config, missing_field",
    [
        (None, "missing offload_config"),
        (SimpleNamespace(prefetch=SimpleNamespace(offload_group_size=0)), "missing offload_config.uva"),
        (SimpleNamespace(uva=SimpleNamespace(cpu_offload_gb=0)), "missing offload_config.prefetch"),
    ],
)
def test_rejects_unknown_offload_config_layout(offload_config, missing_field) -> None:
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(), offload_config=offload_config)
    )
    with pytest.raises(RuntimeError, match=missing_field):
        _assert_weight_offload_disabled(runner)


@pytest.mark.parametrize(
    ("vllm_config", "model_config", "message"),
    [
        (
            SimpleNamespace(quant_config=object(), lora_config=None),
            SimpleNamespace(quantization=None, is_moe=False),
            "unquantized",
        ),
        (
            SimpleNamespace(
                quant_config=None,
                lora_config=object(),
                parallel_config=SimpleNamespace(enable_expert_parallel=False),
            ),
            SimpleNamespace(quantization=None, is_moe=False),
            "LoRA",
        ),
    ],
)
def test_direct_loader_rejects_unsupported_models(vllm_config, model_config, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        _assert_direct_model_supported(vllm_config, model_config)


def test_direct_loader_accepts_dense_unquantized_model() -> None:
    _assert_direct_model_supported(
        SimpleNamespace(
            quant_config=None,
            lora_config=None,
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
        ),
        SimpleNamespace(quantization=None, is_moe=False),
    )


def test_direct_loader_accepts_unquantized_tp_only_moe() -> None:
    _assert_direct_model_supported(
        SimpleNamespace(
            quant_config=None,
            lora_config=None,
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
        ),
        SimpleNamespace(quantization=None, is_moe=True),
    )


def test_direct_loader_accepts_unquantized_ep_moe() -> None:
    _assert_direct_model_supported(
        SimpleNamespace(
            quant_config=None,
            lora_config=None,
            parallel_config=SimpleNamespace(enable_expert_parallel=True),
        ),
        SimpleNamespace(quantization=None, is_moe=True),
    )


def test_ep_metadata_uses_cpu_inside_meta_model_construction() -> None:
    from vllm.model_executor.layers.fused_moe import expert_map_manager

    with torch.device("meta"):
        with _materialize_ep_metadata_on_cpu(True):
            local_experts, expert_map, expert_mask = expert_map_manager.determine_expert_map(
                ep_size=4,
                ep_rank=2,
                global_num_experts=128,
                expert_placement_strategy="linear",
                num_fused_shared_experts=0,
                return_expert_mask=False,
            )

    assert local_experts == 32
    assert expert_map.device.type == "cpu"
    assert expert_mask is None
    assert expert_map[:64].eq(-1).all()
    assert expert_map[64:96].equal(torch.arange(32, dtype=torch.int32))
    assert expert_map[96:].eq(-1).all()


def test_direct_runtime_buffer_is_initialized_in_place() -> None:
    from psrl.utils.weight_arena import materialize_module_weights_in_arena

    model = _RuntimeBufferProbe()
    handle = materialize_module_weights_in_arena(
        model,
        device="cpu",
        max_chunk_bytes=1024,
        alignment_bytes=256,
    )
    original_storage = model.cos_sin_cache.untyped_storage()
    original_address = model.cos_sin_cache.data_ptr()

    initialized = _initialize_direct_runtime_buffers(model)

    assert initialized == ("cos_sin_cache",)
    assert model.cos_sin_cache.untyped_storage().data_ptr() == original_storage.data_ptr()
    assert model.cos_sin_cache.data_ptr() == original_address
    torch.testing.assert_close(
        model.cos_sin_cache,
        torch.arange(12, dtype=torch.float32).view(4, 3),
        rtol=0,
        atol=0,
    )
    handle.assert_module_weights_in_arena(model)


def test_direct_runtime_buffer_rejects_unknown_non_persistent_buffer() -> None:
    from psrl.utils.weight_arena import materialize_module_weights_in_arena

    model = _RuntimeBufferProbe(supported=False)
    materialize_module_weights_in_arena(
        model,
        device="cpu",
        max_chunk_bytes=1024,
        alignment_bytes=256,
    )

    with pytest.raises(RuntimeError, match="without an initializer.*unknown_cache"):
        _initialize_direct_runtime_buffers(model)


def test_materialization_is_role_specific() -> None:
    config = {
        "rollout_materialization": "direct",
        "reward_materialization": "repack",
    }
    assert _role_materialization(config, "rollout") == "direct"
    assert _role_materialization(config, "reward") == "repack"


def test_finalize_pending_weight_arena_replaces_tms_pool(monkeypatch) -> None:
    arena_config = {"max_chunk_gb": 1, "alignment_bytes": 256}
    runner = SimpleNamespace(_psrl_weight_arena_pending_config=arena_config)
    calls = []
    monkeypatch.setattr(
        weight_arena,
        "_pack_loaded_model",
        lambda model_runner, config, *, replace_tms_pool: calls.append((model_runner, config, replace_tms_pool)),
    )

    finalize_pending_weight_arena(runner)

    assert calls == [(runner, arena_config, True)]
    assert not hasattr(runner, "_psrl_weight_arena_pending_config")


def test_finalize_pending_weight_arena_is_noop_without_request(monkeypatch) -> None:
    runner = SimpleNamespace()
    monkeypatch.setattr(
        weight_arena,
        "_pack_loaded_model",
        lambda *args, **kwargs: pytest.fail("unexpected pack"),
    )

    finalize_pending_weight_arena(runner)


@pytest.mark.parametrize(
    ("role", "config", "expected"),
    [
        ("rollout", {"rollout_enabled": True}, True),
        ("rollout", {"reward_enabled": True}, False),
        ("reward", {"reward_enabled": True}, True),
        ("reward", {"rollout_enabled": True}, False),
    ],
)
def test_weight_arena_enablement_is_role_specific(role, config, expected) -> None:
    additional_config = {
        "psrl_role": role,
        "psrl_nixl_weight_arena": config,
    }
    assert _weight_arena_enabled(additional_config) is expected


def test_split_reward_wake_restores_buffers_with_kv_cache() -> None:
    assert not _should_restore_sleep_buffers(["weights"])
    assert _should_restore_sleep_buffers(["kv_cache"])
    assert _should_restore_sleep_buffers(["weights", "kv_cache"])
    assert _should_restore_sleep_buffers(None)
