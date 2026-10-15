"""CPU-only structural tests for PSRL_EngineTrainWorker."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


def test_engine_train_worker_is_importable():
    """The module must import without GPU or Ray initialised."""
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    assert PSRL_EngineTrainWorker is not None


def test_engine_train_worker_inherits_both_bases():
    from psrl.workers.train.base_train_worker import PSRL_BaseTrainWorker
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker
    from verl.workers.engine_workers import ActorRolloutRefWorker

    assert issubclass(PSRL_EngineTrainWorker, ActorRolloutRefWorker)
    assert issubclass(PSRL_EngineTrainWorker, PSRL_BaseTrainWorker)


def test_engine_train_worker_has_required_methods():
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    for method in [
        "init_model",
        "compute_log_prob",
        "update_actor",
        "init_nixl_client",
        "nixl_convert_params",
        "nixl_protocol",
        "nixl_sleep",
        "nixl_wake_up",
        "sleep_model",
        "wake_up_model",
        "ray_push_model",
        "get_replica_id",
        "is_train_representative_rank",
    ]:
        assert hasattr(PSRL_EngineTrainWorker, method), f"Missing: {method}"


def test_engine_train_worker_importable_from_package():
    from psrl.workers.train import PSRL_EngineTrainWorker

    assert PSRL_EngineTrainWorker is not None


def _arena_worker(*, ps_mode="nixl_cpu", tms_range="all", strategy="fsdp2", param_offload=False):
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    worker = object.__new__(PSRL_EngineTrainWorker)
    worker.psrl_config = OmegaConf.create(
        {
            "ps_mode": ps_mode,
            "tms": {"range": tms_range},
            "nixl": {"weight_arena": {"actor_enabled": True, "actor_materialization": "repack"}},
        }
    )
    worker.config = OmegaConf.create(
        {
            "actor": {
                "strategy": strategy,
                "fsdp_config": {"param_offload": param_offload},
            }
        }
    )
    return PSRL_EngineTrainWorker, worker


def test_actor_weight_arena_config_validates_active_worker_requirements():
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker

    _, worker = _arena_worker()
    assert PSRL_EngineTrainWorker._actor_weight_arena_config(worker).actor_enabled

    for worker_kwargs, message in (
        ({"ps_mode": "cpu_ref"}, "ps_mode"),
        ({"tms_range": "null"}, "psrl.tms.range"),
        ({"strategy": "fsdp"}, "fsdp2"),
        ({"param_offload": True}, "parameter offloading"),
    ):
        _, invalid_worker = _arena_worker(**worker_kwargs)
        with pytest.raises(RuntimeError, match=message):
            PSRL_EngineTrainWorker._actor_weight_arena_config(invalid_worker)


def test_actor_weight_arena_materialization_uses_fresh_tms_pool(monkeypatch):
    from psrl.utils import weight_arena

    _, worker = _arena_worker()
    worker.psrl_config.nixl.weight_arena.max_chunk_gb = 2
    worker.psrl_config.nixl.weight_arena.alignment_bytes = 256
    module = object()
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=module))
    worker._rank = 0
    stats = SimpleNamespace(
        arena_count=1,
        unique_storage_count=2,
        tensor_binding_count=2,
        total_storage_bytes=512,
        total_arena_bytes=512,
        largest_storage_bytes=256,
        pack_seconds=0.1,
        tms_source_reserved_bytes=512,
        tms_source_active_bytes=0,
        tms_source_active_allocations=0,
    )
    handle = SimpleNamespace(stats=stats, base_addresses=(1234,))
    unchanged = []
    handle.assert_fsdp_bindings_unchanged = lambda: unchanged.append(True)
    packed = {}

    def fake_pack(actual_module, *, max_chunk_bytes, alignment_bytes, enable_cpu_backup):
        packed.update(
            module=actual_module,
            max_chunk_bytes=max_chunk_bytes,
            alignment_bytes=alignment_bytes,
            enable_cpu_backup=enable_cpu_backup,
        )
        return handle

    monkeypatch.setattr(weight_arena, "pack_fsdp2_model_weights_in_fresh_tms_pool", fake_pack)
    worker._materialize_actor_weight_arena()

    assert packed == {
        "module": module,
        "max_chunk_bytes": 2 * 1024**3,
        "alignment_bytes": 256,
        "enable_cpu_backup": False,
    }
    assert worker._nixl_weight_arena_handle is handle
    assert worker._nixl_weight_arena_materialization == "repack"
    assert unchanged == [True]


def test_weight_arena_uses_model_pool_when_optimizer_pool_exists():
    from psrl.utils.weight_arena import _install_packed_tms_pool, _weight_tms_pool_key

    optimizer_pool = object()
    packed_pool = object()
    mem_pools = {
        ("default", False): object(),
        ("default", True): optimizer_pool,
    }

    model_pool_key = _weight_tms_pool_key(mem_pools, "default", enable_cpu_backup=False)
    assert model_pool_key == ("default", False)
    _install_packed_tms_pool(mem_pools, model_pool_key, packed_pool)
    assert mem_pools[model_pool_key] is packed_pool
    assert mem_pools[("default", True)] is optimizer_pool


def test_weight_arena_selects_cpu_backup_pool_when_requested():
    from psrl.utils.weight_arena import _weight_tms_pool_key

    mem_pools = {
        ("weights", False): object(),
        ("weights", True): object(),
    }

    assert _weight_tms_pool_key(mem_pools, "weights", enable_cpu_backup=True) == ("weights", True)


def test_fsdp2_arena_handle_retains_binding_metadata():
    from types import SimpleNamespace

    from psrl.utils.weight_arena import (
        FSDP2WeightArenaHandle,
        _pack_named_tensors,
    )

    padded = torch.arange(9, dtype=torch.float32)[1:]
    local_tensor = padded[:4]
    fsdp_param = SimpleNamespace(
        _param_fqn="layer.weight",
        _sharded_param_data=padded,
        sharded_param=SimpleNamespace(_local_tensor=local_tensor),
    )
    handle = _pack_named_tensors(
        [
            ("prefix", torch.arange(1, dtype=torch.float32)),
            ("fsdp_padded:layer.weight", padded),
            ("fsdp_local:layer.weight", local_tensor),
        ],
        max_chunk_bytes=128,
        alignment_bytes=16,
        require_cuda=False,
        fsdp_param_specs={"fsdp_padded:layer.weight": (fsdp_param, 4)},
    )

    assert isinstance(handle, FSDP2WeightArenaHandle)
    assert handle.fsdp_param_placements == ((fsdp_param, 0, 20),)
    handle.assert_fsdp_bindings_unchanged()


def test_fsdp2_arena_metadata_uses_unique_names_when_fqn_is_unset():
    from psrl.utils.weight_arena import _fsdp_param_name, _pack_named_tensors

    padded_a = torch.arange(4, dtype=torch.float32)
    padded_b = torch.arange(4, dtype=torch.float32) + 10
    local_a = padded_a[:2]
    local_b = padded_b[:2]
    fsdp_param_a = SimpleNamespace(_param_fqn=None)
    fsdp_param_b = SimpleNamespace(_param_fqn=None)
    fsdp_param_a._sharded_param_data = padded_a
    fsdp_param_a.sharded_param = SimpleNamespace(_local_tensor=local_a)
    fsdp_param_b._sharded_param_data = padded_b
    fsdp_param_b.sharded_param = SimpleNamespace(_local_tensor=local_b)
    assert _fsdp_param_name(fsdp_param_a, 1) == "fsdp_param_1"
    assert _fsdp_param_name(fsdp_param_b, 2) == "fsdp_param_2"
    assert _fsdp_param_name(SimpleNamespace(_param_fqn="layer.weight"), 3) == "layer.weight"

    name_a = f"fsdp_padded:{_fsdp_param_name(fsdp_param_a, 1)}"
    name_b = f"fsdp_padded:{_fsdp_param_name(fsdp_param_b, 2)}"
    handle = _pack_named_tensors(
        [
            (name_a, padded_a),
            ("fsdp_local:fsdp_param_1", local_a),
            (name_b, padded_b),
            ("fsdp_local:fsdp_param_2", local_b),
        ],
        max_chunk_bytes=128,
        alignment_bytes=16,
        require_cuda=False,
        fsdp_param_specs={name_a: (fsdp_param_a, 0), name_b: (fsdp_param_b, 0)},
    )

    assert len(handle.fsdp_param_placements) == 2
    assert [placement[0] for placement in handle.fsdp_param_placements] == [fsdp_param_a, fsdp_param_b]
    handle.assert_fsdp_bindings_unchanged()
