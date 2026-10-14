import pytest
import torch
from psrl.utils.weight_arena import (
    _install_packed_tms_pool,
    _weight_tms_pool_key,
    gb_to_bytes,
    get_fsdp_param_groups,
    materialize_module_weights_in_arena,
    pack_module_weights,
    prepare_module_for_direct_fsdp2_weight_arena,
    restore_weight_arena_from_cpu,
    snapshot_weight_arena_to_cpu,
)
from torch import nn

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("value", "expected"),
    [(4, 4 * 1024**3), ("256", 256 * 1024**3), (0.0625, 64 * 1024**2)],
)
def test_gb_to_bytes(value, expected: int) -> None:
    assert gb_to_bytes(value, field_name="size_gb") == expected


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), "invalid"])
def test_gb_to_bytes_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="size_gb must be"):
        gb_to_bytes(value, field_name="size_gb")


def test_gb_to_bytes_can_allow_zero() -> None:
    assert gb_to_bytes(0, field_name="reserve_gb", allow_zero=True) == 0


class _SharedStorageModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = torch.arange(16, dtype=torch.float32)
        self.left = nn.Parameter(shared[:8].view(2, 4))
        self.tied = self.left
        self.right = nn.Parameter(shared[4:12])
        self.right.weight_loader = object()

        noncontiguous_base = torch.arange(24, dtype=torch.int32).view(4, 6)
        self.register_buffer("noncontiguous", noncontiguous_base[:, ::2])
        self.register_buffer("half_values", torch.arange(10, dtype=torch.float16))


class _MetaSharedStorageModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        with torch.device("meta"):
            shared = torch.empty(16, dtype=torch.float32)
            self.left = nn.Parameter(shared[:8].view(2, 4))
            self.tied = self.left
            self.right = nn.Parameter(shared[4:12])
            self.register_buffer("strided", torch.empty_strided((3, 2), (4, 1)))
            self.register_buffer("empty", torch.empty(0, dtype=torch.float16))
        self.right.weight_loader = object()


def _named_tensors(module: nn.Module) -> dict[str, torch.Tensor]:
    tensors = dict(module.named_parameters(remove_duplicate=False))
    tensors.update(module.named_buffers(remove_duplicate=False))
    return tensors


def test_pack_module_weights_preserves_values_views_and_objects() -> None:
    module = _SharedStorageModule()
    before = _named_tensors(module)
    values = {name: tensor.detach().clone() for name, tensor in before.items()}
    identities = {name: id(tensor) for name, tensor in before.items()}
    strides = {name: tensor.stride() for name, tensor in before.items()}
    shared_relative_offset = module.right.storage_offset() - module.left.storage_offset()
    loader = module.right.weight_loader

    handle = pack_module_weights(
        module,
        max_chunk_bytes=128,
        alignment_bytes=16,
        require_cuda=False,
    )

    after = _named_tensors(module)
    assert handle.stats.unique_storage_count == 3
    assert handle.stats.tensor_binding_count == 4
    assert handle.stats.arena_count >= 2
    assert module.tied is module.left
    assert module.left.untyped_storage().data_ptr() == module.right.untyped_storage().data_ptr()
    assert module.right.storage_offset() - module.left.storage_offset() == shared_relative_offset
    assert module.right.weight_loader is loader
    assert handle.assert_virtual_addresses_unchanged() == handle.base_addresses
    for name, tensor in after.items():
        assert id(tensor) == identities[name]
        assert tensor.stride() == strides[name]
        torch.testing.assert_close(tensor, values[name], rtol=0, atol=0)
        assert any(tensor.untyped_storage().data_ptr() == arena.data_ptr() for arena in handle.arenas)
    assert len({tensor.untyped_storage().data_ptr() for tensor in after.values()}) == handle.stats.arena_count


def test_materialize_meta_module_preserves_views_objects_and_attributes() -> None:
    module = _MetaSharedStorageModule()
    before = _named_tensors(module)
    identities = {name: id(tensor) for name, tensor in before.items()}
    loader = module.right.weight_loader

    handle = materialize_module_weights_in_arena(
        module,
        device="cpu",
        max_chunk_bytes=128,
        alignment_bytes=16,
    )

    after = _named_tensors(module)
    assert handle.stats.unique_storage_count == 2
    assert handle.stats.tensor_binding_count == 4
    assert module.tied is module.left
    assert module.right.weight_loader is loader
    assert module.left.untyped_storage().data_ptr() == module.right.untyped_storage().data_ptr()
    assert module.right.storage_offset() - module.left.storage_offset() == 4
    assert module.strided.stride() == (4, 1)
    assert module.empty.device.type == "cpu"
    for name, tensor in after.items():
        assert id(tensor) == identities[name]
        assert tensor.device.type == "cpu"

    with torch.no_grad():
        module.left.fill_(1)
    torch.testing.assert_close(module.right[:4], torch.ones(4), rtol=0, atol=0)


def test_materialize_cpu_module_copies_values_without_replacing_parameters() -> None:
    module = _SharedStorageModule()
    before = _named_tensors(module)
    identities = {name: id(tensor) for name, tensor in before.items()}
    values = {name: tensor.detach().clone() for name, tensor in before.items()}

    materialize_module_weights_in_arena(
        module,
        device="cpu",
        max_chunk_bytes=128,
        alignment_bytes=16,
    )

    for name, tensor in _named_tensors(module).items():
        assert id(tensor) == identities[name]
        torch.testing.assert_close(tensor, values[name], rtol=0, atol=0)


def test_materialized_module_detects_new_weight_storage() -> None:
    module = _MetaSharedStorageModule()
    handle = materialize_module_weights_in_arena(
        module,
        device="cpu",
        max_chunk_bytes=128,
        alignment_bytes=16,
    )
    handle.assert_module_weights_in_arena(module)

    module.register_buffer("late_buffer", torch.ones(1))
    with pytest.raises(RuntimeError, match="escaped the weight arena"):
        handle.assert_module_weights_in_arena(module)


def test_cpu_arena_snapshot_restores_final_parameter_storage() -> None:
    module = _SharedStorageModule()
    handle = pack_module_weights(
        module,
        max_chunk_bytes=128,
        alignment_bytes=16,
        require_cuda=False,
    )
    expected = {name: tensor.detach().clone() for name, tensor in _named_tensors(module).items()}
    base_addresses = handle.base_addresses
    cached_arenas = snapshot_weight_arena_to_cpu(handle, pin_memory=False)

    for arena in handle.arenas:
        arena.zero_()

    restored_bytes = restore_weight_arena_from_cpu(handle, cached_arenas)

    assert restored_bytes == sum(arena.numel() * arena.element_size() for arena in handle.arenas)
    assert handle.assert_virtual_addresses_unchanged() == base_addresses
    for name, tensor in _named_tensors(module).items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_oversized_storage_gets_its_own_arena() -> None:
    module = nn.Module()
    module.register_buffer("large", torch.arange(80, dtype=torch.uint8))
    module.register_buffer("small", torch.arange(8, dtype=torch.uint8))

    handle = pack_module_weights(
        module,
        max_chunk_bytes=64,
        alignment_bytes=16,
        require_cuda=False,
    )

    assert handle.stats.arena_count == 2
    assert handle.arenas[0].numel() == 80
    assert handle.arenas[1].numel() == 8


def test_empty_view_with_nonempty_storage_is_rebound() -> None:
    module = nn.Module()
    base = torch.arange(8, dtype=torch.float32)
    module.register_buffer("empty", base[4:4])

    handle = pack_module_weights(
        module,
        max_chunk_bytes=64,
        alignment_bytes=16,
        require_cuda=False,
    )

    assert module.empty.numel() == 0
    assert module.empty.untyped_storage().data_ptr() == handle.arenas[0].data_ptr()
    assert module.empty.storage_offset() == 4


def test_pack_copies_storage_padding_outside_visible_tensor_view() -> None:
    module = nn.Module()
    storage_values = torch.arange(32, dtype=torch.uint8)
    module.register_buffer("middle", storage_values[8:24])

    handle = pack_module_weights(
        module,
        max_chunk_bytes=64,
        alignment_bytes=16,
        require_cuda=False,
    )

    assert module.middle.storage_offset() == 8
    torch.testing.assert_close(handle.arenas[0], storage_values, rtol=0, atol=0)


def test_storage_after_exact_chunk_boundary_starts_new_arena() -> None:
    module = nn.Module()
    module.register_buffer("first", torch.arange(32, dtype=torch.uint8))
    module.register_buffer("second", torch.arange(32, dtype=torch.uint8))
    module.register_buffer("third", torch.arange(1, dtype=torch.uint8))

    handle = pack_module_weights(
        module,
        max_chunk_bytes=64,
        alignment_bytes=16,
        require_cuda=False,
    )

    assert [arena.numel() for arena in handle.arenas] == [64, 1]
    assert [(placement.arena_index, placement.offset_bytes) for placement in handle.placements] == [
        (0, 0),
        (0, 32),
        (1, 0),
    ]


@pytest.mark.parametrize(
    ("max_chunk_bytes", "alignment_bytes"),
    [(0, 16), (64, 0), (64, 24), (65, 16)],
)
def test_invalid_arena_config_is_rejected(max_chunk_bytes: int, alignment_bytes: int) -> None:
    module = nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError):
        pack_module_weights(
            module,
            max_chunk_bytes=max_chunk_bytes,
            alignment_bytes=alignment_bytes,
            require_cuda=False,
        )


def test_get_fsdp_param_groups_supports_current_and_legacy_state_layouts() -> None:
    current_group = type("CurrentGroup", (), {"fsdp_params": []})()
    current_state = type("CurrentState", (), {"_fsdp_param_group": current_group})()
    empty_current_state = type("EmptyCurrentState", (), {"_fsdp_param_group": None})()
    legacy_groups = tuple(type("LegacyGroup", (), {"fsdp_params": []})() for _ in range(2))
    legacy_state = type("LegacyState", (), {"_fsdp_param_groups": legacy_groups})()

    assert get_fsdp_param_groups(current_state) == (current_group,)
    assert get_fsdp_param_groups(empty_current_state) == ()
    assert get_fsdp_param_groups(legacy_state) == legacy_groups


def test_get_fsdp_param_groups_rejects_unknown_layout() -> None:
    with pytest.raises(RuntimeError, match="missing _fsdp_param_group"):
        get_fsdp_param_groups(object())


def test_prepare_direct_fsdp2_arena_retains_cpu_state_and_dematerializes_module() -> None:
    module = nn.Linear(4, 3)
    full_state = module.state_dict()
    expected = {name: tensor.clone() for name, tensor in full_state.items()}

    device_counts = prepare_module_for_direct_fsdp2_weight_arena(module, full_state)

    assert device_counts == {"cpu": 2}
    assert all(tensor.device.type == "meta" for tensor in module.state_dict().values())
    for name, tensor in full_state.items():
        assert tensor.device.type == "cpu"
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


def test_prepare_direct_fsdp2_arena_accepts_receiving_rank_meta_state() -> None:
    with torch.device("meta"):
        module = nn.Linear(4, 3)
    parameter_ids = {name: id(parameter) for name, parameter in module.named_parameters()}
    full_state = module.state_dict()

    device_counts = prepare_module_for_direct_fsdp2_weight_arena(module, full_state)

    assert device_counts == {"meta": 2}
    assert {name: id(parameter) for name, parameter in module.named_parameters()} == parameter_ids
    assert all(tensor.device.type == "meta" for tensor in module.state_dict().values())


def test_prepare_direct_fsdp2_arena_rejects_accelerator_state() -> None:
    class _FakeCudaTensor:
        device = torch.device("cuda:0")

    module = nn.Linear(4, 3)
    with pytest.raises(RuntimeError, match=r"device_counts=\{'cuda': 1\}.*model.weight.*cuda:0"):
        prepare_module_for_direct_fsdp2_weight_arena(
            module,
            {"model.weight": _FakeCudaTensor()},  # type: ignore[dict-item]
        )


def test_prepare_direct_fsdp2_arena_rejects_mixed_cpu_meta_state() -> None:
    module = nn.Linear(4, 3)
    with pytest.raises(RuntimeError, match=r"uniform.*device_counts=\{'cpu': 1, 'meta': 1\}"):
        prepare_module_for_direct_fsdp2_weight_arena(
            module,
            {
                "model.weight": torch.empty(1),
                "model.bias": torch.empty(1, device="meta"),
            },
        )


@pytest.mark.parametrize("enable_cpu_backup", [False, True])
def test_weight_tms_pool_key_selects_requested_cpu_backup_flag(enable_cpu_backup: bool) -> None:
    source_key = ("weights", enable_cpu_backup)
    mem_pools = {source_key: object(), ("kv_cache", False): object()}

    assert _weight_tms_pool_key(mem_pools, "weights", enable_cpu_backup) == source_key


def test_weight_tms_pool_key_rejects_missing_requested_flag() -> None:
    with pytest.raises(RuntimeError, match="expected model-weight TMS pool"):
        _weight_tms_pool_key({("weights", True): object()}, "weights", enable_cpu_backup=False)


def test_install_packed_tms_pool_replaces_only_selected_source_key() -> None:
    source_pool = object()
    packed_pool = object()
    mem_pools = {
        ("weights", False): source_pool,
        ("weights", True): object(),
        ("kv_cache", False): object(),
    }

    _install_packed_tms_pool(mem_pools, ("weights", False), packed_pool)

    assert mem_pools[("weights", False)] is packed_pool
    assert mem_pools[("weights", True)] is not packed_pool
    assert ("kv_cache", False) in mem_pools


def test_install_packed_tms_pool_keeps_matching_cpu_backup_key() -> None:
    packed_pool = object()
    mem_pools = {("weights", False): object()}

    _install_packed_tms_pool(mem_pools, ("weights", False), packed_pool)

    assert mem_pools == {("weights", False): packed_pool}
