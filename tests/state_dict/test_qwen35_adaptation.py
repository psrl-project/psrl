from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

# Importing psrl.utils.nixl creates PortScanner.remote() at module load time.
# These converter tests only need the data types, so keep Ray process-free.
from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.fsdp_converter import split_dim0_fused_fsdp_param, split_fused_moe_fsdp_param
from psrl.utils.converter.hf_converter import maybe_convert_to_smaller_parts
from psrl.utils.converter.model_mappings import (
    MappingType,
    get_qkv_tp_layout,
    make_visual_qkv_tp_sharding,
    reshape_visual_block_qkv,
    slice_qkv_proj,
)
from psrl.utils.converter.modeling.hf_modeling import HFParameterMapping
from psrl.utils.converter.param_sync import DTypeCastSync, ParamSyncPlan
from psrl.utils.converter.vllm_converter import VllmConverter
from psrl.utils.nixl.nixl_spec import NIXLSharding
from torch.nn import Parameter
from vllm.model_executor.layers.linear import QKVParallelLinear

pytestmark = pytest.mark.cpu_test


class _Converter(BaseConverter):
    def convert_state_and_sharding_dict(self, model):
        raise NotImplementedError


def _qwen35_config(num_kv_heads: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_attention_heads=16,
            num_key_value_heads=num_kv_heads,
            hidden_size=2560,
            intermediate_size=9216,
            head_dim=256,
            attn_output_gate=True,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_num_value_heads=32,
            linear_value_head_dim=128,
            num_experts=8,
            moe_intermediate_size=4,
        ),
        vision_config=SimpleNamespace(
            num_heads=16,
            hidden_size=1024,
        ),
    )


class _QKVSplitMapping(HFParameterMapping):
    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
        ]


class _FakeQwen35VllmModel:
    def __init__(self, qkv_module):
        self.qkv_module = qkv_module

    def named_modules(self):
        yield "model.language_model.layers.3.self_attn.qkv_proj", self.qkv_module


def _make_fake_qkv_module(tp_size: int, tp_rank: int, num_kv_heads: int = 2):
    num_heads = 32  # Qwen3.5 output-gated Q projection doubles 16 attention heads.
    head_size = 256
    hidden_size = 2
    replicas = tp_size // num_kv_heads
    q_rows = num_heads // tp_size * head_size
    kv_rows = head_size
    fused = torch.cat(
        (
            torch.full((q_rows, hidden_size), float(tp_rank)),
            torch.full((kv_rows, hidden_size), float(100 + tp_rank // replicas)),
            torch.full((kv_rows, hidden_size), float(200 + tp_rank // replicas)),
        )
    )

    module = object.__new__(QKVParallelLinear)
    torch.nn.Module.__init__(module)
    module.tp_size = tp_size
    module.weight = Parameter(fused)
    return module


def test_qwen35_model_info_includes_vision_layout():
    info = HFParameterMapping(_qwen35_config()).get_model_info()

    assert info["vision_num_heads"] == 16
    assert info["vision_num_kv_heads"] == 16
    assert info["vision_head_size"] == 64
    assert info["linear_key_dim"] == 2048
    assert info["linear_value_dim"] == 4096


@pytest.mark.parametrize(
    ("tp_size", "expected_layout"),
    [(4, (8, 1, 2)), (8, (4, 1, 4))],
)
def test_qkv_split_uses_vllm_kv_replication_layout(tp_size, expected_layout):
    num_heads, num_kv_heads, head_size, hidden_size = 32, 2, 128, 2
    layout = get_qkv_tp_layout(num_heads, num_kv_heads, tp_size)
    assert (layout.num_heads_local, layout.num_kv_heads_local, layout.num_kv_head_replicas) == expected_layout
    q_heads_local, kv_heads_local = layout.num_heads_local, layout.num_kv_heads_local
    fused = Parameter(torch.randn((q_heads_local + 2 * kv_heads_local) * head_size, hidden_size))

    q, k, v = slice_qkv_proj(fused, num_heads, num_kv_heads, head_size, tp_size=tp_size)

    assert q.shape == (q_heads_local * head_size, hidden_size)
    assert k.shape == (head_size, hidden_size)
    assert v.shape == (head_size, hidden_size)
    for shard in (q, k, v):
        assert shard.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()


def test_qkv_replication_requires_tp_multiple_of_kv_heads():
    with pytest.raises(AssertionError, match="tp_size = 3 and num_kv_heads = 2"):
        get_qkv_tp_layout(num_heads=12, num_kv_heads=2, tp_size=3)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_qwen35_vllm_converter_expresses_replicated_kv_heads_as_shared_shards(tp_size):
    num_kv_heads = 2
    replicas = tp_size // num_kv_heads
    mapping = _QKVSplitMapping(_qwen35_config(num_kv_heads=num_kv_heads))
    observed_kv_shards = []

    for rank in range(tp_size):
        module = _make_fake_qkv_module(tp_size=tp_size, tp_rank=rank, num_kv_heads=num_kv_heads)
        state, shardings = VllmConverter(mapping, tp_rank=rank).convert_state_and_sharding_dict(
            _FakeQwen35VllmModel(module)
        )

        prefix = "model.language_model.layers.3.self_attn"
        q_key = f"{prefix}.q_proj.weight"
        k_key = f"{prefix}.k_proj.weight"
        v_key = f"{prefix}.v_proj.weight"

        assert state[q_key].shape == (1, 16 // tp_size, 2, 256, 2)
        assert shardings[q_key].shard_mesh == OrderedDict([(0, 2), (1, replicas)])
        assert shardings[q_key].shard_indices == [(rank // replicas, rank % replicas)]

        expected_kv_shard = (rank // replicas,)
        for key, value_offset in ((k_key, 100), (v_key, 200)):
            assert state[key].shape == (1, 256, 2)
            assert shardings[key].shard_mesh == OrderedDict([(0, num_kv_heads)])
            assert shardings[key].shard_indices == [expected_kv_shard]
            assert torch.all(state[key] == value_offset + expected_kv_shard[0])
            assert state[key].untyped_storage().data_ptr() == module.weight.untyped_storage().data_ptr()

        observed_kv_shards.append(expected_kv_shard)

    assert observed_kv_shards == [(rank // replicas,) for rank in range(tp_size)]


def test_visual_qkv_uses_explicit_head_dimension_and_shared_storage():
    original = Parameter(torch.arange(3 * 16 * 64 * 2, dtype=torch.float32).reshape(3 * 16 * 64, 2))

    converted = reshape_visual_block_qkv(original, vision_head_size=64)

    assert converted.shape == (3 * 16, 64, 2)
    assert converted.untyped_storage().data_ptr() == original.untyped_storage().data_ptr()
    assert torch.equal(converted.reshape_as(original), original)


def test_visual_qkv_bias_gets_stable_head_layout():
    original = Parameter(torch.arange(3 * 16 * 64, dtype=torch.float32))

    converted = reshape_visual_block_qkv(original, vision_head_size=64)

    assert converted.shape == (3 * 16, 64, 1)
    assert torch.equal(converted.reshape_as(original), original)


def test_hf_converter_uses_vision_head_size():
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()
    original = Parameter(torch.randn(3 * 16 * 64, 8))

    converted = maybe_convert_to_smaller_parts(
        model_info,
        "model.visual.blocks.0.attn.qkv.weight",
        original,
    )

    assert converted["model.visual.blocks.0.attn.qkv.weight"].shape == (3 * 16, 64, 8)


def test_visual_qkv_layout_aligns_fsdp_and_vllm_tp_shards():
    num_heads = 16
    head_size = 72
    hidden_size = 2
    full_2d = torch.arange(3 * num_heads * head_size * hidden_size, dtype=torch.float32).reshape(
        3 * num_heads * head_size,
        hidden_size,
    )
    full_canonical = reshape_visual_block_qkv(Parameter(full_2d), vision_head_size=head_size)
    finest_mesh = OrderedDict([(0, 24)])

    def refine(tensor, sharding):
        sharding.refactor_based_on_finer_shard_mesh(finest_mesh)
        return dict(zip(sharding.shard_indices, sharding.get_local_sharded_tensors(tensor)))

    expected = refine(full_canonical, NIXLSharding.default())

    fsdp_shards = {}
    for rank, local_2d in enumerate(full_2d.chunk(8, dim=0)):
        local = reshape_visual_block_qkv(Parameter(local_2d), vision_head_size=head_size)
        sharding = NIXLSharding(shard_mesh=OrderedDict([(0, 8)]), shard_indices=[(rank,)])
        fsdp_shards.update(refine(local, sharding))

    vllm_shards = {}
    full_qkv = full_2d.reshape(3, num_heads, head_size, hidden_size)
    for rank in range(4):
        local_2d = torch.cat(
            [full_qkv[qkv, rank * 4 : (rank + 1) * 4].reshape(-1, hidden_size) for qkv in range(3)],
            dim=0,
        )
        local = reshape_visual_block_qkv(Parameter(local_2d), vision_head_size=head_size)
        sharding = make_visual_qkv_tp_sharding(tp_size=4, tp_rank=rank)
        vllm_shards.update(refine(local, sharding))

    assert fsdp_shards.keys() == expected.keys()
    assert vllm_shards.keys() == expected.keys()
    for index, expected_tensor in expected.items():
        assert torch.equal(fsdp_shards[index], expected_tensor)
        assert torch.equal(vllm_shards[index], expected_tensor)


def test_dtype_cast_sync_refreshes_before_push_and_writes_back_after_pull():
    source = torch.tensor([1.25, -2.5], dtype=torch.bfloat16)
    exposed = torch.zeros(2, dtype=torch.float32)
    state_dict = {"model.layers.0.linear_attn.A_log": exposed}
    plan = ParamSyncPlan([DTypeCastSync(key="model.layers.0.linear_attn.A_log", source_param=source)])

    plan.before_push(state_dict)
    assert torch.equal(exposed, source.float())

    exposed.copy_(torch.tensor([3.5, 4.25], dtype=torch.float32))
    plan.after_pull(state_dict)
    assert torch.equal(source, exposed.bfloat16())


def test_qwen35_output_gate_supports_fine_grained_fsdp_sharding():
    converter = _Converter(HFParameterMapping(_qwen35_config()))
    param = Parameter(torch.randn(1024, 2))
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, 8)]),
        shard_indices=[(3,)],
    )

    converted, converted_sharding = converter.maybe_reshape_qkv_to_3d(
        "model.language_model.layers.0.self_attn.q_proj.weight",
        param,
        sharding,
    )

    assert converted.shape == (1, 2, 2, 256, 2)
    assert converted_sharding.shard_mesh == OrderedDict([(0, 4), (1, 2)])
    assert converted_sharding.shard_indices == [(1, 1)]


def _reconstruct_fsdp_canonical_splits(param_name: str, full_param: Parameter):
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()
    shards_by_name: dict[str, dict[tuple[int, ...], torch.Tensor]] = {}
    meshes_by_name: dict[str, OrderedDict] = {}

    for rank, local_param in enumerate(full_param.chunk(8, dim=0)):
        source_sharding = NIXLSharding(
            shard_mesh=OrderedDict([(0, 8)]),
            shard_indices=[(rank,)],
        )
        split_result = split_dim0_fused_fsdp_param(
            local_param,
            source_sharding,
            [param_name + suffix for suffix in ("_q", "_k", "_v")],
            [model_info["linear_key_dim"], model_info["linear_key_dim"], model_info["linear_value_dim"]],
        )
        assert split_result is not None
        converted, converted_shardings = split_result
        for name, tensor in converted.items():
            assert tensor.untyped_storage().data_ptr() == full_param.untyped_storage().data_ptr()
            output_sharding = converted_shardings[name]
            meshes_by_name.setdefault(name, output_sharding.shard_mesh)
            assert meshes_by_name[name] == output_sharding.shard_mesh
            local_shards = output_sharding.get_local_sharded_tensors(tensor)
            shards_by_name.setdefault(name, {}).update(zip(output_sharding.shard_indices, local_shards))

    reconstructed = {
        name: torch.cat([shards[index] for index in sorted(shards)], dim=0) for name, shards in shards_by_name.items()
    }
    return reconstructed, meshes_by_name


def test_qwen35_fsdp_conv1d_split_matches_hf_canonical_layout():
    param_name = "model.language_model.layers.0.linear_attn.conv1d.weight"
    full_param = Parameter(torch.arange(8192 * 4, dtype=torch.float32).reshape(8192, 1, 4))
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()

    reconstructed, meshes = _reconstruct_fsdp_canonical_splits(param_name, full_param)
    expected = maybe_convert_to_smaller_parts(model_info, param_name, full_param)

    assert reconstructed.keys() == expected.keys()
    assert meshes[param_name + "_q"] == OrderedDict([(0, 2)])
    assert meshes[param_name + "_k"] == OrderedDict([(0, 2)])
    assert meshes[param_name + "_v"] == OrderedDict([(0, 4)])
    for name in expected:
        assert torch.equal(reconstructed[name], expected[name])


def test_qwen35_fsdp_in_proj_qkv_split_matches_hf_canonical_layout():
    param_name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    full_param = Parameter(torch.arange(8192 * 2, dtype=torch.float32).reshape(8192, 2))
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()

    reconstructed, meshes = _reconstruct_fsdp_canonical_splits(param_name, full_param)
    expected = maybe_convert_to_smaller_parts(model_info, param_name, full_param)

    assert reconstructed.keys() == expected.keys()
    assert meshes[param_name + "_q"] == OrderedDict([(0, 2)])
    assert meshes[param_name + "_k"] == OrderedDict([(0, 2)])
    assert meshes[param_name + "_v"] == OrderedDict([(0, 4)])
    for name in expected:
        assert torch.equal(reconstructed[name], expected[name])


def _reconstruct_fsdp_moe_splits(
    param_name: str,
    full_param: Parameter,
    shard_dim: int,
    shard_count: int = 4,
):
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()
    shards_by_name: dict[str, dict[tuple[int, ...], torch.Tensor]] = {}
    meshes_by_name: dict[str, OrderedDict] = {}

    for rank, local_param in enumerate(full_param.chunk(shard_count, dim=shard_dim)):
        source_sharding = NIXLSharding(
            shard_mesh=OrderedDict([(shard_dim, shard_count)]),
            shard_indices=[(rank,)],
        )
        split_result = split_fused_moe_fsdp_param(model_info, param_name, local_param, source_sharding)
        assert split_result is not None
        converted, converted_shardings = split_result
        for name, tensor in converted.items():
            assert tensor.untyped_storage().data_ptr() == full_param.untyped_storage().data_ptr()
            output_sharding = converted_shardings[name]
            meshes_by_name.setdefault(name, output_sharding.shard_mesh)
            assert meshes_by_name[name] == output_sharding.shard_mesh
            local_shards = output_sharding.get_local_sharded_tensors(tensor)
            shards_by_name.setdefault(name, {}).update(zip(output_sharding.shard_indices, local_shards))

    reconstructed = {
        name: torch.cat([shards[index] for index in sorted(shards)], dim=0) for name, shards in shards_by_name.items()
    }
    return reconstructed, meshes_by_name


@pytest.mark.parametrize(
    ("param_name", "full_param"),
    [
        (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            Parameter(torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(8, 8, 8)),
        ),
        (
            "model.language_model.layers.0.mlp.experts.down_proj",
            Parameter(torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4)),
        ),
    ],
)
def test_qwen35_fsdp_expert_axis_split_matches_hf_canonical_layout(param_name, full_param):
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()

    reconstructed, meshes = _reconstruct_fsdp_moe_splits(param_name, full_param, shard_dim=0)
    expected = maybe_convert_to_smaller_parts(model_info, param_name, full_param)

    assert reconstructed.keys() == expected.keys()
    assert all(mesh == OrderedDict([(0, 1)]) for mesh in meshes.values())
    for name in expected:
        assert torch.equal(reconstructed[name], expected[name])


@pytest.mark.parametrize(
    ("param_name", "full_param", "expected_shard_count"),
    [
        (
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            Parameter(torch.arange(8 * 8 * 8, dtype=torch.float32).reshape(8, 8, 8)),
            2,
        ),
        (
            "model.language_model.layers.0.mlp.experts.down_proj",
            Parameter(torch.arange(8 * 8 * 4, dtype=torch.float32).reshape(8, 8, 4)),
            4,
        ),
    ],
)
def test_qwen35_fsdp_projection_axis_split_matches_hf_canonical_layout(
    param_name,
    full_param,
    expected_shard_count,
):
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()

    reconstructed, meshes = _reconstruct_fsdp_moe_splits(param_name, full_param, shard_dim=1)
    expected = maybe_convert_to_smaller_parts(model_info, param_name, full_param)

    assert reconstructed.keys() == expected.keys()
    assert all(mesh == OrderedDict([(0, expected_shard_count)]) for mesh in meshes.values())
    for name in expected:
        assert torch.equal(reconstructed[name], expected[name])


@pytest.mark.parametrize("projection", ["gate_up_proj", "down_proj"])
def test_qwen35_hf_fused_expert_mapping_accepts_optional_weight_suffix(projection):
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()
    prefix = "model.language_model.layers.0.mlp.experts"
    if projection == "gate_up_proj":
        param = Parameter(torch.randn(8, 8, 8))
        expected_suffixes = {"gate_proj.weight", "up_proj.weight"}
    else:
        param = Parameter(torch.randn(8, 8, 4))
        expected_suffixes = {"down_proj.weight"}

    without_weight = maybe_convert_to_smaller_parts(model_info, f"{prefix}.{projection}", param)
    with_weight = maybe_convert_to_smaller_parts(model_info, f"{prefix}.{projection}.weight", param)

    expected_names = {f"{prefix}.{expert_id}.{suffix}" for expert_id in range(8) for suffix in expected_suffixes}
    assert without_weight.keys() == expected_names
    assert with_weight.keys() == expected_names
    for name in expected_names:
        assert torch.equal(without_weight[name], with_weight[name])


def test_qwen35_fsdp_expert_axis_split_rejects_uneven_expert_ownership():
    model_info = HFParameterMapping(_qwen35_config()).get_model_info()
    local_param = torch.randn(3, 8, 8)
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, 4)]),
        shard_indices=[(0,)],
    )

    with pytest.raises(ValueError, match="must divide num_experts evenly"):
        split_fused_moe_fsdp_param(
            model_info,
            "model.language_model.layers.0.mlp.experts.gate_up_proj",
            local_param,
            sharding,
        )
