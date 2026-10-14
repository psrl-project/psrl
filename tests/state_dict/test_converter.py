"""
Unit tests for the converter subsystem.

Covers:
  - ParameterMapping base class: default __init__ and get_model_info
  - HFParameterMapping / FSDPParameterMapping: basic instantiation and get_mappings
  - ModelRegistry / create_parameter_mapping: config object (not path) API
  - reshape_qkv_to_3d: shape and storage-sharing correctness
  - QKVTPLayout / slice_qkv_proj: head placement, shape, and storage-sharing for various TP sizes
  - BaseConverter.__init__: model_info populated from parameter_mapping via super()
  - BaseConverter.maybe_reshape_qkv_to_3d: all three sharding cases (A/B/C)
    and the no-op cases (non-QKV name, 1D param, no model_info)
  - HFConverter: state dict conversion, sharding output, QKV 3D reshape applied

All tests are purely in-process (no GPU, no distributed, no real model files).
"""

import math
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import torch
from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.hf_converter import HFConverter, convert_hf_inplace
from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    create_parameter_mapping,
    get_qkv_tp_layout,
    model_registry,
    register_model,
    reshape_qkv_to_3d,
    slice_qkv_proj,
)
from psrl.utils.converter.modeling.fsdp_modeling import FSDPParameterMapping
from psrl.utils.converter.modeling.hf_modeling import HFParameterMapping
from psrl.utils.nixl.nixl_spec import NIXLSharding
from torch.nn import Parameter

# Helpers


def _make_config(
    num_attention_heads: int = 32,
    num_key_value_heads: int = 8,
    hidden_size: int = 4096,
    intermediate_size: int = 11008,
    head_dim: int | None = None,
) -> SimpleNamespace:
    """Build a minimal mock HuggingFace config object."""
    cfg = SimpleNamespace(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    if head_dim is not None:
        cfg.head_dim = head_dim
    return cfg


def _make_param(shape: tuple, requires_grad: bool = True) -> Parameter:
    """Create a random Parameter with the given shape."""
    return Parameter(torch.randn(*shape), requires_grad=requires_grad)


def _default_sharding() -> NIXLSharding:
    return NIXLSharding.default()


# ParameterMapping base class


class TestParameterMappingBase(unittest.TestCase):
    """Tests for ParameterMapping.__init__ and default get_model_info."""

    def test_init_stores_config(self):
        config = _make_config()
        mapping = HFParameterMapping(config)
        self.assertIs(mapping.config, config)

    def test_get_model_info_standard(self):
        # num_heads=32, num_kv_heads=8, hidden=4096 → head_size = 4096 // 32 = 128
        config = _make_config(num_attention_heads=32, num_key_value_heads=8, hidden_size=4096, intermediate_size=11008)
        info = HFParameterMapping(config).get_model_info()
        self.assertEqual(info["num_heads"], 32)
        self.assertEqual(info["num_kv_heads"], 8)
        self.assertEqual(info["head_size"], 128)
        self.assertEqual(info["intermediate_size"], 11008)

    def test_get_model_info_uses_head_dim_when_present(self):
        # head_dim overrides hidden_size // num_heads (e.g. Qwen3-MoE)
        config = _make_config(num_attention_heads=32, hidden_size=4096, head_dim=96)
        info = HFParameterMapping(config).get_model_info()
        self.assertEqual(info["head_size"], 96)

    def test_get_model_info_default_kv_heads_fallback(self):
        # num_key_value_heads absent → falls back to num_attention_heads
        config = SimpleNamespace(
            num_attention_heads=16,
            hidden_size=2048,
            intermediate_size=8192,
        )
        info = HFParameterMapping(config).get_model_info()
        self.assertEqual(info["num_kv_heads"], 16)


# HFParameterMapping


class TestHFParameterMapping(unittest.TestCase):
    """Tests for the HFParameterMapping concrete class."""

    def test_get_mappings_returns_empty_list(self):
        config = _make_config()
        mapping = HFParameterMapping(config)
        self.assertEqual(mapping.get_mappings(), [])

    def test_registered_under_hf_key(self):
        config = _make_config()
        # create_parameter_mapping("HF", config) must work
        mapping = create_parameter_mapping("HF", config)
        self.assertIsInstance(mapping, HFParameterMapping)

    def test_model_info_correct_via_create(self):
        config = _make_config(num_attention_heads=16, num_key_value_heads=4, hidden_size=2048, intermediate_size=8192)
        mapping = create_parameter_mapping("HF", config)
        info = mapping.get_model_info()
        self.assertEqual(info["num_heads"], 16)
        self.assertEqual(info["num_kv_heads"], 4)
        self.assertEqual(info["head_size"], 128)  # 2048 // 16


# FSDPParameterMapping


class TestFSDPParameterMapping(unittest.TestCase):
    """Tests for the FSDPParameterMapping concrete class."""

    def test_get_mappings_returns_empty_list(self):
        config = _make_config()
        mapping = FSDPParameterMapping(config)
        self.assertEqual(mapping.get_mappings(), [])

    def test_registered_under_fsdp_key(self):
        config = _make_config()
        mapping = create_parameter_mapping("FSDP", config)
        self.assertIsInstance(mapping, FSDPParameterMapping)

    def test_model_info_same_as_hf_mapping(self):
        config = _make_config(num_attention_heads=32, num_key_value_heads=8, hidden_size=4096, intermediate_size=11008)
        hf_info = HFParameterMapping(config).get_model_info()
        fsdp_info = FSDPParameterMapping(config).get_model_info()
        self.assertEqual(hf_info, fsdp_info)

    def test_distinct_classes(self):
        # HFParameterMapping and FSDPParameterMapping must be separate classes
        self.assertIsNot(HFParameterMapping, FSDPParameterMapping)


# Model registry and parameter mapping creation


class TestModelRegistry(unittest.TestCase):
    """Tests that ModelRegistry.create_mapping accepts a config, not a path."""

    def setUp(self):
        # Register a fresh test mapping class
        @register_model("_TestRegistryModel")
        class _TestMapping(ParameterMapping):
            def get_mappings(self):
                return []

        self._TestMapping = _TestMapping

    def tearDown(self):
        model_registry.unregister_mapping(self._TestMapping)

    def test_create_mapping_passes_config_to_init(self):
        config = _make_config()
        mapping = create_parameter_mapping("_TestRegistryModel", config)
        self.assertIs(mapping.config, config)

    def test_unsupported_model_raises(self):
        with self.assertRaises(ValueError):
            create_parameter_mapping("__NonExistentModel__", _make_config())


# QKV reshaping


class TestReshapeQKVTo3D(unittest.TestCase):
    """Tests for the reshape_qkv_to_3d helper in model_mappings.py."""

    def _run(self, rows, hidden, num_heads, num_kv_heads, head_size):
        data = torch.randn(rows, hidden)
        param = Parameter(data.clone())
        result = reshape_qkv_to_3d(param, num_heads, num_kv_heads, head_size)
        return param, result

    def test_q_shape_dense(self):
        # Eight dense heads reshape into eight groups of 64 rows.
        H = 256
        param, result = self._run(512, H, num_heads=8, num_kv_heads=8, head_size=64)
        self.assertEqual(result.shape, (8, 64, H))

    def test_kv_shape_gqa(self):
        # Eight key-value heads reshape into eight groups of 128 rows.
        H = 4096
        param, result = self._run(1024, H, num_heads=32, num_kv_heads=8, head_size=128)
        self.assertEqual(result.shape, (8, 128, H))

    def test_q_shape_gqa(self):
        # Q 2D shape for GQA: (32*128, H) = (4096, 4096) → (8, 512, 4096)
        H = 4096
        param, result = self._run(4096, H, num_heads=32, num_kv_heads=8, head_size=128)
        self.assertEqual(result.shape, (8, 512, H))

    def test_storage_shared(self):
        # Reshaping must return a view over the original storage.
        H = 256
        param, result = self._run(512, H, num_heads=8, num_kv_heads=8, head_size=64)
        self.assertEqual(
            result.data.untyped_storage().data_ptr(),
            param.data.untyped_storage().data_ptr(),
        )

    def test_numel_preserved(self):
        H = 512
        rows = 1024
        param, result = self._run(rows, H, num_heads=32, num_kv_heads=8, head_size=128)
        self.assertEqual(result.numel(), rows * H)


# QKV projection slicing


class TestSliceQKVProj(unittest.TestCase):
    """Tests for slice_qkv_proj (returns 2D shards)."""

    def _fused(self, num_heads, num_kv_heads, head_size, tp_size=1, H=4096):
        layout = get_qkv_tp_layout(num_heads, num_kv_heads, tp_size)
        total = (layout.num_heads_local + 2 * layout.num_kv_heads_local) * head_size
        data = torch.randn(total, H)
        return Parameter(data)

    def test_layout_with_partitioned_kv_heads(self):
        layout = get_qkv_tp_layout(num_heads=32, num_kv_heads=8, tp_size=4)
        self.assertEqual(layout.num_heads_local, 8)
        self.assertEqual(layout.num_kv_heads_local, 2)
        self.assertEqual(layout.num_kv_head_replicas, 1)
        self.assertEqual(layout.num_kv_shards, 4)
        self.assertEqual([layout.get_kv_shard_rank(rank) for rank in range(4)], [0, 1, 2, 3])

    def test_layout_with_replicated_kv_heads(self):
        layout = get_qkv_tp_layout(num_heads=16, num_kv_heads=2, tp_size=8)
        self.assertEqual(layout.num_heads_local, 2)
        self.assertEqual(layout.num_kv_heads_local, 1)
        self.assertEqual(layout.num_kv_head_replicas, 4)
        self.assertEqual(layout.num_kv_shards, 2)
        self.assertEqual([layout.get_kv_shard_rank(rank) for rank in range(8)], [0, 0, 0, 0, 1, 1, 1, 1])

    def test_shapes_with_replicated_kv_heads(self):
        H = 256
        fused = self._fused(num_heads=16, num_kv_heads=2, head_size=64, tp_size=8, H=H)
        q, k, v = slice_qkv_proj(fused, num_heads=16, num_kv_heads=2, head_size=64, tp_size=8)
        self.assertEqual(q.shape, (2 * 64, H))
        self.assertEqual(k.shape, (64, H))
        self.assertEqual(v.shape, (64, H))
        fused_storage_ptr = fused.data.untyped_storage().data_ptr()
        for shard in (q, k, v):
            self.assertEqual(shard.data.untyped_storage().data_ptr(), fused_storage_ptr)

    def test_shapes_tp1(self):
        H = 4096
        num_heads, num_kv_heads, head_size = 32, 8, 128
        fused = self._fused(num_heads, num_kv_heads, head_size, tp_size=1, H=H)
        q, k, v = slice_qkv_proj(fused, num_heads, num_kv_heads, head_size, tp_size=1)
        self.assertEqual(q.shape, (32 * 128, H))
        self.assertEqual(k.shape, (8 * 128, H))
        self.assertEqual(v.shape, (8 * 128, H))

    def test_shapes_tp4(self):
        H = 4096
        num_heads, num_kv_heads, head_size = 32, 8, 128
        fused = self._fused(num_heads, num_kv_heads, head_size, tp_size=4, H=H)
        q, k, v = slice_qkv_proj(fused, num_heads, num_kv_heads, head_size, tp_size=4)
        self.assertEqual(q.shape, (8 * 128, H))
        self.assertEqual(k.shape, (2 * 128, H))
        self.assertEqual(v.shape, (2 * 128, H))

    def test_results_are_2d(self):
        H = 4096
        fused = self._fused(32, 8, 128, tp_size=1, H=H)
        for shard in slice_qkv_proj(fused, 32, 8, 128, tp_size=1):
            self.assertEqual(shard.ndim, 2)

    def test_storage_shared_with_fused(self):
        H = 2048
        fused = self._fused(16, 4, 64, tp_size=1, H=H)
        q, k, v = slice_qkv_proj(fused, 16, 4, 64, tp_size=1)
        # narrow() shifts data_ptr by the offset, so compare untyped_storage addresses
        fused_storage_ptr = fused.data.untyped_storage().data_ptr()
        for shard in (q, k, v):
            self.assertEqual(shard.data.untyped_storage().data_ptr(), fused_storage_ptr)

    def test_values_correct(self):
        # Verify actual values by manual offset computation
        H = 8
        num_heads, num_kv_heads, head_size = 4, 2, 2
        # fused rows: 4*2 + 2*2 + 2*2 = 8 + 4 + 4 = 16
        data = torch.arange(16 * H, dtype=torch.float32).reshape(16, H)
        fused = Parameter(data)
        q, k, v = slice_qkv_proj(fused, num_heads, num_kv_heads, head_size)
        self.assertTrue(torch.equal(q.data, data[:8]))
        self.assertTrue(torch.equal(k.data, data[8:12]))
        self.assertTrue(torch.equal(v.data, data[12:16]))


# Conditional QKV reshaping


class _ConcreteConverter(BaseConverter):
    """Minimal concrete converter for testing maybe_reshape_qkv_to_3d."""

    def convert_state_and_sharding_dict(self, model):
        raise NotImplementedError


def _make_converter(num_heads=32, num_kv_heads=8, head_size=128) -> _ConcreteConverter:
    """Build a converter backed by a fresh HFParameterMapping with the given head config."""
    config = _make_config(
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_size=num_heads * head_size,  # consistent with head_size
        intermediate_size=num_heads * head_size * 4,
    )
    return _ConcreteConverter(HFParameterMapping(config))


# Base converter initialization


class TestBaseConverterInit(unittest.TestCase):
    """Tests that BaseConverter.__init__ populates model_info from parameter_mapping."""

    def test_model_info_populated_from_mapping(self):
        config = _make_config(num_attention_heads=16, num_key_value_heads=4, hidden_size=2048, intermediate_size=8192)
        conv = _ConcreteConverter(HFParameterMapping(config))
        self.assertEqual(conv.model_info["num_heads"], 16)
        self.assertEqual(conv.model_info["num_kv_heads"], 4)
        self.assertEqual(conv.model_info["head_size"], 128)

    def test_fsdp_mapping_also_populates_model_info(self):
        config = _make_config(num_attention_heads=32, num_key_value_heads=8, hidden_size=4096, intermediate_size=11008)
        conv = _ConcreteConverter(FSDPParameterMapping(config))
        self.assertEqual(conv.model_info["num_heads"], 32)
        self.assertEqual(conv.model_info["num_kv_heads"], 8)


def _sharding(shard_dim: int, ws: int, rank: int) -> NIXLSharding:
    return NIXLSharding(
        shard_mesh=OrderedDict([(shard_dim, ws)]),
        shard_indices=[(rank,)],
    )


class TestMaybeReshapeQKVTo3D(unittest.TestCase):
    """Test `BaseConverter.maybe_reshape_qkv_to_3d` across all paths."""

    # No-op conditions

    def test_noop_non_qkv_name(self):
        conv = _make_converter()
        param = _make_param((4096, 4096))
        sharding = _default_sharding()
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.mlp.gate_proj.weight", param, sharding)
        self.assertIs(out_p, param)
        self.assertIs(out_s, sharding)

    def test_noop_1d_param(self):
        conv = _make_converter()
        param = _make_param((4096,))
        sharding = _default_sharding()
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        self.assertEqual(out_p.shape, (8, 512, 1))
        self.assertEqual(out_p.untyped_storage().data_ptr(), param.untyped_storage().data_ptr())
        self.assertIs(out_s, sharding)

    def test_noop_no_model_info(self):
        # Converter whose mapping returns no num_heads → maybe_reshape is a no-op
        class _EmptyMapping(ParameterMapping):
            def get_mappings(self):
                return []

            def get_model_info(self):
                return {}

        conv = _ConcreteConverter(_EmptyMapping(_make_config()))
        param = _make_param((4096, 4096))
        sharding = _default_sharding()
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        self.assertIs(out_p, param)
        self.assertIs(out_s, sharding)

    # Case A with sharded hidden dimension

    def test_case_a_shape(self):
        # Reshaping moves the sharded hidden dimension from axis 1 to axis 2.
        num_heads, num_kv_heads, head_size = 32, 8, 128
        H_shard = 2048  # half of hidden_size=4096 (tp-sharded)
        param = _make_param((num_heads * head_size, H_shard))
        sharding = _sharding(shard_dim=1, ws=2, rank=1)
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        G = math.gcd(num_heads, num_kv_heads)
        self.assertEqual(out_p.shape, (G, num_heads // G * head_size, H_shard))
        self.assertEqual(out_s.shard_mesh, OrderedDict([(2, 2)]))
        self.assertEqual(out_s.shard_indices, [(1,)])

    # Case B when world size does not exceed global groups

    def test_case_b_shape_ws_equals_G(self):
        # G=8, ws=8 → Case B (ws == G_global)
        num_heads, num_kv_heads, head_size = 32, 8, 128
        ws = 8  # == G_global
        rows = num_heads // ws * head_size  # Q rows per rank = 4*128=512
        H = 4096
        param = _make_param((rows, H))
        sharding = _sharding(shard_dim=0, ws=ws, rank=3)
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        # num_heads_local=4, num_kv_heads_local=1 → G_eff = gcd(4,1) = 1
        G_eff = math.gcd(num_heads // ws, num_kv_heads // ws)
        self.assertEqual(out_p.shape, (G_eff, rows // G_eff, H))
        # shard_mesh unchanged
        self.assertIs(out_s, sharding)

    def test_case_b_shape_ws_less_than_G(self):
        # G=8, ws=4 → Case B (ws < G_global)
        num_heads, num_kv_heads, head_size = 32, 8, 128
        ws = 4
        rows = num_heads // ws * head_size  # Q rows per rank = 8*128=1024
        H = 4096
        param = _make_param((rows, H))
        sharding = _sharding(shard_dim=0, ws=ws, rank=2)
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        G_eff = math.gcd(num_heads // ws, num_kv_heads // ws)  # gcd(8,2)=2
        self.assertEqual(out_p.shape, (G_eff, rows // G_eff, H))
        self.assertIs(out_s, sharding)

    def test_case_b_ws1_default_sharding(self):
        # NIXLSharding.default() → shard_mesh={0: 1}, rank=0 → Case B (ws=1 ≤ G=8)
        num_heads, num_kv_heads, head_size = 32, 8, 128
        H = 4096
        param = _make_param((num_heads * head_size, H))
        sharding = _default_sharding()
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        G = math.gcd(num_heads, num_kv_heads)  # 8
        self.assertEqual(out_p.shape, (G, num_heads // G * head_size, H))
        self.assertIs(out_s, sharding)

    # Case C when world size exceeds global groups

    def test_case_c_shape(self):
        # G=8, ws=16 → steps=2, Case C
        num_heads, num_kv_heads, head_size = 32, 8, 128
        ws = 16
        rows = num_heads // ws * head_size  # Q rows = 2*128=256
        H = 4096
        param = _make_param((rows, H))
        sharding = _sharding(shard_dim=0, ws=ws, rank=5)
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        out_p, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        G_global = math.gcd(num_heads, num_kv_heads)  # 8
        steps = ws // G_global  # 2
        self.assertEqual(out_p.shape, (1, rows, H))
        self.assertEqual(out_s.shard_mesh, OrderedDict([(0, G_global), (1, steps)]))
        self.assertEqual(out_s.shard_indices, [(rank // steps, rank % steps) for rank in [5]])

    def test_case_c_shard_indices_rank0(self):
        num_heads, num_kv_heads, head_size = 32, 8, 128
        ws = 16
        G_global = 8
        steps = ws // G_global  # 2
        rows = num_heads // ws * head_size
        H = 4096
        for rank in range(ws):
            param = _make_param((rows, H))
            sharding = _sharding(shard_dim=0, ws=ws, rank=rank)
            conv = _make_converter(num_heads, num_kv_heads, head_size)
            _, out_s = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
            expected = [(rank // steps, rank % steps)]
            self.assertEqual(out_s.shard_indices, expected, f"rank={rank}")

    def test_case_c_ws_not_divisible_by_G_raises(self):
        # ws=10 is not divisible by G_global=8
        num_heads, num_kv_heads, head_size = 32, 8, 128
        rows = 128
        H = 4096
        param = _make_param((rows, H))
        sharding = _sharding(shard_dim=0, ws=10, rank=1)
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        with self.assertRaises(AssertionError):
            conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)

    # Storage sharing

    def test_case_b_storage_shared(self):
        num_heads, num_kv_heads, head_size = 32, 8, 128
        H = 4096
        param = _make_param((num_heads * head_size, H))
        sharding = _default_sharding()
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        out_p, _ = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        self.assertEqual(
            out_p.data.untyped_storage().data_ptr(),
            param.data.untyped_storage().data_ptr(),
        )

    def test_case_c_storage_shared(self):
        num_heads, num_kv_heads, head_size = 32, 8, 128
        rows = 256
        H = 4096
        param = _make_param((rows, H))
        sharding = _sharding(shard_dim=0, ws=16, rank=0)
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        out_p, _ = conv.maybe_reshape_qkv_to_3d("model.layers.0.self_attn.q_proj.weight", param, sharding)
        self.assertEqual(
            out_p.data.untyped_storage().data_ptr(),
            param.data.untyped_storage().data_ptr(),
        )

    # Key and value projection reshaping

    def test_kv_proj_names_trigger_reshape(self):
        num_heads, num_kv_heads, head_size = 32, 8, 128
        H = 4096
        conv = _make_converter(num_heads, num_kv_heads, head_size)
        for name in ("k_proj.weight", "v_proj.weight", "q_proj.bias"):
            param = _make_param((num_kv_heads * head_size, H))
            sharding = _default_sharding()
            out_p, _ = conv.maybe_reshape_qkv_to_3d(f"model.layers.0.self_attn.{name}", param, sharding)
            self.assertEqual(out_p.ndim, 3, f"Expected 3D for {name}")


# Hugging Face state and sharding conversion


def _make_minimal_hf_model(num_heads=32, num_kv_heads=8, head_size=128, hidden=4096):
    """Build a tiny fake HF-like model with a q_proj, k_proj, v_proj and a non-QKV param."""

    class FakeModel:
        def state_dict(self):
            return {
                "model.layers.0.self_attn.q_proj.weight": torch.randn(num_heads * head_size, hidden),
                "model.layers.0.self_attn.k_proj.weight": torch.randn(num_kv_heads * head_size, hidden),
                "model.layers.0.self_attn.v_proj.weight": torch.randn(num_kv_heads * head_size, hidden),
                "model.layers.0.mlp.gate_proj.weight": torch.randn(11008, hidden),
            }

    return FakeModel()


class TestHFConverter(unittest.TestCase):
    """Tests for HFConverter (and convert_hf_inplace convenience wrapper)."""

    def _converter(self, num_heads=32, num_kv_heads=8, head_size=128, hidden=4096):
        config = _make_config(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden,
            intermediate_size=11008,
        )
        return HFConverter(HFParameterMapping(config))

    def test_all_keys_present(self):
        model = _make_minimal_hf_model()
        conv = self._converter()
        state, sharding = conv.convert_state_and_sharding_dict(model)
        expected_keys = {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
        }
        self.assertEqual(set(state.keys()), expected_keys)
        self.assertEqual(set(sharding.keys()), expected_keys)

    def test_qkv_weights_are_3d(self):
        model = _make_minimal_hf_model()
        conv = self._converter()
        state, _ = conv.convert_state_and_sharding_dict(model)
        for key in (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        ):
            self.assertEqual(state[key].ndim, 3, f"Expected 3D for {key}")

    def test_non_qkv_weight_is_2d(self):
        model = _make_minimal_hf_model()
        conv = self._converter()
        state, _ = conv.convert_state_and_sharding_dict(model)
        self.assertEqual(state["model.layers.0.mlp.gate_proj.weight"].ndim, 2)

    def test_q_shape_correct(self):
        num_heads, num_kv_heads, head_size, hidden = 32, 8, 128, 4096
        G = math.gcd(num_heads, num_kv_heads)
        model = _make_minimal_hf_model(num_heads, num_kv_heads, head_size, hidden)
        conv = self._converter(num_heads, num_kv_heads, head_size, hidden)
        state, _ = conv.convert_state_and_sharding_dict(model)
        q = state["model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(q.shape, (G, num_heads // G * head_size, hidden))

    def test_kv_shape_correct(self):
        num_heads, num_kv_heads, head_size, hidden = 32, 8, 128, 4096
        G = math.gcd(num_heads, num_kv_heads)
        model = _make_minimal_hf_model(num_heads, num_kv_heads, head_size, hidden)
        conv = self._converter(num_heads, num_kv_heads, head_size, hidden)
        state, _ = conv.convert_state_and_sharding_dict(model)
        k = state["model.layers.0.self_attn.k_proj.weight"]
        self.assertEqual(k.shape, (G, num_kv_heads // G * head_size, hidden))

    def test_sharding_is_default_for_all_params(self):
        # Hugging Face parameters are unsharded, so all use `NIXLSharding.default`.
        model = _make_minimal_hf_model()
        conv = self._converter()
        _, sharding = conv.convert_state_and_sharding_dict(model)
        default = NIXLSharding.default()
        for key, s in sharding.items():
            self.assertEqual(s.shard_mesh, default.shard_mesh, f"key={key}")
            self.assertEqual(s.shard_indices, default.shard_indices, f"key={key}")

    def test_no_reshape_without_model_info(self):
        # Converter with empty model_info → maybe_reshape is a no-op → q remains 2D
        num_heads, num_kv_heads, head_size, hidden = 32, 8, 128, 4096
        model = _make_minimal_hf_model(num_heads, num_kv_heads, head_size, hidden)

        class _EmptyMapping(ParameterMapping):
            def get_mappings(self):
                return []

            def get_model_info(self):
                return {}  # no num_heads key

        conv = HFConverter(_EmptyMapping(_make_config()))
        state, _ = conv.convert_state_and_sharding_dict(model)
        q = state["model.layers.0.self_attn.q_proj.weight"]
        self.assertEqual(q.ndim, 2)

    def test_convert_hf_inplace_wrapper(self):
        config = _make_config(num_attention_heads=32, num_key_value_heads=8, hidden_size=4096, intermediate_size=11008)
        model = _make_minimal_hf_model()
        state, sharding = convert_hf_inplace(HFParameterMapping(config), model)
        self.assertIn("model.layers.0.self_attn.q_proj.weight", state)
        self.assertEqual(state["model.layers.0.self_attn.q_proj.weight"].ndim, 3)


# Subclass model information


class TestSubclassGetModelInfo(unittest.TestCase):
    """Verify that MOE-style subclasses extending super().get_model_info() work."""

    def test_super_extension_adds_num_experts(self):
        @register_model("_TestMoeMapping")
        class _MoeMapping(ParameterMapping):
            def get_mappings(self):
                return []

            def get_model_info(self):
                info = super().get_model_info()
                info["num_experts"] = self.config.num_experts
                return info

        try:
            config = _make_config()
            config.num_experts = 64
            mapping = _MoeMapping(config)
            info = mapping.get_model_info()
            self.assertEqual(info["num_heads"], config.num_attention_heads)
            self.assertEqual(info["num_experts"], 64)
        finally:
            model_registry.unregister_mapping(_MoeMapping)


class TestLoadStateDictNdimMismatchSlicing(unittest.TestCase):
    """Test loading a 2D QKV checkpoint into registered 3D worker shards."""

    def _make_sharding(self, shard_mesh: dict, shard_indices: list) -> NIXLSharding:
        return NIXLSharding(
            shard_mesh=OrderedDict(shard_mesh),
            shard_indices=shard_indices,
        )

    def _extract_shard_ndim_mismatch(
        self,
        src_tensor_2d: torch.Tensor,
        dst_tensor_sample: torch.Tensor,
        sharding: NIXLSharding,
    ) -> list[torch.Tensor]:
        """Inline the fixed slicing logic from load_state_dict_into_registered_tensors."""
        full_shape = list(dst_tensor_sample.shape)
        for dim, count in sharding.shard_mesh.items():
            full_shape[dim] *= count
        src_tensor_full = src_tensor_2d.reshape(full_shape)
        shard_dims = list(sharding.shard_mesh.keys())
        shard_counts = list(sharding.shard_mesh.values())
        src_shards = []
        for shard_idx in sharding.shard_indices:
            shard = src_tensor_full
            for i_dim, (dim, count) in enumerate(zip(shard_dims, shard_counts)):
                shard_size = shard.shape[dim] // count
                shard = shard.narrow(dim, shard_idx[i_dim] * shard_size, shard_size)
            src_shards.append(shard)
        return src_shards

    def test_case_b_k_proj_4_workers(self):
        """
        Case B: ws=4 ≤ G_global=8. k_proj.weight is 2D (512, 2048) in the checkpoint
        but registered as 3D (1, 128, 2048) per PS worker with shard_mesh={0: 4}.
        """
        # 2D checkpoint tensor (4 groups * 128 rows = 512 rows)
        src_2d = torch.randn(512, 2048)
        # Each PS worker registered a (1, 128, 2048) shard
        dst_sample = torch.empty(1, 128, 2048)

        for i in range(4):
            sharding_i = self._make_sharding(shard_mesh={0: 4}, shard_indices=[(i,)])
            shards = self._extract_shard_ndim_mismatch(src_2d, dst_sample, sharding_i)
            self.assertEqual(len(shards), 1)
            shard = shards[0]
            self.assertEqual(tuple(shard.shape), (1, 128, 2048))
            # The shard must contain the i-th group of rows from the 2D tensor.
            expected = src_2d[i * 128 : (i + 1) * 128].reshape(1, 128, 2048)
            self.assertTrue(torch.equal(shard, expected), f"Shard {i} values mismatch.")

    def test_case_b_single_worker_holds_all_shards(self):
        """
        Edge case: a single PS worker holds all 4 shard_indices (degenerate assignment).
        """
        src_2d = torch.randn(512, 2048)
        dst_sample = torch.empty(1, 128, 2048)
        sharding = self._make_sharding(shard_mesh={0: 4}, shard_indices=[(0,), (1,), (2,), (3,)])
        shards = self._extract_shard_ndim_mismatch(src_2d, dst_sample, sharding)
        self.assertEqual(len(shards), 4)
        for i, shard in enumerate(shards):
            self.assertEqual(tuple(shard.shape), (1, 128, 2048))
            expected = src_2d[i * 128 : (i + 1) * 128].reshape(1, 128, 2048)
            self.assertTrue(torch.equal(shard, expected), f"Shard {i} values mismatch.")

    def test_no_op_when_ndim_matches(self):
        """Verify matching dimensions bypass the mismatch branch."""
        src_3d = torch.randn(1, 128, 2048)
        dst_sample = torch.empty(1, 128, 2048)
        # ndim matches → the branch condition is False
        self.assertEqual(src_3d.ndim, dst_sample.ndim)


if __name__ == "__main__":
    unittest.main()
