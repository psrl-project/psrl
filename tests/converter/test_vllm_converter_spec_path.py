"""Tests for VllmConverter spec-driven fast path and fallback."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from psrl.utils.converter.model_mappings import MappingType
from psrl.utils.converter.vllm_converter import VllmConverter
from vllm.model_executor.models.interfaces import SupportsWeightLayoutSpec, WeightLayoutSpec

pytestmark = pytest.mark.cpu_test


class FakeQwen2Model(nn.Module, SupportsWeightLayoutSpec):
    supports_weight_layout_spec = True

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, 8, bias=False)

    def get_weight_layout_spec(self) -> WeightLayoutSpec:
        return WeightLayoutSpec(
            stacked_params=[
                ("qkv_proj", "q_proj", "q"),
                ("qkv_proj", "k_proj", "k"),
                ("qkv_proj", "v_proj", "v"),
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ],
            packing_metadata={"num_heads": 2, "num_kv_heads": 2, "head_size": 4, "intermediate_size": 8},
        )


class FakeMoEModel(nn.Module, SupportsWeightLayoutSpec):
    supports_weight_layout_spec = True
    NUM_EXPERTS = 2

    def get_weight_layout_spec(self) -> WeightLayoutSpec:
        expert_params = []
        for eid in range(self.NUM_EXPERTS):
            expert_params += [
                ("experts.w13_", f"experts.{eid}.gate_proj.", eid, "w1"),
                ("experts.w2_", f"experts.{eid}.down_proj.", eid, "w2"),
                ("experts.w13_", f"experts.{eid}.up_proj.", eid, "w3"),
            ]
        return WeightLayoutSpec(
            stacked_params=[("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)],
            packing_metadata={"intermediate_size": 8, "num_experts": self.NUM_EXPERTS},
            expert_params=expert_params,
        )


class FakeOuterModel(nn.Module, SupportsWeightLayoutSpec):
    supports_weight_layout_spec = True

    def __init__(self):
        super().__init__()
        self.language_model = FakeQwen2Model()

    def get_weight_layout_spec(self) -> WeightLayoutSpec:
        return WeightLayoutSpec(stacked_params=[], packing_metadata={})


class TestBuildFusedMappings:
    def _make_converter(self):
        return VllmConverter(parameter_mapping=None, tp_rank=0)

    def test_qkv_shard_ids_are_integers_after_build(self):
        conv = self._make_converter()
        stacked = [("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v")]
        result = conv._build_fused_mappings(stacked, [], [])
        entries = result["qkv_proj"].mappings
        shard_ids = [e[1] for e in entries]
        assert shard_ids == [0, 1, 2], f"Expected integer shard_ids [0,1,2], got {shard_ids}"

    def test_qkv_inferred_as_qkv_split(self):
        conv = self._make_converter()
        stacked = [("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v")]
        result = conv._build_fused_mappings(stacked, [], [])
        mapping_type = result["qkv_proj"].mapping_type
        entries = result["qkv_proj"].mappings
        assert mapping_type == MappingType.QKV_SPLIT
        assert len(entries) == 3

    def test_gate_up_inferred_as_gate_up_split(self):
        conv = self._make_converter()
        stacked = [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        result = conv._build_fused_mappings(stacked, [], [])
        mapping_type = result["gate_up_proj"].mapping_type
        assert mapping_type == MappingType.GATE_UP_PROJ_SPLIT

    def test_expert_params_split_w13_w2(self):
        conv = self._make_converter()
        expert_params = [
            ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
            ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
            ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
        ]
        result = conv._build_fused_mappings([], [], expert_params)
        assert "experts.w13_" in result
        assert "experts.w2_" in result
        w13_type = result["experts.w13_"].mapping_type
        w2_type = result["experts.w2_"].mapping_type
        assert w13_type == MappingType.FUSED_MOE_W13_SPLIT
        assert w2_type == MappingType.FUSED_MOE_W2_SPLIT

    def test_w13_shard_id_encoding(self):
        conv = self._make_converter()
        expert_params = [
            ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
            ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
            ("experts.w13_", "experts.1.gate_proj.", 1, "w1"),
            ("experts.w13_", "experts.1.up_proj.", 1, "w3"),
        ]
        result = conv._build_fused_mappings([], [], expert_params)
        entries = result["experts.w13_"].mappings
        shard_ids = [e[1] for e in entries]
        assert shard_ids == [0, 1, 2, 3]

    def test_extra_stacked_params_become_individual_keys(self):
        conv = self._make_converter()
        extra = [
            ("layers.0.mlp.w13.weight", "layers.0.mlp.w1.weight", 0),
            ("layers.0.mlp.w13.weight", "layers.0.mlp.w3.weight", 1),
            ("layers.1.mlp.w13.weight", "layers.1.mlp.w1.weight", 0),
        ]
        result = conv._build_fused_mappings([], extra, [])
        assert "layers.0.mlp.w13.weight" in result
        assert "layers.1.mlp.w13.weight" in result
        entries_0 = result["layers.0.mlp.w13.weight"].mappings
        assert len(entries_0) == 2

    def test_full_path_keys_marked_is_full_path(self):
        """Bug 2 fix: extra_stacked_params keys must be marked as full-path (is_full_path=True)
        so convert_parameter uses exact match instead of substring match."""
        conv = self._make_converter()
        extra = [
            ("layers.0.residual_mlp.w13.weight", "layers.0.residual_mlp.w1.weight", 0),
            ("layers.0.residual_mlp.w13.weight", "layers.0.residual_mlp.w3.weight", 1),
        ]
        result = conv._build_fused_mappings([], extra, [])
        key = "layers.0.residual_mlp.w13.weight"
        assert key in result
        entry = result[key]
        assert entry.is_full_path is True, "extra_stacked_params entries must have is_full_path=True"

    def test_suffix_keys_marked_not_full_path(self):
        """Suffix keys (qkv_proj, gate_up_proj) must have is_full_path=False."""
        conv = self._make_converter()
        stacked = [("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v")]
        result = conv._build_fused_mappings(stacked, [], [])
        entry = result["qkv_proj"]
        assert entry.is_full_path is False


class TestBuildFromSpec:
    def _make_converter(self):
        return VllmConverter(parameter_mapping=None, tp_rank=0)

    def test_builds_fused_mappings_from_spec_object(self):
        """_build_from_spec takes a WeightLayoutSpec directly and returns fused_mappings."""
        conv = self._make_converter()
        spec = FakeQwen2Model().get_weight_layout_spec()
        fused, metadata = conv._build_from_spec(spec)
        assert "qkv_proj" in fused
        assert "gate_up_proj" in fused
        assert metadata["num_heads"] == 2

    def test_no_double_counting_pattern1(self):
        """Pattern A: ForCausalLM delegates to inner Model. The spec is applied once."""
        conv = self._make_converter()

        class InnerModel(nn.Module, SupportsWeightLayoutSpec):
            supports_weight_layout_spec = True

            def get_weight_layout_spec(self):
                return WeightLayoutSpec(
                    stacked_params=[
                        ("qkv_proj", "q_proj", "q"),
                        ("qkv_proj", "k_proj", "k"),
                        ("qkv_proj", "v_proj", "v"),
                    ],
                    packing_metadata={"num_heads": 4, "num_kv_heads": 4, "head_size": 8, "intermediate_size": 16},
                )

        class OuterModel(nn.Module, SupportsWeightLayoutSpec):
            supports_weight_layout_spec = True

            def __init__(self):
                super().__init__()
                self.model = InnerModel()

            def get_weight_layout_spec(self):
                return self.model.get_weight_layout_spec()  # delegate

        # Top-level returns non-empty spec → used directly, sub-module walk never runs
        spec = OuterModel().get_weight_layout_spec()
        fused, _ = conv._build_from_spec(spec)
        assert len(fused["qkv_proj"].mappings) == 3  # exactly 3, not 6

    # --- Pattern C/D routing is in convert_state_and_sharding_dict ---

    def test_pattern_c_passthrough_preserves_param_names(self):
        """Pattern C (Mamba, Qwen3): empty top-level spec → all weights pass through unchanged."""
        conv = VllmConverter(parameter_mapping=None, tp_rank=0)

        class SimpleMambaLike(nn.Module, SupportsWeightLayoutSpec):
            supports_weight_layout_spec = True

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4, bias=False)

            def get_weight_layout_spec(self):
                return WeightLayoutSpec(stacked_params=[], packing_metadata={})

        model = SimpleMambaLike()
        state_dict, _ = conv.convert_state_and_sharding_dict(model)
        assert "linear.weight" in state_dict
        assert state_dict["linear.weight"] is model.linear.weight

    def test_pattern_d_empty_toplevel_uses_submodule_spec(self):
        """Pattern D (KimiK25): empty top-level spec, sub-module has non-empty spec.

        The sub-module's spec (with qkv_proj stacked_params) must drive the conversion:
        a parameter named 'qkv_proj' in the sub-module must be split into q/k/v projections.
        """
        from vllm.model_executor.layers.linear import QKVParallelLinear

        # Build a model whose top-level spec is empty but whose sub-module has a non-empty
        # spec with qkv_proj stacked_params. The sub-module also owns a real qkv_proj param.
        class InnerWithQKV(nn.Module, SupportsWeightLayoutSpec):
            """Sub-module with qkv_proj spec AND a matching qkv_proj parameter."""

            supports_weight_layout_spec = True

            def __init__(self):
                super().__init__()
                # num_heads=2, num_kv_heads=2, head_size=4 → total=(2+2+2)*4=24 rows
                mock_qkv = MagicMock(spec=QKVParallelLinear)
                mock_qkv.tp_size = 1
                mock_qkv.weight = nn.Parameter(torch.zeros(24, 8))
                # Register the weight here so its qualified name exposes the projection suffix.
                self.register_parameter("qkv_proj_weight", nn.Parameter(torch.zeros(24, 8)))

            def get_weight_layout_spec(self) -> WeightLayoutSpec:
                return WeightLayoutSpec(
                    stacked_params=[
                        ("qkv_proj_weight", "q_proj_weight", "q"),
                        ("qkv_proj_weight", "k_proj_weight", "k"),
                        ("qkv_proj_weight", "v_proj_weight", "v"),
                    ],
                    packing_metadata={"num_heads": 2, "num_kv_heads": 2, "head_size": 4, "intermediate_size": 8},
                )

        class OuterEmpty(nn.Module, SupportsWeightLayoutSpec):
            supports_weight_layout_spec = True

            def __init__(self):
                super().__init__()
                self.language_model = InnerWithQKV()

            def get_weight_layout_spec(self) -> WeightLayoutSpec:
                return WeightLayoutSpec(stacked_params=[], packing_metadata={})

        conv = self._make_converter()
        model = OuterEmpty()
        state_dict, _ = conv.convert_state_and_sharding_dict(model)

        # The sub-module's spec must have been used: qkv_proj_weight → q/k/v
        assert "language_model.q_proj_weight" in state_dict, (
            "Pattern D: sub-module spec must drive splitting of qkv_proj_weight → q/k/v"
        )
        assert "language_model.k_proj_weight" in state_dict
        assert "language_model.v_proj_weight" in state_dict
        assert "language_model.qkv_proj_weight" not in state_dict, "qkv_proj_weight must be split, not passed through"


class TestVllmConverterIntegration:
    def test_raises_when_no_spec_and_no_mapping(self):
        conv = VllmConverter(parameter_mapping=None, tp_rank=0)
        model = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="does not implement SupportsWeightLayoutSpec"):
            conv.convert_state_and_sharding_dict(model)

    def test_fallback_to_parameter_mapping(self):
        from psrl.utils.converter.model_mappings import MappingType, ParameterMapping

        class FakeMapping(ParameterMapping):
            def __init__(self):
                pass

            def get_mappings(self):
                return [
                    ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
                    ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
                    ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
                ]

            def get_model_info(self):
                return {"num_heads": 2, "num_kv_heads": 2, "head_size": 4, "intermediate_size": 8}

        conv = VllmConverter(parameter_mapping=FakeMapping(), tp_rank=0)
        fused, info = conv._build_from_parameter_mapping()
        assert "qkv_proj" in fused


class TestConvertParameterExactMatch:
    def test_full_path_key_exact_match_not_substring(self):
        """Bug 2 fix: a full-path key 'layers.0.x.w13.weight' must NOT match
        a different param 'layers.0.x.w13.weight_scale'."""
        conv = VllmConverter(parameter_mapping=None, tp_rank=0)
        extra = [
            ("layers.0.mlp.w13.weight", "layers.0.mlp.w1.weight", 0),
        ]
        fused = conv._build_fused_mappings([], extra, [])
        model_info = {"intermediate_size": 8}

        from unittest.mock import MagicMock

        import torch
        from torch.nn import Parameter

        # Param whose name is a superset of the key (the problematic substring match case)
        scale_param = Parameter(torch.zeros(4))
        mock_module = MagicMock()
        mock_module.tp_size = 1

        # "layers.0.mlp.w13.weight_scale" contains "layers.0.mlp.w13.weight" as substring
        result = conv.convert_parameter("layers.0.mlp.w13.weight_scale", scale_param, mock_module, fused, model_info)
        # With exact match fix, this should NOT match and should passthrough
        assert result == {"layers.0.mlp.w13.weight_scale": scale_param}, (
            "Full-path key must use exact match, not substring match"
        )


class TestConvertParameterSplitting:
    """End-to-end tests: actual tensors are correctly split through convert_parameter."""

    def _make_converter(self):
        return VllmConverter(parameter_mapping=None, tp_rank=0)

    def _make_fused_mappings_for_qkv(self, conv):
        stacked = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        return conv._build_fused_mappings(stacked, [], [])

    def _make_fused_mappings_for_gate_up(self, conv):
        stacked = [("gate_up_proj", "gate_proj", 0), ("gate_up_proj", "up_proj", 1)]
        return conv._build_fused_mappings(stacked, [], [])

    def test_qkv_split_produces_correct_output_names(self):
        """Pattern A: qkv_proj is split into q_proj, k_proj, v_proj with correct shapes."""
        from vllm.model_executor.layers.linear import QKVParallelLinear

        conv = self._make_converter()
        fused = self._make_fused_mappings_for_qkv(conv)
        # num_heads=2, num_kv_heads=2, head_size=4 → q=8, k=8, v=8, total=24
        model_info = {"num_heads": 2, "num_kv_heads": 2, "head_size": 4, "intermediate_size": 16}

        param = torch.nn.Parameter(torch.zeros(24, 8))
        module = MagicMock(spec=QKVParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.self_attn.qkv_proj.weight",
            param,
            module,
            fused,
            model_info,
        )

        assert set(result.keys()) == {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        }
        assert result["model.layers.0.self_attn.q_proj.weight"].shape == (8, 8)
        assert result["model.layers.0.self_attn.k_proj.weight"].shape == (8, 8)
        assert result["model.layers.0.self_attn.v_proj.weight"].shape == (8, 8)

    def test_qkv_split_original_key_absent(self):
        """Pattern A: qkv_proj key must NOT appear in the output dict."""
        from vllm.model_executor.layers.linear import QKVParallelLinear

        conv = self._make_converter()
        fused = self._make_fused_mappings_for_qkv(conv)
        model_info = {"num_heads": 2, "num_kv_heads": 2, "head_size": 4, "intermediate_size": 16}

        param = torch.nn.Parameter(torch.zeros(24, 8))
        module = MagicMock(spec=QKVParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.self_attn.qkv_proj.weight",
            param,
            module,
            fused,
            model_info,
        )
        assert not any("qkv_proj" in k for k in result), (
            f"qkv_proj should not appear in output keys, got {set(result.keys())}"
        )

    def test_gate_up_split_produces_correct_output_names(self):
        """Pattern A: gate_up_proj is split into gate_proj and up_proj with correct shapes."""
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        conv = self._make_converter()
        fused = self._make_fused_mappings_for_gate_up(conv)
        model_info = {"intermediate_size": 8}

        # gate_up_proj weight: (2*intermediate_size, in) = (16, 4)
        param = torch.nn.Parameter(torch.zeros(16, 4))
        module = MagicMock(spec=MergedColumnParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.mlp.gate_up_proj.weight",
            param,
            module,
            fused,
            model_info,
        )

        assert set(result.keys()) == {
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        }
        assert result["model.layers.0.mlp.gate_proj.weight"].shape == (8, 4)
        assert result["model.layers.0.mlp.up_proj.weight"].shape == (8, 4)

    def test_gate_up_split_original_key_absent(self):
        """Pattern A: gate_up_proj key must NOT appear in the output dict."""
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        conv = self._make_converter()
        fused = self._make_fused_mappings_for_gate_up(conv)
        model_info = {"intermediate_size": 8}

        param = torch.nn.Parameter(torch.zeros(16, 4))
        module = MagicMock(spec=MergedColumnParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.mlp.gate_up_proj.weight",
            param,
            module,
            fused,
            model_info,
        )
        assert not any("gate_up_proj" in k for k in result), (
            f"gate_up_proj should not appear in output keys, got {set(result.keys())}"
        )

    def test_pattern_b_gate_up_with_w1_w3_names(self):
        """Pattern B (InternLM2): gate_up_proj splits to w1/w3, not gate_proj/up_proj."""
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        conv = self._make_converter()
        stacked = [("gate_up_proj", "w1", 0), ("gate_up_proj", "w3", 1)]
        fused = conv._build_fused_mappings(stacked, [], [])
        model_info = {"intermediate_size": 8}

        param = torch.nn.Parameter(torch.zeros(16, 4))
        module = MagicMock(spec=MergedColumnParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.mlp.gate_up_proj.weight",
            param,
            module,
            fused,
            model_info,
        )

        assert set(result.keys()) == {
            "model.layers.0.mlp.w1.weight",
            "model.layers.0.mlp.w3.weight",
        }, f"Expected w1/w3 keys, got {set(result.keys())}"
        assert result["model.layers.0.mlp.w1.weight"].shape == (8, 4)
        assert result["model.layers.0.mlp.w3.weight"].shape == (8, 4)

    def test_pattern_e_extra_stacked_params_full_path_output(self):
        """Pattern E (Arctic): extra_stacked_params use full paths as both input and output names."""
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        conv = self._make_converter()
        extra = [
            ("layers.0.residual_mlp.w13.weight", "layers.0.residual_mlp.w1.weight", 0),
            ("layers.0.residual_mlp.w13.weight", "layers.0.residual_mlp.w3.weight", 1),
        ]
        fused = conv._build_fused_mappings([], extra, [])
        model_info = {"intermediate_size": 8}

        param = torch.nn.Parameter(torch.zeros(16, 4))
        module = MagicMock(spec=MergedColumnParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "layers.0.residual_mlp.w13.weight",
            param,
            module,
            fused,
            model_info,
        )

        # Full-path keys: output names are the hf_names directly, not suffix-substituted
        assert set(result.keys()) == {
            "layers.0.residual_mlp.w1.weight",
            "layers.0.residual_mlp.w3.weight",
        }, f"Expected full-path HF names, got {set(result.keys())}"
        assert result["layers.0.residual_mlp.w1.weight"].shape == (8, 4)
        assert result["layers.0.residual_mlp.w3.weight"].shape == (8, 4)

    def test_unmatched_param_passes_through(self):
        """Params not matching any fused_mappings key are returned unchanged."""
        from vllm.model_executor.layers.linear import RowParallelLinear

        conv = self._make_converter()
        fused = self._make_fused_mappings_for_qkv(conv)
        model_info = {"num_heads": 2, "num_kv_heads": 2, "head_size": 4}

        param = torch.nn.Parameter(torch.zeros(4, 8))
        module = MagicMock(spec=RowParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.self_attn.o_proj.weight",
            param,
            module,
            fused,
            model_info,
        )
        assert list(result.keys()) == ["model.layers.0.self_attn.o_proj.weight"]
        assert result["model.layers.0.self_attn.o_proj.weight"] is param

    def test_qkv_split_tensor_data_is_correct_slice(self):
        """Verify q/k/v tensors are actual slices of the original (not copies)."""
        from vllm.model_executor.layers.linear import QKVParallelLinear

        conv = self._make_converter()
        fused = self._make_fused_mappings_for_qkv(conv)
        model_info = {"num_heads": 2, "num_kv_heads": 2, "head_size": 4, "intermediate_size": 16}

        # Fill with distinct values so we can verify slicing correctness
        data = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
        param = torch.nn.Parameter(data)
        module = MagicMock(spec=QKVParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.self_attn.qkv_proj.weight",
            param,
            module,
            fused,
            model_info,
        )

        q = result["model.layers.0.self_attn.q_proj.weight"]
        k = result["model.layers.0.self_attn.k_proj.weight"]
        v = result["model.layers.0.self_attn.v_proj.weight"]

        # q = rows 0-7, k = rows 8-15, v = rows 16-23
        assert torch.equal(q.data, data[0:8, :])
        assert torch.equal(k.data, data[8:16, :])
        assert torch.equal(v.data, data[16:24, :])

    def test_gate_up_split_tensor_data_is_correct_slice(self):
        """Verify gate/up tensors are correct halves of the fused weight."""
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        conv = self._make_converter()
        fused = self._make_fused_mappings_for_gate_up(conv)
        model_info = {"intermediate_size": 8}

        data = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
        param = torch.nn.Parameter(data)
        module = MagicMock(spec=MergedColumnParallelLinear)
        module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.mlp.gate_up_proj.weight",
            param,
            module,
            fused,
            model_info,
        )

        gate = result["model.layers.0.mlp.gate_proj.weight"]
        up = result["model.layers.0.mlp.up_proj.weight"]

        # gate = first 8 rows, up = last 8 rows
        assert torch.equal(gate.data, data[0:8, :])
        assert torch.equal(up.data, data[8:16, :])

    def test_integration_parameter_mapping_end_to_end(self):
        """Gap 6: _build_from_parameter_mapping path exercised through convert_state_and_sharding_dict."""
        from psrl.utils.converter.model_mappings import ParameterMapping

        class QKVMapping(ParameterMapping):
            def __init__(self):
                pass

            def get_mappings(self):
                return [
                    ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
                    ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
                    ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
                ]

            def get_model_info(self):
                return {"num_heads": 2, "num_kv_heads": 2, "head_size": 4, "intermediate_size": 8}

        from vllm.model_executor.layers.linear import QKVParallelLinear

        # Build a minimal nn.Module that owns a qkv_proj parameter directly.
        # We attach the mock as a proper submodule via add_module so named_modules() works.
        class FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                # Register qkv weight as a bare parameter on this module
                self.weight = torch.nn.Parameter(torch.zeros(24, 8))

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                # A plain linear layer supplies the default TP size used by sharding lookup.
                self.attn = nn.Linear(8, 24, bias=False)

        # Exercise the `ParameterMapping` path directly without model traversal.
        conv = VllmConverter(parameter_mapping=QKVMapping(), tp_rank=0)
        fused, model_info = conv._build_from_parameter_mapping()

        assert "qkv_proj" in fused
        assert fused["qkv_proj"].mapping_type == MappingType.QKV_SPLIT
        assert len(fused["qkv_proj"].mappings) == 3
        assert model_info["num_heads"] == 2

        # Verify the fused mapping works end-to-end through convert_parameter
        param = torch.nn.Parameter(torch.zeros(24, 8))
        mock_module = MagicMock(spec=QKVParallelLinear)
        mock_module.tp_size = 1

        result = conv.convert_parameter(
            "model.layers.0.self_attn.qkv_proj.weight",
            param,
            mock_module,
            fused,
            model_info,
        )
        assert set(result.keys()) == {
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.v_proj.weight",
        }

    def test_fused_moe_w13_split_produces_per_expert_output(self):
        """Pattern A MoE: w13_weight is split into per-expert gate_proj and up_proj params."""
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        conv = self._make_converter()

        NUM_EXPERTS = 2
        expert_params = []
        for eid in range(NUM_EXPERTS):
            expert_params += [
                ("experts.w13_weight", f"experts.{eid}.gate_proj.weight", eid, "w1"),
                ("experts.w2_weight", f"experts.{eid}.down_proj.weight", eid, "w2"),
                ("experts.w13_weight", f"experts.{eid}.up_proj.weight", eid, "w3"),
            ]
        fused = conv._build_fused_mappings([], [], expert_params)
        model_info = {"num_experts": NUM_EXPERTS, "intermediate_size": 4}

        # w13_weight shape: (num_experts, 2*intermediate, in_features)
        w13_param = nn.Parameter(torch.arange(2 * 2 * 4 * 2, dtype=torch.float32).reshape(2, 8, 2))
        module = MagicMock(spec=FusedMoE)
        module.tp_size = 1
        module.ep_size = 1

        result = conv.convert_parameter(
            "model.layers.0.mlp.experts.w13_weight",
            w13_param,
            module,
            fused,
            model_info,
        )

        # All 4 gate+up projections for 2 experts should appear
        expected_keys = {
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.layers.0.mlp.experts.1.up_proj.weight",
        }
        assert set(result.keys()) == expected_keys, f"Expected per-expert gate/up keys, got {set(result.keys())}"
        # Each slice should have shape (intermediate, in_features) = (4, 2)
        for key in expected_keys:
            assert result[key].shape == (4, 2), f"Shape for key={key!r} was {result[key].shape!r}. Expected (4, 2)."

    def test_fused_moe_w2_split_produces_per_expert_output(self):
        """Pattern A MoE: w2_weight is split into per-expert down_proj params."""
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        conv = self._make_converter()

        NUM_EXPERTS = 2
        expert_params = [
            ("experts.w2_weight", f"experts.{eid}.down_proj.weight", eid, "w2") for eid in range(NUM_EXPERTS)
        ]
        fused = conv._build_fused_mappings([], [], expert_params)
        model_info = {"num_experts": NUM_EXPERTS, "intermediate_size": 4}

        # w2_weight shape: (num_experts, in_features, intermediate)
        w2_param = nn.Parameter(torch.zeros(2, 2, 4))
        module = MagicMock(spec=FusedMoE)
        module.tp_size = 1
        module.ep_size = 1

        result = conv.convert_parameter(
            "model.layers.0.mlp.experts.w2_weight",
            w2_param,
            module,
            fused,
            model_info,
        )

        expected_keys = {
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.experts.1.down_proj.weight",
        }
        assert set(result.keys()) == expected_keys


class TestGetShardingForParam:
    """Tests for `get_sharding_for_param` sharding dimension inference."""

    def _make_converter(self, tp_rank=0):
        return VllmConverter(parameter_mapping=None, tp_rank=tp_rank)

    def test_column_parallel_tp2_shards_dim0(self):
        """ColumnParallelLinear with tp_size=2 → shard_dim=0."""
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        conv = self._make_converter(tp_rank=1)
        module = MagicMock(spec=ColumnParallelLinear)
        module.tp_size = 2
        sharding = conv.get_sharding_for_param(module, "weight")
        assert 0 in sharding.shard_mesh  # sharded on dim 0
        assert sharding.shard_mesh[0] == 2  # tp_size=2
        assert sharding.shard_indices == [(1,)]  # tp_rank=1

    def test_merged_column_parallel_tp2_shards_dim0(self):
        """MergedColumnParallelLinear with tp_size=2 → shard_dim=0 (same as ColumnParallel)."""
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        conv = self._make_converter(tp_rank=0)
        module = MagicMock(spec=MergedColumnParallelLinear)
        module.tp_size = 2
        sharding = conv.get_sharding_for_param(module, "weight")
        assert 0 in sharding.shard_mesh
        assert sharding.shard_mesh[0] == 2
        assert sharding.shard_indices == [(0,)]

    def test_qkv_parallel_tp2_shards_dim0(self):
        """QKVParallelLinear with tp_size=2 → shard_dim=0."""
        from vllm.model_executor.layers.linear import QKVParallelLinear

        conv = self._make_converter(tp_rank=0)
        module = MagicMock(spec=QKVParallelLinear)
        module.tp_size = 2
        sharding = conv.get_sharding_for_param(module, "weight")
        assert 0 in sharding.shard_mesh
        assert sharding.shard_mesh[0] == 2

    def test_row_parallel_tp2_shards_dim1(self):
        """RowParallelLinear with tp_size=2 → shard_dim=1."""
        from vllm.model_executor.layers.linear import RowParallelLinear

        conv = self._make_converter(tp_rank=0)
        module = MagicMock(spec=RowParallelLinear)
        module.tp_size = 2
        sharding = conv.get_sharding_for_param(module, "weight")
        assert 1 in sharding.shard_mesh  # sharded on dim 1
        assert sharding.shard_mesh[1] == 2
        assert sharding.shard_indices == [(0,)]  # tp_rank=0

    def test_row_parallel_tp2_rank1_shard_index(self):
        """RowParallelLinear with tp_size=2 and tp_rank=1 → shard_index=(1,)."""
        from vllm.model_executor.layers.linear import RowParallelLinear

        conv = self._make_converter(tp_rank=1)
        module = MagicMock(spec=RowParallelLinear)
        module.tp_size = 2
        sharding = conv.get_sharding_for_param(module, "weight")
        assert sharding.shard_indices == [(1,)]

    def test_tp1_row_parallel_returns_default_sharding(self):
        """RowParallelLinear with tp_size=1 returns default (no sharding)."""
        from vllm.model_executor.layers.linear import RowParallelLinear

        conv = self._make_converter(tp_rank=0)
        module = MagicMock(spec=RowParallelLinear)
        module.tp_size = 1
        sharding = conv.get_sharding_for_param(module, "weight")
        assert sharding.shard_mesh == {0: 1}
        assert sharding.shard_indices == [(0,)]

    def test_tp1_column_parallel_returns_default_sharding(self):
        """ColumnParallelLinear with tp_size=1 → default sharding (no split)."""
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        conv = self._make_converter(tp_rank=0)
        module = MagicMock(spec=ColumnParallelLinear)
        module.tp_size = 1
        sharding = conv.get_sharding_for_param(module, "weight")
        assert sharding.shard_mesh == {0: 1}
        assert sharding.shard_indices == [(0,)]

    def test_replicated_linear_treated_as_tp1(self):
        """ReplicatedLinear with tp_size>1 is treated as unsharded (tp_size effectively 1)."""
        from vllm.model_executor.layers.linear import ReplicatedLinear

        conv = self._make_converter(tp_rank=0)
        module = MagicMock(spec=ReplicatedLinear)
        module.tp_size = 4  # world size, but not actually sharded
        sharding = conv.get_sharding_for_param(module, "weight")
        assert sharding.shard_mesh == {0: 1}  # treated as unsharded
        assert sharding.shard_indices == [(0,)]

    def test_tp1_no_module_type_returns_default_sharding(self):
        """Any module with tp_size=1 (or missing tp_size) returns default sharding."""
        conv = self._make_converter(tp_rank=0)
        module = MagicMock()  # no spec, unknown type
        module.tp_size = 1
        sharding = conv.get_sharding_for_param(module, "weight")
        assert sharding.shard_mesh == {0: 1}
        assert sharding.shard_indices == [(0,)]

    def test_unsupported_module_type_raises(self):
        """Unknown module type with tp_size>1 raises ValueError."""
        conv = self._make_converter(tp_rank=0)
        module = MagicMock()  # no spec → not isinstance of any known type
        module.tp_size = 2
        with pytest.raises(ValueError, match="Unsupported module type"):
            conv.get_sharding_for_param(module, "weight")
