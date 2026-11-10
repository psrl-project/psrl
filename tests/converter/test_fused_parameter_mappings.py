"""
Test fused parameter mappings used by explicit parameter mapping converters.
"""

import pytest
from psrl.utils.converter.model_mappings import MappingType
from psrl.utils.converter.vllm_converter import VllmConverter

pytestmark = pytest.mark.cpu_test


def test_qkv_shards_use_integer_indices():
    mappings = VllmConverter(None)._build_fused_mappings(
        [("qkv_proj", "q_proj", "q"), ("qkv_proj", "k_proj", "k"), ("qkv_proj", "v_proj", "v")],
        [],
        [],
    )
    entry = mappings["qkv_proj"]
    assert entry.mapping_type == MappingType.QKV_SPLIT, "QKV mappings must split the fused projection."
    assert entry.mappings == [("q_proj", 0), ("k_proj", 1), ("v_proj", 2)], "QKV shard order must be stable."


def test_expert_gate_up_and_down_shards():
    experts = []
    for expert in range(3):
        experts.extend(
            [
                ("experts.w13_", f"experts.{expert}.gate_proj.", expert, "w1"),
                ("experts.w13_", f"experts.{expert}.up_proj.", expert, "w3"),
                ("experts.w2_", f"experts.{expert}.down_proj.", expert, "w2"),
            ]
        )
    mappings = VllmConverter(None)._build_fused_mappings([], [], experts)
    assert [item[1] for item in mappings["experts.w13_"].mappings] == list(range(6)), "Gate and up must interleave."
    assert [item[1] for item in mappings["experts.w2_"].mappings] == list(range(3)), "Down must use expert indices."
