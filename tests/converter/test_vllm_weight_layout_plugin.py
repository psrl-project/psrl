import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch.nn as nn
from vllm_patches.interfaces import SupportsWeightLayout, supports_weight_layout
from vllm_patches.weight_layout import WeightLayoutPlan
from vllm_patches.weight_layouts import _LAYOUTS, register_weight_layouts_for_module

pytestmark = pytest.mark.cpu_test

EXPECTED_LAYOUT_TARGETS = {
    "deepseek_v2.DeepseekV2Model",
    "deepseek_v2.DeepseekV2ForCausalLM",
    "gemma.GemmaForCausalLM",
    "gemma2.Gemma2ForCausalLM",
    "gemma3.Gemma3ForCausalLM",
    "gemma3_mm.Gemma3ForConditionalGeneration",
    "kimi_k25.KimiK25ForConditionalGeneration",
    "kimi_vl.KimiVLForConditionalGeneration",
    "llama.LlamaForCausalLM",
    "llama4.Llama4Model",
    "olmoe.OlmoeForCausalLM",
    "phi.PhiForCausalLM",
    "qwen2.Qwen2Model",
    "qwen2.Qwen2ForCausalLM",
    "qwen2_5_vl.Qwen2_5_VLForConditionalGeneration",
    "qwen2_moe.Qwen2MoeModel",
    "qwen2_moe.Qwen2MoeForCausalLM",
    "qwen2_vl.Qwen2VLForConditionalGeneration",
    "qwen3.Qwen3ForCausalLM",
    "qwen3_5.Qwen3_5ForCausalLMBase",
    "qwen3_5.Qwen3_5ForConditionalGeneration",
    "qwen3_moe.Qwen3MoeModel",
    "qwen3_moe.Qwen3MoeForCausalLM",
    "qwen3_vl.Qwen3VLForConditionalGeneration",
    "qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration",
}


def test_all_migrated_weight_layouts_are_registered():
    assert set(_LAYOUTS) == EXPECTED_LAYOUT_TARGETS


def test_register_weight_layout_for_loaded_model_module(monkeypatch):
    module_name = "vllm.model_executor.models.qwen2"
    module = ModuleType(module_name)

    class Qwen2Model(nn.Module):
        __module__ = module_name

    class Qwen2ForCausalLM(nn.Module):
        __module__ = module_name

    module.Qwen2Model = Qwen2Model
    module.Qwen2ForCausalLM = Qwen2ForCausalLM
    monkeypatch.setitem(sys.modules, module_name, module)

    register_weight_layouts_for_module(module_name)

    model = Qwen2ForCausalLM()
    assert isinstance(model, SupportsWeightLayout)
    assert supports_weight_layout(model)
    assert isinstance(model.build_weight_layout(), WeightLayoutPlan)


def test_mount_module_registers_nested_model_layout(monkeypatch):
    language_module_name = "vllm.model_executor.models.qwen3"
    language_module = ModuleType(language_module_name)

    class Qwen3ForCausalLM(nn.Module):
        __module__ = language_module_name

    language_module.Qwen3ForCausalLM = Qwen3ForCausalLM
    monkeypatch.setitem(sys.modules, language_module_name, language_module)

    vl_module_name = "vllm.model_executor.models.qwen3_vl"
    vl_module = ModuleType(vl_module_name)

    class Qwen3VLForConditionalGeneration(nn.Module):
        __module__ = vl_module_name

        def __init__(self):
            super().__init__()
            self.language_model = Qwen3ForCausalLM()

    vl_module.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    monkeypatch.setitem(sys.modules, vl_module_name, vl_module)

    model = Qwen3VLForConditionalGeneration()
    assert supports_weight_layout(model)

    plan = model.build_weight_layout()
    assert len(plan.mounts) == 1
    assert supports_weight_layout(model.language_model)


def test_deepseek_layout_does_not_require_patched_instance_fields(monkeypatch):
    module_name = "vllm.model_executor.models.deepseek_v2"
    module = ModuleType(module_name)

    class DeepseekV2Model(nn.Module):
        __module__ = module_name

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                model_type="deepseek_v2",
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                q_lora_rank=32,
                kv_lora_rank=16,
                n_routed_experts=8,
                topk_method="noaux_tc",
            )

    class DeepseekV2ForCausalLM(nn.Module):
        __module__ = module_name

    module.DeepseekV2Model = DeepseekV2Model
    module.DeepseekV2ForCausalLM = DeepseekV2ForCausalLM
    monkeypatch.setitem(sys.modules, module_name, module)

    model = DeepseekV2Model()
    assert not hasattr(model, "use_mha")
    assert not hasattr(model, "fuse_qkv_a_proj")
    assert supports_weight_layout(model)

    plan = model.build_weight_layout()
    assert any(rule.vllm_pattern == "fused_qkv_a_proj.weight" for rule in plan.rules)
    assert any(rule.vllm_pattern == "experts.e_score_correction_bias" for rule in plan.rules)
