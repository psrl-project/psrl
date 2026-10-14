"""
Test model layout registration and attention layout selection.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch.nn as nn
from vllm_patches.interfaces import supports_weight_layout
from vllm_patches.weight_layout import WeightLayoutPlan
from vllm_patches.weight_layouts import _LAYOUTS, _deepseek_layout, register_weight_layouts_for_module

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("target", sorted(_LAYOUTS))
def test_registered_model_builds_layout(target, monkeypatch):
    module_suffix, class_name = target.rsplit(".", 1)
    module_name = f"vllm.model_executor.models.{module_suffix}"
    module = ModuleType(module_name)
    model_class = type(class_name, (nn.Module,), {"__module__": module_name})
    setattr(module, class_name, model_class)
    monkeypatch.setitem(sys.modules, module_name, module)
    register_weight_layouts_for_module(module_name)
    model = model_class()
    model.config = SimpleNamespace(num_experts=2, n_routed_experts=2, model_type="deepseek")
    model.num_experts = 2

    class Child(nn.Module):
        def build_weight_layout(self):
            return WeightLayoutPlan()

    model.model = Child()
    model.language_model = Child()
    model.visual = Child()
    assert supports_weight_layout(model), "The registered class must support layouts."
    assert isinstance(model.build_weight_layout(), WeightLayoutPlan), "The model must build a layout plan."


@pytest.mark.parametrize("use_mha", [True, False])
def test_deepseek_attention_layout(use_mha):
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="deepseek" if use_mha else "deepseek_v2",
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=32,
        kv_lora_rank=16,
        n_routed_experts=2,
    )
    patterns = {rule.vllm_pattern for rule in _deepseek_layout(model).rules}
    assert ("qkv_proj.weight" in patterns) == use_mha, "MHA must use the QKV projection."
    assert ("fused_qkv_a_proj.weight" in patterns) != use_mha, "MLA must use the low rank projection."
