"""
Test the weight layout protocol and composed plans.
"""

import pytest
import torch.nn as nn
from vllm_patches.interfaces import SupportsWeightLayout, supports_weight_layout
from vllm_patches.weight_layout import WeightLayoutBuilder, WeightLayoutPlan

pytestmark = pytest.mark.cpu_test


def test_empty_plan():
    plan = WeightLayoutPlan()
    assert plan.flatten().rules == (), "An empty plan must have no rules."


def test_protocol_requires_layout_method():
    class Model(nn.Module):
        def build_weight_layout(self):
            return WeightLayoutPlan()

    model = Model()
    assert isinstance(model, SupportsWeightLayout), "The model must expose the layout protocol."
    assert supports_weight_layout(model), "The model must support weight layouts."
    assert not supports_weight_layout(nn.Linear(2, 2)), "A plain linear layer must not expose a layout."


def test_mounted_plan_preserves_prefix():
    child = nn.Linear(2, 2, bias=False)
    child.build_weight_layout = lambda: WeightLayoutBuilder(child).identity("weight", "projection.weight").build()
    parent = nn.Module()
    parent.child = child
    plan = WeightLayoutBuilder(parent).mount_module("child", child).build().flatten()
    assert len(plan.rules) == 1, "The mounted plan must contain the child rule."
    assert plan.rules[0].prefix == "child", "The child module prefix must survive flattening."
