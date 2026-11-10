"""
Test conversion through explicit weight layout plans.
"""

import pytest
import torch
import torch.nn as nn
from psrl.utils.converter.vllm_converter import VllmConverter
from vllm_patches.weight_layout import WeightLayoutBuilder

pytestmark = pytest.mark.cpu_test


class LayoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)

    def build_weight_layout(self):
        return WeightLayoutBuilder(self).identity("projection.weight", "output.weight").build()


def test_layout_conversion_preserves_values_and_storage():
    model = LayoutModel()
    state, shardings = VllmConverter(None).convert_state_and_sharding_dict(model)
    assert set(state) == {"output.weight"}, "The layout must rename the parameter."
    assert torch.equal(state["output.weight"], model.projection.weight), "The values must be preserved."
    assert state["output.weight"].data_ptr() == model.projection.weight.data_ptr(), "Identity must share storage."
    assert set(shardings) == set(state), "Every exported parameter must have sharding metadata."


def test_layout_errors_propagate():
    class BrokenModel(LayoutModel):
        def build_weight_layout(self):
            raise RuntimeError("Invalid layout.")

    with pytest.raises(RuntimeError, match="Invalid layout"):
        VllmConverter(None).convert_state_and_sharding_dict(BrokenModel())


def test_model_without_layout_or_mapping_is_rejected():
    with pytest.raises(ValueError, match="does not implement SupportsWeightLayout"):
        VllmConverter(None).convert_state_and_sharding_dict(nn.Linear(4, 4))


def test_mounted_layout_exports_nested_parameter():
    class Parent(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LayoutModel()

        def build_weight_layout(self):
            return WeightLayoutBuilder(self).mount_module("language_model", self.language_model).build()

    state, _ = VllmConverter(None).convert_state_and_sharding_dict(Parent())
    assert set(state) == {"language_model.output.weight"}, "The mounted layout must retain its prefix."
