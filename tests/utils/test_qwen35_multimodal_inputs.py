"""
Test multimodal processor input construction through the current processor API.
"""

from unittest.mock import Mock

import pytest
import torch
from verl.utils.tokenizer import build_multimodal_processor_inputs

pytestmark = pytest.mark.cpu_test


def test_processor_receives_images_and_independent_options():
    grid = torch.tensor([[1, 4, 4]])
    processor = Mock(return_value={"image_grid_thw": grid})
    options = {"min_pixels": 256}
    result = build_multimodal_processor_inputs(
        processor, text=["prompt"], images=["image"], mm_processor_kwargs=options
    )
    assert result["image_grid_thw"] is grid, "Processor tensors must be preserved."
    assert processor.call_args.kwargs["images"] == ["image"], "Images must reach the processor."
    assert processor.call_args.kwargs["return_tensors"] == "pt", "The processor must return torch tensors."
    assert options == {"min_pixels": 256}, "Processor options must not be mutated."
