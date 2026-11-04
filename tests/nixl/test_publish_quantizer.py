"""The plan-driven FP8 quantizer reproduces the pre-plan block FP8 path byte for byte."""

from __future__ import annotations

import pytest
import torch
from psrl.utils.nixl.block_fp8 import quantize_blockwise
from psrl.utils.quantization import resolve_quantization_plan
from psrl.utils.quantization.quantizer import VllmFp8BlockQuantizer

pytestmark = pytest.mark.cpu_test

PLAN = resolve_quantization_plan({"scheme": "fp8"})


def test_quantized_bytes_and_scales_match_the_legacy_path():
    torch.manual_seed(1)
    weight = torch.randn(256, 384, dtype=torch.bfloat16) * 3
    reference_bytes, reference_scales = quantize_blockwise(weight, (128, 128))

    got_bytes, got_scales = VllmFp8BlockQuantizer(PLAN).quantize(weight)

    assert got_bytes.dtype == torch.float8_e4m3fn
    assert torch.equal(got_bytes.view(torch.uint8), reference_bytes.view(torch.uint8))
    assert torch.equal(got_scales, reference_scales)


def test_a_strided_shard_quantizes_like_a_standalone_tensor():
    torch.manual_seed(2)
    fused = torch.randn(512, 256, dtype=torch.bfloat16)
    shard = fused[128:384]
    reference_bytes, reference_scales = quantize_blockwise(shard, (128, 128))

    got_bytes, got_scales = VllmFp8BlockQuantizer(PLAN).quantize(shard)

    assert torch.equal(got_bytes.view(torch.uint8), reference_bytes.view(torch.uint8))
    assert torch.equal(got_scales, reference_scales)


def test_quantize_into_writes_the_publish_copies():
    torch.manual_seed(3)
    weight = torch.randn(128, 256, dtype=torch.bfloat16)
    destination = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(1, 2, dtype=torch.float32)

    VllmFp8BlockQuantizer(PLAN).quantize_into(weight, destination, scale)

    reference_bytes, reference_scales = quantize_blockwise(weight, (128, 128))
    assert torch.equal(destination.view(torch.uint8), reference_bytes.view(torch.uint8))
    assert torch.equal(scale, reference_scales)


def test_can_quantize_into_rejects_a_destination_the_kernel_cannot_address(monkeypatch):
    """A view that cannot be flattened for the fused kernel must take the copy path."""
    from psrl.utils.nixl import fp8_kernels

    monkeypatch.setattr(fp8_kernels, "available", lambda tensor: True)
    quantizer = VllmFp8BlockQuantizer(PLAN)
    source = torch.empty(8, 128, 128, dtype=torch.bfloat16)
    scale = torch.empty(8, 1, 1, dtype=torch.float32)
    base = torch.empty(8, 256, 256, dtype=torch.float8_e4m3fn)
    unaddressable = base[:, :128, :128]
    assert quantizer.can_quantize_into(source, unaddressable, scale) is False
    addressable = torch.empty(8, 128, 128, dtype=torch.float8_e4m3fn)
    assert quantizer.can_quantize_into(source, addressable, scale) is True


def test_can_quantize_into_accepts_a_destination_on_the_copy_path(monkeypatch):
    """Without the fused kernel every strided destination is legal, because `copy_` accepts it."""
    from psrl.utils.nixl import fp8_kernels

    monkeypatch.setattr(fp8_kernels, "available", lambda tensor: False)
    quantizer = VllmFp8BlockQuantizer(PLAN)
    source = torch.empty(8, 128, 128, dtype=torch.bfloat16)
    scale = torch.empty(8, 1, 1, dtype=torch.float32)
    base = torch.empty(8, 256, 256, dtype=torch.float8_e4m3fn)
    assert quantizer.can_quantize_into(source, base[:, :128, :128], scale) is True


def test_a_shard_that_cuts_a_tile_is_rejected():
    quantizer = VllmFp8BlockQuantizer(PLAN)
    with pytest.raises(ValueError, match="whole"):
        quantizer.validate_weight_shard("weight", torch.empty(200, 256))


def test_scale_shape_follows_the_publish_tile():
    quantizer = VllmFp8BlockQuantizer(PLAN)
    assert quantizer.scale_shape(torch.empty(256, 384)) == (2, 3)
    assert quantizer.scale_shape_with_tile(torch.empty(256, 384), (64, 64)) == (4, 6)


def test_a_finer_rollout_grid_spreads_the_scale():
    quantizer = VllmFp8BlockQuantizer(PLAN)
    scale = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    refined = quantizer.refine_scale(scale, (2, 2))
    assert refined.shape == (4, 6)
    assert torch.equal(refined[0, 0:2], torch.tensor([0.0, 0.0]))
    assert torch.equal(refined[0, 2:4], torch.tensor([1.0, 1.0]))
