"""Block FP8 scales follow their weight's views, and the quantiser matches vLLM bit for bit."""

from __future__ import annotations

import pytest
import torch
from psrl.utils.nixl.block_fp8 import (
    FP8_DTYPE,
    quantize_blockwise,
    scale_shape,
    scale_view,
    tile_aligned,
)

pytestmark = pytest.mark.cpu_test


def _tiles_of(view_rows: slice, view_cols: slice, scale: torch.Tensor) -> torch.Tensor:
    return scale[view_rows.start // 128 : view_rows.stop // 128, view_cols.start // 128 : view_cols.stop // 128]


@pytest.mark.parametrize(
    "rows,cols",
    [(slice(0, 4096), slice(0, 4096)), (slice(4096, 5120), slice(0, 4096)), (slice(0, 6144), slice(1024, 2048))],
)
def test_a_2d_slice_maps_to_its_tiles(rows, cols):
    weight = torch.empty(6144, 4096, dtype=torch.uint8)
    scale = torch.arange(48 * 32, dtype=torch.float32).reshape(48, 32)
    got = scale_view(weight[rows, cols], weight, scale, (128, 128))
    assert torch.equal(got, _tiles_of(rows, cols, scale))


def test_a_head_grouped_3d_view_maps_to_grouped_tiles():
    """The converter reshapes q to `[kv_groups, rows_per_group, hidden]`, so the scale must follow."""
    weight = torch.empty(6144, 4096, dtype=torch.uint8)
    scale = torch.arange(48 * 32, dtype=torch.float32).reshape(48, 32)
    q = weight[:4096].view(8, 512, 4096)
    got = scale_view(q, weight, scale, (128, 128))
    assert got.shape == (8, 4, 32)
    assert torch.equal(got.reshape(32, 32), scale[:32])


def test_a_view_that_cuts_a_tile_has_no_scale_of_its_own():
    weight = torch.empty(512, 512, dtype=torch.uint8)
    scale = torch.zeros(4, 4)
    assert not tile_aligned(weight[:, 64:192], weight, scale, (128, 128))
    with pytest.raises(ValueError):
        scale_view(weight[:, 64:192], weight, scale, (128, 128))


def test_a_view_of_whole_tiles_is_tile_aligned():
    weight = torch.empty(512, 512, dtype=torch.uint8)
    scale = torch.zeros(4, 4)
    assert tile_aligned(weight[:128], weight, scale, (128, 128))
    assert tile_aligned(weight, weight, scale, (128, 128))


def test_a_partial_last_tile_does_not_let_a_cut_view_own_its_scale():
    """The tile comes from the plan, because `weight // scale` is wrong once a tile is partial."""
    weight = torch.empty(48, 128, dtype=torch.uint8)
    scale = torch.zeros(2, 4)
    assert tile_aligned(weight[:32], weight, scale, (32, 32))
    assert not tile_aligned(weight[:24], weight, scale, (32, 32))
    assert not tile_aligned(weight[24:48], weight, scale, (32, 32))


def test_a_partial_tile_counts_whole_in_the_scale_grid():
    assert scale_shape((64, 2560), (128, 128)) == (1, 20)
    assert scale_shape((300, 2560), (128, 128)) == (3, 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="vLLM's quantiser is compiled for CUDA")
@pytest.mark.parametrize("shape", [(1024, 768), (64, 768), (300, 640)])
def test_quantisation_matches_vllm_bit_for_bit(shape):
    from vllm.utils.deep_gemm import per_block_cast_to_fp8

    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda") * 3
    ref_q, ref_s = per_block_cast_to_fp8(x, [128, 128])
    q, s = quantize_blockwise(x, (128, 128))
    assert q.dtype == FP8_DTYPE
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(s, ref_s)
