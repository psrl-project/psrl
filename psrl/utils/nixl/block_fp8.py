"""
Block scaled FP8 weights on the wire: scale geometry, scale discovery and quantisation.

A block scaled weight carries one scale per tile, and vLLM requires every rank's tiles to
be whole. The scale grid therefore shards exactly like its weight, so a scale travels as
an ordinary key whose sharding is the weight's. The rollout declares which keys are
scales of which weights, read from its own parameters. A parameter is quantized as one
tensor: its checkpoint views are views of that tensor, and a view that cuts a tile is
quantized with the rest of the parameter rather than on its own.
"""

from __future__ import annotations

import torch

from psrl.utils.nixl.moe_grouping import GroupedShards

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
SCALE_SUFFIX = "_scale_inv"


def scale_key_of(weight_key: str) -> str:
    return weight_key + SCALE_SUFFIX


def _scale_view_geometry(
    weight_view: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int],
) -> tuple[list[int], list[int], int]:
    """
    The shape, strides and offset of `scale` holding the tiles of `weight_view`.

    Each view dim walks one dim of `weight`. A walk in steps of whole tiles keeps its extent
    and steps through scale tiles. A unit step walk spans whole tiles and shrinks by the tile.
    """
    if not (weight.is_contiguous() and scale.is_contiguous()):
        raise ValueError("Fused weight and scale must be dense.")
    expected = scale_shape(weight.shape, block)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"Scale {tuple(scale.shape)} is not the {expected} grid of weight {tuple(weight.shape)} at tile {block}."
        )
    item = weight.element_size()
    offset = (weight_view.data_ptr() - weight.data_ptr()) // item
    origin = []
    for stride in weight.stride():
        origin.append(offset // stride)
        offset %= stride
    if any(o % b for o, b in zip(origin, block)):
        raise ValueError(f"View origin {origin} is not tile aligned to {block}.")
    scale_offset = sum((o // b) * s for o, b, s in zip(origin, block, scale.stride()))

    shape, strides = [], []
    for extent, stride in zip(weight_view.shape, weight_view.stride()):
        if extent == 1:
            shape.append(1)
            strides.append(0)
            continue
        dim = next(d for d, s in enumerate(weight.stride()) if stride % s == 0 and stride // s < weight.shape[d])
        steps = stride // weight.stride(dim)
        if steps % block[dim] == 0:
            shape.append(extent)
            strides.append(steps // block[dim] * scale.stride(dim))
        elif steps == 1 and extent % block[dim] == 0:
            shape.append(extent // block[dim])
            strides.append(scale.stride(dim))
        else:
            raise ValueError(
                f"A walk of {extent} in steps of {steps} along dim {dim} does not cover whole tiles of {block[dim]}."
            )
    return shape, strides, scale_offset


def tile_aligned(
    weight_view: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int],
) -> bool:
    """Whether every scale tile of `weight` covers whole elements of `weight_view`."""
    try:
        _scale_view_geometry(weight_view, weight, scale, block)
    except ValueError:
        return False
    return True


def scale_view(
    weight_view: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int],
) -> torch.Tensor:
    """The view of `scale` holding the tiles of `weight_view`, a strided view of `weight`."""
    shape, strides, offset = _scale_view_geometry(weight_view, weight, scale, block)
    return torch.as_strided(scale, shape, strides, offset)


def _first(entry) -> torch.Tensor:
    return entry.shards[0] if isinstance(entry, GroupedShards) else entry


def block_scales_of(
    state_dict: dict,
    *,
    scale_suffix: str = SCALE_SUFFIX,
    weight_dtype: torch.dtype = FP8_DTYPE,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """
    `{scale_key: (weight_key, tile)}` for every quantized weight in `state_dict` that has a scale.

    A grouped expert key has a leading expert dim, whose tile is 1.
    """
    found = {}
    for key, entry in state_dict.items():
        if not key.endswith(scale_suffix):
            continue
        base = key[: -len(scale_suffix)]
        if base not in state_dict or _first(state_dict[base]).dtype != weight_dtype:
            continue
        weight, scale = _first(state_dict[base]), _first(entry)
        tile = tuple(w // s for w, s in zip(weight.shape, scale.shape))
        if isinstance(entry, GroupedShards):
            tile = (1, *tile)
        found[key] = (base, tile)
    return found


def scale_shape(shape, tile) -> tuple[int, ...]:
    """The scale grid of `shape`: one entry per tile, a partial tile counting as one."""
    return tuple(-(-n // t) for n, t in zip(shape, tile))


def quantize_blockwise(x: torch.Tensor, tile: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantise `x` to FP8 with one float32 scale per tile, bit for bit as vLLM's `per_block_cast_to_fp8`.

    A dimension that is not a whole number of tiles is padded with zeros, which leaves each
    tile's amax unchanged, and the result is cropped back to `x`'s shape. A fused parameter
    smaller than one tile, such as a 64 row projection, is therefore quantized by exactly the
    scale its kernel reads.
    """
    blocks = scale_shape(x.shape, tile)
    shape = tuple(x.shape)
    aligned = tuple(b * t for b, t in zip(blocks, tile))
    if aligned != shape:
        padded = x.new_zeros(aligned)
        padded[tuple(slice(0, n) for n in shape)] = x
        x = padded
    split = [d for pair in zip(blocks, tile) for d in pair]
    tiles = x.reshape(split)
    amax = tiles.abs().float().amax(dim=tuple(range(1, 2 * len(tile), 2)), keepdim=True).clamp(1e-4)
    scale = amax / FP8_MAX
    quantized = (tiles * (1.0 / scale)).to(FP8_DTYPE)
    extent = tuple(slice(0, n) for n in shape)
    return quantized.reshape(x.shape)[extent], scale.reshape(blocks)
