"""Publisher-side weight quantizers.

A quantizer answers three questions about one weight key. Which auxiliary parameters
does the rollout declare, what geometry do they have, and how are the weight bytes and
auxiliary bytes produced for a local shard. The transport only moves the resulting
tensors, so a new scheme adds a quantizer instead of a new transfer path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from psrl.utils.quantization.plan import QuantizationPlan


class PublishQuantizer(ABC):
    """The publisher half of one quantization scheme."""

    def __init__(self, plan: QuantizationPlan):
        self.plan = plan

    @property
    @abstractmethod
    def weight_dtype(self) -> torch.dtype:
        """Wire dtype of a quantized weight."""

    @property
    @abstractmethod
    def scale_dtype(self) -> torch.dtype:
        """Wire dtype of the auxiliary scale."""

    @abstractmethod
    def tile_of(self, shard: torch.Tensor) -> tuple[int, ...]:
        """Quantization tile over every dimension of `shard`."""

    @abstractmethod
    def scale_shape(self, shard: torch.Tensor) -> tuple[int, ...]:
        """Shape of the scale for `shard`."""

    @abstractmethod
    def validate_weight_shard(self, key: str, shard: torch.Tensor) -> None:
        """Reject a local shard that cuts a quantization tile."""

    @abstractmethod
    def quantize_into(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        """Write the quantized bytes and scale for `source` into preallocated tensors."""

    def can_quantize_into(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        scale: torch.Tensor,
    ) -> bool:
        """Whether `quantize_into` can write these buffers without a temporary."""
        return True

    def scale_key_of(self, weight_key: str) -> str:
        """The auxiliary key that belongs to `weight_key`."""
        scale_key = self.plan.scale_key_of(weight_key)
        assert scale_key is not None, f"Scheme {self.plan.scheme!r} has no scale key."
        return scale_key

    def base_of_scale_key(self, key: str) -> str | None:
        """The weight key for an auxiliary key."""
        return self.plan.base_of_scale_key(key)


class VllmFp8BlockQuantizer(PublishQuantizer):
    """Block FP8 in vLLM's serialized `Fp8Config` layout.

    The scale is checkpoint-format `_scale_inv`, one fp32 value per 128x128 tile. The
    publisher writes it into the live rollout parameter, which the converter has already
    mapped onto the canonical weight view.
    """

    @property
    def weight_dtype(self) -> torch.dtype:
        from psrl.utils.nixl.block_fp8 import FP8_DTYPE

        return FP8_DTYPE

    @property
    def scale_dtype(self) -> torch.dtype:
        return torch.float32

    def tile_of(self, shard: torch.Tensor) -> tuple[int, ...]:
        block = self.plan.block
        assert block is not None, "A quantized plan must declare a block."
        return (1,) * (shard.dim() - 2) + tuple(block)

    def scale_shape(self, shard: torch.Tensor) -> tuple[int, ...]:
        from psrl.utils.nixl.block_fp8 import scale_shape

        return scale_shape(tuple(shard.shape), self.tile_of(shard))

    def scale_shape_with_tile(
        self,
        shard: torch.Tensor,
        scale_tile: tuple[int, int],
    ) -> tuple[int, ...]:
        """Scale shape when the rollout declares a finer grid than the publish tile."""
        from psrl.utils.nixl.block_fp8 import scale_shape

        tile = (1,) * (shard.dim() - 2) + tuple(scale_tile)
        return scale_shape(tuple(shard.shape), tile)

    def validate_weight_shard(self, key: str, shard: torch.Tensor) -> None:
        block = self.plan.block
        assert block is not None, "A quantized plan must declare a block."
        if any(size % tile != 0 for size, tile in zip(shard.shape[-2:], block)):
            raise ValueError(
                f"{key}: the trainer shard {tuple(shard.shape)} does not hold whole {tuple(block)} "
                "quantization tiles. Adjust the training parallelism or disable FP8."
            )

    def can_quantize_in_place(self, shard: torch.Tensor) -> bool:
        """Whether the fused kernel can write straight into a publish copy."""
        block = self.plan.block
        assert block is not None, "A quantized plan must declare a block."
        if any(size % tile != 0 for size, tile in zip(shard.shape[-2:], block)):
            # A partial tile is padded before it is quantized, which the fused kernel does not do.
            return False
        from psrl.utils.nixl import fp8_kernels

        return fp8_kernels.available(shard)

    def can_quantize_into(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        scale: torch.Tensor,
    ) -> bool:
        """Whether the fused kernel can address these destination buffers.

        `quantize_into` flattens the weight and the scale before it launches. A target
        that cannot be flattened without a copy must take the `copy_` path instead.
        """
        if not self.can_quantize_in_place(source):
            return True
        block = self.plan.block
        assert block is not None, "A quantized plan must declare a block."
        rows = source.numel() // source.shape[-1]
        cols = source.shape[-1]
        try:
            destination.view(rows, cols)
            scale.view(rows // block[0], cols // block[1])
        except RuntimeError:
            return False
        return True

    def quantize_into(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        scale: torch.Tensor,
    ) -> None:
        from psrl.utils.nixl import fp8_kernels
        from psrl.utils.nixl.block_fp8 import quantize_blockwise

        block = self.plan.block
        assert block is not None, "A quantized plan must declare a block."
        if self.can_quantize_in_place(source):
            fp8_kernels.quantize_into(source, destination, scale, tuple(block))
            return
        quantized, derived = quantize_blockwise(source, self.tile_of(source))
        destination.copy_(quantized)
        scale.copy_(derived)

    def quantize(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize `source` into fresh tensors, for callers without publish copies."""
        destination = torch.empty(source.shape, dtype=self.weight_dtype, device=source.device)
        scale = torch.empty(self.scale_shape(source), dtype=self.scale_dtype, device=source.device)
        self.quantize_into(source, destination, scale)
        return destination, scale

    def refine_scale(self, scale: torch.Tensor, repeat: tuple[int, int]) -> torch.Tensor:
        """Spread a coarse scale over the finer rollout grid it covers."""
        if repeat == (1, 1):
            return scale
        return scale.repeat_interleave(repeat[0], dim=-2).repeat_interleave(repeat[1], dim=-1)


def quantizer_for_plan(plan: QuantizationPlan) -> PublishQuantizer | None:
    """Build the publisher quantizer for a resolved plan."""
    if not plan.is_quantized:
        return None
    if plan.scheme == "fp8":
        return VllmFp8BlockQuantizer(plan)
    raise ValueError(f"Scheme {plan.scheme!r} is resolved but has no publisher quantizer.")
