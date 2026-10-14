"""
MetaBuffer: a single 1D buffer (per dtype) for meta tensor allocation.

Allocates one contiguous block per dtype, registers it once, then hands out
views by (offset, length) + reshape. Only dtype is needed in the mapping.
offset and shape are stored per key.
"""

from collections import defaultdict
from collections.abc import Hashable

import torch


class MetaBuffer:
    """
    One long 1D buffer per dtype. The caller registers each buffer
    externally. Tensors are views into slices (offset, numel) reshaped to the requested shape.
    """

    def __init__(self, device: torch.device):
        self.device = device
        # dtype -> 1D tensor (the big buffer)
        self._buffers: dict[torch.dtype, torch.Tensor] = {}
        # key -> (dtype, offset_in_elements, shape)
        self._slices: dict[Hashable, tuple[torch.dtype, int, tuple[int, ...]]] = {}

    def allocate(
        self,
        entries: list[tuple[Hashable, tuple[int, ...], torch.dtype]],
    ) -> None:
        """
        Perform actual allocation: one 1D buffer per dtype.
        entries: list of (key, shape, dtype). Caller should register each buffer from buffers() after.
        """
        # dtype -> list of (key, shape, numel)
        by_dtype: dict[torch.dtype, list[tuple[Hashable, tuple[int, ...], int]]] = defaultdict(list)
        for key, shape, dtype in entries:
            numel = 1
            for s in shape:
                numel *= s
            by_dtype[dtype].append((key, shape, numel))

        for dtype, key_shape_numel_list in by_dtype.items():
            total_numel = sum(n for _, _, n in key_shape_numel_list)
            buf = torch.empty(
                total_numel,
                dtype=dtype,
                device=self.device,
                memory_format=torch.contiguous_format,
            )
            self._buffers[dtype] = buf
            offset = 0
            for key, shape, numel in key_shape_numel_list:
                self._slices[key] = (dtype, offset, shape)
                offset += numel

    def buffers(self):
        """Yield each 1D buffer (one per dtype). Caller should register them externally."""
        yield from self._buffers.values()

    def get_tensor(self, key: Hashable) -> torch.Tensor:
        """Return a view of the slice for this key (offset + length, then reshape)."""
        dtype, offset, shape = self._slices[key]
        buf = self._buffers[dtype]
        numel = 1
        for s in shape:
            numel *= s
        return buf[offset : offset + numel].view(shape)

    def __contains__(self, key: Hashable) -> bool:
        return key in self._slices
