from dataclasses import dataclass, field
from typing import Protocol

import torch

from psrl.utils.converter.model_mappings import reshape_visual_block_qkv, slice_qkv_proj_megatron
from psrl.utils.nixl.nixl_spec import NIXLSharding


class ParamSyncAction(Protocol):
    """Lifecycle hook for tensors whose canonical PSRL view is not a plain model-param alias."""

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None: ...

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None: ...

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None: ...


@dataclass
class ParamSyncPlan:
    """Executable plan for keeping PSRL canonical tensors and train parameters synchronized."""

    actions: list[ParamSyncAction] = field(default_factory=list)

    def add(self, action: ParamSyncAction) -> None:
        self.actions.append(action)

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for action in self.actions:
                action.before_push(state_dict)

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for action in reversed(self.actions):
                action.after_push(state_dict)

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for action in self.actions:
                action.after_pull(state_dict)

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for action in self.actions:
            name = type(action).__name__
            counts[name] = counts.get(name, 0) + 1
        return counts


@dataclass
class ConversionResult:
    state_dict: dict[str, torch.Tensor]
    sharding_dict: dict[str, NIXLSharding]
    sync_plan: ParamSyncPlan = field(default_factory=ParamSyncPlan)


@dataclass
class ZeroCenteredGammaSync:
    """Expose Megatron zero-centered gamma as standard gamma during PS push."""

    key: str

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].add_(1)

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].sub_(1)

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].sub_(1)


@dataclass
class DTypeCastSync:
    """Keep an externally exposed dtype copy synchronized with the real train parameter."""

    key: str
    source_param: torch.Tensor

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            state_dict[self.key].copy_(self.source_param.to(dtype=state_dict[self.key].dtype))

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            self.source_param.copy_(state_dict[self.key].to(dtype=self.source_param.dtype))


@dataclass
class ConcatenatedQKVSync:
    """Synchronize a packed HF QKV tensor with a Megatron interleaved QKV parameter."""

    key: str
    megatron_name: str
    source_param: torch.Tensor
    num_heads: int
    num_kv_heads: int
    head_size: int
    tp_size: int
    vision_head_size: int | None = None

    @property
    def is_visual_qkv(self) -> bool:
        return "visual.blocks" in self.key and "qkv" in self.key

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key not in state_dict:
            return
        q, k, v = slice_qkv_proj_megatron(
            fused_param=self.source_param,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            attn_output_gate=False,
            tp_size=self.tp_size,
        )
        qkv = torch.cat((q, k, v), dim=0)
        if self.is_visual_qkv:
            qkv = reshape_visual_block_qkv(qkv, vision_head_size=self.vision_head_size)
        state_dict[self.key].copy_(qkv.to(dtype=state_dict[self.key].dtype))

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key not in state_dict:
            return
        hf_qkv = state_dict[self.key]
        if hf_qkv.dim() == 4:
            hf_qkv = hf_qkv.reshape(-1, hf_qkv.shape[-1])
        elif hf_qkv.dim() == 3:
            hf_qkv = hf_qkv.reshape(-1, hf_qkv.shape[-1])

        total_rows = hf_qkv.shape[0]
        denom = self.num_heads + 2 * self.num_kv_heads
        if denom == 0 or total_rows % denom != 0:
            return

        unit = total_rows // denom
        q_size = unit * self.num_heads
        k_size = unit * self.num_kv_heads
        v_size = unit * self.num_kv_heads
        q_flat, k_flat, v_flat = hf_qkv.split([q_size, k_size, v_size], dim=0)

        heads_per_group = self.num_heads // self.num_kv_heads
        num_kv_groups = k_size // self.head_size
        q_grouped = q_flat.reshape(num_kv_groups, heads_per_group * self.head_size, -1)
        k_grouped = k_flat.reshape(num_kv_groups, self.head_size, -1)
        v_grouped = v_flat.reshape(num_kv_groups, self.head_size, -1)
        megatron_qkv = torch.cat([q_grouped, k_grouped, v_grouped], dim=1)
        self.source_param.copy_(megatron_qkv.reshape(self.source_param.shape).to(dtype=self.source_param.dtype))
