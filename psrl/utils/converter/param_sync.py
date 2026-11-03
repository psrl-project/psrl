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
class ResidentZeroCenteredGammaSync:
    """Keep a *separate* exposed gamma correct at all times, for direct supply.

    `ZeroCenteredGammaSync` works by borrowing the train parameter: `add_(1)` before a
    push and `sub_(1)` after it, so the tensor is only briefly in exposed form. Direct
    supply has no "after the push" -- a rollout reads the registered tensor whenever it
    pulls -- so the pair cannot be used. Reverting would corrupt whatever read is in
    flight, and *not* reverting would leave the trainer's own forward reading gamma+1.

    So the exposed value gets its own tensor. `source_param` stays zero-centred for the
    trainer. `state_dict[key]` holds `source_param + 1` and is refreshed at each version
    publish. The copy costs one gamma per norm layer, which is negligible beside the
    weights themselves -- and it is what lets the registered address stay valid, since
    NIXL descriptors point at this tensor for the life of the run.
    """

    key: str
    source_param: torch.Tensor

    def before_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Refresh the exposed copy. Under direct supply this is the publish point."""
        if self.key in state_dict:
            exposed = state_dict[self.key]
            exposed.copy_(self.source_param)
            exposed.add_(1)

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Deliberately nothing: the exposed copy must stay correct between pulls."""

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key in state_dict:
            self.source_param.copy_(state_dict[self.key])
            self.source_param.sub_(1)


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
        # GQA models produce mismatched shapes (e.g., q: [8, 512, 4096], k/v: [8, 128, 4096]).
        # cat(dim=0) only works for MHA. cat(dim=1) concatenates group-local QKV, then flatten.
        if self.source_param.dim() == 1:
            # A fused qkv bias has no hidden dimension. Both sides use the same
            # group-major order, so the group-local concat is the whole tensor.
            qkv = torch.cat((q, k, v), dim=1).reshape(-1)
        else:
            qkv = torch.cat((q, k, v), dim=1).reshape(-1, self.source_param.size(1))
            if self.is_visual_qkv:
                qkv = reshape_visual_block_qkv(qkv, vision_head_size=self.vision_head_size)
        state_dict[self.key].copy_(qkv.to(dtype=state_dict[self.key].dtype))

    def after_push(self, state_dict: dict[str, torch.Tensor]) -> None:
        pass

    def after_pull(self, state_dict: dict[str, torch.Tensor]) -> None:
        if self.key not in state_dict:
            return
        import math
        hf_qkv = state_dict[self.key]
        # Flatten visual QKV layouts to 2-D first.
        if hf_qkv.dim() > 2:
            hf_qkv = hf_qkv.reshape(-1, hf_qkv.shape[-1])

        if hf_qkv.dim() == 1:
            # A fused qkv bias has no hidden dimension. `before_push` wrote the same
            # group-major order, so the inverse is a copy.
            self.source_param.copy_(hf_qkv.reshape(self.source_param.shape).to(dtype=self.source_param.dtype))
            return

        # `before_push` writes via cat(dim=1).reshape: the flat tensor is organised as
        # nsh groups, each containing (q_len + k_len + v_len) rows, where
        #   nsh       = gcd(num_heads, num_kv_heads)
        #   q_len     = head_size * num_heads  // nsh
        #   k_len     = head_size * num_kv_heads // nsh
        # Invert that by reshaping to (nsh, q_len+2*k_len, hidden), splitting on dim=1,
        # concatenating back in Megatron's interleaved order, and copying into source_param.
        nsh = math.gcd(self.num_heads, self.num_kv_heads)
        q_len = self.head_size * self.num_heads // nsh
        k_len = self.head_size * self.num_kv_heads // nsh
        total_group_rows = q_len + 2 * k_len

        expected_rows = nsh * total_group_rows
        if hf_qkv.shape[0] != expected_rows:
            # Size mismatch. Cannot safely reconstruct, so skip to avoid corruption.
            return

        hidden = hf_qkv.shape[1]
        qkv_3d = hf_qkv.reshape(nsh, total_group_rows, hidden)
        q_part = qkv_3d[:, :q_len, :]
        k_part = qkv_3d[:, q_len:q_len + k_len, :]
        v_part = qkv_3d[:, q_len + k_len:, :]
        megatron_qkv = torch.cat([q_part, k_part, v_part], dim=1)
        self.source_param.copy_(megatron_qkv.reshape(self.source_param.shape).to(dtype=self.source_param.dtype))
