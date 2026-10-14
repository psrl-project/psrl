"""
Execute declarative weight layout transforms for vLLM tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from vllm_patches.weight_layout import (
    WeightPiece,
    WeightTransform,
)

from psrl.utils.nixl.nixl_spec import NIXLSharding


@dataclass
class TransformFragment:
    """Output fragment from a transform.

    Attributes:
        name: HF parameter name, or None to preserve the original name
        param: Converted tensor (may be a view of the source)
        shard_id: Optional shard identifier for expert/head selection
        sharding: Optional NIXL sharding metadata
    """

    name: str | None
    param: torch.Tensor
    shard_id: int | str | tuple[int, ...] | None = None
    sharding: NIXLSharding | None = None


class TransformExecutor:
    """Executes WeightTransform instances to convert vLLM tensors to HF format.

    Usage::

        executor = TransformExecutor(tp_rank=0, ep_rank=0)
        for frag in executor.execute(transform, param, module):
            print(f"{frag.name}: {frag.param.shape}")
    """

    def __init__(self, tp_rank: int = 0, ep_rank: int = 0):
        self.tp_rank = tp_rank
        self.ep_rank = ep_rank

    def execute(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None = None,
        full_name: str | None = None,
    ) -> Iterable[TransformFragment]:
        """Execute a transform and yield output fragments."""
        kind = transform.kind
        if kind == "identity":
            yield from self._transform_identity(transform, param, module)
        elif kind == "qkv":
            yield from self._transform_qkv(transform, param, module)
        elif kind in ("merged_column", "split"):
            yield from self._transform_merged_column(transform, param, module)
        elif kind == "qkv_interleaved":
            yield from self._transform_qkv_interleaved(transform, param, module)
        elif kind == "fused_moe":
            yield from self._transform_fused_moe(transform, param, module, full_name)
        elif kind == "expert_matrix":
            yield from self._transform_expert_matrix(transform, param, module)
        elif kind == "transpose":
            yield from self._transform_transpose(transform, param, module)
        elif kind == "reshape":
            yield from self._transform_reshape(transform, param, module)
        elif kind == "permute_qk_rotary":
            yield from self._transform_permute_qk_rotary(transform, param, module)
        elif kind == "scalar_extract":
            yield from self._transform_scalar_extract(transform, param, module)
        elif kind == "index_select":
            yield from self._transform_index_select(transform, param, module)
        elif kind == "alias":
            yield from self._transform_alias(transform, param, module)
        elif kind == "derive":
            yield from self._transform_derive(transform, param, module)
        elif kind == "custom":
            yield from self._transform_custom(transform, param, module)
        else:
            raise ValueError(f"Unknown transform kind: {kind!r}")

    # --- Identity ---

    def _transform_identity(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Yield the tensor unchanged.

        If pieces is non-empty, the first piece's hf_name is used as the output
        name. If pieces is empty, the parameter name is not remapped (the caller
        is responsible for applying any name_map).
        """
        if transform.pieces:
            piece = transform.pieces[0]
            name = piece.hf_name
        else:
            # Preserve the source name when no rename is configured.
            name = None

        yield TransformFragment(name=name, param=param)

    # --- QKV with GQA ---

    def _transform_qkv(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Split fused QKV into Q, K, V tensors.

        Reads num_heads, num_kv_heads, head_size from module attributes
        to correctly handle GQA where Q != K == V in head count.
        """
        if not transform.pieces or len(transform.pieces) < 3:
            raise ValueError(f"QKV transform requires 3 pieces (Q, K, V), got {len(transform.pieces)}")

        meta = transform.metadata
        num_heads_attr = meta.get("num_heads_attr", "num_heads")
        num_kv_heads_attr = meta.get("num_kv_heads_attr", "num_kv_heads")
        head_size_attr = meta.get("head_size_attr", "head_size")

        # Read from module or fall back to equal-split heuristic
        num_heads = self._get_module_attr(module, num_heads_attr)
        num_kv_heads = self._get_module_attr(module, num_kv_heads_attr)
        head_size = self._get_module_attr(module, head_size_attr)

        if num_heads is not None and num_kv_heads is not None and head_size is not None:
            q_size = int(num_heads) * int(head_size)
            k_size = int(num_kv_heads) * int(head_size)
            v_size = int(num_kv_heads) * int(head_size)
        else:
            # Fall back: assume equal split (MHA)
            total = param.shape[0]
            if total % 3 != 0:
                raise ValueError(
                    f"QKV parameter size {total} is not divisible by 3 "
                    f"and module attributes ({num_heads_attr}, {num_kv_heads_attr}, "
                    f"{head_size_attr}) not found on module {type(module).__name__}"
                )
            q_size = k_size = v_size = total // 3

        axis = transform.axis or 0
        total_check = q_size + k_size + v_size
        if param.shape[axis] != total_check:
            raise ValueError(
                f"QKV param shape[{axis}]={param.shape[axis]} != q({q_size})+k({k_size})+v({v_size})={total_check}"
            )

        q_param = torch.narrow(param, axis, 0, q_size)
        k_param = torch.narrow(param, axis, q_size, k_size)
        v_param = torch.narrow(param, axis, q_size + k_size, v_size)

        yield TransformFragment(name=transform.pieces[0].hf_name, param=q_param, shard_id="q")
        yield TransformFragment(name=transform.pieces[1].hf_name, param=k_param, shard_id="k")
        yield TransformFragment(name=transform.pieces[2].hf_name, param=v_param, shard_id="v")

    # --- Merged column and split ---

    def _transform_merged_column(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """
        Split a merged column weight into named pieces.

        Explicit lengths are honored, and unspecified pieces divide the
        remaining width equally.
        """
        if not transform.pieces:
            raise ValueError("merged_column/split transform requires at least one piece")

        axis = transform.axis
        total = param.shape[axis]

        # Resolve lengths
        lengths = self._resolve_lengths(transform.pieces, total)

        offset = 0
        for piece, length in zip(transform.pieces, lengths):
            sliced = torch.narrow(param, axis, offset, length)
            offset += length
            yield TransformFragment(name=piece.hf_name, param=sliced)

    @staticmethod
    def _resolve_lengths(
        pieces: Sequence[WeightPiece],
        total: int,
    ) -> list[int]:
        """Compute actual lengths for all pieces given total dimension size."""
        explicit = [p.length for p in pieces]
        n_none = sum(1 for x in explicit if x is None)

        if n_none == 0:
            # All explicit
            return [x for x in explicit]  # type: ignore[return-value]

        sum_explicit = sum(x for x in explicit if x is not None)
        remaining = total - sum_explicit
        if remaining < 0:
            raise ValueError(f"Sum of explicit lengths ({sum_explicit}) exceeds total ({total})")
        if remaining % n_none != 0:
            raise ValueError(f"Remaining {remaining} not evenly divisible among {n_none} auto-sized pieces")
        auto_len = remaining // n_none

        return [(x if x is not None else auto_len) for x in explicit]

    # --- Interleaved QKV ---

    def _transform_qkv_interleaved(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """
        Convert packed vLLM QKV heads to the interleaved HF layout.
        """
        if not transform.pieces:
            raise ValueError("qkv_interleaved requires at least one output piece")

        meta = transform.metadata
        num_heads_attr = meta.get("num_heads_attr", "num_heads")
        num_kv_heads_attr = meta.get("num_kv_heads_attr", "num_kv_heads")
        head_dim_attr = meta.get("head_dim_attr", "head_dim")

        num_heads = self._get_module_attr(module, num_heads_attr)
        num_kv_heads = self._get_module_attr(module, num_kv_heads_attr, default=num_heads)
        head_dim = self._get_module_attr(module, head_dim_attr)

        if num_heads is None or head_dim is None:
            raise ValueError(
                f"qkv_interleaved requires num_heads and head_dim; module {type(module).__name__} missing attributes"
            )
        num_heads = int(num_heads)
        num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else num_heads
        head_dim = int(head_dim)

        q_size = num_heads * head_dim
        k_size = num_kv_heads * head_dim
        v_size = num_kv_heads * head_dim
        total = q_size + k_size + v_size

        original_shape = param.shape
        rest = original_shape[1:]  # e.g. (hidden_size,)

        # In vLLM layout: [Q0..Qn, K0..Kn, V0..Vn]
        q_all = param[:q_size]  # (num_heads * head_dim, ...)
        k_all = param[q_size : q_size + k_size]
        v_all = param[q_size + k_size :]

        # Reshape to per-head tensors
        q_heads = q_all.reshape(num_heads, head_dim, *rest)  # (nh, hd, ...)
        k_heads = k_all.reshape(num_kv_heads, head_dim, *rest)  # (nkv, hd, ...)
        v_heads = v_all.reshape(num_kv_heads, head_dim, *rest)  # (nkv, hd, ...)

        # For GQA: repeat K/V heads to match Q heads if needed
        if num_kv_heads != num_heads:
            # GQA retains vLLM layout because shared KV heads cannot be interleaved per query head.
            yield TransformFragment(name=transform.pieces[0].hf_name, param=param)
            return

        # Interleave: [Q0,K0,V0, Q1,K1,V1, ...]
        # Stack: (num_heads, 3, head_dim, ...)
        qkv_interleaved = torch.stack([q_heads, k_heads, v_heads], dim=1)
        # Reshape back to (total, ...)
        result = qkv_interleaved.reshape(total, *rest)

        yield TransformFragment(name=transform.pieces[0].hf_name, param=result)

    # --- Fused MoE ---

    def _transform_fused_moe(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
        full_name: str | None,
    ) -> Iterable[TransformFragment]:
        """
        Decompose fused MoE `w13` and `w2` tensors into per-expert HF tensors.
        """
        meta = transform.metadata
        if full_name is None:
            raise ValueError("fused_moe transform requires the full vLLM parameter name")

        # Path 1: from expert_params_mapping format
        if "expert_mapping" in meta:
            yield from self._fused_moe_from_expert_mapping(transform, param, module, full_name)
            return

        # Path 2: from explicit w13/w2/gate/up/down naming
        gate_name = meta.get("gate_name", "")
        up_name = meta.get("up_name", "")
        down_name = meta.get("down_name", "")
        w13_name = meta.get("w13_name", "w13_weight")
        w2_name = meta.get("w2_name", "w2_weight")
        num_experts = meta.get("num_experts")
        num_experts_attr = meta.get("num_experts_attr", "num_experts")
        ep_aware = meta.get("ep_aware", True)

        if num_experts is None and module is not None:
            num_experts = self._get_module_attr(module, num_experts_attr)
        if num_experts is None:
            raise ValueError("fused_moe transform requires num_experts")
        num_experts = int(num_experts)

        # Determine local expert range for EP
        ep_size = getattr(module, "ep_size", 1) if module is not None else 1
        if ep_aware and ep_size > 1:
            ep_rank = self.ep_rank
            num_local = num_experts // ep_size
            local_start = ep_rank * num_local
        else:
            num_local = param.shape[0]
            local_start = 0

        is_w13 = full_name.endswith(w13_name)
        is_w2 = full_name.endswith(w2_name)

        if is_w13:
            for local_i in range(num_local):
                global_i = local_start + local_i
                expert = param[local_i]
                split = expert.shape[0] // 2
                gate_slice = expert.narrow(0, 0, split)
                up_slice = expert.narrow(0, split, split)
                hf_gate = gate_name.format(i=global_i)
                hf_up = up_name.format(i=global_i)
                yield TransformFragment(name=hf_gate, param=gate_slice)
                yield TransformFragment(name=hf_up, param=up_slice)
        elif is_w2:
            for local_i in range(num_local):
                global_i = local_start + local_i
                down_slice = param[local_i]  # (hidden, intermediate)
                hf_down = down_name.format(i=global_i)
                yield TransformFragment(name=hf_down, param=down_slice)
        else:
            raise ValueError(
                f"fused_moe: {full_name!r} does not match {w13_name!r} or {w2_name!r}; param shape is {param.shape}"
            )

    def _fused_moe_from_expert_mapping(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
        full_name: str | None,
    ) -> Iterable[TransformFragment]:
        """
        Convert fused MoE weights described by `expert_params_mapping`.
        """
        expert_mapping = transform.metadata.get("expert_mapping", [])

        ep_size = getattr(module, "ep_size", 1) if module is not None else 1
        num_experts = len(set(eid for _, _, eid, _ in expert_mapping))

        if ep_size > 1:
            ep_rank = self.ep_rank
            num_local = num_experts // ep_size
            local_start = ep_rank * num_local
            local_expert_ids = set(range(local_start, local_start + num_local))
        else:
            local_expert_ids = set(eid for _, _, eid, _ in expert_mapping)
            local_start = 0

        for packed_suffix, hf_name, expert_id, shard_id in expert_mapping:
            if full_name is not None and not full_name.endswith(packed_suffix):
                continue
            if expert_id not in local_expert_ids:
                continue
            local_i = expert_id - local_start
            if shard_id == "w1":
                expert = param[local_i]
                split = expert.shape[0] // 2
                slice_tensor = expert.narrow(0, 0, split)
            elif shard_id == "w3":
                expert = param[local_i]
                split = expert.shape[0] // 2
                slice_tensor = expert.narrow(0, split, split)
            elif shard_id == "w2":
                # down_proj = row local_i of w2
                slice_tensor = param[local_i]
            else:
                raise ValueError(f"Unknown expert shard_id: {shard_id!r}; expected 'w1', 'w2', or 'w3'")
            yield TransformFragment(name=hf_name, param=slice_tensor)

    # --- Expert matrix ---

    def _transform_expert_matrix(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Reshape flat expert matrix to (num_experts, dim1, dim2).

        Used by bert_with_rope-style models where all expert weights are packed
        into a single 2D matrix.
        """
        if not transform.pieces:
            raise ValueError("expert_matrix requires an output piece")

        num_experts = transform.metadata.get("num_experts")
        transpose_w2 = transform.metadata.get("transpose_w2", False)
        num_experts_attr = transform.metadata.get("num_experts_attr", "num_experts")

        if num_experts is None and module is not None:
            num_experts = self._get_module_attr(module, num_experts_attr)
        if num_experts is None:
            raise ValueError("expert_matrix requires num_experts in metadata")
        num_experts = int(num_experts)

        # Reshape to (num_experts, ...)
        remaining_shape = param.shape[0] // num_experts
        reshaped = param.reshape(num_experts, remaining_shape, *param.shape[1:])

        if transpose_w2:
            reshaped = reshaped.transpose(-2, -1).contiguous()

        yield TransformFragment(name=transform.pieces[0].hf_name, param=reshaped)

    # --- Transpose ---

    def _transform_transpose(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Swap two tensor dimensions."""
        if not transform.pieces:
            raise ValueError("transpose requires an output piece")

        dims = transform.metadata.get("dims", (0, 1))
        if len(dims) < 2:
            raise ValueError("transpose requires dims tuple of length >= 2")

        transposed = param.transpose(int(dims[0]), int(dims[1]))
        yield TransformFragment(name=transform.pieces[0].hf_name, param=transposed)

    # --- Reshape ---

    def _transform_reshape(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Reshape tensor with optional expression evaluation."""
        if not transform.pieces:
            raise ValueError("reshape requires an output piece")

        shape_expr = transform.metadata.get("shape_expr")
        if shape_expr is None:
            raise ValueError("reshape requires 'shape_expr' in metadata")

        if isinstance(shape_expr, (tuple, list)):
            new_shape = tuple(shape_expr)
        elif isinstance(shape_expr, str):
            # Build eval locals from module attributes
            eval_locals: dict = {"param": param, "shape": param.shape}
            if module is not None:
                for attr in dir(module):
                    if not attr.startswith("_"):
                        try:
                            eval_locals[attr] = getattr(module, attr)
                        except Exception:
                            pass
            try:
                new_shape = eval(shape_expr, {"__builtins__": {}}, eval_locals)  # noqa: S307
            except Exception as exc:
                raise ValueError(f"Failed to evaluate shape_expr {shape_expr!r}: {exc}") from exc
        else:
            new_shape = shape_expr

        yield TransformFragment(
            name=transform.pieces[0].hf_name,
            param=param.reshape(new_shape),
        )

    # --- QK rotary permutation ---

    def _transform_permute_qk_rotary(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Reverse the rotary weight permutation used during loading.

        During HF→vLLM loading, Q/K weights go through::

            # from convert_hf_to_vllm or permute_rotary_embeddings
            param.view(n_heads, head_dim // 2, 2, hidden).transpose(1, 2).reshape(original)

        The inverse (vLLM→HF) is::

            param.view(n_heads, 2, head_dim // 2, hidden).transpose(1, 2).reshape(original)
        """
        if not transform.pieces:
            raise ValueError("permute_qk_rotary requires an output piece")

        head_dim = transform.metadata.get("head_dim")
        head_dim_attr = transform.metadata.get("head_dim_attr", "head_dim")

        if head_dim is None and module is not None:
            head_dim = self._get_module_attr(module, head_dim_attr)

        if head_dim is None:
            raise ValueError(
                f"permute_qk_rotary: head_dim not found. "
                f"Set head_dim in metadata or ensure module has {head_dim_attr!r} attribute."
            )
        head_dim = int(head_dim)
        half = head_dim // 2

        original_shape = param.shape
        n_heads = original_shape[0] // head_dim
        rest = original_shape[1:]  # e.g. (hidden_size,) for weight matrices

        # Reverse the loaded layout by swapping its pair and half head dimensions.
        reshaped = param.reshape(n_heads, 2, half, *rest)
        permuted = reshaped.transpose(1, 2)  # (n_heads, half, 2, ...)
        result = permuted.reshape(original_shape)

        yield TransformFragment(name=transform.pieces[0].hf_name, param=result)

    # --- Scalar extraction ---

    def _transform_scalar_extract(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Extract a single element from a batched/stacked scalar parameter."""
        if not transform.pieces:
            raise ValueError("scalar_extract requires an output piece")

        index = transform.metadata.get("index", 0)
        extracted = param[int(index)]
        yield TransformFragment(name=transform.pieces[0].hf_name, param=extracted)

    # --- Index selection ---

    def _transform_index_select(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Select rows/columns using an index list."""
        if not transform.pieces:
            raise ValueError("index_select requires an output piece")

        dim = int(transform.metadata.get("dim", 0))
        indices = transform.metadata.get("indices")
        index_attr = transform.metadata.get("index_attr")

        if indices is None and index_attr is not None and module is not None:
            indices = self._get_module_attr(module, index_attr)

        if indices is None:
            raise ValueError(
                "index_select requires 'indices' in metadata or 'index_attr' pointing to a module attribute"
            )

        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(list(indices), dtype=torch.long, device=param.device)

        selected = torch.index_select(param, dim, indices)
        yield TransformFragment(name=transform.pieces[0].hf_name, param=selected)

    # --- Alias ---

    def _transform_alias(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Passthrough with a renamed HF name (shared parameter alias)."""
        if not transform.pieces:
            raise ValueError("alias requires an output piece")
        yield TransformFragment(name=transform.pieces[0].hf_name, param=param)

    # --- Derivation ---

    def _transform_derive(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Compute a derived tensor from a source parameter.

        The derivation function is provided via metadata["fn"] (callable) or
        metadata["fn_name"] (string key into a registry on the module).

        The callable signature is: fn(param, module) -> tensor
        """
        if not transform.pieces:
            raise ValueError("derive requires an output piece")

        fn: Callable | None = transform.metadata.get("fn")
        fn_name: str | None = transform.metadata.get("fn_name")

        if fn is None and fn_name is not None and module is not None:
            fn = getattr(module, fn_name, None)

        if fn is None:
            raise ValueError("derive transform requires 'fn' callable or 'fn_name' pointing to a method on the module")

        try:
            result = fn(param, module)
        except Exception as exc:
            raise ValueError(f"derive fn failed for {transform.pieces[0].hf_name!r}: {exc}") from exc

        if isinstance(result, (list, tuple)):
            # fn may return multiple (name, tensor) pairs
            for name, tensor in result:
                yield TransformFragment(name=name, param=tensor)
        else:
            yield TransformFragment(name=transform.pieces[0].hf_name, param=result)

    # --- Custom transforms ---

    def _transform_custom(
        self,
        transform: WeightTransform,
        param: torch.Tensor,
        module: nn.Module | None,
    ) -> Iterable[TransformFragment]:
        """Delegate to a ModelWeightTransform.vllm_to_hf() implementation."""
        custom_transform = transform.metadata.get("transform_instance")
        if custom_transform is None:
            raise ValueError("custom transform requires 'transform_instance' in metadata")
        if not hasattr(custom_transform, "vllm_to_hf"):
            raise ValueError(
                f"Custom transform {type(custom_transform).__name__!r} is missing the vllm_to_hf() method"
            )

        # Determine full_name (best effort)
        full_name = transform.pieces[0].hf_name if transform.pieces else getattr(param, "_vllm_full_name", "unknown")

        for name, tensor in custom_transform.vllm_to_hf(
            full_name=full_name,
            param=param,
            module=module if module is not None else _DummyModule(),
            tp_rank=self.tp_rank,
        ):
            yield TransformFragment(name=name, param=tensor)

    # --- Utilities ---

    @staticmethod
    def _get_module_attr(
        module: nn.Module | None,
        attr: str,
        default: object | None = None,
    ) -> object | None:
        """Safely read an attribute from a module."""
        if module is None:
            return default
        return getattr(module, attr, default)


class _DummyModule(nn.Module):
    """Placeholder module for transforms called without a real module."""

    pass
