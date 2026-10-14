"""
Weight Layout Infrastructure for vLLM Models

This module provides a unified builder API for describing how vLLM model weights
map to HuggingFace checkpoint weights, including packed/fused parameter transforms,
parameter renaming, nested model mounting, and sharding metadata.

Key concepts:
- WeightLayoutPlan: Complete weight layout for a model or sub-model
- WeightLayoutRule: A vLLM -> HF parameter mapping with optional transforms
- WeightTransform: Tensor-level transformations (split, concat, reshape, etc.)
- WeightLayoutBuilder: Public API for models to declare their weight layout
- WeightLayoutMount: Composition of nested model layouts via module mounting

Design philosophy:
- Transforms declarative, not procedural
- Built-in transforms for common patterns (qkv, merged columns, MoE, etc.)
- Model-specific custom logic via ModelWeightTransform protocol
- All mapping metadata explicit in the plan
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    Protocol,
)

import torch
import torch.nn as nn

# ============================================================================
# Enums
# ============================================================================


class MatchMode(enum.Enum):
    """Pattern matching modes for weight layout rules."""

    SUFFIX = "suffix"
    PREFIX = "prefix"
    EXACT = "exact"
    REGEX = "regex"


# ============================================================================
# Core Data Structures
# ============================================================================


@dataclass(frozen=True)
class WeightPiece:
    """A fragment of a tensor produced by a WeightTransform.

    Attributes:
        hf_name: HuggingFace parameter name for this piece
        shard_id: Optional shard identifier (int, str, or tuple of ints)
                 Used for expert parameters, multi-head selections, etc.
        offset: Optional start position along split axis (for view-based slicing)
        length: Optional slice length along split axis (for view-based slicing)
        transform: Optional nested transform to apply to this piece
    """

    hf_name: str
    shard_id: str | int | tuple[int, ...] | None = None
    offset: int | None = None
    length: int | None = None
    transform: WeightTransform | None = None


@dataclass(frozen=True)
class WeightTransform:
    """Tensor-level transformation specification.

    Attributes:
        kind: Type of transformation. Supported built-in kinds:
              "identity", "qkv", "merged_column", "split",
              "qkv_interleaved", "fused_moe", "expert_matrix",
              "transpose", "reshape", "permute_qk_rotary",
              "scalar_extract", "index_select", "alias",
              "derive", "custom"
        pieces: Output fragments for split/unpack transforms
        axis: Primary axis for split/concat operations
        metadata: Transform-specific parameters (see per-kind docs)
    """

    kind: str
    pieces: tuple[WeightPiece, ...] = ()
    axis: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def identity(cls, hf_name: str | None = None) -> WeightTransform:
        """Passthrough transform (no tensor modification).

        Args:
            hf_name: Optional HF name override. If None, name is unchanged.
        """
        pieces = (WeightPiece(hf_name=hf_name),) if hf_name is not None else ()
        return cls(kind="identity", pieces=pieces)

    @classmethod
    def qkv(
        cls,
        q_name: str,
        k_name: str,
        v_name: str,
        *,
        num_heads_attr: str = "num_heads",
        num_kv_heads_attr: str = "num_kv_heads",
        head_size_attr: str = "head_size",
    ) -> WeightTransform:
        """Split a fused QKV weight into separate Q, K, V.

        The actual sizes are read from module attributes at transform time
        to support GQA (num_kv_heads < num_heads).

        Args:
            q_name: HF name for the Q weight
            k_name: HF name for the K weight
            v_name: HF name for the V weight
            num_heads_attr: Module attribute name for number of Q heads
            num_kv_heads_attr: Module attribute name for number of KV heads
            head_size_attr: Module attribute name for per-head dimension
        """
        return cls(
            kind="qkv",
            pieces=(
                WeightPiece(hf_name=q_name, shard_id="q"),
                WeightPiece(hf_name=k_name, shard_id="k"),
                WeightPiece(hf_name=v_name, shard_id="v"),
            ),
            metadata={
                "num_heads_attr": num_heads_attr,
                "num_kv_heads_attr": num_kv_heads_attr,
                "head_size_attr": head_size_attr,
            },
        )

    @classmethod
    def merged_column(
        cls,
        pieces: Sequence[tuple[str, int | None]],
        axis: int = 0,
    ) -> WeightTransform:
        """Split a merged column weight (e.g. gate_up_proj -> gate_proj, up_proj).

        Args:
            pieces: List of (hf_name, length) pairs.
                    If length is None the pieces are split equally.
            axis: Axis along which to split (default 0)
        """
        weight_pieces = tuple(WeightPiece(hf_name=name, length=length) for name, length in pieces)
        return cls(kind="merged_column", pieces=weight_pieces, axis=axis)

    @classmethod
    def split(
        cls,
        pieces: Sequence[tuple[str, int | None]],
        axis: int = 0,
    ) -> WeightTransform:
        """Generic split transform.

        Args:
            pieces: List of (hf_name, length) pairs.
                    If length is None the pieces are split equally.
            axis: Axis along which to split
        """
        weight_pieces = tuple(WeightPiece(hf_name=name, length=length) for name, length in pieces)
        return cls(kind="split", pieces=weight_pieces, axis=axis)

    @classmethod
    def fused_moe(
        cls,
        w13_name: str,
        w2_name: str,
        gate_name: str,
        up_name: str,
        down_name: str,
        *,
        num_experts: int,
        num_experts_attr: str = "num_experts",
        ep_aware: bool = True,
    ) -> WeightTransform:
        """Decompose fused MoE weights (w13, w2) to per-expert HF tensors.

        vLLM fuses all experts into:
          w13_weight: [num_local_experts * 2, intermediate, hidden]
                      (gate+up interleaved)
          w2_weight:  [num_local_experts,     hidden, intermediate]

        The transform unfuses to:
          gate_name.{i}.weight, up_name.{i}.weight, down_name.{i}.weight

        Args:
            w13_name: vLLM suffix for the w13 fused weight (e.g. "w13_weight")
            w2_name: vLLM suffix for the w2 weight (e.g. "w2_weight")
            gate_name: HF name pattern for gate weights (must contain {i})
            up_name: HF name pattern for up weights (must contain {i})
            down_name: HF name pattern for down weights (must contain {i})
            num_experts: Total number of experts across all ranks
            num_experts_attr: Module attribute for num_experts (fallback)
            ep_aware: Whether to use EP rank for local expert selection
        """
        return cls(
            kind="fused_moe",
            metadata={
                "w13_name": w13_name,
                "w2_name": w2_name,
                "gate_name": gate_name,
                "up_name": up_name,
                "down_name": down_name,
                "num_experts": num_experts,
                "num_experts_attr": num_experts_attr,
                "ep_aware": ep_aware,
            },
        )

    @classmethod
    def fused_moe_from_mapping(
        cls,
        expert_mapping: Sequence[tuple[str, str, int, str]],
    ) -> WeightTransform:
        """Build a fused_moe transform from vLLM expert_params_mapping format.

        Args:
            expert_mapping: List of (packed_suffix, hf_name, expert_id, shard_id)
                where shard_id is "w1" / "w2" / "w3".
        """
        return cls(
            kind="fused_moe",
            metadata={"expert_mapping": list(expert_mapping)},
        )

    def expert_matrix(
        cls,
        hf_name: str,
        *,
        num_experts: int,
        transpose_w2: bool = False,
    ) -> WeightTransform:
        """Reshape expert weight to (num_experts, intermediate, hidden).

        Used by models like bert_with_rope where expert weights are stored
        as a single 2D matrix.
        """
        return cls(
            kind="expert_matrix",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={
                "num_experts": num_experts,
                "transpose_w2": transpose_w2,
            },
        )

    @classmethod
    def transpose(
        cls,
        hf_name: str,
        dims: tuple[int, int] = (0, 1),
    ) -> WeightTransform:
        """Swap two dimensions of a tensor."""
        return cls(
            kind="transpose",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={"dims": dims},
        )

    @classmethod
    def reshape(
        cls,
        hf_name: str,
        shape_expr: tuple | str,
    ) -> WeightTransform:
        """Reshape a tensor.

        Args:
            hf_name: Output HF name
            shape_expr: New shape as tuple or string expression.
                String may reference module attributes, e.g.
                "(num_experts, -1, hidden_size)".
        """
        return cls(
            kind="reshape",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={"shape_expr": shape_expr},
        )

    @classmethod
    def permute_qk_rotary(
        cls,
        hf_name: str,
        *,
        head_dim: int | None = None,
        head_dim_attr: str = "head_dim",
    ) -> WeightTransform:
        """Permute Q or K weight for rotary embedding compatibility.

        Standard Llama/Mistral permutation:
          view(n_heads, 2, half_head_dim, hidden)
          -> transpose(1, 2) -> reshape(original)

        Args:
            hf_name: Output HF name
            head_dim: Fixed head dimension (if known at build time)
            head_dim_attr: Module attribute for head dimension (fallback)
        """
        return cls(
            kind="permute_qk_rotary",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={"head_dim": head_dim, "head_dim_attr": head_dim_attr},
        )

    @classmethod
    def scalar_extract(
        cls,
        hf_name: str,
        index: int = 0,
    ) -> WeightTransform:
        """Extract a single element from a batched scalar parameter."""
        return cls(
            kind="scalar_extract",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={"index": index},
        )

    @classmethod
    def index_select(
        cls,
        hf_name: str,
        *,
        indices: Sequence[int] | None = None,
        index_attr: str | None = None,
        dim: int = 0,
    ) -> WeightTransform:
        """Select rows/elements using an index list.

        Args:
            hf_name: Output HF name
            indices: Explicit list of indices
            index_attr: Module attribute name containing indices (runtime)
            dim: Dimension to select along
        """
        return cls(
            kind="index_select",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={"indices": list(indices) if indices else None, "index_attr": index_attr, "dim": dim},
        )

    @classmethod
    def alias(cls, hf_name: str) -> WeightTransform:
        """Alias a parameter under a different HF name (shared weights)."""
        return cls(
            kind="alias",
            pieces=(WeightPiece(hf_name=hf_name),),
        )

    @classmethod
    def derive(
        cls,
        hf_name: str,
        *,
        fn: Callable | None = None,
        fn_name: str | None = None,
    ) -> WeightTransform:
        """Derive a parameter from one or more source parameters.

        Used for parameters that do not appear in HF checkpoints but are
        computed (e.g. LongCat w_kc/w_vc derived from kv_b_proj).

        Args:
            hf_name: Output HF name of the derived tensor
            fn: Callable(param, module) -> tensor
            fn_name: String key for function registry (alternative to fn)
        """
        return cls(
            kind="derive",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={"fn": fn, "fn_name": fn_name},
        )

    @classmethod
    def qkv_interleaved(
        cls,
        hf_name: str,
        *,
        num_heads_attr: str = "num_heads",
        num_kv_heads_attr: str = "num_kv_heads",
        head_dim_attr: str = "head_dim",
    ) -> WeightTransform:
        """Reorder interleaved QKV layout to vLLM packed QKV layout.

        BLOOM/GPT-NeoX/Persimmon/Falcon store QKV as interleaved per-head:
          checkpoint: [Q0, K0, V0, Q1, K1, V1, ..., Qn, Kn, Vn]
          vLLM:       [Q0..Qn, K0..Kn, V0..Vn]

        This transform is used for HF->vLLM direction only (loading).
        For PSRL export (vLLM->HF), use qkv_interleaved_reverse.
        """
        return cls(
            kind="qkv_interleaved",
            pieces=(WeightPiece(hf_name=hf_name),),
            metadata={
                "num_heads_attr": num_heads_attr,
                "num_kv_heads_attr": num_kv_heads_attr,
                "head_dim_attr": head_dim_attr,
            },
        )

    @classmethod
    def custom(
        cls,
        transform_instance: ModelWeightTransform,
    ) -> WeightTransform:
        """Custom model-specific transform.

        The transform_instance must implement ModelWeightTransform protocol.
        """
        return cls(
            kind="custom",
            metadata={"transform_instance": transform_instance},
        )


# ============================================================================
# ModelWeightTransform Protocol
# ============================================================================


class ModelWeightTransform(Protocol):
    """Protocol for model-local custom weight transforms.

    Models with complex, non-reusable weight transformations should implement
    this protocol and register it via WeightTransform.custom().

    The transform must be deterministic given (full_name, param, module, tp_rank).
    """

    name: str

    def vllm_to_hf(
        self,
        *,
        full_name: str,
        param: torch.Tensor,
        module: nn.Module,
        tp_rank: int,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Convert vLLM parameter to HF format.

        Yields:
            (hf_name, tensor) pairs
        """
        ...

    def hf_to_vllm(
        self,
        *,
        name: str,
        weight: torch.Tensor,
        module: nn.Module,
        tp_rank: int,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Convert HF weight to vLLM format (for loading).

        Yields:
            (vllm_name, tensor) pairs
        """
        ...


# ============================================================================
# Name Mapping
# ============================================================================


class NameMapper:
    """Unidirectional name mapper supporting prefix/substring replacement.

    This is distinct from the vLLM utils.py WeightsMapper (which handles
    checkpoint -> vLLM loading). This class handles bidirectional HF <-> vLLM
    name transformations for PSRL export.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        mode: Literal["prefix", "substr", "suffix", "regex"] = "substr",
    ):
        self._pairs = list(pairs)
        self._mode = mode

    def apply(self, name: str) -> str | None:
        """Apply the first matching substitution."""
        for src, dst in self._pairs:
            if self._mode == "prefix" and name.startswith(src):
                return dst + name[len(src) :]
            elif self._mode == "substr" and src in name:
                return name.replace(src, dst, 1)
            elif self._mode == "suffix" and name.endswith(src):
                return name[: -len(src)] + dst
            elif self._mode == "regex":
                result = re.sub(src, dst, name)
                if result != name:
                    return result
        return None

    def apply_or_identity(self, name: str) -> str:
        result = self.apply(name)
        return result if result is not None else name


@dataclass(frozen=True)
class ReversibleNameMap:
    """Bidirectional name mapping between vLLM and HuggingFace parameter names.

    Both directions must be explicitly provided for correctness.
    Use from_prefix_pairs() for simple prefix substitutions.

    Attributes:
        vllm_to_hf: Mapping applied when converting vLLM names to HF names
        hf_to_vllm: Mapping applied when converting HF names to vLLM names
    """

    vllm_to_hf: NameMapper | Mapping[str, str] | None = None
    hf_to_vllm: NameMapper | Mapping[str, str] | None = None

    def to_hf(self, vllm_name: str) -> str | None:
        """Convert a vLLM parameter name to HF name."""
        if self.vllm_to_hf is None:
            return None
        if isinstance(self.vllm_to_hf, NameMapper):
            return self.vllm_to_hf.apply(vllm_name)
        return self.vllm_to_hf.get(vllm_name)

    def to_hf_or_identity(self, vllm_name: str) -> str:
        result = self.to_hf(vllm_name)
        return result if result is not None else vllm_name

    def to_vllm(self, hf_name: str) -> str | None:
        """Convert an HF parameter name to vLLM name."""
        if self.hf_to_vllm is None:
            return None
        if isinstance(self.hf_to_vllm, NameMapper):
            return self.hf_to_vllm.apply(hf_name)
        return self.hf_to_vllm.get(hf_name)

    def is_reversible(self) -> bool:
        return self.vllm_to_hf is not None and self.hf_to_vllm is not None

    @classmethod
    def from_prefix_pairs(
        cls,
        pairs: Mapping[str, str],
        *,
        mode: Literal["prefix", "substr", "suffix", "regex"] = "substr",
    ) -> ReversibleNameMap:
        """Create a reversible map from (vllm_prefix -> hf_prefix) pairs.

        Example::

            ReversibleNameMap.from_prefix_pairs(
                {
                    "language_model.layers.": "language_model.model.layers.",
                }
            )

        The reverse mapping is automatically derived by swapping keys/values.
        """
        fwd_pairs = list(pairs.items())
        rev_pairs = [(v, k) for k, v in fwd_pairs]
        return cls(
            vllm_to_hf=NameMapper(fwd_pairs, mode=mode),
            hf_to_vllm=NameMapper(rev_pairs, mode=mode),
        )

    @classmethod
    def from_exact_pairs(cls, pairs: Mapping[str, str]) -> ReversibleNameMap:
        """Create a reversible map from exact vllm_name -> hf_name pairs."""
        fwd = dict(pairs)
        rev = {v: k for k, v in fwd.items()}
        return cls(vllm_to_hf=fwd, hf_to_vllm=rev)


# ============================================================================
# Sync Policy
# ============================================================================


@dataclass(frozen=True)
class WeightSyncPolicy:
    """Policy controlling which parameters are included in PSRL synchronization.

    Default: include all parameters, no exclusions.
    Use exclude_prefixes / exclude_substrs to skip derived buffers, rotary
    caches, vision-only parameters, etc.

    Note: MTP and Eagle parameters are included by default.
    """

    include_prefixes: tuple[str, ...] = ()
    exclude_prefixes: tuple[str, ...] = ()
    exclude_substrs: tuple[str, ...] = ()

    def should_include(self, name: str) -> bool:
        """Return True if this parameter should be included in synchronization."""
        # Explicit exclusion wins over everything
        if any(name.startswith(p) for p in self.exclude_prefixes):
            return False
        if any(s in name for s in self.exclude_substrs):
            return False
        # If include_prefixes is set, only names matching those prefixes are included
        if self.include_prefixes:
            return any(name.startswith(p) for p in self.include_prefixes)
        return True

    def merge(self, other: WeightSyncPolicy) -> WeightSyncPolicy:
        """Merge two policies (union of exclusions, intersection of inclusions)."""
        return WeightSyncPolicy(
            include_prefixes=self.include_prefixes + other.include_prefixes,
            exclude_prefixes=self.exclude_prefixes + other.exclude_prefixes,
            exclude_substrs=self.exclude_substrs + other.exclude_substrs,
        )


# ============================================================================
# Layout Rule and Mount
# ============================================================================


@dataclass(frozen=True)
class WeightLayoutRule:
    """A rule mapping vLLM parameter(s) to HuggingFace parameter(s).

    Attributes:
        vllm_pattern: Pattern for matching vLLM parameter names
        hf_patterns: Target HF parameter name(s) (may be empty for derive)
        transform: Transformation to apply
        match: How to match vllm_pattern against parameter names
        module_types: Optional filter on module type
        metadata: Extra metadata for this rule
    """

    vllm_pattern: str
    hf_patterns: tuple[str, ...]
    transform: WeightTransform
    match: MatchMode = MatchMode.SUFFIX
    module_types: tuple[type, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, name: str, module: nn.Module | None = None) -> bool:
        """Return True if this rule applies to the given parameter name."""
        pattern = self.vllm_pattern

        if self.match == MatchMode.EXACT:
            if name != pattern:
                return False
        elif self.match == MatchMode.SUFFIX:
            if not name.endswith(pattern):
                return False
        elif self.match == MatchMode.PREFIX:
            if not name.startswith(pattern):
                return False
        elif self.match == MatchMode.REGEX:
            if not re.fullmatch(pattern, name):
                return False

        if self.module_types and module is not None and not isinstance(module, tuple(self.module_types)):
            return False

        return True


@dataclass(frozen=True)
class WeightLayoutMount:
    """Mount a nested WeightLayoutPlan under a module prefix.

    Represents composing sub-model layouts (e.g. language_model within a VL
    wrapper). The prefix is the vLLM module path component (e.g. "language_model").

    Attributes:
        prefix: Module prefix at which to mount the sub-plan
        plan: The sub-model's WeightLayoutPlan
        name_map: Optional additional name map applied at this mount boundary
    """

    prefix: str
    plan: WeightLayoutPlan
    name_map: ReversibleNameMap | None = None


# ============================================================================
# Resolved Plan (after flatten)
# ============================================================================


@dataclass(frozen=True)
class FlattenedRule:
    """A WeightLayoutRule with its resolved module prefix."""

    prefix: str
    rule: WeightLayoutRule


@dataclass(frozen=True)
class ResolvedWeightLayoutPlan:
    """Flattened weight layout plan with all mounts resolved into flat rules."""

    rules: tuple[FlattenedRule, ...] = ()
    name_map: ReversibleNameMap | None = None
    sync: WeightSyncPolicy = field(default_factory=WeightSyncPolicy)

    def __post_init__(self):
        # Build prefix index for O(1) lookup instead of O(rules) per param.
        # Groups rules by their prefix for efficient matching.
        index: dict[str, list[FlattenedRule]] = {}
        for flattened in self.rules:
            index.setdefault(flattened.prefix, []).append(flattened)
        # frozen=True requires object.__setattr__ for post-init mutation
        object.__setattr__(self, "_prefix_index", index)

    def should_exclude(self, name: str) -> bool:
        return not self.sync.should_include(name)

    def matches_rules(
        self,
        param_name: str,
        module: nn.Module | None = None,
    ) -> Sequence[FlattenedRule]:
        """Find all rules whose pattern matches the given parameter name."""
        matches = []
        index: dict[str, list[FlattenedRule]] = getattr(self, "_prefix_index", {})

        # Check rules with empty prefix (apply to all params)
        for flattened in index.get("", []):
            if flattened.rule.matches(param_name, module):
                matches.append(flattened)

        # Check rules whose prefix matches the param_name prefix
        # Extract candidate prefixes from param_name by splitting on "."
        parts = param_name.split(".")
        candidate = ""
        for part in parts[:-1]:  # skip last part (param name)
            candidate = f"{candidate}.{part}" if candidate else part
            rules_for_prefix = index.get(candidate)
            if rules_for_prefix is None:
                continue
            prefix_dot = candidate + "."
            relative_name = param_name[len(prefix_dot) :]
            for flattened in rules_for_prefix:
                if flattened.rule.matches(relative_name, module):
                    matches.append(flattened)

        return matches


# ============================================================================
# WeightLayoutPlan
# ============================================================================


@dataclass(frozen=True)
class WeightLayoutPlan:
    """Complete weight layout description for a vLLM model.

    Describes all parameter transformations from vLLM runtime format to
    HuggingFace checkpoint format.

    Attributes:
        rules: Direct mapping rules for this model's own parameters
        mounts: Sub-model plans mounted under specific module prefixes
        name_map: Global HF <-> vLLM name mapping applied after transforms
        sync: Policy controlling which parameters are synchronised
    """

    rules: tuple[WeightLayoutRule, ...] = ()
    mounts: tuple[WeightLayoutMount, ...] = ()
    name_map: ReversibleNameMap | None = None
    sync: WeightSyncPolicy = field(default_factory=WeightSyncPolicy)

    def flatten(self) -> ResolvedWeightLayoutPlan:
        """Flatten all nested mounts into a single resolved plan."""
        flattened_rules: list[FlattenedRule] = []
        merged_sync = self.sync

        # Direct rules (no prefix)
        for rule in self.rules:
            flattened_rules.append(FlattenedRule(prefix="", rule=rule))

        # Mounted plans
        for mount in self.mounts:
            mounted_resolved = mount.plan.flatten()
            merged_sync = merged_sync.merge(mounted_resolved.sync)
            for flattened in mounted_resolved.rules:
                new_prefix = f"{mount.prefix}.{flattened.prefix}".rstrip(".") if flattened.prefix else mount.prefix
                flattened_rules.append(FlattenedRule(prefix=new_prefix, rule=flattened.rule))

        return ResolvedWeightLayoutPlan(
            rules=tuple(flattened_rules),
            name_map=self.name_map,
            sync=merged_sync,
        )

    def should_exclude(self, name: str) -> bool:
        return not self.sync.should_include(name)


# ============================================================================
# Builder API
# ============================================================================


class WeightLayoutBuilder:
    """Builder for declaring weight layouts in model code.

    Usage::

        def build_weight_layout(self) -> WeightLayoutPlan:
            return (
                WeightLayoutBuilder(self)
                .qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
                .merged("gate_up_proj", [("gate_proj", None), ("up_proj", None)])
                .mount_module("language_model", self.language_model)
                .build()
            )

    Note on ``qkv`` / ``merged`` argument convention:

    The ``vllm`` and target piece names passed to :meth:`qkv` and
    :meth:`merged` are *module-name suffixes* (e.g. ``"qkv_proj"``,
    ``"q_proj"``), **not** parameter names with a ``.weight`` / ``.bias``
    suffix. The builder automatically expands them into a rule for
    ``<suffix>.weight`` and, when the matching module declares a non-None
    ``bias``, a corresponding ``<suffix>.bias`` rule with the same
    splitting layout. This avoids forgetting to register a bias rule for
    layers like Qwen2 attention which use ``QKVParallelLinear(bias=True)``.
    """

    def __init__(self, model: nn.Module | None = None):
        self._model = model
        self._rules: list[WeightLayoutRule] = []
        self._mounts: list[WeightLayoutMount] = []
        self._name_map: ReversibleNameMap | None = None
        self._include_prefixes: list[str] = []
        self._exclude_prefixes: list[str] = []
        self._exclude_substrs: list[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _module_has_bias(self, module_suffix: str) -> bool:
        """Return True if any module under ``self._model`` whose name ends with
        ``module_suffix`` exposes a non-None ``bias`` parameter.

        When ``self._model`` is None we conservatively return True so that the
        caller still emits a ``.bias`` rule — having an unmatched rule is
        harmless, but missing one silently leaks fused bias names through to
        the output.
        """
        if self._model is None:
            return True
        # The suffix is a module-name fragment such as ``"qkv_proj"`` or
        # ``"gate_up_proj"``. We accept either an exact match (top-level
        # module) or a dotted-suffix match (nested module path).
        dotted = "." + module_suffix
        for name, mod in self._model.named_modules():
            if name == module_suffix or name.endswith(dotted):
                if getattr(mod, "bias", None) is not None:
                    return True
        return False

    # ------------------------------------------------------------------
    # Name mapping
    # ------------------------------------------------------------------

    def name_map(self, name_map: ReversibleNameMap) -> WeightLayoutBuilder:
        """Set a global reversible name map for this plan."""
        self._name_map = name_map
        return self

    # ------------------------------------------------------------------
    # Sync policy
    # ------------------------------------------------------------------

    def include_all(self) -> WeightLayoutBuilder:
        """Include all parameters (default behaviour)."""
        self._include_prefixes = []
        return self

    def include_prefix(self, prefix: str) -> WeightLayoutBuilder:
        """Only include parameters whose names start with prefix."""
        self._include_prefixes.append(prefix)
        return self

    def exclude_prefix(self, prefix: str) -> WeightLayoutBuilder:
        """Exclude parameters whose names start with prefix."""
        self._exclude_prefixes.append(prefix)
        return self

    def exclude_substr(self, substr: str) -> WeightLayoutBuilder:
        """Exclude parameters whose names contain substr."""
        self._exclude_substrs.append(substr)
        return self

    # ------------------------------------------------------------------
    # Low-level rule addition
    # ------------------------------------------------------------------

    def add_rule(
        self,
        vllm_pattern: str,
        hf_patterns: str | Sequence[str],
        transform: WeightTransform,
        match: MatchMode | str = MatchMode.SUFFIX,
        module_types: Sequence[type] | None = None,
    ) -> WeightLayoutBuilder:
        """Add a raw WeightLayoutRule."""
        if isinstance(hf_patterns, str):
            hf_patterns = (hf_patterns,)
        else:
            hf_patterns = tuple(hf_patterns)
        if isinstance(match, str):
            match = MatchMode(match)
        rule = WeightLayoutRule(
            vllm_pattern=vllm_pattern,
            hf_patterns=hf_patterns,
            transform=transform,
            match=match,
            module_types=tuple(module_types or []),
        )
        self._rules.append(rule)
        return self

    # ------------------------------------------------------------------
    # Shorthand transform builders
    # ------------------------------------------------------------------

    def identity(
        self,
        vllm: str,
        hf: str | None = None,
        *,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add an identity (passthrough) rule.

        Args:
            vllm: vLLM parameter pattern
            hf: HF parameter name. Defaults to same as vllm if None.
            match: Pattern matching mode
        """
        transform = WeightTransform.identity(hf_name=hf)
        hf_name = hf if hf is not None else vllm
        return self.add_rule(vllm, hf_name, transform, match=match)

    def qkv(
        self,
        vllm: str,
        q: str,
        k: str,
        v: str,
        *,
        match: MatchMode | str = MatchMode.SUFFIX,
        num_heads_attr: str = "num_heads",
        num_kv_heads_attr: str = "num_kv_heads",
        head_size_attr: str = "head_size",
    ) -> WeightLayoutBuilder:
        """Add QKV split rule(s) for a fused QKV linear layer (GQA-aware).

        ``vllm``, ``q``, ``k`` and ``v`` are *module-name suffixes* (e.g.
        ``"qkv_proj"``, ``"q_proj"``) — they must NOT include the trailing
        ``.weight`` / ``.bias``. The builder expands them into:

        * A rule for ``<vllm>.weight`` → ``<q>.weight`` / ``<k>.weight`` /
          ``<v>.weight`` (always emitted).
        * A rule for ``<vllm>.bias``   → ``<q>.bias``   / ``<k>.bias``   /
          ``<v>.bias`` — emitted only when the model exposes at least one
          matching module with a non-None ``bias`` parameter.

        Args:
            vllm: vLLM module-name suffix for the fused QKV linear
                (e.g. ``"qkv_proj"``).
            q, k, v: HF module-name suffixes for the per-head splits
                (e.g. ``"q_proj"``, ``"k_proj"``, ``"v_proj"``).
            match: Pattern matching mode (default suffix).
            num_heads_attr: Module attribute for num Q heads.
            num_kv_heads_attr: Module attribute for num KV heads.
            head_size_attr: Module attribute for head dimension.
        """
        for sub in ("weight", "bias"):
            if sub == "bias" and not self._module_has_bias(vllm):
                continue
            vllm_p = f"{vllm}.{sub}"
            q_p, k_p, v_p = f"{q}.{sub}", f"{k}.{sub}", f"{v}.{sub}"
            transform = WeightTransform.qkv(
                q_p,
                k_p,
                v_p,
                num_heads_attr=num_heads_attr,
                num_kv_heads_attr=num_kv_heads_attr,
                head_size_attr=head_size_attr,
            )
            self.add_rule(vllm_p, (q_p, k_p, v_p), transform, match=match)
        return self

    def merged(
        self,
        vllm: str,
        pieces: Sequence[tuple[str, int | None]],
        *,
        axis: int = 0,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add merged-column split rule(s) (gate_up_proj, in_proj_qkvz, etc.).

        ``vllm`` and the names in ``pieces`` are *module-name suffixes*
        (e.g. ``"gate_up_proj"``, ``"gate_proj"``) — they must NOT include
        the trailing ``.weight`` / ``.bias``. The builder expands them into:

        * A rule for ``<vllm>.weight`` (always emitted).
        * A rule for ``<vllm>.bias`` — emitted only when the model exposes
          at least one matching module with a non-None ``bias`` parameter.

        The same per-piece ``length`` values are reused for both the weight
        and bias rules; for a Linear layer, the bias is split along the
        same output partition as the weight's row partition.

        Args:
            vllm: vLLM module-name suffix for the fused parameter.
            pieces: List of ``(piece_module_suffix, length)`` pairs.
                ``length=None`` means equal split.
            axis: Split axis (default 0).
            match: Pattern matching mode (default suffix).
        """
        for sub in ("weight", "bias"):
            if sub == "bias" and not self._module_has_bias(vllm):
                continue
            vllm_p = f"{vllm}.{sub}"
            sub_pieces = [(f"{name}.{sub}", length) for name, length in pieces]
            transform = WeightTransform.merged_column(sub_pieces, axis=axis)
            hf_names = tuple(name for name, _ in sub_pieces)
            self.add_rule(vllm_p, hf_names, transform, match=match)
        return self

    def fused_moe(
        self,
        w13: str,
        w2: str,
        gate: str,
        up: str,
        down: str,
        *,
        num_experts: int,
        num_experts_attr: str = "num_experts",
        ep_aware: bool = True,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add fused MoE decompose rules.

        Args:
            w13: vLLM suffix for gate+up fused weight (e.g. "w13_weight")
            w2: vLLM suffix for down weight (e.g. "w2_weight")
            gate, up, down: HF name templates (e.g. "experts.{i}.gate_proj.weight")
            num_experts: Total number of experts
            ep_aware: Whether to use EP rank for local expert slicing
            match: Pattern matching mode
        """
        transform = WeightTransform.fused_moe(
            w13,
            w2,
            gate,
            up,
            down,
            num_experts=num_experts,
            num_experts_attr=num_experts_attr,
            ep_aware=ep_aware,
        )
        self.add_rule(w13, [gate, up], transform, match=match)
        return self.add_rule(w2, down, transform, match=match)

    def fused_moe_from_mapping(
        self,
        mapping: Sequence[tuple[str, str, int, str]],
        *,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add fused MoE rules from vLLM expert_params_mapping format.

        Args:
            mapping: list of (packed_suffix, hf_name, expert_id, shard_id)
                     where shard_id in {"w1", "w2", "w3"}
            match: Pattern matching mode
        """
        transform = WeightTransform.fused_moe_from_mapping(mapping)
        # Collect unique packed suffixes
        seen_packed: dict[str, list] = {}
        for packed_suffix, hf_name, _expert_id, _shard_id in mapping:
            seen_packed.setdefault(packed_suffix, []).append(hf_name)

        for packed_suffix, hf_names in seen_packed.items():
            self.add_rule(packed_suffix, tuple(hf_names), transform, match=match)
        return self

    def split_param(
        self,
        vllm: str,
        pieces: Sequence[tuple[str, int | None]],
        *,
        axis: int = 0,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add a generic split rule for an exact parameter suffix.

        Unlike :meth:`merged`, this helper does not append ``.weight`` /
        ``.bias`` to names. It is intended for parameter-level layouts whose
        output names are not module-style suffixes, for example
        ``in_proj_qkv.weight_q``.
        """
        transform = WeightTransform.split(pieces, axis=axis)
        hf_names = tuple(name for name, _length in pieces)
        return self.add_rule(vllm, hf_names, transform, match=match)

    def transpose(
        self,
        vllm: str,
        hf: str,
        dims: tuple[int, int] = (0, 1),
        *,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add a transpose rule."""
        transform = WeightTransform.transpose(hf, dims=dims)
        return self.add_rule(vllm, hf, transform, match=match)

    def reshape(
        self,
        vllm: str,
        hf: str,
        shape_expr: tuple | str,
        *,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add a reshape rule."""
        transform = WeightTransform.reshape(hf, shape_expr)
        return self.add_rule(vllm, hf, transform, match=match)

    def permute_qk_rotary(
        self,
        patterns: Sequence[str],
        *,
        head_dim: int | None = None,
        head_dim_attr: str = "head_dim",
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add rotary Q/K permutation rules for multiple patterns.

        Args:
            patterns: vLLM parameter patterns (e.g. ["q_proj.weight", "k_proj.weight"])
            head_dim: Fixed head dimension if known
            head_dim_attr: Module attribute name for head dimension
            match: Pattern matching mode
        """
        for pattern in patterns:
            transform = WeightTransform.permute_qk_rotary(pattern, head_dim=head_dim, head_dim_attr=head_dim_attr)
            self.add_rule(pattern, pattern, transform, match=match)
        return self

    def scalar_extract(
        self,
        vllm: str,
        hf: str,
        index: int = 0,
        *,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add a scalar extraction rule."""
        transform = WeightTransform.scalar_extract(hf, index=index)
        return self.add_rule(vllm, hf, transform, match=match)

    def index_select(
        self,
        vllm: str,
        hf: str,
        *,
        indices: Sequence[int] | None = None,
        index_attr: str | None = None,
        dim: int = 0,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add an index-select rule."""
        transform = WeightTransform.index_select(hf, indices=indices, index_attr=index_attr, dim=dim)
        return self.add_rule(vllm, hf, transform, match=match)

    def alias(
        self,
        vllm: str,
        hf: str,
        *,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add an alias rule (same tensor, different HF name)."""
        transform = WeightTransform.alias(hf)
        return self.add_rule(vllm, hf, transform, match=match)

    def derive(
        self,
        vllm: str,
        hf: str,
        *,
        fn: Callable | None = None,
        fn_name: str | None = None,
        match: MatchMode | str = MatchMode.SUFFIX,
    ) -> WeightLayoutBuilder:
        """Add a derive rule (compute derived tensor from source)."""
        transform = WeightTransform.derive(hf, fn=fn, fn_name=fn_name)
        return self.add_rule(vllm, hf, transform, match=match)

    def custom(
        self,
        rule: WeightLayoutRule,
        transform: ModelWeightTransform,
    ) -> WeightLayoutBuilder:
        """Register a custom model-local transform.

        Args:
            rule: The WeightLayoutRule specifying the pattern
            transform: The ModelWeightTransform instance
        """
        # Inject the transform instance into the rule's transform metadata
        new_transform = WeightTransform(
            kind="custom",
            pieces=rule.transform.pieces,
            metadata={**rule.transform.metadata, "transform_instance": transform},
        )
        new_rule = WeightLayoutRule(
            vllm_pattern=rule.vllm_pattern,
            hf_patterns=rule.hf_patterns,
            transform=new_transform,
            match=rule.match,
            module_types=rule.module_types,
            metadata=rule.metadata,
        )
        self._rules.append(new_rule)
        return self

    # ------------------------------------------------------------------
    # Mounting sub-models
    # ------------------------------------------------------------------

    def mount_plan(
        self,
        prefix: str,
        plan: WeightLayoutPlan,
        name_map: ReversibleNameMap | None = None,
    ) -> WeightLayoutBuilder:
        """Mount another WeightLayoutPlan under a module prefix."""
        mount = WeightLayoutMount(prefix=prefix, plan=plan, name_map=name_map)
        self._mounts.append(mount)
        return self

    def mount_module(
        self,
        prefix: str,
        module: nn.Module,
        name_map: ReversibleNameMap | None = None,
    ) -> WeightLayoutBuilder:
        """Mount a sub-module's WeightLayoutPlan.

        The sub-module must implement build_weight_layout().
        Falls back to an empty plan if the module does not.
        """
        from vllm_patches.weight_layouts import register_weight_layouts_for_module

        register_weight_layouts_for_module(type(module).__module__)
        if hasattr(module, "build_weight_layout") and callable(module.build_weight_layout):
            sub_plan = module.build_weight_layout()  # type: ignore[union-attr]
        else:
            sub_plan = WeightLayoutPlan()
        return self.mount_plan(prefix, sub_plan, name_map=name_map)

    def extend(self, plan: WeightLayoutPlan) -> WeightLayoutBuilder:
        """Extend this builder with rules from another plan (for inheritance)."""
        for rule in plan.rules:
            self._rules.append(rule)
        for mount in plan.mounts:
            self._mounts.append(mount)
        if plan.name_map is not None and self._name_map is None:
            self._name_map = plan.name_map
        if plan.sync.exclude_prefixes:
            self._exclude_prefixes.extend(plan.sync.exclude_prefixes)
        if plan.sync.exclude_substrs:
            self._exclude_substrs.extend(plan.sync.exclude_substrs)
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> WeightLayoutPlan:
        """Build the final WeightLayoutPlan."""
        sync = WeightSyncPolicy(
            include_prefixes=tuple(self._include_prefixes),
            exclude_prefixes=tuple(self._exclude_prefixes),
            exclude_substrs=tuple(self._exclude_substrs),
        )
        return WeightLayoutPlan(
            rules=tuple(self._rules),
            mounts=tuple(self._mounts),
            name_map=self._name_map,
            sync=sync,
        )


# ============================================================================
# Auto-inference helper
# ============================================================================


def build_auto_weight_layout(
    model: nn.Module,
    *,
    name_map: ReversibleNameMap | None = None,
    packed_modules: Mapping[str, Any] | None = None,
) -> WeightLayoutPlan:
    """Automatically infer weight layout from model structure.

    Traverses the module tree and generates rules for standard vLLM layer types:
    - QKVParallelLinear → qkv transform
    - MergedColumnParallelLinear → merged_column transform
    - FusedMoE / SharedFusedMoE → fused_moe transform
    - ColumnParallelLinear / RowParallelLinear / ReplicatedLinear → identity

    Sub-modules that implement build_weight_layout() are mounted automatically.
    """
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        has_fused_moe = True
    except ImportError:
        FusedMoE = None  # type: ignore[assignment, misc]
        has_fused_moe = False

    builder = WeightLayoutBuilder(model)
    if name_map:
        builder.name_map(name_map)

    seen_modules: set[str] = set()

    for mod_name, mod in model.named_modules():
        if mod_name in seen_modules or mod is model:
            continue
        seen_modules.add(mod_name)

        # Sub-models that declare their own layout
        if mod is not model and hasattr(mod, "build_weight_layout") and callable(mod.build_weight_layout):
            builder.mount_module(mod_name, mod)
            continue

        # Standard vLLM linear layers
        if isinstance(mod, QKVParallelLinear):
            builder.add_rule(
                mod_name + ".weight",
                (mod_name + ".q_proj.weight", mod_name + ".k_proj.weight", mod_name + ".v_proj.weight"),
                WeightTransform.qkv(
                    mod_name + ".q_proj.weight",
                    mod_name + ".k_proj.weight",
                    mod_name + ".v_proj.weight",
                ),
                match=MatchMode.EXACT,
            )
            if getattr(mod, "bias", None) is not None:
                builder.add_rule(
                    mod_name + ".bias",
                    (mod_name + ".q_proj.bias", mod_name + ".k_proj.bias", mod_name + ".v_proj.bias"),
                    WeightTransform.qkv(
                        mod_name + ".q_proj.bias",
                        mod_name + ".k_proj.bias",
                        mod_name + ".v_proj.bias",
                    ),
                    match=MatchMode.EXACT,
                )
        elif isinstance(mod, MergedColumnParallelLinear):
            if packed_modules and mod_name in packed_modules:
                spec = packed_modules[mod_name]
                builder.add_rule(
                    mod_name + ".weight",
                    tuple(n + ".weight" for n in spec),
                    WeightTransform.merged_column([(n + ".weight", None) for n in spec]),
                    match=MatchMode.EXACT,
                )
                if getattr(mod, "bias", None) is not None:
                    builder.add_rule(
                        mod_name + ".bias",
                        tuple(n + ".bias" for n in spec),
                        WeightTransform.merged_column([(n + ".bias", None) for n in spec]),
                        match=MatchMode.EXACT,
                    )
        elif has_fused_moe and isinstance(mod, FusedMoE):
            # FusedMoE requires explicit expert mapping - skip auto for now
            pass

    return builder.build()
