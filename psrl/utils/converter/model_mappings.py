from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch.nn import Parameter

from psrl.utils.nixl.nixl_spec import NIXLSharding

_MISSING = object()


def make_slice_parameter(slice_data: torch.Tensor, param: Parameter) -> Parameter:
    """
    Make a slice parameter from a slice data tensor and a parameter.
    This is inplace operation.
    """
    new_param = Parameter(slice_data, requires_grad=param.requires_grad)
    for attr_name, attr_value in param.__dict__.items():
        if attr_name not in ["data", "_grad", "_grad_fn", "_backward_hooks"]:
            setattr(new_param, attr_name, attr_value)
    return new_param


# Suffix patterns that identify Q, K, V projection weights in HF/FSDP state dicts.
_QKV_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_proj.bias",
    "k_proj.bias",
    "v_proj.bias",
)

# Suffix patterns that identify Q projection weights in HF/FSDP state dicts.
_Q_WEIGHT_SUFFIXES = (
    "q_proj.weight",
    "q_proj.bias",
)


def is_qkv_weight(param_name: str) -> bool:
    """
    Return True if param_name ends with a Q/K/V projection weight suffix.

    Args:
        param_name (str): Fully-qualified parameter name from a state dict.

    Returns:
        bool: True if the parameter is a Q, K, or V projection weight or bias.
    """
    return any(param_name.endswith(suffix) for suffix in _QKV_WEIGHT_SUFFIXES)


def is_q_weight(param_name: str) -> bool:
    """
    Return True if param_name ends with a Q projection weight suffix.

    Args:
        param_name (str): Fully-qualified parameter name from a state dict.

    Returns:
        bool: True if the parameter is a Q projection weight or bias.
    """
    return any(param_name.endswith(suffix) for suffix in _Q_WEIGHT_SUFFIXES)


def reshape_qkv_to_3d(
    param: Parameter,
    num_heads_local: int,
    num_kv_heads_local: int,
    head_size: int,
) -> Parameter:
    """
    Reshape a 2D Q/K/V weight tensor into the 3D group-interleaved layout used by
    slice_qkv_proj_megatron, so that all three sides (Megatron/PS/vLLM) register
    the same shape with NIXL and can_xfer_to passes.

    Args:
        param (Parameter): 2D tensor of shape (rows, hidden) where rows is either
            num_heads_local * head_size (for Q) or num_kv_heads_local * head_size (for K/V).
        num_heads_local (int): Number of Q heads on this TP rank.
        num_kv_heads_local (int): Number of KV heads on this TP rank.
        head_size (int): Dimension of each head.

    Returns:
        Parameter: View of shape (num_groups_local, heads_per_group * head_size, hidden)
            where num_groups_local = gcd(num_heads_local, num_kv_heads_local).
    """
    import math

    num_groups_local = math.gcd(num_heads_local, num_kv_heads_local)
    rows = param.shape[0]
    hidden = param.shape[1] if param.ndim == 2 else 1
    assert rows % num_groups_local == 0, f"rows={rows} is not divisible by num_groups_local={num_groups_local}."
    new_shape = (num_groups_local, rows // num_groups_local, hidden)
    # NOTE(lhy): The `param.data` storage comes from contiguous HF state or a vLLM
    # fused split, so reshaping returns a view rather than a copy.
    reshaped_data = param.data.reshape(new_shape)
    return make_slice_parameter(reshaped_data, param)


def reshape_visual_block_qkv(param, vision_head_size: int | None = None):
    """
    Reshape a visual QKV tensor to PSRL's flattened-head transfer layout.

    The canonical layout is `[3 * num_heads_local, head_size, ...]`. Flattening
    the Q/K/V component and head axes lets both of these native layouts describe
    zero-copy views of the same global tensor:

    - FSDP's contiguous flat-row shards.
    - Tensor parallelism's non-contiguous Q/K/V head groups.

    Their different ownership is represented by `NIXLSharding`, rather than by
    forcing one backend to materialize or communicate a reordered tensor.
    """
    rows = param.shape[0]
    assert rows % 3 == 0, f"Expected rows={rows} to be divisible by 3 for visual block qkv weights."
    if vision_head_size is None:
        raise ValueError("vision_head_size is required for visual QKV canonicalization.")
    if param.ndim >= 2 and param.shape[1] == vision_head_size:
        # Megatron's concatenated Q/K/V view is already in flattened-head form.
        return make_slice_parameter(param.data, param)
    assert rows % vision_head_size == 0, (
        f"Expected rows={rows} to be divisible by vision_head_size={vision_head_size}."
    )
    if len(param.shape) == 1:
        reshaped_data = param.data.reshape(rows // vision_head_size, vision_head_size, 1)
    else:
        reshaped_data = param.data.reshape(rows // vision_head_size, vision_head_size, *param.shape[1:])
    return make_slice_parameter(reshaped_data, param)


def make_visual_qkv_tp_sharding(
    tp_size: int,
    tp_rank: int,
) -> NIXLSharding:
    """
    Describe TP ownership in the flattened-head visual QKV layout.

    A TP rank owns one head range from each of Q, K, and V. In a mesh of
    `3 * tp_size` equal head shards, rank `r` therefore owns shard indices
    `r`, `tp_size + r`, and `2 * tp_size + r`.
    """
    assert tp_size >= 1, f"tp_size must be positive, got {tp_size}."
    assert 0 <= tp_rank < tp_size, f"tp_rank must be in [0, {tp_size}), got {tp_rank}."
    return NIXLSharding(
        shard_mesh=OrderedDict([(0, 3 * tp_size)]),
        shard_indices=[
            (tp_rank,),
            (tp_size + tp_rank,),
            (2 * tp_size + tp_rank,),
        ],
    )


def reshape_q_to_5d(
    param: Parameter,
    num_heads_local: int,
    head_size: int,
) -> Parameter:
    """
    Reshape a 3D Q weight tensor into the 5D group-interleaved layout used by
    slice_qkv_proj_megatron, so that all three sides (Megatron/PS/vLLM) register
    the same shape with NIXL and can_xfer_to passes.

    Args:
        param (Parameter): 3D tensor of shape (num_groups_local, heads_per_group * head_size, hidden)
        num_heads_local (int): Number of Q heads on this TP rank.
        head_size (int): Dimension of each head.

    Returns:
        Parameter: View of shape (num_groups_local, q_heads_per_group, 2, head_size, hidden)
            where num_groups_local = gcd(num_heads_local, num_kv_heads_local).
    """
    num_groups_local, rows, hidden = param.shape
    assert rows == num_heads_local // num_groups_local * head_size, (
        f"Expected rows={num_heads_local * head_size} but got {rows}."
    )
    q_num_heads_per_group = num_heads_local // num_groups_local // 2
    new_shape = (num_groups_local, q_num_heads_per_group, 2, head_size, hidden)
    # NOTE(lhy): The `param.data` storage comes from contiguous HF state or a vLLM
    # fused split, so reshaping returns a view rather than a copy.
    reshaped_data = param.data.reshape(new_shape)
    return make_slice_parameter(reshaped_data, param)


def slice_gate_up_proj(
    fused_param: Parameter,
    output_sizes: list[int],
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
    """
    Split a fused `gate_up_proj` parameter into gate and up projection views.

    Args:
        fused_param (Parameter): Fused parameter to split.
        output_sizes (list[int]): Global gate and up projection sizes.
        tp_size (int): Tensor parallel size.
        output_dim (int): Dimension along which to split.

    Returns:
        list[Parameter]: Gate and up projection views.
    """
    assert len(output_sizes) == 2, "Expected exactly two shards for gate_up_proj."
    assert all([output_size % tp_size == 0 for output_size in output_sizes]), (
        "Output sizes must be divisible by tensor parallel size."
    )
    gate_size, up_size = [output_size // tp_size for output_size in output_sizes]

    # Preserve storage sharing when splitting the fused parameter.
    assert fused_param.data.shape[output_dim] == (gate_size + up_size), (
        f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} "
        f"must match the sum of gate and up sizes {[gate_size, up_size]}."
    )
    gate_data = fused_param.data.narrow(output_dim, 0, gate_size)
    up_data = fused_param.data.narrow(output_dim, gate_size, up_size)
    return [
        make_slice_parameter(gate_data, fused_param),
        make_slice_parameter(up_data, fused_param),
    ]


@dataclass(frozen=True)
class QKVTPLayout:
    """
    Describe Q/KV head placement across a tensor-parallel group.

    Attributes:
        tp_size (int): Tensor-parallel world size.
        num_heads_local (int): Number of query heads stored by each TP rank.
        num_kv_heads_local (int): Number of key/value heads stored by each TP rank.
        num_kv_head_replicas (int): Number of TP ranks sharing each KV shard.
    """

    tp_size: int
    num_heads_local: int
    num_kv_heads_local: int
    num_kv_head_replicas: int

    @property
    def num_kv_shards(self) -> int:
        """
        Return the number of distinct KV shards across all TP ranks.
        """
        return self.tp_size // self.num_kv_head_replicas

    def get_kv_shard_rank(self, tp_rank: int) -> int:
        """
        Return the distinct KV shard index owned by a TP rank.

        Args:
            tp_rank (int): Tensor-parallel rank.

        Returns:
            int: KV shard index shared by this TP rank and its replicas.
        """
        assert 0 <= tp_rank < self.tp_size, (
            f"Tensor parallel rank must be in [0, {self.tp_size}), got tp_rank = {tp_rank}."
        )
        return tp_rank // self.num_kv_head_replicas


def get_qkv_tp_layout(
    num_heads: int,
    num_kv_heads: int,
    tp_size: int,
) -> QKVTPLayout:
    """
    Build the complete Q/KV head layout for tensor parallelism.

    vLLM partitions KV heads when there are at least as many KV heads as TP
    ranks. When TP is larger, each KV head is replicated across a contiguous
    group of TP ranks.

    Args:
        num_heads (int): Global number of query heads.
        num_kv_heads (int): Global number of key/value heads.
        tp_size (int): Tensor-parallel world size.

    Returns:
        QKVTPLayout: Local head counts and KV shard replication metadata.
    """
    assert tp_size > 0, f"Tensor parallel size must be positive, got tp_size = {tp_size}."
    assert num_kv_heads > 0, f"Number of KV heads must be positive, got num_kv_heads = {num_kv_heads}."
    assert num_heads % tp_size == 0, (
        "Number of heads must be divisible by tensor parallel size, "
        f"but got num_heads = {num_heads} and tp_size = {tp_size}."
    )

    if num_kv_heads >= tp_size:
        assert num_kv_heads % tp_size == 0, (
            "Number of KV heads must be divisible by tensor parallel size when KV heads are partitioned, "
            f"but got num_kv_heads = {num_kv_heads} and tp_size = {tp_size}."
        )
        num_kv_heads_local = num_kv_heads // tp_size
        num_kv_head_replicas = 1
    else:
        assert tp_size % num_kv_heads == 0, (
            "Tensor parallel size must be divisible by the number of KV heads when KV heads are replicated, "
            f"but got tp_size = {tp_size} and num_kv_heads = {num_kv_heads}."
        )
        num_kv_heads_local = 1
        num_kv_head_replicas = tp_size // num_kv_heads

    return QKVTPLayout(
        tp_size=tp_size,
        num_heads_local=num_heads // tp_size,
        num_kv_heads_local=num_kv_heads_local,
        num_kv_head_replicas=num_kv_head_replicas,
    )


def slice_qkv_proj(
    fused_param: Parameter,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
    """
    Split a fused qkv_proj parameter into three 2D shards:
      - q_param: query heads
      - k_param: key heads
      - v_param: value heads

    Args:
        fused_param (Parameter): The fused parameter of shape (total_qkv, hidden).
        num_heads (int): Number of query heads (global, across all TP ranks).
        num_kv_heads (int): Number of key/value heads (global, across all TP ranks).
        head_size (int): Dimensionality of each head.
        tp_size (int): Tensor parallel size (default is 1).
        output_dim (int): Dimension along which to split (default is 0).

    Returns:
        list[Parameter]: A list of three 2D parameters sharing storage with fused_param:
            [
              q_param, # shape (num_heads // tp_size * head_size, hidden)
              k_param, # shape (num_kv_heads_local * head_size, hidden)
              v_param, # shape (num_kv_heads_local * head_size, hidden)
            ]
            When `num_kv_heads < tp_size`, `num_kv_heads_local` is 1 and the
            same KV shard is replicated across multiple TP ranks.
    """
    layout = get_qkv_tp_layout(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        tp_size=tp_size,
    )
    q_len = layout.num_heads_local * head_size
    k_len = layout.num_kv_heads_local * head_size
    v_len = layout.num_kv_heads_local * head_size

    assert fused_param.data.shape[output_dim] == (q_len + k_len + v_len), (
        f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} "
        f"must match the sum of qkv lengths {[q_len, k_len, v_len]}."
    )
    offset_and_sizes = [
        (0, q_len),
        (q_len, k_len),
        (q_len + k_len, v_len),
    ]

    qkv_params: list[Parameter] = []
    for offset, size in offset_and_sizes:
        data = fused_param.data.narrow(output_dim, offset, size)
        qkv_params.append(make_slice_parameter(data, fused_param))
    return qkv_params


def slice_qkv_proj_megatron(
    fused_param: Parameter,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    tp_size: int = 1,
    attn_output_gate: bool = False,
    output_dim: int = 1,
) -> list[Parameter]:
    """
    Split a fused qkv_proj parameter into three shards according to Megatron-style:
      - q_param: query heads
      - k_param: key heads
      - v_param: value heads

    Args
    ----------
    fused_param : Parameter
        The fused parameter of shape [..., total_qkv, ...].
    num_heads : int
        Number of query heads.
    num_kv_heads : int
        Number of key/value heads.
    head_size : int
        Dimensionality of each head.
    tp_size : int, optional
        Tensor parallel size (default is 1).
    attn_output_gate : bool, optional
        Whether this is an attention output gate parameter,
        which requires special handling for Q weights in Qwen3.5 (default is False).
    output_dim : int, optional
        Dimension along which to split (default is 0).

    Returns
    -------
    List[Parameter]
        A list of three parameters, each a 3D non-contiguous view sharing storage with fused_param:
        [
          q_param, # view of shape (num_groups, q_heads_per_group * head_size, hidden)
          k_param, # view of shape (num_groups, kv_heads_per_group * head_size, hidden)
          v_param, # view of shape (num_groups, kv_heads_per_group * head_size, hidden)
        ]
        where num_groups = gcd(num_heads, num_kv_heads) // tp_size.
        These views share storage with fused_param, so NIXL pull/push correctly operate
        on the actual fused linear_qkv.weight used by Megatron forward.
    """
    import math

    num_split_heads = math.gcd(num_heads, num_kv_heads)
    assert num_split_heads % tp_size == 0, (
        "Number of split heads must be divisible by tensor parallel size, "
        f"but got num_split_heads = {num_split_heads} and tp_size = {tp_size}"
    )

    q_len = head_size * num_heads // num_split_heads
    k_len = head_size * num_kv_heads // num_split_heads
    v_len = head_size * num_kv_heads // num_split_heads

    fused_param = fused_param.reshape(num_split_heads // tp_size, q_len + k_len + v_len, -1)
    assert fused_param.data.shape[output_dim] == (q_len + k_len + v_len), (
        f"Dim {output_dim} of fused parameter shape {fused_param.data.shape} must "
        f"match the sum of qkv lengths {[q_len, k_len, v_len]}"
    )
    offset_and_sizes = [
        (0, q_len),
        (q_len, k_len),
        (q_len + k_len, v_len),
    ]

    qkv_params: list[Parameter] = []
    for i, (offset, size) in enumerate(offset_and_sizes):
        # NOTE(lhy): Keep the 3D noncontiguous view because flattening copies storage.
        # NIXL updates must reach the original Megatron `linear_qkv.weight`.
        data = fused_param.data.narrow(output_dim, offset, size)
        if attn_output_gate and i == 0:
            # NOTE(zym): Qwen3.5 Megatron query weights require a gated layout.
            q_num_heads_per_group = num_heads // num_split_heads // 2
            data = data.view(num_split_heads // tp_size, 2, q_num_heads_per_group, head_size, -1).transpose(1, 2)
        qkv_params.append(make_slice_parameter(data, fused_param))

    return qkv_params


def slice_attn_conv1d(
    fused_param: Parameter,
    num_k_heads: int,
    num_v_heads: int,
    k_head_size: int,
    v_head_size: int,
    tp_size: int = 1,
    output_dim: int = 0,
):
    """
    For Qwen3.5, slice attn_con1d weights to support correct tp sharding.
    """
    k_dim = num_k_heads * k_head_size // tp_size
    v_dim = num_v_heads * v_head_size // tp_size
    assert fused_param.data.shape[output_dim] == (k_dim * 2 + v_dim), (
        f"Dim {output_dim} of fused parameter shape "
        f"{fused_param.data.shape} must match the sum "
        f"of k and v dims {[k_dim, v_dim]}"
    )
    offset_and_sizes = [
        (0, k_dim),
        (k_dim, k_dim),
        (k_dim * 2, v_dim),
    ]
    params: list[Parameter] = []
    for offset, size in offset_and_sizes:
        data = fused_param.data.narrow(output_dim, offset, size)
        params.append(make_slice_parameter(data, fused_param))
    return params


def slice_fused_moe_w13_weight(
    fused_param: Parameter,
) -> list[Parameter]:
    expert_params: list[Parameter] = []
    expert_num = fused_param.shape[0]
    shard_size = fused_param.shape[1] // 2
    for expert_id in range(expert_num):
        expert = fused_param.data[expert_id]
        gate = expert.narrow(0, 0, shard_size)
        up = expert.narrow(0, shard_size, shard_size)
        expert_params.append(make_slice_parameter(gate, fused_param))
        expert_params.append(make_slice_parameter(up, fused_param))
    return expert_params


def slice_fused_moe_w2_weight(
    fused_param: Parameter,
) -> list[Parameter]:
    expert_params: list[Parameter] = []
    expert_num = fused_param.shape[0]
    for expert_id in range(expert_num):
        down = fused_param.data[expert_id]
        expert_params.append(make_slice_parameter(down, fused_param))
    return expert_params


def get_fused_moe_expert_prefix(param_name: str, projection: str) -> str | None:
    """Return the canonical `...mlp.experts` prefix for a fused MoE parameter."""
    for suffix in (f".{projection}.weight", f".{projection}"):
        if param_name.endswith(suffix):
            prefix = param_name[: -len(suffix)]
            if prefix.endswith(".mlp.experts"):
                return prefix
    return None


def slice_in_proj_qkvz(
    fused_param: Parameter,
    key_dim: int,
    value_dim: int,
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
    """
    Split Qwen3.5 GDN `in_proj_qkvz` into `in_proj_qkv` and `in_proj_z`.

    The checkpoint separates the first three projection shards from the final
    value projection shard used by vLLM's fused parameter.

    Args:
        fused_param (Parameter): Fused parameter tensor.
        key_dim (int): Global key projection dimension (linear_num_key_heads * linear_key_head_dim).
        value_dim (int): Global value projection dimension (linear_num_value_heads * linear_value_head_dim).
        tp_size (int): Tensor-parallel size.
        output_dim (int): Dimension along which to split.

    Returns:
        list[Parameter]: Projection views that share storage with `fused_param`.
    """
    assert tp_size >= 1, f"tp_size must be >= 1, got: {tp_size!r}."
    qkv_size = 2 * key_dim + value_dim
    z_size = value_dim
    assert qkv_size % tp_size == 0, (
        f"in_proj_qkv size must be divisible by tensor parallel size, got qkv_size={qkv_size} and tp_size={tp_size}."
    )
    assert z_size % tp_size == 0, (
        f"in_proj_z size must be divisible by tensor parallel size, got z_size={z_size} and tp_size={tp_size}."
    )
    qkv_size_local = qkv_size // tp_size
    z_size_local = z_size // tp_size
    expected = qkv_size_local + z_size_local
    assert fused_param.data.shape[output_dim] == expected, (
        "Fused in_proj_qkvz has unexpected size on output_dim. "
        f"Expected {expected}, got {fused_param.data.shape[output_dim]}."
    )

    qkv_data = fused_param.data.narrow(output_dim, 0, qkv_size_local)
    z_data = fused_param.data.narrow(output_dim, qkv_size_local, z_size_local)
    return [
        make_slice_parameter(qkv_data, fused_param),
        make_slice_parameter(z_data, fused_param),
    ]


def slice_in_proj_ba(
    fused_param: Parameter,
    num_v_heads: int,
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
    """
    Split Qwen3.5 GDN fused in_proj_ba into (in_proj_b, in_proj_a).

    Args:
        fused_param (Parameter): Fused parameter tensor.
        num_v_heads (int): Global number of value heads for linear attention.
        tp_size (int): Tensor-parallel size.
        output_dim (int): Dimension along which to split.

    Returns:
        list[Parameter]: [in_proj_b, in_proj_a] views sharing storage with fused_param.
    """
    assert tp_size >= 1, f"tp_size must be >= 1, got: {tp_size!r}."
    assert num_v_heads % tp_size == 0, (
        f"num_v_heads must be divisible by tensor parallel size, got num_v_heads={num_v_heads} and tp_size={tp_size}."
    )
    size_local = num_v_heads // tp_size
    expected = 2 * size_local
    assert fused_param.data.shape[output_dim] == expected, (
        "Fused in_proj_ba has unexpected size on output_dim. "
        f"Expected {expected}, got {fused_param.data.shape[output_dim]}."
    )

    b_data = fused_param.data.narrow(output_dim, 0, size_local)
    a_data = fused_param.data.narrow(output_dim, size_local, size_local)
    return [
        make_slice_parameter(b_data, fused_param),
        make_slice_parameter(a_data, fused_param),
    ]


def slice_qwen3_5_in_proj(
    fused_param: Parameter,
    key_dim: int,
    value_dim: int,
    num_v_heads: int,
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
    """
    Split Megatron Qwen3.5 linear attention in_proj into 4 HF-like parameters.

    Megatron-Bridge maps Megatron's single in_proj.weight into:
      - linear_attn.in_proj_qkv.weight
      - linear_attn.in_proj_z.weight
      - linear_attn.in_proj_b.weight
      - linear_attn.in_proj_a.weight

    Args:
        fused_param (Parameter): Fused parameter tensor.
        key_dim (int): Global key projection dimension.
        value_dim (int): Global value projection dimension.
        num_v_heads (int): Global number of value heads.
        tp_size (int): Tensor-parallel size.
        output_dim (int): Dimension along which to split.

    Returns:
        list[Parameter]: [in_proj_qkv, in_proj_z, in_proj_b, in_proj_a] views.
    """
    assert tp_size >= 1, f"tp_size must be >= 1, got: {tp_size!r}."
    qkv_size = 2 * key_dim + value_dim
    z_size = value_dim
    b_size = num_v_heads
    a_size = num_v_heads
    for name, size in (
        ("in_proj_qkv", qkv_size),
        ("in_proj_z", z_size),
        ("in_proj_b", b_size),
        ("in_proj_a", a_size),
    ):
        assert size % tp_size == 0, (
            f"{name} size must be divisible by tensor parallel size, got size={size} and tp_size={tp_size}."
        )

    qkv_local = qkv_size // tp_size
    z_local = z_size // tp_size
    b_local = b_size // tp_size
    a_local = a_size // tp_size
    expected = qkv_local + z_local + b_local + a_local
    assert fused_param.data.shape[output_dim] == expected, (
        "Fused in_proj has unexpected size on output_dim. "
        f"Expected {expected}, got {fused_param.data.shape[output_dim]}."
    )

    offset = 0
    qkv_data = fused_param.data.narrow(output_dim, offset, qkv_local)
    offset += qkv_local
    z_data = fused_param.data.narrow(output_dim, offset, z_local)
    offset += z_local
    b_data = fused_param.data.narrow(output_dim, offset, b_local)
    offset += b_local
    a_data = fused_param.data.narrow(output_dim, offset, a_local)
    return [
        make_slice_parameter(qkv_data, fused_param),
        make_slice_parameter(z_data, fused_param),
        make_slice_parameter(b_data, fused_param),
        make_slice_parameter(a_data, fused_param),
    ]


def slice_qwen3_5_in_proj_qkv(
    fused_param: Parameter,
    key_dim: int,
    value_dim: int,
    tp_size: int = 1,
    output_dim: int = 0,
) -> list[Parameter]:
    """
    For Qwen3.5, slice in_proj_qkv to support correct tp sharding.
    """
    q_local = key_dim // tp_size
    k_local = key_dim // tp_size
    v_local = value_dim // tp_size
    expected = q_local + k_local + v_local
    assert fused_param.data.shape[output_dim] == expected, (
        "Fused in_proj_qkv has unexpected size on output_dim. "
        f"Expected {expected}, got {fused_param.data.shape[output_dim]}."
    )
    offset = 0
    q_data = fused_param.data.narrow(output_dim, offset, q_local)
    offset += q_local
    k_data = fused_param.data.narrow(output_dim, offset, k_local)
    offset += k_local
    v_data = fused_param.data.narrow(output_dim, offset, v_local)
    return [
        make_slice_parameter(q_data, fused_param),
        make_slice_parameter(k_data, fused_param),
        make_slice_parameter(v_data, fused_param),
    ]


class MappingType(Enum):
    """Enum for mapping prototypes."""

    DIRECT = "direct"
    QKV_SPLIT = "qkv_split"
    GATE_UP_PROJ_SPLIT = "gate_up_proj_split"
    IN_PROJ_QKVZ_SPLIT = "in_proj_qkvz_split"
    IN_PROJ_BA_SPLIT = "in_proj_ba_split"
    FUSED_MOE_W13_SPLIT = "fused_moe_w13_split"
    FUSED_MOE_W2_SPLIT = "fused_moe_w2_split"


class ParameterMapping(ABC):
    """Abstract base class for parameter mappings."""

    def __init__(self, config):
        """
        Args:
            config: HuggingFace model config (e.g. from AutoConfig.from_pretrained).
        """
        self.config = config

    @abstractmethod
    def get_mappings(self) -> list[tuple[str, str, MappingType, int]]:
        """Return list of (original_param_name, hf_param_name, mapping_prototype, shard_id) mappings."""
        pass

    def get_model_info(self) -> dict[str, Any]:
        """
        Return model-specific information needed for parameter splitting.

        Subclasses that need extra fields (e.g. num_experts) should call
        super().get_model_info() and extend the returned dict.
        """
        cfg = self.config
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is None:
            text_cfg = cfg

        def _get_attr(obj, name: str, default=_MISSING):
            if isinstance(obj, dict):
                if default is _MISSING:
                    return obj[name]
                return obj.get(name, default)
            if default is _MISSING:
                return getattr(obj, name)
            return getattr(obj, name, default)

        def _is_int(value) -> bool:
            return isinstance(value, int) and not isinstance(value, bool)

        num_heads = _get_attr(text_cfg, "num_attention_heads")
        hidden_size = _get_attr(text_cfg, "hidden_size")
        intermediate_size = _get_attr(text_cfg, "intermediate_size", None)
        num_kv_heads = _get_attr(text_cfg, "num_key_value_heads", num_heads)

        attn_output_gate = _get_attr(text_cfg, "attn_output_gate", False)
        if attn_output_gate:
            num_heads *= 2

        qk_nope_head_dim = _get_attr(text_cfg, "qk_nope_head_dim", None)
        qk_rope_head_dim = _get_attr(text_cfg, "qk_rope_head_dim", None)
        v_head_dim = _get_attr(text_cfg, "v_head_dim", None)
        kv_lora_rank = _get_attr(text_cfg, "kv_lora_rank", None)
        q_lora_rank = _get_attr(text_cfg, "q_lora_rank", None)

        uses_mla_attention = (
            _is_int(qk_nope_head_dim)
            and _is_int(qk_rope_head_dim)
            and _is_int(kv_lora_rank)
            and (qk_nope_head_dim + qk_rope_head_dim) > 0
        )

        head_size = _get_attr(text_cfg, "head_dim", None)
        if head_size is None and uses_mla_attention:
            head_size = qk_nope_head_dim + qk_rope_head_dim
        if head_size is None:
            head_size = hidden_size // num_heads

        info: dict[str, Any] = {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "intermediate_size": intermediate_size,
            "attn_output_gate": attn_output_gate,
            "uses_mla_attention": uses_mla_attention,
        }

        if uses_mla_attention:
            info.update(
                {
                    "qk_nope_head_dim": qk_nope_head_dim,
                    "qk_rope_head_dim": qk_rope_head_dim,
                    "qk_head_dim": qk_nope_head_dim + qk_rope_head_dim,
                    "v_head_dim": v_head_dim,
                    "kv_lora_rank": kv_lora_rank,
                    "q_lora_rank": q_lora_rank,
                }
            )

        num_experts = _get_attr(text_cfg, "num_experts", None)
        if num_experts is None:
            num_experts = _get_attr(text_cfg, "n_routed_experts", None)
        moe_intermediate_size = _get_attr(text_cfg, "moe_intermediate_size", None)
        n_shared_experts = _get_attr(text_cfg, "n_shared_experts", None)
        shared_expert_intermediate_size = _get_attr(text_cfg, "shared_expert_intermediate_size", None)
        if num_experts is not None:
            info["num_experts"] = num_experts
        if moe_intermediate_size is not None:
            info["moe_intermediate_size"] = moe_intermediate_size
        if n_shared_experts is not None:
            info["n_shared_experts"] = n_shared_experts
        if shared_expert_intermediate_size is not None:
            info["shared_expert_intermediate_size"] = shared_expert_intermediate_size
        elif moe_intermediate_size is not None and n_shared_experts is not None:
            info["shared_expert_intermediate_size"] = moe_intermediate_size * n_shared_experts

        linear_num_key_heads = _get_attr(text_cfg, "linear_num_key_heads", None)
        linear_key_head_dim = _get_attr(text_cfg, "linear_key_head_dim", None)
        linear_num_value_heads = _get_attr(text_cfg, "linear_num_value_heads", None)
        linear_value_head_dim = _get_attr(text_cfg, "linear_value_head_dim", None)
        if all(
            x is not None
            for x in (
                linear_num_key_heads,
                linear_key_head_dim,
                linear_num_value_heads,
                linear_value_head_dim,
            )
        ):
            info.update(
                {
                    "linear_num_key_heads": linear_num_key_heads,
                    "linear_key_head_dim": linear_key_head_dim,
                    "linear_num_value_heads": linear_num_value_heads,
                    "linear_value_head_dim": linear_value_head_dim,
                    "linear_key_dim": linear_num_key_heads * linear_key_head_dim,
                    "linear_value_dim": linear_num_value_heads * linear_value_head_dim,
                }
            )

        vision_cfg = _get_attr(cfg, "vision_config", None)
        if vision_cfg is not None:
            vision_num_heads = _get_attr(vision_cfg, "num_heads", None)
            vision_hidden_size = _get_attr(vision_cfg, "hidden_size", None)
            vision_head_size = _get_attr(vision_cfg, "head_dim", None)
            if vision_head_size is None and _is_int(vision_num_heads) and _is_int(vision_hidden_size):
                vision_head_size = vision_hidden_size // vision_num_heads
            if _is_int(vision_num_heads):
                info["vision_num_heads"] = vision_num_heads
                info["vision_num_kv_heads"] = _get_attr(vision_cfg, "num_key_value_heads", vision_num_heads)
            if _is_int(vision_head_size):
                info["vision_head_size"] = vision_head_size

        vision_cfg = _get_attr(cfg, "vision_config", None)
        if vision_cfg is not None:
            vision_num_heads = _get_attr(vision_cfg, "num_heads", None)
            vision_hidden_size = _get_attr(vision_cfg, "hidden_size", None)
            vision_head_size = _get_attr(vision_cfg, "head_dim", None)
            if vision_head_size is None and _is_int(vision_num_heads) and _is_int(vision_hidden_size):
                vision_head_size = vision_hidden_size // vision_num_heads
            if _is_int(vision_num_heads):
                info["vision_num_heads"] = vision_num_heads
                info["vision_num_kv_heads"] = _get_attr(vision_cfg, "num_key_value_heads", vision_num_heads)
            if _is_int(vision_head_size):
                info["vision_head_size"] = vision_head_size

        return info

    def get_external_fp32_param_patterns(self) -> tuple[str, ...]:
        """Return unified key substrings for params that must be exposed to PS/vLLM as fp32."""
        return ("A_log",)


class ModelRegistry:
    """Registry for different model parameter mappings based on model class types."""

    def __init__(self):
        self._mappings: dict[type | str, type[ParameterMapping]] = {}
        self._reverse_mappings: dict[type[ParameterMapping], list[type | str]] = {}

    def register(
        self,
        model_classes: type | str | list[type | str],
        mapping_class: type[ParameterMapping],
    ):
        """Register a parameter mapping for one or more model classes or class names."""
        if not isinstance(model_classes, list):
            model_classes = [model_classes]

        # Register the mapping for each model class
        for model_class in model_classes:
            self._mappings[model_class] = mapping_class

        # Store reverse mapping for cleanup
        if mapping_class not in self._reverse_mappings:
            self._reverse_mappings[mapping_class] = []
        self._reverse_mappings[mapping_class].extend(model_classes)

    def create_mapping(self, model_class: type | str, config) -> ParameterMapping:
        """
        Create a parameter mapping instance for a model class or class name.

        Args:
            model_class (type | str): The model class or its string name.
            config: HuggingFace model config (e.g. from AutoConfig.from_pretrained).

        Returns:
            ParameterMapping: An instantiated parameter mapping for the given model.
        """
        if model_class not in self._mappings:
            supported_classes = list(self._mappings.keys())
            supported_names = [str(c) for c in supported_classes]
            raise ValueError(f"Unsupported model class: {model_class}. Supported classes: {supported_names}")

        mapping_class = self._mappings[model_class]
        return mapping_class(config)

    def get_supported_models(self) -> list[type | str]:
        """Get list of supported model classes."""
        return list(self._mappings.keys())

    def unregister_mapping(self, mapping_class: type[ParameterMapping]):
        """Unregister a parameter mapping and all its associated model classes."""
        if mapping_class in self._reverse_mappings:
            for model_class in self._reverse_mappings[mapping_class]:
                if model_class in self._mappings:
                    del self._mappings[model_class]
            del self._reverse_mappings[mapping_class]


def register_model(model_classes: type | str | list[type | str]):
    """Decorator to register a model parameter mapping for one or more model classes."""

    def decorator(mapping_class: type[ParameterMapping]):
        model_registry.register(model_classes, mapping_class)
        return mapping_class

    return decorator


# Factory function for creating parameter mappings
def create_parameter_mapping(model_class: type | str, config) -> ParameterMapping:
    """
    Create parameter mapping for a specific model class or class name.

    Args:
        model_class (type | str): The model class or its string name.
        config: HuggingFace model config (e.g. from AutoConfig.from_pretrained).

    Returns:
        ParameterMapping: An instantiated parameter mapping for the given model.
    """
    return model_registry.create_mapping(model_class, config)


# Global registry instance
model_registry = ModelRegistry()
