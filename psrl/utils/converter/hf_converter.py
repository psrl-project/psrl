from collections import OrderedDict

import torch

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    get_fused_moe_expert_prefix,
    make_visual_qkv_tp_sharding,
    reshape_visual_block_qkv,
    slice_attn_conv1d,
    slice_qwen3_5_in_proj_qkv,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding


class HFConverter(BaseConverter):
    """Convert HuggingFace model to a unified format and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping):
        """
        Args:
            parameter_mapping (ParameterMapping): Parameter mapping instance carrying
                model_info (num_heads, num_kv_heads, head_size). Use HFParameterMapping
                to enable Q/K/V 3D reshaping for NIXL compatibility with Megatron workers.
        """
        super().__init__(parameter_mapping)

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert HuggingFace model to unified state dict and sharding info.

        Args:
            model: The HuggingFace model instance.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
                (converted_state_dict, sharding_dict).
        """
        hf_state_dict = model.state_dict()

        converted_state_dict = {}
        sharding_dict = {}
        for param_name, param in hf_state_dict.items():
            assert isinstance(param, torch.Tensor), f"Expected Tensor for {param_name}, got {type(param)}."
            sharding = NIXLSharding.default()
            # NOTE(lhy): PS holds full (non-TP-sharded) weights, so num_heads_local = global num_heads.
            # maybe_reshape_qkv_to_3d is a no-op when model_info does not contain num_heads.
            param, sharding = self.maybe_reshape_qkv_to_3d(param_name, param, sharding)
            converted_params = convert_hf_local_shard(
                self.model_info,
                param_name,
                param,
                sharding,
            )
            for new_param_name, (new_param, new_sharding) in converted_params.items():
                converted_state_dict[new_param_name] = new_param
                sharding_dict[new_param_name] = new_sharding
        return converted_state_dict, sharding_dict


def maybe_convert_to_smaller_parts(model_info, param_name, param):
    """
    Convert an unsharded HF checkpoint tensor to canonical transfer tensors.
    """
    if "linear_attn.conv1d.weight" in param_name:
        new_param_names = [param_name + "_q", param_name + "_k", param_name + "_v"]
        new_params = slice_attn_conv1d(
            fused_param=param,
            num_k_heads=model_info["linear_num_key_heads"],
            num_v_heads=model_info["linear_num_value_heads"],
            k_head_size=model_info["linear_key_head_dim"],
            v_head_size=model_info["linear_value_head_dim"],
            tp_size=1,
        )
        return dict(zip(new_param_names, new_params))
    if "linear_attn.in_proj_qkv.weight" in param_name:
        new_param_names = [param_name + "_q", param_name + "_k", param_name + "_v"]
        new_params = slice_qwen3_5_in_proj_qkv(
            fused_param=param,
            key_dim=model_info["linear_key_dim"],
            value_dim=model_info["linear_value_dim"],
            tp_size=1,
        )
        return dict(zip(new_param_names, new_params))
    if "visual.blocks" in param_name and "qkv" in param_name:
        param = reshape_visual_block_qkv(param, vision_head_size=model_info.get("vision_head_size"))
    gate_up_prefix = get_fused_moe_expert_prefix(param_name, "gate_up_proj")
    if gate_up_prefix is not None:
        param_dict = {}
        num_experts = model_info["num_experts"]
        if param.ndim != 3 or param.shape[0] != num_experts:
            raise ValueError(
                f"Expected fused gate_up_proj with shape [num_experts, 2 * intermediate_size, hidden_size], "
                f"got {tuple(param.shape)} for {param_name} with num_experts={num_experts}."
            )
        intermediate_size = model_info.get("moe_intermediate_size")
        if intermediate_size is not None and param.shape[1] != 2 * intermediate_size:
            raise ValueError(
                f"Expected fused gate_up_proj dim 1 to be {2 * intermediate_size}, "
                f"got {param.shape[1]} for {param_name}."
            )
        for i in range(num_experts):
            gate_name = f"{gate_up_prefix}.{i}.gate_proj.weight"
            up_name = f"{gate_up_prefix}.{i}.up_proj.weight"
            gate_param, up_param = param[i].chunk(2, dim=0)
            param_dict[gate_name] = gate_param
            param_dict[up_name] = up_param
        return param_dict
    down_prefix = get_fused_moe_expert_prefix(param_name, "down_proj")
    if down_prefix is not None:
        param_dict = {}
        num_experts = model_info["num_experts"]
        if param.ndim != 3 or param.shape[0] != num_experts:
            raise ValueError(
                f"Expected fused down_proj with shape [num_experts, hidden_size, intermediate_size], "
                f"got {tuple(param.shape)} for {param_name} with num_experts={num_experts}."
            )
        intermediate_size = model_info.get("moe_intermediate_size")
        if intermediate_size is not None and param.shape[2] != intermediate_size:
            raise ValueError(
                f"Expected fused down_proj dim 2 to be {intermediate_size}, got {param.shape[2]} for {param_name}."
            )
        for i in range(num_experts):
            down_name = f"{down_prefix}.{i}.down_proj.weight"
            param_dict[down_name] = param[i]
        return param_dict
    return {param_name: param}


def convert_hf_local_shard(
    model_info,
    param_name: str,
    param: torch.Tensor,
    sharding: NIXLSharding,
) -> dict[str, tuple[torch.Tensor, NIXLSharding]]:
    """
    Convert an HF-format local shard to the canonical PSRL transfer layout.

    Unlike `maybe_convert_to_smaller_parts`, this function also updates the
    sharding dimensions after a reshape. It is therefore suitable for FSDP
    local shards, where the original tensor is already partitioned.

    Args:
        model_info: Model metadata returned by `ParameterMapping.get_model_info`.
        param_name (str): Fully-qualified HF parameter name.
        param (torch.Tensor): Local tensor shard.
        sharding (NIXLSharding): Sharding descriptor for the input tensor.

    Returns:
        dict[str, tuple[torch.Tensor, NIXLSharding]]: Canonical key, tensor-view,
            and matching sharding descriptor.
    """
    shard_dim = next(iter(sharding.shard_mesh))
    shard_size = next(iter(sharding.shard_mesh.values()))
    shard_rank = sharding.shard_indices[0][0]

    if shard_dim == 0 and shard_size > 1 and "linear_attn.in_proj_qkv.weight" in param_name:
        return _split_dim0_fsdp_shard(
            param_name,
            param,
            shard_size=shard_size,
            shard_rank=shard_rank,
            output_specs=[
                (param_name + "_q", model_info["linear_key_dim"]),
                (param_name + "_k", model_info["linear_key_dim"]),
                (param_name + "_v", model_info["linear_value_dim"]),
            ],
        )

    if shard_dim == 0 and shard_size > 1 and "linear_attn.conv1d.weight" in param_name:
        key_dim = model_info["linear_num_key_heads"] * model_info["linear_key_head_dim"]
        value_dim = model_info["linear_num_value_heads"] * model_info["linear_value_head_dim"]
        return _split_dim0_fsdp_shard(
            param_name,
            param,
            shard_size=shard_size,
            shard_rank=shard_rank,
            output_specs=[
                (param_name + "_q", key_dim),
                (param_name + "_k", key_dim),
                (param_name + "_v", value_dim),
            ],
        )

    if "visual.blocks" in param_name and "qkv" in param_name:
        shard_indices = sharding.shard_indices
        if shard_dim == 0:
            # Flattened heads preserve the FSDP row partition without copying.
            if shard_size > 1:
                new_sharding = sharding
            else:
                new_sharding = make_visual_qkv_tp_sharding(tp_size=1, tp_rank=0)
        elif shard_dim == 1:
            # The input hidden dimension moves one position to the right.
            new_sharding = NIXLSharding(
                shard_mesh=OrderedDict([(2, shard_size)]),
                shard_indices=list(shard_indices),
            )
        else:
            raise ValueError(
                f"Unsupported visual QKV shard dimension {shard_dim} for {param_name!r}; expected 0 or 1."
            )
        return {
            param_name: (
                reshape_visual_block_qkv(param, vision_head_size=model_info.get("vision_head_size")),
                new_sharding,
            )
        }

    converted = maybe_convert_to_smaller_parts(model_info, param_name, param)
    return {name: (tensor, sharding) for name, tensor in converted.items()}


def _split_dim0_fsdp_shard(
    param_name: str,
    param: torch.Tensor,
    *,
    shard_size: int,
    shard_rank: int,
    output_specs: list[tuple[str, int]],
) -> dict[str, tuple[torch.Tensor, NIXLSharding]]:
    """
    Split a flat FSDP row shard into canonical component keys without copies.

    FSDP partitions the concatenated HF tensor uniformly, so a rank can own
    multiple canonical shards of one component and none of another. The output
    descriptors encode that sparse ownership explicitly.
    """
    global_rows = sum(size for _, size in output_specs)
    assert global_rows % shard_size == 0, (
        f"Global dim 0 of {param_name!r} has size {global_rows}, which is not divisible by "
        f"FSDP shard size {shard_size}."
    )
    local_rows = global_rows // shard_size
    assert param.shape[0] == local_rows, (
        f"Expected local dim 0 of {param_name!r} to have size {local_rows}, got {param.shape[0]}."
    )

    source_start = shard_rank * local_rows
    source_end = source_start + local_rows
    output: dict[str, tuple[torch.Tensor, NIXLSharding]] = {}
    component_start = 0
    for output_name, component_rows in output_specs:
        component_end = component_start + component_rows
        overlap_start = max(source_start, component_start)
        overlap_end = min(source_end, component_end)
        if overlap_start >= overlap_end:
            output[output_name] = (
                param.narrow(0, 0, 0),
                NIXLSharding.empty(),
            )
            component_start = component_end
            continue

        assert component_rows % shard_size == 0, (
            f"Component {output_name!r} has {component_rows} rows, which is not divisible by "
            f"FSDP shard size {shard_size}."
        )
        canonical_rows = component_rows // shard_size
        relative_start = overlap_start - component_start
        overlap_rows = overlap_end - overlap_start
        assert relative_start % canonical_rows == 0 and overlap_rows % canonical_rows == 0, (
            f"FSDP shard boundaries for {param_name!r} do not align with canonical component {output_name!r}."
        )
        first_index = relative_start // canonical_rows
        index_count = overlap_rows // canonical_rows
        local_offset = overlap_start - source_start
        output[output_name] = (
            param.narrow(0, local_offset, overlap_rows),
            NIXLSharding(
                shard_mesh=OrderedDict([(0, shard_size)]),
                shard_indices=[(index,) for index in range(first_index, first_index + index_count)],
            ),
        )
        component_start = component_end
    return output


def convert_hf_inplace(
    parameter_mapping: ParameterMapping,
    model,
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convert HuggingFace model to unified state dict and sharding info.

    Args:
        model: The HuggingFace model instance.
        parameter_mapping (ParameterMapping): Parameter mapping instance carrying
            model_info. Use HFParameterMapping to enable Q/K/V 3D reshaping for
            NIXL shape compatibility between PS and Megatron train workers.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
            (converted_state_dict, sharding_dict).
    """
    converter = HFConverter(parameter_mapping)
    return converter.convert_state_and_sharding_dict(model)
