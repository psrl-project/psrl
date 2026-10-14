import math
import warnings
from collections import OrderedDict

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import StateDictType
from torch.distributed.tensor import DTensor
from verl.utils.fsdp_utils import fsdp_version

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.hf_converter import convert_hf_local_shard
from psrl.utils.converter.model_mappings import ParameterMapping, get_fused_moe_expert_prefix
from psrl.utils.nixl.nixl_spec import NIXLSharding


def split_dim0_fused_fsdp_param(
    local_param: torch.Tensor,
    sharding: NIXLSharding,
    component_names: list[str],
    component_sizes: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Split a fused dimension-zero parameter while preserving local FSDP views.

    A local FSDP shard can overlap one or more canonical components. The
    returned descriptors express each overlap using a component-local mesh.
    """
    if len(component_names) != len(component_sizes):
        raise ValueError(
            "component_names and component_sizes must have the same length, "
            f"got {len(component_names)} and {len(component_sizes)}."
        )
    if not component_sizes or any(size <= 0 for size in component_sizes):
        raise ValueError(f"component_sizes must be positive, got {component_sizes}.")
    if len(sharding.shard_mesh) != 1 or len(sharding.shard_indices) != 1:
        raise ValueError(
            "FSDP fused-parameter splitting requires one sharded dimension and one local shard, "
            f"got shard_mesh={sharding.shard_mesh}, shard_indices={sharding.shard_indices}."
        )

    total_size = sum(component_sizes)
    shard_dim, shard_count = next(iter(sharding.shard_mesh.items()))
    shard_rank = sharding.shard_indices[0][0]
    if not 0 <= shard_rank < shard_count:
        raise ValueError(f"Invalid shard rank {shard_rank} for shard count {shard_count}.")

    converted_state_dict: dict[str, torch.Tensor] = {}
    converted_sharding_dict: dict[str, NIXLSharding] = {}

    if shard_dim != 0:
        if local_param.shape[0] != total_size:
            raise ValueError(
                f"Expected the unsharded fused dimension to have size {total_size}, "
                f"got local shape {tuple(local_param.shape)} with shard_dim={shard_dim}."
            )
        offset = 0
        for name, size in zip(component_names, component_sizes):
            converted_state_dict[name] = local_param.narrow(0, offset, size)
            converted_sharding_dict[name] = NIXLSharding(
                shard_mesh=sharding.shard_mesh.copy(),
                shard_indices=list(sharding.shard_indices),
            )
            offset += size
        return converted_state_dict, converted_sharding_dict

    local_size = local_param.shape[0]
    if local_size * shard_count != total_size:
        raise ValueError(
            "FSDP dim-0 fused parameter must be evenly sharded before canonical splitting: "
            f"local_size={local_size}, shard_count={shard_count}, expected_global_size={total_size}."
        )

    local_global_start = shard_rank * local_size
    local_global_end = local_global_start + local_size
    component_start = 0
    for name, component_size in zip(component_names, component_sizes):
        component_end = component_start + component_size
        overlap_start = max(local_global_start, component_start)
        overlap_end = min(local_global_end, component_end)
        if overlap_start < overlap_end:
            shard_unit = math.gcd(local_size, component_start, component_size)
            component_shard_count = component_size // shard_unit
            first_shard = (overlap_start - component_start) // shard_unit
            past_last_shard = (overlap_end - component_start) // shard_unit
            local_offset = overlap_start - local_global_start
            overlap_size = overlap_end - overlap_start
            if overlap_size % shard_unit != 0:
                raise ValueError(
                    f"Local overlap for {name} is not aligned to canonical shard size: "
                    f"overlap_size={overlap_size}, shard_unit={shard_unit}."
                )
            converted_state_dict[name] = local_param.narrow(0, local_offset, overlap_size)
            converted_sharding_dict[name] = NIXLSharding(
                shard_mesh=OrderedDict([(0, component_shard_count)]),
                shard_indices=[(index,) for index in range(first_shard, past_last_shard)],
            )
        component_start = component_end

    return converted_state_dict, converted_sharding_dict


def split_fused_moe_fsdp_param(
    model_info: dict,
    param_name: str,
    local_param: torch.Tensor,
    sharding: NIXLSharding,
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]] | None:
    """
    Split fused MoE expert weights while preserving FSDP ownership.

    Supports FSDP sharding across either the expert or projection dimension.
    """
    gate_up_prefix = get_fused_moe_expert_prefix(param_name, "gate_up_proj")
    down_prefix = get_fused_moe_expert_prefix(param_name, "down_proj")
    if gate_up_prefix is None and down_prefix is None:
        return None

    num_experts = model_info.get("num_experts")
    if not isinstance(num_experts, int) or isinstance(num_experts, bool) or num_experts <= 0:
        raise ValueError(f"A positive num_experts is required to split {param_name}, got {num_experts}.")
    if local_param.ndim != 3:
        raise ValueError(f"Expected a 3D fused expert parameter for {param_name}, got {tuple(local_param.shape)}.")
    if len(sharding.shard_mesh) != 1 or len(sharding.shard_indices) != 1:
        raise ValueError(
            "FSDP fused expert splitting requires one sharded dimension and one local shard, "
            f"got shard_mesh={sharding.shard_mesh}, shard_indices={sharding.shard_indices}."
        )

    shard_dim, shard_count = next(iter(sharding.shard_mesh.items()))
    shard_rank = sharding.shard_indices[0][0]
    if not 0 <= shard_rank < shard_count:
        raise ValueError(f"Invalid shard rank {shard_rank} for shard count {shard_count}.")

    intermediate_size = model_info.get("moe_intermediate_size")
    if gate_up_prefix is not None and (
        not isinstance(intermediate_size, int) or isinstance(intermediate_size, bool) or intermediate_size <= 0
    ):
        raise ValueError(
            f"A positive moe_intermediate_size is required to split {param_name}, got {intermediate_size}."
        )

    converted_state_dict: dict[str, torch.Tensor] = {}
    converted_sharding_dict: dict[str, NIXLSharding] = {}

    if shard_dim == 0:
        local_expert_count = local_param.shape[0]
        if local_expert_count * shard_count != num_experts:
            raise ValueError(
                "FSDP expert-dimension sharding must divide num_experts evenly: "
                f"local_experts={local_expert_count}, shard_count={shard_count}, "
                f"num_experts={num_experts} for {param_name}."
            )
        first_expert_id = shard_rank * local_expert_count
        for local_expert_id in range(local_expert_count):
            expert_id = first_expert_id + local_expert_id
            expert_param = local_param[local_expert_id]
            if gate_up_prefix is not None:
                if expert_param.shape[0] != 2 * intermediate_size:
                    raise ValueError(
                        f"Expected fused gate_up_proj dim 1 to be {2 * intermediate_size}, "
                        f"got {expert_param.shape[0]} for {param_name}."
                    )
                gate_param, up_param = expert_param.chunk(2, dim=0)
                names_and_params = (
                    (f"{gate_up_prefix}.{expert_id}.gate_proj.weight", gate_param),
                    (f"{gate_up_prefix}.{expert_id}.up_proj.weight", up_param),
                )
            else:
                if intermediate_size is not None and expert_param.shape[1] != intermediate_size:
                    raise ValueError(
                        f"Expected fused down_proj dim 2 to be {intermediate_size}, "
                        f"got {expert_param.shape[1]} for {param_name}."
                    )
                names_and_params = ((f"{down_prefix}.{expert_id}.down_proj.weight", expert_param),)

            for name, tensor in names_and_params:
                converted_state_dict[name] = tensor
                converted_sharding_dict[name] = NIXLSharding.default()
        return converted_state_dict, converted_sharding_dict

    if shard_dim == 1:
        if local_param.shape[0] != num_experts:
            raise ValueError(
                f"Expected all {num_experts} experts when FSDP shards projection dim 1 of {param_name}, "
                f"got {local_param.shape[0]}."
            )
        canonical_projection_sharding = NIXLSharding(
            shard_mesh=OrderedDict([(0, shard_count)]),
            shard_indices=[(shard_rank,)],
        )
        for expert_id in range(num_experts):
            expert_param = local_param[expert_id]
            if gate_up_prefix is not None:
                split_state, split_sharding = split_dim0_fused_fsdp_param(
                    local_param=expert_param,
                    sharding=canonical_projection_sharding,
                    component_names=[
                        f"{gate_up_prefix}.{expert_id}.gate_proj.weight",
                        f"{gate_up_prefix}.{expert_id}.up_proj.weight",
                    ],
                    component_sizes=[intermediate_size, intermediate_size],
                )
                converted_state_dict.update(split_state)
                converted_sharding_dict.update(split_sharding)
            else:
                if intermediate_size is not None and expert_param.shape[1] != intermediate_size:
                    raise ValueError(
                        f"Expected fused down_proj dim 2 to be {intermediate_size}, "
                        f"got {expert_param.shape[1]} for {param_name}."
                    )
                name = f"{down_prefix}.{expert_id}.down_proj.weight"
                converted_state_dict[name] = expert_param
                converted_sharding_dict[name] = NIXLSharding(
                    shard_mesh=canonical_projection_sharding.shard_mesh.copy(),
                    shard_indices=list(canonical_projection_sharding.shard_indices),
                )
        return converted_state_dict, converted_sharding_dict

    raise ValueError(
        f"FSDP fused expert parameters support sharding on dimensions 0 or 1, got dim {shard_dim} for {param_name}."
    )


class FSDPConverter(BaseConverter):
    """Converter for FSDP/FSDP2 model."""

    def __init__(self, fsdp_strategy: str, parameter_mapping: ParameterMapping):
        """
        Args:
            fsdp_strategy (str): FSDP strategy, either 'fsdp' or 'fsdp2'.
            parameter_mapping (ParameterMapping): Parameter mapping instance carrying
                model_info (num_heads, num_kv_heads, head_size). Use FSDPParameterMapping
                to enable Q/K/V 3D reshaping for NIXL shape compatibility between
                FSDP train workers and the PS.
        """
        super().__init__(parameter_mapping)
        self.fsdp_strategy = fsdp_strategy

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert FSDP/FSDP2 model to unified state dict and sharding info.

        Args:
            model: The FSDP/FSDP2 model instance.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
                (converted_state_dict, sharding_dict).
        """
        # Determine the FSDP strategy and convert accordingly.
        # fsdp_state_dict will be (name, DTensor) pairs.
        if self.fsdp_strategy == "fsdp":
            warnings.warn(
                "FSDP strategy is deprecated beacause it cannot "
                "guarantee the in-place representation of the state dict.",
                stacklevel=2,
            )
            with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
                fsdp_state_dict = model.state_dict()
        elif self.fsdp_strategy == "fsdp2":
            fsdp_state_dict = model.state_dict()
        else:
            raise ValueError(f"Unsupported FSDP strategy: {self.fsdp_strategy}")

        # Convert the FSDP state dict to a unified format.
        converted_state_dict = {}
        sharding_dict = {}
        for param_name, param in fsdp_state_dict.items():
            assert isinstance(param, DTensor), f"Expected DTensor for {param_name}, got {type(param)}."
            local_param = param.to_local()
            # Compute sharding from original 2D DTensor placements before any reshape.
            sharding = self.get_sharding_for_param(param_name, param)
            moe_split = split_fused_moe_fsdp_param(
                self.model_info,
                param_name,
                local_param,
                sharding,
            )
            if moe_split is not None:
                split_state_dict, split_sharding_dict = moe_split
                converted_state_dict.update(split_state_dict)
                sharding_dict.update(split_sharding_dict)
                continue
            # NOTE(lhy): Reshape Q/K/V local shards to 3D to match slice_qkv_proj_megatron layout
            # and update sharding to reflect the new tensor shape.
            local_param, sharding = self.maybe_reshape_qkv_to_3d(param_name, local_param, sharding)
            converted_params = convert_hf_local_shard(
                self.model_info,
                param_name,
                local_param,
                sharding,
            )
            for converted_name, (converted_param, converted_sharding) in converted_params.items():
                converted_state_dict[converted_name] = converted_param
                sharding_dict[converted_name] = converted_sharding
        return converted_state_dict, sharding_dict

    def get_sharding_for_param(self, param_name: str, param: DTensor) -> NIXLSharding:
        """
        Generate sharding info for a parameter.
        Returns a NIXLSharding object.
        """
        # FSDP
        if len(param.placements) == 1:
            assert param.device_mesh and param.device_mesh.ndim == 1, (
                f"Expected 1 dim device mesh for {param_name}, got {param.device_mesh}"
            )
            if param.placements[0].is_shard(dim=0):
                shard_dim = 0
            elif param.placements[0].is_shard(dim=1):
                shard_dim = 1
            else:
                raise ValueError(f"Unexpected shard_dim for {param_name}, got {param.placements}")
            kwargs = {
                "shard_mesh": OrderedDict([(shard_dim, param.device_mesh.size())]),
                "shard_indices": [(param.device_mesh.get_rank(),)],
            }
        # HSDP
        else:
            assert len(param.placements) == 2 and param.placements[0].is_replicate(), (
                f"Expected two shards (first replicate, second shard on dim 0) for {param_name} "
                f"when using hybrid FSDP, got {param.placements}"
            )
            assert param.device_mesh and param.device_mesh.ndim == 2, (
                f"Expected 2 dim device mesh for {param_name}, got {param.device_mesh}"
            )
            if param.placements[1].is_shard(dim=0):
                shard_dim = 0
            elif param.placements[1].is_shard(dim=1):
                shard_dim = 1
            else:
                raise ValueError(f"Unexpected shard_dim for {param_name}, got {param.placements}")
            kwargs = {
                "shard_mesh": OrderedDict([(shard_dim, param.device_mesh.size(mesh_dim=1))]),
                "shard_indices": [(param.device_mesh.get_local_rank(mesh_dim=1),)],
            }
        return NIXLSharding(**kwargs)


def convert_fsdp_inplace(
    parameter_mapping: ParameterMapping,
    model,
    fsdp_strategy: str = "fsdp2",
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convert FSDP/FSDP2 model to unified state dict and sharding info.

    Args:
        fsdp_strategy (str): FSDP strategy, either 'fsdp' or 'fsdp2'.
        model: The FSDP/FSDP2 model instance.
        parameter_mapping (ParameterMapping): Parameter mapping instance carrying
            model_info. Use FSDPParameterMapping to enable Q/K/V 3D reshaping for
            NIXL shape compatibility between FSDP train workers and the PS.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]: A pair of
            (converted_state_dict, sharding_dict).
    """
    if fsdp_strategy == "fsdp":
        assert fsdp_version(model) == 1, (
            f"FSDP version 1 is expected for 'fsdp' strategy, but got {fsdp_version(model)}"
        )
    elif fsdp_strategy == "fsdp2":
        assert fsdp_version(model) == 2, (
            f"FSDP version 2 is expected for 'fsdp2' strategy, but got {fsdp_version(model)}"
        )
    else:
        raise ValueError(f"Unsupported FSDP strategy: {fsdp_strategy}")
    converter = FSDPConverter(fsdp_strategy, parameter_mapping)
    return converter.convert_state_and_sharding_dict(model)
