import math
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch.nn import Parameter

from psrl.utils.converter.model_mappings import (
    ParameterMapping,
    is_q_weight,
    is_qkv_weight,
    make_slice_parameter,
    reshape_q_to_5d,
    reshape_qkv_to_3d,
)
from psrl.utils.nixl.nixl_spec import NIXLSharding


class BaseConverter(ABC):
    """Base class for state dict and sharding converter."""

    def __init__(self, parameter_mapping: ParameterMapping | None):
        """
        Initialize the converter with model info from the given parameter mapping.

        Args:
            parameter_mapping (ParameterMapping | None): Parameter mapping for the model.
                When None (e.g. for models implementing SupportsWeightLayout),
                model_info is populated lazily by the subclass during conversion.
        """
        self.model_info = parameter_mapping.get_model_info() if parameter_mapping is not None else {}

    def maybe_reshape_qkv_to_3d(
        self,
        param_name: str,
        param: Parameter,
        sharding: NIXLSharding,
    ) -> tuple[Parameter, NIXLSharding]:
        """
        Reshape a QKV parameter and update its sharding descriptor.

        The returned descriptor preserves physical shard ownership after the tensor
        rank changes.

        Args:
            param_name (str): Fully-qualified parameter name.
            param (Parameter): The 1D or 2D parameter tensor.
            sharding (NIXLSharding): Existing sharding descriptor for this param.

        Returns:
            tuple[Parameter, NIXLSharding]: Reshaped param and updated sharding.
        """
        num_heads = self.model_info.get("num_heads")
        if num_heads is None or not is_qkv_weight(param_name) or param.ndim not in (1, 2):
            return param, sharding
        if self.model_info.get("uses_mla_attention") and is_q_weight(param_name):
            return param, sharding

        num_kv_heads = self.model_info["num_kv_heads"]
        head_size = self.model_info["head_size"]
        G_global = math.gcd(num_heads, num_kv_heads)
        shard_dim_2d = next(iter(sharding.shard_mesh.keys()))
        ws = next(iter(sharding.shard_mesh.values()))
        rank = sharding.shard_indices[0][0]
        # For 1D bias tensors, treat hidden dimension as 1.
        rows = param.shape[0]
        H = param.shape[1] if param.ndim == 2 else 1
        attn_output_gate = self.model_info.get("attn_output_gate", False)

        # Derive local head counts from the sharding descriptor.
        if shard_dim_2d == 0:
            num_heads_local = num_heads // ws
            num_kv_heads_local = num_kv_heads // ws
        else:
            # dim=1 (hidden sharded): all heads are present on this rank.
            num_heads_local = num_heads
            num_kv_heads_local = num_kv_heads

        if shard_dim_2d == 1:
            # Case A: the sharded hidden dimension moves from dimension one to two.
            reshaped = reshape_qkv_to_3d(
                make_slice_parameter(param.data, param),
                num_heads_local=num_heads_local,
                num_kv_heads_local=num_kv_heads_local,
                head_size=head_size,
            )
            new_sharding = NIXLSharding(
                shard_mesh=OrderedDict([(2, ws)]),
                shard_indices=[(rank,)],
            )
            if attn_output_gate and is_q_weight(param_name):
                reshaped = reshape_q_to_5d(
                    reshaped,
                    num_heads_local=num_heads_local,
                    head_size=head_size,
                )
                new_sharding = NIXLSharding(
                    shard_mesh=OrderedDict([(4, ws)]),
                    shard_indices=[(rank,)],
                )
        elif ws <= G_global:
            # Case B: coarse head sharding, shard_mesh={0: ws} stays valid in 3D.
            reshaped = reshape_qkv_to_3d(
                make_slice_parameter(param.data, param),
                num_heads_local=num_heads_local,
                num_kv_heads_local=num_kv_heads_local,
                head_size=head_size,
            )
            new_sharding = sharding
            if attn_output_gate and is_q_weight(param_name):
                reshaped = reshape_q_to_5d(
                    reshaped,
                    num_heads_local=num_heads_local,
                    head_size=head_size,
                )
        else:
            # Case C: fine-grained sharding spills from dim=0 into dim=1.
            assert ws % G_global == 0, (
                f"FSDP world size ({ws}) is not divisible by G_global={G_global} for {param_name}."
            )
            steps = ws // G_global
            if attn_output_gate and is_q_weight(param_name):
                # The fine grained Q shard contains only a fraction of one attention group.
                assert rows % (2 * head_size) == 0, (
                    f"rows={rows} must be divisible by 2*head_size={2 * head_size} "
                    f"for attn_output_gate 5D reshape in Case C for {param_name}."
                )
                q_heads_per_group_local = rows // (2 * head_size)
                reshaped = make_slice_parameter(param.data.reshape(1, q_heads_per_group_local, 2, head_size, H), param)
                new_sharding = NIXLSharding(
                    shard_mesh=OrderedDict([(0, G_global), (1, steps)]),
                    shard_indices=[(rank // steps, rank % steps)],
                )
            else:
                reshaped = make_slice_parameter(param.data.reshape(1, rows, H), param)
                new_sharding = NIXLSharding(
                    shard_mesh=OrderedDict([(0, G_global), (1, steps)]),
                    shard_indices=[(rank // steps, rank % steps)],
                )

        return reshaped, new_sharding

    @abstractmethod
    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        pass
