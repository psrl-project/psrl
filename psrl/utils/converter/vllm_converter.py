from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from torch.nn import Parameter
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    set_weight_attrs,
)
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.qwen3_5 import QwenGatedDeltaNetAttention
from vllm_patches.interfaces import supports_weight_layout

from psrl.utils.converter.base_converter import BaseConverter
from psrl.utils.converter.model_mappings import (
    MappingType,
    ParameterMapping,
    get_qkv_tp_layout,
    make_visual_qkv_tp_sharding,
    reshape_visual_block_qkv,
    slice_attn_conv1d,
    slice_fused_moe_w2_weight,
    slice_fused_moe_w13_weight,
    slice_gate_up_proj,
    slice_in_proj_ba,
    slice_in_proj_qkvz,
    slice_qkv_proj,
    slice_qwen3_5_in_proj_qkv,
)
from psrl.utils.converter.weight_layout_plan import PlanExecutor
from psrl.utils.nixl.nixl_spec import NIXLSharding

# vLLM QKV shard identifiers map to split tensor indices.
QKV_SHARD_ID_MAP: dict[str, int] = {"q": 0, "k": 1, "v": 2}


@dataclass
class ParamMappingEntry:
    mapping_type: MappingType
    mappings: list = field(default_factory=list)
    is_full_path: bool = False


def enable_sharded_weight_attrs(params: dict[str, Parameter]):
    for name, param in params.items():
        set_weight_attrs(param, {"is_sharded_weight": True})
    return params


class VllmConverter(BaseConverter):
    """Convert vLLM model to a unified format (i.e., HuggingFace) and generate sharding info."""

    def __init__(self, parameter_mapping: ParameterMapping | None, tp_rank: int | None = 0, ep_rank: int = 0):
        super().__init__(parameter_mapping)
        self.parameter_mapping = parameter_mapping
        self.tp_rank = tp_rank
        self.ep_rank = ep_rank

    def _build_fused_mappings(
        self,
        stacked_params: list,
        extra_stacked_params: list,
        expert_params: list,
    ) -> dict:
        fused_mappings: dict[str, ParamMappingEntry] = {}

        for packed_suffix, hf_suffix, shard_id in stacked_params:
            if packed_suffix not in fused_mappings:
                mapping_type = MappingType.QKV_SPLIT if isinstance(shard_id, str) else MappingType.GATE_UP_PROJ_SPLIT
                fused_mappings[packed_suffix] = ParamMappingEntry(mapping_type=mapping_type, is_full_path=False)
            int_shard_id = QKV_SHARD_ID_MAP[shard_id] if isinstance(shard_id, str) else shard_id
            fused_mappings[packed_suffix].mappings.append((hf_suffix, int_shard_id))

        for full_packed, full_hf, shard_id in extra_stacked_params:
            if full_packed not in fused_mappings:
                fused_mappings[full_packed] = ParamMappingEntry(
                    mapping_type=MappingType.GATE_UP_PROJ_SPLIT, is_full_path=True
                )
            int_shard_id = QKV_SHARD_ID_MAP[shard_id] if isinstance(shard_id, str) else shard_id
            fused_mappings[full_packed].mappings.append((full_hf, int_shard_id))

        # "w1" (gate_proj), "w2" (down_proj), "w3" (up_proj)
        w13_by_suffix: dict[str, list] = {}
        w2_by_suffix: dict[str, list] = {}
        for packed_suffix, hf_suffix, expert_id, shard_id in expert_params:
            if shard_id in ("w1", "w3"):
                int_shard_id = expert_id * 2 if shard_id == "w1" else expert_id * 2 + 1
                w13_by_suffix.setdefault(packed_suffix, []).append((hf_suffix, int_shard_id))
            else:
                w2_by_suffix.setdefault(packed_suffix, []).append((hf_suffix, expert_id))

        for suffix, entries in w13_by_suffix.items():
            fused_mappings[suffix] = ParamMappingEntry(
                mapping_type=MappingType.FUSED_MOE_W13_SPLIT, mappings=entries, is_full_path=False
            )
        for suffix, entries in w2_by_suffix.items():
            fused_mappings[suffix] = ParamMappingEntry(
                mapping_type=MappingType.FUSED_MOE_W2_SPLIT, mappings=entries, is_full_path=False
            )

        return fused_mappings

    def _build_from_parameter_mapping(self) -> tuple[dict, dict]:
        fused_mappings: dict = {}
        for vllm_name, hf_name, mapping_type, shard_id in self.parameter_mapping.get_mappings():
            if vllm_name not in fused_mappings:
                fused_mappings[vllm_name] = ParamMappingEntry(mapping_type=mapping_type, is_full_path=False)
            else:
                assert mapping_type != MappingType.DIRECT, f"Mapping type should not be DIRECT for {vllm_name}"
                assert mapping_type == fused_mappings[vllm_name].mapping_type, (
                    f"Mapping type for {vllm_name} must be the same, "
                    f"but got {mapping_type} and {fused_mappings[vllm_name].mapping_type}"
                )
            fused_mappings[vllm_name].mappings.append((hf_name, shard_id))
        model_info = self.parameter_mapping.get_model_info()
        return fused_mappings, model_info

    def _convert_with_weight_layout_plan(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert vLLM model using the unified WeightLayoutPlan from build_weight_layout().

        This is the canonical conversion path for all models implementing
        SupportsWeightLayout. Errors propagate to the caller.
        """
        if not self.model_info:
            config = getattr(model, "config", None)
            if config is None and hasattr(model, "model"):
                config = getattr(model.model, "config", None)
            if config is not None:
                from psrl.utils.converter.modeling.hf_modeling import HFParameterMapping

                self.model_info = HFParameterMapping(config).get_model_info()

        plan = model.build_weight_layout()
        resolved_plan = plan.flatten()

        executor = PlanExecutor(
            resolved_plan,
            tp_rank=self.tp_rank or 0,
            ep_rank=self.ep_rank,
        )

        converted_state_dict: dict[str, torch.Tensor] = {}
        sharding_dict: dict[str, NIXLSharding] = {}

        for module_prefix, module in model.named_modules():
            # ── Parameters ────────────────────────────────────────────────
            for param_name, param in module.named_parameters(recurse=False):
                # Skip pipeline-parallel placeholder tensors (wrong stage)
                if getattr(param, "is_pp_missing_parameter", False):
                    continue

                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name

                for converted in executor.execute(full_name, param, module):
                    sharding = converted.sharding
                    if sharding is None:
                        sharding = self.get_sharding_for_param(module, param_name, full_name)
                    converted_name = converted.name
                    converted_param = converted.param
                    if "visual.blocks" in converted_name and "qkv" in converted_name:
                        converted_param = reshape_visual_block_qkv(
                            converted_param,
                            vision_head_size=self.model_info.get("vision_head_size"),
                        )
                        sharding = make_visual_qkv_tp_sharding(
                            tp_size=getattr(module, "tp_size", 1),
                            tp_rank=self.tp_rank or 0,
                        )
                    sharding = self._adjust_kv_sharding(converted_name, sharding, module, self.model_info)
                    converted_param, sharding = self.maybe_reshape_qkv_to_3d(converted_name, converted_param, sharding)
                    converted_state_dict[converted_name] = converted_param
                    sharding_dict[converted_name] = sharding

            # Export only buffers with an explicit layout rule.
            for buf_name, buf in module.named_buffers(recurse=False):
                if buf is None:
                    continue
                full_name = f"{module_prefix}.{buf_name}" if module_prefix else buf_name

                # Only process if there's an explicit rule for this buffer
                if not resolved_plan.matches_rules(full_name, module):
                    continue

                for converted in executor.execute(full_name, buf, module):
                    sharding = converted.sharding
                    if sharding is None:
                        # Buffers are typically replicated (EP/TP handled by transform)
                        sharding = NIXLSharding(
                            shard_mesh=OrderedDict([(0, 1)]),
                            shard_indices=[(0,)],
                        )
                    converted_param, sharding = self.maybe_reshape_qkv_to_3d(converted.name, converted.param, sharding)
                    converted_state_dict[converted.name] = converted_param
                    sharding_dict[converted.name] = sharding

        return converted_state_dict, sharding_dict

    def convert_state_and_sharding_dict(self, model) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
        """
        Convert vLLM model to unified state dict and generate sharding info.
        Args:
            model: The vLLM model instance
        Returns:
            (converted_state_dict, sharding_dict)
        """
        if supports_weight_layout(model):
            return self._convert_with_weight_layout_plan(model)
        if self.parameter_mapping is not None:
            fused_mappings, model_info = self._build_from_parameter_mapping()
        else:
            raise ValueError(
                f"{type(model).__name__} does not implement SupportsWeightLayout "
                "and no parameter_mapping was provided. Either implement "
                "build_weight_layout() in the model, or pass parameter_mapping= "
                "to convert_vllm_inplace()."
            )

        converted_state_dict = {}
        sharding_dict = {}

        # Workaround: for lm_head, we do not care if it shares the weight with wte
        lm_head_module = None
        lm_head_module_prefix = None
        if hasattr(model, "lm_head"):
            lm_head_module = model.lm_head
            lm_head_module_prefix = "lm_head"
        if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
            lm_head_module = model.language_model.lm_head
            lm_head_module_prefix = "lm_head"

        seen_module_prefixes = set()
        for module_prefix, module in model.named_modules():
            if module_prefix.startswith("visual"):
                module_prefix = f"model.{module_prefix}"
            if module_prefix.startswith("language_model.model"):
                module_prefix = f"model.language_model.{module_prefix[21:]}"
            if module_prefix.startswith("language_model.lm_head"):
                module_prefix = "lm_head"
            seen_module_prefixes.add(module_prefix)
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name
                new_params = self.convert_parameter(full_name, param, module, fused_mappings, model_info)
                sharding = self.get_sharding_for_param(module, param_name, full_name)
                for new_param_name, new_param in new_params.items():
                    if "visual.blocks" in new_param_name and "qkv" in new_param_name:
                        sharding = make_visual_qkv_tp_sharding(
                            tp_size=getattr(module, "tp_size", 1),
                            tp_rank=self.tp_rank or 0,
                        )
                    adjusted_sharding = self._adjust_kv_sharding(new_param_name, sharding, module, model_info)
                    new_param, sharding_for_param = self.maybe_reshape_qkv_to_3d(
                        new_param_name, new_param, adjusted_sharding
                    )
                    converted_state_dict[new_param_name] = new_param
                    sharding_dict[new_param_name] = sharding_for_param

        # Handle lm_head separately
        if lm_head_module is not None and (lm_head_module_prefix not in seen_module_prefixes):
            module = lm_head_module
            module_prefix = lm_head_module_prefix
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}" if module_prefix else param_name
                new_params = self.convert_parameter(full_name, param, module, fused_mappings, model_info)
                sharding = self.get_sharding_for_param(module, param_name, full_name)
                for new_param_name, new_param in new_params.items():
                    sharding = self.get_sharding_for_param(module, param_name, new_param_name)
                    new_param, sharding_for_param = self.maybe_reshape_qkv_to_3d(new_param_name, new_param, sharding)
                    converted_state_dict[new_param_name] = new_param
                    sharding_dict[new_param_name] = sharding_for_param

        return converted_state_dict, sharding_dict

    def convert_parameter(
        self,
        full_name: str,
        param: Parameter,
        module,
        fused_mappings: dict,
        model_info: dict,
    ) -> dict:
        """
        Convert the parameter, may need to split inplace
        if it matches a split mapping type (e.g., qkv_proj, gate_up_proj).
        """
        tp_size = getattr(module, "tp_size", 1)
        for vllm_name, entry in fused_mappings.items():
            mapping_type = entry.mapping_type
            mappings = entry.mappings
            is_full_path = entry.is_full_path
            matched = (full_name == vllm_name) if is_full_path else (vllm_name in full_name)
            if not matched:
                continue
            if mapping_type == MappingType.DIRECT:
                assert len(mappings) == 1, f"Mapping type is DIRECT for {vllm_name}, but got {len(mappings)} mappings"
                new_param = param
                hf_name = mappings[0][0]
                new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                return {new_param_name: new_param}
            elif mapping_type == MappingType.QKV_SPLIT:
                try:
                    sliced_params = slice_qkv_proj(
                        fused_param=param,
                        num_heads=model_info["num_heads"],
                        num_kv_heads=model_info["num_kv_heads"],
                        head_size=model_info["head_size"],
                        tp_size=tp_size,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice qkv parameter {full_name}: {e}") from e
                out = {}
                for hf_name, shard_id in mappings:
                    assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                    new_param = sliced_params[shard_id]
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            elif mapping_type == MappingType.GATE_UP_PROJ_SPLIT:
                intermediate_size = model_info["intermediate_size"]
                if intermediate_size is None:
                    intermediate_size = self.model_info["moe_intermediate_size"]
                assert intermediate_size is not None, (
                    "Intermediate size must be specified in model_info for gate_up_proj split"
                )
                try:
                    sliced_params = slice_gate_up_proj(
                        fused_param=param,
                        output_sizes=[intermediate_size, intermediate_size],
                        tp_size=tp_size,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice gate up proj parameter {full_name}: {e}") from e
                out = {}
                for hf_name, shard_id in mappings:
                    assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                    new_param = sliced_params[shard_id]
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            elif mapping_type == MappingType.IN_PROJ_QKVZ_SPLIT:
                key_dim = self.model_info.get("linear_key_dim")
                value_dim = self.model_info.get("linear_value_dim")
                if key_dim is None or value_dim is None:
                    raise ValueError(
                        "Qwen3.5 linear attention dims are missing in model_info; "
                        f"got linear_key_dim={key_dim} and linear_value_dim={value_dim}."
                    )
                try:
                    sliced_params = slice_in_proj_qkvz(
                        fused_param=param,
                        key_dim=key_dim,
                        value_dim=value_dim,
                        tp_size=tp_size,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice in_proj_qkvz parameter {full_name}: {e}") from e
                out = {}
                for hf_name, shard_id in mappings:
                    assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                    new_param = sliced_params[shard_id]
                    new_param_name = full_name.replace(vllm_name, hf_name)
                    if shard_id == 0:
                        qkv_names = [new_param_name + "_q", new_param_name + "_k", new_param_name + "_v"]
                        qkv_params = slice_qwen3_5_in_proj_qkv(
                            fused_param=new_param,
                            key_dim=key_dim,
                            value_dim=value_dim,
                            tp_size=tp_size,
                        )
                        out.update(dict(zip(qkv_names, qkv_params)))
                        continue
                    out[new_param_name] = new_param
                return out
            elif mapping_type == MappingType.IN_PROJ_BA_SPLIT:
                num_v_heads = self.model_info.get("linear_num_value_heads")
                if num_v_heads is None:
                    raise ValueError(
                        "Qwen3.5 linear attention num_v_heads is missing in model_info; "
                        f"got linear_num_value_heads={num_v_heads}."
                    )
                try:
                    sliced_params = slice_in_proj_ba(
                        fused_param=param,
                        num_v_heads=num_v_heads,
                        tp_size=tp_size,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice in_proj_ba parameter {full_name}: {e}") from e
                out = {}
                for hf_name, shard_id in mappings:
                    assert shard_id < len(sliced_params), f"Shard id {shard_id} is out of range for {vllm_name}"
                    new_param = sliced_params[shard_id]
                    new_param_name = full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            elif mapping_type == MappingType.FUSED_MOE_W13_SPLIT:
                try:
                    sliced_params = slice_fused_moe_w13_weight(
                        fused_param=param,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice w13_weight parameter {full_name}: {e}") from e
                out = {}
                ep_size = getattr(module, "ep_size", 1)
                # NOTE(zym): The `ep_rank` value is unreliable with DP one, so use `tp_rank`.
                ep_rank = self.tp_rank if ep_size > 1 else 0
                num_experts = model_info["num_experts"]
                num_experts_per_ep_rank = num_experts // ep_size
                local_experts_start_id = ep_rank * num_experts_per_ep_rank
                local_experts_end_id = local_experts_start_id + num_experts_per_ep_rank
                local_shard_ids = list(range(local_experts_start_id * 2, local_experts_end_id * 2))
                for hf_name, shard_id in mappings:
                    if shard_id not in local_shard_ids:
                        continue
                    slice_idx = shard_id - local_experts_start_id * 2
                    assert slice_idx < len(sliced_params), f"Slice idx {slice_idx} is out of range for {vllm_name}"
                    new_param = sliced_params[slice_idx]
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            elif mapping_type == MappingType.FUSED_MOE_W2_SPLIT:
                try:
                    sliced_params = slice_fused_moe_w2_weight(
                        fused_param=param,
                    )
                except Exception as e:
                    raise ValueError(f"Failed to slice w2_weight parameter {full_name}: {e}") from e
                out = {}
                ep_size = getattr(module, "ep_size", 1)
                ep_rank = self.tp_rank if ep_size > 1 else 0
                num_experts = model_info["num_experts"]
                num_experts_per_ep_rank = num_experts // ep_size
                local_experts_start_id = ep_rank * num_experts_per_ep_rank
                local_experts_end_id = local_experts_start_id + num_experts_per_ep_rank
                local_shard_ids = list(range(local_experts_start_id, local_experts_end_id))
                for hf_name, shard_id in mappings:
                    if shard_id not in local_shard_ids:
                        continue
                    slice_idx = shard_id - local_experts_start_id
                    assert slice_idx < len(sliced_params), f"Slice idx {slice_idx} is out of range for {vllm_name}"
                    new_param = sliced_params[slice_idx]
                    new_param_name = hf_name if is_full_path else full_name.replace(vllm_name, hf_name)
                    out[new_param_name] = new_param
                return out
            else:
                raise ValueError(f"Unsupported mapping type: {mapping_type}")

        if "linear_attn.conv1d.weight" in full_name:
            new_param_names = [full_name + "_q", full_name + "_k", full_name + "_v"]
            new_params = slice_attn_conv1d(
                fused_param=param,
                num_k_heads=self.model_info["linear_num_key_heads"],
                num_v_heads=self.model_info["linear_num_value_heads"],
                k_head_size=self.model_info["linear_key_head_dim"],
                v_head_size=self.model_info["linear_value_head_dim"],
                tp_size=tp_size,
            )
            return dict(zip(new_param_names, new_params))

        if "visual.blocks" in full_name and "qkv" in full_name:
            param = reshape_visual_block_qkv(param, vision_head_size=self.model_info.get("vision_head_size"))

        # Default: No conversion needed
        return {full_name: param}

    def _adjust_kv_sharding(self, param_name: str, sharding: NIXLSharding, module, model_info: dict) -> NIXLSharding:
        """Adjust sharding for K/V projection weights when KV heads are replicated.

        In GQA with num_kv_heads < tp_size, vLLM replicates KV heads across TP ranks.
        For example, with num_kv_heads=2 and tp_size=4: ranks 0,1 share KV head 0,
        ranks 2,3 share KV head 1. The effective KV tp_size is num_kv_heads (not tp_size).

        Without this adjustment, maybe_reshape_qkv_to_3d uses the full tp_size and
        produces a 3D shape incompatible with the PS side.
        """
        if not isinstance(module, QKVParallelLinear):
            return sharding
        # Only adjust for K/V weights (not Q)
        if not (
            param_name.endswith("k_proj.weight")
            or param_name.endswith("v_proj.weight")
            or param_name.endswith("k_proj.bias")
            or param_name.endswith("v_proj.bias")
        ):
            return sharding

        num_heads = model_info.get("num_heads")
        num_kv_heads = model_info.get("num_kv_heads")
        tp_size = getattr(module, "tp_size", 1)
        if num_heads is None or num_kv_heads is None or tp_size <= 1:
            return sharding

        layout = get_qkv_tp_layout(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tp_size=tp_size,
        )
        if layout.num_kv_head_replicas == 1:
            return sharding

        shard_dim = next(iter(sharding.shard_mesh.keys()))
        return NIXLSharding(
            shard_mesh=OrderedDict([(shard_dim, layout.num_kv_shards)]),
            shard_indices=[(layout.get_kv_shard_rank(self.tp_rank),)],
        )

    def get_sharding_for_param(self, module, param_name, full_name=None) -> NIXLSharding:
        """
        Generate sharding info for a parameter given its module and tp_rank.
        Returns a NIXLSharding object.
        """
        tp_size = getattr(module, "tp_size", 1)
        if tp_size > 1:
            assert tp_size > self.tp_rank, (
                f"Tensor parallel size ({tp_size}) must be "
                f"greater than tensor parallel rank ({self.tp_rank}), "
                f"please check the tensor parallel size and rank."
            )
            shard_indices = [(self.tp_rank,)] if self.tp_rank is not None else [(0,)]
            if isinstance(
                module,
                (
                    ColumnParallelLinear,
                    MergedColumnParallelLinear,
                    QKVParallelLinear,
                    VocabParallelEmbedding,
                ),
            ):
                shard_dim = 0
            elif isinstance(module, RowParallelLinear):
                if param_name == "bias":
                    # NOTE(zym): Bias remains replicated.
                    tp_size = 1
                    shard_indices = [(0,)]
                    shard_dim = 0
                else:
                    shard_dim = 1
            elif isinstance(module, FusedMoE):
                if "w13" in param_name:
                    shard_dim = 0
                else:
                    assert "w2" in param_name, f"FusedMoE param can only be w13 and w2, but get {param_name}"
                    shard_dim = 1
            elif isinstance(module, ReplicatedLinear):
                # NOTE(zym): `ReplicatedLinear` carries the global TP size despite
                # storing a replicated parameter.
                tp_size = 1
                shard_indices = [(0,)]
                shard_dim = 0
            elif isinstance(module, QwenGatedDeltaNetAttention):
                shard_dim = 0
            else:
                raise ValueError(f"Unsupported module type for sharding: {type(module)}")
        else:
            shard_indices = [(0,)]
            shard_dim = 0
        kwargs = {
            "shard_mesh": OrderedDict([(shard_dim, tp_size)]),
            "shard_indices": shard_indices,
        }
        return NIXLSharding(**kwargs)


def convert_vllm_inplace(
    model,
    tp_rank: int = 0,
    ep_rank: int = 0,
    *,
    parameter_mapping: ParameterMapping | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, NIXLSharding]]:
    """
    Convert a vLLM model to unified HF-format state dict with NIXL sharding info.

    For models implementing SupportsWeightLayout, no parameter_mapping is needed.
    For custom models not yet supporting the interface, pass parameter_mapping as
    a keyword argument.
    """
    converter = VllmConverter(parameter_mapping=parameter_mapping, tp_rank=tp_rank, ep_rank=ep_rank)
    return converter.convert_state_and_sharding_dict(model)
