import warnings

from psrl.utils.converter.model_mappings import (
    MappingType,
    ParameterMapping,
    register_model,
)

# Qwen2
vllm_qwen2_classes = []
try:
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM, Qwen2Model

    vllm_qwen2_classes = [Qwen2ForCausalLM, Qwen2Model]
except ImportError as e:
    warnings.warn(f"Could not import Qwen2 classes: {e}", stacklevel=2)


@register_model(["VllmQwen2ForCausalLM", "VllmQwen2Model", "VllmQwen2ForSequenceClassification"] + vllm_qwen2_classes)
class VllmQwen2ParameterMapping(ParameterMapping):
    """Parameter mapping for Qwen2 model."""

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]


# Qwen2Moe
vllm_qwen2_moe_classes = []
try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeModel

    vllm_qwen2_moe_classes = [Qwen2MoeForCausalLM, Qwen2MoeModel]
except ImportError as e:
    warnings.warn(f"Could not import Qwen2Moe classes: {e}", stacklevel=2)


@register_model(["VllmQwen2MoeForCausalLM", "VllmQwen2MoeModel"] + vllm_qwen2_moe_classes)
class VllmQwen2MoeParameterMapping(ParameterMapping):
    """Parameter mapping for Qwen2Moe model."""

    def get_mappings(self):
        mapping = [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]
        expert_num = self.config.num_experts
        for expert_id in range(expert_num):
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.gate_proj.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id,
                )
            )
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.up_proj.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id + 1,
                )
            )
            mapping.append(
                (
                    "w2_weight",
                    f"{expert_id}.down_proj.weight",
                    MappingType.FUSED_MOE_W2_SPLIT,
                    expert_id,
                )
            )
        return mapping

    def get_model_info(self):
        info = super().get_model_info()
        info["num_experts"] = self.config.num_experts
        return info


# Qwen3 (dense)
vllm_qwen3_classes = []
try:
    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model

    vllm_qwen3_classes = [Qwen3ForCausalLM, Qwen3Model]
except ImportError as e:
    warnings.warn(f"Could not import Qwen3 classes: {e}", stacklevel=2)


@register_model(["VllmQwen3ForCausalLM", "VllmQwen3Model", "VllmQwen3ForSequenceClassification"] + vllm_qwen3_classes)
class VllmQwen3ParameterMapping(ParameterMapping):
    """Parameter mapping for Qwen3 (dense) model."""

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]


# Qwen3Moe
vllm_qwen3_moe_classes = []
try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM, Qwen3MoeModel

    vllm_qwen3_moe_classes = [Qwen3MoeForCausalLM, Qwen3MoeModel]
except ImportError as e:
    warnings.warn(f"Could not import Qwen3Moe classes: {e}", stacklevel=2)


@register_model(["VllmQwen3MoeForCausalLM", "VllmQwen3MoeModel"] + vllm_qwen3_moe_classes)
class VllmQwen3MoeParameterMapping(ParameterMapping):
    """Parameter mapping for Qwen3Moe model."""

    def get_mappings(self):
        mapping = [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]
        expert_num = self.config.num_experts
        for expert_id in range(expert_num):
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.gate_proj.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id,
                )
            )
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.up_proj.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id + 1,
                )
            )
            mapping.append(
                (
                    "w2_weight",
                    f"{expert_id}.down_proj.weight",
                    MappingType.FUSED_MOE_W2_SPLIT,
                    expert_id,
                )
            )
        return mapping

    def get_model_info(self):
        # NOTE(zym): Qwen3 MoE directly provides `head_dim`,
        # which isn't equal to hidden_size // num_attention_heads.
        # The default get_model_info already handles head_dim via getattr fallback.
        info = super().get_model_info()
        info["num_experts"] = self.config.num_experts
        return info


# Mixtral
vllm_mixtral_classes = []
try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM, MixtralModel

    vllm_mixtral_classes = [MixtralForCausalLM, MixtralModel]
except ImportError as e:
    warnings.warn(f"Could not import Mixtral classes: {e}", stacklevel=2)


@register_model(["VllmMixtralForCausalLM", "VllmMixtralModel"] + vllm_mixtral_classes)
class VllmMixtralParameterMapping(ParameterMapping):
    """Parameter mapping for Mixtral model."""

    def get_mappings(self):
        mapping = [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
        ]
        expert_num = self.config.num_local_experts
        for expert_id in range(expert_num):
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.w1.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id,
                )
            )
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.w3.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id + 1,
                )
            )
            mapping.append(
                (
                    "w2_weight",
                    f"{expert_id}.w2.weight",
                    MappingType.FUSED_MOE_W2_SPLIT,
                    expert_id,
                )
            )
        return mapping

    def get_model_info(self):
        info = super().get_model_info()
        info["num_experts"] = self.config.num_local_experts
        return info


# Llama
vllm_llama_classes = []
try:
    from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel

    vllm_llama_classes = [LlamaForCausalLM, LlamaModel]
except ImportError as e:
    warnings.warn(f"Could not import Llama classes: {e}", stacklevel=2)


@register_model(
    [
        "VllmLlamaForCausalLM",
        "VllmLlamaModel",
        "VllmLLaMAForCausalLM",
        "VllmInternLMForCausalLM",
        "VllmInternLM3ForCausalLM",
        "VllmXverseForCausalLM",
    ]
    + vllm_llama_classes
)
class VllmLlamaParameterMapping(ParameterMapping):
    """Parameter mapping for Llama model."""

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]


# Mistral (implemented as LlamaForCausalLM in vLLM)
@register_model(
    [
        "VllmMistralForCausalLM",
        "VllmMistralModel",
        *(vllm_llama_classes if vllm_llama_classes else []),
    ]
)
class VllmMistralParameterMapping(ParameterMapping):
    """Parameter mapping for Mistral model (uses Llama implementation)."""

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]


# Phi
vllm_phi_classes = []
try:
    from vllm.model_executor.models.phi import PhiForCausalLM

    vllm_phi_classes = [PhiForCausalLM]
except ImportError as e:
    warnings.warn(f"Could not import Phi classes: {e}", stacklevel=2)


@register_model(["VllmPhiForCausalLM"] + vllm_phi_classes)
class VllmPhiParameterMapping(ParameterMapping):
    """Parameter mapping for Phi model."""

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]


# Gemma
vllm_gemma_classes = []
try:
    from vllm.model_executor.models.gemma import GemmaForCausalLM

    vllm_gemma_classes = [GemmaForCausalLM]
except ImportError as e:
    warnings.warn(f"Could not import Gemma classes: {e}", stacklevel=2)


@register_model(["VllmGemmaForCausalLM"] + vllm_gemma_classes)
class VllmGemmaParameterMapping(ParameterMapping):
    """Parameter mapping for Gemma model."""

    def get_mappings(self):
        return [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]


# OLMoE
vllm_olmoe_classes = []
try:
    from vllm.model_executor.models.olmoe import OlmoeForCausalLM

    vllm_olmoe_classes = [OlmoeForCausalLM]
except ImportError as e:
    warnings.warn(f"Could not import OLMoE classes: {e}", stacklevel=2)


@register_model(vllm_olmoe_classes)
class VllmOLMoEParameterMapping(ParameterMapping):
    """Parameter mapping for OLMoE model."""

    def get_mappings(self):
        mapping = [
            ("qkv_proj", "q_proj", MappingType.QKV_SPLIT, 0),
            ("qkv_proj", "k_proj", MappingType.QKV_SPLIT, 1),
            ("qkv_proj", "v_proj", MappingType.QKV_SPLIT, 2),
            ("gate_up_proj", "gate_proj", MappingType.GATE_UP_PROJ_SPLIT, 0),
            ("gate_up_proj", "up_proj", MappingType.GATE_UP_PROJ_SPLIT, 1),
        ]
        expert_num = 64
        for expert_id in range(expert_num):
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.gate_proj.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id,
                )
            )
            mapping.append(
                (
                    "w13_weight",
                    f"{expert_id}.up_proj.weight",
                    MappingType.FUSED_MOE_W13_SPLIT,
                    2 * expert_id + 1,
                )
            )
            mapping.append(
                (
                    "w2_weight",
                    f"{expert_id}.down_proj.weight",
                    MappingType.FUSED_MOE_W2_SPLIT,
                    expert_id,
                )
            )
        return mapping
