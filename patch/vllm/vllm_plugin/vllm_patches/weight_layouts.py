from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from typing import Any

from vllm_patches.weight_layout import ReversibleNameMap, WeightLayoutBuilder, WeightLayoutPlan

logger = logging.getLogger(__name__)

LayoutBuilder = Callable[[Any], WeightLayoutPlan]


def _dense_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
    builder.merged("gate_up_proj", [("gate_proj", None), ("up_proj", None)])
    builder.exclude_substr("rotary_emb.inv_freq")
    return builder.build()


def _qkv_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
    builder.exclude_substr("rotary_emb.inv_freq")
    return builder.build()


def _moe_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
    builder.merged("gate_up_proj", [("gate_proj", None), ("up_proj", None)])
    builder.fused_moe(
        "w13_weight",
        "w2_weight",
        "{i}.gate_proj.weight",
        "{i}.up_proj.weight",
        "{i}.down_proj.weight",
        num_experts=model.config.num_experts,
    )
    builder.exclude_substr("rotary_emb.inv_freq")
    return builder.build()


def _olmoe_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
    builder.fused_moe(
        "w13_weight",
        "w2_weight",
        "{i}.gate_proj.weight",
        "{i}.up_proj.weight",
        "{i}.down_proj.weight",
        num_experts=model.config.num_experts,
    )
    builder.exclude_substr("rotary_emb.inv_freq")
    return builder.build()


def _llama4_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
    builder.merged("gate_up_proj", [("gate_proj", None), ("up_proj", None)])
    builder.fused_moe(
        "w13_weight",
        "w2_weight",
        "{i}.gate_proj.weight",
        "{i}.up_proj.weight",
        "{i}.down_proj.weight",
        num_experts=model.num_experts,
    )
    builder.exclude_substr("rotary_emb.inv_freq")
    return builder.build()


def _deepseek_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.merged("gate_up_proj", [("gate_proj", None), ("up_proj", None)])
    qk_nope_head_dim = getattr(model.config, "qk_nope_head_dim", 0)
    qk_rope_head_dim = getattr(model.config, "qk_rope_head_dim", 0)
    use_mha = model.config.model_type == "deepseek" or all(dim == 0 for dim in (qk_nope_head_dim, qk_rope_head_dim))
    fuse_qkv_a_proj = getattr(model.config, "q_lora_rank", None) is not None
    if use_mha:
        builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
    elif fuse_qkv_a_proj:
        builder.merged(
            "fused_qkv_a_proj",
            [
                ("q_a_proj", model.config.q_lora_rank),
                (
                    "kv_a_proj_with_mqa",
                    model.config.kv_lora_rank + model.config.qk_rope_head_dim,
                ),
            ],
        )
    builder.fused_moe(
        "w13_weight",
        "w2_weight",
        "{i}.gate_proj.weight",
        "{i}.up_proj.weight",
        "{i}.down_proj.weight",
        num_experts=model.config.n_routed_experts,
    )
    if getattr(model.config, "topk_method", None) == "noaux_tc":
        builder.alias("experts.e_score_correction_bias", "gate.e_score_correction_bias")
    builder.exclude_substr("rotary_emb.inv_freq")
    return builder.build()


def _delegate_model_layout(model: Any) -> WeightLayoutPlan:
    return model.model.build_weight_layout()


def _language_model_layout(
    model: Any,
    *,
    exclude_substrs: tuple[str, ...] = (),
) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.mount_module("language_model", model.language_model)
    for substring in exclude_substrs:
        builder.exclude_substr(substring)
    return builder.build()


def _gemma3_mm_layout(model: Any) -> WeightLayoutPlan:
    return _language_model_layout(model, exclude_substrs=("vision_tower",))


def _kimi_k25_layout(model: Any) -> WeightLayoutPlan:
    return _language_model_layout(
        model,
        exclude_substrs=("vision_tower", "mm_projector"),
    )


def _kimi_vl_layout(model: Any) -> WeightLayoutPlan:
    return _language_model_layout(model, exclude_substrs=("vision_tower",))


def _qwen2_vl_layout(model: Any) -> WeightLayoutPlan:
    return _language_model_layout(model, exclude_substrs=("visual",))


def _vl_name_map() -> ReversibleNameMap:
    return ReversibleNameMap.from_prefix_pairs(
        {
            "language_model.model.": "model.language_model.",
            "language_model.lm_head.": "lm_head.",
            "visual.": "model.visual.",
        },
        mode="prefix",
    )


def _qwen_vl_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.name_map(_vl_name_map())
    builder.mount_module("language_model", model.language_model)
    return builder.build()


def _qwen3_5_vl_gen_layout(model: Any) -> WeightLayoutPlan:
    """Layout for Qwen3.5-VL gen/validate workers.

    Visual encoder
        Vision encoder weights (model.visual.*) are synced via NIXL so that
        rollout servers started with load_format=dummy receive correct vision
        weights.  Qwen3.5's ViT uses separate linear_fc1 / linear_fc2 in its
        MLP (unlike Qwen2.5-VL which has a fused mlp.gate_up_proj), so no
        merged() call is needed on the visual_builder.

    MTP exclusion
        vLLM's Qwen3_5ForConditionalGeneration does NOT instantiate MTP
        (mtp.*) sub-modules — those are skipped in load_weights and kept in a
        separate Eagle3 speculative-decoding model class.  Excluding "mtp"
        makes the intent explicit and guards against future accidental
        registration if the class hierarchy changes.

    GDN (Gated DeltaNet) TP sharding
        The language_model sub-module is a Qwen3_5ForCausalLM.  Mounting it
        via mount_module() triggers build_weight_layout() on that sub-module,
        which runs _qwen3_5_layout().  That function handles all GDN-specific
        TP splits (in_proj_qkvz.weight → q/k/v/z, conv1d.weight → k/k/v)
        automatically, so no extra split_param() calls are needed here.
    """
    builder = WeightLayoutBuilder(model)
    builder.name_map(_vl_name_map())
    builder.mount_module("language_model", model.language_model)
    # Mount visual encoder — no MLP fusing (Qwen3.5 ViT uses separate linear_fc1/fc2)
    visual_builder = WeightLayoutBuilder(model.visual)
    builder.mount_plan("visual", visual_builder.build())
    # vLLM has no MTP sub-module in Qwen3_5ForConditionalGeneration;
    # exclude defensively in case class hierarchy changes.
    builder.exclude_substr("mtp")
    return builder.build()


def _qwen2_5_vl_layout(model: Any) -> WeightLayoutPlan:
    builder = WeightLayoutBuilder(model)
    builder.name_map(_vl_name_map())
    builder.mount_module("language_model", model.language_model)
    visual_builder = WeightLayoutBuilder(model.visual)
    visual_builder.merged(
        "mlp.gate_up_proj",
        [("mlp.gate_proj", None), ("mlp.up_proj", None)],
    )
    builder.mount_plan("visual", visual_builder.build())
    return builder.build()


def _qwen3_5_layout(model: Any) -> WeightLayoutPlan:
    """Layout for Qwen3.5 language-model workers (Qwen3_5ForCausalLMBase).

    This covers both stand-alone CausalLM gen workers and the language_model
    sub-module of Qwen3_5ForConditionalGeneration (mounted by
    _qwen3_5_vl_gen_layout via mount_module).

    GDN (Gated DeltaNet) TP sharding
    ---------------------------------
    Each GDN layer has two fused projection tensors that must be split along the
    TP dimension before NIXL transfer:

      in_proj_qkvz.weight  [4·head_dim × hidden]
          → in_proj_qkv.weight_q  [q_size × hidden]   (column-parallel Q)
          → in_proj_qkv.weight_k  [k_size × hidden]   (column-parallel K)
          → in_proj_qkv.weight_v  [v_size × hidden]   (column-parallel V)
          → in_proj_z.weight      [z_size × hidden]   (column-parallel gate)

      conv1d.weight  [(k_size + k_size + v_size) × ...]
          → conv1d.weight_q  [conv_k_size × ...]
          → conv1d.weight_k  [conv_k_size × ...]
          → conv1d.weight_v  [conv_v_size × ...]

    MTP exclusion
    -------------
    vLLM's Qwen3_5ForCausalLMBase does NOT instantiate MTP (mtp.*) layers.
    Those are handled by the separate Qwen3_5ForEagle3MTP class used only in
    Eagle3 speculative-decoding mode.  Excluding "mtp" makes the intent
    explicit and prevents accidental registration if the vLLM class hierarchy
    changes in the future.
    """
    builder = WeightLayoutBuilder(model)
    builder.qkv("qkv_proj", "q_proj", "k_proj", "v_proj")
    builder.merged("gate_up_proj", [("gate_proj", None), ("up_proj", None)])

    num_experts = getattr(model.config, "num_experts", 0)
    if num_experts:
        builder.fused_moe(
            "w13_weight",
            "w2_weight",
            "{i}.gate_proj.weight",
            "{i}.up_proj.weight",
            "{i}.down_proj.weight",
            num_experts=num_experts,
        )

    for module in model.modules():
        if hasattr(module, "in_proj_qkvz") and hasattr(module.in_proj_qkvz, "output_sizes"):
            tp_size = getattr(module.in_proj_qkvz, "tp_size", 1)
            q_size, k_size, v_size, z_size = [size // tp_size for size in module.in_proj_qkvz.output_sizes]
            builder.split_param(
                "in_proj_qkvz.weight",
                (
                    ("in_proj_qkv.weight_q", q_size),
                    ("in_proj_qkv.weight_k", k_size),
                    ("in_proj_qkv.weight_v", v_size),
                    ("in_proj_z.weight", z_size),
                ),
            )
            builder.merged("in_proj_ba", [("in_proj_b", None), ("in_proj_a", None)])
            conv_tp_size = getattr(module.conv1d, "tp_size", 1)
            conv_k_size = module.num_k_heads * module.head_k_dim // conv_tp_size
            conv_v_size = module.num_v_heads * module.head_v_dim // conv_tp_size
            builder.split_param(
                "conv1d.weight",
                (
                    ("conv1d.weight_q", conv_k_size),
                    ("conv1d.weight_k", conv_k_size),
                    ("conv1d.weight_v", conv_v_size),
                ),
            )
            break
    builder.exclude_substr("rotary_emb.inv_freq")
    # vLLM does not instantiate MTP layers in Qwen3_5ForCausalLMBase.
    # Exclude defensively in case of future class hierarchy changes.
    builder.exclude_substr("mtp.")
    return builder.build()


_LAYOUTS: dict[str, LayoutBuilder] = {
    "deepseek_v2.DeepseekV2Model": _deepseek_layout,
    "deepseek_v2.DeepseekV2ForCausalLM": _delegate_model_layout,
    "gemma.GemmaForCausalLM": _dense_layout,
    "gemma2.Gemma2ForCausalLM": _dense_layout,
    "gemma3.Gemma3ForCausalLM": _dense_layout,
    "gemma3_mm.Gemma3ForConditionalGeneration": _gemma3_mm_layout,
    "kimi_k25.KimiK25ForConditionalGeneration": _kimi_k25_layout,
    "kimi_vl.KimiVLForConditionalGeneration": _kimi_vl_layout,
    "llama.LlamaForCausalLM": _dense_layout,
    "llama4.Llama4Model": _llama4_layout,
    "olmoe.OlmoeForCausalLM": _olmoe_layout,
    "phi.PhiForCausalLM": _qkv_layout,
    "qwen2.Qwen2Model": _dense_layout,
    "qwen2.Qwen2ForCausalLM": _dense_layout,
    "qwen2_5_vl.Qwen2_5_VLForConditionalGeneration": _qwen2_5_vl_layout,
    "qwen2_moe.Qwen2MoeModel": _moe_layout,
    "qwen2_moe.Qwen2MoeForCausalLM": _moe_layout,
    "qwen2_vl.Qwen2VLForConditionalGeneration": _qwen2_vl_layout,
    "qwen3.Qwen3ForCausalLM": _dense_layout,
    "qwen3_5.Qwen3_5ForCausalLMBase": _qwen3_5_layout,
    "qwen3_5.Qwen3_5ForConditionalGeneration": _qwen3_5_vl_gen_layout,
    "qwen3_moe.Qwen3MoeModel": _moe_layout,
    "qwen3_moe.Qwen3MoeForCausalLM": _moe_layout,
    "qwen3_vl.Qwen3VLForConditionalGeneration": _qwen_vl_layout,
    "qwen3_vl_moe.Qwen3VLMoeForConditionalGeneration": _qwen_vl_layout,
}


def register_weight_layouts_for_module(module_name: str) -> None:
    """Attach layouts for a vLLM model module that has already been imported."""
    module = sys.modules.get(module_name)
    if module is None:
        return

    prefix = "vllm.model_executor.models."
    relative_module = module_name.removeprefix(prefix)
    for target, layout_builder in _LAYOUTS.items():
        target_module, class_name = target.rsplit(".", 1)
        if target_module != relative_module:
            continue
        model_class = getattr(module, class_name, None)
        if model_class is None:
            logger.warning("Could not register weight layout for missing class %s", target)
            continue

        existing = getattr(model_class, "build_weight_layout", None)
        if existing is not None and existing is not layout_builder:
            logger.warning("Replacing existing weight layout for %s", target)
        model_class.build_weight_layout = layout_builder


def register_weight_layouts() -> None:
    """Attach layouts to supported vLLM model modules already in memory."""
    modules = {f"vllm.model_executor.models.{target.rsplit('.', 1)[0]}" for target in _LAYOUTS}
    for module_name in modules:
        register_weight_layouts_for_module(module_name)


def install_weight_layout_registry_hook() -> None:
    """Patch vLLM's lazy model loader so layouts are attached after import."""
    registry = importlib.import_module("vllm.model_executor.models.registry")
    lazy_model = getattr(registry, "_LazyRegisteredModel", None)
    if lazy_model is None:
        logger.warning(
            "vLLM has no _LazyRegisteredModel; layouts will be attached on supports_weight_layout() checks instead"
        )
        return
    if getattr(lazy_model, "_psrl_weight_layout_hook", False):
        return

    original_load_model_cls = lazy_model.load_model_cls

    def load_model_cls(self: Any) -> type:
        model_class = original_load_model_cls(self)
        register_weight_layouts_for_module(model_class.__module__)
        return model_class

    lazy_model.load_model_cls = load_model_cls
    lazy_model._psrl_weight_layout_hook = True
