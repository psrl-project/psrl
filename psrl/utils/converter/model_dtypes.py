import torch

# Fallbacks for architectures whose Transformers implementation does not yet
# expose every strict-fp32 tensor through _keep_in_fp32_modules_strict.
FP32_PATTERNS: dict[str, tuple[str, ...]] = {
    "Qwen3_5": ("A_log",),
}


def fix_meta_model_dtypes(meta_model: torch.nn.Module) -> int:
    """Restore architecture-constrained fp32 tensors on a meta model."""
    keep_in_fp32_strict: set[str] = set()
    keep_in_fp32: set[str] = set()

    for module in meta_model.modules():
        if patterns := getattr(module, "_keep_in_fp32_modules_strict", None):
            keep_in_fp32_strict.update(patterns)
        if patterns := getattr(module, "_keep_in_fp32_modules", None):
            keep_in_fp32.update(patterns)
        class_name = type(module).__name__
        for model_name, patterns in FP32_PATTERNS.items():
            if model_name in class_name:
                keep_in_fp32_strict.update(patterns)

    if not keep_in_fp32_strict and not keep_in_fp32:
        return 0

    fixed_count = 0
    for param_name, param in meta_model.named_parameters():
        strict_match = any(pattern in param_name for pattern in keep_in_fp32_strict)
        fp16_match = param.dtype == torch.float16 and any(pattern in param_name for pattern in keep_in_fp32)
        if param.dtype in (torch.float16, torch.bfloat16) and (strict_match or fp16_match):
            param.data = torch.empty(param.shape, dtype=torch.float32, device=param.device)
            fixed_count += 1

    for buffer_name, buffer in meta_model.named_buffers():
        if buffer is None:
            continue
        strict_match = any(pattern in buffer_name for pattern in keep_in_fp32_strict)
        fp16_match = buffer.dtype == torch.float16 and any(pattern in buffer_name for pattern in keep_in_fp32)
        if buffer.dtype not in (torch.float16, torch.bfloat16) or not (strict_match or fp16_match):
            continue

        parent_path, separator, attr_name = buffer_name.rpartition(".")
        parent_module = meta_model.get_submodule(parent_path) if separator else meta_model
        parent_module.register_buffer(
            attr_name,
            torch.empty(buffer.shape, dtype=torch.float32, device=buffer.device),
            persistent=attr_name not in parent_module._non_persistent_buffers_set,
        )
        fixed_count += 1

    return fixed_count
