import logging
from typing import Any

import numpy as np
import torch
from verl.protocol import DataProto


def log_tensor(
    tensor: torch.Tensor,
    psrl_logger: logging.Logger,
    log_prefix: str,
    name: str = "tensor",
    max_elements: int = 8,
    level: int = logging.INFO,
    extra_fields: dict[str, Any] | None = None,
    **kwargs: Any,
):
    assert isinstance(tensor, torch.Tensor), f"{tensor} is not a torch.Tensor, got {type(tensor)}"

    def _safe_value(expr):
        try:
            value = expr()
            return value.item() if hasattr(value, "item") else value
        except Exception:
            return None

    def _format_stat(name: str, value):
        if value is None:
            return f"{name}=N/A"
        try:
            return f"{name}={float(value):.6g}"
        except Exception:
            return f"{name}={value}"

    try:
        merged_extra_fields: dict[str, Any] = {}
        if extra_fields is not None:
            merged_extra_fields.update(extra_fields)
        merged_extra_fields.update(kwargs)
        extra_text = ", ".join(f"{k}={v!r}" for k, v in merged_extra_fields.items()) if merged_extra_fields else ""
        if extra_text:
            extra_text = f", {extra_text}"

        tensor_detached = tensor.detach()
        numel = tensor_detached.numel()
        if numel == 0:
            psrl_logger.log(
                level,
                f"[{log_prefix}]: {name}{extra_text}, shape={tuple(tensor_detached.shape)}, "
                f"dtype={tensor_detached.dtype}, numel=0 (empty tensor)",
            )
            return

        tensor_float = tensor_detached.float()
        tensor_flat = tensor_float.flatten()
        tensor_head = tensor_flat[:max_elements]
        try:
            first_vals = tensor_head.tolist()
        except Exception:
            first_vals = str(tensor_head)

        t_min = _safe_value(lambda: tensor_float.min())
        t_max = _safe_value(lambda: tensor_float.max())
        t_mean = _safe_value(lambda: tensor_float.mean())
        t_sum = _safe_value(lambda: tensor_float.sum())
        t_norm = _safe_value(lambda: tensor_float.norm())
        shape = _safe_value(lambda: tuple(tensor_detached.shape))
        dtype = _safe_value(lambda: tensor_detached.dtype)
        device = _safe_value(lambda: tensor_detached.device)

        psrl_logger.log(
            level,
            f"[{log_prefix}]: {name}{extra_text}, shape={shape}, dtype={dtype}, device={device}, "
            f"{_format_stat('min', t_min)}, {_format_stat('max', t_max)}, "
            f"{_format_stat('mean', t_mean)}, {_format_stat('sum', t_sum)}, "
            f"{_format_stat('norm', t_norm)}, "
            f"first_{max_elements}_vals={first_vals}",
        )
    except Exception as e:
        psrl_logger.warning(f"[{log_prefix}]: failed to log {name}: {e}")


def log_data_protocol(
    inputs: DataProto,
    psrl_logger: logging.Logger,
    log_prefix: str,
    level: int = logging.INFO,
):
    input_tensor_types = {}
    if inputs.batch is not None:
        for k, v in inputs.batch.items():
            input_tensor_types[k] = f"torch.Tensor with dtype {v.dtype} and shape {v.shape}"
    input_non_tensor_types = {}
    for k, v in inputs.non_tensor_batch.items():
        if isinstance(v, torch.Tensor):
            input_non_tensor_types[k] = f"torch.Tensor with dtype {v.dtype} and shape {v.shape}"
        elif isinstance(v, np.ndarray):
            input_non_tensor_types[k] = f"np.ndarray with dtype {v.dtype} and shape {v.shape}"
        elif isinstance(v, list):
            input_non_tensor_types[k] = f"list with length {len(v)}"
        else:
            input_non_tensor_types[k] = f"single value with {type(v)}"

    psrl_logger.log(level, f"[{log_prefix}]: tensor batch have {input_tensor_types}")
    psrl_logger.log(level, f"[{log_prefix}]: non tensor batch have {input_non_tensor_types}")
