"""
Compare segmented vLLM logprobs with PSRL-style trainer recomputation.

The script performs inference only. It first generates interrupt/reprefill
sequences with vLLM and records historical decode logprobs (concat), online
reprefill replacement logprobs, and a final full-prefix prefill score. It then
shuts vLLM down and loads the trainer checkpoint into a single-rank FSDP2 model.
Trainer recomputation calls verl's DataParallelPPOActor.compute_log_prob with
PSRL-style padded tensors and metadata. No optimizer is created and no model
parameter is updated. A continuous-decode baseline can be enabled explicitly,
but is excluded by
default because it need not produce the same token IDs.

Example:

  source env/env_311.sh
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
      scripts/test_vllm_reprefill_vs_fsdp.py \
      --model models/Qwen2.5-7B \
      --response-length 1024 \
      --segment-lengths 32,64,128,256 \
      --output-dir /tmp/qwen2.5-7b-reprefill-vs-psrl

For the mode5-shaped 1K / 8-interrupt run:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python \
      scripts/test_vllm_reprefill_vs_fsdp.py \
      --model models/Qwen2.5-7B --response-length 1024 \
      --reprefill-count 8 --match-mode5 \
      --output-dir /tmp/qwen2.5-7b-reprefill-8x
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import hashlib
import json
import logging
import math
import os
import statistics
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from test_vllm_reprefill_logprobs import (
    _collect_request,
    _parse_positive_int_list,
    _parse_token_ids,
)

psrl_logger = logging.getLogger(__file__)

DEFAULT_PROMPTS = [
    "Explain why numerical reproducibility matters in large language model inference.",
    "Write a concise technical discussion of floating-point error accumulation.",
    "Describe how KV-cache execution paths can affect autoregressive generation.",
    "Analyze why teacher-forced scoring can differ from online token decoding.",
]


@dataclass
class SegmentedCase:
    """
    Store one segmented rollout before trainer recomputation.
    """

    case_id: str
    prompt_id: int
    prompt: str | None
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    logprob_concat: list[float]
    # Backward-compatible name for the online, actual interrupt/re-prefill path.
    logprob_replace: list[float]
    # Diagnostic path where a final full-prefix prefill overwrites every token.
    logprob_replace_final: list[float]
    replacement_source: list[str]
    final_replacement_source: list[str]
    final_prefill_available: list[bool]
    reprefill_history: list[dict[str, Any]]
    segment_id: list[int]
    boundaries: list[int]
    segments: list[dict[str, Any]]
    final_prefill: dict[str, Any]


@dataclass
class ContinuousCase:
    """
    Store one continuous rollout before trainer recomputation.
    """

    case_id: str
    prompt_id: int
    prompt: str | None
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    logprob_continuous: list[float]


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _parse_float_list(value: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be comma-separated floating-point numbers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def _torch_dtype(name: str) -> Any:
    import torch

    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported trainer dtype: {name!r}") from exc


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _finite_ratio(delta: float) -> float:
    if delta > 709.0:
        return float("inf")
    if delta < -745.0:
        return 0.0
    return math.exp(delta)


def mismatch_statistics(
    trainer: Iterable[float],
    rollout: Iterable[float],
    *,
    outlier_thresholds: list[float],
    ratio_low: float,
    ratio_high: float,
) -> dict[str, Any]:
    """
    Calculate signed, absolute, ratio, and outlier mismatch statistics.

    Args:
        trainer (Iterable[float]): Trainer reference logprobs.
        rollout (Iterable[float]): Rollout logprobs aligned to trainer.
        outlier_thresholds (list[float]): Absolute-delta thresholds.
        ratio_low (float): Lower hypothetical ratio threshold.
        ratio_high (float): Upper hypothetical ratio threshold.

    Returns:
        dict[str, Any]: Summary statistics for the aligned token arrays.
    """
    trainer_values = [float(value) for value in trainer]
    rollout_values = [float(value) for value in rollout]
    if len(trainer_values) != len(rollout_values):
        raise ValueError(
            "trainer and rollout logprobs must have the same length, "
            f"got {len(trainer_values)} and {len(rollout_values)}"
        )
    if not trainer_values:
        raise ValueError("cannot calculate mismatch statistics for empty arrays")

    delta = [ref - candidate for ref, candidate in zip(trainer_values, rollout_values, strict=True)]
    absolute = [abs(value) for value in delta]
    ratio = [_finite_ratio(value) for value in delta]
    finite_ratio = [value for value in ratio if math.isfinite(value)]
    masked = [value < ratio_low or value > ratio_high for value in ratio]

    return {
        "num_tokens": len(delta),
        "signed_mean": statistics.fmean(delta),
        "signed_median": statistics.median(delta),
        "mean_absolute_error": statistics.fmean(absolute),
        "median_absolute_error": statistics.median(absolute),
        "std": statistics.pstdev(delta),
        "p90_absolute_error": _percentile(absolute, 90.0),
        "p95_absolute_error": _percentile(absolute, 95.0),
        "p99_absolute_error": _percentile(absolute, 99.0),
        "p99_9_absolute_error": _percentile(absolute, 99.9),
        "max_absolute_error": max(absolute),
        "ratio": {
            "mean": statistics.fmean(finite_ratio) if finite_ratio else None,
            "median": statistics.median(finite_ratio) if finite_ratio else None,
            "p95": _percentile(finite_ratio, 95.0),
            "p99": _percentile(finite_ratio, 99.0),
            "max": max(ratio),
            "hypothetical_low": ratio_low,
            "hypothetical_high": ratio_high,
            "hypothetical_masked_count": sum(masked),
            "hypothetical_masked_fraction": sum(masked) / len(masked),
        },
        "absolute_outliers": {
            str(threshold): {
                "count": sum(value > threshold for value in absolute),
                "fraction": sum(value > threshold for value in absolute) / len(absolute),
            }
            for threshold in outlier_thresholds
        },
    }


def _nearest_boundary_distance(position: int, boundaries: list[int]) -> int | None:
    if not boundaries:
        return None
    return min((position - boundary for boundary in boundaries), key=lambda value: (abs(value), value))


def _boundary_category(distance: int | None) -> str:
    if distance is None:
        return "far"
    if -8 <= distance <= -1:
        return "before_-8_-1"
    if distance == 0:
        return "boundary_0"
    if 1 <= distance <= 8:
        return "after_1_8"
    if 9 <= distance <= 32:
        return "after_9_32"
    return "far"


def token_boundary_metadata(
    position: int,
    boundaries: list[int],
) -> dict[str, Any]:
    """
    Calculate boundary-relative metadata for one response token.

    Args:
        position (int): Zero-based response-token position.
        boundaries (list[int]): Segment-start positions after reprefill.

    Returns:
        dict[str, Any]: Previous/next distances and nearest-distance category.
    """
    previous = [boundary for boundary in boundaries if boundary <= position]
    following = [boundary for boundary in boundaries if boundary > position]
    nearest = _nearest_boundary_distance(position, boundaries)
    return {
        "is_segment_boundary": position in boundaries,
        "distance_to_previous_boundary": (position - max(previous) if previous else None),
        "distance_to_next_boundary": min(following) - position if following else None,
        "nearest_boundary_signed_distance": nearest,
        "boundary_category": _boundary_category(nearest),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_snapshot(model_path: str) -> dict[str, Any]:
    """
    Capture a lightweight checkpoint identity and mutation snapshot.

    Args:
        model_path (str): Local model checkpoint directory.

    Returns:
        dict[str, Any]: File metadata and hashes for checkpoint index/config files.
    """
    root = Path(model_path).resolve()
    if not root.is_dir():
        return {"path": model_path, "local_directory": False}
    patterns = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "*.safetensors",
    )
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(root.glob(pattern))
    files = []
    for path in sorted(paths):
        stat = path.stat()
        item: dict[str, Any] = {
            "name": path.relative_to(root).as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if path.name.endswith(".json"):
            item["sha256"] = _sha256(path)
        files.append(item)
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "path": str(root),
        "local_directory": True,
        "identity_sha256": identity,
        "files": files,
    }


def _load_prompts(args: argparse.Namespace) -> list[tuple[str | None, list[int] | None]]:
    if args.prompt_token_ids:
        return [(None, _parse_token_ids(value)) for value in args.prompt_token_ids]

    prompts = list(args.prompt or [])
    if args.prompts_file:
        path = Path(args.prompts_file)
        if path.suffix == ".jsonl":
            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if isinstance(item, str):
                        prompts.append(item)
                    else:
                        prompts.append(str(item[args.prompt_key]))
        else:
            prompts.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    return [(prompt, None) for prompt in prompts]


def _sampling_params(args: argparse.Namespace, max_tokens: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=(args.sampling_temperature if args.sampling_temperature is not None else args.temperature),
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=max_tokens,
        ignore_eos=True,
        logprobs=args.vllm_logprobs,
        prompt_logprobs=args.vllm_prompt_logprobs,
    )


def _response_prompt_records(
    prompt_records: list[dict[str, Any]],
    *,
    prompt_length: int,
    response_ids: list[int],
) -> tuple[list[float | None], list[bool]]:
    """Align vLLM prompt logprobs to the already generated response tokens."""
    if not prompt_records:
        # Some vLLM backends/configurations may omit prompt logprobs even when
        # requested. Preserve the experiment and make the missing values
        # explicit so the online replacement buffer remains the fallback.
        return [None] * len(response_ids), [False] * len(response_ids)
    response_records = prompt_records[prompt_length:]
    if len(response_records) != len(response_ids):
        raise RuntimeError(
            "reprefill prompt logprob length mismatch: "
            f"got {len(response_records)}, expected {len(response_ids)}"
        )
    values: list[float | None] = []
    available: list[bool] = []
    for position, (record, token_id) in enumerate(zip(response_records, response_ids, strict=True)):
        if int(record["token_id"]) != int(token_id):
            raise RuntimeError(
                f"reprefill token mismatch at response position {position}: "
                f"{record['token_id']} != {token_id}"
            )
        value = record.get("logprob")
        values.append(float(value) if value is not None else None)
        available.append(value is not None and math.isfinite(float(value)))
    return values, available


def _geometric_rs_statistics(
    trainer: Iterable[float],
    rollout: Iterable[float],
    *,
    upper_threshold: float,
) -> dict[str, Any]:
    """Compute mode5's geometric sequence rejection statistics offline."""
    trainer_values = [float(value) for value in trainer]
    rollout_values = [float(value) for value in rollout]
    if len(trainer_values) != len(rollout_values) or not trainer_values:
        raise ValueError("geometric RS statistics require equally sized non-empty sequences")
    lower_threshold = 1.0 / upper_threshold
    log_ratio = [ref - candidate for ref, candidate in zip(trainer_values, rollout_values, strict=True)]
    geometric_ratio = math.exp(max(-20.0, min(20.0, statistics.fmean(log_ratio))))
    rejected_low = geometric_ratio < lower_threshold
    rejected_high = geometric_ratio > upper_threshold
    rejected = rejected_low or rejected_high
    return {
        "geometric_ratio": geometric_ratio,
        "threshold_upper": upper_threshold,
        "threshold_lower": lower_threshold,
        "sequence_rejected": rejected,
        "sequence_fraction_low": float(rejected_low),
        "sequence_fraction_high": float(rejected_high),
        # For geometric RS all valid tokens in a rejected sequence are masked.
        "token_masked_fraction": float(rejected),
        "response_tokens": len(log_ratio),
    }


def _offpolicy_statistics(
    trainer: Iterable[float],
    rollout: Iterable[float],
) -> dict[str, float]:
    """Mirror rollout_corr's scalar diagnostics for one fixed sequence."""
    trainer_values = [float(value) for value in trainer]
    rollout_values = [float(value) for value in rollout]
    if len(trainer_values) != len(rollout_values) or not trainer_values:
        raise ValueError("off-policy statistics require equally sized non-empty sequences")
    log_ratio = [ref - candidate for ref, candidate in zip(trainer_values, rollout_values, strict=True)]
    mean_training_log_ppl = -statistics.fmean(trainer_values)
    mean_rollout_log_ppl = -statistics.fmean(rollout_values)
    safe_log_ratio = [max(-20.0, min(20.0, value)) for value in log_ratio]
    sequence_log_ratio = sum(log_ratio)
    safe_sequence_log_ratio = max(-20.0, min(20.0, sequence_log_ratio))
    return {
        "training_log_ppl": mean_training_log_ppl,
        "training_ppl": math.exp(mean_training_log_ppl),
        "rollout_log_ppl": mean_rollout_log_ppl,
        "rollout_ppl": math.exp(mean_rollout_log_ppl),
        # verl reports mean(log p_rollout) - mean(log p_training), which is
        # equivalent to training_log_ppl - rollout_log_ppl.
        "log_ppl_diff": mean_training_log_ppl - mean_rollout_log_ppl,
        "log_ppl_abs_diff": abs(mean_training_log_ppl - mean_rollout_log_ppl),
        "kl": statistics.fmean(rollout_values[index] - trainer_values[index] for index in range(len(log_ratio))),
        "k3_kl": statistics.fmean(
            math.exp(value) - value - 1.0 for value in safe_log_ratio
        ),
        "ppl_ratio": math.exp(mean_training_log_ppl - mean_rollout_log_ppl),
        "chi2_token": statistics.fmean(math.exp(2.0 * value) for value in safe_log_ratio) - 1.0,
        "chi2_seq": math.exp(2.0 * safe_sequence_log_ratio) - 1.0,
    }


def _offpolicy_batch_statistics(
    pairs: Iterable[tuple[Iterable[float], Iterable[float]]],
) -> dict[str, float]:
    """Aggregate off-policy diagnostics with the same token/sequence reductions as verl."""
    pair_values = [
        ([float(value) for value in trainer], [float(value) for value in rollout])
        for trainer, rollout in pairs
    ]
    if not pair_values:
        raise ValueError("off-policy batch statistics require at least one sequence")
    for trainer_values, rollout_values in pair_values:
        if len(trainer_values) != len(rollout_values) or not trainer_values:
            raise ValueError("off-policy batch pairs must be equally sized and non-empty")

    token_pairs = [
        (trainer_value, rollout_value)
        for trainer_values, rollout_values in pair_values
        for trainer_value, rollout_value in zip(trainer_values, rollout_values, strict=True)
    ]
    log_ratios = [trainer_value - rollout_value for trainer_value, rollout_value in token_pairs]
    safe_log_ratios = [max(-20.0, min(20.0, value)) for value in log_ratios]
    sequence_stats = [_offpolicy_statistics(trainer, rollout) for trainer, rollout in pair_values]
    sequence_log_ratios = [
        sum(
            trainer_value - rollout_value
            for trainer_value, rollout_value in zip(trainer, rollout, strict=True)
        )
        for trainer, rollout in pair_values
    ]
    sequence_log_ppl_diffs = [stat["log_ppl_diff"] for stat in sequence_stats]
    return {
        "training_ppl": statistics.fmean(stat["training_ppl"] for stat in sequence_stats),
        "training_log_ppl": statistics.fmean(stat["training_log_ppl"] for stat in sequence_stats),
        "rollout_ppl": statistics.fmean(stat["rollout_ppl"] for stat in sequence_stats),
        "rollout_log_ppl": statistics.fmean(stat["rollout_log_ppl"] for stat in sequence_stats),
        "log_ppl_diff": statistics.fmean(sequence_log_ppl_diffs),
        "log_ppl_abs_diff": statistics.fmean(abs(value) for value in sequence_log_ppl_diffs),
        "log_ppl_diff_max": max(sequence_log_ppl_diffs),
        "log_ppl_diff_min": min(sequence_log_ppl_diffs),
        "ppl_ratio": statistics.fmean(math.exp(value) for value in sequence_log_ppl_diffs),
        "kl": statistics.fmean(rollout_value - trainer_value for trainer_value, rollout_value in token_pairs),
        "k3_kl": statistics.fmean(math.exp(value) - value - 1.0 for value in safe_log_ratios),
        "chi2_token": statistics.fmean(math.exp(2.0 * value) for value in safe_log_ratios) - 1.0,
        "chi2_seq": statistics.fmean(
            math.exp(2.0 * max(-20.0, min(20.0, value))) for value in sequence_log_ratios
        )
        - 1.0,
    }


def _geometric_rs_batch_statistics(
    pairs: Iterable[tuple[Iterable[float], Iterable[float]]],
    *,
    upper_threshold: float,
) -> dict[str, Any]:
    """Aggregate geometric RS statistics, preserving sequence and token denominators."""
    sequence_stats = [
        _geometric_rs_statistics(trainer, rollout, upper_threshold=upper_threshold)
        for trainer, rollout in pairs
    ]
    total_tokens = sum(int(stat["response_tokens"]) for stat in sequence_stats)
    rejected_tokens = sum(
        int(stat["response_tokens"])
        for stat in sequence_stats
        if stat["sequence_rejected"]
    )
    geometric_ratio_mean = statistics.fmean(stat["geometric_ratio"] for stat in sequence_stats)
    geometric_ratio_min = min(stat["geometric_ratio"] for stat in sequence_stats)
    geometric_ratio_max = max(stat["geometric_ratio"] for stat in sequence_stats)
    sequence_fraction_high = statistics.fmean(stat["sequence_fraction_high"] for stat in sequence_stats)
    sequence_fraction_low = statistics.fmean(stat["sequence_fraction_low"] for stat in sequence_stats)
    sequence_masked_fraction = statistics.fmean(
        float(stat["sequence_rejected"]) for stat in sequence_stats
    )
    token_masked_fraction = rejected_tokens / total_tokens if total_tokens else 0.0
    return {
        "threshold_upper": upper_threshold,
        "threshold_lower": 1.0 / upper_threshold,
        "geometric_ratio_mean": geometric_ratio_mean,
        "geometric_ratio_min": geometric_ratio_min,
        "geometric_ratio_max": geometric_ratio_max,
        "sequence_fraction_high": sequence_fraction_high,
        "sequence_fraction_low": sequence_fraction_low,
        "sequence_masked_fraction": sequence_masked_fraction,
        "token_masked_fraction": token_masked_fraction,
        # Names match third_party/verl's rollout_corr metrics for direct log
        # comparison with a mode5 run.
        "rollout_rs_mean": geometric_ratio_mean,
        "rollout_rs_min": geometric_ratio_min,
        "rollout_rs_max": geometric_ratio_max,
        "rollout_rs_ratio_fraction_high": sequence_fraction_high,
        "rollout_rs_ratio_fraction_low": sequence_fraction_low,
        "rollout_rs_masked_fraction": token_masked_fraction,
        "rollout_rs_seq_masked_fraction": sequence_masked_fraction,
        "num_sequences": len(sequence_stats),
        "num_tokens": total_tokens,
    }


async def _build_vllm_engine(args: argparse.Namespace) -> Any:
    import inspect

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    requested_logprobs = (args.vllm_logprobs, args.vllm_prompt_logprobs)
    max_logprobs = -1 if -1 in requested_logprobs else max(*requested_logprobs, 1)
    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.vllm_dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.vllm_max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_logprobs": max_logprobs,
        "logprobs_mode": args.logprobs_mode,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enforce_eager": args.enforce_eager,
        "attention_backend": args.vllm_attention_backend,
        "stream_interval": 1,
        "disable_log_stats": True,
        "enable_log_requests": False,
    }
    accepted = set(inspect.signature(AsyncEngineArgs).parameters)
    kwargs = {key: value for key, value in kwargs.items() if key in accepted and value is not None}
    return AsyncLLM.from_engine_args(AsyncEngineArgs(**kwargs))


def _vllm_metadata(engine: Any, args: argparse.Namespace) -> dict[str, Any]:
    config = engine.vllm_config
    model_config = config.model_config
    cache_config = config.cache_config
    scheduler_config = config.scheduler_config
    compilation_config = config.compilation_config
    attention_config = getattr(config, "attention_config", None)
    return {
        "version": __import__("vllm").__version__,
        "dtype": str(model_config.dtype),
        "kv_cache_dtype": str(cache_config.cache_dtype),
        "tensor_parallel_size": config.parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": config.parallel_config.pipeline_parallel_size,
        "attention_backend": str(getattr(attention_config, "backend", None)),
        "chunked_prefill": scheduler_config.enable_chunked_prefill,
        "prefix_caching": cache_config.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "cuda_graph_enabled": not args.enforce_eager,
        "max_model_len": model_config.max_model_len,
        "max_num_seqs": scheduler_config.max_num_seqs,
        "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "gpu_memory_utilization": cache_config.gpu_memory_utilization,
        "logprobs_mode": args.logprobs_mode,
        "compilation_config": str(compilation_config),
        "temperature": args.temperature,
        "sampling_temperature": (
            args.sampling_temperature if args.sampling_temperature is not None else args.temperature
        ),
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "returned_logprobs": args.vllm_logprobs,
        "returned_prompt_logprobs": args.vllm_prompt_logprobs,
        "mode5_preset_enabled": args.match_mode5,
        "mode5_rollout_is": "token" if args.match_mode5 else None,
        "mode5_rollout_is_threshold": 2.0 if args.match_mode5 else None,
        "mode5_rollout_rs": "geometric" if args.match_mode5 else None,
        "mode5_rollout_rs_threshold": args.mode5_rs_threshold if args.match_mode5 else None,
        "mode5_rollout_rs_threshold_lower": (
            1.0 / args.mode5_rs_threshold if args.match_mode5 else None
        ),
        "mode5_rollout_token_veto_threshold": None,
    }


async def _generate_continuous(
    engine: Any,
    args: argparse.Namespace,
    *,
    prompt_id: int,
    prompt: str | None,
    prompt_token_ids: list[int],
) -> ContinuousCase:
    records, _, aborted = await _collect_request(
        engine,
        prompt_token_ids=prompt_token_ids,
        sampling_params=_sampling_params(args, args.response_length),
        request_id=f"continuous-p{prompt_id}",
        abort_after=None,
        timeout_s=args.timeout_s,
    )
    if aborted or len(records) != args.response_length:
        raise RuntimeError(
            f"continuous prompt {prompt_id} returned {len(records)} tokens, expected {args.response_length}"
        )
    return ContinuousCase(
        case_id=f"continuous-p{prompt_id}",
        prompt_id=prompt_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        response_token_ids=[int(record["token_id"]) for record in records],
        logprob_continuous=[float(record["logprob"]) for record in records],
    )


async def _generate_segmented(
    engine: Any,
    args: argparse.Namespace,
    *,
    prompt_id: int,
    prompt: str | None,
    prompt_token_ids: list[int],
    segment_length: int,
) -> SegmentedCase:
    response_ids: list[int] = []
    concat: list[float] = []
    replacement: list[float] = []
    source: list[str] = []
    reprefill_history: list[dict[str, Any]] = []
    segment_ids: list[int] = []
    boundaries: list[int] = []
    segments: list[dict[str, Any]] = []
    segment_index = 0

    while len(response_ids) < args.response_length:
        consumed = len(response_ids)
        if segment_index > 0:
            boundaries.append(consumed)
        new_tokens = min(segment_length, args.response_length - consumed)
        request_prompt = prompt_token_ids + response_ids
        records, prompt_records, aborted = await _collect_request(
            engine,
            prompt_token_ids=request_prompt,
            sampling_params=_sampling_params(args, args.response_length - consumed),
            request_id=f"seg-p{prompt_id}-n{segment_length}-s{segment_index}",
            abort_after=new_tokens if consumed + new_tokens < args.response_length else None,
            timeout_s=args.timeout_s,
        )
        should_abort = consumed + new_tokens < args.response_length
        if should_abort and not aborted:
            raise RuntimeError(
                f"segment {segment_index} for prompt {prompt_id}, length {segment_length} "
                "did not finish with vLLM abort"
            )
        if len(records) < new_tokens:
            raise RuntimeError(f"segment {segment_index} returned {len(records)} tokens, expected {new_tokens}")
        records = records[:new_tokens]

        replacement_count = 0
        replacement_available_count = 0
        if segment_index > 0:
            values, available = _response_prompt_records(
                prompt_records,
                prompt_length=len(prompt_token_ids),
                response_ids=response_ids,
            )
            for position, (value, is_available) in enumerate(zip(values, available, strict=True)):
                if is_available:
                    assert value is not None
                    replacement[position] = value
                    source[position] = f"reprefill_{segment_index}"
                    replacement_available_count += 1
                replacement_count += 1
            reprefill_history.append(
                {
                    "reprefill_index": segment_index,
                    "prefix_length": len(request_prompt),
                    "response_length_scored": consumed,
                    "response_logprobs": values,
                    "available": available,
                    "available_fraction": replacement_available_count / consumed if consumed else 1.0,
                }
            )

        response_ids.extend(int(record["token_id"]) for record in records)
        decoded_logprobs = [float(record["logprob"]) for record in records]
        concat.extend(decoded_logprobs)
        replacement.extend(decoded_logprobs)
        source.extend("decode" for _ in records)
        segment_ids.extend(segment_index for _ in records)
        segments.append(
            {
                "segment_id": segment_index,
                "response_start": consumed,
                "response_end_exclusive": consumed + len(records),
                "prompt_length": len(request_prompt),
                "decoded_tokens": len(records),
                "reprefill_index": segment_index if segment_index > 0 else None,
                "replacement_count": replacement_count,
                "replacement_available_count": replacement_available_count,
                "aborted": aborted,
            }
        )
        segment_index += 1

    # The final segment has no subsequent interrupt in the online path. Score
    # the complete fixed prefix once more so the report can separate that
    # actual online behavior from an all-prefill diagnostic variant.
    final_prefill_values: list[float | None] = [None] * len(response_ids)
    final_prefill_available = [False] * len(response_ids)
    final_replacement = list(replacement)
    final_source = list(source)
    final_prefill_record: dict[str, Any] = {
        "enabled": bool(getattr(args, "final_prefix_score", False)),
        "prefix_length": len(prompt_token_ids) + len(response_ids),
    }
    if getattr(args, "final_prefix_score", False):
        final_prompt = prompt_token_ids + response_ids
        _, final_prompt_records, final_aborted = await _collect_request(
            engine,
            prompt_token_ids=final_prompt,
            sampling_params=_sampling_params(args, 1),
            request_id=f"final-prefix-p{prompt_id}-n{segment_length}",
            abort_after=None,
            timeout_s=args.timeout_s,
        )
        if final_aborted:
            raise RuntimeError("final full-prefix score unexpectedly aborted")
        final_prefill_values, final_prefill_available = _response_prompt_records(
            final_prompt_records,
            prompt_length=len(prompt_token_ids),
            response_ids=response_ids,
        )
        for position, (value, is_available) in enumerate(
            zip(final_prefill_values, final_prefill_available, strict=True)
        ):
            if is_available:
                assert value is not None
                final_replacement[position] = value
                final_source[position] = "final_prefill"
        final_prefill_record.update(
            {
                "available_count": sum(final_prefill_available),
                "available_fraction": (
                    sum(final_prefill_available) / len(final_prefill_available)
                    if final_prefill_available
                    else 1.0
                ),
            }
        )
    else:
        final_prefill_record.update({"available_count": 0, "available_fraction": 0.0})

    return SegmentedCase(
        case_id=f"segmented-p{prompt_id}-n{segment_length}",
        prompt_id=prompt_id,
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        response_token_ids=response_ids,
        logprob_concat=concat,
        logprob_replace=replacement,
        logprob_replace_final=final_replacement,
        replacement_source=source,
        final_replacement_source=final_source,
        final_prefill_available=final_prefill_available,
        reprefill_history=reprefill_history,
        segment_id=segment_ids,
        boundaries=boundaries,
        segments=segments,
        final_prefill=final_prefill_record,
    )


async def _run_vllm_phase(
    args: argparse.Namespace,
    prompt_inputs: list[tuple[str | None, list[int] | None]],
) -> tuple[list[ContinuousCase], list[SegmentedCase], dict[str, Any]]:
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
    )
    resolved_prompts = []
    for prompt, token_ids in prompt_inputs:
        if token_ids is None:
            try:
                token_ids = tokenizer.encode(prompt, add_special_tokens=args.add_special_tokens)
            except TypeError:
                token_ids = tokenizer.encode(prompt)
        if len(token_ids) + args.response_length > args.max_model_len:
            raise ValueError(
                f"prompt length {len(token_ids)} plus response length "
                f"{args.response_length} exceeds max model length {args.max_model_len}"
            )
        resolved_prompts.append((prompt, [int(token_id) for token_id in token_ids]))

    engine = await _build_vllm_engine(args)
    metadata = _vllm_metadata(engine, args)
    continuous_cases: list[ContinuousCase] = []
    segmented_cases: list[SegmentedCase] = []
    try:
        for prompt_id, (prompt, prompt_token_ids) in enumerate(resolved_prompts):
            if args.include_continuous_baseline:
                psrl_logger.info("[reprefill_exp] Generating continuous case for prompt %d.", prompt_id)
                continuous_cases.append(
                    await _generate_continuous(
                        engine,
                        args,
                        prompt_id=prompt_id,
                        prompt=prompt,
                        prompt_token_ids=prompt_token_ids,
                    )
                )
            for segment_length in args.segment_lengths:
                psrl_logger.info(
                    "[reprefill_exp] Generating segmented case for prompt %d with segment length %d.",
                    prompt_id,
                    segment_length,
                )
                segmented_cases.append(
                    await _generate_segmented(
                        engine,
                        args,
                        prompt_id=prompt_id,
                        prompt=prompt,
                        prompt_token_ids=prompt_token_ids,
                        segment_length=segment_length,
                    )
                )
    finally:
        engine.shutdown()
        del engine
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return continuous_cases, segmented_cases, metadata


def _init_single_rank_process_group() -> str:
    import torch.distributed as dist

    if dist.is_initialized():
        if dist.get_world_size() != 1:
            raise RuntimeError("this minimal experiment requires a single-rank FSDP process group")
        return "existing"
    fd, path = tempfile.mkstemp(prefix="reprefill-fsdp-rdzv-")
    os.close(fd)
    os.unlink(path)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{path}",
        rank=0,
        world_size=1,
    )
    return path


def _build_psrl_batch_tensors(
    cases: list[ContinuousCase | SegmentedCase],
    *,
    pad_token_id: int,
    prompt_length: int,
) -> dict[str, Any]:
    """
    Build the padded tensors consumed by PSRL's actor recompute path.

    Args:
        cases (list[ContinuousCase | SegmentedCase]): Fixed sequences to score.
        pad_token_id (int): Token used for left prompt padding.
        prompt_length (int): Fixed padded prompt width.

    Returns:
        dict[str, Any]: Input IDs, response IDs, attention mask, and position IDs.
    """
    import torch
    from verl.utils.model import compute_position_id_with_mask

    if not cases:
        raise ValueError("cannot build a trainer batch without cases")
    response_length = len(cases[0].response_token_ids)
    if response_length < 1:
        raise ValueError("trainer cases must contain response tokens")

    input_rows = []
    response_rows = []
    attention_rows = []
    for case in cases:
        if len(case.response_token_ids) != response_length:
            raise ValueError("all trainer cases must have the same response length")
        left_padding = prompt_length - len(case.prompt_token_ids)
        if left_padding < 0:
            raise ValueError(
                f"case {case.case_id} has prompt length {len(case.prompt_token_ids)}, "
                f"which exceeds trainer prompt length {prompt_length}"
            )
        input_rows.append([pad_token_id] * left_padding + case.prompt_token_ids + case.response_token_ids)
        response_rows.append(case.response_token_ids)
        attention_rows.append([0] * left_padding + [1] * (len(case.prompt_token_ids) + response_length))

    input_ids = torch.tensor(input_rows, dtype=torch.long)
    responses = torch.tensor(response_rows, dtype=torch.long)
    attention_mask = torch.tensor(attention_rows, dtype=torch.long)
    position_ids = compute_position_id_with_mask(attention_mask)
    return {
        "input_ids": input_ids,
        "responses": responses,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }


def _run_fsdp_phase(
    args: argparse.Namespace,
    continuous_cases: list[ContinuousCase],
    segmented_cases: list[SegmentedCase],
) -> tuple[dict[str, list[float]], dict[str, Any], Any]:
    import torch
    import torch.distributed as dist
    import transformers
    import verl
    import verl.utils.torch_functional as verl_F
    from omegaconf import OmegaConf
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl import DataProto
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.fsdp_utils import (
        apply_fsdp2,
        fsdp2_load_full_state_dict,
        get_shard_placement_fn,
    )
    from verl.workers.actor.dp_actor import DataParallelPPOActor

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    rendezvous = _init_single_rank_process_group()
    model_dtype = _torch_dtype(args.trainer_model_dtype)
    forward_dtype = _torch_dtype(args.trainer_dtype)
    trainer_model_path = args.trainer_model or args.model
    model = None
    actor = None
    try:
        psrl_logger.info("[reprefill_exp] Loading trainer checkpoint for PSRL-style FSDP2 recompute.")
        model = AutoModelForCausalLM.from_pretrained(
            trainer_model_path,
            torch_dtype=model_dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.trainer_attention_implementation,
            low_cpu_mem_usage=True,
            local_files_only=args.local_files_only,
        )
        apply_monkey_patch(
            model=model,
            use_remove_padding=args.trainer_use_remove_padding,
            ulysses_sp_size=1,
            use_fused_kernels=args.trainer_use_fused_kernels,
            fused_kernels_backend=None,
        )
        model.to(model_dtype)
        if args.trainer_enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(dist.get_world_size(),),
            mesh_dim_names=("fsdp",),
        )
        mixed_precision = MixedPrecisionPolicy(
            param_dtype=forward_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        fsdp_config = {
            "wrap_policy": {},
            "reshard_after_forward": args.trainer_reshard_after_forward,
        }
        full_state = model.state_dict()
        apply_fsdp2(
            model,
            {
                "mesh": device_mesh,
                "mp_policy": mixed_precision,
                "offload_policy": None,
                "reshard_after_forward": args.trainer_reshard_after_forward,
                "shard_placement_fn": get_shard_placement_fn(fsdp_size=dist.get_world_size()),
            },
            fsdp_config,
        )
        fsdp2_load_full_state_dict(
            model,
            full_state,
            device_mesh=device_mesh,
            cpu_offload=None,
        )
        del full_state

        tokenizer = AutoTokenizer.from_pretrained(
            # Use the rollout tokenizer unless explicitly overridden. A
            # separate trainer checkpoint must still score the same token IDs.
            args.tokenizer or args.model,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        if tokenizer.pad_token_id is None:
            raise RuntimeError("trainer tokenizer must define pad_token_id")

        actor_config = OmegaConf.create(
            {
                "use_remove_padding": args.trainer_use_remove_padding,
                "use_fused_kernels": args.trainer_use_fused_kernels,
                "ulysses_sequence_parallel_size": 1,
                "entropy_from_logits_with_chunking": (args.trainer_entropy_from_logits_with_chunking),
                "entropy_checkpointing": False,
                "use_torch_compile": args.trainer_use_torch_compile,
                "fsdp_config": {"dtype": args.trainer_dtype},
            }
        )
        actor = DataParallelPPOActor(
            config=actor_config,
            actor_module=model,
            actor_optimizer=None,
        )
        all_cases: list[ContinuousCase | SegmentedCase] = [
            *continuous_cases,
            *segmented_cases,
        ]
        tensors = _build_psrl_batch_tensors(
            all_cases,
            pad_token_id=tokenizer.pad_token_id,
            prompt_length=args.trainer_prompt_length,
        )
        trainer_data = DataProto.from_dict(
            tensors=tensors,
            meta_info={
                "micro_batch_size": args.trainer_log_prob_micro_batch_size,
                "max_token_len": args.trainer_log_prob_max_token_len,
                "use_dynamic_bsz": args.trainer_log_prob_use_dynamic_bsz,
                "temperature": args.temperature,
            },
        )
        psrl_logger.info(
            "[reprefill_exp] Scoring %d fixed sequences through DataParallelPPOActor.compute_log_prob.",
            len(all_cases),
        )
        log_probs, entropies = actor.compute_log_prob(
            data=trainer_data,
            calculate_entropy=True,
        )
        if log_probs.shape != (
            len(all_cases),
            args.response_length,
        ):
            raise RuntimeError(
                f"PSRL actor returned logprob shape {tuple(log_probs.shape)}, expected "
                f"{(len(all_cases), args.response_length)}"
            )
        if entropies is None or entropies.shape != log_probs.shape:
            raise RuntimeError("PSRL actor did not return response-aligned entropy")
        log_probs = log_probs.detach().float().cpu()
        scores = {case.case_id: [float(value) for value in log_probs[index]] for index, case in enumerate(all_cases)}

        config = model.config
        metadata = {
            "backend": "verl.workers.actor.dp_actor.DataParallelPPOActor.compute_log_prob",
            "model_path": trainer_model_path,
            "fsdp_strategy": "fsdp2",
            "requested_sharding_strategy": "FULL_SHARD via fully_shard",
            "effective_world_size": dist.get_world_size(),
            "single_rank_no_shard_expected": dist.get_world_size() == 1,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "verl_version": getattr(verl, "__version__", None),
            "model_storage_dtype": str(model_dtype),
            "forward_param_dtype": str(forward_dtype),
            "mixed_precision": {
                "param_dtype": str(forward_dtype),
                "reduce_dtype": str(torch.float32),
                "cast_forward_inputs": True,
            },
            "tensor_parallel_size": 1,
            "sequence_parallel_size": 1,
            "attention_implementation": getattr(config, "_attn_implementation", None),
            "flash_attention": "flash" in str(getattr(config, "_attn_implementation", "")),
            "logprob_implementation": "verl.utils.torch_functional.logprobs_from_logits",
            "flash_attention_cross_entropy_available": (verl_F.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE),
            "logits_cast_to_fp32_before_normalization": False,
            "temperature": args.temperature,
            "prompt_padded_length": args.trainer_prompt_length,
            "micro_batch_size": args.trainer_log_prob_micro_batch_size,
            "use_dynamic_batch_size": args.trainer_log_prob_use_dynamic_bsz,
            "max_token_len_per_gpu": args.trainer_log_prob_max_token_len,
            "use_remove_padding": args.trainer_use_remove_padding,
            "use_fused_kernels": args.trainer_use_fused_kernels,
            "use_torch_compile": args.trainer_use_torch_compile,
            "calculate_entropy": True,
            "gradient_checkpointing_enabled": (args.trainer_enable_gradient_checkpointing),
            "reshard_after_forward": args.trainer_reshard_after_forward,
            "explicit_attention_mask": True,
            "explicit_position_ids": True,
            "eval_mode": not model.training,
            "no_grad": True,
            "optimizer_created": False,
            "optimizer_step": False,
            "full_psrl_runtime_differences": [
                "single process and single-rank FSDP2 instead of Ray worker dispatch",
                "Ulysses sequence parallel manager is not entered because sequence_parallel_size is 1",
                "PSRL weight-arena and parameter offload integrations are not enabled",
            ],
            "vocab_size": config.vocab_size,
            "tokenizer_class": tokenizer.__class__.__name__,
            "model_type": config.model_type,
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "rope_scaling": getattr(config, "rope_scaling", None),
            "rope_theta": getattr(config, "rope_theta", None),
            "rendezvous": rendezvous,
        }
        return scores, metadata, tokenizer
    finally:
        if actor is not None:
            del actor
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
        if dist.is_initialized() and rendezvous != "existing":
            dist.destroy_process_group()
        if rendezvous not in {"existing", ""}:
            Path(rendezvous).unlink(missing_ok=True)


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right, strict=False):
        if left_token != right_token:
            break
        count += 1
    return count


def _position_bucket(position: int, size: int) -> str:
    start = position // size * size
    return f"{start}-{start + size - 1}"


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"num_tokens": 0, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "num_tokens": len(values),
        "mean": statistics.fmean(values),
        "p95": _percentile(values, 95.0),
        "p99": _percentile(values, 99.0),
        "max": max(values),
    }


def _case_token_rows(
    case: SegmentedCase,
    trainer: list[float],
    tokenizer: Any,
    position_bucket_size: int,
) -> list[dict[str, Any]]:
    rows = []
    for position, token_id in enumerate(case.response_token_ids):
        boundary = token_boundary_metadata(position, case.boundaries)
        rows.append(
            {
                "case_id": case.case_id,
                "prompt_id": case.prompt_id,
                "segment_length": int(case.case_id.rsplit("n", 1)[1]),
                "token_position": position,
                "token_id": token_id,
                "token_text": tokenizer.decode([token_id]),
                "segment_id": case.segment_id[position],
                "reprefill_count_before_token": case.segment_id[position],
                **boundary,
                "position_bucket": _position_bucket(position, position_bucket_size),
                "replacement_source": case.replacement_source[position],
                "final_replacement_source": case.final_replacement_source[position],
                "final_prefill_available": case.final_prefill_available[position],
                "logprob_concat": case.logprob_concat[position],
                "logprob_replace": case.logprob_replace[position],
                "logprob_replace_online": case.logprob_replace[position],
                "logprob_replace_final": case.logprob_replace_final[position],
                "logprob_trainer": trainer[position],
                "delta_concat": trainer[position] - case.logprob_concat[position],
                "delta_replace": trainer[position] - case.logprob_replace[position],
                "delta_replace_online": trainer[position] - case.logprob_replace[position],
                "delta_replace_final": trainer[position] - case.logprob_replace_final[position],
                "abs_delta_concat": abs(trainer[position] - case.logprob_concat[position]),
                "abs_delta_replace": abs(trainer[position] - case.logprob_replace[position]),
                "abs_delta_replace_online": abs(trainer[position] - case.logprob_replace[position]),
                "abs_delta_replace_final": abs(trainer[position] - case.logprob_replace_final[position]),
                "ratio_concat": _finite_ratio(trainer[position] - case.logprob_concat[position]),
                "ratio_replace": _finite_ratio(trainer[position] - case.logprob_replace[position]),
                "ratio_replace_online": _finite_ratio(trainer[position] - case.logprob_replace[position]),
                "ratio_replace_final": _finite_ratio(
                    trainer[position] - case.logprob_replace_final[position]
                ),
            }
        )
    return rows


def _aggregate_grouped_rows(
    rows: list[dict[str, Any]],
    group_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        base = dict(zip(group_fields, key, strict=True))
        output_row = {
            **base,
            "concat": _summarize_values([row["abs_delta_concat"] for row in group]),
            "replace": _summarize_values([row["abs_delta_replace"] for row in group]),
        }
        if "abs_delta_replace_final" in group[0]:
            output_row["replace_final"] = _summarize_values(
                [row["abs_delta_replace_final"] for row in group]
            )
        output.append(output_row)
    return output


def _aggregate_method_metrics(
    args: argparse.Namespace,
    continuous_cases: list[ContinuousCase],
    segmented_cases: list[SegmentedCase],
    trainer_scores: dict[str, list[float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    aggregate_values: dict[tuple[int, str], list[tuple[list[float], list[float]]]] = {}
    continuous_by_prompt = {case.prompt_id: case for case in continuous_cases}

    for segment_length in args.segment_lengths:
        method_pairs: dict[str, list[tuple[list[float], list[float]]]] = defaultdict(list)
        for segmented in segmented_cases:
            if int(segmented.case_id.rsplit("n", 1)[1]) != segment_length:
                continue
            continuous = continuous_by_prompt.get(segmented.prompt_id)
            if continuous is not None:
                method_pairs["continuous"].append(
                    (trainer_scores[continuous.case_id], continuous.logprob_continuous)
                )
            trainer = trainer_scores[segmented.case_id]
            method_pairs["concat"].append((trainer, segmented.logprob_concat))
            # Keep the historical key "replace" as an alias for online semantics.
            method_pairs["replace"].append((trainer, segmented.logprob_replace))
            method_pairs["replace_online"].append((trainer, segmented.logprob_replace))
            method_pairs["replace_final"].append((trainer, segmented.logprob_replace_final))
        for method, pairs in method_pairs.items():
            if pairs:
                aggregate_values[(segment_length, method)] = pairs

    summary: dict[str, Any] = {}
    for (segment_length, method), pairs in aggregate_values.items():
        reference = [value for trainer, _ in pairs for value in trainer]
        rollout = [value for _, candidate in pairs for value in candidate]
        metrics = mismatch_statistics(
            reference,
            rollout,
            outlier_thresholds=args.outlier_thresholds,
            ratio_low=args.ratio_low,
            ratio_high=args.ratio_high,
        )
        metrics["mode5_geometric_rs"] = _geometric_rs_batch_statistics(
            pairs,
            upper_threshold=getattr(args, "mode5_rs_threshold", 1.005),
        )
        metrics["offpolicy"] = _offpolicy_batch_statistics(pairs)
        rows.append({"segment_length": segment_length, "method": method, **metrics})
        summary.setdefault(str(segment_length), {})[method] = metrics
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _fractional_reduction(before: float, after: float) -> float | None:
    if before == 0.0:
        return None
    return (before - after) / before


def _build_automated_conclusions(
    aggregate_summary: dict[str, Any],
    token_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    conclusions = {}
    comparison_fields = (
        "mean_absolute_error",
        "p95_absolute_error",
        "p99_absolute_error",
        "max_absolute_error",
    )
    for segment_length, methods in aggregate_summary.items():
        continuous = methods.get("continuous")
        concat = methods["concat"]
        replace = methods["replace"]
        replace_final = methods.get("replace_final")
        matching_rows = [row for row in token_rows if row["segment_length"] == int(segment_length)]
        near_rows = [row for row in matching_rows if row["boundary_category"] in {"boundary_0", "after_1_8"}]
        far_rows = [row for row in matching_rows if row["boundary_category"] == "far"]

        def mean_or_none(rows: list[dict[str, Any]], field: str) -> float | None:
            return statistics.fmean(row[field] for row in rows) if rows else None

        concat_near = mean_or_none(near_rows, "abs_delta_concat")
        replace_near = mean_or_none(near_rows, "abs_delta_replace")
        concat_far = mean_or_none(far_rows, "abs_delta_concat")
        replace_far = mean_or_none(far_rows, "abs_delta_replace")

        reductions = {field: _fractional_reduction(concat[field], replace[field]) for field in comparison_fields}
        concat_masked = concat["ratio"]["hypothetical_masked_fraction"]
        replace_masked = replace["ratio"]["hypothetical_masked_fraction"]
        replacement_improves_core_metrics = (
            all(replace[field] < concat[field] for field in comparison_fields) and replace_masked <= concat_masked
        )

        boundary = {
            "near_boundary_categories": ["boundary_0", "after_1_8"],
            "concat_near_mean_absolute_error": concat_near,
            "replace_near_mean_absolute_error": replace_near,
            "concat_far_mean_absolute_error": concat_far,
            "replace_far_mean_absolute_error": replace_far,
            "concat_near_to_far_ratio": (
                concat_near / concat_far if concat_near is not None and concat_far not in {None, 0.0} else None
            ),
            "replace_near_to_far_ratio": (
                replace_near / replace_far if replace_near is not None and replace_far not in {None, 0.0} else None
            ),
            "replacement_reduces_near_boundary_error": (
                replace_near < concat_near if replace_near is not None and concat_near is not None else None
            ),
        }
        concat_boundary_ratio = boundary["concat_near_to_far_ratio"]
        replace_boundary_ratio = boundary["replace_near_to_far_ratio"]
        boundary["replacement_reduces_boundary_concentration"] = (
            abs(replace_boundary_ratio - 1.0) < abs(concat_boundary_ratio - 1.0)
            if concat_boundary_ratio is not None and replace_boundary_ratio is not None
            else None
        )

        result = {
            "replacement_improves_core_metrics": replacement_improves_core_metrics,
            "fractional_reduction_concat_to_replace": reductions,
            "signed_mean_magnitude_reduction": abs(concat["signed_mean"]) - abs(replace["signed_mean"]),
            "signed_median_magnitude_reduction": abs(concat["signed_median"]) - abs(replace["signed_median"]),
            "tail_reduction_exceeds_mae_reduction": (
                reductions["p99_absolute_error"] is not None
                and reductions["mean_absolute_error"] is not None
                and reductions["p99_absolute_error"] > reductions["mean_absolute_error"]
            ),
            "hypothetical_masked_fraction": {
                "concat": concat_masked,
                "replace": replace_masked,
            },
            "boundary": boundary,
            "canonicalization_candidate_supported": replacement_improves_core_metrics,
        }
        if replace_final is not None:
            final_reductions = {
                field: _fractional_reduction(replace[field], replace_final[field])
                for field in comparison_fields
            }
            result["final_prefill"] = {
                "fractional_reduction_online_to_final": final_reductions,
                "improves_over_online": all(
                    replace_final[field] < replace[field] for field in comparison_fields
                ),
                "hypothetical_masked_fraction": replace_final["ratio"]["hypothetical_masked_fraction"],
                "mode5_geometric_rs": replace_final.get("mode5_geometric_rs"),
            }
        if continuous is not None:
            continuous_masked = continuous["ratio"]["hypothetical_masked_fraction"]
            closer_to_continuous = {
                field: abs(replace[field] - continuous[field]) < abs(concat[field] - continuous[field])
                for field in comparison_fields
            }
            replacement_approaches_baseline = all(closer_to_continuous.values()) and abs(
                replace_masked - continuous_masked
            ) <= abs(concat_masked - continuous_masked)
            result.update(
                {
                    "segmented_adds_mae_over_continuous": (
                        concat["mean_absolute_error"] > continuous["mean_absolute_error"]
                    ),
                    "replacement_approaches_continuous_baseline": replacement_approaches_baseline,
                    "closer_to_continuous_by_metric": closer_to_continuous,
                }
            )
            result["hypothetical_masked_fraction"]["continuous"] = continuous_masked
        conclusions[segment_length] = result
    return conclusions


def _plot_results(
    output_dir: Path,
    token_rows: list[dict[str, Any]],
    aggregate_summary: dict[str, Any],
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/psrl-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    first_case = token_rows[0]["case_id"]
    example = [row for row in token_rows if row["case_id"] == first_case]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        [row["token_position"] for row in example],
        [row["delta_concat"] for row in example],
        label="trainer - concat",
    )
    ax.plot(
        [row["token_position"] for row in example],
        [row["delta_replace"] for row in example],
        label="trainer - replace",
        alpha=0.8,
    )
    if "delta_replace_final" in example[0]:
        ax.plot(
            [row["token_position"] for row in example],
            [row["delta_replace_final"] for row in example],
            label="trainer - final prefill",
            alpha=0.8,
        )
    for row in example:
        if row["is_segment_boundary"]:
            ax.axvline(row["token_position"], color="black", alpha=0.12, linewidth=0.7)
    ax.set_xlabel("Response token position")
    ax.set_ylabel("Signed logprob delta")
    ax.set_title(first_case)
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "per_token_delta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    segment_lengths = sorted(int(key) for key in aggregate_summary)
    percentile_fields = (
        ("median_absolute_error", "P50"),
        ("p95_absolute_error", "P95"),
        ("p99_absolute_error", "P99"),
        ("max_absolute_error", "Max"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, (field, label) in zip(axes.flat, percentile_fields, strict=True):
        methods = ["concat", "replace"]
        if all("replace_final" in aggregate_summary[str(length)] for length in segment_lengths):
            methods.append("replace_final")
        for method in methods:
            ax.plot(
                segment_lengths,
                [aggregate_summary[str(length)][method][field] for length in segment_lengths],
                marker="o",
                label=method,
            )
        ax.set_title(f"{label} absolute error")
        ax.set_xlabel("Segment length")
        ax.set_ylabel("Absolute logprob delta")
        ax.legend()
    fig.tight_layout()
    path = plot_dir / "concat_replace_percentiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    concat_abs = np.asarray([row["abs_delta_concat"] for row in token_rows])
    replace_abs = np.asarray([row["abs_delta_replace"] for row in token_rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(concat_abs, bins=100, alpha=0.55, label="concat", density=True)
    axes[0].hist(replace_abs, bins=100, alpha=0.55, label="replace", density=True)
    axes[0].set_xlabel("Absolute logprob delta")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    for values, label in ((concat_abs, "concat"), (replace_abs, "replace")):
        ordered = np.sort(values)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=label)
    axes[1].set_xlabel("Absolute logprob delta")
    axes[1].set_ylabel("ECDF")
    axes[1].legend()
    fig.tight_layout()
    path = plot_dir / "absolute_delta_hist_ecdf.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    methods = ["concat", "replace"]
    if all("replace_final" in aggregate_summary[str(length)] for length in segment_lengths):
        methods.append("replace_final")
    if all("continuous" in aggregate_summary[str(length)] for length in segment_lengths):
        methods.insert(0, "continuous")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in methods:
        axes[0].plot(
            segment_lengths,
            [aggregate_summary[str(length)][method]["mean_absolute_error"] for length in segment_lengths],
            marker="o",
            label=method,
        )
        axes[1].plot(
            segment_lengths,
            [aggregate_summary[str(length)][method]["p99_absolute_error"] for length in segment_lengths],
            marker="o",
            label=method,
        )
    axes[0].set_title("Mean absolute error")
    axes[1].set_title("P99 absolute error")
    for ax in axes:
        ax.set_xlabel("Segment length")
        ax.set_ylabel("Absolute logprob delta")
        ax.legend()
    fig.tight_layout()
    path = plot_dir / "segment_length_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    categories = ["before_-8_-1", "boundary_0", "after_1_8", "after_9_32", "far"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(categories))
    width = 0.38
    concat_means = []
    replace_means = []
    for category in categories:
        selected = [row for row in token_rows if row["boundary_category"] == category]
        concat_means.append(
            statistics.fmean(row["abs_delta_concat"] for row in selected) if selected else float("nan")
        )
        replace_means.append(
            statistics.fmean(row["abs_delta_replace"] for row in selected) if selected else float("nan")
        )
    ax.bar(x - width / 2, concat_means, width, label="concat")
    ax.bar(x + width / 2, replace_means, width, label="replace")
    ax.set_xticks(x, categories, rotation=20)
    ax.set_ylabel("Mean absolute logprob delta")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "boundary_distance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.8 / len(methods)
    x = np.arange(len(segment_lengths))
    for index, method in enumerate(methods):
        masked = [
            aggregate_summary[str(length)][method]["ratio"]["hypothetical_masked_fraction"]
            for length in segment_lengths
        ]
        offset = (index - (len(methods) - 1) / 2) * width
        ax.bar(x + offset, masked, width, label=method)
    ax.set_xticks(x, [str(value) for value in segment_lengths])
    ax.set_xlabel("Segment length")
    ax.set_ylabel("Hypothetical RS masked fraction")
    ax.legend()
    fig.tight_layout()
    path = plot_dir / "hypothetical_rs_masked_fraction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _build_report(
    args: argparse.Namespace,
    continuous_cases: list[ContinuousCase],
    segmented_cases: list[SegmentedCase],
    trainer_scores: dict[str, list[float]],
    tokenizer: Any,
    vllm_metadata: dict[str, Any],
    trainer_metadata: dict[str, Any],
    checkpoint_before: dict[str, Any],
    checkpoint_after: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    token_rows = []
    segmented_reports = []
    continuous_reports = []
    continuous_by_prompt = {case.prompt_id: case for case in continuous_cases}

    for case in continuous_cases:
        trainer = trainer_scores[case.case_id]
        metrics = mismatch_statistics(
            trainer,
            case.logprob_continuous,
            outlier_thresholds=args.outlier_thresholds,
            ratio_low=args.ratio_low,
            ratio_high=args.ratio_high,
        )
        continuous_reports.append(
            {
                **asdict(case),
                "logprob_trainer": trainer,
                "mismatch": metrics,
            }
        )

    for case in segmented_cases:
        trainer = trainer_scores[case.case_id]
        rows = _case_token_rows(
            case,
            trainer,
            tokenizer,
            args.position_bucket_size,
        )
        token_rows.extend(rows)
        continuous = continuous_by_prompt.get(case.prompt_id)
        concat_metrics = mismatch_statistics(
            trainer,
            case.logprob_concat,
            outlier_thresholds=args.outlier_thresholds,
            ratio_low=args.ratio_low,
            ratio_high=args.ratio_high,
        )
        replace_metrics = mismatch_statistics(
            trainer,
            case.logprob_replace,
            outlier_thresholds=args.outlier_thresholds,
            ratio_low=args.ratio_low,
            ratio_high=args.ratio_high,
        )
        final_metrics = mismatch_statistics(
            trainer,
            case.logprob_replace_final,
            outlier_thresholds=args.outlier_thresholds,
            ratio_low=args.ratio_low,
            ratio_high=args.ratio_high,
        )
        method_pairs = {
            "concat": (trainer, case.logprob_concat),
            "replace_online": (trainer, case.logprob_replace),
            "replace_final": (trainer, case.logprob_replace_final),
        }
        mode5_metrics = {
            method: {
                "geometric_rs": _geometric_rs_statistics(
                    reference,
                    candidate,
                    upper_threshold=getattr(args, "mode5_rs_threshold", 1.005),
                ),
                "offpolicy": _offpolicy_statistics(reference, candidate),
            }
            for method, (reference, candidate) in method_pairs.items()
        }
        case_report = {
            **asdict(case),
            "logprob_trainer": trainer,
            "mismatch": {
                "concat": concat_metrics,
                "replace": replace_metrics,
                "replace_online": replace_metrics,
                "replace_final": final_metrics,
            },
            "mode5_rollout_corr": mode5_metrics,
            "alignment_preview": rows[:20],
        }
        if continuous is not None:
            lcp = _longest_common_prefix(
                continuous.response_token_ids,
                case.response_token_ids,
            )
            case_report.update(
                {
                    "continuous_case_id": continuous.case_id,
                    "continuous_vs_segmented_longest_common_prefix": lcp,
                    "continuous_vs_segmented_same_tokens": lcp
                    == len(case.response_token_ids)
                    == len(continuous.response_token_ids),
                }
            )
        segmented_reports.append(case_report)

    if segmented_reports:
        preview = segmented_reports[0]["alignment_preview"]
        psrl_logger.info(
            "[reprefill_exp] Alignment preview for %s (first %d tokens):\n%s",
            segmented_reports[0]["case_id"],
            len(preview),
            json.dumps(preview, indent=2, ensure_ascii=False),
        )

    aggregate_rows, aggregate_summary = _aggregate_method_metrics(
        args,
        continuous_cases,
        segmented_cases,
        trainer_scores,
    )
    boundary_summary = _aggregate_grouped_rows(
        token_rows,
        ["segment_length", "boundary_category"],
    )
    position_summary = _aggregate_grouped_rows(
        token_rows,
        ["segment_length", "position_bucket"],
    )

    _write_csv(output_dir / "tokens.csv", token_rows)
    _write_csv(output_dir / "summary.csv", aggregate_rows)
    _write_csv(output_dir / "boundary_summary.csv", boundary_summary)
    _write_csv(output_dir / "position_summary.csv", position_summary)
    plot_paths = [] if args.skip_plots else _plot_results(output_dir, token_rows, aggregate_summary)

    conclusions = _build_automated_conclusions(aggregate_summary, token_rows)

    rollout_checkpoint_before = checkpoint_before.get("rollout", checkpoint_before)
    rollout_checkpoint_after = checkpoint_after.get("rollout", checkpoint_after)
    trainer_checkpoint_before = checkpoint_before.get("trainer")
    trainer_checkpoint_after = checkpoint_after.get("trainer")
    return {
        "status": "completed",
        "experiment": {
            "model": args.model,
            "trainer_model": args.trainer_model or args.model,
            "tokenizer": args.tokenizer or args.model,
            "response_length": args.response_length,
            "segment_lengths": args.segment_lengths,
            "num_prompts": len({case.prompt_id for case in segmented_cases}),
            "continuous_baseline_enabled": args.include_continuous_baseline,
            "final_prefix_score_enabled": getattr(args, "final_prefix_score", True),
            "reprefill_count_requested": getattr(args, "reprefill_count", None),
            "reprefill_count_actual": sorted(
                {len(case.reprefill_history) for case in segmented_cases}
            ),
            "mode5_preset_enabled": getattr(args, "match_mode5", False),
            "mode5_rollout_corr_metrics_are_offline": True,
            "total_continuous_response_tokens": sum(len(case.response_token_ids) for case in continuous_cases),
            "total_segmented_response_tokens": sum(len(case.response_token_ids) for case in segmented_cases),
            "total_scored_response_tokens": sum(len(case.response_token_ids) for case in continuous_cases)
            + sum(len(case.response_token_ids) for case in segmented_cases),
            "weight_updates": 0,
            "optimizer_steps": 0,
            "trainer_token_alignment_verified_by_construction": True,
            "replacement_semantics": (
                "At each actual reprefill, every previously generated response token is "
                "overwritten by that request's prompt_logprob. Final-segment tokens remain "
                "from decode because no reprefill follows completion."
            ),
            "final_prefill_semantics": (
                "A separate full-prefix prompt_logprob request overwrites every response token; "
                "tokens for which vLLM omits prompt logprob fall back to online replacement and "
                "are marked unavailable."
            ),
            "continuous_semantics": (
                "Continuous generation is scored by FSDP on its own fixed final sequence. "
                "It is not position-aligned to a divergent segmented sequence."
                if args.include_continuous_baseline
                else "Disabled; default conclusions compare only concat and replace on identical token IDs."
            ),
        },
        # Keep the historical flat rollout checkpoint fields for downstream
        # report consumers; trainer snapshots are additional fields.
        "checkpoint_before": rollout_checkpoint_before,
        "checkpoint_after": rollout_checkpoint_after,
        "trainer_checkpoint_before": trainer_checkpoint_before,
        "trainer_checkpoint_after": trainer_checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "vllm": vllm_metadata,
        "trainer": trainer_metadata,
        "continuous_cases": continuous_reports,
        "segmented_cases": segmented_reports,
        "aggregate_summary": aggregate_summary,
        "boundary_summary": boundary_summary,
        "position_summary": position_summary,
        "automated_conclusions": conclusions,
        "artifacts": {
            "token_csv": str(output_dir / "tokens.csv"),
            "summary_csv": str(output_dir / "summary.csv"),
            "boundary_csv": str(output_dir / "boundary_summary.csv"),
            "position_csv": str(output_dir / "position_summary.csv"),
            "plots": plot_paths,
        },
    }


def _self_test() -> None:
    metrics = mismatch_statistics(
        [-0.1, -0.2, -0.3],
        [-0.1, -0.25, -0.2],
        outlier_thresholds=[0.05],
        ratio_low=0.5,
        ratio_high=2.0,
    )
    assert metrics["num_tokens"] == 3, "self-test token count mismatch."
    assert math.isclose(metrics["mean_absolute_error"], 0.05), "self-test mean absolute error mismatch."
    boundary = token_boundary_metadata(101, [100, 200])
    assert boundary["boundary_category"] == "after_1_8", "self-test boundary category mismatch."
    assert _longest_common_prefix([1, 2, 3], [1, 2, 4]) == 2, "self-test longest common prefix mismatch."
    offpolicy = _offpolicy_statistics([-1.0, -1.0], [-1.1, -0.9])
    assert math.isclose(offpolicy["log_ppl_diff"], 0.0), "self-test log-ppl sign mismatch."
    rs = _geometric_rs_statistics([-1.0, -1.0], [-1.01, -1.0], upper_threshold=1.005)
    assert rs["sequence_rejected"] is True, "self-test geometric RS threshold mismatch."
    values, available = _response_prompt_records([], prompt_length=2, response_ids=[3, 4])
    assert values == [None, None] and available == [False, False], (
        "self-test missing prompt logprob handling mismatch."
    )
    psrl_logger.info("[reprefill_exp] Self-test passed.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-7B")
    parser.add_argument(
        "--trainer-model",
        default=None,
        help="optional trainer checkpoint; defaults to --model for an A/A comparison",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--prompt-token-ids", action="append", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--response-length", type=int, default=1024)
    parser.add_argument(
        "--segment-lengths",
        type=_parse_positive_int_list,
        default=[32, 64, 128, 256],
    )
    parser.add_argument(
        "--reprefill-count",
        type=int,
        default=None,
        help="number of intermediate reprefills; derives one segment length from response length",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--sampling-temperature",
        type=float,
        default=None,
        help="vLLM sampling temperature; trainer scoring always uses --temperature",
    )
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-continuous-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also generate a continuous-decode reference; excluded by default because tokens may diverge",
    )
    parser.add_argument("--vllm-dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--logprobs-mode", default="raw_logprobs")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=1)
    parser.add_argument("--vllm-logprobs", type=int, default=1)
    parser.add_argument("--vllm-prompt-logprobs", type=int, default=1)
    parser.add_argument("--vllm-attention-backend", default=None)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--final-prefix-score",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score the complete final prefix to produce replace_final",
    )
    parser.add_argument(
        "--match-mode5",
        action="store_true",
        help="apply the effective mode5 vLLM/trainer settings before validation",
    )
    parser.add_argument("--trainer-model-dtype", default="float32")
    parser.add_argument("--trainer-dtype", default="bfloat16")
    parser.add_argument(
        "--trainer-attention-implementation",
        default="flash_attention_2",
    )
    parser.add_argument("--trainer-prompt-length", type=int, default=512)
    parser.add_argument(
        "--trainer-log-prob-micro-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--trainer-log-prob-use-dynamic-bsz",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trainer-log-prob-max-token-len",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--trainer-use-remove-padding",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trainer-use-fused-kernels",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trainer-use-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trainer-entropy-from-logits-with-chunking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--trainer-enable-gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trainer-reshard-after-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout-s", type=float, default=1200.0)
    parser.add_argument("--position-bucket-size", type=int, default=128)
    parser.add_argument(
        "--outlier-thresholds",
        type=_parse_float_list,
        default=[0.05, 0.1, 0.5, 1.0],
    )
    parser.add_argument("--ratio-low", type=float, default=0.5)
    parser.add_argument("--ratio-high", type=float, default=2.0)
    parser.add_argument(
        "--mode5-rs-threshold",
        type=float,
        default=1.005,
        help="geometric RS threshold used for offline mode5-compatible diagnostics",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/qwen2.5-7b-reprefill-vs-psrl",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _apply_mode5_defaults(args: argparse.Namespace) -> None:
    """Apply the effective Qwen7B mode5 execution settings as an opt-in preset."""
    if not args.match_mode5:
        return
    args.vllm_max_num_seqs = 512
    args.max_num_batched_tokens = 32768
    args.gpu_memory_utilization = 0.7
    args.enable_prefix_caching = True
    args.enable_chunked_prefill = True
    args.enforce_eager = False
    args.trainer_prompt_length = max(args.trainer_prompt_length, 1024)
    # Qwen2.5-7B exposes a 32K context window and mode5 leaves max_model_len
    # at the model limit while packing up to 32K tokens.
    args.max_model_len = max(args.max_model_len, 32768, args.response_length + args.trainer_prompt_length)
    args.trainer_log_prob_use_dynamic_bsz = True
    args.trainer_use_remove_padding = True
    args.trainer_use_torch_compile = True
    args.trainer_log_prob_max_token_len = max(args.trainer_log_prob_max_token_len, 16384)
    args.vllm_logprobs = max(args.vllm_logprobs, 1)
    args.vllm_prompt_logprobs = max(args.vllm_prompt_logprobs, 1)


def _validate_args(args: argparse.Namespace) -> None:
    if args.response_length < 1:
        raise ValueError("--response-length must be positive")
    if args.max_model_len < args.response_length + 1:
        raise ValueError("--max-model-len must exceed --response-length")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive, matching PSRL trainer recompute requirements")
    if args.sampling_temperature is not None and args.sampling_temperature < 0:
        raise ValueError("--sampling-temperature must be non-negative")
    if not 0 < args.ratio_low < args.ratio_high:
        raise ValueError("ratio thresholds must satisfy 0 < low < high")
    if args.trainer_prompt_length < 1:
        raise ValueError("--trainer-prompt-length must be positive")
    if args.trainer_prompt_length + args.response_length > args.max_model_len:
        raise ValueError("trainer prompt length plus response length must not exceed max model length")
    if args.trainer_log_prob_micro_batch_size < 1:
        raise ValueError("trainer logprob micro-batch size must be positive")
    if args.trainer_log_prob_max_token_len < 1 or args.position_bucket_size < 1:
        raise ValueError("trainer max-token length and position bucket size must be positive")
    if args.reprefill_count is not None:
        if args.reprefill_count < 1:
            raise ValueError("--reprefill-count must be positive")
        if args.reprefill_count >= args.response_length:
            raise ValueError("--reprefill-count must be smaller than --response-length")
    if args.vllm_max_num_seqs < 1 or args.max_num_batched_tokens < 1:
        raise ValueError("vLLM sequence and batched-token limits must be positive")
    for name in ("vllm_logprobs", "vllm_prompt_logprobs"):
        value = getattr(args, name)
        if value == 0 or value < -1:
            raise ValueError(f"--{name.replace('_', '-')} must be -1 or a positive integer")
    if args.mode5_rs_threshold <= 1.0:
        raise ValueError("--mode5-rs-threshold must be greater than 1")


def _prepare_args(args: argparse.Namespace) -> None:
    _apply_mode5_defaults(args)
    if args.reprefill_count is not None:
        # N intermediate reprefills require N+1 decode segments. The final
        # segment may be shorter, which preserves the requested total length.
        args.segment_lengths = [math.ceil(args.response_length / (args.reprefill_count + 1))]


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    _prepare_args(args)
    if args.self_test:
        _self_test()
        return 0
    try:
        _validate_args(args)
    except ValueError as exc:
        psrl_logger.error("[reprefill_exp] Invalid arguments: %s.", exc)
        return 2

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        psrl_logger.error("[reprefill_exp] No CUDA device is visible.")
        return 1 if args.require_gpu else 0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_inputs = _load_prompts(args)
    checkpoint_before = {
        "rollout": checkpoint_snapshot(args.model),
        "trainer": checkpoint_snapshot(args.trainer_model or args.model),
    }
    started = time.time()
    try:
        continuous_cases, segmented_cases, vllm_metadata = asyncio.run(_run_vllm_phase(args, prompt_inputs))
        trainer_scores, trainer_metadata, tokenizer = _run_fsdp_phase(
            args,
            continuous_cases,
            segmented_cases,
        )
        checkpoint_after = {
            "rollout": checkpoint_snapshot(args.model),
            "trainer": checkpoint_snapshot(args.trainer_model or args.model),
        }
        if checkpoint_before != checkpoint_after:
            raise RuntimeError("checkpoint files changed between vLLM rollout and FSDP recompute")
        report = _build_report(
            args,
            continuous_cases,
            segmented_cases,
            trainer_scores,
            tokenizer,
            vllm_metadata,
            trainer_metadata,
            checkpoint_before,
            checkpoint_after,
            output_dir,
        )
        report["duration_s"] = time.time() - started
        report["hostname"] = os.uname().nodename
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        psrl_logger.info("[reprefill_exp] Full report written to %s.", report_path)
        psrl_logger.info(
            "[reprefill_exp] Aggregate summary:\n%s",
            json.dumps(report["aggregate_summary"], indent=2),
        )
        return 0
    except Exception:
        psrl_logger.exception("[reprefill_exp] Experiment failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
