"""
Start one engine instance, record its E3b memory breakdown, and exit.

Run one process per `max_num_batched_tokens` value because vLLM fixes
`peak_activation` during engine startup.

Usage:

    # Probe N=8192
    PYTHONUNBUFFERED=1 python -m psrl.bench.chunked_prefill.memory_probe \\
        --model /path/to/model \\
        --max-batched-tokens 8192 \\
        --gpu-util 0.90 \\
        --tp 1 \\
        --output results/e3b_N8192.json

    # Shell loop for E3b sweep (see run_chunked_prefill_memory_sweep.sh)
    for N in 512 1024 2048 4096 8192 16384 32768 65536
    do
        python -m psrl.bench.chunked_prefill.memory_probe --max-batched-tokens $N ...
    done

Each invocation writes one JSON object to `--output`, or stdout when omitted,
with complete environment metadata and the memory breakdown from rank 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from typing import Any

import torch
from vllm import LLM

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=os.getenv("PSRL_LOGGING_LEVEL", "INFO"),
)
psrl_logger = logging.getLogger(__name__)


def _get_gpu_name() -> str:
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def _get_vllm_version() -> str:
    try:
        import vllm

        return vllm.__version__
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "E3b memory probe: start an engine, read memory breakdown, exit. "
            "Run once per max_num_batched_tokens value."
        )
    )
    parser.add_argument("--model", required=True, help="Path to the HuggingFace model.")
    parser.add_argument(
        "--max-batched-tokens",
        type=int,
        required=True,
        help="max_num_batched_tokens value to probe (determines activation reservation).",
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size.")
    parser.add_argument("--gpu-util", type=float, default=0.90, help="gpu_memory_utilization.")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype.")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--max-seqs",
        type=int,
        default=1024,
        help="max_num_seqs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON result.  If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code when loading the model.",
    )
    args = parser.parse_args()

    # vLLM SchedulerConfig requires max_num_batched_tokens >= max_num_seqs.
    # E3b sweeps N down to 512 while the default max_num_seqs is 1024, so clamp.
    max_seqs = min(args.max_seqs, args.max_batched_tokens)
    if max_seqs != args.max_seqs:
        psrl_logger.info(
            "Clamping max_num_seqs from %d to %d (must be <= max_num_batched_tokens=%d).",
            args.max_seqs,
            max_seqs,
            args.max_batched_tokens,
        )

    psrl_logger.info(
        "Starting engine for E3b probe: model=%r N=%d TP=%d util=%.2f.",
        args.model,
        args.max_batched_tokens,
        args.tp,
        args.gpu_util,
    )

    # Start engine.
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        max_num_seqs=max_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        trust_remote_code=args.trust_remote_code,
        seed=0,
        worker_extension_cls=("psrl.bench.chunked_prefill.vllm_extension.ChunkedPrefillProbeExtension"),
    )
    psrl_logger.info("Engine initialised.")

    # Report rank zero while retaining every rank in the result.
    mem_per_rank: list[dict[str, Any]] = llm.collective_rpc("get_memory_breakdown")
    rank0 = mem_per_rank[0]

    # Build result record.
    result: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "vllm_version": _get_vllm_version(),
        "gpu_name": _get_gpu_name(),
        "gpu_count": torch.cuda.device_count(),
        "model_path": args.model,
        "max_num_batched_tokens": args.max_batched_tokens,
        "max_num_seqs": max_seqs,
        "tensor_parallel_size": args.tp,
        "gpu_memory_utilization": args.gpu_util,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "experiment": "e3b",
        # Memory accounting from rank 0.
        "total_gpu_bytes": rank0.get("total_gpu_bytes", 0),
        "requested_bytes": rank0.get("requested_bytes", 0),
        "weights_bytes": rank0.get("weights_bytes", 0),
        "peak_activation_bytes": rank0.get("peak_activation_bytes", 0),
        "non_torch_bytes": rank0.get("non_torch_bytes", 0),
        "available_kv_bytes": rank0.get("available_kv_bytes", 0),
        "num_gpu_blocks": rank0.get("num_gpu_blocks", 0),
        "block_size_tokens": rank0.get("block_size_tokens", 0),
        "kv_token_capacity": rank0.get("kv_token_capacity", 0),
        "memory_breakdown_per_rank": mem_per_rank,
    }

    psrl_logger.info(
        "Memory probe result: N=%d peak_act=%.2fGiB kv_tokens=%dk.",
        args.max_batched_tokens,
        result["peak_activation_bytes"] / 1024**3,
        result["kv_token_capacity"] // 1000,
    )

    out_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        import pathlib

        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n", encoding="utf-8")
        psrl_logger.info("Wrote result to %r.", args.output)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
