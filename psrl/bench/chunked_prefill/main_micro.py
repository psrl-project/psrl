"""
Chunked prefill micro-benchmark main entry point.

Usage::

    # Quick smoke test (E1 only, TP=1, 7B model)
    python -m psrl.bench.chunked_prefill.main_micro \\
        model.path=/path/to/SWE-agent-LM-7B \\
        micro.experiment=e1 \\
        micro.e1.m_values=[512,2048,8192] \\
        micro.warmup=2 micro.iters=5

    # Full sweep, TP=4
    python -m psrl.bench.chunked_prefill.main_micro \\
        model.path=/path/to/model \\
        rollout.tensor_parallel_size=4 \\
        micro.experiment=all

Results are written as JSONL to ``micro.output_dir``, one line per measurement.
Each line contains complete environment metadata (vLLM version, GPU model,
TP, model path, budget) so results from different runs can be safely aggregated.

See ``psrl/bench/chunked_prefill/plot.py`` for visualisation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import time
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from psrl.bench.chunked_prefill.batch_spec import (
    BatchRequest,
    format_batch_spec,
    total_query_tokens,
)
from psrl.utils.logger import DualOutputHandler
from vllm import LLM

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# --- Helpers ---


def _get_gpu_name() -> str:
    """Return the first GPU's name or 'unknown'."""
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


def _make_run_meta(config: DictConfig) -> dict[str, Any]:
    """Build a dict of run-level metadata to embed in every result line."""
    return {
        "run_id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": platform.node(),
        "vllm_version": _get_vllm_version(),
        "gpu_name": _get_gpu_name(),
        "gpu_count": torch.cuda.device_count(),
        "model_path": config.model.path,
        "tensor_parallel_size": config.rollout.tensor_parallel_size,
        "max_num_batched_tokens": config.rollout.max_num_batched_tokens,
        "max_num_seqs": config.rollout.max_num_seqs,
        "dtype": config.rollout.dtype,
        "gpu_memory_utilization": config.rollout.gpu_memory_utilization,
        "peak_tflops": config.micro.peak_tflops,
    }


def _compute_mfu(flops: int, latency_ms: float, peak_tflops: float) -> float | None:
    """
    Compute model FLOPs utilisation.

    Args:
        flops (int): Theoretical FLOPs for this step.
        latency_ms (float): Elapsed time in milliseconds.
        peak_tflops (float): Device peak bf16 TFLOPs.

    Returns:
        float | None: MFU in [0, 1], or None if inputs are invalid.
    """
    if flops <= 0 or latency_ms <= 0 or peak_tflops <= 0:
        return None
    peak_flops_per_sec = peak_tflops * 1e12
    elapsed_s = latency_ms / 1000.0
    return flops / (elapsed_s * peak_flops_per_sec)


def _write_result(out_file: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to the output file."""
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _probe_batch(
    llm: LLM,
    requests: list[BatchRequest],
    warmup: int,
    iters: int,
    experiment: str,
    meta: dict[str, Any],
    extra: dict[str, Any],
    out_file: Path,
    peak_tflops: float,
) -> None:
    """
    Run one probe and write the result to the output JSONL file.

    Args:
        llm: Initialised ``LLM`` instance.
        requests: The batch composition to probe.
        warmup: Warm-up iterations.
        iters: Measured iterations.
        experiment: Experiment label (``"e1"``, ``"e2"``, ``"e3a"``).
        meta: Run-level metadata dict.
        extra: Experiment-specific fields to embed in the record.
        out_file: Output JSONL path.
        peak_tflops: Device peak FLOPs for MFU calculation.
    """
    q_lens = [r.q_len for r in requests]
    kv_lens = [r.kv_len for r in requests]
    desc = format_batch_spec(requests)
    psrl_logger.info(
        "Probing experiment=%s batch=%s total_q=%d num_reqs=%d.",
        experiment,
        desc,
        total_query_tokens(requests),
        len(requests),
    )

    per_rank: list[dict[str, Any]] = llm.collective_rpc(
        "probe_step",
        args=(q_lens, kv_lens, warmup, iters),
    )

    # Aggregate across ranks: take max latency (step wall-clock = slowest rank).
    non_skipped = [r for r in per_rank if not r.get("skipped")]
    if not non_skipped:
        reason = per_rank[0].get("skip_reason", "unknown") if per_rank else "no ranks"
        psrl_logger.warning("Probe skipped for %r: %s.", desc, reason)
        record: dict[str, Any] = {
            **meta,
            "experiment": experiment,
            **extra,
            "batch_spec_desc": desc,
            "q_lens": q_lens,
            "kv_lens": kv_lens,
            "skipped": True,
            "skip_reason": reason,
        }
        _write_result(out_file, record)
        return

    lat_med = max(r["latency_ms_median"] for r in non_skipped)
    lat_p10 = max(r["latency_ms_p10"] for r in non_skipped)
    lat_p90 = max(r["latency_ms_p90"] for r in non_skipped)
    act_med = max((r["activation_bytes_median"] or 0) for r in non_skipped)
    flops = max(r["flops"] for r in non_skipped)
    total_q = sum(q_lens)

    throughput_tok_per_s = (total_q / lat_med * 1000.0) if lat_med > 0 else None
    mfu = _compute_mfu(flops, lat_med, peak_tflops)

    record = {
        **meta,
        "experiment": experiment,
        **extra,
        "batch_spec_desc": desc,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "num_reqs": len(requests),
        "total_q_tokens": total_q,
        "latency_ms_median": lat_med,
        "latency_ms_p10": lat_p10,
        "latency_ms_p90": lat_p90,
        "activation_bytes_median": act_med,
        "flops": flops,
        "throughput_tok_per_s": throughput_tok_per_s,
        "mfu": mfu,
        "skipped": False,
        "per_rank": non_skipped,
    }
    _write_result(out_file, record)
    psrl_logger.info(
        "Recorded lat_ms=%.2f throughput=%.0f tok/s MFU=%.3f.",
        lat_med,
        throughput_tok_per_s or 0,
        mfu or 0,
    )


# --- Experiment Runners ---


def run_e1(
    llm: LLM,
    cfg: DictConfig,
    meta: dict[str, Any],
    out_file: Path,
) -> None:
    """
    Measure E1 MFU against total query tokens `M`.

    Each `M` can compare single-request and multiple-request decompositions.
    """
    m_values: list[int] = list(cfg.micro.e1.m_values)
    test_decomp: bool = bool(cfg.micro.e1.test_decompositions)
    multi_chunk_size: int = int(cfg.micro.e1.multi_chunk_size)
    warmup = cfg.micro.warmup
    iters = cfg.micro.iters
    peak_tflops = cfg.micro.peak_tflops
    max_seqs = cfg.rollout.max_num_seqs

    for m in m_values:
        # Decomposition 1: single request of q_len = m (pure prefill).
        reqs_single = [BatchRequest(q_len=m, kv_len=m)]
        _probe_batch(
            llm,
            reqs_single,
            warmup,
            iters,
            "e1",
            meta,
            {"m": m, "decomposition": "single"},
            out_file,
            peak_tflops,
        )

        if test_decomp and m > 1:
            # Decomposition 2: split into multiple smaller requests.
            # Grow `multi_chunk_size` when the request count would exceed `max_seqs`.
            target = max(1, multi_chunk_size)
            chunk = max(1, m // min(max_seqs, max(1, m // target)))
            num_reqs = math.ceil(m / chunk)
            leftover = m - chunk * (num_reqs - 1)
            reqs_multi = [BatchRequest(q_len=chunk, kv_len=chunk)] * (num_reqs - 1)
            reqs_multi.append(BatchRequest(q_len=leftover, kv_len=leftover))
            _probe_batch(
                llm,
                reqs_multi,
                warmup,
                iters,
                "e1",
                meta,
                {"m": m, "decomposition": "multi"},
                out_file,
                peak_tflops,
            )


def run_e2(
    llm: LLM,
    cfg: DictConfig,
    meta: dict[str, Any],
    out_file: Path,
) -> None:
    """
    E2: Mixed-batch overhead.

    For each (D, C, L) combination:
    1. Probe the mixed batch: D decode requests (q=1, kv=L) + 1 prefill (q=C, kv=C).
    2. Probe D decode-only.
    3. Probe 1 prefill-only.
    Overhead = t_mixed / (t_decode_only + t_prefill_only).
    """
    decode_counts: list[int] = list(cfg.micro.e2.decode_counts)
    prefill_chunks: list[int] = list(cfg.micro.e2.prefill_chunks)
    context_lens: list[int] = list(cfg.micro.e2.context_lens)
    warmup = cfg.micro.warmup
    iters = cfg.micro.iters
    peak_tflops = cfg.micro.peak_tflops

    for d in decode_counts:
        for c in prefill_chunks:
            for l_ctx in context_lens:
                # Decode requests: q=1, kv=l_ctx.
                dec_reqs = [BatchRequest(q_len=1, kv_len=l_ctx)] * d
                # Prefill request: q=c, kv=c (no prior context).
                pf_reqs = [BatchRequest(q_len=c, kv_len=c)]

                extra = {"decode_count": d, "prefill_chunk": c, "context_len": l_ctx}

                # Mixed.
                if d + 1 > 0:
                    mixed_reqs = dec_reqs + pf_reqs
                    _probe_batch(
                        llm,
                        mixed_reqs,
                        warmup,
                        iters,
                        "e2",
                        meta,
                        {**extra, "variant": "mixed"},
                        out_file,
                        peak_tflops,
                    )

                # Decode-only (skip if d == 0).
                if d > 0:
                    _probe_batch(
                        llm,
                        dec_reqs,
                        warmup,
                        iters,
                        "e2",
                        meta,
                        {**extra, "variant": "decode_only"},
                        out_file,
                        peak_tflops,
                    )

                # Prefill-only.
                _probe_batch(
                    llm,
                    pf_reqs,
                    warmup,
                    iters,
                    "e2",
                    meta,
                    {**extra, "variant": "prefill_only"},
                    out_file,
                    peak_tflops,
                )


def run_e3a(
    llm: LLM,
    cfg: DictConfig,
    meta: dict[str, Any],
    out_file: Path,
) -> None:
    """
    Measure the E3a per-step activation increment against `M`.

    Reuse the `probe_step` field `activation_bytes_median`. The difference
    between `max_memory_allocated()` and `allocated_before` captures only the
    transient activation peak above the KV-cache baseline.
    """
    m_values: list[int] = list(cfg.micro.e3a.m_values)
    warmup = cfg.micro.warmup
    iters = cfg.micro.iters
    peak_tflops = cfg.micro.peak_tflops

    for m in m_values:
        reqs = [BatchRequest(q_len=m, kv_len=m)]
        _probe_batch(
            llm,
            reqs,
            warmup,
            iters,
            "e3a",
            meta,
            {"m": m},
            out_file,
            peak_tflops,
        )


def run_e_chunked(
    llm: LLM,
    cfg: DictConfig,
    meta: dict[str, Any],
    out_file: Path,
) -> None:
    """
    Measure per-step timing for E_chunked prefill sequences.

    For each `(total_len, chunk_size, prefix_len, decode_scenario)` combination,
    call `probe_chunked_sequence` on all workers and write one
    JSONL record containing the full per-step latency breakdown.
    """
    ec = cfg.micro.e_chunked
    total_lens: list[int] = list(ec.total_lens)
    chunk_sizes: list[int] = list(ec.chunk_sizes)
    prefix_lens: list[int] = list(ec.prefix_lens)
    decode_scenarios: list[list[int]] = [list(s) for s in ec.decode_scenarios]
    warmup: int = ec.warmup
    iters: int = ec.iters

    for total_len in total_lens:
        for chunk_size in chunk_sizes:
            # Skip if chunk_size >= total_len (would be a single-step prefill,
            # which is already covered by e1).
            if chunk_size >= total_len:
                continue
            for prefix_len in prefix_lens:
                for dec_scenario in decode_scenarios:
                    num_dec = len(dec_scenario)
                    dec_label = f"dec{num_dec}" if num_dec > 0 else "nodec"
                    psrl_logger.info(
                        "E_chunked: total=%d chunk=%d prefix=%d %s...",
                        total_len,
                        chunk_size,
                        prefix_len,
                        dec_label,
                    )
                    per_rank: list[dict[str, Any]] = llm.collective_rpc(
                        "probe_chunked_sequence",
                        args=(total_len, chunk_size),
                        kwargs={
                            "prefix_len": prefix_len,
                            "decode_contexts": dec_scenario,
                            "warmup": warmup,
                            "iters": iters,
                        },
                    )

                    non_skipped = [r for r in per_rank if not r.get("skipped")]
                    if not non_skipped:
                        reason = per_rank[0].get("skip_reason", "unknown") if per_rank else "no ranks"
                        psrl_logger.warning(
                            "E_chunked skipped total=%d chunk=%d prefix=%d %s: %s.",
                            total_len,
                            chunk_size,
                            prefix_len,
                            dec_label,
                            reason,
                        )
                        record: dict[str, Any] = {
                            **meta,
                            "experiment": "e_chunked",
                            "total_len": total_len,
                            "chunk_size": chunk_size,
                            "prefix_len": prefix_len,
                            "num_decode_reqs": num_dec,
                            "decode_contexts": dec_scenario,
                            "skipped": True,
                            "skip_reason": reason,
                        }
                        _write_result(out_file, record)
                        continue

                    # Take per-step max across ranks (wall-clock = slowest rank).
                    r0 = non_skipped[0]
                    num_steps = r0["num_steps"]

                    def _max_field_across_ranks(
                        ranks: list[dict[str, Any]],
                        field: str,
                        step_idx: int,
                    ) -> float | None:
                        vals = [r[field][step_idx] for r in ranks if r[field][step_idx] is not None]
                        return max(vals) if vals else None

                    step_lat_med = [
                        _max_field_across_ranks(non_skipped, "step_latency_ms_median", s) for s in range(num_steps)
                    ]
                    step_lat_p10 = [
                        _max_field_across_ranks(non_skipped, "step_latency_ms_p10", s) for s in range(num_steps)
                    ]
                    step_lat_p90 = [
                        _max_field_across_ranks(non_skipped, "step_latency_ms_p90", s) for s in range(num_steps)
                    ]
                    step_act = [
                        _max_field_across_ranks(non_skipped, "step_activation_bytes_median", s)
                        for s in range(num_steps)
                    ]
                    seq_lat_med = max(
                        r["sequence_latency_ms_median"]
                        for r in non_skipped
                        if r["sequence_latency_ms_median"] is not None
                    )
                    seq_lat_p10 = max(
                        r["sequence_latency_ms_p10"] for r in non_skipped if r["sequence_latency_ms_p10"] is not None
                    )
                    seq_lat_p90 = max(
                        r["sequence_latency_ms_p90"] for r in non_skipped if r["sequence_latency_ms_p90"] is not None
                    )

                    # Throughput: total compute tokens / sequence wall-clock.
                    total_q_tokens = sum(r0["step_q_tokens"])
                    throughput = (total_q_tokens / seq_lat_med * 1000.0) if seq_lat_med else None

                    record = {
                        **meta,
                        "experiment": "e_chunked",
                        "total_len": total_len,
                        "chunk_size": chunk_size,
                        "prefix_len": prefix_len,
                        "num_decode_reqs": num_dec,
                        "decode_contexts": dec_scenario,
                        "num_steps": num_steps,
                        "step_q_tokens": r0["step_q_tokens"],
                        # Per-step breakdown.
                        "step_latency_ms_median": step_lat_med,
                        "step_latency_ms_p10": step_lat_p10,
                        "step_latency_ms_p90": step_lat_p90,
                        "step_activation_bytes_median": step_act,
                        # Sequence totals.
                        "sequence_latency_ms_median": seq_lat_med,
                        "sequence_latency_ms_p10": seq_lat_p10,
                        "sequence_latency_ms_p90": seq_lat_p90,
                        "sequence_throughput_tok_per_s": throughput,
                        "skipped": False,
                        "per_rank": non_skipped,
                    }
                    _write_result(out_file, record)
                    psrl_logger.info(
                        "E_chunked done: total=%d chunk=%d prefix=%d %s seq_ms=%.1f (%d steps: [%s]).",
                        total_len,
                        chunk_size,
                        prefix_len,
                        dec_label,
                        seq_lat_med,
                        num_steps,
                        ", ".join(f"{v:.1f}" for v in step_lat_med if v is not None),
                    )


# --- Main Entry Point ---


@hydra.main(
    config_path="config",
    config_name="chunked_prefill_micro",
    version_base=None,
)
def main(config: DictConfig) -> None:
    """
    Launch the chunked prefill micro-benchmark.

    Reads configuration from ``config/chunked_prefill_micro.yaml``, starts a
    ``vllm.LLM`` instance with ``ChunkedPrefillProbeExtension`` injected, runs
    the requested experiments, and writes JSONL results to ``micro.output_dir``.
    """
    OmegaConf.resolve(config)

    # Set up logging.
    tp = config.rollout.tensor_parallel_size
    log_prefix = f"micro_TP{tp}"
    psrl_logger.addHandler(DualOutputHandler(config.psrl.logging_path, log_prefix))
    psrl_logger.info("Starting chunked prefill micro-benchmark (experiment=%r).", config.micro.experiment)

    # Output directory.
    out_dir = Path(config.micro.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{config.micro.output_prefix}_TP{tp}_{ts}.jsonl"
    psrl_logger.info("Results will be written to %r.", str(out_file))

    # Initialise LLM with the probe extension injected.
    rollout = config.rollout
    psrl_logger.info("Initialising LLM (model=%r, TP=%d)...", config.model.path, tp)
    llm = LLM(
        model=config.model.path,
        tensor_parallel_size=tp,
        pipeline_parallel_size=rollout.pipeline_parallel_size,
        dtype=rollout.dtype,
        gpu_memory_utilization=rollout.gpu_memory_utilization,
        max_model_len=rollout.max_model_len,
        max_num_seqs=rollout.max_num_seqs,
        max_num_batched_tokens=rollout.max_num_batched_tokens,
        enable_chunked_prefill=rollout.enable_chunked_prefill,
        enable_prefix_caching=rollout.enable_prefix_caching,
        trust_remote_code=config.model.trust_remote_code,
        seed=rollout.seed,
        worker_extension_cls="psrl.bench.chunked_prefill.vllm_extension.ChunkedPrefillProbeExtension",
    )
    psrl_logger.info("LLM initialised.")

    meta = _make_run_meta(config)

    # Also capture memory breakdown and write as a special "setup" record.
    try:
        mem_per_rank: list[dict[str, Any]] = llm.collective_rpc("get_memory_breakdown")
        setup_record: dict[str, Any] = {
            **meta,
            "experiment": "setup",
            "memory_breakdown_per_rank": mem_per_rank,
        }
        _write_result(out_file, setup_record)
        psrl_logger.info(
            "Memory breakdown: kv_token_capacity=%dk (rank 0).",
            (mem_per_rank[0].get("kv_token_capacity") or 0) // 1000,
        )
    except Exception as exc:
        psrl_logger.warning("Could not collect memory breakdown: %r.", exc)

    # Run experiments.
    experiment = config.micro.experiment
    if experiment in ("e1", "all"):
        psrl_logger.info("Running E1 (MFU vs M)...")
        run_e1(llm, config, meta, out_file)
    if experiment in ("e2", "all"):
        psrl_logger.info("Running E2 (mixed batch overhead)...")
        run_e2(llm, config, meta, out_file)
    if experiment in ("e3a", "all"):
        psrl_logger.info("Running E3a (activation increment vs M)...")
        run_e3a(llm, config, meta, out_file)

    if experiment in ("e_chunked", "all"):
        psrl_logger.info("Running E_chunked (per-step chunked prefill timing)...")
        run_e_chunked(llm, config, meta, out_file)

    psrl_logger.info("Benchmark complete. Results at %r.", str(out_file))


if __name__ == "__main__":
    main()
