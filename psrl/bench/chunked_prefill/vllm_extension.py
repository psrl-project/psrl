"""
Expose worker-side probes for chunked prefill micro-benchmarks.

`probe_step` measures one synthetic engine step, `probe_chunked_sequence` measures
a complete chunked sequence, and `get_memory_breakdown` reports worker memory.

The normal `execute_model` path defers sampling, so each call must be followed by
`sample_tokens(None)`. Cleanup removes request mappings but preserves physical KV
blocks, allowing later chunks to reuse earlier KV values.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.metrics.perf import ModelMetrics

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# Sentinel token ID used to fill synthetic prompts.
_DUMMY_TOKEN_ID = 1


def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _build_scheduler_output(
    req_ids: list[str],
    q_lens: list[int],
    kv_lens: list[int],
    block_size: int,
    block_offset: int,
) -> tuple[SchedulerOutput, int]:
    """
    Build a synthetic ``SchedulerOutput`` for one probe step.

    Returns ``(scheduler_output, next_block_offset)`` where ``next_block_offset``
    is the first block ID that was *not* used by this call (so the caller can
    advance a global counter to avoid ID collisions between probes).

    Args:
        req_ids (list[str]): Request IDs, one per request.
        q_lens (list[int]): Query lengths per request.
        kv_lens (list[int]): KV-cache lengths per request.
        block_size (int): Block size in tokens.
        block_offset (int): First block ID to allocate from.
            Must be >= 1 (block 0 is the null block).

    Returns:
        tuple[SchedulerOutput, int]: Synthetic output and updated block offset.
    """
    new_reqs: list[NewRequestData] = []
    num_scheduled_tokens: dict[str, int] = {}

    cur_block = block_offset
    for req_id, q_len, kv_len in zip(req_ids, q_lens, kv_lens):
        n_blocks = _cdiv(kv_len, block_size)
        block_ids = list(range(cur_block, cur_block + n_blocks))
        cur_block += n_blocks

        # NOTE(lhy): Prompt token IDs must cover `kv_len` so `_update_states` can
        # write them into `input_batch`.
        prompt_token_ids = [_DUMMY_TOKEN_ID] * kv_len

        new_reqs.append(
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                mm_features=[],
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=1,
                    ignore_eos=True,
                ),
                pooling_params=None,
                block_ids=(block_ids,),
                num_computed_tokens=kv_len - q_len,
                lora_request=None,
            )
        )
        num_scheduled_tokens[req_id] = q_len

    sched_out = SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(q_lens),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return sched_out, cur_block


def _build_cleanup_output(req_ids: list[str]) -> SchedulerOutput:
    """
    Build a zero-token ``SchedulerOutput`` that only marks requests finished.

    The empty `execute_model` path calls `_update_states` and returns
    `EMPTY_MODEL_RUNNER_OUTPUT` without deferred sampling.
    """
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(req_ids),
        free_encoder_mm_hashes=[],
    )


class ChunkedPrefillProbeExtension:
    """
    vLLM worker extension for chunked prefill micro-benchmarks.

    Registered via ``LLM(worker_extension_cls=...)`` and called through
    ``LLM.collective_rpc("probe_step", ...)`` / ``collective_rpc("get_memory_breakdown")``.

    The extension accesses ``self.model_runner`` and ``self.vllm_config``
    which vLLM injects automatically for any ``worker_extension_cls``.
    """

    # --- Public RPC Methods ---

    def probe_step(
        self,
        q_lens: list[int],
        kv_lens: list[int],
        warmup: int = 3,
        iters: int = 10,
    ) -> dict[str, Any]:
        """
        Measure a synthetic engine step with the requested geometry.

        Args:
            q_lens (list[int]): Query lengths per request (tokens computed this step).
            kv_lens (list[int]): KV-cache lengths per request (context + query).
            warmup (int): Number of warm-up iterations (not measured).
            iters (int): Number of measured iterations.

        Returns:
            dict[str, Any]: Measurement results for this rank. Keys:
                - ``"rank"``: local rank index.
                - ``"num_reqs"``: number of requests in the batch.
                - ``"total_q_tokens"``: total query tokens.
                - ``"latency_ms_median"`` / ``"p10"`` / ``"p90"``: CUDA-event latency
                  in milliseconds.
                - ``"activation_bytes_median"``: median activation memory increase
                  in bytes.
                - ``"flops"``: theoretical FLOPs for this step (0 if unavailable).
                - ``"skipped"``: True if the step was skipped due to capacity overflow.
                - ``"skip_reason"``: human-readable reason when ``skipped`` is True.
        """
        assert hasattr(self, "vllm_config"), "vllm_config must be set on this extension."
        assert hasattr(self, "model_runner"), "model_runner must be set on this extension."

        vllm_config = self.vllm_config
        model_runner = self.model_runner
        rank = int(os.getenv("RANK", "0"))

        total_q = sum(q_lens)
        num_reqs = len(q_lens)
        base_result: dict[str, Any] = {
            "rank": rank,
            "num_reqs": num_reqs,
            "total_q_tokens": total_q,
            "latency_ms_median": None,
            "latency_ms_p10": None,
            "latency_ms_p90": None,
            "activation_bytes_median": None,
            "flops": 0,
            "skipped": False,
            "skip_reason": "",
        }

        # --- Capacity Validation ---
        max_tokens = model_runner.max_num_tokens
        max_reqs = model_runner.max_num_reqs
        if total_q > max_tokens:
            msg = (
                f"total_q_tokens={total_q} exceeds max_num_batched_tokens={max_tokens}. "
                "Increase max_num_batched_tokens or reduce the batch size."
            )
            psrl_logger.warning(msg)
            base_result["skipped"] = True
            base_result["skip_reason"] = msg
            return base_result
        if num_reqs > max_reqs:
            msg = (
                f"num_reqs={num_reqs} exceeds max_num_seqs={max_reqs}. "
                "Increase max_num_seqs or reduce the number of requests."
            )
            psrl_logger.warning(msg)
            base_result["skipped"] = True
            base_result["skip_reason"] = msg
            return base_result

        # Actual block size in tokens comes from the first KV cache group spec.
        kv_cache_config = getattr(model_runner, "kv_cache_config", None)
        if kv_cache_config and kv_cache_config.kv_cache_groups:
            blk_sz = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        else:
            # Fallback: parse from vllm_config.
            blk_sz = vllm_config.cache_config.block_size or 16

        total_blocks_needed = sum(_cdiv(kv, blk_sz) for kv in kv_lens)
        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        # Reserve block 0 (null block) and a small safety margin.
        usable_blocks = max(0, num_gpu_blocks - 1)
        if total_blocks_needed > usable_blocks:
            msg = (
                f"Requested {total_blocks_needed} blocks but only {usable_blocks} usable "
                f"(num_gpu_blocks={num_gpu_blocks}, block_size={blk_sz}). "
                "Reduce kv_lens or increase gpu_memory_utilization."
            )
            psrl_logger.warning(msg)
            base_result["skipped"] = True
            base_result["skip_reason"] = msg
            return base_result

        # --- FLOPs Estimation ---
        flops = 0
        try:
            # Build a probe SchedulerOutput once for FLOPs computation.
            probe_ids_for_flops = [f"__flops_probe_{i}" for i in range(num_reqs)]
            sched_flops, _ = _build_scheduler_output(probe_ids_for_flops, q_lens, kv_lens, blk_sz, block_offset=1)
            model_metrics = ModelMetrics(vllm_config)
            if model_metrics.is_enabled():
                perf = model_metrics.get_step_perf_stats_per_gpu(sched_flops)
                flops = perf.num_flops_per_gpu
        except Exception as exc:
            psrl_logger.debug("FLOPs estimation failed: %r.", exc)

        # --- Probe Iterations ---
        latencies_ms: list[float] = []
        activation_deltas: list[int] = []

        for iteration in range(warmup + iters):
            measuring = iteration >= warmup
            req_ids = [f"__probe_{i}" for i in range(num_reqs)]
            # Stagger block allocation by iteration to avoid ID reuse within one run.
            block_start = 1 + iteration * total_blocks_needed
            # Wrap around if we exceed available blocks (safe because we measured above).
            block_start = (block_start % usable_blocks) + 1

            sched_out, _ = _build_scheduler_output(req_ids, q_lens, kv_lens, blk_sz, block_offset=block_start)
            cleanup_out = _build_cleanup_output(req_ids)

            if measuring:
                torch.cuda.reset_peak_memory_stats()
                alloc_before = torch.cuda.memory_allocated()
                t_start = torch.cuda.Event(enable_timing=True)
                t_end = torch.cuda.Event(enable_timing=True)
                t_start.record()

            # --- forward pass ---
            with torch.inference_mode():
                out = model_runner.execute_model(sched_out)
                # execute_model always returns None (deferred sampling path).
                # Must call sample_tokens to clear execute_model_state.
                if out is None:
                    model_runner.sample_tokens(None)

            if measuring:
                t_end.record()
                torch.cuda.synchronize()
                latencies_ms.append(t_start.elapsed_time(t_end))
                peak_alloc = torch.cuda.max_memory_allocated()
                activation_deltas.append(max(0, peak_alloc - alloc_before))

            with torch.inference_mode():
                model_runner.execute_model(cleanup_out)
                # Cleanup returns EMPTY_MODEL_RUNNER_OUTPUT, so sampling is unnecessary.

        # --- Result Aggregation ---
        import statistics

        lat_sorted = sorted(latencies_ms)
        act_sorted = sorted(activation_deltas)
        n = len(lat_sorted)

        def _pct(lst: list[float], p: float) -> float:
            idx = min(int(p * n), n - 1)
            return lst[idx]

        base_result["latency_ms_median"] = statistics.median(lat_sorted)
        base_result["latency_ms_p10"] = _pct(lat_sorted, 0.10)
        base_result["latency_ms_p90"] = _pct(lat_sorted, 0.90)
        base_result["activation_bytes_median"] = statistics.median(act_sorted)
        base_result["flops"] = flops

        psrl_logger.info(
            "Probe result rank=%d total_q=%d num_reqs=%d lat_ms=%.2f (p10=%.2f p90=%.2f) act_MB=%.1f.",
            rank,
            total_q,
            num_reqs,
            base_result["latency_ms_median"],
            base_result["latency_ms_p10"],
            base_result["latency_ms_p90"],
            (base_result["activation_bytes_median"] or 0) / 1024**2,
        )
        return base_result

    def probe_chunked_sequence(
        self,
        total_len: int,
        chunk_size: int,
        prefix_len: int = 0,
        decode_contexts: list[int] | None = None,
        warmup: int = 2,
        iters: int = 5,
    ) -> dict[str, Any]:
        """
        Measure a full chunked prefill sequence across multiple engine steps.

        Each trial processes `total_len` in chunks of `chunk_size`. Optional
        `decode_contexts` requests run beside every chunk.

        Args:
            total_len (int): Total tokens to prefill.
            chunk_size (int): Tokens computed per step. Must not exceed
                `max_num_batched_tokens` minus the decode request budget.
            prefix_len (int): Tokens already present in the KV cache.
            decode_contexts (list[int] | None): Context lengths of concurrent decode
                requests. Each entry uses query length one.
                If None or empty, no decode requests are included.
            warmup (int): Number of discarded warm-up trials.
            iters (int): Number of measured full-sequence trials.

        Returns:
            dict[str, Any]: Per-step and per-sequence latency, activation, and metadata.
        """
        assert hasattr(self, "vllm_config"), "vllm_config must be set on this extension."
        assert hasattr(self, "model_runner"), "model_runner must be set on this extension."

        vllm_config = self.vllm_config
        model_runner = self.model_runner
        rank = int(os.getenv("RANK", "0"))
        dec_ctxs: list[int] = decode_contexts or []
        num_dec = len(dec_ctxs)

        # --- Step Plan ---
        remaining = total_len
        step_q_tokens_prefill: list[int] = []
        while remaining > 0:
            step_q_tokens_prefill.append(min(remaining, chunk_size))
            remaining -= chunk_size
        num_steps = len(step_q_tokens_prefill)
        # Total query tokens per step = prefill chunk + one token per decode req.
        step_total_q = [c + num_dec for c in step_q_tokens_prefill]

        base_result: dict[str, Any] = {
            "rank": rank,
            "total_len": total_len,
            "chunk_size": chunk_size,
            "prefix_len": prefix_len,
            "decode_contexts": dec_ctxs,
            "num_decode_reqs": num_dec,
            "num_steps": num_steps,
            "step_q_tokens": step_total_q,
            "sequence_latency_ms_median": None,
            "sequence_latency_ms_p10": None,
            "sequence_latency_ms_p90": None,
            "step_latency_ms_median": [None] * num_steps,
            "step_latency_ms_p10": [None] * num_steps,
            "step_latency_ms_p90": [None] * num_steps,
            "step_activation_bytes_median": [None] * num_steps,
            "skipped": False,
            "skip_reason": "",
        }

        # --- Capacity Validation ---
        max_tokens = model_runner.max_num_tokens
        max_reqs = model_runner.max_num_reqs

        for s, tq in enumerate(step_total_q):
            if tq > max_tokens:
                msg = (
                    f"Step {s}: total_q_tokens={tq} exceeds max_num_batched_tokens={max_tokens}. "
                    "Reduce chunk_size or decode_contexts count."
                )
                psrl_logger.warning(msg)
                base_result["skipped"] = True
                base_result["skip_reason"] = msg
                return base_result

        if 1 + num_dec > max_reqs:
            msg = f"num_reqs={1 + num_dec} (1 prefill + {num_dec} decode) exceeds max_num_seqs={max_reqs}."
            psrl_logger.warning(msg)
            base_result["skipped"] = True
            base_result["skip_reason"] = msg
            return base_result

        kv_cache_config = getattr(model_runner, "kv_cache_config", None)
        if kv_cache_config and kv_cache_config.kv_cache_groups:
            blk_sz = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        else:
            blk_sz = vllm_config.cache_config.block_size or 16

        num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks or 0
        usable_blocks = max(0, num_gpu_blocks - 1)

        # Full KV length at end of sequence = prefix_len + total_len.
        prefill_kv_len_final = prefix_len + total_len
        blocks_for_prefill = _cdiv(prefill_kv_len_final, blk_sz)
        # Decode requests keep fixed KV length throughout.
        blocks_for_decode = sum(_cdiv(ctx, blk_sz) for ctx in dec_ctxs)
        total_blocks_needed = blocks_for_prefill + blocks_for_decode

        if total_blocks_needed > usable_blocks:
            msg = (
                f"Requested {total_blocks_needed} blocks but only {usable_blocks} usable "
                f"(num_gpu_blocks={num_gpu_blocks}, block_size={blk_sz}, "
                f"prefill_kv_final={prefill_kv_len_final}, decode={blocks_for_decode} blocks). "
                "Reduce total_len, prefix_len, or increase gpu_memory_utilization."
            )
            psrl_logger.warning(msg)
            base_result["skipped"] = True
            base_result["skip_reason"] = msg
            return base_result

        # --- Trials ---
        step_latencies: list[list[float]] = [[] for _ in range(num_steps)]
        step_activations: list[list[int]] = [[] for _ in range(num_steps)]
        sequence_latencies: list[float] = []

        for trial in range(warmup + iters):
            measuring = trial >= warmup
            # Stagger block IDs by trial to avoid write-after-write hazards.
            # Wrap so we stay within usable_blocks.
            pf_block_start = 1 + (trial * total_blocks_needed) % usable_blocks
            dec_block_start = pf_block_start + blocks_for_prefill

            pf_req_id = f"__cp_pf_{trial}"
            dec_req_ids = [f"__cp_dec{d}_{trial}" for d in range(num_dec)]
            all_req_ids = [pf_req_id] + dec_req_ids

            # Preallocate the full sequence so later steps can address KV positions
            # written by earlier steps.
            pf_all_blocks = list(range(pf_block_start, pf_block_start + blocks_for_prefill))

            seq_t_start = torch.cuda.Event(enable_timing=True)
            seq_t_end = torch.cuda.Event(enable_timing=True)

            if measuring:
                seq_t_start.record()

            for step_idx, pf_chunk in enumerate(step_q_tokens_prefill):
                computed_so_far = prefix_len + sum(step_q_tokens_prefill[:step_idx])

                # Re-register the prefill request with the current computed prefix.
                pf_new_req = NewRequestData(
                    req_id=pf_req_id,
                    prompt_token_ids=[_DUMMY_TOKEN_ID] * (prefix_len + total_len),
                    mm_features=[],
                    sampling_params=SamplingParams(
                        temperature=0.0,
                        max_tokens=1,
                        ignore_eos=True,
                    ),
                    pooling_params=None,
                    block_ids=(pf_all_blocks,),
                    num_computed_tokens=computed_so_far,
                    lora_request=None,
                )

                new_reqs = [pf_new_req]
                num_sched: dict[str, int] = {pf_req_id: pf_chunk}

                # Decode requests: also fresh each step (fixed context, q=1).
                cur_dec_block = dec_block_start
                for d_idx, d_ctx in enumerate(dec_ctxs):
                    d_id = dec_req_ids[d_idx]
                    d_n_blocks = _cdiv(d_ctx, blk_sz)
                    d_blocks = list(range(cur_dec_block, cur_dec_block + d_n_blocks))
                    cur_dec_block += d_n_blocks
                    new_reqs.append(
                        NewRequestData(
                            req_id=d_id,
                            prompt_token_ids=[_DUMMY_TOKEN_ID] * d_ctx,
                            mm_features=[],
                            sampling_params=SamplingParams(
                                temperature=0.0,
                                max_tokens=1,
                                ignore_eos=True,
                            ),
                            pooling_params=None,
                            block_ids=(d_blocks,),
                            num_computed_tokens=d_ctx - 1,
                            lora_request=None,
                        )
                    )
                    num_sched[d_id] = 1

                total_q_this_step = pf_chunk + num_dec
                sched_out = SchedulerOutput(
                    scheduled_new_reqs=new_reqs,
                    scheduled_cached_reqs=CachedRequestData.make_empty(),
                    num_scheduled_tokens=num_sched,
                    total_num_scheduled_tokens=total_q_this_step,
                    scheduled_spec_decode_tokens={},
                    scheduled_encoder_inputs={},
                    num_common_prefix_blocks=[],
                    finished_req_ids=set(),
                    free_encoder_mm_hashes=[],
                )
                cleanup_out = _build_cleanup_output(all_req_ids)

                if measuring:
                    torch.cuda.reset_peak_memory_stats()
                    alloc_before = torch.cuda.memory_allocated()
                    t_start = torch.cuda.Event(enable_timing=True)
                    t_end = torch.cuda.Event(enable_timing=True)
                    t_start.record()

                with torch.inference_mode():
                    out = model_runner.execute_model(sched_out)
                    if out is None:
                        model_runner.sample_tokens(None)

                if measuring:
                    t_end.record()
                    torch.cuda.synchronize()
                    step_latencies[step_idx].append(t_start.elapsed_time(t_end))
                    peak_alloc = torch.cuda.max_memory_allocated()
                    step_activations[step_idx].append(max(0, peak_alloc - alloc_before))

                # Remove all requests from input_batch before the next step
                # so the next step's NewRequestData registration starts clean.
                with torch.inference_mode():
                    model_runner.execute_model(cleanup_out)

            if measuring:
                seq_t_end.record()
                torch.cuda.synchronize()
                sequence_latencies.append(seq_t_start.elapsed_time(seq_t_end))

        # --- Aggregation ---
        import statistics

        def _stats(lst: list[float]) -> tuple[float, float, float]:
            lst_s = sorted(lst)
            n = len(lst_s)
            p10 = lst_s[max(0, int(0.10 * n))]
            p90 = lst_s[min(n - 1, int(0.90 * n))]
            return statistics.median(lst_s), p10, p90

        seq_med, seq_p10, seq_p90 = _stats(sequence_latencies)
        base_result["sequence_latency_ms_median"] = seq_med
        base_result["sequence_latency_ms_p10"] = seq_p10
        base_result["sequence_latency_ms_p90"] = seq_p90

        for s in range(num_steps):
            med, p10, p90 = _stats(step_latencies[s])
            base_result["step_latency_ms_median"][s] = med
            base_result["step_latency_ms_p10"][s] = p10
            base_result["step_latency_ms_p90"][s] = p90
            base_result["step_activation_bytes_median"][s] = statistics.median(sorted(step_activations[s]))

        psrl_logger.info(
            "Chunked probe rank=%d total_len=%d chunk=%d prefix=%d num_dec=%d num_steps=%d seq_ms=%.2f (steps: %s).",
            rank,
            total_len,
            chunk_size,
            prefix_len,
            num_dec,
            num_steps,
            seq_med,
            " ".join(f"{v:.1f}" for v in base_result["step_latency_ms_median"]),
        )
        return base_result

    def get_memory_breakdown(self) -> dict[str, Any]:
        """
        Return a snapshot of this worker's GPU memory accounting.

        All byte values are raw integers. Callers may divide by `1024**3` for GiB.

        Returns:
            dict[str, Any]: Device, weight, activation, and KV-cache memory fields.
        """
        assert hasattr(self, "vllm_config"), "vllm_config must be set on this extension."
        assert hasattr(self, "model_runner"), "model_runner must be set on this extension."

        vllm_config = self.vllm_config
        model_runner = self.model_runner
        rank = int(os.getenv("RANK", "0"))

        # Worker-level memory attributes (set in gpu_worker.py:375-526).
        worker = self  # extension is mixed into Worker
        requested_bytes: int = getattr(worker, "requested_memory", 0)
        peak_activation_bytes: int = getattr(worker, "peak_activation_memory", 0)
        non_torch_bytes: int = getattr(worker, "non_torch_memory", 0)
        available_kv_bytes: int = getattr(worker, "available_kv_cache_memory_bytes", 0)

        # Total device memory.
        _, total_gpu_bytes = torch.cuda.mem_get_info()

        # Weight memory from model runner.
        weights_bytes: int = int(getattr(model_runner, "model_memory_usage", 0))

        # KV cache configuration.
        cache_config = vllm_config.cache_config
        num_gpu_blocks: int = cache_config.num_gpu_blocks or 0
        gpu_memory_utilization: float = cache_config.gpu_memory_utilization

        kv_cache_config = getattr(model_runner, "kv_cache_config", None)
        if kv_cache_config and kv_cache_config.kv_cache_groups:
            block_size_tokens: int = kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            kv_token_capacity = int(
                get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)
                * vllm_config.model_config.max_model_len
            )
        else:
            block_size_tokens = cache_config.block_size or 16
            kv_token_capacity = num_gpu_blocks * block_size_tokens

        result: dict[str, Any] = {
            "rank": rank,
            "total_gpu_bytes": total_gpu_bytes,
            "requested_bytes": requested_bytes,
            "weights_bytes": weights_bytes,
            "peak_activation_bytes": peak_activation_bytes,
            "non_torch_bytes": non_torch_bytes,
            "available_kv_bytes": available_kv_bytes,
            "num_gpu_blocks": num_gpu_blocks,
            "block_size_tokens": block_size_tokens,
            "kv_token_capacity": kv_token_capacity,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        psrl_logger.info(
            "Memory breakdown rank=%d total=%.1fGiB requested=%.1fGiB weights=%.1fGiB peak_act=%.1fGiB kv_tokens=%dk.",
            rank,
            total_gpu_bytes / 1024**3,
            requested_bytes / 1024**3,
            weights_bytes / 1024**3,
            peak_activation_bytes / 1024**3,
            kv_token_capacity // 1000,
        )
        return result
