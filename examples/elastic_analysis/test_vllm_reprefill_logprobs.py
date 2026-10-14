"""Compare vLLM continuous decode with abort + full-prefix reprefill.

The experiment uses one vLLM ``AsyncLLM`` instance and the same tokenized
prompt for every run:

  continuous decode -> completion
  decode a chunk -> abort -> submit the complete prefix again -> ...

The public vLLM generation API exposes sampled-token logprobs, not raw logits.
This script therefore compares chosen-token logprobs and, when requested,
the returned top-k/all-token logprob maps.  It also requests ``prompt_logprobs``
for reprefill requests.  Those values are the direct prefill-vs-decode check
for tokens that were already generated before an interruption.

Run in the repository environment, for example:

  source env/env_311.sh
  PYTHONPATH=. python scripts/test_vllm_reprefill_logprobs.py \
      --model /path/to/model --prompt-token-ids '[1, 2, 3, 4]' \
      --total-tokens 32 --chunk-size 4 --output /tmp/reprefill.json

Use ``--self-test`` to validate the comparison logic without a model or GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _parse_token_ids(value: str) -> list[int]:
    """Parse a JSON list or a comma-separated token-id list."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(parsed, list) or not parsed:
        raise argparse.ArgumentTypeError("prompt token ids must be a non-empty list")
    try:
        token_ids = [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("prompt token ids must contain integers") from exc
    if any(token_id < 0 for token_id in token_ids):
        raise argparse.ArgumentTypeError("prompt token ids must be non-negative")
    return token_ids


def _parse_positive_int_list(value: str) -> list[int]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    try:
        parsed = [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("chunk sizes must be comma-separated integers") from exc
    if not parsed or any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("chunk sizes must all be positive")
    return parsed


def _float_logprob(value: Any) -> float:
    raw = getattr(value, "logprob", value)
    if isinstance(raw, dict):
        raw = raw["logprob"]
    return float(raw)


def _logprob_map(entry: Any) -> dict[str, float]:
    """Convert list[dict] and vLLM FlatLogprobs entries to JSON-safe data."""
    if entry is None:
        return {}
    if hasattr(entry, "items"):
        items = entry.items()
    else:
        items = []
    return {str(int(token_id)): _float_logprob(logprob) for token_id, logprob in items}


def _records_from_output(token_ids: Any, logprobs: Any) -> list[dict[str, Any]]:
    """Extract one record per token from a vLLM CompletionOutput field."""
    token_ids = [int(token_id) for token_id in token_ids]
    if logprobs is None:
        raise RuntimeError("vLLM returned no output logprobs; keep --logprobs enabled")
    entries = list(logprobs)
    if len(entries) != len(token_ids):
        raise RuntimeError(
            "vLLM output token/logprob length mismatch: "
            f"tokens={len(token_ids)} logprobs={len(entries)}"
        )

    records = []
    for token_id, entry in zip(token_ids, entries, strict=True):
        values = _logprob_map(entry)
        chosen = values.get(str(token_id))
        if chosen is None:
            raise RuntimeError(
                f"chosen token {token_id} is absent from the returned logprob map"
            )
        records.append(
            {
                "token_id": token_id,
                "logprob": chosen,
                "top_logprobs": values,
            }
        )
    return records


def _prompt_records(prompt_token_ids: Any, prompt_logprobs: Any) -> list[dict[str, Any]]:
    """Extract prompt-token logprobs, retaining None for vLLM's first token."""
    token_ids = [int(token_id) for token_id in prompt_token_ids]
    if prompt_logprobs is None:
        return []
    entries = list(prompt_logprobs)
    if len(entries) != len(token_ids):
        raise RuntimeError(
            "vLLM prompt token/logprob length mismatch: "
            f"tokens={len(token_ids)} logprobs={len(entries)}"
        )
    records = []
    for token_id, entry in zip(token_ids, entries, strict=True):
        values = _logprob_map(entry)
        chosen = values.get(str(token_id))
        records.append(
            {
                "token_id": token_id,
                "logprob": chosen,
                "top_logprobs": values,
            }
        )
    return records


def _compare_records(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    significance: float,
) -> dict[str, Any]:
    """Compare aligned token records and their chosen/top-k logprobs."""
    common = min(len(reference), len(candidate))
    token_mismatches = []
    chosen_diffs = []
    top_diffs = []
    for index in range(common):
        ref = reference[index]
        got = candidate[index]
        if ref["token_id"] != got["token_id"]:
            token_mismatches.append(
                {
                    "index": index,
                    "reference": ref["token_id"],
                    "candidate": got["token_id"],
                }
            )
        if ref.get("logprob") is not None and got.get("logprob") is not None:
            chosen_diffs.append(abs(float(ref["logprob"]) - float(got["logprob"])))
        ref_top = ref.get("top_logprobs", {})
        got_top = got.get("top_logprobs", {})
        for token_id in set(ref_top) | set(got_top):
            if token_id in ref_top and token_id in got_top:
                top_diffs.append(abs(float(ref_top[token_id]) - float(got_top[token_id])))

    return {
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
        "compared_positions": common,
        "same_length": len(reference) == len(candidate),
        "same_token_ids": not token_mismatches and len(reference) == len(candidate),
        "token_mismatches": token_mismatches[:20],
        "token_mismatch_count": len(token_mismatches),
        "max_abs_chosen_logprob_diff": max(chosen_diffs, default=None),
        "mean_abs_chosen_logprob_diff": (
            sum(chosen_diffs) / len(chosen_diffs) if chosen_diffs else None
        ),
        "max_abs_returned_logprob_diff": max(top_diffs, default=None),
        "num_chosen_diffs_above_significance": sum(
            diff > significance for diff in chosen_diffs
        ),
        "classification": _classification(
            same_tokens=not token_mismatches and len(reference) == len(candidate),
            max_diff=max(chosen_diffs, default=None),
            significance=significance,
        ),
    }


def _classification(
    *, same_tokens: bool, max_diff: float | None, significance: float
) -> str:
    if not same_tokens:
        return "token_sequence_diverged"
    if max_diff is None:
        return "logprob_unavailable"
    if max_diff > significance:
        return "logprob_difference_observed"
    return "no_difference_above_significance"


def _chunk_schedule(total_tokens: int, sizes: list[int]) -> list[int]:
    boundaries = []
    consumed = 0
    index = 0
    while consumed < total_tokens:
        consumed = min(total_tokens, consumed + sizes[index % len(sizes)])
        boundaries.append(consumed)
        index += 1
    return boundaries


async def _collect_request(
    engine: Any,
    *,
    prompt_token_ids: list[int],
    sampling_params: Any,
    request_id: str,
    abort_after: int | None,
    timeout_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Collect a request, optionally aborting after N emitted tokens."""
    from vllm import TokensPrompt

    latest_output = None
    abort_sent = False
    async def consume() -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
        nonlocal latest_output, abort_sent
        stream = engine.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        async for output in stream:
            latest_output = output
            completion = output.outputs[0]
            current_tokens = list(completion.token_ids)
            if (
                abort_after is not None
                and not abort_sent
                and len(current_tokens) >= abort_after
                and not output.finished
            ):
                await engine.abort(request_id)
                abort_sent = True
            if output.finished:
                break
        if latest_output is None:
            raise RuntimeError(f"request {request_id} produced no output")
        completion = latest_output.outputs[0]
        was_aborted = abort_sent and str(completion.finish_reason).lower() == "abort"
        return (
            _records_from_output(completion.token_ids, completion.logprobs),
            _prompt_records(latest_output.prompt_token_ids or [], latest_output.prompt_logprobs),
            was_aborted,
        )

    return await asyncio.wait_for(consume(), timeout=timeout_s)


def _sampling_params(args: argparse.Namespace, max_tokens: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=max_tokens,
        ignore_eos=True,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
    )


async def _run_experiment(args: argparse.Namespace, prompt_token_ids: list[int]) -> dict[str, Any]:
    import inspect

    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "trust_remote_code": args.trust_remote_code,
        "download_dir": args.download_dir,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 1,
        "max_logprobs": (
            -1
            if args.logprobs == -1 or args.prompt_logprobs == -1
            else max(args.logprobs, args.prompt_logprobs, 1)
        ),
        "enable_prefix_caching": args.enable_prefix_caching,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enforce_eager": args.enforce_eager,
        "stream_interval": 1,
        "disable_log_stats": True,
        "enable_log_requests": False,
    }
    accepted = set(inspect.signature(AsyncEngineArgs).parameters)
    engine_kwargs = {key: value for key, value in engine_kwargs.items() if key in accepted and value is not None}
    engine_args = AsyncEngineArgs(**engine_kwargs)
    engine = AsyncLLM.from_engine_args(engine_args)

    def params(max_tokens: int) -> Any:
        return _sampling_params(args, max_tokens=max_tokens)

    try:
        continuous_runs = []
        for run_index in range(2):
            records, prompt_records, aborted = await _collect_request(
                engine,
                prompt_token_ids=prompt_token_ids,
                sampling_params=params(args.total_tokens),
                request_id=f"continuous-{run_index}",
                abort_after=None,
                timeout_s=args.timeout_s,
            )
            if aborted or len(records) != args.total_tokens:
                raise RuntimeError(
                    f"continuous run returned {len(records)} tokens; expected {args.total_tokens}"
                )
            continuous_runs.append(
                {"records": records, "prompt_records": prompt_records}
            )

        boundaries = _chunk_schedule(args.total_tokens, args.chunk_sizes)
        interrupted_records: list[dict[str, Any]] = []
        prefill_by_index: dict[int, dict[str, Any]] = {}
        segments = []
        consumed = 0
        for segment_index, boundary in enumerate(boundaries):
            current_prompt = prompt_token_ids + [item["token_id"] for item in interrupted_records]
            segment_records, prompt_records, aborted = await _collect_request(
                engine,
                prompt_token_ids=current_prompt,
                sampling_params=params(args.total_tokens - consumed),
                request_id=f"interrupted-{segment_index}",
                abort_after=boundary - consumed,
                timeout_s=args.timeout_s,
            )
            expected_segment_len = boundary - consumed
            if not segment_records:
                raise RuntimeError(f"interrupted segment {segment_index} returned no tokens")
            if boundary < args.total_tokens and not aborted:
                raise RuntimeError(
                    f"interrupted segment {segment_index} did not finish with vLLM abort"
                )
            if len(segment_records) > expected_segment_len:
                segment_records = segment_records[:expected_segment_len]
            interrupted_records.extend(segment_records)
            consumed = len(interrupted_records)

            for prompt_index, record in enumerate(prompt_records):
                generated_index = prompt_index - len(prompt_token_ids)
                if generated_index >= 0 and record.get("logprob") is not None:
                    prefill_by_index[generated_index] = record

            segments.append(
                {
                    "segment_index": segment_index,
                    "prompt_length": len(current_prompt),
                    "requested_new_tokens": expected_segment_len,
                    "returned_new_tokens": len(segment_records),
                    "abort_sent": aborted,
                    "records": segment_records,
                    "prompt_logprob_records": prompt_records,
                }
            )
            if consumed >= args.total_tokens:
                break

        if len(interrupted_records) != args.total_tokens:
            raise RuntimeError(
                f"interrupted run returned {len(interrupted_records)} tokens; "
                f"expected {args.total_tokens}"
            )

        # Score the final complete prefix too. This covers the final chunk,
        # which has no later reprefill request in the segmented run.
        final_prompt = prompt_token_ids + [item["token_id"] for item in interrupted_records]
        _, final_prompt_records, _ = await _collect_request(
            engine,
            prompt_token_ids=final_prompt,
            sampling_params=params(1),
            request_id="final-prefix-score",
            abort_after=None,
            timeout_s=args.timeout_s,
        )
        for prompt_index, record in enumerate(final_prompt_records):
            generated_index = prompt_index - len(prompt_token_ids)
            if generated_index >= 0 and record.get("logprob") is not None:
                prefill_by_index[generated_index] = record

        prefill_records = [prefill_by_index[index] for index in sorted(prefill_by_index)]
        baseline_repeat = _compare_records(
            continuous_runs[0]["records"],
            continuous_runs[1]["records"],
            significance=args.significance,
        )
        interrupted_decode = _compare_records(
            continuous_runs[0]["records"], interrupted_records, significance=args.significance
        )
        interrupted_prefill = _compare_records(
            continuous_runs[0]["records"], prefill_records, significance=args.significance
        )

        return {
            "status": "completed",
            "metadata": {
                "model": args.model,
                "tokenizer": args.tokenizer or args.model,
                "prompt_token_ids": prompt_token_ids,
                "prompt_length": len(prompt_token_ids),
                "total_tokens": args.total_tokens,
                "chunk_sizes": args.chunk_sizes,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "seed": args.seed,
                "logprobs_requested": args.logprobs,
                "prompt_logprobs_requested": args.prompt_logprobs,
                "enable_prefix_caching": args.enable_prefix_caching,
                "enable_chunked_prefill": args.enable_chunked_prefill,
                "raw_logits_returned": False,
                "note": "vLLM public output exposes logprobs, not raw per-token logits.",
            },
            "continuous": continuous_runs[0],
            "continuous_repeat": continuous_runs[1],
            "interrupted_reprefill": {
                "segments": segments,
                "records": interrupted_records,
                "final_prompt_logprob_records": final_prompt_records,
                "prefill_records_aligned_to_generated_tokens": prefill_records,
            },
            "comparisons": {
                "continuous_vs_repeat_baseline": baseline_repeat,
                "continuous_decode_vs_interrupted_decode": interrupted_decode,
                "continuous_decode_vs_reprefill_prompt": interrupted_prefill,
            },
        }
    finally:
        engine.shutdown()


def _self_test() -> None:
    reference = [
        {"token_id": 1, "logprob": -0.1, "top_logprobs": {"1": -0.1, "2": -1.0}},
        {"token_id": 2, "logprob": -0.2, "top_logprobs": {"2": -0.2}},
    ]
    same = _compare_records(reference, list(reference), significance=1e-3)
    assert same["same_token_ids"] and same["max_abs_chosen_logprob_diff"] == 0.0
    different = [dict(reference[0]), {**reference[1], "logprob": -0.203}]
    changed = _compare_records(reference, different, significance=1e-3)
    assert changed["classification"] == "logprob_difference_observed"
    assert _chunk_schedule(10, [3, 2]) == [3, 5, 8, 10]
    print("self-test: PASS")


def _prompt_ids(args: argparse.Namespace) -> list[int]:
    if args.prompt_token_ids is not None:
        return _parse_token_ids(args.prompt_token_ids)
    from vllm.tokenizers import get_tokenizer

    tokenizer = get_tokenizer(
        args.tokenizer or args.model,
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
    )
    try:
        encoded = tokenizer.encode(
            args.prompt, add_special_tokens=args.add_special_tokens
        )
        return [int(token_id) for token_id in encoded]
    except TypeError:
        return [int(token_id) for token_id in tokenizer.encode(args.prompt)]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="vLLM/Hugging Face model path or model ID")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--prompt", default="Explain why numerical reproducibility matters.")
    parser.add_argument("--prompt-token-ids", default=None, help="JSON list or comma-separated exact token IDs")
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument("--total-tokens", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--chunk-sizes", type=_parse_positive_int_list, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logprobs", type=int, default=1, help="-1 requests all returned vocabulary logprobs")
    parser.add_argument("--prompt-logprobs", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--significance", type=float, default=1e-3)
    parser.add_argument("--output", default="vllm_reprefill_logprobs_report.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_test:
        _self_test()
        return 0
    if not args.model:
        print("error: --model is required unless --self-test is used", file=sys.stderr)
        return 2
    if args.total_tokens < 1 or args.chunk_size < 1:
        print("error: --total-tokens and --chunk-size must be positive", file=sys.stderr)
        return 2
    args.chunk_sizes = args.chunk_sizes or [args.chunk_size]
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception as exc:  # pragma: no cover - environment-specific import failure
        cuda_available = False
        print(f"GPU probe failed: {exc}", file=sys.stderr)
    if not cuda_available:
        result = {
            "status": "skipped_no_gpu",
            "reason": "torch.cuda.is_available() is false or no CUDA device is visible",
            "require_gpu": args.require_gpu,
        }
        print(json.dumps(result, indent=2))
        return 1 if args.require_gpu else 0

    started = time.time()
    try:
        prompt_token_ids = _prompt_ids(args)
        result = asyncio.run(_run_experiment(args, prompt_token_ids))
    except Exception as exc:
        print(f"experiment failed: {exc}", file=sys.stderr)
        return 1
    result["metadata"] = {
        **result["metadata"],
        "started_unix_s": started,
        "duration_s": time.time() - started,
        "hostname": os.uname().nodename,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["comparisons"], indent=2))
    print(f"full report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
