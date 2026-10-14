#!/usr/bin/env python3
"""
Profile GPU memory for Transformers causal-LM inference with TP / EP.

Loads a model via ``AutoModelForCausalLM`` (same kwargs as
``PSRL_TransformersRollout``), feeds a random fixed-length token sequence,
runs one forward pass, and returns top-k logprobs for each scored token in the
sequence (same slicing as ``PSRL_TransformersRollout._score_single`` / OPD teacher).
Memory is logged at each phase on every visible GPU.

Launch with ``torchrun`` when ``--tp-size`` > 1 (and optionally ``--ep-size``).
Example::

    torchrun --nproc_per_node=8 examples/opd_trainer/profile_transformers_infer_memory.py \\
        --model-path /path/to/model \\
        --seq-len 4096 --topk 16 --tp-size 8 --ep-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from psrl.utils.logger.memory_logger import format_memory_message, get_all_gpu_memory_info, log_gpu_memory_now
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.distributed import DistributedConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile Transformers TP/EP inference memory.")
    parser.add_argument("--model-path", type=str, required=True, help="HuggingFace model path or hub id.")
    parser.add_argument("--seq-len", type=int, default=4096, help="Fixed random input sequence length.")
    parser.add_argument("--topk", type=int, default=16, help="Top-k logprobs per scored token position.")
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=0,
        help="Treat the first prompt_len tokens as prompt (not scored). "
        "OPD teacher uses prompt+response; only response tokens are scored.",
    )
    parser.add_argument(
        "--response-len",
        type=int,
        default=None,
        help="Number of trailing tokens to score (teacher response length). "
        "Default: seq_len - prompt_len (score all non-prompt tokens).",
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size (must equal torchrun world_size)."
    )
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size; must equal tp-size when > 1.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32", "auto"))
    parser.add_argument("--use-kv-cache", action="store_true", help="Pass use_cache=True to the model forward.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-warmup", type=int, default=1, help="Warmup forward passes before profiling.")
    parser.add_argument("--num-iters", type=int, default=1, help="Timed/profiled forward passes.")
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Enable torch.profiler (profile_memory=True) for a single forward pass; exports TensorBoard traces.",
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=str,
        default="torch_profiler_traces",
        help="Base directory for torch profiler traces.",
    )
    parser.add_argument(
        "--torch-profiler-top-events",
        type=int,
        default=30,
        help="How many operator events to summarize by CUDA memory usage.",
    )
    parser.add_argument(
        "--torch-profiler-every-rank",
        action="store_true",
        help="If not set, only rank 0 will run torch.profiler to reduce overhead.",
    )
    parser.add_argument("--log-dir", type=str, default=".", help="Directory for rank-0 memory log file.")
    parser.add_argument("--log-prefix", type=str, default="profile_transformers_infer")
    return parser.parse_args()


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype in ("fp16", "float16", "half"):
        return torch.float16
    if dtype in ("fp32", "float32"):
        return torch.float32
    if dtype == "auto":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype!r}")


def init_distributed() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and not dist.is_initialized():
        backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size


def select_device(rank: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    local_rank = local_rank % max(torch.cuda.device_count(), 1)
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def validate_parallel_args(tp_size: int, ep_size: int, world_size: int) -> None:
    if world_size != tp_size:
        raise ValueError(f"world_size ({world_size}) must equal --tp-size ({tp_size}).")
    if ep_size > 1 and ep_size != tp_size:
        raise ValueError(f"When --ep-size > 1 it must equal --tp-size; got ep_size={ep_size}, tp_size={tp_size}.")


def load_model(
    model_path: str,
    dtype: torch.dtype,
    device_mesh,
    tp_size: int,
    ep_size: int,
    trust_remote_code: bool,
    device: torch.device,
):
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    kwargs: dict = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": getattr(hf_config, "_attn_implementation", None),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}

    if tp_size > 1:
        kwargs["device_mesh"] = device_mesh
        kwargs["tp_size"] = tp_size
        if ep_size > 1:
            if not getattr(hf_config, "base_model_ep_plan", None):
                raise NotImplementedError(
                    f"ep_size > 1 requires a model config with base_model_ep_plan (model={model_path!r})."
                )
            kwargs["distributed_config"] = DistributedConfig(enable_expert_parallel=True)
        else:
            kwargs["tp_plan"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if tp_size == 1:
        model.to(device)
    model.config.use_cache = False
    model.eval()
    return model, hf_config


def build_random_input_ids(
    seq_len: int,
    vocab_size: int,
    pad_token_id: int | None,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=gen, dtype=torch.long)
    if pad_token_id is not None:
        input_ids = input_ids.clone()
        input_ids[input_ids == pad_token_id] = (pad_token_id + 1) % vocab_size
    return input_ids.to(device)


def resolve_response_len(seq_len: int, prompt_len: int, response_len: int | None) -> int:
    if prompt_len < 0 or prompt_len >= seq_len:
        raise ValueError(f"prompt_len must be in [0, seq_len), got {prompt_len=} {seq_len=}.")
    if response_len is None:
        # OPD teacher scores response tokens only; when prompt_len==0, at most seq_len-1 positions
        # have a prior logit (token 0 is never scored).
        resolved = seq_len - prompt_len if prompt_len > 0 else seq_len - 1
    else:
        resolved = response_len
    if resolved <= 0 or resolved >= seq_len:
        raise ValueError(f"response_len must be in (0, seq_len), got {resolved=} {seq_len=} {prompt_len=}.")
    if prompt_len > 0 and prompt_len + resolved > seq_len:
        raise ValueError(f"prompt_len + response_len exceeds seq_len: {prompt_len=} {resolved=} {seq_len=}.")
    return resolved


@torch.no_grad()
def forward_sequence_topk(
    model,
    input_ids: torch.Tensor,
    response_len: int,
    topk: int,
    use_cache: bool,
) -> tuple[list[list[float]], list[list[int]]]:
    """
    Single forward on a length-L sequence; top-k logprobs for ``response_len`` trailing tokens.

    Matches ``PSRL_TransformersRollout._score_single``: for each response token at index t,
    use logits[t - 1] to compute logprobs (teacher prefill path in OPD).
    """
    seq_len = input_ids.shape[1]
    if response_len <= 0 or response_len >= seq_len:
        raise ValueError(f"response_len must be in (0, seq_len), got {response_len=} {seq_len=}.")

    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
    logits = outputs.logits
    if hasattr(logits, "to_local"):
        logits = logits.to_local()
    logits = logits[0]

    response_start = seq_len - response_len
    # Slice on GPU, free forward intermediates, then compute logprobs on CPU.
    target_logits = logits[response_start - 1 : seq_len - 1].detach().float().contiguous()
    del outputs, logits, attention_mask
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    target_logits = target_logits.cpu()

    log_probs = torch.log_softmax(target_logits, dim=-1)
    del target_logits

    k = min(topk, log_probs.shape[-1])
    top_log_probs, top_ids = torch.topk(log_probs, k=k, dim=-1)
    del log_probs
    return top_log_probs.tolist(), top_ids.tolist()


def log_phase(rank: int, phase: str, log_dir: str, log_prefix: str) -> str:
    prefix = f"[rank={rank}] {phase}"
    message = log_gpu_memory_now(prefix=prefix)
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{log_prefix}_memory.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    return message


def main() -> None:
    args = parse_args()
    rank, world_size = init_distributed()
    validate_parallel_args(args.tp_size, args.ep_size, world_size)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU memory profiling.")

    device = select_device(rank)
    dtype = resolve_dtype(args.dtype)
    device_mesh = init_device_mesh("cuda", (world_size,)) if world_size > 1 else None

    torch.manual_seed(args.seed)
    if rank == 0:
        print(
            f"Profiling model={args.model_path!r} seq_len={args.seq_len} topk={args.topk} "
            f"tp={args.tp_size} ep={args.ep_size} dtype={dtype} use_cache={args.use_kv_cache}"
        )

    log_phase(rank, "baseline_before_load", args.log_dir, args.log_prefix)

    t0 = time.monotonic()
    model, hf_config = load_model(
        args.model_path,
        dtype,
        device_mesh,
        args.tp_size,
        args.ep_size,
        args.trust_remote_code,
        device,
    )
    model.config.use_cache = args.use_kv_cache
    load_time = time.monotonic() - t0
    log_phase(rank, f"after_model_load ({load_time:.2f}s)", args.log_dir, args.log_prefix)

    vocab_size = int(getattr(hf_config, "vocab_size", 32000))
    pad_token_id = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        pad_token_id = tokenizer.pad_token_id
        if vocab_size <= 0:
            vocab_size = len(tokenizer)
    except Exception:
        pass

    input_ids = build_random_input_ids(args.seq_len, vocab_size, pad_token_id, args.seed, device)
    response_len = resolve_response_len(args.seq_len, args.prompt_len, args.response_len)
    if rank == 0:
        print(f"Scoring response_len={response_len} tokens (prompt_len={args.prompt_len}, seq_len={args.seq_len})")

    def run_forward() -> tuple[list[list[float]], list[list[int]]]:
        return forward_sequence_topk(model, input_ids, response_len, args.topk, args.use_kv_cache)

    for i in range(args.num_warmup):
        run_forward()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    log_phase(rank, f"after_warmup (n={args.num_warmup})", args.log_dir, args.log_prefix)

    times: list[float] = []
    top_log_probs: list[list[float]] | None = None
    top_ids: list[list[int]] | None = None

    def maybe_run_torch_profiler_one_forward() -> tuple[list[list[float]] | None, list[list[int]] | None]:
        """Run exactly one forward under torch profiler and summarize CUDA memory events (rank0)."""
        if not args.torch_profiler:
            return None, None
        if not args.torch_profiler_every_rank and rank != 0:
            # Run the forward normally on non-profiler ranks to keep ranks' outputs aligned.
            run_forward()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            return None, None

        trace_dir = os.path.join(
            args.log_dir, args.torch_profiler_dir, f"{args.log_prefix}_seq{args.seq_len}_rank{rank}"
        )
        os.makedirs(trace_dir, exist_ok=True)

        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=False,
            with_stack=False,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        ) as prof:
            # One forward only (the operator-level memory stats come from this step).
            out_log_probs, out_ids = run_forward()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            prof.step()

        if rank == 0:
            # Summarize operator memory usage; useful for locating the biggest activation/temporary tensors.
            key_avgs = prof.key_averages()
            rows: list[dict] = []
            for avg in key_avgs:
                self_mem = getattr(avg, "self_cuda_memory_usage", None)
                total_mem = getattr(avg, "total_cuda_memory_usage", None)
                if self_mem is None and total_mem is None:
                    continue
                rows.append(
                    {
                        "key": str(avg.key),
                        "count": int(getattr(avg, "count", 0)),
                        "self_cuda_memory_usage_bytes": int(self_mem) if self_mem is not None else None,
                        "total_cuda_memory_usage_bytes": int(total_mem) if total_mem is not None else None,
                    }
                )

            # Prefer sorting by self_cuda_memory_usage (peak-ish for the op itself), fallback to total.
            rows.sort(
                key=lambda x: (
                    x["self_cuda_memory_usage_bytes"] if x["self_cuda_memory_usage_bytes"] is not None else -1
                ),
                reverse=True,
            )
            rows = rows[: args.torch_profiler_top_events]

            summary_path = os.path.join(args.log_dir, f"{args.log_prefix}_torch_profiler_top_ops.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

            try:
                print(key_avgs.table(sort_by="self_cuda_memory_usage", row_limit=args.torch_profiler_top_events))
            except Exception:
                # Table output is best-effort; json summary is the reliable artifact.
                pass
            print(f"[torch profiler] trace_dir={trace_dir}")
            print(f"[torch profiler] top-ops json={summary_path}")

        return out_log_probs, out_ids

    # Optional: run one torch-profiler forward (operator-level memory events).
    prof_log_probs, prof_ids = maybe_run_torch_profiler_one_forward()
    if prof_log_probs is not None:
        top_log_probs, top_ids = prof_log_probs, prof_ids

    for i in range(args.num_iters):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        t_start = time.monotonic()
        top_log_probs, top_ids = run_forward()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        times.append(time.monotonic() - t_start)
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        log_phase(
            rank,
            f"after_forward iter={i} ({times[-1] * 1000:.2f}ms peak_alloc={peak_gb:.2f}GB)",
            args.log_dir,
            args.log_prefix,
        )

    if rank == 0:
        device_infos = get_all_gpu_memory_info()
        summary = format_memory_message(device_infos, prefix="[final rank=0]")
        print(summary)
        print(f"Forward latency (ms): {[t * 1000 for t in times]}")
        num_positions = len(top_log_probs or [])
        num_topk = len((top_log_probs or [[]])[0]) if num_positions > 0 else 0
        print(f"teacher topk logprobs shape: ({num_positions}, {num_topk})  # (response_len, topk)")
        if num_positions > 0:
            print(f"sample top_ids[0][:5]: {top_ids[0][:5]}")
            print(f"sample top_log_probs[0][:5]: {top_log_probs[0][:5]}")
            print(f"sample top_ids[-1][:5]: {top_ids[-1][:5]}")
        log_path = os.path.join(args.log_dir, f"{args.log_prefix}_memory.log")
        print(f"Memory log: {log_path}")
        profiler_json = os.path.join(args.log_dir, f"{args.log_prefix}_torch_profiler_top_ops.json")
        if args.torch_profiler:
            print(f"Torch profiler top-ops json: {profiler_json}")
            print(
                f"Torch profiler traces: "
                f"{os.path.join(args.log_dir, args.torch_profiler_dir, f'{args.log_prefix}_seq{args.seq_len}_rank0')}"
            )
        else:
            print("Torch profiler json not generated (rerun with --torch-profiler or TORCH_PROFILER=true).")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
