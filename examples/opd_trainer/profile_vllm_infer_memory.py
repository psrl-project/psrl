#!/usr/bin/env python3
"""
Profile GPU memory for vLLM causal-LM inference with TP / EP (OPD teacher path).

Uses ``AsyncLLM`` with ``prompt_logprobs`` + ``max_tokens=1`` (same as OPD teacher in
``PSRL_vLLMRollout`` / ``opd.py``), feeds a random fixed-length token prompt, and
extracts top-k logprobs for the trailing ``response_len`` tokens.

Single-process launch (vLLM manages TP/EP internally). Example::

    python examples/opd_trainer/profile_vllm_infer_memory.py \\
        --model-path models/Qwen3-235B-A22B \\
        --seq-len 16384 --topk 16 --tp-size 8 --ep-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import time
import uuid

import torch
from psrl.utils.logger.memory_logger import format_memory_message, get_all_gpu_memory_info, log_gpu_memory_now
from transformers import AutoConfig, AutoTokenizer
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from vllm.v1.engine.async_llm import AsyncLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile vLLM TP/EP inference memory (OPD teacher path).")
    parser.add_argument("--model-path", type=str, required=True, help="HuggingFace model path or hub id.")
    parser.add_argument("--seq-len", type=int, default=4096, help="Fixed random prompt length (token ids).")
    parser.add_argument("--topk", type=int, default=16, help="Top-k logprobs per scored token (prompt_logprobs).")
    parser.add_argument(
        "--prompt-len", type=int, default=0, help="Leading prompt tokens (not scored); see --response-len."
    )
    parser.add_argument(
        "--response-len",
        type=int,
        default=None,
        help=(
            "Trailing tokens to score via prompt_logprobs. "
            "Default: seq_len - prompt_len (or seq_len-1 if prompt_len=0)."
        ),
    )
    parser.add_argument("--tp-size", type=int, default=1, help="vLLM tensor_parallel_size.")
    parser.add_argument(
        "--ep-size", type=int, default=1, help="vLLM expert parallel (enable_expert_parallel when > 1)."
    )
    parser.add_argument("--pp-size", type=int, default=1, help="vLLM pipeline_parallel_size.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32", "auto"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-model-len", type=int, default=None, help="Default: seq_len + 1.")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None, help="Default: max_model_len.")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--num-iters", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default=".")
    parser.add_argument("--log-prefix", type=str, default="profile_vllm_infer")
    parser.add_argument(
        "--vllm-profiler",
        action="store_true",
        help="Enable vLLM built-in torch profiler via profiler_config + llm.start_profile/stop_profile.",
    )
    parser.add_argument(
        "--vllm-profiler-dir",
        type=str,
        default="torch_profiler_traces_vllm",
        help="Trace base directory. If relative, it's resolved under --log-dir. Must be absolute in vLLM config.",
    )
    parser.add_argument("--vllm-profiler-wait", type=int, default=0, help="Profiler schedule wait iterations.")
    parser.add_argument("--vllm-profiler-warmup", type=int, default=1, help="Profiler schedule warmup iterations.")
    parser.add_argument("--vllm-profiler-active", type=int, default=1, help="Profiler schedule active iterations.")
    parser.add_argument(
        "--vllm-profiler-ignore-frontend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer ignore frontend profiling to reduce overhead.",
    )
    return parser.parse_args()


def resolve_response_len(seq_len: int, prompt_len: int, response_len: int | None) -> int:
    if prompt_len < 0 or prompt_len >= seq_len:
        raise ValueError(f"prompt_len must be in [0, seq_len), got {prompt_len=} {seq_len=}.")
    if response_len is None:
        resolved = seq_len - prompt_len if prompt_len > 0 else seq_len - 1
    else:
        resolved = response_len
    if resolved <= 0 or resolved >= seq_len:
        raise ValueError(f"response_len must be in (0, seq_len), got {resolved=} {seq_len=} {prompt_len=}.")
    if prompt_len > 0 and prompt_len + resolved > seq_len:
        raise ValueError(f"prompt_len + response_len exceeds seq_len: {prompt_len=} {resolved=} {seq_len=}.")
    return resolved


def build_random_prompt_ids(
    seq_len: int,
    vocab_size: int,
    pad_token_id: int | None,
    seed: int,
) -> list[int]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    token_ids = torch.randint(0, vocab_size, (seq_len,), generator=gen, dtype=torch.long)
    if pad_token_id is not None:
        token_ids = token_ids.clone()
        token_ids[token_ids == pad_token_id] = (pad_token_id + 1) % vocab_size
    return token_ids.tolist()


def extract_teacher_topk_from_output(
    vllm_output: RequestOutput,
    response_len: int,
    topk: int,
) -> tuple[list[list[float]], list[list[int]]]:
    """Match ``PSRL_vLLMRollout`` teacher extraction (prompt_logprobs path)."""
    if not hasattr(vllm_output, "prompt_logprobs") or vllm_output.prompt_logprobs is None:
        raise RuntimeError("vLLM output missing prompt_logprobs; set SamplingParams.prompt_logprobs.")

    prompt_token_ids = list(vllm_output.prompt_token_ids)
    if response_len > 0:
        prompt_logprobs = vllm_output.prompt_logprobs[-response_len:]
        prompt_token_ids = prompt_token_ids[-response_len:]
    else:
        prompt_logprobs = []
        prompt_token_ids = []

    log_prob_list: list[list[float]] = []
    teacher_id_list: list[list[int]] = []
    for token_id, logprob_dict in zip(prompt_token_ids, prompt_logprobs, strict=False):
        if logprob_dict is None:
            log_prob_list.append([float("nan")] * topk if topk > 0 else [])
            teacher_id_list.append([])
            continue

        topk_entries = sorted(
            logprob_dict.items(),
            key=lambda item: getattr(item[1], "rank", None)
            if getattr(item[1], "rank", None) is not None
            else -float(item[1].logprob),
        )
        if topk > 0:
            topk_entries = topk_entries[:topk]
            log_prob_list.append([float(entry.logprob) for _, entry in topk_entries])
            teacher_id_list.append([int(tid) for tid, _ in topk_entries])
        elif token_id in logprob_dict:
            log_prob_list.append([float(logprob_dict[token_id].logprob)])
            teacher_id_list.append([int(token_id)])
        else:
            log_prob_list.append([float(logprob_dict[str(token_id)].logprob)])
            teacher_id_list.append([int(token_id)])

    return log_prob_list, teacher_id_list


def build_async_llm(args: argparse.Namespace) -> AsyncLLM:
    max_model_len = args.max_model_len if args.max_model_len is not None else args.seq_len + 1
    max_num_batched_tokens = args.max_num_batched_tokens if args.max_num_batched_tokens is not None else max_model_len

    llm_kwargs: dict = {
        "model": args.model_path,
        "tensor_parallel_size": args.tp_size,
        "pipeline_parallel_size": args.pp_size,
        "enable_expert_parallel": args.ep_size > 1,
        "dtype": args.dtype,
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "enable_prefix_caching": args.enable_prefix_caching,
        "trust_remote_code": args.trust_remote_code,
        "disable_log_stats": True,
        "disable_custom_all_reduce": True,
        "seed": args.seed,
    }

    if args.vllm_profiler:
        profiler_dir = args.vllm_profiler_dir
        # vLLM requires an absolute directory for torch_profiler_dir.
        if not os.path.isabs(profiler_dir):
            profiler_dir = os.path.abspath(os.path.join(args.log_dir, profiler_dir))
        else:
            profiler_dir = os.path.abspath(os.path.expanduser(profiler_dir))

        llm_kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": profiler_dir,
            "torch_profiler_with_memory": True,
            "ignore_frontend": bool(args.vllm_profiler_ignore_frontend),
            "wait_iterations": int(args.vllm_profiler_wait),
            "warmup_iterations": int(args.vllm_profiler_warmup),
            "active_iterations": int(args.vllm_profiler_active),
            "delay_iterations": 0,
            "max_iterations": 0,
        }

    async_engine_params = inspect.signature(AsyncEngineArgs.__init__).parameters
    supports_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in async_engine_params.values())
    if not supports_var_keyword:
        supported_keys = {key for key, param in async_engine_params.items() if key != "self"}
        llm_kwargs = {key: value for key, value in llm_kwargs.items() if key in supported_keys}

    engine_args = AsyncEngineArgs(**llm_kwargs)
    return AsyncLLM.from_engine_args(engine_args)


def log_phase(phase: str, log_dir: str, log_prefix: str) -> str:
    message = log_gpu_memory_now(prefix=phase)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{log_prefix}_memory.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    return message


async def run_teacher_prefill(
    llm: AsyncLLM,
    prompt_token_ids: list[int],
    topk: int,
    response_len: int,
) -> tuple[RequestOutput, list[list[float]], list[list[int]]]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        detokenize=False,
        prompt_logprobs=topk,
    )
    request_id = str(uuid.uuid4())
    final_output: RequestOutput | None = None
    async for output in llm.generate(
        prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        final_output = output

    if final_output is None:
        raise RuntimeError("vLLM generate returned no output.")

    top_log_probs, top_ids = extract_teacher_topk_from_output(final_output, response_len, topk)
    return final_output, top_log_probs, top_ids


async def async_main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU memory profiling.")

    response_len = resolve_response_len(args.seq_len, args.prompt_len, args.response_len)
    print(
        f"Profiling vLLM model={args.model_path!r} seq_len={args.seq_len} response_len={response_len} "
        f"topk={args.topk} tp={args.tp_size} ep={args.ep_size} pp={args.pp_size} dtype={args.dtype}"
    )

    hf_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    vocab_size = int(getattr(hf_config, "vocab_size", 32000))
    pad_token_id = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
        pad_token_id = tokenizer.pad_token_id
        if vocab_size <= 0:
            vocab_size = len(tokenizer)
    except Exception:
        pass

    prompt_token_ids = build_random_prompt_ids(args.seq_len, vocab_size, pad_token_id, args.seed)

    log_phase("baseline_before_engine", args.log_dir, args.log_prefix)

    t0 = time.monotonic()
    llm = build_async_llm(args)
    load_time = time.monotonic() - t0
    log_phase(f"after_engine_init ({load_time:.2f}s)", args.log_dir, args.log_prefix)

    for _ in range(args.num_warmup):
        await run_teacher_prefill(llm, prompt_token_ids, args.topk, response_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    log_phase(f"after_warmup (n={args.num_warmup})", args.log_dir, args.log_prefix)

    times: list[float] = []
    top_log_probs: list[list[float]] = []
    top_ids: list[list[int]] = []
    for i in range(args.num_iters):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_start = time.monotonic()
        profile_prefix = f"teacher_prefill_iter{i}"
        if args.vllm_profiler:
            await llm.start_profile(profile_prefix=profile_prefix)
        try:
            _, top_log_probs, top_ids = await run_teacher_prefill(llm, prompt_token_ids, args.topk, response_len)
        finally:
            if args.vllm_profiler:
                await llm.stop_profile()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.monotonic() - t_start
        times.append(elapsed)
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        log_phase(
            f"after_generate iter={i} ({elapsed * 1000:.2f}ms peak_alloc={peak_gb:.2f}GB)",
            args.log_dir,
            args.log_prefix,
        )

    device_infos = get_all_gpu_memory_info()
    print(format_memory_message(device_infos, prefix="[final]"))
    print(f"Generate latency (ms): {[t * 1000 for t in times]}")
    num_positions = len(top_log_probs)
    num_topk = len(top_log_probs[0]) if num_positions > 0 else 0
    print(f"teacher topk logprobs shape: ({num_positions}, {num_topk})  # (response_len, topk)")
    if num_positions > 0:
        print(f"sample top_ids[0][:5]: {top_ids[0][:5]}")
        print(f"sample top_log_probs[0][:5]: {top_log_probs[0][:5]}")
        print(f"sample top_ids[-1][:5]: {top_ids[-1][:5]}")
    print(f"Memory log: {os.path.join(args.log_dir, f'{args.log_prefix}_memory.log')}")

    if hasattr(llm, "shutdown"):
        llm.shutdown()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
