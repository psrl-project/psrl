"""Single-GPU vLLM accuracy probe for the direct weight arena loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


def _snapshot_arena(worker) -> int:
    from psrl.utils.weight_arena import snapshot_weight_arena_to_cpu

    handle = worker.model_runner._psrl_weight_arena_handle
    worker._weight_arena_smoke_cpu_cache = snapshot_weight_arena_to_cpu(
        handle,
        pin_memory=False,
    )
    return sum(tensor.numel() * tensor.element_size() for tensor in worker._weight_arena_smoke_cpu_cache)


def _restore_arena(worker) -> int:
    from psrl.utils.weight_arena import restore_weight_arena_from_cpu

    return restore_weight_arena_from_cpu(
        worker.model_runner._psrl_weight_arena_handle,
        worker._weight_arena_smoke_cpu_cache,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("baseline", "direct"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cudagraph", action="store_true")
    parser.add_argument("--tms-cycles", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    direct = args.mode == "direct"
    arena_config = {
        "rollout_enabled": direct,
        "rollout_materialization": "direct",
        "max_chunk_gb": 0.0625,
        "alignment_bytes": 256,
    }
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        load_format="psrl_arena_dummy" if direct else "dummy",
        enforce_eager=not args.cudagraph,
        enable_sleep_mode=args.tms_cycles > 0,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        gpu_memory_utilization=0.2,
        distributed_executor_backend="uni",
        disable_custom_all_reduce=True,
        disable_log_stats=True,
        worker_extension_cls="psrl.workers.gen.vllm_extension.vLLMWorkerExtension",
        additional_config={
            "psrl_role": "rollout",
            "psrl_nixl_weight_arena": arena_config,
        },
    )

    def generate_once() -> dict:
        outputs = llm.generate(
            ["Explain why 2 + 2 = 4 in one sentence."],
            SamplingParams(temperature=0.0, max_tokens=8, logprobs=5),
            use_tqdm=False,
        )
        completion = outputs[0].outputs[0]
        selected_logprobs = []
        for token_id, candidates in zip(completion.token_ids, completion.logprobs, strict=True):
            selected_logprobs.append(float(candidates[token_id].logprob))
        return {
            "token_ids": list(completion.token_ids),
            "selected_logprobs": selected_logprobs,
            "text": completion.text,
        }

    generations = [generate_once()]
    arena_infos = [llm.collective_rpc("get_weight_arena_info")]
    snapshot_bytes = llm.collective_rpc(_snapshot_arena) if args.tms_cycles else []
    restored_bytes = []
    wake_tags = ["weights", "kv_cache", "graph"] if args.cudagraph else ["weights", "kv_cache"]
    for _ in range(args.tms_cycles):
        llm.sleep(level=2)
        llm.wake_up(tags=wake_tags)
        restored_bytes.append(llm.collective_rpc(_restore_arena))
        generations.append(generate_once())
        arena_infos.append(llm.collective_rpc("get_weight_arena_info"))

    first_generation = generations[0]
    result = {
        "mode": args.mode,
        "cudagraph": args.cudagraph,
        "tms_cycles": args.tms_cycles,
        **first_generation,
        "generations": generations,
        "arena_info": arena_infos[0],
        "arena_infos": arena_infos,
        "snapshot_bytes": snapshot_bytes,
        "restored_bytes": restored_bytes,
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
