"""
Benchmark script: launch two vLLM engines (different models) in sequence and
record timing and GPU/RAM usage. Both engines run on the same machine and
share 8 GPUs: each engine uses all 8 cards (TP * PP when running); they
execute sequentially.

1) Phase 1: launch each engine, then sleep immediately; record time from
   launch to sleep and GPU/RAM after sleep.
2) Phase 2: launch each engine, run one inference, then sleep; record launch
   time, sleep time, and GPU/RAM before/after launch and before/after sleep.
"""

import json
import multiprocessing
import os
import time
from pathlib import Path

# Comment line length limit: 72 characters (wrap longer lines).
# ---------- Hardcoded config: single machine, both engines share 8 GPUs -----
# Both engines use the same 8 GPUs on this host; each run uses all 8 when
# executing.
SINGLE_MACHINE_GPU_IDS = "0,1,2,3,4,5,6,7"
DEVICES_A = "0,1,2,3,4,5,6,7"
DEVICES_B = "0,1,2,3,4,5,6,7"

MODEL_A = "Qwen/Qwen2.5-0.5B"
MODEL_B = "Qwen/Qwen2.5-1.5B"
SLEEP_SEC = 10.0
TP = 2   # tensor parallel size
PP = 4   # pipeline parallel size; total GPUs per engine = TP * PP
OUT_DIR = "./vllm_bench_out"

# Worker run in subprocess: load vLLM on given GPUs and run sleep-only or
# inference-then-sleep per phase.


def get_gpu_memory_mb():
    """Return per-device GPU memory allocated/reserved in MB (uses torch)."""
    import torch
    if not torch.cuda.is_available():
        return {}
    out = {}
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
        out[i] = {"allocated_mb": round(allocated, 2), "reserved_mb": round(reserved, 2)}
    return out


def get_system_memory_mb():
    try:
        import psutil
        v = psutil.virtual_memory()
        return {
            "used_mb": round(v.used / (1024 ** 2), 2),
            "available_mb": round(v.available / (1024 ** 2), 2),
            "total_mb": round(v.total / (1024 ** 2), 2),
        }
    except Exception:
        return {}


def worker_main(
    model: str,
    cuda_devices: str,
    phase: str,
    sleep_sec: float,
    tp: int,
    pp: int,
    out_path: str,
) -> None:
    """Run in child process: set CUDA devices, load vLLM, then sleep or infer.
    Writes result dict as JSON to out_path.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    result = {
        "phase": phase,
        "model": model,
        "cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tp": tp,
        "pp": pp,
    }

    # GPU/RAM before launch (no model loaded yet)
    result["mem_before_launch"] = {
        "gpu": get_gpu_memory_mb(),
        "system": get_system_memory_mb(),
    }

    t_launch_start = time.perf_counter()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        gpu_memory_utilization=0.9,
    )
    t_launch_end = time.perf_counter()
    result["launch_duration_sec"] = round(t_launch_end - t_launch_start, 3)
    result["mem_after_launch"] = {
        "gpu": get_gpu_memory_mb(),
        "system": get_system_memory_mb(),
    }

    if phase == "sleep_only":
        t_sleep_start = time.perf_counter()
        time.sleep(sleep_sec)
        t_sleep_end = time.perf_counter()
        result["sleep_duration_sec"] = round(t_sleep_end - t_sleep_start, 3)
        result["mem_after_sleep"] = {
            "gpu": get_gpu_memory_mb(),
            "system": get_system_memory_mb(),
        }
    else:
        # inference_then_sleep: run one inference
        t_infer_start = time.perf_counter()
        sampling_params = SamplingParams(max_tokens=64, temperature=0)
        _ = llm.generate(["Hello, one inference."], sampling_params)
        t_infer_end = time.perf_counter()
        result["inference_duration_sec"] = round(t_infer_end - t_infer_start, 3)
        result["mem_before_sleep"] = {
            "gpu": get_gpu_memory_mb(),
            "system": get_system_memory_mb(),
        }
        t_sleep_start = time.perf_counter()
        time.sleep(sleep_sec)
        t_sleep_end = time.perf_counter()
        result["sleep_duration_sec"] = round(t_sleep_end - t_sleep_start, 3)
        result["mem_after_sleep"] = {
            "gpu": get_gpu_memory_mb(),
            "system": get_system_memory_mb(),
        }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)


def run_worker(
    model: str,
    cuda_devices: str,
    phase: str,
    sleep_sec: float,
    tp: int,
    pp: int,
    out_path: str,
) -> dict:
    """Run worker in subprocess; return main-process timing and
    subprocess-written result.
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Main process: record start/end (subprocess total wall time)
    t_main_start = time.perf_counter()
    proc = multiprocessing.Process(
        target=worker_main,
        args=(model, cuda_devices, phase, sleep_sec, tp, pp, str(out_file)),
    )
    proc.start()
    proc.join(timeout=600)
    t_main_end = time.perf_counter()

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10)
        raise RuntimeError("Worker timed out after 600s")
    if proc.exitcode != 0:
        raise RuntimeError(f"Worker exited with code {proc.exitcode}")

    with open(out_file) as f:
        sub_result = json.load(f)
    sub_result["main_process_wall_sec"] = round(t_main_end - t_main_start, 3)
    return sub_result


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "phase1_sleep_only": {},
        "phase2_inference_then_sleep": {},
        "main_memory_before_phase1": get_system_memory_mb(),
    }

    # ---------- Phase 1: launch each engine, then sleep ----------
    print("===== Phase 1: Engine A launch -> sleep =====")
    r1a = run_worker(
        MODEL_A, DEVICES_A, "sleep_only", SLEEP_SEC, TP, PP,
        str(out_dir / "phase1_engine_a.json"),
    )
    report["phase1_sleep_only"]["engine_a"] = r1a
    print(f"  launch-to-sleep (subprocess): {r1a['launch_duration_sec']} s, sleep: {r1a['sleep_duration_sec']} s")
    print(f"  GPU after sleep: {r1a['mem_after_sleep']['gpu']}")
    print(f"  RAM after sleep: {r1a['mem_after_sleep']['system']}")

    print("===== Phase 1: Engine B launch -> sleep =====")
    r1b = run_worker(
        MODEL_B, DEVICES_B, "sleep_only", SLEEP_SEC, TP, PP,
        str(out_dir / "phase1_engine_b.json"),
    )
    report["phase1_sleep_only"]["engine_b"] = r1b
    print(f"  launch-to-sleep (subprocess): {r1b['launch_duration_sec']} s, sleep: {r1b['sleep_duration_sec']} s")
    print(f"  GPU after sleep: {r1b['mem_after_sleep']['gpu']}")
    print(f"  RAM after sleep: {r1b['mem_after_sleep']['system']}")

    # ---------- Phase 2: launch each engine, one inference, then sleep -------
    print("===== Phase 2: Engine A launch -> one inference -> sleep =====")
    r2a = run_worker(
        MODEL_A, DEVICES_A, "inference_then_sleep", SLEEP_SEC, TP, PP,
        str(out_dir / "phase2_engine_a.json"),
    )
    report["phase2_inference_then_sleep"]["engine_a"] = r2a
    print(f"  launch: {r2a['launch_duration_sec']} s, inference: {r2a.get('inference_duration_sec')} s, sleep: {r2a['sleep_duration_sec']} s")
    print(f"  before launch GPU/RAM: {r2a['mem_before_launch']}")
    print(f"  after launch GPU/RAM: {r2a['mem_after_launch']}")
    print(f"  before sleep GPU/RAM: {r2a['mem_before_sleep']}")
    print(f"  after sleep GPU/RAM: {r2a['mem_after_sleep']}")

    print("===== Phase 2: Engine B launch -> one inference -> sleep =====")
    r2b = run_worker(
        MODEL_B, DEVICES_B, "inference_then_sleep", SLEEP_SEC, TP, PP,
        str(out_dir / "phase2_engine_b.json"),
    )
    report["phase2_inference_then_sleep"]["engine_b"] = r2b
    print(f"  launch: {r2b['launch_duration_sec']} s, inference: {r2b.get('inference_duration_sec')} s, sleep: {r2b['sleep_duration_sec']} s")
    print(f"  before launch GPU/RAM: {r2b['mem_before_launch']}")
    print(f"  after launch GPU/RAM: {r2b['mem_after_launch']}")
    print(f"  before sleep GPU/RAM: {r2b['mem_before_sleep']}")
    print(f"  after sleep GPU/RAM: {r2b['mem_after_sleep']}")

    report["main_memory_after_phase2"] = get_system_memory_mb()
    report_path = out_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
