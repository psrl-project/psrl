"""Small repeatable latency/memory benchmark for the persistent Docker backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid

from psrl.sandbox import SandboxManager, SandboxSource, SandboxSpec
from psrl.sandbox.backends.docker import DockerBackend


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


async def _benchmark(image: str, iterations: int, concurrency: int) -> dict:
    backend = DockerBackend(auto_pull=True)
    manager = SandboxManager({"docker": backend}, "docker")
    gate = asyncio.Semaphore(concurrency)

    async def run_one(index: int) -> tuple[float, float, int]:
        async with gate:
            started = time.perf_counter()
            lease = await manager.acquire(
                SandboxSpec(
                    SandboxSource.image(image),
                    idempotency_key=f"benchmark-{uuid.uuid4().hex}-{index}",
                    idle_timeout_s=300,
                )
            )
            created = time.perf_counter()
            result = await lease.session.exec("printf ok")
            executed = time.perf_counter()
            if result.stdout != "ok":
                raise RuntimeError(f"Unexpected Docker output: {result!r}.")
            usage = await lease.session.stats()
            await lease.release()
            return created - started, executed - created, usage.peak_memory_bytes

    try:
        samples = await asyncio.gather(*(run_one(index) for index in range(iterations)))
        metrics = manager.metrics_snapshot()["docker"].as_dict()
    finally:
        await manager.shutdown()
    creates = [sample[0] for sample in samples]
    execs = [sample[1] for sample in samples]
    return {
        "iterations": iterations,
        "concurrency": concurrency,
        "create_s": {
            "mean": statistics.fmean(creates),
            "p50": _percentile(creates, 0.50),
            "p95": _percentile(creates, 0.95),
            "max": max(creates),
        },
        "exec_s": {
            "mean": statistics.fmean(execs),
            "p50": _percentile(execs, 0.50),
            "p95": _percentile(execs, 0.95),
            "max": max(execs),
        },
        "max_container_peak_memory_mib": max(sample[2] for sample in samples) / (1024 * 1024),
        "backend_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.iterations < 1 or args.concurrency < 1:
        parser.error("--iterations and --concurrency must be positive")
    print(json.dumps(asyncio.run(_benchmark(args.image, args.iterations, args.concurrency)), indent=2))


if __name__ == "__main__":
    main()
