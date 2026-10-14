#!/usr/bin/env python3
"""Benchmark the Python elastic simulator and compare it with C++ output."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    RoleCandidatePlan,
    RoleEvaluationContext,
    RoleEvaluationResult,
    RoleSnapshot,
    evaluate_role_candidate,
    prepare_role_evaluation_context,
)

ROLE_ORDER = ("Rollout", "RewardModel")
FNV_OFFSET_BASIS = 14695981039346656037
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class TaskSpec:
    role: str
    plan: RoleCandidatePlan
    context: RoleEvaluationContext


@dataclass(frozen=True)
class TaskBatch:
    tasks: tuple[TaskSpec, ...]
    logical_task_count: int
    candidate_count: int


def plan_from_mapping(raw: Mapping[str, Any]) -> RoleCandidatePlan:
    return RoleCandidatePlan(
        wake_instance_ids=frozenset(int(value) for value in raw.get("wake_instance_ids", ())),
        sleep_instance_ids=frozenset(
            int(value) for value in raw.get("sleep_instance_ids", ())
        ),
        primary_scale_up=bool(raw.get("primary_scale_up", False)),
    )


def build_task_batch(
    payload: Mapping[str, Any],
    contexts: Mapping[str, RoleEvaluationContext],
) -> TaskBatch:
    tasks: list[TaskSpec] = []
    task_ids: dict[tuple[str, RoleCandidatePlan], int] = {}

    def add_once(role: str, plan: RoleCandidatePlan) -> None:
        key = (role, plan)
        if key not in task_ids:
            task_ids[key] = len(tasks)
            tasks.append(TaskSpec(role=role, plan=plan, context=contexts[role]))

    roles = tuple(role for role in ROLE_ORDER if role in contexts)
    baseline = RoleCandidatePlan()
    for role in roles:
        add_once(role, baseline)
    candidates = payload.get("candidates", ())
    for candidate in candidates:
        plans = candidate.get("role_plans", {})
        for role in roles:
            add_once(role, plan_from_mapping(plans.get(role, {})))
    return TaskBatch(
        tasks=tuple(tasks),
        logical_task_count=len(roles) * (len(candidates) + 1),
        candidate_count=len(candidates),
    )


def execute_batch(
    executor: ThreadPoolExecutor,
    batch: TaskBatch,
) -> tuple[RoleEvaluationResult, ...]:
    futures = [
        executor.submit(evaluate_role_candidate, task.context, task.plan)
        for task in batch.tasks
    ]
    return tuple(future.result() for future in futures)


def fnv_append(value: int, data: bytes) -> int:
    for byte in data:
        value ^= byte
        value = (value * FNV_PRIME) & UINT64_MASK
    return value


def functional_checksum(results: Iterable[RoleEvaluationResult]) -> str:
    """Match tools/elastic_simulator's native little-endian FNV checksum."""
    checksum = FNV_OFFSET_BASIS
    for result in results:
        checksum = fnv_append(checksum, struct.pack("<d", result.throughput))
        checksum = fnv_append(checksum, struct.pack("<q", result.routed_count))
        checksum = fnv_append(checksum, struct.pack("<q", result.unrouted_count))
        for move in result.rebalance_moves:
            checksum = fnv_append(checksum, move.request_id.encode("utf-8"))
            checksum = fnv_append(checksum, struct.pack("<i", move.source_instance_id))
            checksum = fnv_append(
                checksum,
                struct.pack("<i", move.destination_instance_id),
            )
        for instance_id, throughput in result.instance_throughputs:
            checksum = fnv_append(checksum, struct.pack("<i", instance_id))
            checksum = fnv_append(checksum, struct.pack("<d", throughput))
    return f"{checksum:016x}"


def percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank - 1, len(ordered) - 1)]


def summarize(values: Sequence[int]) -> dict[str, int | float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
    }


def interval_union_ns(intervals: Iterable[tuple[float, float]]) -> int:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0
    current_start, current_end = valid[0]
    total_s = 0.0
    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total_s += current_end - current_start
        current_start, current_end = start, end
    total_s += current_end - current_start
    return int(total_s * 1e9)


def load_cpp_rows(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            cycle_id = int(row["cycle_id"])
            if cycle_id in rows:
                raise ValueError(f"duplicate C++ cycle {cycle_id} at line {line_number}")
            rows[cycle_id] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--cpp-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--warmup-first-cycle", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if args.threads <= 0 or args.repeat <= 0 or args.warmup_first_cycle < 0:
        parser.error("threads and repeat must be positive; warmup must be non-negative")

    cpp_rows = load_cpp_rows(args.cpp_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    emitted = 0
    first_cycle = True
    seen_cycles: set[int] = set()
    with (
        args.input.open("r", encoding="utf-8") as input_file,
        args.output.open("w", encoding="utf-8") as output_file,
        ThreadPoolExecutor(max_workers=args.threads) as executor,
    ):
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            cycle_id = int(payload["cycle_id"])
            cpp = cpp_rows.get(cycle_id)
            if cpp is None:
                raise ValueError(f"input cycle {cycle_id} has no C++ result")

            preparation_started = time.perf_counter_ns()
            contexts = {
                role: prepare_role_evaluation_context(
                    RoleSnapshot.from_mapping(payload["roles"][role])
                )
                for role in ROLE_ORDER
                if role in payload["roles"]
            }
            preparation_ns = time.perf_counter_ns() - preparation_started
            batch = build_task_batch(payload, contexts)
            expected_counts = {
                "candidates": batch.candidate_count,
                "logical_tasks": batch.logical_task_count,
                "unique_tasks": len(batch.tasks),
                "deduplicated_tasks": batch.logical_task_count - len(batch.tasks),
            }
            if expected_counts != cpp["counts"]:
                raise AssertionError(
                    f"cycle {cycle_id} task counts differ: "
                    f"python={expected_counts}, cpp={cpp['counts']}"
                )

            if first_cycle:
                for _ in range(args.warmup_first_cycle):
                    execute_batch(executor, batch)
                first_cycle = False

            evaluation_values: list[int] = []
            rebalance_worker_values: list[int] = []
            router_worker_values: list[int] = []
            other_worker_values: list[int] = []
            rebalance_wall_values: list[int] = []
            router_wall_values: list[int] = []
            simulation_wall_values: list[int] = []
            overlap_wall_values: list[int] = []
            result_checksum: str | None = None
            for _ in range(args.repeat):
                evaluation_started = time.perf_counter_ns()
                results = execute_batch(executor, batch)
                evaluation_ns = time.perf_counter_ns() - evaluation_started
                checksum = functional_checksum(results)
                if result_checksum is not None and checksum != result_checksum:
                    raise AssertionError(f"cycle {cycle_id} produced nondeterministic results")
                result_checksum = checksum

                rebalance_ns = sum(
                    int(result.rebalance_simulation_s * 1e9) for result in results
                )
                router_ns = sum(int(result.router_simulation_s * 1e9) for result in results)
                other_ns = sum(int(result.other_simulation_s * 1e9) for result in results)
                rebalance_intervals = [
                    (result.rebalance_started_s, result.rebalance_finished_s)
                    for result in results
                    if result.rebalance_started_s is not None
                    and result.rebalance_finished_s is not None
                ]
                router_intervals = [
                    (result.router_started_s, result.router_finished_s)
                    for result in results
                    if result.router_started_s is not None
                    and result.router_finished_s is not None
                ]
                rebalance_wall_ns = interval_union_ns(rebalance_intervals)
                router_wall_ns = interval_union_ns(router_intervals)
                simulation_wall_ns = interval_union_ns(
                    rebalance_intervals + router_intervals
                )
                evaluation_values.append(evaluation_ns)
                rebalance_worker_values.append(rebalance_ns)
                router_worker_values.append(router_ns)
                other_worker_values.append(other_ns)
                rebalance_wall_values.append(rebalance_wall_ns)
                router_wall_values.append(router_wall_ns)
                simulation_wall_values.append(simulation_wall_ns)
                overlap_wall_values.append(
                    max(0, rebalance_wall_ns + router_wall_ns - simulation_wall_ns)
                )

            cpp_checksum = str(cpp["functional_checksum"])
            functional_match = result_checksum == cpp_checksum
            if not functional_match:
                raise AssertionError(
                    f"cycle {cycle_id} checksum differs: "
                    f"python={result_checksum}, cpp={cpp_checksum}"
                )
            output = {
                "schema_version": 1,
                "record_type": "python_elastic_simulation_benchmark",
                "cycle_id": cycle_id,
                "config": {
                    "threads": args.threads,
                    "warmup_first_cycle": args.warmup_first_cycle,
                    "repeat": args.repeat,
                },
                "counts": expected_counts,
                "timing_ns": {
                    "context_preparation": preparation_ns,
                    "candidate_evaluation_wall": summarize(evaluation_values),
                    "rebalance_worker_sum": summarize(rebalance_worker_values),
                    "router_worker_sum": summarize(router_worker_values),
                    "other_worker_sum": summarize(other_worker_values),
                    "rebalance_wall_union": summarize(rebalance_wall_values),
                    "router_wall_union": summarize(router_wall_values),
                    "rebalance_router_wall_union": summarize(simulation_wall_values),
                    "rebalance_router_wall_overlap": summarize(overlap_wall_values),
                },
                "python_functional_checksum": result_checksum,
                "cpp_functional_checksum": cpp_checksum,
                "functional_match": functional_match,
                "cpp_timing_ns": cpp["timing_ns"],
                "production_timing_seconds": cpp.get("reference_timing_seconds"),
            }
            output_file.write(json.dumps(output, separators=(",", ":"), allow_nan=False))
            output_file.write("\n")
            output_file.flush()
            emitted += 1
            seen_cycles.add(cycle_id)
            if args.progress_every > 0 and emitted % args.progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"processed {emitted} cycles in {elapsed:.1f}s "
                    f"(input line {line_number})",
                    file=sys.stderr,
                    flush=True,
                )

    missing_cycles = sorted(set(cpp_rows) - seen_cycles)
    if missing_cycles:
        raise ValueError(f"C++ output has unmatched cycles: {missing_cycles[:10]}")
    print(
        f"benchmarked and matched {emitted} cycles in {time.perf_counter() - started:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
