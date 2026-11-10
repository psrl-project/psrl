"""
Persistent subprocess bridge for the C++ request-level candidate evaluator.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import select
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    InstanceSnapshot,
    RequestMigration,
    RequestSnapshot,
    RoleCandidatePlan,
    RoleEvaluationResult,
    RoleSnapshot,
)


class CppCandidateEvaluatorError(RuntimeError):
    """
    Raised when the C++ evaluator cannot start or violates its protocol.
    """


@dataclass(frozen=True)
class CppBatchEvaluation:
    """
    One complete baseline plus candidate evaluation returned by C++.
    """

    baseline_results: Mapping[str, RoleEvaluationResult]
    candidate_results: tuple[Mapping[str, RoleEvaluationResult], ...]
    logical_task_count: int
    unique_task_count: int
    deduplicated_task_count: int
    bridge_wall_s: float
    evaluation_wall_s: float
    rebalance_simulation_s: float
    router_simulation_s: float
    simulation_wall_s: float = 0.0
    rebalance_router_overlap_s: float = 0.0


def default_cpp_candidate_evaluator_binary() -> Path:
    """
    Return the conventional in-repository release binary path.
    """
    return Path(__file__).resolve().parents[5] / "build" / "elastic_simulator" / "elastic_simulator"


def _json_priority(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if math.isnan(number):
            raise CppCandidateEvaluatorError("priority values cannot contain NaN")
        if math.isinf(number):
            return {"__psrl_float__": "inf" if number > 0.0 else "-inf"}
        return number
    if isinstance(value, (list, tuple)):
        return [_json_priority(item) for item in value]
    return str(value)


def _request_dict(request: RequestSnapshot) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "seq_len": request.seq_len,
        "source_instance_id": request.source_instance_id,
        "is_waiting": request.is_waiting,
        "movable": request.movable,
        "route_order": request.route_order,
        "routing_priority": [_json_priority(value) for value in request.routing_priority],
        "eligible_instance_ids": (
            None if request.eligible_instance_ids is None else list(request.eligible_instance_ids)
        ),
        "fallback_instance_ids": (
            None if request.fallback_instance_ids is None else list(request.fallback_instance_ids)
        ),
        "candidate_priorities": [
            [instance_id, _json_priority(priority)] for instance_id, priority in request.candidate_priorities
        ],
    }


def _instance_dict(instance: InstanceSnapshot) -> dict[str, Any]:
    return {
        "instance_id": instance.instance_id,
        "is_awake": instance.is_awake,
        "model_version": instance.model_version,
        "requests": [_request_dict(request) for request in instance.requests],
        "route_request_count": instance.route_request_count,
        "running_count": instance.running_count,
        "waiting_count": instance.waiting_count,
        "token_count": instance.token_count,
        "max_model_len": instance.max_model_len,
        "throughput_params": list(instance.throughput_params),
        "route_cost_params": (None if instance.route_cost_params is None else list(instance.route_cost_params)),
    }


def _role_dict(snapshot: RoleSnapshot) -> dict[str, Any]:
    return {
        "role": snapshot.role,
        "strategy": snapshot.strategy,
        "instances": [_instance_dict(instance) for instance in snapshot.instances],
        "pending_requests": [_request_dict(request) for request in snapshot.pending_requests],
        "queue_scope": snapshot.queue_scope,
        "max_concurrent_requests": snapshot.max_concurrent_requests,
        "delta_throughput_threshold": snapshot.delta_throughput_threshold,
        "itl_max_itl": snapshot.itl_max_itl,
    }


def _plan_dict(plan: RoleCandidatePlan) -> dict[str, Any]:
    return {
        "wake_instance_ids": sorted(plan.wake_instance_ids),
        "sleep_instance_ids": sorted(plan.sleep_instance_ids),
        "primary_scale_up": plan.primary_scale_up,
    }


def _role_result(raw: Mapping[str, Any]) -> RoleEvaluationResult:
    try:
        moves = tuple(
            RequestMigration(
                request_id=str(move["request_id"]),
                source_instance_id=int(move["source_instance_id"]),
                destination_instance_id=int(move["destination_instance_id"]),
            )
            for move in raw["rebalance_moves"]
        )
        instance_throughputs = tuple(
            (int(row["instance_id"]), float(row["throughput"])) for row in raw["instance_throughputs"]
        )
        return RoleEvaluationResult(
            throughput=float(raw["throughput"]),
            routed_count=int(raw["routed_count"]),
            unrouted_count=int(raw["unrouted_count"]),
            rebalance_moves=moves,
            instance_throughputs=instance_throughputs,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CppCandidateEvaluatorError(f"invalid C++ role evaluation result: {exc}") from exc


class CppCandidateEvaluator:
    """
    Keep one C++ evaluator process alive across policy decision cycles.
    """

    _EXPECTED_ROLES = ("Rollout", "RewardModel")

    def __init__(
        self,
        *,
        binary_path: str | os.PathLike[str] | None,
        max_workers: int,
        timeout_s: float,
    ) -> None:
        path = (
            default_cpp_candidate_evaluator_binary()
            if binary_path is None
            else Path(binary_path).expanduser().resolve()
        )
        if not path.is_file() or not os.access(path, os.X_OK):
            raise CppCandidateEvaluatorError(
                f"C++ candidate evaluator is missing or not executable: {path}. "
                "Build it with `cmake -S tools/elastic_simulator "
                "-B build/elastic_simulator -DCMAKE_BUILD_TYPE=Release && "
                "cmake --build build/elastic_simulator -j`."
            )
        if int(max_workers) <= 0:
            raise ValueError("max_workers must be positive")
        if float(timeout_s) <= 0.0:
            raise ValueError("timeout_s must be positive")

        self.binary_path = path
        self.max_workers = int(max_workers)
        self.timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._next_cycle_id = 0
        try:
            self._process = subprocess.Popen(
                [
                    str(path),
                    "--input",
                    "-",
                    "--output",
                    "-",
                    "--threads",
                    str(self.max_workers),
                    "--warmup",
                    "0",
                    "--repeat",
                    "1",
                    "--emit-migrations",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise CppCandidateEvaluatorError(f"failed to start C++ candidate evaluator {path}: {exc}") from exc

    def evaluate(
        self,
        *,
        snapshots: Mapping[str, RoleSnapshot],
        candidate_plans: Sequence[Mapping[str, RoleCandidatePlan]],
    ) -> CppBatchEvaluation:
        """
        Evaluate a full decision cycle in one JSONL request/response.
        """
        if set(snapshots) != set(self._EXPECTED_ROLES):
            raise CppCandidateEvaluatorError("C++ evaluation requires Rollout and RewardModel snapshots")
        for index, plans in enumerate(candidate_plans):
            if set(plans) != set(self._EXPECTED_ROLES):
                raise CppCandidateEvaluatorError(f"candidate {index} does not contain both role plans")

        with self._lock:
            process = self._process
            if process is None:
                raise CppCandidateEvaluatorError("C++ candidate evaluator is closed")
            if process.poll() is not None:
                stderr = self._read_stderr_locked()
                raise CppCandidateEvaluatorError(
                    f"C++ candidate evaluator exited with code {process.returncode}: {stderr}"
                )
            if process.stdin is None or process.stdout is None:
                raise CppCandidateEvaluatorError("C++ candidate evaluator pipes are unavailable")

            cycle_id = self._next_cycle_id
            self._next_cycle_id += 1
            payload = {
                "schema_version": 1,
                "record_type": "elastic_simulation_input",
                "cycle_id": cycle_id,
                "roles": {role: _role_dict(snapshots[role]) for role in self._EXPECTED_ROLES},
                "candidates": [
                    {
                        "index": index,
                        "role_plans": {role: _plan_dict(plans[role]) for role in self._EXPECTED_ROLES},
                    }
                    for index, plans in enumerate(candidate_plans)
                ],
            }
            try:
                encoded = json.dumps(
                    payload,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError) as exc:
                raise CppCandidateEvaluatorError(f"invalid input for C++ candidate evaluator: {exc}") from exc

            started_s = time.monotonic()
            try:
                process.stdin.write(encoded + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                stderr = self._stop_locked()
                raise CppCandidateEvaluatorError(
                    f"failed to send input to C++ candidate evaluator: {exc}; {stderr}"
                ) from exc

            ready, _, _ = select.select(
                [process.stdout.fileno()],
                [],
                [],
                self.timeout_s,
            )
            if not ready:
                stderr = self._stop_locked()
                raise CppCandidateEvaluatorError(
                    f"C++ candidate evaluation timed out after {self.timeout_s:.3f}s; {stderr}"
                )
            response_line = process.stdout.readline()
            if not response_line:
                stderr = self._stop_locked()
                raise CppCandidateEvaluatorError(f"C++ candidate evaluator closed stdout without a response; {stderr}")
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError as exc:
                stderr = self._stop_locked()
                raise CppCandidateEvaluatorError(
                    f"C++ candidate evaluator returned invalid JSON: {exc}; {stderr}"
                ) from exc
            bridge_wall_s = time.monotonic() - started_s

        return self._parse_response(
            response,
            cycle_id=cycle_id,
            candidate_count=len(candidate_plans),
            bridge_wall_s=bridge_wall_s,
        )

    def close(self) -> None:
        """
        Stop the owned evaluator process. Safe to call more than once.
        """
        with self._lock:
            self._stop_locked()

    def _parse_response(
        self,
        response: Any,
        *,
        cycle_id: int,
        candidate_count: int,
        bridge_wall_s: float,
    ) -> CppBatchEvaluation:
        try:
            if response["schema_version"] != 1:
                raise ValueError("unsupported response schema")
            if response["record_type"] != "elastic_simulation_benchmark":
                raise ValueError("unexpected response record type")
            if int(response["cycle_id"]) != cycle_id:
                raise ValueError(f"cycle id mismatch: expected {cycle_id}, got {response['cycle_id']}")

            functional = response["functional_results"]
            raw_baseline = functional["baseline"]
            if set(raw_baseline) != set(self._EXPECTED_ROLES):
                raise ValueError("baseline does not contain both roles")
            baseline_results = {role: _role_result(raw_baseline[role]) for role in self._EXPECTED_ROLES}

            raw_candidates = functional["candidates"]
            if len(raw_candidates) != candidate_count:
                raise ValueError(f"expected {candidate_count} candidates, got {len(raw_candidates)}")
            candidate_results = []
            for expected_index, row in enumerate(raw_candidates):
                if int(row["index"]) != expected_index:
                    raise ValueError(f"expected candidate index {expected_index}, got {row['index']}")
                roles = row["roles"]
                if set(roles) != set(self._EXPECTED_ROLES):
                    raise ValueError(f"candidate {expected_index} does not contain both roles")
                candidate_results.append({role: _role_result(roles[role]["result"]) for role in self._EXPECTED_ROLES})

            counts = response["counts"]
            timing = response["timing_ns"]
            nanoseconds_to_seconds = 1e-9
            return CppBatchEvaluation(
                baseline_results=baseline_results,
                candidate_results=tuple(candidate_results),
                logical_task_count=int(counts["logical_tasks"]),
                unique_task_count=int(counts["unique_tasks"]),
                deduplicated_task_count=int(counts["deduplicated_tasks"]),
                bridge_wall_s=bridge_wall_s,
                evaluation_wall_s=(float(timing["candidate_evaluation_wall"]["mean"]) * nanoseconds_to_seconds),
                rebalance_simulation_s=(float(timing["rebalance_wall_union"]["mean"]) * nanoseconds_to_seconds),
                router_simulation_s=(float(timing["router_wall_union"]["mean"]) * nanoseconds_to_seconds),
                simulation_wall_s=(float(timing["rebalance_router_wall_union"]["mean"]) * nanoseconds_to_seconds),
                rebalance_router_overlap_s=(
                    float(timing["rebalance_router_wall_overlap"]["mean"]) * nanoseconds_to_seconds
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CppCandidateEvaluatorError(f"invalid C++ candidate evaluator response: {exc}") from exc

    def _read_stderr_locked(self) -> str:
        process = self._process
        if process is None or process.stderr is None:
            return ""
        try:
            return process.stderr.read().strip()
        except OSError:
            return ""

    def _stop_locked(self) -> str:
        process = self._process
        if process is None:
            return ""
        self._process = None
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        stderr = ""
        if process.stderr is not None:
            try:
                stderr = process.stderr.read().strip()
            except OSError:
                pass
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except OSError:
                    pass
        return stderr

    def __enter__(self) -> CppCandidateEvaluator:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
