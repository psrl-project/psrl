from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from psrl.workers.elastic.scaling_policy.evaluation.cpp_candidate_evaluator import (
    CppCandidateEvaluator,
    CppCandidateEvaluatorError,
    default_cpp_candidate_evaluator_binary,
)
from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    RoleCandidatePlan,
    RoleSnapshot,
    evaluate_role_candidate,
    prepare_role_evaluation_context,
)

pytestmark = pytest.mark.cpu_test

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL_DIR = _REPO_ROOT / "tools" / "elastic_simulator"


@pytest.fixture(scope="module")
def elastic_simulator_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which("cmake") is None or shutil.which("c++") is None:
        pytest.skip("C++ toolchain is unavailable")
    if not any(
        (include_root / "rapidjson" / "document.h").is_file()
        for include_root in (Path("/usr/include"), Path("/usr/local/include"))
    ):
        pytest.skip("RapidJSON headers are unavailable")

    build_dir = tmp_path_factory.mktemp("elastic-simulator-build")
    subprocess.run(
        [
            "cmake",
            "-S",
            str(_TOOL_DIR),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    return build_dir / "elastic_simulator"


def _instance(instance_id: int, requests: list[tuple[str, int]], *, awake: bool = True) -> dict[str, object]:
    return {
        "instance_id": instance_id,
        "is_awake": awake,
        "requests": [
            {
                "request_id": request_id,
                "seq_len": seq_len,
                "source_instance_id": instance_id,
                "is_waiting": False,
                "route_order": index,
            }
            for index, (request_id, seq_len) in enumerate(requests)
        ],
        "route_request_count": len(requests),
        "running_count": len(requests),
        "waiting_count": 0,
        "token_count": sum(seq_len for _, seq_len in requests),
        "max_model_len": 10_000,
        "throughput_params": [0.1, 1.0, 0.0, 0.0],
        "route_cost_params": [0.0, 0.0, 0.0, 1.0, 0.01],
    }


def _fixture() -> dict[str, object]:
    rollout = {
        "role": "Rollout",
        "strategy": "throughput_optimal",
        "queue_scope": "running",
        "instances": [
            _instance(0, [("10", 10), ("2", 2), ("1", 1)]),
            _instance(1, [], awake=False),
        ],
        "pending_requests": [{"request_id": "pending", "seq_len": 3, "route_order": 0}],
        "max_concurrent_requests": 32,
        "delta_throughput_threshold": 0.0,
        "itl_max_itl": None,
    }
    reward_model = {
        "role": "RewardModel",
        "strategy": "itl",
        "queue_scope": "running",
        "instances": [_instance(0, []), _instance(1, [])],
        "pending_requests": [
            {"request_id": f"rm-{index}", "seq_len": index + 1, "route_order": index} for index in range(5)
        ],
        "max_concurrent_requests": 8,
        "delta_throughput_threshold": 0.0,
        "itl_max_itl": None,
    }
    return {
        "schema_version": 1,
        "record_type": "elastic_simulation_input",
        "cycle_id": 7,
        "roles": {"Rollout": rollout, "RewardModel": reward_model},
        "candidates": [
            {
                "index": 0,
                "role_plans": {
                    "Rollout": {
                        "wake_instance_ids": [1],
                        "sleep_instance_ids": [],
                        "primary_scale_up": True,
                    },
                    "RewardModel": {
                        "wake_instance_ids": [],
                        "sleep_instance_ids": [],
                        "primary_scale_up": False,
                    },
                },
            }
        ],
        "reference_timing": {"router_simulation_s": 0.25},
    }


def _run(binary: Path, tmp_path: Path, payload: dict[str, object], threads: int) -> dict[str, object]:
    input_path = tmp_path / f"input-{threads}.jsonl"
    output_path = tmp_path / f"output-{threads}.jsonl"
    input_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    subprocess.run(
        [
            str(binary),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--threads",
            str(threads),
            "--warmup",
            "0",
            "--repeat",
            "2",
            "--emit-migrations",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _assert_result_matches(actual: dict[str, object], expected: object) -> None:
    assert actual["throughput"] == pytest.approx(expected.throughput, rel=1e-12)
    assert actual["routed_count"] == expected.routed_count
    assert actual["unrouted_count"] == expected.unrouted_count
    assert actual["rebalance_moves"] == [move.as_dict() for move in expected.rebalance_moves]
    assert [row["instance_id"] for row in actual["instance_throughputs"]] == [
        instance_id for instance_id, _ in expected.instance_throughputs
    ]
    assert [row["throughput"] for row in actual["instance_throughputs"]] == pytest.approx(
        [throughput for _, throughput in expected.instance_throughputs], rel=1e-12
    )


def test_cpp_matches_python_and_is_thread_deterministic(
    elastic_simulator_binary: Path,
    tmp_path: Path,
) -> None:
    payload = _fixture()
    serial = _run(elastic_simulator_binary, tmp_path, payload, threads=1)
    parallel = _run(elastic_simulator_binary, tmp_path, payload, threads=4)

    assert serial["functional_checksum"] == parallel["functional_checksum"]
    assert serial["counts"] == {
        "candidates": 1,
        "logical_tasks": 4,
        "unique_tasks": 3,
        "deduplicated_tasks": 1,
    }
    assert serial["reference_timing_seconds"]["router_simulation_s"] == 0.25
    timing = parallel["timing_ns"]
    evaluation_wall = timing["candidate_evaluation_wall"]["mean"]
    rebalance_wall = timing["rebalance_wall_union"]["mean"]
    router_wall = timing["router_wall_union"]["mean"]
    simulation_wall = timing["rebalance_router_wall_union"]["mean"]
    overlap = timing["rebalance_router_wall_overlap"]["mean"]
    assert 0 <= simulation_wall <= evaluation_wall
    assert rebalance_wall + router_wall - overlap == pytest.approx(simulation_wall, abs=1.0)
    assert "rebalance_parallel_wall_attribution" not in timing
    assert "router_parallel_wall_attribution" not in timing

    contexts = {
        role: prepare_role_evaluation_context(RoleSnapshot.from_mapping(snapshot))
        for role, snapshot in payload["roles"].items()
    }
    for role, actual in serial["functional_results"]["baseline"].items():
        expected = evaluate_role_candidate(contexts[role], RoleCandidatePlan())
        _assert_result_matches(actual, expected)

    candidate = serial["functional_results"]["candidates"][0]
    for role, row in candidate["roles"].items():
        raw_plan = payload["candidates"][0]["role_plans"][role]
        plan = RoleCandidatePlan(
            wake_instance_ids=frozenset(raw_plan["wake_instance_ids"]),
            sleep_instance_ids=frozenset(raw_plan["sleep_instance_ids"]),
            primary_scale_up=raw_plan["primary_scale_up"],
        )
        expected = evaluate_role_candidate(contexts[role], plan)
        _assert_result_matches(row["result"], expected)


def test_nonresumable_requests_are_neither_migrated_nor_requeued(elastic_simulator_binary, tmp_path):
    payload = _fixture()
    payload["roles"]["RewardModel"]["instances"] = [
        _instance(0, [("nonresumable-0", 12), ("nonresumable-1", 9)]),
        _instance(1, [], awake=False),
    ]
    for instance in payload["roles"]["RewardModel"]["instances"]:
        for request in instance["requests"]:
            request["movable"] = False
    payload["candidates"][0]["role_plans"]["RewardModel"] = {
        "wake_instance_ids": [1],
        "sleep_instance_ids": [0],
        "primary_scale_up": True,
    }
    result = _run(elastic_simulator_binary, tmp_path, payload, threads=2)
    context = prepare_role_evaluation_context(RoleSnapshot.from_mapping(payload["roles"]["RewardModel"]))
    plan = RoleCandidatePlan(
        wake_instance_ids=frozenset({1}), sleep_instance_ids=frozenset({0}), primary_scale_up=True
    )
    expected = evaluate_role_candidate(context, plan)
    actual = result["functional_results"]["candidates"][0]["roles"]["RewardModel"]["result"]
    _assert_result_matches(actual, expected)
    assert not expected.rebalance_moves
    assert all(not instance.reroute_requests for instance in context.instance_by_id.values())


def test_new_reward_instance_admits_128_requests_in_python_and_cpp(elastic_simulator_binary, tmp_path):
    payload = _fixture()
    payload["roles"]["RewardModel"]["instances"] = [_instance(0, [], awake=False)]
    payload["roles"]["RewardModel"]["max_concurrent_requests"] = 128
    payload["roles"]["RewardModel"]["pending_requests"] = [
        {"request_id": f"rm-{index}", "seq_len": 32, "route_order": index} for index in range(129)
    ]
    payload["candidates"][0]["role_plans"]["RewardModel"] = {
        "wake_instance_ids": [0],
        "sleep_instance_ids": [],
        "primary_scale_up": True,
    }
    plan = RoleCandidatePlan(wake_instance_ids=frozenset({0}), primary_scale_up=True)
    expected = evaluate_role_candidate(RoleSnapshot.from_mapping(payload["roles"]["RewardModel"]), plan)
    result = _run(elastic_simulator_binary, tmp_path, payload, threads=2)
    actual = result["functional_results"]["candidates"][0]["roles"]["RewardModel"]["result"]
    _assert_result_matches(actual, expected)
    assert expected.routed_count == 128
    assert expected.unrouted_count == 1


def test_cpp_backend_bridge_reuses_process_and_matches_python(
    elastic_simulator_binary: Path,
) -> None:
    payload = _fixture()
    snapshots = {role: RoleSnapshot.from_mapping(snapshot) for role, snapshot in payload["roles"].items()}
    raw_plans = payload["candidates"][0]["role_plans"]
    plans = {
        role: RoleCandidatePlan(
            wake_instance_ids=frozenset(raw_plan["wake_instance_ids"]),
            sleep_instance_ids=frozenset(raw_plan["sleep_instance_ids"]),
            primary_scale_up=raw_plan["primary_scale_up"],
        )
        for role, raw_plan in raw_plans.items()
    }

    with CppCandidateEvaluator(
        binary_path=elastic_simulator_binary,
        max_workers=4,
        timeout_s=10.0,
    ) as evaluator:
        first = evaluator.evaluate(snapshots=snapshots, candidate_plans=[plans])
        second = evaluator.evaluate(snapshots=snapshots, candidate_plans=[plans])

    assert first.baseline_results == second.baseline_results
    assert first.candidate_results == second.candidate_results
    assert first.logical_task_count == 4
    assert first.unique_task_count == 3
    assert first.deduplicated_task_count == 1
    assert first.bridge_wall_s > 0.0
    assert 0.0 <= first.simulation_wall_s <= first.evaluation_wall_s
    assert (
        first.rebalance_simulation_s + first.router_simulation_s - first.rebalance_router_overlap_s
    ) == pytest.approx(first.simulation_wall_s, abs=1e-9)
    for role, snapshot in snapshots.items():
        context = prepare_role_evaluation_context(snapshot)
        assert first.baseline_results[role] == evaluate_role_candidate(
            context,
            RoleCandidatePlan(),
        )
        assert first.candidate_results[0][role] == evaluate_role_candidate(
            context,
            plans[role],
        )


def test_cpp_backend_bridge_preserves_infinite_priorities(
    elastic_simulator_binary: Path,
) -> None:
    payload = _fixture()
    pending = payload["roles"]["Rollout"]["pending_requests"][0]
    pending.update(
        {
            "routing_priority": [float("-inf"), float("inf")],
            "eligible_instance_ids": [0, 1],
            "fallback_instance_ids": [0, 1],
            "candidate_priorities": [
                [0, [float("inf"), 0]],
                [1, [float("-inf"), 0]],
            ],
        }
    )
    snapshots = {role: RoleSnapshot.from_mapping(snapshot) for role, snapshot in payload["roles"].items()}
    plans = {
        role: RoleCandidatePlan(
            wake_instance_ids=frozenset(raw_plan["wake_instance_ids"]),
            sleep_instance_ids=frozenset(raw_plan["sleep_instance_ids"]),
            primary_scale_up=raw_plan["primary_scale_up"],
        )
        for role, raw_plan in payload["candidates"][0]["role_plans"].items()
    }

    with CppCandidateEvaluator(
        binary_path=elastic_simulator_binary,
        max_workers=2,
        timeout_s=10.0,
    ) as evaluator:
        first = evaluator.evaluate(snapshots=snapshots, candidate_plans=[plans])
        second = evaluator.evaluate(snapshots=snapshots, candidate_plans=[plans])

    assert first.baseline_results == second.baseline_results
    assert first.candidate_results == second.candidate_results
    for role, snapshot in snapshots.items():
        context = prepare_role_evaluation_context(snapshot)
        assert first.baseline_results[role] == evaluate_role_candidate(
            context,
            RoleCandidatePlan(),
        )
        assert first.candidate_results[0][role] == evaluate_role_candidate(
            context,
            plans[role],
        )


def test_cpp_backend_bridge_reports_exited_process(
    elastic_simulator_binary: Path,
) -> None:
    payload = _fixture()
    snapshots = {role: RoleSnapshot.from_mapping(snapshot) for role, snapshot in payload["roles"].items()}
    evaluator = CppCandidateEvaluator(
        binary_path=elastic_simulator_binary,
        max_workers=1,
        timeout_s=10.0,
    )
    try:
        process = evaluator._process
        assert process is not None
        process.terminate()
        process.wait(timeout=5.0)
        with pytest.raises(CppCandidateEvaluatorError, match="exited with code"):
            evaluator.evaluate(snapshots=snapshots, candidate_plans=[])
    finally:
        evaluator.close()


def test_cpp_rejects_unknown_schema(
    elastic_simulator_binary: Path,
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "invalid-schema.jsonl"
    input_path.write_text(
        json.dumps({"schema_version": 2, "record_type": "elastic_simulation_input"}) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [str(elastic_simulator_binary), "--input", str(input_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "unsupported schema version 2" in completed.stderr


def test_default_candidate_evaluator_binary_is_under_repository_build():
    assert default_cpp_candidate_evaluator_binary() == (
        _REPO_ROOT / "build" / "elastic_simulator" / "elastic_simulator"
    )
