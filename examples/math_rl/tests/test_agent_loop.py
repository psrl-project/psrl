"""Tests for the math harness agent loop."""

from __future__ import annotations

import asyncio
import json

import pytest
from examples.math_rl.agent_loop import (
    MathHarnessAgentLoop,
    MathHarnessArtifact,
    MathHarnessTaskState,
    _parse_duration_s,
    last_assistant_message,
)
from examples.math_rl.prompt import SOLUTION_PATH
from examples.math_rl.verifier import Verdict, VerifierConfig, VerifierError
from psrl.sandbox import MountSpec, SandboxPathNotFoundError, SandboxSource, SandboxSpec
from psrl.workers.agent_loop.harness import HarnessConfig, HarnessResult, HarnessTaskContext


def _mock_verifier() -> VerifierConfig:
    return VerifierConfig(
        endpoint="http://127.0.0.1:1/verify",
        verifier_harness_version="mock-v1",
        model="mock-verifier",
    )


def _bare_loop(runtime_in_image: bool = False, verifier: VerifierConfig | None = None) -> MathHarnessAgentLoop:
    loop = object.__new__(MathHarnessAgentLoop)
    loop.harness_config = HarnessConfig(
        kind="codex",
        executable="codex",
        home_dir="/root",
        runtime_mount="/usr/local",
    )
    loop.backend = "p3b"
    loop.runtime_in_image = runtime_in_image
    loop.solution_cap = 200_000
    loop.image = "math-rl-codex:local"
    loop.workdir = "/workspace"
    loop.memory = "8g"
    loop.container_timeout = "2h"
    loop.resource_class = "rollout"
    loop.cpu_count = 2.0
    loop.sandbox_env = {}
    loop.snapshot_verifier = False
    loop.collect_resource_metrics = False
    loop.verifier_config = verifier or _mock_verifier()
    loop.grader_unavailable = False
    loop.grader_capacity_timeout = False
    return loop


def _task(problem: str = "Prove P.") -> HarnessTaskContext[MathHarnessTaskState]:
    state = MathHarnessTaskState(
        uid="u_1",
        problem=problem,
        reference_context="Reference.",
        solution_path=SOLUTION_PATH,
    )
    spec = SandboxSpec(source=SandboxSource.image("math-rl-codex:local"))
    return HarnessTaskContext(state=state, prompt="prompt", sandbox_spec=spec)


class _MissingSandbox:
    async def read_bytes(self, path: str) -> bytes:
        raise SandboxPathNotFoundError("missing")


class _FileSandbox:
    async def read_bytes(self, path: str) -> bytes:
        return b"A complete proof."


class _LostSandbox:
    async def read_bytes(self, path: str) -> bytes:
        raise RuntimeError("sandbox is gone")


def test_parse_duration_accepts_compact_units():
    assert _parse_duration_s("2h") == 7200.0
    assert _parse_duration_s("30m") == 1800.0
    assert _parse_duration_s(15) == 15.0
    assert _parse_duration_s(None) is None
    with pytest.raises(ValueError, match="Invalid duration"):
        _parse_duration_s("soon")


def test_last_assistant_message_reads_codex_events():
    log = (
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}})
        + "\n"
        + json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "final"}})
        + "\n"
    ).encode()
    assert last_assistant_message(log) == "final"
    assert last_assistant_message(b"not json") is None


def test_prepare_harness_task_builds_the_sandbox():
    loop = _bare_loop()
    request = {
        "uid": "u_1",
        "extra_info": {
            "uid": "u_1",
            "problem": "Prove P.",
            "reference_context": "Reference.",
        },
    }
    task = asyncio.run(loop.prepare_harness_task(request))
    assert task.state.problem == "Prove P."
    assert "Prove P." in task.prompt
    assert task.sandbox_spec.workdir == "/workspace"
    assert task.sandbox_spec.env["CODEX_HOME"] == "/root/.codex"
    assert task.sandbox_spec.source.reference == "math-rl-codex:local"
    assert task.sandbox_spec.resource_class == "rollout"
    assert task.sandbox_spec.idempotency_key == "math-rl:u_1:rollout"
    assert task.backend == "p3b"


def test_two_rollouts_of_one_problem_get_distinct_sandbox_keys():
    loop = _bare_loop()
    extra = {"uid": "u_1", "problem": "Prove P."}

    async def prepare(request_uid: str):
        return await loop.prepare_harness_task({"uid": request_uid, "extra_info": extra})

    first = asyncio.run(prepare("rollout_a"))
    second = asyncio.run(prepare("rollout_b"))
    assert first.sandbox_spec.idempotency_key != second.sandbox_spec.idempotency_key


def test_runtime_mount_is_skipped_when_the_image_has_codex():
    loop = _bare_loop(runtime_in_image=True)
    task = _task()
    assert loop.attach_runtime_mount(task) is task


def test_runtime_mount_is_added_when_the_image_lacks_codex(monkeypatch):
    loop = _bare_loop(runtime_in_image=False)
    task = _task()
    monkeypatch.setattr(
        "psrl.workers.agent_loop.loops.harness_agent_loop.runtime_mount_spec",
        lambda kind, target: MountSpec("/host/codex", target, read_only=True),
    )
    result = loop.attach_runtime_mount(task)
    assert result.runtime_mount_target == "/usr/local"
    assert result.sandbox_spec.mounts[0].target == "/usr/local"


def test_collect_artifact_falls_back_to_the_harness_log():
    loop = _bare_loop()
    log = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "fallback proof"}}).encode()
    artifact = asyncio.run(
        loop.collect_harness_artifact(
            _task(),
            _MissingSandbox(),
            None,
            HarnessResult(exit_code=0),
            harness_log=log,
        )
    )
    assert artifact.source == "harness_log"
    assert artifact.solution == "fallback proof"
    assert artifact.chars == len("fallback proof")


def test_collect_artifact_prefers_the_solution_file():
    loop = _bare_loop()
    artifact = asyncio.run(
        loop.collect_harness_artifact(
            _task(),
            _FileSandbox(),
            None,
            HarnessResult(exit_code=0),
            harness_log=None,
        )
    )
    assert artifact.source == "solution_file"
    assert artifact.solution == "A complete proof."


def test_collect_artifact_propagates_a_lost_sandbox():
    loop = _bare_loop()
    with pytest.raises(RuntimeError, match="sandbox is gone"):
        asyncio.run(
            loop.collect_harness_artifact(
                _task(),
                _LostSandbox(),
                None,
                HarnessResult(exit_code=0),
                harness_log=None,
            )
        )


def test_finalize_marks_a_missing_solution_as_zero():
    loop = _bare_loop()
    artifact = MathHarnessArtifact(solution="", source="missing", chars=0)
    result = asyncio.run(loop.finalize_harness_task(_task(), artifact, None, {}, 30))
    assert result["reward_score"] == 0.0
    assert result["verifier_verdict"] == "missing_solution"
    assert result["verifier_graded"] is True
    assert loop.grader_unavailable is False


def test_finalize_carries_the_verifier_verdict_and_receipt(monkeypatch):
    loop = _bare_loop()
    artifact = MathHarnessArtifact(solution="proof", source="solution_file", chars=5)

    async def fake_evaluate(config, problem, solution, timeout_s=None):
        return Verdict(
            verdict="6",
            reason="one gap",
            run_id="run-1",
            duration_s=12.5,
            usage={"input_tokens": 10},
            proof_sha256="a" * 64,
        )

    monkeypatch.setattr("examples.math_rl.agent_loop.evaluate_proof", fake_evaluate)
    result = asyncio.run(loop.finalize_harness_task(_task(), artifact, None, {}, 30))
    assert result["reward_score"] == 0.8
    assert result["verifier_verdict"] == "6"
    assert result["verifier_graded"] is True
    assert result["verifier_run_id"] == "run-1"
    assert result["proof_sha256"] == "a" * 64
    assert result["verifier_cached"] is False
    assert loop.grader_unavailable is False


def test_finalize_uses_the_binary_mapping_when_configured(monkeypatch):
    verifier = VerifierConfig(
        endpoint="http://127.0.0.1:1/verify",
        verifier_harness_version="mock-v1",
        model="mock-verifier",
        reward_mode="binary",
    )
    loop = _bare_loop(verifier=verifier)
    artifact = MathHarnessArtifact(solution="proof", source="solution_file", chars=5)

    async def fake_evaluate(config, problem, solution, timeout_s=None):
        return Verdict(verdict="6", reason="minor")

    monkeypatch.setattr("examples.math_rl.agent_loop.evaluate_proof", fake_evaluate)
    result = asyncio.run(loop.finalize_harness_task(_task(), artifact, None, {}, 30))
    assert result["reward_score"] == 1.0


def test_finalize_marks_a_verifier_failure_as_ungraded(monkeypatch):
    loop = _bare_loop()
    artifact = MathHarnessArtifact(solution="proof", source="solution_file", chars=5)

    async def fake_evaluate(config, problem, solution, timeout_s=None):
        raise VerifierError("endpoint down", kind="transport")

    monkeypatch.setattr("examples.math_rl.agent_loop.evaluate_proof", fake_evaluate)
    result = asyncio.run(loop.finalize_harness_task(_task(), artifact, None, {}, 30))
    assert result["reward_score"] == 0.0
    assert result["verifier_verdict"] == "error"
    assert result["verifier_graded"] is False
    assert loop.grader_unavailable is True
