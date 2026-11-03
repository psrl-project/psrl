"""Codex harness agent loop for mathematics problems."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from examples.math_rl.prompt import SCRATCH_DIR, SOLUTION_PATH, build_task_prompt
from examples.math_rl.verifier import VerifierConfig, VerifierError
from examples.math_rl.verifier import evaluate as evaluate_proof
from omegaconf import DictConfig, OmegaConf
from psrl.sandbox import (
    ExecMode,
    ResourceSpec,
    SandboxPathNotFoundError,
    SandboxSession,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
)
from psrl.sandbox.capacity import parse_memory_mb
from psrl.workers.agent_loop.harness import (
    HarnessConfig,
    HarnessResult,
    HarnessRuntime,
    HarnessTaskContext,
)
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop
from psrl.workers.agent_loop.loops.utils import register

psrl_logger = logging.getLogger(__name__)

_PROBLEM_PATH = "/workspace/problem.md"
_DEFAULT_SOLUTION_CAP = 200_000
_DURATION_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd]?)$")


@dataclass(frozen=True)
class MathHarnessTaskState:
    """Task data the loop needs after the sandbox is gone."""

    uid: str
    problem: str
    reference_context: str
    solution_path: str


@dataclass(frozen=True)
class MathHarnessArtifact:
    """The candidate solution collected from the sandbox."""

    solution: str
    source: str
    chars: int


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        return dict(OmegaConf.to_container(value, resolve=True) or {})
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        return dict(value[0])
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _as_mapping(value.tolist())
        except Exception:
            return {}
    return {}


def _parse_duration_s(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    match = _DURATION_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"Invalid duration {value!r}, expected a number with s, m, h, or d.")
    multipliers = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    return float(match.group(1)) * multipliers[match.group(2)]


def _request_problem(request: Mapping[str, Any]) -> str:
    extra = _as_mapping(request.get("extra_info"))
    problem = str(extra.get("problem") or "").strip()
    if problem:
        return problem
    raw_prompt = request.get("raw_prompt")
    if isinstance(raw_prompt, (list, tuple)):
        for message in reversed(raw_prompt):
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    prompt = request.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    return ""


def last_assistant_message(log_bytes: bytes | None) -> str | None:
    """
    Return the last assistant message in a Codex JSONL log.

    Args:
        log_bytes (bytes | None): The CLI output.

    Returns:
        str | None: The last assistant text, or None when none is present.
    """
    if not log_bytes:
        return None
    messages: list[str] = []
    for raw_line in log_bytes.splitlines():
        line = raw_line.strip()
        if not line.startswith(b"{"):
            continue
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = str(item.get("text") or "").strip()
                if text:
                    messages.append(text)
        elif event_type == "agent_message":
            text = str(event.get("text") or "").strip()
            if text:
                messages.append(text)
        elif event_type == "response_item":
            payload = event.get("payload")
            if isinstance(payload, dict) and payload.get("type") == "message" and payload.get("role") == "assistant":
                text = _content_text(payload.get("content"))
                if text:
                    messages.append(text)
        elif event_type == "message" and event.get("role") == "assistant":
            text = _content_text(event.get("content"))
            if text:
                messages.append(text)
    return messages[-1] if messages else None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


@register("math_harness")
class MathHarnessAgentLoop(HarnessAgentLoop):
    """Run Codex on a math problem and grade the written solution."""

    def __init__(
        self,
        context,
        harness: HarnessConfig | DictConfig | Mapping[str, Any],
        sandbox_config: DictConfig | Mapping[str, Any] | None = None,
        verifier: DictConfig | Mapping[str, Any] | None = None,
        runtime_in_image: bool = False,
        solution_cap: int = _DEFAULT_SOLUTION_CAP,
        **kwargs,
    ) -> None:
        super().__init__(context=context, harness=harness)
        sandbox = _as_mapping(sandbox_config)
        environment = _as_mapping(sandbox.get("environment"))
        self.backend = str(sandbox.get("backend") or "p3b")
        self.runtime_in_image = bool(runtime_in_image)
        self.solution_cap = int(solution_cap)
        self.image = str(environment.get("image") or "python:3.12-slim")
        self.workdir = str(environment.get("cwd") or "/workspace")
        self.memory = environment.get("memory") or "8g"
        self.container_timeout = environment.get("container_timeout") or "2h"
        self.resource_class = str(environment.get("resource_class") or "rollout")
        self.cpu_count = float(sandbox.get("sandbox_cpu_count") or 2)
        self.sandbox_env = {str(key): str(value) for key, value in _as_mapping(environment.get("env")).items()}
        self.snapshot_verifier = bool(sandbox.get("snapshot_verifier", False))
        self.collect_resource_metrics = bool(sandbox.get("collect_resource_metrics", False))
        self.verifier_config = (
            VerifierConfig.from_mapping(verifier) if verifier is not None else VerifierConfig.from_env()
        )

    def attach_runtime_mount(self, task: HarnessTaskContext) -> HarnessTaskContext:
        """
        Skip the runtime mount when the sandbox image already ships Codex.

        Args:
            task (HarnessTaskContext): The task to prepare.

        Returns:
            HarnessTaskContext: The task, with a runtime mount only when needed.
        """
        if self.runtime_in_image:
            return task
        return super().attach_runtime_mount(task)

    async def prepare_harness_task(self, request: dict) -> HarnessTaskContext[MathHarnessTaskState]:
        """
        Build the prompt and sandbox specification for one problem.

        Args:
            request (dict): The rollout request.

        Returns:
            HarnessTaskContext[MathHarnessTaskState]: The task context.

        Raises:
            ValueError: If the request carries no problem statement.
        """
        extra = _as_mapping(request.get("extra_info"))
        problem = _request_problem(request)
        if not problem:
            raise ValueError(f"Request {request.get('uid')!r} carries no problem statement.")
        uid = str(extra.get("uid") or request.get("uid") or "")
        state = MathHarnessTaskState(
            uid=uid,
            problem=problem,
            reference_context=str(extra.get("reference_context") or ""),
            solution_path=SOLUTION_PATH,
        )
        return HarnessTaskContext(
            state=state,
            prompt=build_task_prompt(problem),
            sandbox_spec=self._build_sandbox_spec(uid, str(request.get("uid") or uid)),
            backend=self.backend,
            clean_sandbox_spec=None,
            collect_resource_metrics=self.collect_resource_metrics,
        )

    def _build_sandbox_spec(self, uid: str, request_uid: str) -> SandboxSpec:
        env = dict(self.sandbox_env)
        env.setdefault("HOME", self.harness_config.home_dir)
        env.setdefault("CODEX_HOME", f"{self.harness_config.home_dir}/.codex")
        return SandboxSpec(
            source=SandboxSource.image(self.image),
            resources=ResourceSpec(
                cpu_count=self.cpu_count,
                memory_mb=parse_memory_mb(self.memory),
            ),
            workdir=self.workdir,
            env=env,
            metadata={"psrl.math_uid": uid},
            idle_timeout_s=_parse_duration_s(self.container_timeout),
            # One rollout per request uid, so two rollouts of the same problem
            # never adopt each other's sandbox.
            idempotency_key=f"math-rl:{request_uid}:rollout",
            state_policy=SandboxStatePolicy(enabled=self.snapshot_verifier),
            resource_class=self.resource_class,
            backend=self.backend,
            exec_mode=ExecMode.ONE_SHOT,
        )

    async def prepare_harness_sandbox(
        self,
        task: HarnessTaskContext[MathHarnessTaskState],
        session: SandboxSession,
        timing: dict[str, float],
    ) -> None:
        """
        Create the scratch directory and stage the problem file.

        Args:
            task (HarnessTaskContext[MathHarnessTaskState]): The task.
            session (SandboxSession): The live sandbox.
            timing (dict[str, float]): Timing sink.
        """
        await session.exec(f"mkdir -p {SCRATCH_DIR}", timeout_s=60)
        await session.write_bytes(_PROBLEM_PATH, task.state.problem.encode("utf-8"))

    async def collect_harness_artifact(
        self,
        task: HarnessTaskContext[MathHarnessTaskState],
        sandbox: SandboxSession,
        runtime: HarnessRuntime,
        harness_result: HarnessResult,
        *,
        harness_log: bytes | None = None,
    ) -> MathHarnessArtifact:
        """
        Read the written solution, falling back to the last CLI message.

        Args:
            task (HarnessTaskContext[MathHarnessTaskState]): The task.
            sandbox (SandboxSession): The live sandbox.
            runtime (HarnessRuntime): Harness runtime values.
            harness_result (HarnessResult): The CLI result.
            harness_log (bytes | None): The CLI output.

        Returns:
            MathHarnessArtifact: The collected solution.
        """
        solution = ""
        source = "missing"
        try:
            raw = await sandbox.read_bytes(task.state.solution_path)
            solution = raw.decode("utf-8", errors="replace").strip()
            source = "solution_file" if solution else "missing"
        except SandboxPathNotFoundError:
            psrl_logger.warning(f"No solution file for task {task.state.uid!r}.")
        # A lost session or a transport fault must reach the worker's classifier, not
        # become a missing solution that trains an infrastructure failure as a zero.
        if not solution and harness_log:
            fallback = last_assistant_message(harness_log)
            if fallback:
                solution = fallback
                source = "harness_log"
        if len(solution) > self.solution_cap:
            solution = solution[: self.solution_cap]
            source = f"{source}_truncated"
        return MathHarnessArtifact(solution=solution, source=source, chars=len(solution))

    async def finalize_harness_task(
        self,
        task: HarnessTaskContext[MathHarnessTaskState],
        artifact: MathHarnessArtifact,
        clean_snapshot,
        timing: dict[str, float],
        grading_timeout_s: float | None,
    ) -> dict:
        """
        Grade the final proof and build reward metadata.

        Args:
            task (HarnessTaskContext[MathHarnessTaskState]): The task.
            artifact (MathHarnessArtifact): The candidate solution.
            clean_snapshot: Unused.
            timing (dict[str, float]): Timing sink.
            grading_timeout_s (float | None): Budget for the verifier call.

        Returns:
            dict: Reward metadata consumed by the harness loop.
        """
        started = time.perf_counter()
        common = {
            "solution_source": artifact.source,
            "solution_chars": artifact.chars,
        }
        if not artifact.solution:
            timing["grading_s"] = time.perf_counter() - started
            return {
                **common,
                "reward_score": 0.0,
                "verifier_verdict": "missing_solution",
                "verifier_reason": "The agent did not write a solution.",
                "verifier_graded": True,
            }
        timeout_s = grading_timeout_s or self.verifier_config.client_timeout_s
        try:
            verdict = await asyncio.wait_for(
                evaluate_proof(
                    self.verifier_config,
                    task.state.problem,
                    artifact.solution,
                    timeout_s=grading_timeout_s,
                ),
                timeout=timeout_s,
            )
        except (VerifierError, asyncio.TimeoutError) as error:
            self.grader_unavailable = True
            timing["grading_s"] = time.perf_counter() - started
            psrl_logger.error(f"Verifier failed for task {task.state.uid!r}: {error!r}.")
            return {
                **common,
                "reward_score": 0.0,
                "verifier_verdict": "error",
                "verifier_reason": "",
                "verifier_graded": False,
                "verifier_error": str(error),
            }
        timing["grading_s"] = time.perf_counter() - started
        return {
            **common,
            "reward_score": verdict.score(self.verifier_config),
            "verifier_verdict": verdict.verdict,
            "verifier_reason": verdict.reason,
            "verifier_graded": True,
            "verifier_run_id": verdict.run_id,
            "verifier_model": self.verifier_config.model or self.verifier_config.profile,
            "verifier_version": self.verifier_config.verifier_harness_version,
            "verifier_latency_s": verdict.duration_s,
            "verifier_cached": verdict.cached,
            "verifier_usage": json.dumps(dict(verdict.usage), sort_keys=True)[:2000],
            "proof_sha256": verdict.proof_sha256,
        }

    async def close_harness_task(self, task: HarnessTaskContext[MathHarnessTaskState]) -> None:
        """
        Release task resources. The base loop owns the sandbox and session.

        Args:
            task (HarnessTaskContext[MathHarnessTaskState]): The task.
        """
        return None
