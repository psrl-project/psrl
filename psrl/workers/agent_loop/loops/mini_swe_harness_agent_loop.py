"""Mini-SWE task hooks for the generic harness agent loop."""

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import asdict, dataclass

from examples.mini_swe.config import MiniSWEAgentRuntimeConfig, build_runtime_config
from examples.mini_swe.runner import build_grader_spec, build_sandbox_spec, grade_patch
from examples.mini_swe.swebench_grader import analyze_patch_policy
from examples.mini_swe.utils.git_sanitize import ensure_git_sanitized, repository_mount_targets
from examples.mini_swe.utils.harness_task import build_harness_prompt, collect_git_patch
from examples.mini_swe.utils.integrity import scan_trajectory_integrity

from psrl.environments import Environment
from psrl.sandbox import SandboxSession, SnapshotRef
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.harness import HarnessResult, HarnessRuntime, HarnessTaskContext
from psrl.workers.agent_loop.loops.harness_agent_loop import HarnessAgentLoop
from psrl.workers.agent_loop.loops.utils import register

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class MiniSWEHarnessTaskState:
    """
    Store Mini-SWE resources and grader payload owned by the task hook.
    """

    environment: Environment
    payload: dict


@dataclass(frozen=True)
class MiniSWEHarnessArtifact:
    """Store patch and post-hoc harness integrity diagnostics."""

    patch: str
    integrity: dict


@register("mini_swe_harness")
class MiniSWEHarnessAgentLoop(HarnessAgentLoop):
    """
    Supply Mini-SWE environment, patch, grader, and reward hooks.
    """

    def __init__(self, context: AgentLoopContext, **kwargs) -> None:
        if "harness" not in kwargs:
            raise ValueError("mini-SWE harness loop requires a harness configuration.")
        super().__init__(context=context, harness=kwargs["harness"])
        runtime_kwargs = {key: kwargs[key] for key in ("sandbox_config", "agent", "model") if key in kwargs}
        self.runtime_config: MiniSWEAgentRuntimeConfig = build_runtime_config(
            runtime_kwargs,
            require_agent_templates=False,
        )

    async def prepare_harness_task(self, request: dict) -> HarnessTaskContext[MiniSWEHarnessTaskState]:
        """
        Prepare Mini-SWE task data without taking ownership of generic resources.
        """
        env_class = request.get("env_class", self.config.gen_actor_rollout_ref.rollout.agent.env.name)
        environment = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            runtime_config=self.runtime_config,
        )
        try:
            observation, _ = await environment.reset(task=request, seed=request.get("seed"))
            runtime_config = observation["runtime_config"]
            payload = self._build_task_payload(observation, runtime_config)
            sandbox_spec = build_sandbox_spec(payload)
            grader_spec = build_grader_spec(payload)
            state = MiniSWEHarnessTaskState(
                environment=environment,
                payload=payload,
            )
            return HarnessTaskContext(
                state=state,
                prompt=build_harness_prompt(observation.get("problem_statement", "")),
                sandbox_spec=sandbox_spec,
                backend=runtime_config.sandbox_config.backend,
                clean_sandbox_spec=grader_spec,
                collect_resource_metrics=runtime_config.sandbox_config.collect_resource_metrics,
            )
        except BaseException:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(environment.close())
            raise

    async def prepare_harness_sandbox(
        self,
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        session: SandboxSession,
        timing: dict[str, float],
    ) -> None:
        """
        Drop leaked git metadata before the harness starts.

        Baked task images are already clean, so the probe normally returns
        immediately. Only unbaked images pay for the fallback purge. A host
        bind-mounted repository is skipped because purging would mutate the
        host checkout.
        """
        observation = task.state.payload["observation"]
        if self._uses_host_mounted_repository(task, observation):
            psrl_logger.debug("Skipping git sanitization for a host-mounted repository.")
            return
        swe_problem = observation.get("swe_problem", {}) or {}
        timing.update(
            await ensure_git_sanitized(
                session,
                task.sandbox_spec.workdir or "/testbed",
                base_commit=str(swe_problem.get("base_commit") or "") or None,
            )
        )

    @staticmethod
    def _uses_host_mounted_repository(
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        observation: dict,
    ) -> bool:
        """Whether the task workdir aliases a host bind-mounted checkout."""
        if observation.get("repo_path") and not observation.get("use_preexisting_repo", True):
            return True
        workdir = task.sandbox_spec.workdir
        return bool(workdir) and workdir in repository_mount_targets(task.sandbox_spec.mounts)

    async def collect_harness_artifact(
        self,
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        sandbox: SandboxSession,
        runtime: HarnessRuntime,
        harness_result: HarnessResult,
    ) -> MiniSWEHarnessArtifact:
        """
        Capture the repository patch and harness tool trace before teardown.
        """
        swe_problem = task.state.payload["observation"].get("swe_problem", {})
        patch = await collect_git_patch(
            sandbox,
            runtime.workdir,
            base_commit=str(swe_problem.get("base_commit") or "") or None,
        )
        integrity: dict = {
            "violated": False,
            "scannable": True,
            "reasons": [],
            "violations": [],
            "parsed_lines": 0,
            "malformed_lines": 0,
            "tool_calls": 0,
        }
        if harness_result.stdout_path:
            try:
                log_bytes = await sandbox.read_bytes(harness_result.stdout_path)
            except Exception as exc:
                integrity.update(
                    {
                        "violated": True,
                        "reasons": ["integrity_trace_unavailable"],
                        "scan_error": str(exc),
                    }
                )
                psrl_logger.warning("Could not read the harness tool trace for integrity analysis.", exc_info=True)
            else:
                trajectory_format = self.harness_config.resolved_trajectory_format()
                integrity = scan_trajectory_integrity(
                    log_bytes,
                    trajectory_format,
                    str(swe_problem.get("repo") or ""),
                )
                if not integrity.get("scannable", True):
                    psrl_logger.warning(
                        f"Harness integrity scan could not parse the {trajectory_format!r} trajectory: "
                        f"{integrity.get('scan_note', 'unknown reason')}."
                    )
        return MiniSWEHarnessArtifact(patch=patch, integrity=integrity)

    async def finalize_harness_task(
        self,
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        artifact: MiniSWEHarnessArtifact,
        clean_snapshot: SnapshotRef | None,
        timing: dict[str, float],
    ) -> dict:
        """
        Grade the Mini-SWE patch and return reward-specific metadata.
        """
        grading_started = time.perf_counter()
        observation = task.state.payload["observation"]
        swe_problem = observation.get("swe_problem", {}) or {}
        # The final-patch re-check runs independently of the trajectory scan, so a trajectory violation
        # never hides a protected-path edit in the patch.
        #
        # Only fresh-container tasks enforce the patch policy, toy tasks gain no new test/config-file penalty.
        patch_policy: dict = {}
        if artifact.patch and observation.get("swe_grader") == "swebench_fresh_container":
            patch_policy = analyze_patch_policy(artifact.patch, swe_problem)
        if artifact.integrity.get("violated") or patch_policy.get("violated"):
            grader_result = self._integrity_failure_result(swe_problem, artifact.integrity, patch_policy)
        else:
            grader_result = await self._grade_patch(task, artifact.patch, clean_snapshot)
        timing["grading_s"] = time.perf_counter() - grading_started
        result = grader_result or {}
        return {
            "patch": artifact.patch or None,
            "integrity": artifact.integrity,
            "patch_policy": patch_policy,
            "alignment_failed": False,
            "alignment_failure_reason": "",
            "grader_result": result,
            "acc": float(bool(result.get("resolved", False))),
        }

    @staticmethod
    def _integrity_failure_result(swe_problem: dict, integrity: dict, patch_policy: dict | None = None) -> dict:
        """Return a grader-shaped failure without executing protected output."""
        policy_reasons = [f"trajectory:{reason}" for reason in integrity.get("reasons", [])]
        policy_reasons += [f"patch:{reason}" for reason in (patch_policy or {}).get("reasons", [])]
        return {
            "policy_violated": True,
            "policy_reasons": policy_reasons,
            "resolved": False,
            "apply_ok": False,
            "f2p_pass": 0,
            "f2p_total": len(swe_problem.get("FAIL_TO_PASS", [])),
            "p2p_pass": 0,
            "p2p_total": len(swe_problem.get("PASS_TO_PASS", [])),
            "timeout": False,
            "error": None,
            "elapsed_s": 0.0,
            "output_tail": "",
            "resolved_by": "integrity_blocked",
        }

    async def close_harness_task(self, task: HarnessTaskContext[MiniSWEHarnessTaskState]) -> None:
        """
        Close the Mini-SWE environment.
        """
        await task.state.environment.close()

    def _build_task_payload(
        self,
        observation: dict,
        runtime_config: MiniSWEAgentRuntimeConfig,
    ) -> dict:
        return {
            "observation": {key: value for key, value in observation.items() if key != "runtime_config"},
            "runtime_config": {"sandbox_config": asdict(runtime_config.sandbox_config)},
        }

    async def _grade_patch(
        self,
        task: HarnessTaskContext[MiniSWEHarnessTaskState],
        patch: str,
        clean_snapshot: SnapshotRef | None,
    ) -> dict | None:
        if not patch:
            return None
        return await self._run_sync_sandbox_operation(
            task.backend,
            lambda sync_sandbox: grade_patch(
                task.state.payload,
                patch,
                sync_sandbox,
                clean_snapshot,
                task.clean_sandbox_spec,
            ),
        )
