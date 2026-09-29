"""Generic task lifecycle for sandboxed coding harness training."""

import asyncio
import contextlib
import logging
import os
import time
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any

from omegaconf import DictConfig

from psrl.sandbox import SandboxFeature, SandboxLease, SandboxSession, SnapshotKind, SnapshotRef, SyncSandboxManager
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.harness import (
    Harness,
    HarnessConfig,
    HarnessResult,
    HarnessRuntime,
    HarnessTaskContext,
    clean_snapshot_compatible,
    create_harness,
    runtime_mount_spec,
)
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class HarnessAgentLoop(SessionAgentLoop):
    """
    Run one task-scoped harness and turn its TITO session into training data.

    Subclasses define task preparation and may override artifact collection,
    post-rollout finalization, and task cleanup. This class remains the sole
    owner of the TITO session, sandbox lease, harness process, and snapshot.
    """

    def __init__(
        self,
        context: AgentLoopContext,
        harness: HarnessConfig | DictConfig | Mapping[str, Any],
    ) -> None:
        super().__init__(context=context)
        self.harness_config = HarnessConfig.from_value(harness)
        self.rollout_budget = int(self.rollout_config.prompt_length) + int(self.rollout_config.response_length)
        self.compaction_budget: tuple[int, int] | None = None
        self.prompt_too_long_limit: int | None = None
        multi_turn = context.config.gen_actor_rollout_ref.rollout.multi_turn
        if not getattr(multi_turn, "enable", False):
            raise ValueError("Harness training requires rollout.multi_turn.enable=True.")
        if self.trajectory_id_strategy != "auto":
            raise ValueError(
                "Harness training requires psrl.rollout_gateway.trajectory_id_strategy=auto so TITO can assign "
                "multi-agent and compaction branches from its prefix tree."
            )
        if context.config.gen_actor_rollout_ref.rollout.agent.traj_reward_mode != "traj":
            raise ValueError("Harness training supports only agent.traj_reward_mode=traj.")

    @abstractmethod
    async def prepare_harness_task(self, request: dict) -> HarnessTaskContext:
        """
        Build task state, prompt, and sandbox specifications.

        A subclass that acquires partial resources before returning must release
        them if preparation raises. Once returned, `close_harness_task` is always
        called by the generic lifecycle.
        """

    async def collect_harness_artifact(
        self,
        task: HarnessTaskContext,
        sandbox: SandboxSession,
        runtime: HarnessRuntime,
        harness_result: HarnessResult,
    ) -> Any:
        """
        Collect an optional task artifact before the agent sandbox is destroyed.
        """
        return None

    async def finalize_harness_task(
        self,
        task: HarnessTaskContext,
        artifact: Any,
        clean_snapshot: SnapshotRef | None,
        timing: dict[str, float],
    ) -> dict:
        """
        Run optional task evaluation and return task-specific reward fields.
        """
        return {}

    async def close_harness_task(self, task: HarnessTaskContext) -> None:
        """
        Release task-specific state after generic resources are cleaned up.
        """
        return None

    async def _run_sync_sandbox_operation(
        self,
        backend: str | None,
        operation: Callable[[SyncSandboxManager], Any],
    ) -> Any:
        """
        Run blocking harness code with cancellation-safe sandbox ownership.
        """
        sync_sandbox = self.sandbox_manager.sync(backend=backend)
        operation_task = asyncio.create_task(asyncio.to_thread(operation, sync_sandbox))
        try:
            return await asyncio.shield(operation_task)
        except asyncio.CancelledError:
            await sync_sandbox.aclose()
            with contextlib.suppress(Exception):
                await asyncio.shield(operation_task)
            raise
        finally:
            await sync_sandbox.aclose()

    async def prepare_harness_sandbox(
        self,
        task: HarnessTaskContext,
        session: SandboxSession,
        timing: dict[str, float],
    ) -> None:
        """
        Optionally initialize the acquired sandbox before the harness starts.

        Runs after the clean snapshot and before the harness is created, so any
        sandbox mutation is not captured by the reusable clean snapshot. An
        override may add its own float entries to ``timing``. The generic loop
        records the total as ``sandbox_init_s``.
        """
        return None

    def attach_runtime_mount(self, task: HarnessTaskContext) -> HarnessTaskContext:
        """Bind this harness's read-only runtime tree into the rollout sandbox.

        The tree carries the harness executable only (no task content), so it is
        added to the rollout spec and excluded from clean-snapshot compatibility.
        A backend without host-mount support is rejected by the sandbox manager
        when it sees the mount.
        """
        mount = runtime_mount_spec(self.harness_config.kind, self.harness_config.runtime_mount)
        spec = replace(task.sandbox_spec, mounts=(*task.sandbox_spec.mounts, mount))
        return replace(task, sandbox_spec=spec, runtime_mount_target=mount.target)

    def resolve_request_settings(self, request: dict) -> None:
        """Resolve per-request turn budget and context compaction settings."""
        rollout_config = (
            self.config.train_actor_rollout_ref.rollout
            if request.get("validate", False)
            else self.config.gen_actor_rollout_ref.rollout
        )

        self.max_turns = rollout_config.multi_turn.max_turns
        context_window = rollout_config.max_model_len or (
            rollout_config.prompt_length + rollout_config.response_length
        )
        if context_window > self.rollout_budget:
            psrl_logger.warning(
                f"Harness context window ({context_window}) exceeds the trainable budget "
                f"({self.rollout_budget}). A compaction branch may not fit. Keep "
                "rollout.max_model_len <= rollout.prompt_length + rollout.response_length."
            )
        self.compaction_budget = self.harness_config.compaction.resolve(context_window)
        self.prompt_too_long_limit = self.compaction_budget[1] if self.compaction_budget else None

    async def run(
        self,
        request: dict,
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """
        Execute the generic harness lifecycle for one training task.
        """
        self.resolve_request_settings(request)
        task: HarnessTaskContext | None = None
        session_id: str | None = None
        lease: SandboxLease | None = None
        harness: Harness | None = None
        clean_snapshot: SnapshotRef | None = None
        prepare_tasks: list[asyncio.Task[None]] = []
        run_start = time.perf_counter()
        timing = {
            "prep_s": 0.0,
            "assistant_s": 0.0,
            "env_s": 0.0,
            "elapsed_s": 0.0,
            "task_prepare_s": 0.0,
            "sandbox_create_s": 0.0,
            "snapshot_s": 0.0,
            "sandbox_init_s": 0.0,
            "harness_prepare_s": 0.0,
        }
        try:
            task = await self.prepare_harness_task(request)
            task = self.attach_runtime_mount(task)
            timing["task_prepare_s"] = time.perf_counter() - run_start
            prepare_tasks.append(
                asyncio.create_task(self.sandbox_manager.prepare(task.sandbox_spec, backend=task.backend))
            )
            if task.clean_sandbox_spec is not None:
                prepare_tasks.append(
                    asyncio.create_task(self.sandbox_manager.prepare(task.clean_sandbox_spec, backend=task.backend))
                )
            session_id = await self.create_session(request)
            await prepare_tasks[0]
            sandbox_started = time.perf_counter()
            # Retain lease ownership if cancellation races sandbox creation.
            acquire_task = asyncio.create_task(
                self.sandbox_manager.acquire(task.sandbox_spec, backend=task.backend),
            )
            try:
                lease = await asyncio.shield(acquire_task)
            except asyncio.CancelledError:
                lease = await acquire_task
                raise
            timing["sandbox_create_s"] = time.perf_counter() - sandbox_started
            snapshot_started = time.perf_counter()
            clean_snapshot = await self._try_snapshot_clean_sandbox(task, lease)
            timing["snapshot_s"] = time.perf_counter() - snapshot_started

            sandbox_init_started = time.perf_counter()
            await self.prepare_harness_sandbox(task, lease.session, timing)
            timing["sandbox_init_s"] = time.perf_counter() - sandbox_init_started

            session_root_url = self.session_root_url(session_id, self.harness_config.callback_base_url)
            session_root_url = lease.session.resolve_callback_url(session_root_url)
            harness = create_harness(self.harness_config, lease.session)
            harness_runtime = HarnessRuntime(
                session_id=session_id,
                session_root_url=session_root_url,
                workdir=task.sandbox_spec.workdir or "/",
                model=str(self.model_config.path),
                context_window_tokens=(self.compaction_budget[0] if self.compaction_budget else None),
                compaction_token_limit=(self.compaction_budget[1] if self.compaction_budget else None),
                max_turns=int(self.max_turns) if self.max_turns is not None else None,
            )
            prepare_started = time.perf_counter()
            await harness.prepare(harness_runtime)
            timing["harness_prepare_s"] = time.perf_counter() - prepare_started
            timing["prep_s"] = time.perf_counter() - run_start

            harness_started = time.perf_counter()
            try:
                harness_result = await harness.run(task.prompt, harness_runtime)
            except TimeoutError:
                return None, TerminateReason.TRAJECTORY_TIMEOUT
            timing["assistant_s"] = time.perf_counter() - harness_started

            artifact = await self.collect_harness_artifact(
                task,
                lease.session,
                harness_runtime,
                harness_result,
            )
            await self._capture_resource_metrics(task, lease, timing)
            training_data = await self.get_training_data(session_id)

            await lease.release()
            lease = None
            await self.delete_session(session_id)
            session_id = None

            if not training_data or any(item["num_turns"] <= 0 or not item["response_ids"] for item in training_data):
                harness_result = await harness.collect_diagnostics(harness_result)
                await self._dump_harness_training_data(
                    request,
                    training_data,
                    TerminateReason.ROLLOUT_ERROR,
                    self._build_partial_reward_info(training_data, harness_result),
                )
                raise RuntimeError(
                    f"{self.harness_config.kind} produced no usable TITO trajectory.\n"
                    f"Harness process diagnostics:\n{harness_result.diagnostic_text()}"
                )

            # Preparation is speculative: a clean snapshot may already provide
            # the grader image, and finalization can retry a failed image pull.
            prepare_results = await asyncio.gather(*prepare_tasks, return_exceptions=True)
            for result in prepare_results:
                if isinstance(result, Exception):
                    psrl_logger.warning(f"Sandbox artifact preparation failed: {result!r}.")
            task_reward_info = await self.finalize_harness_task(
                task,
                artifact,
                clean_snapshot,
                timing,
            )
            timing["elapsed_s"] = time.perf_counter() - run_start
            reward_info = self._build_reward_info(
                task_reward_info,
                training_data,
                timing,
                harness_result,
            )
            try:
                outputs = [self._build_capped_output(item) for item in training_data]
            except Exception:
                # The raw TITO trajectory is still valuable when the training budget cannot represent it,
                # for example when the harness system/tool prompt already exceeds prompt+response.
                #
                # AgentLoopBase's normal dump is unreachable here, because run_with_termination_handling
                # receives no output.
                await self._dump_harness_training_data(
                    request,
                    training_data,
                    TerminateReason.ROLLOUT_ERROR,
                    reward_info,
                )
                raise
            for output in outputs:
                output.agent_reward_info = dict(reward_info)
            self.attach_tito_tree_metadata(training_data, outputs)
            output_value: TokenOutput | list[TokenOutput] = outputs[0] if len(outputs) == 1 else outputs
            scored_output = await self.compute_reward_score(output_value, **request)
            if scored_output is None:
                return None, TerminateReason.ABORTED
            return scored_output, self.get_harness_terminate_reason(training_data)
        finally:
            for prepare_task in prepare_tasks:
                prepare_task.cancel()
            await asyncio.gather(*prepare_tasks, return_exceptions=True)
            await self._cleanup_harness_run(task, session_id, lease, harness, clean_snapshot)

    async def _try_snapshot_clean_sandbox(
        self,
        task: HarnessTaskContext,
        lease: SandboxLease,
    ) -> SnapshotRef | None:
        caps = lease.session.capabilities
        # The sandbox has not been prepared yet. Docker already shares the
        # original image layers. Committing here adds I/O without caching setup.
        if not caps.supports(SandboxFeature.FULL_STATE_SNAPSHOT):
            return None
        kind = SnapshotKind.FULL_STATE
        if not caps.supports(SandboxFeature.RESTORE):
            return None
        if not clean_snapshot_compatible(task, kind):
            return None
        try:
            return await self.sandbox_manager.checkpoint(
                lease.session,
                kind,
                task.sandbox_spec.state_policy,
            )
        except Exception:
            psrl_logger.warning(
                "Could not create a safe clean-sandbox snapshot. Task finalization will provision a fresh sandbox.",
                exc_info=True,
            )
            return None

    async def _cleanup_harness_run(
        self,
        task: HarnessTaskContext | None,
        session_id: str | None,
        lease: SandboxLease | None,
        harness: Harness | None,
        clean_snapshot: SnapshotRef | None,
    ) -> None:
        cleanup_task = asyncio.create_task(
            self._cleanup_harness_resources(task, session_id, lease, harness, clean_snapshot),
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            await cleanup_task
            raise

    async def _cleanup_harness_resources(
        self,
        task: HarnessTaskContext | None,
        session_id: str | None,
        lease: SandboxLease | None,
        harness: Harness | None,
        clean_snapshot: SnapshotRef | None,
    ) -> None:
        if harness is not None:
            await self._run_cleanup_operation("harness abort", harness.abort())
        if lease is not None:
            await self._run_cleanup_operation("sandbox release", lease.release())
        if session_id is not None:
            await self._run_cleanup_operation("session deletion", self.delete_session(session_id))
        if clean_snapshot is not None:
            await self._run_cleanup_operation(
                "clean snapshot deletion",
                self.sandbox_manager.delete_snapshot(clean_snapshot),
            )
        if task is not None:
            await self._run_cleanup_operation("task cleanup", self.close_harness_task(task))

    @staticmethod
    async def _run_cleanup_operation(name: str, operation: Awaitable[None]) -> None:
        """
        Isolate one cleanup failure so subsequent resources are still released.
        """
        try:
            await operation
        except Exception as exc:
            psrl_logger.warning(f"Harness cleanup operation {name!r} failed: {exc!r}.")

    @staticmethod
    async def _capture_resource_metrics(
        task: HarnessTaskContext,
        lease: SandboxLease,
        timing: dict[str, float],
    ) -> None:
        if not task.collect_resource_metrics:
            return
        try:
            usage = await lease.session.stats()
            timing["sandbox_memory_mib"] = usage.memory_bytes / (1024 * 1024)
            timing["sandbox_peak_memory_mib"] = usage.peak_memory_bytes / (1024 * 1024)
            timing["sandbox_cpu_total_s"] = usage.cpu_total_ns / 1_000_000_000
        except Exception:
            psrl_logger.warning("Could not collect harness sandbox resource metrics.", exc_info=True)

    def _build_reward_info(
        self,
        task_reward_info: dict,
        training_data: list[dict],
        timing: dict[str, float],
        harness_result: HarnessResult,
    ) -> dict:
        return {
            **task_reward_info,
            **self._build_partial_reward_info(training_data, harness_result),
            "timing": timing,
            "harness_stdout_path": harness_result.stdout_path,
            "harness_stderr_path": harness_result.stderr_path,
        }

    def _build_partial_reward_info(
        self,
        training_data: list[dict],
        harness_result: HarnessResult,
    ) -> dict:
        """Build diagnostics that are safe even for an incomplete trajectory."""
        num_turns = max((item.get("num_turns", 0) for item in training_data), default=0)
        return {
            "num_turns": num_turns,
            "actual_num_turns": num_turns,
            "trajectory_count": len(training_data),
            "trajectory_token_lengths": [
                len(item.get("prompt_ids", [])) + len(item.get("response_ids", [])) for item in training_data
            ],
            "compaction_context_window_tokens": (self.compaction_budget[0] if self.compaction_budget else None),
            "compaction_token_limit": self.compaction_budget[1] if self.compaction_budget else None,
            "harness": self.harness_config.kind,
            "harness_exit_code": harness_result.exit_code,
            "harness_stderr_tail": harness_result.stderr_tail if harness_result.exit_code != 0 else "",
        }

    async def _dump_harness_training_data(
        self,
        request: dict,
        training_data: list[dict],
        terminate_reason: TerminateReason,
        reward_info: dict,
    ) -> None:
        """Persist raw TITO text before an error prevents normal output handling.

        Successful rollouts are written by ``AgentLoopBase`` after the final
        output is scored. Error paths can fail before that hook, so use the
        untruncated TITO data here. This preserves the actual prompt,
        assistant turns, observations, and compaction branches for debugging.
        """
        if not training_data or not getattr(self, "traj_writer", None) or not self.traj_writer.enable:
            return
        try:
            outputs = [self.build_token_output(item) for item in training_data]
            for output in outputs:
                output.agent_reward_info = dict(reward_info)
            self.attach_tito_tree_metadata(training_data, outputs)
            output_value: TokenOutput | list[TokenOutput] = outputs[0] if len(outputs) == 1 else outputs
            await self._resolve_version_for_dump(output_value, request)
            self._attach_loop_timing(output_value)
            self._dump_trajectory_text(request, output_value, terminate_reason)
        except Exception:
            psrl_logger.warning(
                "Failed to dump raw harness trajectory for uid=%s.",
                request.get("uid", "N/A"),
                exc_info=True,
            )

    def _build_capped_output(self, training_data: dict) -> TokenOutput:
        output = self.build_token_output(training_data)
        # Budget the trainable trajectory against the rollout packing budget (prompt_length + response_length),
        # not the CLI compaction trigger.
        #
        # A TITO compaction branch legitimately carries the whole pre-compaction context as its prompt,
        # which can be far larger than the initial prompt.
        remaining_response_tokens = self.rollout_budget - len(output.prompt_ids)
        if remaining_response_tokens <= 0:
            raise RuntimeError(
                f"TITO trajectory {training_data.get('trajectory_id', '<unknown>')} has prompt length "
                f"{len(output.prompt_ids)} >= trajectory budget {self.rollout_budget}. "
                "The harness prompt/system context is larger than the configured training budget "
                "(rollout.prompt_length + rollout.response_length). Increase the training budget "
                "(keeping the sum within the model context window) or reduce the harness "
                "system/tool prompt."
            )
        response_length = remaining_response_tokens
        output.response_ids = output.response_ids[:response_length]
        output.response_mask = output.response_mask[:response_length]
        if output.response_log_probs is not None:
            output.response_log_probs = output.response_log_probs[:response_length]
        if output.routed_experts is not None:
            output.routed_experts = output.routed_experts[: len(output.prompt_ids) + response_length]
        return output

    def get_harness_terminate_reason(self, training_data: list[dict]) -> TerminateReason:
        """
        Map captured TITO data to the framework's successful stop reasons.
        """
        if any(len(item["response_ids"]) >= int(self.rollout_config.response_length) for item in training_data):
            return TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
        if max(item["num_turns"] for item in training_data) >= self.max_turns:
            return TerminateReason.MAX_TURNS_EXCEEDED
        return TerminateReason.FINISHED
