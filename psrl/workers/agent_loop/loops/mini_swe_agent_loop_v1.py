import asyncio
import concurrent.futures
import logging
import os
import threading
from dataclasses import asdict

from examples.mini_swe.config import MiniSWEAgentRuntimeConfig, build_runtime_config
from examples.mini_swe.runner import build_grader_spec, parse_duration_seconds, run_agent

from psrl.environments import Environment
from psrl.sandbox import SyncSandboxManager
from psrl.workers.agent_loop.agent_data import AgentData, MiniSWEAgentData
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_RUNNER_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("MINI_SWE_RUNNER_THREADS", "128")),
    thread_name_prefix="mini-swe-runner",
)


@register("mini_swe_agent")
class MiniSWEAgentLoopV1(SessionAgentLoop):
    """Dispatch mini-SWE-agent as a black box and collect its TITO trajectory."""

    def __init__(
        self,
        context: AgentLoopContext,
        **kwargs,
    ):
        super().__init__(context=context)
        runtime_kwargs = {key: kwargs[key] for key in ("sandbox_config", "agent", "model") if key in kwargs}
        self.runtime_config: MiniSWEAgentRuntimeConfig = build_runtime_config(runtime_kwargs)
        multi_turn = context.config.gen_actor_rollout_ref.rollout.multi_turn
        if not getattr(multi_turn, "enable", False):
            raise ValueError("mini-SWE-agent v1 requires rollout.multi_turn.enable=True.")
        if context.config.gen_actor_rollout_ref.rollout.agent.traj_reward_mode != "traj":
            raise ValueError("mini-SWE-agent v1 supports only agent.traj_reward_mode=traj.")

    async def _run_agent(
        self,
        request: dict,
        observation: dict,
        runtime_config: MiniSWEAgentRuntimeConfig,
        session_id: str,
        sandbox: SyncSandboxManager,
    ) -> dict:
        runner_observation = {key: value for key, value in observation.items() if key != "runtime_config"}
        payload = {
            "base_url": self.session_api_url(session_id),
            "model": f"openai/{self.model_config.path}",
            "sampling_params": self.get_session_sampling_params(request),
            "task": observation.get("problem_statement", ""),
            "observation": runner_observation,
            "runtime_config": asdict(runtime_config),
            "max_turns": self.max_turns,
            "trajectory_id_strategy": self.trajectory_id_strategy,
            "sandbox_prefix": session_id,
        }
        episode_timeout = parse_duration_seconds(runtime_config.sandbox_config.environment.container_timeout)
        if episode_timeout is None:
            raise ValueError("MiniSWE container_timeout must be configured.")
        timeout = episode_timeout + 1200.0
        grader_spec = build_grader_spec(payload)
        prepare_task = (
            asyncio.create_task(
                self.sandbox_manager.prepare(grader_spec, backend=runtime_config.sandbox_config.backend)
            )
            if grader_spec is not None
            else None
        )
        cancel_event = threading.Event()
        runner = asyncio.get_running_loop().run_in_executor(
            _RUNNER_THREAD_POOL,
            run_agent,
            payload,
            sandbox,
            cancel_event,
        )
        try:
            return await asyncio.wait_for(asyncio.shield(runner), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            cancel_event.set()
            try:
                await asyncio.shield(runner)
            except Exception:
                pass
            raise
        finally:
            if prepare_task is not None:
                prepare_task.cancel()
                results = await asyncio.gather(prepare_task, return_exceptions=True)
                if isinstance(results[0], Exception):
                    psrl_logger.warning(f"Grader image preparation failed: {results[0]!r}.")

    async def run(
        self,
        request: dict,
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        env_class = request.get("env_class", self.config.gen_actor_rollout_ref.rollout.agent.env.name)
        data_class = request.get("data_class", self.config.gen_actor_rollout_ref.rollout.agent.data.name)
        env = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            runtime_config=self.runtime_config,
        )
        agent_data = AgentData.get_agent_data(data_class, self.config, self.reward_manager, env)
        if not isinstance(agent_data, MiniSWEAgentData):
            raise TypeError(f"mini-SWE-agent v1 requires MiniSWEAgentData, got {type(agent_data).__name__}.")
        agent_data.reset()
        observation, _ = await env.reset(task=request, seed=request.get("seed"))
        agent_data.init_trajectory(request)

        session_id: str | None = None
        sync_sandbox = None
        try:
            runtime_config = observation["runtime_config"]
            session_id = await self.create_session(request)
            sync_sandbox = self.sandbox_manager.sync(
                backend=runtime_config.sandbox_config.backend,
            )
            try:
                result = await self._run_agent(request, observation, runtime_config, session_id, sync_sandbox)
            except asyncio.TimeoutError:
                return None, TerminateReason.TRAJECTORY_TIMEOUT
            if result.get("exit_status") == "error":
                psrl_logger.error("mini-SWE-agent runner failed: %s.", result.get("error", "unknown error"))
                return None, TerminateReason.ROLLOUT_ERROR

            # Completed turns remain valid training data after a context overflow.
            context_exceeded = result.get("exit_status") == "context_exceeded"

            training_data = await self.get_primary_training_data(session_id)
            if training_data["num_turns"] == 0:
                return None, TerminateReason.UNKNOWN
            self.attach_training_data(agent_data, training_data, update_turn_counts=True)
            agent_data.set_patch(result.get("submission") or None)
            if result.get("timing") is not None:
                agent_data.set_timing(result["timing"])
            if result.get("grader_result") is not None:
                agent_data.set_grader_result(result["grader_result"])

            terminate_reason = TerminateReason.FINISHED
            if context_exceeded or len(training_data["response_ids"]) >= int(self.rollout_config.response_length):
                terminate_reason = TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
            elif training_data["num_turns"] >= self.max_turns:
                terminate_reason = TerminateReason.MAX_TURNS_EXCEEDED
            finalized = await agent_data.finalize_output()
            return (finalized, terminate_reason) if finalized is not None else (None, TerminateReason.ABORTED)
        finally:
            try:
                if sync_sandbox is not None:
                    await sync_sandbox.aclose()
            finally:
                try:
                    await env.close()
                finally:
                    if session_id is not None:
                        await self.delete_session(session_id)
