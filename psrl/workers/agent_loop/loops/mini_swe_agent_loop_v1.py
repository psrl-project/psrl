import asyncio
import concurrent.futures
import logging
import os
import re
from dataclasses import asdict

from examples.mini_swe.config import MiniSWEAgentRuntimeConfig, build_runtime_config
from examples.mini_swe.runner import run_agent

from psrl.environments import Environment
from psrl.utils.concurrency import SlotManager
from psrl.workers.agent_loop.agent_data import AgentData, MiniSWEAgentData
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_DEFAULT_EPISODE_TIMEOUT_SECS = 7200.0
_RUNNER_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=int(os.getenv("MINI_SWE_RUNNER_THREADS", "128")),
    thread_name_prefix="mini-swe-runner",
)


def _parse_timeout_secs(value: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms]?)", str(value).strip().lower())
    if match is None:
        return _DEFAULT_EPISODE_TIMEOUT_SECS
    return float(match.group(1)) * {"h": 3600.0, "m": 60.0, "s": 1.0, "": 1.0}[match.group(2)]


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
            "actor_id": os.getenv("PSRL_ACTOR_ID", ""),
        }
        timeout = _parse_timeout_secs(runtime_config.sandbox_config.environment.container_timeout) + 1200.0
        return await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(_RUNNER_THREAD_POOL, run_agent, payload),
            timeout=timeout,
        )

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
        run_slot: tuple[int, int] | None = None
        try:
            runtime_config = observation["runtime_config"]
            parallelism = runtime_config.sandbox_config.max_parallel_tasks_per_worker
            if parallelism > 0:
                namespace = os.path.join(
                    str(self.config.trainer.project_name),
                    str(self.config.trainer.experiment_name),
                )
                run_slot = await SlotManager.acquire(parallelism, namespace, prefix="psrl_mini_swe_agent_slots")

            session_id = await self.create_session(request)
            try:
                result = await self._run_agent(request, observation, runtime_config, session_id)
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
                await env.close()
            finally:
                SlotManager.release(run_slot)
                if session_id is not None:
                    await self.delete_session(session_id)
