"""mini-SWE-agent loop backed by PSRL's asynchronous generation path."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import queue
import re
import time
from dataclasses import dataclass

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

import litellm  # noqa: E402
import ray  # noqa: E402
from examples.mini_swe.config import MiniSWEAgentRuntimeConfig, build_runtime_config  # noqa: E402
from litellm import ModelResponse  # noqa: E402

# litellm prints a "Give Feedback / Get Help" banner via bare print() on every mapped
# exception (e.g. vLLM 400 on overlong prompts, an expected case here). Silence it.
litellm.suppress_debug_info = True
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.environments.docker import DockerEnvironment  # noqa: E402
from minisweagent.exceptions import FormatError, InterruptAgentFlow  # noqa: E402
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402
from verl.utils.dataset.rl_dataset import RLHFDataset  # noqa: E402

from psrl.environments import Environment  # noqa: E402
from psrl.utils.concurrency import SlotManager  # noqa: E402
from psrl.workers.agent_loop.agent_data import (  # noqa: E402
    AgentData,
    MiniSWEAgentData,
    normalize_openai_messages,
)
from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase  # noqa: E402
from psrl.workers.agent_loop.loops.utils import DictConfigWrap, TerminateReason, register  # noqa: E402
from psrl.workers.gen.utils import TokenOutput  # noqa: E402

_MIN_GEN_TOKENS = 256
_QUEUE_POLL_INTERVAL = 0.05
_DEFAULT_EPISODE_TIMEOUT_SECS = 7200.0
_XML_FC_SENTINEL = "__XML_FUNCTION_CALLING__"
_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
]

_AGENT_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=512,
    thread_name_prefix="mini-swe-agent",
)
_GRADER_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="mini-swe-grader",
)


def _parse_timeout_secs(container_timeout: str) -> float:
    """Parse a Docker-style duration string into seconds."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms]?)", str(container_timeout).strip().lower())
    if match is None:
        return _DEFAULT_EPISODE_TIMEOUT_SECS
    multipliers = {"h": 3600.0, "m": 60.0, "s": 1.0, "": 1.0}
    return float(match.group(1)) * multipliers[match.group(2)]


def _classify_termination(
    thread_exit_status: str,
    num_turns: int,
    max_turns: int,
) -> TerminateReason:
    """Map mini-SWE-agent exit state to PSRL termination semantics."""
    if thread_exit_status == "PromptBudgetExhausted":
        return TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
    if thread_exit_status == "EpisodeTimeout":
        return TerminateReason.TRAJECTORY_TIMEOUT
    if thread_exit_status == "RolloutReturnedNone":
        return TerminateReason.ABORTED
    if thread_exit_status in {
        "AgentThreadTimeout",
        "QueryTimeout",
        "RolloutError",
        "error",
    }:
        return TerminateReason.ROLLOUT_ERROR
    if num_turns >= max_turns:
        return TerminateReason.MAX_TURNS_EXCEEDED
    if num_turns == 0:
        return TerminateReason.UNKNOWN
    return TerminateReason.FINISHED


@dataclass
class _TerminateSignal:
    """Tell the synchronous mini-SWE-agent thread to stop."""

    exit_status: str
    content: str = ""


class _PSRLModel(LitellmTextbasedModel):
    """Bridge mini-SWE-agent's synchronous model API to PSRL generation."""

    def __init__(
        self,
        request_queue: queue.Queue,
        response_queue: queue.Queue,
        query_timeout: int,
        **model_kwargs,
    ):
        super().__init__(**model_kwargs)
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._query_timeout = query_timeout

    def query(self, messages: list[dict], **kwargs) -> dict:
        """Send messages to the async loop and wait for generated text."""
        self._request_queue.put(messages)
        try:
            item = self._response_queue.get(timeout=self._query_timeout)
        except queue.Empty:
            item = _TerminateSignal("QueryTimeout")

        if isinstance(item, _TerminateSignal):
            raise InterruptAgentFlow(
                {
                    "role": "exit",
                    "content": item.content or item.exit_status,
                    "extra": {
                        "exit_status": item.exit_status,
                        "submission": "",
                        "cost": 0.0,
                        "timestamp": time.time(),
                    },
                }
            )

        response = ModelResponse(
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": item},
                    "finish_reason": "stop",
                }
            ],
            model="psrl-rollout",
        )
        message = response.choices[0].message.model_dump()
        message["extra"] = {
            "actions": self._parse_actions(response),
            "cost": 0.0,
            "timestamp": time.time(),
        }
        return message

    def _parse_actions(self, response) -> list[dict]:
        """Support the XML function-calling format used by some SWE models."""
        if self.config.action_regex != _XML_FC_SENTINEL:
            return super()._parse_actions(response)

        from psrl.tools.tool_parser.xml_fc_tool_parser import parse_xml_fc_to_bash

        content = response.choices[0].message.content or ""
        command = parse_xml_fc_to_bash(content)
        if command is not None:
            return [{"command": command}]
        raise FormatError(
            {
                "role": "user",
                "content": self.config.format_error_template,
                "extra": {
                    "interrupt_type": "FormatError",
                    "n_actions": 0,
                    "model_response": content,
                },
            }
        )


@register("mini_swe_agent_deprecated")
class MiniSWEAgentLoop(AgentLoopBase):
    """Run mini-SWE-agent while using PSRL for every model turn."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        rollout_router: ray.actor.ActorHandle | str,
        reward_manager: ray.actor.ActorHandle,
        ps_manager_handle: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        data_config: DictConfigWrap,
        **kwargs,
    ):
        super().__init__(
            trainer_config=trainer_config,
            rollout_router=rollout_router,
            reward_manager=reward_manager,
            ps_manager_handle=ps_manager_handle,
            tokenizer=tokenizer,
            processor=processor,
            dataset_cls=dataset_cls,
            data_config=data_config,
            **kwargs,
        )
        runtime_kwargs = {key: kwargs[key] for key in ("sandbox_config", "agent", "model") if key in kwargs}
        self.runtime_config: MiniSWEAgentRuntimeConfig = build_runtime_config(runtime_kwargs)
        multi_turn = trainer_config.config.gen_actor_rollout_ref.rollout.multi_turn
        if not getattr(multi_turn, "enable", False):
            raise ValueError("mini-SWE-agent requires rollout.multi_turn.enable=True.")
        self.max_turns = int(multi_turn.max_turns)

    def get_generate_fields(self) -> list[str]:
        """Include fields used to select the environment and agent data."""
        fields = super().get_generate_fields()
        fields.extend(["env_class", "data_class", "seed"])
        return fields

    async def run(
        self,
        request: dict,
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Execute one mini-SWE-agent episode."""
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
        agent_data = AgentData.get_agent_data(
            data_class,
            self.config,
            self.reward_manager,
            env,
        )
        if not isinstance(agent_data, MiniSWEAgentData):
            raise TypeError(f"mini-SWE-agent requires MiniSWEAgentData, got {type(agent_data).__name__}.")
        agent_data.reset()
        observation, _ = await env.reset(task=request, seed=request.get("seed"))
        agent_data.init_trajectory(request)

        run_slot: tuple[int, int] | None = None
        response_queue: queue.Queue | None = None
        agent_future: asyncio.Future | None = None
        try:
            runtime_config = observation["runtime_config"]
            sandbox_config = runtime_config.sandbox_config
            if sandbox_config.max_parallel_tasks_per_worker > 0:
                slot_namespace = os.path.join(
                    str(self.config.trainer.project_name),
                    str(self.config.trainer.experiment_name),
                )
                run_slot = await SlotManager.acquire(
                    sandbox_config.max_parallel_tasks_per_worker,
                    slot_namespace,
                    prefix="psrl_mini_swe_agent_slots",
                )

            docker_env_kwargs = self._build_docker_env_kwargs(observation, runtime_config)
            agent_kwargs = {
                "system_template": runtime_config.agent.system_template,
                "instance_template": runtime_config.agent.problem_template,
                "step_limit": self.max_turns,
                "cost_limit": runtime_config.agent.cost_limit,
                "output_path": None,
            }
            request_queue: queue.Queue = queue.Queue()
            response_queue = queue.Queue()
            model = _PSRLModel(
                request_queue,
                response_queue,
                query_timeout=sandbox_config.query_timeout,
                model_name="psrl/rollout",
                cost_tracking="ignore_errors",
                action_regex=runtime_config.model.action_regex,
                observation_template=runtime_config.model.observation_template,
                format_error_template=runtime_config.model.format_error_template,
            )

            event_loop = asyncio.get_running_loop()
            agent_future = event_loop.run_in_executor(
                _AGENT_THREAD_POOL,
                self._run_agent_sync,
                model,
                docker_env_kwargs,
                agent_kwargs,
                observation.get("problem_statement", ""),
            )
            num_turns, loop_exit_status = await self._generation_loop(
                agent_future=agent_future,
                agent_data=agent_data,
                request=request,
                request_queue=request_queue,
                response_queue=response_queue,
                container_timeout=sandbox_config.environment.container_timeout,
                rollout_turn_timeout=sandbox_config.rollout_turn_timeout,
            )

            result = {}
            try:
                result = await asyncio.wait_for(agent_future, timeout=60.0)
            except asyncio.TimeoutError:
                loop_exit_status = loop_exit_status or "AgentThreadTimeout"

            patch = result.get("submission", "") if isinstance(result, dict) else ""
            thread_exit_status = result.get("exit_status", "") if isinstance(result, dict) else ""
            terminate_reason = _classify_termination(
                loop_exit_status or thread_exit_status,
                num_turns,
                self.max_turns,
            )
            if terminate_reason.is_aborted or terminate_reason.needs_worker_retry():
                return None, terminate_reason

            agent_data.set_patch(patch or None)
            grader_result = await self._grade_patch(
                observation=observation,
                patch=patch,
                runtime_config=runtime_config,
            )
            if grader_result is not None:
                agent_data.set_grader_result(grader_result)

            finalized = await agent_data.finalize_output()
            if finalized is None:
                return None, TerminateReason.ABORTED
            return finalized, terminate_reason
        finally:
            if response_queue is not None and agent_future is not None and not agent_future.done():
                response_queue.put(_TerminateSignal("AgentLoopCancelled"))
            await env.close()
            SlotManager.release(run_slot)

    @staticmethod
    def _build_docker_env_kwargs(
        observation: dict,
        runtime_config: MiniSWEAgentRuntimeConfig,
    ) -> dict:
        """Build per-episode `DockerEnvironment` arguments."""
        environment = runtime_config.sandbox_config.environment
        use_preexisting_repo = observation.get("use_preexisting_repo", True)
        preexisting_repo_name = observation.get("preexisting_repo_name", "")
        effective_cwd = environment.cwd
        if use_preexisting_repo and preexisting_repo_name:
            effective_cwd = f"/{preexisting_repo_name}"

        run_args = list(environment.run_args)
        swe_task_id = observation.get("swe_task_id", "")
        if swe_task_id:
            run_args.extend(["--label", f"psrl.swe_task_id={swe_task_id}"])
        actor_id = os.environ.get("PSRL_ACTOR_ID", "")
        if actor_id:
            run_args.extend(["--label", f"psrl.actor_id={actor_id}"])
        if not use_preexisting_repo and observation.get("repo_path"):
            run_args.extend(["--volume", f"{observation['repo_path']}:/testbed"])

        return {
            "image": environment.image,
            "cwd": effective_cwd,
            "env": environment.env if isinstance(environment.env, dict) else {},
            "forward_env": _PROXY_ENV_KEYS,
            "run_args": run_args,
            "container_timeout": environment.container_timeout,
        }

    @staticmethod
    def _run_agent_sync(
        model: _PSRLModel,
        docker_env_kwargs: dict,
        agent_kwargs: dict,
        task: str,
    ) -> dict:
        """Run mini-SWE-agent synchronously in a dedicated worker thread."""
        docker_env = None
        try:
            docker_env = DockerEnvironment(**docker_env_kwargs)
            cwd = docker_env_kwargs.get("cwd", "")
            if cwd:
                check = docker_env.execute(
                    {"command": f"test -d {cwd} && echo EXISTS || echo MISSING"},
                    cwd="/",
                )
                if "MISSING" in check.get("output", ""):
                    raise RuntimeError(f"Working directory {cwd!r} does not exist in the Docker container.")
            agent = DefaultAgent(model, docker_env, **agent_kwargs)
            return agent.run(task)
        except Exception as exc:
            return {
                "exit_status": "error",
                "submission": "",
                "error": str(exc),
            }
        finally:
            if docker_env is not None:
                docker_env.cleanup()

    async def _generation_loop(
        self,
        agent_future: asyncio.Future,
        agent_data: MiniSWEAgentData,
        request: dict,
        request_queue: queue.Queue,
        response_queue: queue.Queue,
        container_timeout: str,
        rollout_turn_timeout: int,
    ) -> tuple[int, str]:
        """Consume synchronous model requests and generate through PSRL."""
        num_turns = 0
        exit_status = ""
        episode_timeout = _parse_timeout_secs(container_timeout) * 0.8
        episode_start = time.monotonic()
        context_limit = self._effective_context_limit()

        while not agent_future.done():
            if time.monotonic() - episode_start > episode_timeout:
                exit_status = "EpisodeTimeout"
                response_queue.put(_TerminateSignal(exit_status))
                break

            try:
                messages = request_queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(_QUEUE_POLL_INTERVAL)
                continue

            normalized = normalize_openai_messages(messages)
            observation = normalized if num_turns == 0 else [normalized[-1]]
            overlong = await agent_data.update_from_env(observation, 0, False, {})
            trajectory = agent_data.session_data.trajectories[-1]
            token_count = len(trajectory.prompt_ids) + len(trajectory.response_ids)
            if overlong or (context_limit and context_limit - token_count < _MIN_GEN_TOKENS):
                exit_status = "PromptBudgetExhausted"
                response_queue.put(_TerminateSignal(exit_status))
                break

            try:
                output = await asyncio.wait_for(
                    self.generate_sequence(
                        agent_data.prepare_generation_request(request),
                        is_sticky_session=self.config.psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky,
                    ),
                    timeout=rollout_turn_timeout,
                )
            except asyncio.TimeoutError:
                exit_status = "RolloutError"
                response_queue.put(_TerminateSignal(exit_status))
                break
            except Exception:
                response_queue.put(_TerminateSignal("RolloutError"))
                raise

            if output is None:
                exit_status = "RolloutReturnedNone"
                response_queue.put(_TerminateSignal(exit_status))
                break

            _, overlong = await agent_data.update_from_model_token_ids(output)
            response_text = agent_data.get_current_step().model_response
            num_turns += 1
            response_queue.put(response_text)

            if overlong:
                exit_status = "PromptBudgetExhausted"
                response_queue.put(_TerminateSignal(exit_status))
                break
            if num_turns >= self.max_turns:
                exit_status = "MaxTurnsExceeded"
                response_queue.put(_TerminateSignal(exit_status))
                break

        return num_turns, exit_status

    def _effective_context_limit(self) -> int:
        """Return the strictest configured model context limit."""
        prompt_length = int(getattr(self.rollout_config, "prompt_length", 0) or 0)
        response_length = int(getattr(self.rollout_config, "response_length", 0) or 0)
        max_model_len = int(getattr(self.rollout_config, "max_model_len", 0) or 0)
        rollout_limit = prompt_length + response_length if prompt_length else 0
        limits = [limit for limit in (rollout_limit, max_model_len) if limit > 0]
        return min(limits) if limits else 0

    async def _grade_patch(
        self,
        observation: dict,
        patch: str,
        runtime_config: MiniSWEAgentRuntimeConfig,
    ) -> dict | None:
        """Run optional fresh-container grading without affecting rollout collection."""
        grader = str(observation.get("swe_grader", "") or "")
        if not patch or grader != "swebench_fresh_container":
            return None

        swe_problem = observation.get("swe_problem", {})
        swe_image = str(observation.get("swe_problem_image", "") or "")
        if not swe_problem or not swe_image:
            return self._grader_failure(swe_problem, "missing_grader_input")

        try:
            from examples.mini_swe.swebench_grader import grade_fresh_container
        except ImportError as exc:
            return self._grader_failure(swe_problem, f"grader_unavailable: {exc}")

        grader_kind = (
            "smith"
            if observation.get("swe_restore_tests", False)
            else "gym"
            if isinstance(swe_problem, dict) and swe_problem.get("eval_script")
            else "verified"
        )
        try:
            return await asyncio.get_running_loop().run_in_executor(
                _GRADER_THREAD_POOL,
                grade_fresh_container,
                swe_problem,
                patch,
                grader_kind,
                swe_image,
                900,
                observation.get("swe_task_id", ""),
                runtime_config.sandbox_config.environment.grader_memory,
            )
        except Exception as exc:
            return self._grader_failure(swe_problem, str(exc))

    @staticmethod
    def _grader_failure(swe_problem: dict, error: str) -> dict:
        """Build a stable failed-grading result."""
        return {
            "policy_violated": False,
            "policy_reasons": [],
            "resolved": False,
            "apply_ok": False,
            "f2p_pass": 0,
            "f2p_total": len(swe_problem.get("FAIL_TO_PASS", [])),
            "p2p_pass": 0,
            "p2p_total": len(swe_problem.get("PASS_TO_PASS", [])),
            "timeout": False,
            "error": error,
            "elapsed_s": 0.0,
            "output_tail": "",
            "resolved_by": "grader_error",
        }
