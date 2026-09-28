"""
Harbor Job execution wrapper for SciBuddy rollout.

Runs one Harbor episode (agent container + verifier) and returns the shaped
reward from the verifier. The model endpoint is pointed at PSRL's SessionRouter
so TITO captures all tokens when using smg_local, or at a plain completion URL
when using openai_api.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from examples.scibuddy_rollout.config import SciBuddyRuntimeConfig
from harbor.job import Job
from harbor.models.job.config import AgentConfig, JobConfig
from harbor.models.trial.config import TaskConfig
from psrl.utils.agent.thinking import MULTI_TRAJ, harness_extra_body
from psrl.utils.common.docker_utils import (
    CLEANUP_EXECUTOR,
    force_remove_compose_images,
    force_remove_compose_project,
    prune_dangling_images,
)

psrl_logger = logging.getLogger("psrl.scibuddy_rollout.runner")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_CONTEXT_SAFETY_MARGIN = 8192


async def _in_cleanup_pool(fn, *args):
    """Run a blocking Docker call on the dedicated cleanup pool."""
    return await asyncio.get_running_loop().run_in_executor(CLEANUP_EXECUTOR, fn, *args)


@dataclass
class HarborEpisodeResult:
    """Result of one Harbor episode."""

    task_name: str
    reward: float
    rewards: dict = field(default_factory=dict)
    exception: str | None = None
    exception_type: str | None = None


async def run_harbor_episode(
    task_path: str,
    model_base_url: str,
    model_name: str,
    config: SciBuddyRuntimeConfig,
    session_id: str = "",
    max_model_len: int = 40960,
    max_turns: int | None = None,
    actor_id: str = "",
    thinking_template: str = MULTI_TRAJ,
) -> HarborEpisodeResult:
    """Run one Harbor episode and return the verifier reward.

    Args:
        task_path: Path to the Harbor task directory (containing task.toml).
        model_base_url: Session URL or plain API base for the model endpoint.
        model_name: Model name sent to the agent.
        config: Runtime config with Harbor and timeout settings.
        session_id: TITO session ID, used as the job directory name.
        max_model_len: vLLM context window size forwarded to terminus-2.
        max_turns: Hard cap on agent turns.
        actor_id: Ray actor ID label attached to the Compose service.
        thinking_template: Chain-of-thought policy. See psrl/utils/agent/thinking.py.

    Returns:
        HarborEpisodeResult with the verifier reward.
    """
    extra_body = harness_extra_body(thinking_template)

    main_override: dict = {}
    if actor_id:
        main_override["labels"] = [f"psrl.actor_id={actor_id}"]
    if config.harbor.memory_mb_override > 0:
        main_override["mem_limit"] = f"{config.harbor.memory_mb_override}m"

    env_kwargs: dict = {}
    if main_override:
        import tempfile

        import yaml

        override_file = Path(tempfile.mktemp(suffix=".yaml", prefix="harbor-override-"))
        override_file.write_text(yaml.dump({"services": {"main": main_override}}))
        env_kwargs["extra_docker_compose"] = [override_file]

    job_name = session_id or uuid.uuid4().hex[:12]
    job_dir = Path(config.harbor.jobs_dir) / job_name
    job_config = JobConfig(
        jobs_dir=job_dir,
        tasks=[TaskConfig(path=task_path)],
        agents=[
            AgentConfig(
                name=config.harbor.agent_name,
                model_name=f"openai/{model_name}",
                env={"OPENAI_API_KEY": "EMPTY"},
                override_timeout_sec=config.task_timeout_sec,
                kwargs={
                    "api_base": model_base_url,
                    "enable_summarize": False,
                    "collect_rollout_details": True,
                    "record_terminal_session": False,
                    "temperature": 1.0,
                    **({"max_turns": max_turns} if max_turns else {}),
                    "suppress_max_turns_warning": True,
                    "model_info": {
                        "max_input_tokens": max(1024, max_model_len - _CONTEXT_SAFETY_MARGIN),
                        "max_output_tokens": max_model_len,
                        "input_cost_per_token": 0.0,
                        "output_cost_per_token": 0.0,
                    },
                    "llm_kwargs": {
                        "timeout": 900,
                        "max_retries": 0,
                        **({"extra_body": extra_body} if extra_body else {}),
                    },
                },
            )
        ],
        n_attempts=1,
        n_concurrent_trials=1,
        **({"environment": env_kwargs} if env_kwargs else {}),
    )

    timeout = config.task_timeout_sec + config.verifier_timeout_sec + 600.0

    try:
        try:
            job = await Job.create(job_config)
            result = await asyncio.wait_for(job.run(), timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            psrl_logger.warning("Harbor job failed before returning a trial: %s.", exc, exc_info=True)
            return HarborEpisodeResult(
                task_name="unknown",
                reward=0.0,
                exception=str(exc) or type(exc).__name__,
                exception_type=type(exc).__name__,
            )

        if not result.trial_results:
            return HarborEpisodeResult(
                task_name="unknown",
                reward=0.0,
                exception="no_trials",
                exception_type="NoTrialsError",
            )

        tr = result.trial_results[0]
        rewards = dict(tr.verifier_result.rewards) if tr.verifier_result and tr.verifier_result.rewards else {}
        exception = tr.exception_info.exception_message if tr.exception_info else None
        exception_type = tr.exception_info.exception_type if tr.exception_info else None

        return HarborEpisodeResult(
            task_name=tr.task_name,
            reward=float(rewards.get("reward", 0.0)),
            rewards=rewards,
            exception=exception,
            exception_type=exception_type,
        )
    finally:
        removed = await _in_cleanup_pool(force_remove_compose_project, job_name)
        if removed:
            psrl_logger.info(
                "Reclaimed %d container(s) for episode %s after the job returned.",
                len(removed),
                job_name,
            )
        await _in_cleanup_pool(force_remove_compose_images, job_name)
        await _in_cleanup_pool(prune_dangling_images)
