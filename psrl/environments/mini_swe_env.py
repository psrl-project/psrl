"""
mini-SWE-agent Environment for PSRL.

This environment adapts the mini-SWE-agent integration to PSRL's `Environment`
interface. It handles:
- `reset()`: Parse task metadata, build per-instance config, create workspace.
- `close()`: Clean up temporary directories and safety-net Docker container cleanup.

It does NOT use `step()` because mini-swe-agent manages its own tool execution
loop internally via `DefaultAgent.run()`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

import numpy as np
from examples.mini_swe.config import (
    MiniSWEAgentRuntimeConfig,
    apply_data_overrides,
    build_runtime_config,
)
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

from psrl.environments.base import Environment, EnvStepOutput
from psrl.utils.common.docker_utils import force_remove_containers_by_label
from psrl.workers.reward.client import RewardManagerClient

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _grading_metadata(
    data_source: str,
    extra_info: dict,
) -> dict:
    """
    Require harness inputs only for datasets graded by test execution.
    """
    if data_source == "mini_swe_agent_simple":
        return {}
    if data_source not in ("swebench_verified", "swe_smith_py", "swe_gym"):
        raise ValueError(f"Unsupported mini-SWE data source: {data_source!r}.")

    swe_grader = extra_info.get("swe_grader")
    if swe_grader != "swebench_fresh_container":
        raise ValueError(f"Expected swe_grader='swebench_fresh_container' for {data_source!r}, got {swe_grader!r}.")
    swe_problem = extra_info.get("swe_problem")
    if not isinstance(swe_problem, dict) or not swe_problem.get("instance_id"):
        raise ValueError(f"Expected swe_problem with an instance_id for {data_source!r}, got {swe_problem!r}.")
    swe_problem_image = extra_info.get("swe_problem_image")
    if not isinstance(swe_problem_image, str) or not swe_problem_image:
        raise ValueError(f"Expected swe_problem_image for {data_source!r}, got {swe_problem_image!r}.")

    return {
        "swe_grader": swe_grader,
        "swe_problem": swe_problem,
        "swe_problem_image": swe_problem_image,
        "swe_restore_tests": bool(extra_info.get("swe_restore_tests", False)),
    }


@Environment.register("mini_swe_env")
class MiniSWEEnvironment(Environment[dict, None]):
    """
    Environment adapter for mini-SWE-agent in-process episodes.

    Prepares per-episode workspaces, applies per-instance config overrides,
    and provides safety-net Docker cleanup. Docker container lifecycle is
    primarily managed by mini-swe-agent's `DockerEnvironment` internally.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: RewardManagerClient,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor | None = None,
        dataset_cls=None,
        runtime_config: MiniSWEAgentRuntimeConfig | None = None,
    ):
        """
        Initialize the mini-SWE-agent environment.

        Args:
            config (DictConfig): PSRL trainer configuration.
            reward_manager (RewardManagerClient): Shared client for submitting reward requests.
            runtime_config (MiniSWEAgentRuntimeConfig | None): Pre-built runtime config
                from the agent loop. If None, a default config is built from scratch.
        """
        super().__init__(config, reward_manager, tokenizer, processor, dataset_cls)
        self._base_runtime_config = runtime_config
        self._swe_task_id: str = ""
        self._runtime_config: MiniSWEAgentRuntimeConfig | None = None

    async def reset(self, task: dict, **kwargs) -> tuple[dict, dict]:
        """
        Parse task data and prepare per-episode workspace.

        Args:
            task: Single request dict containing the SWE task metadata.

        Returns:
            Tuple of (observation_dict, info_dict).
        """
        self.task = task

        # Log available keys for debugging data pipeline issues.
        task_keys = list(task.keys()) if isinstance(task, dict) else []
        psrl_logger.debug(f"MiniSWEEnvironment reset: task keys={task_keys}.")

        # The reward manager retains `extra_info` if the data pipeline omits it.
        extra_info_raw = task.get("extra_info", {}) if isinstance(task, dict) else {}
        if isinstance(extra_info_raw, str):
            try:
                extra_info = json.loads(extra_info_raw)
            except json.JSONDecodeError:
                extra_info = {}
        elif isinstance(extra_info_raw, dict):
            extra_info = extra_info_raw
        else:
            extra_info = {}

        grading_metadata = _grading_metadata(task.get("data_source"), extra_info)

        # Fall back to raw_prompt for problem_statement if extra_info is empty.
        problem_statement = extra_info.get("problem_statement", "") or ""
        if not problem_statement:
            raw_prompt = task.get("raw_prompt", None) if isinstance(task, dict) else None
            if raw_prompt is not None:
                if isinstance(raw_prompt, (list, np.ndarray)) and len(raw_prompt) > 0:
                    first_msg = raw_prompt[0] if isinstance(raw_prompt[0], dict) else {}
                    problem_statement = str(first_msg.get("content", ""))
                elif isinstance(raw_prompt, str):
                    problem_statement = raw_prompt

        repo_path = extra_info.get("repo_path", None)
        swe_problem_id = str(extra_info.get("swe_problem_id", "") or "")
        sandbox_overrides = extra_info.get("sandbox_overrides", {}) or {}

        # Build runtime config from pre-built base or defaults.
        base_config = self._base_runtime_config or build_runtime_config({})
        self._runtime_config = apply_data_overrides(base_config, extra_info)

        # A fresh ID prevents concurrent rollouts for one `swe_problem_id` from sharing Docker cleanup labels.
        task_uuid = f"{uuid.uuid4().hex[:12]}-{int(time.time())}"
        self._swe_task_id = task_uuid
        if swe_problem_id:
            # Append the problem ID for easier log tracing, but keep the UUID
            # as the uniqueness guarantee.
            self._swe_task_id = f"{task_uuid}-{swe_problem_id[:24]}"

        # Determine repo type and preexisting repo name.
        use_preexisting_repo = bool(sandbox_overrides.get("use_preexisting_repo", False))
        preexisting_repo_name = str(sandbox_overrides.get("preexisting_repo_name", "") or "")
        if not use_preexisting_repo and not repo_path:
            use_preexisting_repo = True

        observation = {
            "problem_statement": problem_statement,
            "swe_task_id": self._swe_task_id,
            "runtime_config": self._runtime_config,
            "repo_path": repo_path,
            "use_preexisting_repo": use_preexisting_repo,
            "preexisting_repo_name": preexisting_repo_name,
            **grading_metadata,
        }

        _problem_short = f"{problem_statement[:80]}..." if len(problem_statement) > 80 else problem_statement
        psrl_logger.debug(
            f"[mini-SWE-agent, task_id={self._swe_task_id}] MiniSWEEnvironment reset: "
            f"problem={_problem_short!r}, "
            f"repo={preexisting_repo_name!r}, preexisting={use_preexisting_repo}."
        )

        return observation, {}

    async def step(self, action: None) -> EnvStepOutput:
        """
        Not used. mini-SWE-agent manages its own step loop internally.
        """
        raise NotImplementedError(
            "MiniSWEEnvironment does not support step(). "
            "mini-SWE-agent manages its own tool execution loop in-process."
        )

    async def close(self) -> None:
        """
        Safety-net cleanup for Docker containers and temporary directories.

        Primary Docker cleanup is handled by mini-swe-agent's
        `DockerEnvironment.cleanup()`. This method provides a fallback
        via label-based container cleanup in case the primary path fails.

        Uses ``docker rm -f`` (not ``docker stop``) because rollout sandboxes
        are disposable and ``docker stop`` has been observed to silently
        succeed without actually killing containers that have many runaway
        in-container children or active ``docker exec`` sessions racing the
        SIGTERM/SIGKILL window.
        """
        if self._swe_task_id:
            await asyncio.to_thread(
                force_remove_containers_by_label,
                "psrl.swe_task_id",
                self._swe_task_id,
            )

        psrl_logger.debug(f"[mini-SWE-agent, task_id={self._swe_task_id}] MiniSWEEnvironment closed.")

    @property
    def state(self) -> dict:
        """
        Return current environment state for debugging.
        """
        return {
            "swe_task_id": self._swe_task_id,
            "runtime_config": self._runtime_config,
        }
