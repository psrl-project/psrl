"""
mini-SWE-agent Environment for PSRL.

This environment adapts the mini-SWE-agent integration to PSRL's `Environment`
interface. It handles:
- `reset()`: Parse task metadata and build per-instance config.
- `close()`: Close episode-local state, since sandbox leases own runtime cleanup.

It does NOT use `step()` because mini-swe-agent manages its own tool execution
loop internally via `DefaultAgent.run()`.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid

import numpy as np
import ray
from examples.mini_swe.config import (
    MiniSWEAgentRuntimeConfig,
    apply_data_overrides,
    build_runtime_config,
)
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

from psrl.environments.base import Environment, EnvStepOutput

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@Environment.register("mini_swe_env")
class MiniSWEEnvironment(Environment[dict, None]):
    """
    Environment adapter for mini-SWE-agent in-process episodes.

    Prepares per-episode metadata and applies per-instance config overrides.
    Runtime lifecycle is managed by PSRL's sandbox lease abstraction.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor | None = None,
        dataset_cls=None,
        runtime_config: MiniSWEAgentRuntimeConfig | None = None,
    ):
        """
        Initialize the mini-SWE-agent environment.

        Args:
            config (DictConfig): PSRL trainer configuration.
            reward_manager (ray.actor.ActorHandle): Ray actor handle for computing rewards.
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

        # Prepared datasets may use either supported grader field schema.
        swe_grader = str(extra_info.get("swe_grader", "") or extra_info.get("grader", "") or "")
        swe_problem = extra_info.get("swe_problem", None) or extra_info.get("instance", None) or {}
        swe_problem_image = str(extra_info.get("swe_problem_image", "") or extra_info.get("image_name", "") or "")
        swe_problem_template = str(extra_info.get("swe_problem_template", "") or "")
        swe_restore_tests = bool(
            extra_info.get("swe_restore_tests", False) or extra_info.get("needs_head_minus_one", False)
        )

        psrl_logger.debug(
            f"MiniSWEEnvironment reset: extra_info type={type(extra_info_raw).__name__}, "
            f"extra_info keys={list(extra_info)}, "
            f"swe_grader={swe_grader or 'MISSING'!r}, "
            f"has_swe_problem_image={bool(swe_problem_image)}, "
            f"task_keys={list(task) if isinstance(task, dict) else []}."
        )

        if swe_grader == "swebench_fresh_container" and (
            not isinstance(swe_problem, dict) or not swe_problem.get("instance_id") or not swe_problem_image
        ):
            raise ValueError(
                "Fresh-container grading requires swe_problem with instance_id and swe_problem_image; "
                f"available extra_info keys: {sorted(extra_info.keys())}."
            )
        # Grading is host-independent: the eval script must travel with the row.
        # Re-run the dataset preparation step if a split predates that change.
        if swe_grader == "swebench_fresh_container" and not str(swe_problem.get("eval_script") or "").strip():
            raise ValueError(
                "Fresh-container grading requires swe_problem.eval_script. Re-run the preparation "
                "step for this split (see examples/mini_swe/prepare/README.md)."
            )

        observation = {
            "problem_statement": problem_statement,
            "swe_task_id": self._swe_task_id,
            "runtime_config": self._runtime_config,
            "repo_path": repo_path,
            "use_preexisting_repo": use_preexisting_repo,
            "preexisting_repo_name": preexisting_repo_name,
            "swe_grader": swe_grader,
            "swe_problem": swe_problem,
            "swe_problem_image": swe_problem_image,
            "swe_problem_template": swe_problem_template,
            "swe_restore_tests": swe_restore_tests,
        }

        problem_short = f"{problem_statement[:80]}..." if len(problem_statement) > 80 else problem_statement
        psrl_logger.debug(
            f"[mini-SWE-agent, task_id={self._swe_task_id}] MiniSWEEnvironment reset: "
            f"problem={problem_short!r}, "
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
        Close episode-local environment state.

        Sandbox lifecycle belongs to the lease created by the agent loop, so
        this dataset adapter intentionally performs no backend-specific work.
        """
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
