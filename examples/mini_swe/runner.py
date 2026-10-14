"""Run one mini-SWE-agent task through its standard Python bindings."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

psrl_logger = logging.getLogger(__name__)
logging.getLogger("minisweagent.environment").setLevel(logging.WARNING)


def _silence_litellm() -> None:
    """Suppress LiteLLM's BadRequestError banner (print, not logging).

    LiteLLM's exception mapper always ``print``s
    ``Give Feedback / Get Help`` + ``LiteLLM.Info: ...`` unless
    ``litellm.suppress_debug_info`` is True. Logger level tweaks do nothing.
    """
    try:
        import litellm

        litellm.suppress_debug_info = True
    except Exception:
        pass
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)


_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
]


def _build_environment_config(payload: dict[str, Any]) -> dict[str, Any]:
    environment = dict(payload["runtime_config"]["sandbox_config"]["environment"])
    observation = payload["observation"]
    use_preexisting_repo = observation.get("use_preexisting_repo", True)
    preexisting_repo_name = observation.get("preexisting_repo_name", "")
    if use_preexisting_repo and preexisting_repo_name:
        environment["cwd"] = f"/{preexisting_repo_name}"

    run_args = list(environment.get("run_args", []))
    swe_task_id = observation.get("swe_task_id", "")
    if swe_task_id:
        run_args.extend(["--label", f"psrl.swe_task_id={swe_task_id}"])
    actor_id = payload.get("actor_id", "")
    if actor_id:
        run_args.extend(["--label", f"psrl.actor_id={actor_id}"])
    if not use_preexisting_repo and observation.get("repo_path"):
        run_args.extend(["--volume", f"{observation['repo_path']}:/testbed"])

    environment.update(
        {
            "environment_class": environment.get("environment_class", "docker"),
            "forward_env": _PROXY_ENV_KEYS,
            "run_args": run_args,
        }
    )
    environment.pop("grader_memory", None)
    return environment


def _build_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(payload["runtime_config"]["model"])
    model_kwargs = dict(payload["sampling_params"])
    top_k = model_kwargs.pop("top_k", None)
    if top_k is not None and int(top_k) >= 0:
        model_kwargs["extra_body"] = {"top_k": int(top_k)}
    extra_headers = dict(model_kwargs.get("extra_headers") or {})
    if payload.get("trajectory_id_strategy", "manual") == "manual":
        extra_headers["x-smg-tito-trajectory-id"] = "0"
    else:
        extra_headers.pop("x-smg-tito-trajectory-id", None)
    if extra_headers:
        model_kwargs["extra_headers"] = extra_headers
    else:
        model_kwargs.pop("extra_headers", None)
    model_kwargs.update(
        {
            "api_base": payload["base_url"],
            "api_key": "EMPTY",
            "timeout": payload["runtime_config"]["sandbox_config"]["rollout_turn_timeout"],
        }
    )
    model_config.update(
        {
            "model_name": payload["model"],
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        }
    )
    return model_config


def _grader_failure(swe_problem: dict[str, Any], error: str) -> dict[str, Any]:
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


def _grade_patch(payload: dict[str, Any], patch: str) -> dict[str, Any] | None:
    observation = payload["observation"]
    if not patch or observation.get("swe_grader") != "swebench_fresh_container":
        return None

    swe_problem = observation.get("swe_problem", {})
    swe_image = str(observation.get("swe_problem_image", "") or "")
    if not swe_problem or not swe_image:
        return _grader_failure(swe_problem, "missing_grader_input")

    try:
        from examples.mini_swe.swebench_grader import grade_fresh_container

        grader_kind = (
            "smith"
            if observation.get("swe_restore_tests", False)
            else "gym"
            if swe_problem.get("eval_script")
            else "verified"
        )
        grader_memory = payload["runtime_config"]["sandbox_config"]["environment"].get("grader_memory", "")
        return grade_fresh_container(
            swe_problem,
            patch,
            grader_kind,
            swe_image,
            900,
            observation.get("swe_task_id", ""),
            grader_memory,
        )
    except Exception as exc:
        return _grader_failure(swe_problem, str(exc))


def run_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one task through mini-SWE-agent's standard Python bindings."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model
    from psrl.utils.agent.overflow import PromptOverflowError, ensure_overflow_handling

    _silence_litellm()

    class _TimedAgent(DefaultAgent):
        """DefaultAgent that accumulates wall-clock model vs environment time.

        ``step()`` decomposes into ``query()`` (model/assistant turn) and
        ``execute_actions()`` (environment/tool execution), so timing each gives a
        clean assistant-vs-env split for the trajectory summary.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.assistant_s = 0.0
            self.env_s = 0.0

        def query(self, *args: Any, **kwargs: Any) -> Any:
            t = time.time()
            try:
                return super().query(*args, **kwargs)
            finally:
                self.assistant_s += time.time() - t

        def execute_actions(self, *args: Any, **kwargs: Any) -> Any:
            t = time.time()
            try:
                return super().execute_actions(*args, **kwargs)
            finally:
                self.env_s += time.time() - t

    environment = None
    agent = None
    run_start = time.time()
    # Return dictionaries share `timing`, so the finalizer records elapsed values
    # for success and exception paths.
    timing: dict[str, float] = {"prep_s": 0.0, "assistant_s": 0.0, "env_s": 0.0, "grading_s": 0.0, "elapsed_s": 0.0}
    try:
        environment = get_environment(_build_environment_config(payload))
        model = get_model(config=_build_model_config(payload))
        ensure_overflow_handling(model)
        agent_config = dict(payload["runtime_config"]["agent"])
        agent_config["instance_template"] = agent_config.pop("problem_template")
        agent_config["step_limit"] = payload["max_turns"]
        agent_config["output_path"] = None
        timing["prep_s"] = time.time() - run_start
        agent = _TimedAgent(model, environment, **agent_config)
        result = agent.run(payload["task"])
        patch = result.get("submission", "") or ""
        try:
            environment.cleanup()
            environment = None
        except Exception:
            psrl_logger.warning("mini-SWE-agent environment cleanup before grading failed.", exc_info=True)
        grade_start = time.time()
        grader_result = _grade_patch(payload, patch)
        timing["grading_s"] = time.time() - grade_start
        return {
            "exit_status": result.get("exit_status", ""),
            "submission": patch,
            "grader_result": grader_result,
            "timing": timing,
        }
    except PromptOverflowError as exc:
        # Preserve turns generated before context overflow and terminate normally.
        psrl_logger.warning("mini-SWE-agent stopped on context overflow: %s.", exc)
        return {
            "exit_status": "context_exceeded",
            "submission": "",
            "grader_result": None,
            "timing": timing,
        }
    except Exception as exc:
        psrl_logger.warning("mini-SWE-agent task failed: %s.", exc, exc_info=True)
        return {
            "exit_status": "error",
            "submission": "",
            "grader_result": None,
            "error": str(exc),
            "timing": timing,
        }
    finally:
        # Shared return dictionaries receive final timing before control leaves.
        if agent is not None:
            timing["assistant_s"] = agent.assistant_s
            timing["env_s"] = agent.env_s
        timing["elapsed_s"] = time.time() - run_start
        if environment is not None:
            try:
                environment.cleanup()
            except Exception:
                psrl_logger.warning("mini-SWE-agent environment cleanup failed.", exc_info=True)
