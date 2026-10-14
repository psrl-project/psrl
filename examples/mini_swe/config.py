"""
mini-SWE-agent Runtime Configuration for PSRL.

Dataclass-based config for the mini-SWE-agent integration. These configs
control the PSRL-side orchestration (sandbox timeouts, parallelism, templates).
mini-swe-agent's own components (`DockerEnvironment`, `DefaultAgent`,
`LitellmTextbasedModel`) are configured directly via their Python APIs
in the black-box runner -- no YAML generation is needed.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# --- Structured configuration ---


@dataclass
class MiniEnvironmentConfig:
    """
    Environment settings passed to mini-swe-agent's `get_environment()`.
    """

    environment_class: str = "docker"
    image: str = "python:3.11-slim"
    cwd: str = "/testbed"
    env: dict = field(
        default_factory=lambda: {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
        }
    )
    run_args: list = field(
        default_factory=lambda: [
            "--rm",
            "--memory=8g",
            "--network",
            "host",
            "--add-host",
            "host.docker.internal:host-gateway",
        ]
    )
    container_timeout: str = "2h"
    # Installing heavy repositories can briefly exceed rollout memory during grading.
    # A separate 30 GB limit prevents cgroup OOM failures.
    grader_memory: str = "30g"


@dataclass
class MiniSandboxConfig:
    """
    PSRL-side orchestration settings (not passed to mini-swe-agent directly).
    """

    max_parallel_tasks_per_worker: int = 0
    environment: MiniEnvironmentConfig = field(default_factory=MiniEnvironmentConfig)

    # Per-turn timeout forwarded to mini-swe-agent's model client.
    rollout_turn_timeout: int = 480

    # Legacy queue-bridge loop compatibility.
    query_timeout: int = 600


@dataclass
class MiniAgentConfig:
    """
    Agent settings passed to `DefaultAgent(**kwargs)`.

    `system_template` and `problem_template` are required by mini-swe-agent's
    `AgentConfig` (no defaults in upstream). `problem_template` maps to
    mini-swe-agent's `instance_template` kwarg.

    Both templates must come from `simple_agent_config.yaml`. Empty defaults make
    `build_runtime_config` reject incomplete merged configurations.
    """

    cost_limit: float = 0.0
    system_template: str = ""
    problem_template: str = ""


@dataclass
class MiniModelConfig:
    """
    Model format settings forwarded to mini-swe-agent's `get_model()` during
    session-routed training and standalone eval.

    Defaults match `LitellmTextbasedModelConfig` class-level defaults so that
    omitting this section from a YAML config is fully backwards-compatible.

    Fields:
        model_class: mini-swe-agent model implementation name or import path.
        action_regex: Regex applied to the model's text output to extract the
            shell command. Change this to switch action formats, e.g.
            ``"```bash\\\\s*\\\\n(.*?)\\\\n```"`` for a plain bash block.
        observation_template: Jinja2 template that formats Docker command output
            into the next user message. Training and eval must use the same
            template or the model sees unfamiliar observation formatting.
        format_error_template: Message sent back to the model when
            ``action_regex`` finds 0 or >1 matches.
    """

    model_class: str = "litellm_textbased"
    action_regex: str = r"```mswea_bash_command\s*\n(.*?)\n```"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    format_error_template: str = (
        "Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions."
    )


@dataclass
class MiniSWEAgentRuntimeConfig:
    """
    Top-level config for the mini-SWE-agent PSRL integration.
    """

    sandbox_config: MiniSandboxConfig = field(default_factory=MiniSandboxConfig)
    agent: MiniAgentConfig = field(default_factory=MiniAgentConfig)
    model: MiniModelConfig = field(default_factory=MiniModelConfig)


# --- Helpers ---


def _ensure_dict(val: Any) -> dict:
    """
    Coerce `val` to a plain dict, handling JSON strings and `OmegaConf`.
    """
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(val, DictConfig):
        result = OmegaConf.to_container(val, resolve=True)
        return result if isinstance(result, dict) else {}
    if isinstance(val, dict):
        return val
    return {}


# --- Factory ---


def build_runtime_config(yaml_kwargs: dict[str, Any]) -> MiniSWEAgentRuntimeConfig:
    """
    Build config by merging YAML kwargs onto the `OmegaConf` structured schema.
    """
    raw = OmegaConf.to_container(OmegaConf.create(yaml_kwargs), resolve=True)
    if not isinstance(raw, dict):
        raw = {}
    raw.pop("name", None)
    raw.pop("_target_", None)

    schema = OmegaConf.structured(MiniSWEAgentRuntimeConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create(raw))
    cfg: MiniSWEAgentRuntimeConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

    if not cfg.agent.system_template or not cfg.agent.problem_template:
        raise ValueError(
            "agent.system_template and agent.problem_template must be provided "
            "in simple_agent_config.yaml (they have no hardcoded defaults)."
        )

    return cfg


# --- Per-problem overrides ---

_SANDBOX_FIELDS = frozenset(MiniSandboxConfig.__dataclass_fields__)
_AGENT_OVERRIDE_FIELDS = frozenset(("cost_limit", "system_template", "problem_template"))


def apply_data_overrides(
    base: MiniSWEAgentRuntimeConfig,
    extra_info: dict[str, Any],
) -> MiniSWEAgentRuntimeConfig:
    """
    Per-SWE-problem copy of `base` with data-affine overrides. `base` is not mutated.
    """
    sandbox_ov = _ensure_dict(extra_info.get("sandbox_overrides", {}))
    agent_ov = _ensure_dict(extra_info.get("agent_overrides", {}))
    if not sandbox_ov and not agent_ov:
        return base

    patch: dict[str, Any] = {}
    if sandbox_ov:
        sandbox_patch = {k: v for k, v in sandbox_ov.items() if k in _SANDBOX_FIELDS}
        if sandbox_patch:
            patch["sandbox_config"] = sandbox_patch
    if agent_ov:
        agent_patch = {k: v for k, v in agent_ov.items() if k in _AGENT_OVERRIDE_FIELDS}
        if agent_patch:
            patch["agent"] = agent_patch

    if not patch:
        return base

    base_cfg = OmegaConf.structured(base)
    return OmegaConf.to_object(OmegaConf.merge(base_cfg, OmegaConf.create(patch)))  # type: ignore[return-value]
