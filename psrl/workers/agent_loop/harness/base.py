"""Task-scoped lifecycle contract for sandboxed coding harnesses."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from omegaconf import DictConfig, OmegaConf

from psrl.sandbox import ExecResult, SandboxSession
from psrl.workers.agent_loop.harness.runtime import executable_path

# Trajectory output formats the post-rollout integrity scanner can dispatch on.
# `auto` resolves to the kind default below, but the scan dispatch key is always a concrete format.
SUPPORTED_TRAJECTORY_FORMATS = (
    "auto",
    "claude_code_stream_json",
    "codex_jsonl",
    "plain_text",
)
_KIND_DEFAULT_TRAJECTORY_FORMATS: dict[str, str] = {
    "claude_code": "claude_code_stream_json",
    "codex": "codex_jsonl",
}


@dataclass(frozen=True)
class HarnessCompactionConfig:
    """Context-compaction policy shared by external coding harnesses.

    The CLI context capacity is the effective rollout ``max_model_len`` (the
    harness adds no separate window knob). ``compact_percent`` is the share of
    that window at which the harness triggers compaction, and is forwarded
    verbatim as Claude Code's ``CLAUDE_AUTOCOMPACT_PCT_OVERRIDE``.
    """

    enabled: bool = True
    compact_percent: float = 75.0

    def resolve(self, context_window_tokens: int) -> tuple[int, int] | None:
        """Resolve CLI capacity and the absolute compaction trigger."""
        if not self.enabled:
            return None
        window = int(context_window_tokens)
        if window <= 0:
            raise ValueError("Harness compaction context window must be greater than zero.")
        if not 0 < self.compact_percent <= 100:
            raise ValueError(f"Harness compaction compact_percent must be in (0, 100], got {self.compact_percent!r}.")
        trigger = max(1, (window * self.compact_percent) // 100)
        return window, trigger


@dataclass(frozen=True)
class HarnessConfig:
    """Backend-neutral configuration shared by all coding harnesses."""

    kind: str
    executable: str
    model: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    home_dir: str = "/root"
    runtime_mount: str = "/opt/harness"
    time_budget_s: float = 7200.0
    output_tail_chars: int = 16_384
    callback_base_url: str | None = None
    permission_mode: str | None = None
    allowed_permissions: tuple[str, ...] = ()
    system_prompt: str | None = None
    system_prompt_mode: str = "replace"
    setting_sources: str | None = None
    tools: str | None = None
    max_output_tokens: int | None = None
    thinking_enabled: bool = True
    thinking_budget_tokens: int | None = None
    reasoning_effort: str | None = None
    interleaved_thinking: bool = True
    supported_capabilities: tuple[str, ...] = ()
    disable_prompt_caching: bool = False
    disable_experimental_betas: bool = False
    subagents_enabled: bool = True
    compaction: HarnessCompactionConfig = field(default_factory=HarnessCompactionConfig)
    trajectory_format: str = "auto"

    def __post_init__(self) -> None:
        if not self.kind or not self.executable:
            raise ValueError("Harness kind and executable cannot be empty.")
        if not self.runtime_mount.startswith("/"):
            raise ValueError(f"Harness runtime_mount must be an absolute path, got {self.runtime_mount!r}.")
        if self.time_budget_s <= 0 or self.output_tail_chars <= 0:
            raise ValueError("Harness time_budget_s and output_tail_chars must be greater than zero.")
        if self.system_prompt_mode not in ("append", "replace", "none"):
            raise ValueError(f"Unsupported harness system_prompt_mode {self.system_prompt_mode!r}.")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("Harness max_output_tokens must be greater than zero when configured.")
        if self.thinking_budget_tokens is not None and self.thinking_budget_tokens <= 0:
            raise ValueError("Harness thinking_budget_tokens must be greater than zero when configured.")
        if not self.thinking_enabled and self.thinking_budget_tokens is not None:
            raise ValueError("Harness thinking_budget_tokens cannot be set when thinking is disabled.")
        if self.reasoning_effort is not None and not self.reasoning_effort.strip():
            raise ValueError("Harness reasoning_effort cannot be empty when configured.")
        if self.trajectory_format not in SUPPORTED_TRAJECTORY_FORMATS:
            raise ValueError(
                f"Unsupported harness trajectory_format {self.trajectory_format!r}; "
                f"expected one of {SUPPORTED_TRAJECTORY_FORMATS}."
            )

    def resolved_trajectory_format(self) -> str:
        """Return the concrete format the integrity scanner should dispatch on."""
        if self.trajectory_format != "auto":
            return self.trajectory_format
        return _KIND_DEFAULT_TRAJECTORY_FORMATS.get(self.kind, "plain_text")

    @classmethod
    def from_value(cls, value: HarnessConfig | DictConfig | Mapping[str, Any]) -> HarnessConfig:
        """Normalize a Hydra mapping into an immutable harness configuration."""
        if isinstance(value, cls):
            return value
        if isinstance(value, DictConfig):
            raw = OmegaConf.to_container(value, resolve=True)
        else:
            raw = dict(value)
        if not isinstance(raw, Mapping):
            raise TypeError("Harness configuration must be a mapping.")
        normalized = dict(raw)
        normalized["args"] = tuple(str(item) for item in normalized.get("args", ()))
        normalized["allowed_permissions"] = tuple(str(item) for item in normalized.get("allowed_permissions", ()))
        normalized["supported_capabilities"] = tuple(
            str(item) for item in normalized.get("supported_capabilities", ())
        )
        normalized["env"] = {str(key): str(item) for key, item in dict(normalized.get("env", {})).items()}
        compaction = normalized.get("compaction", {})
        normalized["compaction"] = (
            compaction
            if isinstance(compaction, HarnessCompactionConfig)
            else HarnessCompactionConfig(**dict(compaction or {}))
        )
        return cls(**normalized)


@dataclass(frozen=True)
class HarnessRuntime:
    """Per-task values bound after the TITO session and sandbox exist."""

    session_id: str
    session_root_url: str
    workdir: str
    model: str
    context_window_tokens: int | None = None
    compaction_token_limit: int | None = None
    max_turns: int | None = None


@dataclass(frozen=True)
class HarnessResult:
    """Small process result retained outside the disposable sandbox."""

    exit_code: int
    stdout_tail: str = ""
    stderr_tail: str = ""
    command_name: str = ""
    workdir: str = ""
    stdout_path: str = ""
    stderr_path: str = ""

    def diagnostic_text(self) -> str:
        """Return bounded, actionable process diagnostics for rollout failures."""
        return "\n".join(
            (
                f"exit_code={self.exit_code}",
                f"command={self.command_name or '<unknown>'}",
                f"workdir={self.workdir or '<unknown>'}",
                f"stdout_path={self.stdout_path or '<unknown>'}",
                f"stderr_path={self.stderr_path or '<unknown>'}",
                "stdout_tail:",
                self.stdout_tail or "<empty>",
                "stderr_tail:",
                self.stderr_tail or "<empty>",
            )
        )


class Harness(ABC):
    """One task-owned coding harness, including preparation, run, and abort.

    The object deliberately combines adapter and active-run state. It is never
    shared between tasks, so the sandbox lease remains the single lifecycle
    owner while an optional abort cancels the currently active CLI request.
    """

    def __init__(self, config: HarnessConfig, sandbox: SandboxSession) -> None:
        self.config = config
        self.sandbox = sandbox
        self._active_exec: asyncio.Task[ExecResult] | None = None
        self._log_dir = str(PurePosixPath(config.home_dir) / ".psrl-harness")

    def config_dir(self) -> str | None:
        """Harness-specific config directory created during preparation."""
        return None

    async def prepare(self, runtime: HarnessRuntime) -> None:
        """Verify the mounted executable and write harness-specific configuration.

        The executable comes from the read-only runtime tree mounted at
        ``config.runtime_mount``. There is no in-sandbox installation, no
        network fetch and no mutation of the task image's global toolchain. All
        directories are created in the same probe, so per-trajectory setup stays
        to one round trip.
        """
        exe = executable_path(self.config.runtime_mount, self.config.executable)
        directories = [self._log_dir, self.config_dir()]
        mkdir = "mkdir -p " + " ".join(shlex.quote(d) for d in directories if d)
        check = await self.sandbox.exec(f"{mkdir} && {shlex.quote(exe)} --version", timeout_s=60)
        if check.exit_code != 0:
            raise RuntimeError(
                f"Harness executable {exe!r} is unavailable in the sandbox (exit {check.exit_code}). "
                f"Mount the {self.config.kind!r} runtime tree at {self.config.runtime_mount!r} and ensure "
                f"<runtime-root>/{self.config.kind}/bin/{self.config.executable} exists on the worker. "
                f"stdout={check.stdout[-self.config.output_tail_chars :]!r}, "
                f"stderr={check.stderr[-self.config.output_tail_chars :]!r}."
            )
        await self._prepare(runtime)

    @staticmethod
    def inherited_proxy_env() -> dict[str, str]:
        """Copy proxy variables from the Ray worker environment without values in code."""
        keys = (
            "http_proxy",
            "https_proxy",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "all_proxy",
            "ALL_PROXY",
            "no_proxy",
            "NO_PROXY",
        )
        return {key: os.environ[key] for key in keys if os.environ.get(key)}

    @staticmethod
    def callback_no_proxy(runtime: HarnessRuntime, env: Mapping[str, str]) -> dict[str, str]:
        """Keep the in-cluster callback endpoint outside the public proxy."""
        result = dict(env)
        host = urlsplit(runtime.session_root_url).hostname
        if not host:
            return result
        for key in ("no_proxy", "NO_PROXY"):
            entries = [item.strip() for item in result.get(key, "").split(",") if item.strip()]
            if host not in entries:
                entries.append(host)
            result[key] = ",".join(entries)
        return result

    @abstractmethod
    async def _prepare(self, runtime: HarnessRuntime) -> None:
        """Write adapter-specific files after the executable check succeeds."""

    @abstractmethod
    def build_command(self, prompt: str, runtime: HarnessRuntime) -> Sequence[str]:
        """Build the CLI argv without embedding environment secrets."""

    @abstractmethod
    def build_env(self, runtime: HarnessRuntime) -> Mapping[str, str]:
        """Build environment variables for one task-scoped inference endpoint."""

    async def run(self, prompt: str, runtime: HarnessRuntime) -> HarnessResult:
        """Run the harness CLI and retain bounded stdout/stderr diagnostic tails."""
        stderr_path = str(PurePosixPath(self._log_dir) / "stderr.log")
        stdout_path = str(PurePosixPath(self._log_dir) / "stdout.log")
        argv = self.build_command(prompt, runtime)
        command = (
            f"{shlex.join([str(item) for item in argv])} > {shlex.quote(stdout_path)} 2> {shlex.quote(stderr_path)}"
        )
        task = asyncio.create_task(
            self.sandbox.exec(
                command,
                cwd=runtime.workdir,
                env=self.build_env(runtime),
                timeout_s=self.config.time_budget_s,
            )
        )
        self._active_exec = task
        try:
            result = await task
        finally:
            if self._active_exec is task:
                self._active_exec = None
        if result.exit_code == 0:
            return HarnessResult(
                exit_code=0,
                command_name=str(argv[0]) if argv else self.config.executable,
                workdir=runtime.workdir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        return HarnessResult(
            exit_code=result.exit_code,
            stdout_tail=await self._read_tail(stdout_path),
            stderr_tail=await self._read_tail(stderr_path),
            command_name=str(argv[0]) if argv else self.config.executable,
            workdir=runtime.workdir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    async def collect_diagnostics(self, result: HarnessResult) -> HarnessResult:
        """Read process output on demand for failures discovered after exit 0."""
        if result.stdout_tail or result.stderr_tail or not result.stdout_path or not result.stderr_path:
            return result
        return replace(
            result,
            stdout_tail=await self._read_tail(result.stdout_path),
            stderr_tail=await self._read_tail(result.stderr_path),
        )

    async def abort(self) -> None:
        """Cancel the active exec. Sandbox lease release supplies the hard stop."""
        task = self._active_exec
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _read_tail(self, path: str) -> str:
        result = await self.sandbox.exec(
            f"tail -c {self.config.output_tail_chars} {shlex.quote(path)}",
            timeout_s=30,
        )
        return result.stdout if result.exit_code == 0 else ""
