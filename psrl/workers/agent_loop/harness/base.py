"""Task-scoped lifecycle contract for sandboxed coding harnesses."""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
from psrl.workers.agent_loop.harness.episode import (
    CLAUDE_CODE_STREAM_JSON,
    CODEX_JSONL,
    HarnessEpisodeSummary,
    episode_summary,
)
from psrl.workers.agent_loop.harness.runtime import executable_path

psrl_logger = logging.getLogger(__file__)

# The exit code a `HarnessResult` carries when the harness stopped the CLI at its
# deadline rather than the CLI exiting on its own.
EXIT_TIME_BUDGET_EXCEEDED = -1

# How long past the episode clock this layer keeps waiting for the sandbox's reply.
# The sandbox enforces the deadline itself and answers with a typed timeout; this
# wait only covers a transport that stops answering, so it sits above the SDK's own
# reply grace and never races the real answer.
_REPLY_BACKSTOP_S = 60.0

# Trajectory output formats the post-rollout integrity scanner and the episode
# summary parser can dispatch on. `auto` resolves to the kind default below, but the
# scan dispatch key is always a concrete format.
SUPPORTED_TRAJECTORY_FORMATS = (
    "auto",
    CLAUDE_CODE_STREAM_JSON,
    CODEX_JSONL,
    "plain_text",
)
_KIND_DEFAULT_TRAJECTORY_FORMATS: dict[str, str] = {
    "claude_code": CLAUDE_CODE_STREAM_JSON,
    "codex": CODEX_JSONL,
}

# Harnesses whose CLI takes no turn cap of its own. Codex runs an episode from one
# `exec` process and exposes no `--max-turns`-style flag, so a cap cannot be handed to
# it; the session it talks to is held to the budget instead, which is the one component
# that sees every model call. Listed by kind because taking a cap is a property of the
# CLI, and a new harness whose CLI takes none belongs here.
_KINDS_WITHOUT_CLI_TURN_CAP = frozenset({"codex"})


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
    # Optional task-level instruction appended to every episode prompt. Use it for
    # experiment-specific guidance that should not become a system prompt.
    prompt_suffix: str = ""
    # `none` keeps the historical shared workspace. `logical` makes the task
    # workdir the main artifact and sends each subagent to its own directory.
    workspace_isolation: str = "none"
    agent_workspace_root: str = "/workspace/agents"
    compaction: HarnessCompactionConfig = field(default_factory=HarnessCompactionConfig)
    trajectory_format: str = "auto"

    def __post_init__(self) -> None:
        if not self.kind or not self.executable:
            raise ValueError("Harness kind and executable cannot be empty.")
        if not self.runtime_mount.startswith("/"):
            raise ValueError(f"Harness runtime_mount must be an absolute path, got {self.runtime_mount!r}.")
        if self.output_tail_chars <= 0:
            raise ValueError("Harness output_tail_chars must be greater than zero.")
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
        if self.workspace_isolation not in ("none", "logical"):
            raise ValueError(
                f"Harness workspace_isolation must be 'none' or 'logical', got {self.workspace_isolation!r}."
            )
        if not self.agent_workspace_root.startswith("/"):
            raise ValueError(
                f"Harness agent_workspace_root must be absolute, got {self.agent_workspace_root!r}."
            )

    def state_dir(self) -> str:
        """The harness's own directory inside the sandbox.

        It holds the CLI's logs and any payload the framework stages for this episode,
        so it is the one directory on the guest side that the framework owns. Nothing
        the agent is told to use lives here, which is what makes it safe to stage a file
        in without competing with the agent's own scratch space.
        """
        return str(PurePosixPath(self.home_dir) / ".psrl-harness")

    def task_prompt(self, prompt: str, workdir: str) -> str:
        """Return the task prompt with the configured suffix and isolation rule."""
        return f"{prompt}{self.prompt_suffix}{self.isolate_prompt(workdir)}"

    def isolate_prompt(self, workdir: str) -> str:
        """Return the prompt suffix that states the logical workspace convention."""
        if self.workspace_isolation != "logical":
            return ""
        return (
            "\n\nWorkspace isolation: the main task root is "
            f"{workdir!r}; the main artifact is collected from that directory. "
            "When you delegate, require every subagent to write only under "
            f"{self.agent_workspace_root!r}/<agent-id>/ and never modify the main task root. "
            "Do not write outside those directories."
        )

    def resolved_trajectory_format(self) -> str:
        """Return the concrete format the integrity scanner should dispatch on."""
        if self.trajectory_format != "auto":
            return self.trajectory_format
        return _KIND_DEFAULT_TRAJECTORY_FORMATS.get(self.kind, "plain_text")

    @property
    def cli_takes_turn_cap(self) -> bool:
        """Return whether this harness's CLI accepts `rollout.multi_turn.max_turns`.

        A CLI that takes one stops itself at the configured number. A CLI that takes none
        is held to the same number by the session it talks to, declared as that session's
        turn budget, so a configured cap is honoured either way.
        """
        return self.kind not in _KINDS_WITHOUT_CLI_TURN_CAP

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


# The sentinel the gateway returns on an aborted request, echoed by the CLI into its
# event stream. Kept as a literal here because the CLI's output is bytes this service
# does not parse; matching the code is enough and parsing the JSON would couple this
# to the CLI's schema.
_ABORT_SENTINEL = "request_aborted"


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
    # Set when the harness stopped the CLI at its deadline rather than the CLI exiting.
    # The output files keep whatever the CLI had produced, so a `keep` caller can still
    # collect the partial trajectory and grade it.
    timed_out: bool = False

    def aborted_by_gateway(self) -> bool:
        """Report that the CLI stopped because the gateway aborted its request.

        The abort reaches the CLI as an ordinary HTTP error on an LLM call *inside the
        sandbox*, so by the time this service sees it, it looks like any other harness
        failure. That is the whole problem: an abort is not a failure of the harness, and
        treating it as one both retries work that was deliberately abandoned and feeds the
        breaker that exists to catch a genuinely broken CLI.

        Detected from the sentinel the gateway returns, which the CLI echoes into its own
        event stream. See `psrl.utils.common.http_utils` for the sentinel's definition.
        """
        return _ABORT_SENTINEL in (self.stdout_tail or "") or _ABORT_SENTINEL in (self.stderr_tail or "")

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
        self._log_dir = config.state_dir()

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

        What the probe checks is what the mount owns: that the executable is present
        and executable in this sandbox. Running it is not the probe's job, because a
        CLI is hundreds of megabytes and this ran once per trajectory, so every step
        paid for that read and its dynamic linking on the critical path of its whole
        batch, at the moment that step was also creating all of its sandboxes. Whether
        the binary runs in this image is answered by the first real invocation, which
        reports the CLI's own diagnostics when it does not.
        """
        exe = executable_path(self.config.runtime_mount, self.config.executable)
        directories = [self._log_dir, self.config_dir()]
        if self.config.workspace_isolation == "logical":
            directories.append(self.config.agent_workspace_root)
        mkdir = "mkdir -p " + " ".join(shlex.quote(d) for d in directories if d)
        check = await self.sandbox.exec(f"{mkdir} && test -x {shlex.quote(exe)}", timeout_s=60)
        if check.exit_code != 0:
            # `test -x` answers 1 for "not there or not executable" and nothing else. Any
            # other code is the probe failing, not the mount: a sandbox that cannot run
            # `mkdir`, or one whose filesystem is gone, reports something else entirely.
            # Reporting both as "missing or not executable" sent a hunt for a mount
            # problem that was really a container being torn down underneath the episode.
            details = (
                f"stdout={check.stdout[-self.config.output_tail_chars :]!r}, "
                f"stderr={check.stderr[-self.config.output_tail_chars :]!r}."
            )
            if check.exit_code == 1:
                raise RuntimeError(
                    f"Harness executable {exe!r} is missing or not executable in the sandbox. "
                    f"Mount the {self.config.kind!r} runtime tree at {self.config.runtime_mount!r} and ensure "
                    f"<runtime-root>/{self.config.kind}/bin/{self.config.executable} exists on the worker. "
                    f"{details}"
                )
            raise RuntimeError(
                f"Harness executable probe for {exe!r} failed (exit {check.exit_code}), which is "
                f"not the same as the mount being absent: the sandbox could not run the probe "
                f"at all, so the episode is lost to an infrastructure fault rather than to a "
                f"missing runtime tree. {details}"
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

    async def run(self, prompt: str, runtime: HarnessRuntime, *, deadline_s: float | None = None) -> HarnessResult:
        """Run the harness CLI and retain bounded stdout/stderr diagnostic tails.

        ``deadline_s`` is the episode clock the loop hands down, so the harness stops
        at the same instant the loop's budget expires rather than a second timer. When
        it fires, the CLI is cancelled and the result is returned with ``timed_out=True``
        and the output files intact, so a caller that keeps partial work can still read
        and grade what the CLI produced. ``None`` runs the CLI to completion and relies
        on an outside cancellation to stop it.
        """
        stderr_path = str(PurePosixPath(self._log_dir) / "stderr.log")
        stdout_path = str(PurePosixPath(self._log_dir) / "stdout.log")
        argv = self.build_command(prompt, runtime)
        command = (
            f"{shlex.join([str(item) for item in argv])} > {shlex.quote(stdout_path)} "
            f"2> {shlex.quote(stderr_path)} < /dev/null"
        )
        env = dict(self.build_env(runtime))
        if self.config.workspace_isolation == "logical":
            env["PSRL_AGENT_WORKSPACE_ROOT"] = self.config.agent_workspace_root
            env["PSRL_MAIN_WORKSPACE"] = runtime.workdir
        task = asyncio.create_task(
            self.sandbox.exec(
                command,
                cwd=runtime.workdir,
                env=env,
                # The episode clock is the command's own deadline: the sandbox stops the
                # CLI at it and keeps partial output, so the artifact collector can run
                # on the same sandbox immediately instead of queuing behind a turn
                # nobody is waiting for.
                timeout_s=deadline_s,
            )
        )
        self._active_exec = task
        try:
            if deadline_s is None:
                result = await task
            else:
                result = await asyncio.wait_for(task, timeout=deadline_s + _REPLY_BACKSTOP_S)
        except asyncio.TimeoutError:
            # Either the sandbox stopped the CLI at the episode deadline (its typed
            # command timeout is also a TimeoutError) or nothing answered at all.
            # Cancelling the exec is the best stop this layer can offer; the sandbox
            # lease release is the hard one. Return the partial output rather than
            # raising, so the caller decides whether to keep it.
            return HarnessResult(
                exit_code=EXIT_TIME_BUDGET_EXCEEDED,
                timed_out=True,
                stdout_tail=await self._read_tail(stdout_path),
                stderr_tail=await self._read_tail(stderr_path),
                command_name=str(argv[0]) if argv else self.config.executable,
                workdir=runtime.workdir,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
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

    def episode_summary(self, log_bytes: bytes) -> HarnessEpisodeSummary | None:
        """Parse what this harness's CLI reported about the episode it just ran.

        The CLI is the only component that knows how many turns it ran and why it
        stopped, so its own output is what a caller should label the episode from.
        Returns ``None`` when this harness's output format carries no such report.
        """
        return episode_summary(self.config.resolved_trajectory_format(), log_bytes)

    async def abort(self) -> None:
        """Cancel the active exec. Sandbox lease release supplies the hard stop."""
        task = self._active_exec
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task

    async def _read_tail(self, path: str) -> str:
        """Return a bounded tail of one log file, or an empty string if it cannot be read.

        This is diagnostics, so every failure is absorbed. A sandbox whose container has
        stopped or whose runtime has stalled raises here, and both `SandboxCommandTimeout`
        and `SandboxSessionLostError` would otherwise propagate out of a harness run that
        had already finished its work — turning a complete episode into a reported budget
        expiry or a lost container. Losing the tail costs a post mortem; losing the
        trajectory costs the episode.

        It reads the file rather than running `tail` in the sandbox, because a command is
        not free: on a backend whose exec cannot be signalled (a one-shot exec), a command
        that stalls is stopped by destroying the container. Swallowing that exception here
        would leave a sandbox already destroyed under an episode that still has a patch to
        collect, and the collection is what would then report the failure. A diagnostic
        must not be able to cost the episode its sandbox.
        """
        try:
            payload = await self.sandbox.read_bytes(path)
        except Exception:
            psrl_logger.warning(f"Could not read harness log tail {path!r}.", exc_info=True)
            return ""
        return payload[-self.config.output_tail_chars :].decode(errors="replace")
