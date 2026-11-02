"""Unit tests for task-scoped coding harness adapters."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
from examples.mini_swe.config import build_runtime_config
from examples.mini_swe.utils.harness_task import (
    _HARNESS_DIAG_NAME,
    _HARNESS_PATCH_NAME,
    build_harness_prompt,
    collect_git_patch,
)
from omegaconf import OmegaConf
from psrl.sandbox import (
    ExecResult,
    MountSpec,
    ResourceSpec,
    SandboxCapabilities,
    SandboxCollectorError,
    SandboxCommandTimeout,
    SandboxExitReason,
    SandboxPathNotFoundError,
    SandboxRef,
    SandboxSession,
    SandboxSessionLostError,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SandboxStatus,
)
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.backends.docker.policy import DockerPolicyProfile
from psrl.sandbox.backends.docker.session import DockerSession
from psrl.workers.agent_loop.harness import (
    EXIT_TIME_BUDGET_EXCEEDED,
    HarnessCompactionConfig,
    HarnessConfig,
    HarnessResult,
    HarnessRuntime,
    HarnessTaskContext,
    clean_snapshot_compatible,
    create_harness,
)
from psrl.workers.agent_loop.harness.claude_code import ClaudeCodeHarness
from psrl.workers.agent_loop.harness.codex import CodexHarness
from psrl.workers.agent_loop.harness.runtime import executable_path, host_runtime_dir, runtime_mount_spec
from psrl.workers.agent_loop.loops.mini_swe_harness_agent_loop import MiniSWEHarnessAgentLoop

# Where an episode stages the collector's payload: the harness's own state directory,
# next to the CLI's logs. Deliberately not /tmp, which is the agent's scratch space.
_STAGING_DIR = "/root/.psrl-harness"
_PATCH_PATH = f"{_STAGING_DIR}/{_HARNESS_PATCH_NAME}"
_DIAG_PATH = f"{_STAGING_DIR}/{_HARNESS_DIAG_NAME}"


class FakeSandbox(SandboxSession):
    """Minimal async data plane with deterministic command behavior."""

    def __init__(
        self,
        *,
        runtime_available: bool = True,
        block_cli: bool = False,
        cli_exit_code: int = 0,
        collect_exit_code: int = 0,
        diag_missing: bool = False,
        status_value: SandboxStatus = SandboxStatus.RUNNING,
        status_error: BaseException | None = None,
    ) -> None:
        self.runtime_available = runtime_available
        self.block_cli = block_cli
        self.cli_exit_code = cli_exit_code
        self.collect_exit_code = collect_exit_code
        # A missing diagnostics file in a live sandbox is the case that used to be
        # reported as a lost session; the status is scripted separately so a 137
        # can be attributed to OOM or to a killed child.
        self.diag_missing = diag_missing
        self.status_value = status_value
        self.status_error = status_error
        self.commands: list[tuple[str, str | None, Mapping[str, str] | None]] = []
        # The deadline each command carried, so a test can pin that the harness hands
        # its episode clock to the sandbox instead of relying on its own timer.
        self.timeouts: list[float | None] = []
        self.writes: dict[str, bytes] = {}
        self.cli_started = asyncio.Event()
        self.cli_cancelled = False

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef("fake", "sandbox-1")

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        self.commands.append((command, cwd, env))
        self.timeouts.append(timeout_s)
        # The prepare probe creates the state dir and checks the mounted CLI.
        if command.startswith("mkdir -p /root/.psrl-harness"):
            return ExecResult(0 if self.runtime_available else 1, "", "" if self.runtime_available else "no such file")
        if "ulimit -u" in command:
            # The failed-collector environment probe: names the missing tool.
            return ExecResult(0, "cwd=/testbed\npath=/usr/bin\nwc=MISSING\n", "")
        if _HARNESS_PATCH_NAME in command:
            # The patch collector. Its diagnostics are read back through the file API,
            # never from this response (the backend does not carry stderr).
            return ExecResult(self.collect_exit_code, "", "")
        if "/opt/harness/bin/" in command and self.block_cli:
            self.cli_started.set()
            try:
                if timeout_s is None:
                    await asyncio.Future()
                else:
                    # The sandbox enforces a command's deadline itself. Modeling it here
                    # is what makes a harness that relies on its own timer fail this
                    # test instead of passing slowly.
                    await asyncio.wait_for(asyncio.Future(), timeout=timeout_s)
            except asyncio.TimeoutError as exc:  # a deadline stop, not a cancellation
                self.cli_cancelled = True
                raise SandboxCommandTimeout(
                    "the CLI ran out of its command budget", sandbox_preserved=True
                ) from exc
            except asyncio.CancelledError:
                self.cli_cancelled = True
                raise
        if "/opt/harness/bin/" in command:
            return ExecResult(self.cli_exit_code, "", "")
        return ExecResult(0, "", "")

    async def read_bytes(self, path: str) -> bytes:
        if path not in self.writes:
            raise SandboxPathNotFoundError(f"{path!r} is not present in the sandbox")
        return self.writes[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.writes[path] = data

    async def status(self) -> SandboxStatus:
        if self.status_error is not None:
            raise self.status_error
        return self.status_value

    async def terminate(self) -> None:
        return None


class LocalSession(SandboxSession):
    """Run the collector's real shell script against a local temporary repository.

    The rest of the harness tests script command results, which cannot catch a
    shell-level regression in the collector. This session executes the exact
    command string, so the size caps and the per-file diff loop are exercised.
    """

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef("local", "local")

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return ExecResult(completed.returncode, completed.stdout, completed.stderr)

    async def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> None:
        Path(path).write_bytes(data)

    async def status(self) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def terminate(self) -> None:
        return None


def _runtime(
    context_window_tokens: int | None = None,
    compaction_token_limit: int | None = None,
    max_turns: int | None = None,
) -> HarnessRuntime:
    return HarnessRuntime(
        session_id="session-1",
        session_root_url="http://router/sessions/session-1",
        workdir="/testbed",
        model="Qwen/Qwen3",
        context_window_tokens=context_window_tokens,
        compaction_token_limit=compaction_token_limit,
        max_turns=max_turns,
    )


def test_harness_compaction_budget_uses_configured_percent_of_window() -> None:
    assert HarnessCompactionConfig().resolve(10_240) == (10_240, 7_680)
    assert HarnessCompactionConfig(compact_percent=100).resolve(10_240) == (10_240, 10_240)
    assert HarnessCompactionConfig(enabled=False).resolve(10_240) is None
    with pytest.raises(ValueError, match="compact_percent"):
        HarnessCompactionConfig(compact_percent=0).resolve(10_240)
    with pytest.raises(ValueError, match="compact_percent"):
        HarnessCompactionConfig(compact_percent=101).resolve(10_240)
    with pytest.raises(ValueError, match="context window"):
        HarnessCompactionConfig().resolve(0)


def test_harness_prompt_preserves_native_miniswe_task_boundary() -> None:
    prompt = build_harness_prompt("Fix the parser.")

    assert prompt == (
        "<pr_description>\nFix the parser.\n</pr_description>\n\n"
        "Implement the required changes in the current repository and verify the fix.\n\n"
        "Integrity rules:\n"
        "- Do not modify tests, pytest configuration, or evaluation harness files.\n"
        "- Do not retrieve a solution, patch, commit, or pull request from the task repository or its mirrors.\n"
        "- Do not create nested git repositories (directories containing a .git) inside the working directory; "
        "if you need a scratch repository to reproduce the issue, create it under /tmp instead."
    )


@pytest.mark.asyncio
async def test_claude_code_uses_session_root_and_bounded_process_output() -> None:
    sandbox = FakeSandbox(cli_exit_code=1)
    config = HarnessConfig.from_value(
        {
            "kind": "claude_code",
            "executable": "claude",
            "args": ["--max-budget-usd", "0"],
            "permission_mode": "acceptEdits",
            "allowed_permissions": ["Bash(*)", "Read(*)", "Edit(*)"],
            "system_prompt": "minimal system",
            "tools": "Bash,Read,Edit",
            "env": {"ANTHROPIC_BASE_URL": "http://must-not-escape", "CUSTOM": "value"},
        }
    )
    harness = create_harness(config, sandbox)

    await harness.prepare(_runtime())
    # What the CLI had written to its log before it exited non-zero.
    sandbox.writes[f"{_STAGING_DIR}/stderr.log"] = b"bounded tail"
    result = await harness.run("fix the bug", _runtime())

    assert isinstance(harness, ClaudeCodeHarness)
    assert result.exit_code == 1
    assert result.stderr_tail == "bounded tail"
    assert b"hasCompletedOnboarding" in sandbox.writes["/root/.claude.json"]
    settings = json.loads(sandbox.writes["/root/.claude/settings.json"])
    assert settings["permissions"]["allow"] == ["Bash(*)", "Read(*)", "Edit(*)"]
    assert settings["permissions"]["deny"] == []
    assert "defaultMode" not in settings["permissions"]
    assert settings["env"]["CLAUDE_CODE_GATEWAY_HINT_HEADERS"] == "1"
    cli_command, cwd, env = next(item for item in sandbox.commands if "--output-format" in item[0])
    assert "--output-format stream-json" in cli_command
    assert "--permission-mode acceptEdits" in cli_command
    assert "--allowedTools" not in cli_command
    assert "--system-prompt 'minimal system'" in cli_command
    assert "--tools Bash,Read,Edit" in cli_command
    assert "--max-turns" not in cli_command
    assert "> /root/.psrl-harness/stdout.log" in cli_command
    assert cwd == "/testbed"
    assert env is not None
    assert env["ANTHROPIC_BASE_URL"] == "http://router/sessions/session-1"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "session-1"
    assert env["CUSTOM"] == "value"


@pytest.mark.asyncio
async def test_successful_harness_skips_diagnostic_tail_commands() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)

    await harness.prepare(_runtime())
    result = await harness.run("fix", _runtime())

    assert result.exit_code == 0
    assert result.stdout_tail == ""
    assert result.stderr_tail == ""
    # Only the prepare probe and the CLI itself: a successful run spends nothing on
    # diagnostics, and reading them costs no command at all.
    assert len(sandbox.commands) == 2


@pytest.mark.asyncio
async def test_the_harness_stops_at_the_deadline_and_keeps_partial_output() -> None:
    """A `keep` loop hands the harness its episode deadline, so the CLI stops at it.

    The deadline travels to the sandbox as the command's own budget, and the sandbox
    stops the CLI and answers with a command timeout. The run has to return rather
    than raise, with `timed_out` set and the output files still pointing at what the
    CLI had written, because the loop that chose `keep` collects the partial
    trajectory and grades it.
    """
    sandbox = FakeSandbox(block_cli=True)
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)
    await harness.prepare(_runtime())
    # What the CLI had written to its log before the deadline stopped it.
    sandbox.writes[f"{_STAGING_DIR}/stdout.log"] = b"bounded tail"

    result = await harness.run("fix", _runtime(), deadline_s=0.05)

    assert result.timed_out is True
    assert result.exit_code == EXIT_TIME_BUDGET_EXCEEDED
    assert sandbox.cli_cancelled, "the deadline stops the CLI rather than leaving it running"
    # The episode clock travels to the sandbox as the command's own deadline, so the
    # sandbox stops the CLI and the collector finds a free shell behind it.
    assert sandbox.timeouts[-1] == pytest.approx(0.05)
    # The deadline stops the CLI; it does not discard what the CLI produced.
    assert result.stdout_path == "/root/.psrl-harness/stdout.log"
    assert result.stdout_tail == "bounded tail"


@pytest.mark.asyncio
async def test_the_harness_without_a_deadline_runs_the_cli_to_completion() -> None:
    # `discard` mode passes no deadline, so the harness never stops the CLI on its own;
    # the base cancels the whole episode instead.
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)
    await harness.prepare(_runtime())

    result = await harness.run("fix", _runtime(), deadline_s=None)

    assert result.timed_out is False
    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_claude_code_receives_percent_based_compaction_settings() -> None:
    sandbox = FakeSandbox()
    config = HarnessConfig.from_value(
        {
            "kind": "claude_code",
            "executable": "claude",
            "max_output_tokens": 8192,
            "compaction": {"enabled": True, "compact_percent": 80},
        }
    )
    harness = create_harness(config, sandbox)

    env = harness.build_env(_runtime(10_240, 8_192))

    assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "10240"
    assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "80"
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8192"


def test_claude_code_applies_turn_bound_and_leaves_output_limit_config_gated() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="claude_code", executable="claude"), sandbox)
    configured = create_harness(
        HarnessConfig(kind="claude_code", executable="claude", max_output_tokens=8192),
        sandbox,
    )
    runtime = _runtime(max_turns=12)

    command = harness.build_command("fix", runtime)

    assert "--max-turns 12" in " ".join(command)
    # `CLAUDE_CODE_MAX_OUTPUT_TOKENS` is config-gated: when unset, Claude Code's
    # default applies. Only an explicit `max_output_tokens` writes the env var.
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in harness.build_env(runtime)
    assert configured.build_env(runtime)["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "8192"


@pytest.mark.asyncio
async def test_failed_trajectory_can_collect_output_after_successful_cli_exit() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)
    await harness.prepare(_runtime())

    result = await harness.run("fix", _runtime())
    sandbox.writes[result.stdout_path] = b"stdout diagnostics"
    sandbox.writes[result.stderr_path] = b"stderr diagnostics"

    diagnostic_result = await harness.collect_diagnostics(result)

    assert diagnostic_result.stdout_tail == "stdout diagnostics"
    assert diagnostic_result.stderr_tail == "stderr diagnostics"


@pytest.mark.asyncio
async def test_a_log_tail_is_read_as_a_file_rather_than_run_as_a_command() -> None:
    """A diagnostic must not be able to cost the episode its sandbox.

    On a one-shot exec a command that stalls is stopped by destroying the container, so
    reading the tail with `tail -c` left the collection that follows to fail against a
    sandbox this read had already destroyed. The file API cannot do that.
    """
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)
    log_path = f"{_STAGING_DIR}/stdout.log"
    sandbox.writes[log_path] = b"x" * 40_000 + b"THE-TAIL"

    tail = await harness._read_tail(log_path)

    assert tail.endswith("THE-TAIL")
    assert len(tail) == harness.config.output_tail_chars, "The reported tail stays bounded."
    assert not any(command.startswith("tail -c") for command, _, _ in sandbox.commands)
    # An unreadable log still costs only a post mortem, never the episode.
    assert await harness._read_tail(f"{_STAGING_DIR}/missing.log") == ""


@pytest.mark.asyncio
async def test_patch_collection_reads_the_patch_file_in_full() -> None:
    """The patch is fetched as a file, so no command-output budget can cut it short.

    Reading it back as command output capped the patch at the diagnostic budget, and a
    patch cut short would be graded as if it were the agent's whole solution.
    """
    sandbox = FakeSandbox()
    collected = b"diff --git a/x b/x\n+line\n" * 100_000
    sandbox.writes[_PATCH_PATH] = collected

    patch = await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    script, cwd, _ = sandbox.commands[-1]
    assert cwd == "/testbed"
    # Staged (index vs base/HEAD) and unstaged (worktree vs index) tracked changes,
    # emitted one bounded diff per file so a large artifact cannot OOM the sandbox.
    assert "cached=--cached" in script
    assert "git diff $cached --binary --submodule=diff" in script
    assert "skipped_tracked" in script
    assert "patch_capped" in script
    # Untracked files become new-file diffs, but nested repositories are skipped.
    assert "git ls-files --others --exclude-standard" in script
    assert '[ -e "$d/.git" ]' in script
    # The sandbox runs commands with /bin/sh, where bash's `read -d` silently skipped
    # every untracked file; the NUL list is consumed with `xargs -0` instead.
    assert "read -r -d" not in script
    assert "xargs -0" in script
    # The collector never stages, so a nested repo without a commit cannot break it.
    assert "git add" not in script
    # The patch leaves the sandbox as a file rather than as command output.
    assert f"out={_PATCH_PATH}" in script
    # The staging directory is re-asserted by the collector itself, so a directory the
    # agent removed cannot turn a collectable patch into a failed collection.
    assert f"mkdir -p {_STAGING_DIR}" in script
    # A self-report the caller can read back when the backend reports no stderr.
    assert f"diag={_DIAG_PATH}" in script
    assert "command -v" in script
    assert "cat " not in script
    assert patch == collected.decode()


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
@pytest.mark.asyncio
async def test_collector_skips_an_untracked_file_above_the_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large untracked artifact is skipped and named, not diffed into the sandbox.

    An unbounded `git diff --binary` on an agent-created file is what turned a
    collector failure into a sandbox OOM and then into a whole-group replacement.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"], cwd=repo, check=True, capture_output=True
    )
    (repo / "a.txt").write_text("one\ntwo\n")
    (repo / "big.bin").write_bytes(b"x" * 4096)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(
        "examples.mini_swe.utils.harness_task._MAX_UNTRACKED_FILE_BYTES", 1024
    )

    patch = await collect_git_patch(LocalSession(), str(repo), staging_dir=str(staging))

    assert "+two" in patch
    assert "big.bin" not in patch
    diag = (staging / _HARNESS_DIAG_NAME).read_text()
    assert "skipped_untracked=big.bin" in diag
    assert "collector=done" in diag


@pytest.mark.skipif(shutil.which("git") is None, reason="git is required")
@pytest.mark.asyncio
async def test_collector_stops_at_the_total_patch_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A patch over the budget fails as a collector fault, not as an OOM.

    The sandbox stays alive; the episode is replaced by the collector's own
    classification and the diagnostics name the cap.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "base"], cwd=repo, check=True, capture_output=True
    )
    (repo / "a.txt").write_text("one\n" + ("two\n" * 4096))
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(
        "examples.mini_swe.utils.harness_task._MAX_PATCH_BYTES", 512
    )

    with pytest.raises(SandboxCollectorError) as excinfo:
        await collect_git_patch(LocalSession(), str(repo), staging_dir=str(staging))

    assert "patch_capped=1" in str(excinfo.value)
    assert "status='running'" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_collector_that_times_out_grades_what_it_wrote() -> None:
    """A collector stopped at its budget leaves a gradeable patch behind.

    The collector writes each file's diff as it goes, so a budget stop is not a failed
    collection: discarding what it wrote would cost a finished harness run to the size
    of the repository rather than to the solution.
    """

    class SlowCollector(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None):
            if _HARNESS_PATCH_NAME in command:
                self.writes[_PATCH_PATH] = b"diff --git a/x b/x\n+done\n"
                raise SandboxCommandTimeout("collector stopped at its budget", sandbox_preserved=True)
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s)

    patch = await collect_git_patch(SlowCollector(), "/testbed", staging_dir=_STAGING_DIR)

    assert patch == "diff --git a/x b/x\n+done\n"


@pytest.mark.asyncio
async def test_a_collector_that_times_out_before_writing_anything_is_a_collector_fault() -> None:
    class EmptyCollector(FakeSandbox):
        async def exec(self, command, *, cwd=None, env=None, timeout_s=None):
            if _HARNESS_PATCH_NAME in command:
                raise SandboxCommandTimeout("collector stopped at its budget", sandbox_preserved=True)
            return await super().exec(command, cwd=cwd, env=env, timeout_s=timeout_s)

    with pytest.raises(SandboxCollectorError) as excinfo:
        await collect_git_patch(EmptyCollector(), "/testbed", staging_dir=_STAGING_DIR)

    assert "before writing any patch" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_loss_before_collection_logs_the_runtime_evidence(caplog) -> None:
    """A lost sandbox leaves the daemon's evidence in the collector log.

    The collector has nothing else to add once the sandbox is gone, so the
    exception's exit code, finish time and state_error are the post mortem's
    sandbox-side record.
    """

    class LostSession(FakeSandbox):
        async def exec(self, command, **kwargs):
            raise SandboxSessionLostError(
                "gone",
                sandbox_id="sandbox-1",
                exit_reason=SandboxExitReason.OOM,
                exit_code=137,
                finished_at="2026-10-23T11:05:23Z",
                state_error="status=exited exit_code=137 oom_killed=true",
            )

    with caplog.at_level(logging.WARNING):
        with pytest.raises(SandboxSessionLostError):
            await collect_git_patch(LostSession(), "/testbed", staging_dir=_STAGING_DIR)

    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "exit_code=137" in message
    assert "state_error=status=exited exit_code=137 oom_killed=true" in message


@pytest.mark.asyncio
async def test_patch_collection_failure_surfaces_collector_diagnostics() -> None:
    """A failed collection reports the collector's own diagnostics, not just an exit code.

    The sandbox exec channel does not carry stderr on every backend, so the exit code
    alone said nothing about which step failed.
    """
    sandbox = FakeSandbox(collect_exit_code=127)
    sandbox.writes[_DIAG_PATH] = b"cwd=/testbed\nwc=MISSING\n"

    with pytest.raises(RuntimeError) as excinfo:
        await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    message = str(excinfo.value)
    assert "exit 127" in message
    assert "cwd='/testbed'" in message
    assert "wc=MISSING" in message


@pytest.mark.asyncio
async def test_a_sigkill_after_the_sandbox_died_is_an_oom_loss() -> None:
    """137 plus a sandbox that is no longer running is the kernel killing the sandbox."""
    sandbox = FakeSandbox(
        collect_exit_code=137,
        diag_missing=True,
        status_value=SandboxStatus.TERMINATED,
    )

    with pytest.raises(SandboxSessionLostError) as excinfo:
        await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    assert excinfo.value.exit_reason is SandboxExitReason.OOM


@pytest.mark.asyncio
async def test_a_sigkill_inside_a_live_sandbox_is_a_collector_fault() -> None:
    """A killed child is not a lost sandbox: the container is still running and usable."""
    sandbox = FakeSandbox(
        collect_exit_code=137,
        diag_missing=True,
        status_value=SandboxStatus.RUNNING,
    )

    with pytest.raises(SandboxCollectorError) as excinfo:
        await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    assert excinfo.value.exit_code == 137


@pytest.mark.asyncio
async def test_a_missing_diagnostics_file_in_a_live_sandbox_is_a_collector_fault() -> None:
    """A live sandbox whose collector never wrote its report did not lose its session.

    The container is still running (the runtime answers `running`), so the
    failure is the collector's execution environment. Reporting it as a lost
    session would destroy a healthy sandbox and blame the wrong layer.
    """
    sandbox = FakeSandbox(collect_exit_code=128, diag_missing=True)

    with pytest.raises(SandboxCollectorError) as excinfo:
        await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    assert excinfo.value.exit_code == 128
    assert "status='running'" in str(excinfo.value)
    assert "environment probe" in str(excinfo.value)
    assert "wc=MISSING" in str(excinfo.value)
    assert "sandbox='sandbox-1'" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_missing_diagnostics_file_after_the_sandbox_died_is_a_lost_session() -> None:
    """The same collector failure with a dead sandbox is a lost session."""
    sandbox = FakeSandbox(
        collect_exit_code=128,
        diag_missing=True,
        status_value=SandboxStatus.TERMINATED,
    )

    with pytest.raises(SandboxSessionLostError) as excinfo:
        await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    assert excinfo.value.exit_reason is SandboxExitReason.CONTAINER_EXITED
    assert excinfo.value.exit_code == 128


@pytest.mark.asyncio
async def test_a_status_that_reports_the_loss_keeps_its_exit_attribution() -> None:
    """The service's exit code and finish time are stronger than the collector's."""
    lost = SandboxSessionLostError(
        "sandbox is gone",
        sandbox_id="sb-attributed",
        exit_reason=SandboxExitReason.OOM,
        exit_code=137,
        finished_at="2026-10-22T02:13:07Z",
    )
    sandbox = FakeSandbox(
        collect_exit_code=137,
        diag_missing=True,
        status_error=lost,
    )

    with pytest.raises(SandboxSessionLostError) as excinfo:
        await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    assert excinfo.value.sandbox_id == "sandbox-1"
    assert excinfo.value.exit_reason is SandboxExitReason.OOM
    assert excinfo.value.exit_code == 137
    assert excinfo.value.finished_at == "2026-10-22T02:13:07Z"
    assert "environment probe" in str(excinfo.value)


@pytest.mark.asyncio
async def test_missing_tools_are_a_collector_fault_with_diagnostics() -> None:
    """126/127 mean the sandbox could not execute the collector's tools."""
    sandbox = FakeSandbox(collect_exit_code=127)
    sandbox.writes[_DIAG_PATH] = b"git=MISSING\nwc=MISSING\n"

    with pytest.raises(SandboxCollectorError) as excinfo:
        await collect_git_patch(sandbox, "/testbed", staging_dir=_STAGING_DIR)

    assert excinfo.value.exit_code == 127
    assert "git=MISSING" in str(excinfo.value)


@pytest.mark.asyncio
async def test_mini_swe_stages_the_patch_in_the_harness_state_dir() -> None:
    """The mini-SWE collector stages where the harness already owns the directory.

    Not /tmp: that is the scratch space the harness prompt hands to the agent, and a
    sandbox whose /tmp was gone reported a collectable patch as a failed collection.
    """
    loop = MiniSWEHarnessAgentLoop.__new__(MiniSWEHarnessAgentLoop)
    loop.harness_config = HarnessConfig(kind="codex", executable="codex")
    sandbox = FakeSandbox()
    sandbox.writes[_PATCH_PATH] = b"diff --git a/x b/x\n"
    task = HarnessTaskContext(
        state=SimpleNamespace(payload={"observation": {"swe_problem": {}}}),
        prompt="fix",
        sandbox_spec=SandboxSpec(SandboxSource.image("image")),
    )

    artifact = await loop.collect_harness_artifact(
        task,
        sandbox,
        SimpleNamespace(workdir="/testbed"),
        HarnessResult(exit_code=0),
    )

    script, cwd, _ = sandbox.commands[-1]
    assert cwd == "/testbed"
    assert f"out={_PATCH_PATH}" in script
    assert f"mkdir -p {_STAGING_DIR}" in script
    assert artifact.patch == "diff --git a/x b/x\n"


@pytest.mark.asyncio
async def test_codex_writes_responses_provider_config_and_uses_session_as_key() -> None:
    sandbox = FakeSandbox()
    config = HarnessConfig(kind="codex", executable="codex")
    harness = create_harness(config, sandbox)

    await harness.prepare(_runtime())

    assert isinstance(harness, CodexHarness)
    codex_config = sandbox.writes["/root/.codex/config.toml"].decode()
    assert 'base_url = "http://router/sessions/session-1/v1"' in codex_config
    assert 'wire_api = "responses"' in codex_config
    assert "requires_openai_auth = false" in codex_config
    assert "supports_websockets = false" in codex_config
    assert 'model_reasoning_effort = "medium"' in codex_config
    assert 'model_reasoning_summary = "auto"' in codex_config
    assert 'model_catalog_json = "/root/.codex/model_catalog.json"' in codex_config
    assert "enabled = false" in codex_config  # analytics / feedback disabled
    # Hosted web_search must be off: the gateway rejects its schema and the task
    # forbids fetching the solution from the web. Must be a top-level key.
    parsed = tomllib.loads(codex_config)
    assert parsed["web_search"] == "disabled"
    assert parsed["features"]["multi_agent"] is True
    assert harness.build_env(_runtime())["OPENAI_API_KEY"] == "session-1"
    command = harness.build_command("fix", _runtime())
    assert command[0] == "/opt/harness/bin/codex"
    assert command[1:4] == ("exec", "--skip-git-repo-check", "--json")


@pytest.mark.asyncio
async def test_codex_catalog_enables_apply_patch_and_carries_base_instructions() -> None:
    from psrl.workers.agent_loop.harness.codex import _base_instructions

    sandbox = FakeSandbox()
    harness = create_harness(
        HarnessConfig(kind="codex", executable="codex", reasoning_effort="high"),
        sandbox,
    )

    await harness.prepare(_runtime())

    catalog = json.loads(sandbox.writes["/root/.codex/model_catalog.json"])
    assert len(catalog["models"]) == 1
    entry = catalog["models"][0]
    assert entry["slug"] == "Qwen/Qwen3"
    # apply_patch is only exposed when the catalog marks the model freeform.
    assert entry["apply_patch_tool_type"] == "freeform"
    assert entry["shell_type"] == "unified_exec"
    assert entry["default_reasoning_level"] == "high"
    assert entry["multi_agent_version"] == "v1"
    assert entry["model_messages"]["instructions_template"] == _base_instructions()
    assert _base_instructions().strip() != ""


@pytest.mark.asyncio
async def test_codex_writes_compaction_threshold_with_context_headroom() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)

    await harness.prepare(_runtime(10_240, 7_680))

    codex_config = sandbox.writes["/root/.codex/config.toml"].decode()
    parsed = tomllib.loads(codex_config)
    # Compaction keys must be top-level, not nested under a preceding table.
    assert parsed["model_context_window"] == 10240
    assert parsed["model_auto_compact_token_limit"] == 7680
    assert parsed["analytics"]["enabled"] is False
    assert parsed["feedback"]["enabled"] is False
    assert parsed["model_providers"]["psrl"]["wire_api"] == "responses"


@pytest.mark.asyncio
async def test_prepare_fails_clearly_when_runtime_tree_is_missing() -> None:
    sandbox = FakeSandbox(runtime_available=False)
    config = HarnessConfig(kind="codex", executable="codex")

    with pytest.raises(RuntimeError, match="not executable in the sandbox"):
        await create_harness(config, sandbox).prepare(_runtime())

    # The explicit probe is the only setup command. There is no install path.
    assert all("npm" not in command for command, _, _ in sandbox.commands)


@pytest.mark.asyncio
async def test_prepare_probes_the_mount_without_running_the_cli() -> None:
    # Running the CLI once per trajectory put a several-hundred-megabyte read and its
    # dynamic linking on the critical path of every episode, in the same moment that step
    # was creating its whole batch of sandboxes. The probe only has to answer what the
    # mount owns, and whether the binary runs is answered by the first real invocation.
    sandbox = FakeSandbox()
    config = HarnessConfig(kind="codex", executable="codex")

    await create_harness(config, sandbox).prepare(_runtime())

    probes = [command for command, _, _ in sandbox.commands if command.startswith("mkdir -p /root/.psrl-harness")]
    assert len(probes) == 1
    assert "test -x" in probes[0]
    assert "--version" not in probes[0]


@pytest.mark.asyncio
async def test_abort_cancels_active_cli_exec() -> None:
    sandbox = FakeSandbox(block_cli=True)
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)
    await harness.prepare(_runtime())
    run_task = asyncio.create_task(harness.run("fix", _runtime()))
    await sandbox.cli_started.wait()

    await harness.abort()

    with pytest.raises(asyncio.CancelledError):
        await run_task
    assert sandbox.cli_cancelled


def test_registry_returns_new_task_owned_harnesses() -> None:
    config = HarnessConfig(kind="codex", executable="codex")
    sandbox = FakeSandbox()

    assert create_harness(config, sandbox) is not create_harness(config, sandbox)


def test_example_config_selects_both_harnesses() -> None:
    path = Path(__file__).parents[2] / "examples/mini_swe/config/swebench_harness_config.yaml"
    configs = OmegaConf.load(path)

    assert [item.name for item in configs] == ["mini_swe_claude_code", "mini_swe_codex"]
    assert [HarnessConfig.from_value(item.harness).kind for item in configs] == ["claude_code", "codex"]
    assert all(item.sandbox_config.environment.cwd == "/testbed" for item in configs)

    runtime = build_runtime_config(
        {"sandbox_config": OmegaConf.to_container(configs[0].sandbox_config, resolve=True)},
        require_agent_templates=False,
    )
    assert runtime.agent.system_template == ""
    assert runtime.sandbox_config.environment.cwd == "/testbed"


def test_subagent_example_config_enables_delegation_without_training_it() -> None:
    path = Path(__file__).parents[2] / "examples/mini_swe/config/swebench_harness_config_codex_subagent.yaml"
    configs = OmegaConf.load(path)

    assert [item.name for item in configs] == ["mini_swe_codex_subagent"]
    harness = HarnessConfig.from_value(configs[0].harness)
    assert harness.kind == "codex"
    assert harness.subagents_enabled is True
    assert harness.workspace_isolation == "logical"
    assert "delegate" in harness.prompt_suffix.lower()
    assert configs[0].sandbox_config.environment.cwd == "/testbed"


def test_claude_code_subagent_example_config_enables_delegation_without_training_it() -> None:
    path = Path(__file__).parents[2] / "examples/mini_swe/config/swebench_harness_config_cc_subagent.yaml"
    configs = OmegaConf.load(path)

    assert [item.name for item in configs] == ["mini_swe_claude_code_subagent"]
    harness = HarnessConfig.from_value(configs[0].harness)
    assert harness.kind == "claude_code"
    assert harness.subagents_enabled is True
    assert harness.workspace_isolation == "logical"
    # The subagent tool is named Agent in current Claude Code and was Task before it;
    # both must be admitted for the delegation path to be available.
    assert {"Agent", "Task"} <= set(harness.allowed_permissions)
    assert "delegate" in harness.prompt_suffix.lower()
    assert configs[0].sandbox_config.environment.cwd == "/testbed"


def test_task_context_only_enables_compatible_clean_snapshot() -> None:
    state_policy = SandboxStatePolicy(enabled=True)
    runtime_mount = MountSpec("/host/runtimes/claude_code", "/opt/harness", read_only=True)
    rollout = SandboxSpec(source=SandboxSource.image("image"), state_policy=state_policy)
    rollout_with_runtime = SandboxSpec(
        source=SandboxSource.image("image"),
        mounts=(runtime_mount,),
        state_policy=state_policy,
    )
    compatible = HarnessTaskContext(
        state={"task": "opaque"},
        prompt="solve",
        sandbox_spec=rollout,
        clean_sandbox_spec=SandboxSpec(source=SandboxSource.image("image"), state_policy=state_policy),
    )
    # The read-only harness runtime mount is excluded from mount compatibility.
    compatible_with_runtime = HarnessTaskContext(
        state={"task": "opaque"},
        prompt="solve",
        sandbox_spec=rollout_with_runtime,
        clean_sandbox_spec=SandboxSpec(source=SandboxSource.image("image"), state_policy=state_policy),
        runtime_mount_target="/opt/harness",
    )
    incompatible = HarnessTaskContext(
        state=None,
        prompt="solve",
        sandbox_spec=rollout,
        clean_sandbox_spec=SandboxSpec(source=SandboxSource.image("other-image"), state_policy=state_policy),
    )
    disabled = HarnessTaskContext(
        state=None,
        prompt="solve",
        sandbox_spec=SandboxSpec(source=SandboxSource.image("image")),
        clean_sandbox_spec=SandboxSpec(source=SandboxSource.image("image")),
    )

    assert clean_snapshot_compatible(compatible)
    assert clean_snapshot_compatible(compatible_with_runtime)
    assert not clean_snapshot_compatible(incompatible)
    assert not clean_snapshot_compatible(disabled)


def test_filesystem_clean_snapshot_ignores_source_and_resources() -> None:
    """A committed image seeds the grader regardless of its own flags.

    This is what lets a lightweight rollout sandbox (8GiB, baked derivative)
    seed the heavier grader sandbox (30GiB, original image): ``docker commit``
    captures the filesystem only, and restore recreates the container with the
    grader's own source and resources.
    """
    from psrl.sandbox import SnapshotKind

    state_policy = SandboxStatePolicy(enabled=True)
    rollout = SandboxSpec(
        source=SandboxSource.image("psrl/swebench-harness:abc"),
        resources=ResourceSpec(cpu_count=2, memory_mb=8 * 1024),
        state_policy=state_policy,
    )
    grader = SandboxSpec(
        source=SandboxSource.image("swebench/sweb.eval.x86_64.repo_1776_repo-1:latest"),
        resources=ResourceSpec(cpu_count=None, memory_mb=30 * 1024),
        state_policy=state_policy,
    )
    task = HarnessTaskContext(state=None, prompt="solve", sandbox_spec=rollout, clean_sandbox_spec=grader)

    assert clean_snapshot_compatible(task, SnapshotKind.FILESYSTEM)
    # A full-state snapshot must be rebuilt with matching source and resources.
    assert not clean_snapshot_compatible(task, SnapshotKind.FULL_STATE)


def test_runtime_tree_paths_are_derived_from_mount_and_root(monkeypatch, tmp_path) -> None:
    assert executable_path("/opt/harness", "claude") == "/opt/harness/bin/claude"

    monkeypatch.delenv("PSRL_HARNESS_RUNTIME_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="PSRL_HARNESS_RUNTIME_ROOT"):
        host_runtime_dir("claude_code")

    monkeypatch.setenv("PSRL_HARNESS_RUNTIME_ROOT", str(tmp_path))
    (tmp_path / "claude_code" / "bin").mkdir(parents=True)
    mount = runtime_mount_spec("claude_code", "/opt/harness")
    assert mount.source == str(tmp_path / "claude_code")
    assert mount.target == "/opt/harness"
    assert mount.read_only

    with pytest.raises(RuntimeError, match="missing on this worker"):
        runtime_mount_spec("codex", "/opt/harness")


def test_generic_harness_loop_has_no_task_specific_imports() -> None:
    path = Path(__file__).parents[2] / "psrl/workers/agent_loop/loops/harness_agent_loop.py"
    module = ast.parse(path.read_text())
    imports = {
        node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(name.startswith("examples.") for name in imports)


def test_docker_session_translates_loopback_callback_through_policy() -> None:
    backend = DockerBackend(
        policy_profiles={
            "mini_swe": DockerPolicyProfile(host_gateway_alias="host.docker.internal"),
        },
        engine=object(),
    )
    session = DockerSession(
        backend,
        "container-1",
        spec=SandboxSpec(
            source=SandboxSource.image("image"),
            policy_profile="mini_swe",
        ),
    )

    assert session.resolve_callback_url("http://127.0.0.1:8080/sessions/s1") == (
        "http://host.docker.internal:8080/sessions/s1"
    )
    assert session.resolve_callback_url("https://router.internal/sessions/s1") == (
        "https://router.internal/sessions/s1"
    )


def test_the_abort_path_can_name_a_dict_request() -> None:
    """The loop's request is a dict, and the abort path used to require an attribute.

    `run_with_termination_handling` unpacks the TensorDict before calling `run`,
    so `request.request_id` raised exactly when the gateway had already aborted
    the episode -- turning a deliberate abort into a rollout error that refilled
    the whole group.
    """
    from psrl.workers.agent_loop.loops.harness_agent_loop import _request_id_of

    assert _request_id_of({"uid": 7}) == "7"
    assert _request_id_of({}) == "N/A"


@pytest.mark.asyncio
async def test_codex_disabled_subagents_emit_no_multi_agent_surface() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(
        HarnessConfig(kind="codex", executable="codex", subagents_enabled=False),
        sandbox,
    )

    await harness.prepare(_runtime())

    catalog = json.loads(sandbox.writes["/root/.codex/model_catalog.json"])
    assert catalog["models"][0]["multi_agent_version"] is None
    config = tomllib.loads(sandbox.writes["/root/.codex/config.toml"].decode())
    assert "features" not in config


@pytest.mark.asyncio
async def test_claude_code_disabled_subagents_keep_the_agent_deny_rule() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(
        HarnessConfig(kind="claude_code", executable="claude", subagents_enabled=False),
        sandbox,
    )

    await harness.prepare(_runtime())

    settings = json.loads(sandbox.writes["/root/.claude/settings.json"])
    assert settings["permissions"]["deny"] == ["Agent"]


def test_logical_workspace_isolation_creates_the_agent_root_and_states_the_rule() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(
        HarnessConfig(
            kind="codex",
            executable="codex",
            workspace_isolation="logical",
            agent_workspace_root="/workspace/agents",
        ),
        sandbox,
    )

    import asyncio

    asyncio.run(harness.prepare(_runtime()))
    probes = [command for command, _, _ in sandbox.commands if command.startswith("mkdir -p /root/.psrl-harness")]
    assert probes and "/workspace/agents" in probes[0]
    assert "/testbed" in harness.config.isolate_prompt("/testbed")
    assert "/workspace/agents" in harness.config.isolate_prompt("/testbed")


def test_task_prompt_appends_the_suffix_before_the_isolation_rule() -> None:
    config = HarnessConfig(
        kind="codex",
        executable="codex",
        prompt_suffix="\n\nDelegate one investigation to a subagent.",
        workspace_isolation="logical",
    )

    prompt = config.task_prompt("fix the bug", "/testbed")

    assert prompt.startswith("fix the bug\n\nDelegate one investigation to a subagent.")
    assert prompt.index("Delegate one investigation") < prompt.index("Workspace isolation")
    assert config.task_prompt("fix the bug", "/testbed").count("fix the bug") == 1


def test_task_prompt_without_suffix_or_isolation_is_the_plain_prompt() -> None:
    config = HarnessConfig(kind="codex", executable="codex")

    assert config.task_prompt("fix the bug", "/testbed") == "fix the bug"


def test_logical_workspace_isolation_is_invalid_with_a_relative_root() -> None:
    import pytest

    with pytest.raises(ValueError, match="agent_workspace_root"):
        HarnessConfig(
            kind="codex",
            executable="codex",
            workspace_isolation="logical",
            agent_workspace_root="agents",
        )


def test_claude_episode_reports_live_background_subagents() -> None:
    from psrl.workers.agent_loop.harness.episode import CLAUDE_CODE_STREAM_JSON, episode_summary

    event = {
        "type": "result",
        "num_turns": 2,
        "terminal_reason": "success",
        "subagent_stats": {
            "started_in_background": 2,
            "completed": 1,
            "failed": 0,
            "killed": {"parent": 0, "user": 0, "system": 0},
        },
    }
    summary = episode_summary(CLAUDE_CODE_STREAM_JSON, json.dumps(event).encode())

    assert summary is not None
    assert summary.background_subagents == 1


def test_codex_episode_reports_agents_still_live_at_main_close() -> None:
    from psrl.workers.agent_loop.harness.episode import CODEX_JSONL, episode_summary

    completed = {
        "type": "item.completed",
        "item": {
            "type": "collab_tool_call",
            "tool": "wait",
            "agents_states": {"a": {"status": "completed"}, "b": {"status": "running"}},
        },
    }
    summary = episode_summary(CODEX_JSONL, json.dumps(completed).encode())

    assert summary is not None
    assert summary.background_subagents == 1
