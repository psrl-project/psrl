"""Unit tests for task-scoped coding harness adapters."""

from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from examples.mini_swe.config import build_runtime_config
from examples.mini_swe.utils.harness_task import build_harness_prompt, collect_git_patch
from omegaconf import OmegaConf
from psrl.sandbox import (
    ExecResult,
    MountSpec,
    ResourceSpec,
    SandboxCapabilities,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SandboxStatus,
)
from psrl.sandbox.backends.docker import DockerBackend, DockerPolicyProfile, DockerSession
from psrl.workers.agent_loop.harness import (
    HarnessCompactionConfig,
    HarnessConfig,
    HarnessRuntime,
    HarnessTaskContext,
    clean_snapshot_compatible,
    create_harness,
)
from psrl.workers.agent_loop.harness.claude_code import ClaudeCodeHarness
from psrl.workers.agent_loop.harness.codex import CodexHarness
from psrl.workers.agent_loop.harness.runtime import executable_path, host_runtime_dir, runtime_mount_spec


class FakeSandbox(SandboxSession):
    """Minimal async data plane with deterministic command behavior."""

    def __init__(
        self,
        *,
        runtime_available: bool = True,
        block_cli: bool = False,
        cli_exit_code: int = 0,
    ) -> None:
        self.runtime_available = runtime_available
        self.block_cli = block_cli
        self.cli_exit_code = cli_exit_code
        self.commands: list[tuple[str, str | None, Mapping[str, str] | None]] = []
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
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        self.commands.append((command, cwd, env))
        # The prepare probe creates the state dir and checks the mounted CLI.
        if command.startswith("mkdir -p /root/.psrl-harness"):
            return ExecResult(0 if self.runtime_available else 1, "", "" if self.runtime_available else "no such file")
        if command.startswith("tail -c"):
            return ExecResult(0, "bounded tail", "")
        if "/opt/harness/bin/" in command and self.block_cli:
            self.cli_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.cli_cancelled = True
                raise
        if "/opt/harness/bin/" in command:
            return ExecResult(self.cli_exit_code, "", "")
        return ExecResult(0, "", "")

    async def read_bytes(self, path: str) -> bytes:
        return self.writes[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.writes[path] = data

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
    result = await harness.run("fix the bug", _runtime())

    assert isinstance(harness, ClaudeCodeHarness)
    assert result.exit_code == 1
    assert result.stderr_tail == "bounded tail"
    assert b"hasCompletedOnboarding" in sandbox.writes["/root/.claude.json"]
    settings = json.loads(sandbox.writes["/root/.claude/settings.json"])
    assert settings["permissions"]["allow"] == ["Bash(*)", "Read(*)", "Edit(*)"]
    assert settings["permissions"]["deny"] == []
    assert "defaultMode" not in settings["permissions"]
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
    assert not any(command.startswith("tail -c") for command, _, _ in sandbox.commands)


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
    diagnostic_result = await harness.collect_diagnostics(result)

    assert diagnostic_result.stdout_tail == "bounded tail"
    assert diagnostic_result.stderr_tail == "bounded tail"
    assert sum(command.startswith("tail -c") for command, _, _ in sandbox.commands) == 2


@pytest.mark.asyncio
async def test_patch_collection_includes_staged_and_untracked_changes() -> None:
    sandbox = FakeSandbox()

    await collect_git_patch(sandbox, "/testbed")

    script, cwd, _ = sandbox.commands[-1]
    assert cwd == "/testbed"
    # Staged (index vs base/HEAD) and unstaged (worktree vs index) tracked changes.
    assert "git diff --cached --binary --submodule=diff --" in script
    assert "git diff --binary --submodule=diff --" in script
    # Untracked files become new-file diffs, but nested repositories are skipped.
    assert "git ls-files --others --exclude-standard" in script
    assert '[ -e "$d/.git" ]' in script
    # The collector never stages, so a nested repo without a commit cannot break it.
    assert "git add" not in script


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
    assert harness.build_env(_runtime())["OPENAI_API_KEY"] == "session-1"
    command = harness.build_command("fix", _runtime())
    assert command[0] == "/opt/harness/bin/codex"
    assert command[1:4] == ("exec", "--skip-git-repo-check", "--json")


@pytest.mark.asyncio
async def test_codex_writes_compaction_threshold_with_context_headroom() -> None:
    sandbox = FakeSandbox()
    harness = create_harness(HarnessConfig(kind="codex", executable="codex"), sandbox)

    await harness.prepare(_runtime(10_240, 7_680))

    codex_config = sandbox.writes["/root/.codex/config.toml"].decode()
    assert "model_context_window = 10240" in codex_config
    assert "model_auto_compact_token_limit = 7680" in codex_config


@pytest.mark.asyncio
async def test_prepare_fails_clearly_when_runtime_tree_is_missing() -> None:
    sandbox = FakeSandbox(runtime_available=False)
    config = HarnessConfig(kind="codex", executable="codex")

    with pytest.raises(RuntimeError, match="unavailable in the sandbox"):
        await create_harness(config, sandbox).prepare(_runtime())

    # The explicit probe is the only setup command. There is no install path.
    assert all("npm" not in command for command, _, _ in sandbox.commands)


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
