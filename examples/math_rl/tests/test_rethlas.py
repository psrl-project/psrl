"""Tests for the Rethlas staging helpers and harness adapter."""

from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest
from examples.math_rl.agent_loop import MathHarnessTaskState, stage_reference_material
from examples.math_rl.rethlas_harness import RethlasCommandHarness
from examples.math_rl.rethlas_loop import (
    _wheelhouse_download_command,
    _wheelhouse_key,
    build_rethlas_wheelhouse,
    episode_id_for,
    link_workspace_paths,
    pack_rethlas,
    patch_mcp_verifier_url,
    patch_web_search,
)
from psrl.sandbox import SandboxPathNotFoundError
from psrl.workers.agent_loop.harness.base import HarnessConfig, HarnessRuntime


class FakeSession:
    """A sandbox session that keeps files in memory."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = dict(files or {})
        self.commands: list[str] = []

    async def read_bytes(self, path: str) -> bytes:
        if path not in self.files:
            raise SandboxPathNotFoundError(path)
        return self.files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def exec(self, command: str, timeout_s: float | None = None):
        self.commands.append(command)

        class Result:
            exit_code = 0
            stdout = ""
            stderr = ""

        return Result()


def _runtime() -> HarnessRuntime:
    return HarnessRuntime(
        session_id="sid",
        session_root_url="http://worker:8000",
        workdir="/workspace",
        model="/models/test",
    )


def test_pack_rethlas_skips_vcs_and_run_state(tmp_path: Path):
    repo = tmp_path / "Rethlas"
    (repo / "agents" / "generation" / "tests").mkdir(parents=True)
    (repo / "agents" / "generation" / "tests" / "run_example.sh").write_text("#!/bin/bash\n")
    (repo / "agents" / "generation" / "mcp").mkdir(parents=True)
    (repo / "agents" / "generation" / "mcp" / "server.py").write_text("x = 1\n")
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("gitdir\n")
    (repo / "results" / "psrl").mkdir(parents=True)
    (repo / "results" / "psrl" / "old.md").write_text("stale\n")

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    (wheelhouse / "fastmcp-2.0.0-py3-none-any.whl").write_bytes(b"wheel")

    archive = pack_rethlas(str(repo), wheelhouse=wheelhouse)
    names = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz").getnames()

    assert "agents/generation/tests/run_example.sh" in names
    assert "agents/generation/mcp/server.py" in names
    assert "agents/generation/.psrl-wheelhouse/fastmcp-2.0.0-py3-none-any.whl" in names
    assert not any(name.startswith(".git") for name in names)
    assert not any(name.startswith("results") for name in names)


def test_episode_ids_are_url_safe():
    assert episode_id_for("opp/abc 12") == "opp_abc_12"
    assert episode_id_for("") == "episode"


async def test_the_mcp_server_is_pointed_at_the_gateway():
    path = "/workspace/rethlas/agents/generation/mcp/server.py"
    session = FakeSession(
        {
            path: (
                b'VERIFY_PROOF_URL = "http://127.0.0.1:8091/verify"\n'
                b"def verify_proof_service(statement: str, proof: str, timeout_seconds: int = 3600):\n"
            )
        }
    )
    await patch_mcp_verifier_url(
        session, "/workspace/rethlas", "http://worker:9000/verify/ep-1", "11100"
    )
    assert b"http://worker:9000/verify/ep-1" in session.files[path]
    assert b"8091" not in session.files[path]
    assert b"timeout_seconds: int = 11100" in session.files[path]


async def test_a_missing_default_url_fails_loudly():
    path = "/workspace/rethlas/agents/generation/mcp/server.py"
    session = FakeSession({path: b"VERIFY_PROOF_URL = 'moved'\n"})
    with pytest.raises(RuntimeError, match="expected local verifier URL"):
        await patch_mcp_verifier_url(
            session, "/workspace/rethlas", "http://worker:9000/verify/ep-1", "11100"
        )


async def test_the_workspace_paths_resolve_into_the_staged_checkout():
    session = FakeSession()
    await link_workspace_paths(session, "/workspace/rethlas")
    assert len(session.commands) == 1
    command = session.commands[0]
    assert 'link="/workspace/$name"' in command
    assert 'target="/workspace/rethlas/$name"' in command
    assert "ln -sfn" in command
    assert 'mv "$link" "$link.psrl-image"' in command
    for name in ("results", ".scratch", ".memory"):
        assert name in command


async def test_the_reference_material_is_staged_for_the_prompt():
    session = FakeSession()
    state = MathHarnessTaskState(
        uid="u_1",
        problem="Prove P.",
        solution_path="/workspace/solution.md",
        references=("Bri15 (arXiv:1310.0186)",),
        literature_review="Brion studies the Picard variety.",
        literature_reviewed_at="2026-10-22T00:00:00+00:00",
    )
    await stage_reference_material(session, "/workspace/rethlas/data/psrl/problem.refs", state)
    base = "/workspace/rethlas/data/psrl/problem.refs"
    references = session.files[f"{base}/references.md"].decode()
    literature = session.files[f"{base}/literature.md"].decode()
    assert "Bri15 (arXiv:1310.0186)" in references
    assert "research material" in references
    assert "Brion studies the Picard variety." in literature
    assert "last reviewed at 2026-10-22" in literature
    assert any("mkdir -p" in command for command in session.commands)


async def test_a_task_without_references_stages_nothing():
    session = FakeSession()
    state = MathHarnessTaskState(uid="u_1", problem="Prove P.", solution_path="/workspace/solution.md")
    await stage_reference_material(session, "/workspace/rethlas/data/psrl/problem.refs", state)
    assert session.files == {}
    assert session.commands == []


async def test_a_workspace_link_failure_is_reported():
    class FailingSession(FakeSession):
        async def exec(self, command: str, timeout_s: float | None = None):
            class Result:
                exit_code = 1
                stdout = ""
                stderr = "ln: cannot create symbolic link"

            return Result()

    with pytest.raises(RuntimeError, match="Linking the workspace paths"):
        await link_workspace_paths(FailingSession(), "/workspace/rethlas")


def test_the_wheelhouse_command_targets_the_sandbox_interpreter(tmp_path: Path):
    command = _wheelhouse_download_command(tmp_path / "requirements.txt", tmp_path / "wheels", "3.11")
    assert command[:3] == [sys.executable, "-m", "pip"]
    assert "--only-binary=:all:" in command
    assert command[command.index("--python-version") + 1] == "3.11"
    assert command[command.index("--abi") + 1] == "cp311"
    assert command[command.index("--platform") + 1] == "manylinux2014_x86_64"


def test_the_wheelhouse_key_tracks_requirements_and_python():
    base = _wheelhouse_key(b"fastmcp>=2.0.0\n", "3.11")
    assert base == _wheelhouse_key(b"fastmcp>=2.0.0\n", "3.11")
    assert base != _wheelhouse_key(b"fastmcp>=3.0.0\n", "3.11")
    assert base != _wheelhouse_key(b"fastmcp>=2.0.0\n", "3.12")


def test_a_missing_requirements_file_skips_the_wheelhouse(tmp_path: Path):
    repo = tmp_path / "Rethlas"
    repo.mkdir()
    cache = tmp_path / "cache"
    assert build_rethlas_wheelhouse(str(repo), "3.11", cache_root=str(cache)) is None
    assert not cache.exists()


async def test_web_search_is_disabled_on_every_round():
    path = "/workspace/rethlas/agents/generation/tests/run_example.sh"
    session = FakeSession({path: b'--config "web_search=\\"live\\""\n'})
    await patch_web_search(session, "/workspace/rethlas")
    assert b'web_search=\\"disabled\\"' in session.files[path]


async def test_the_rethlas_harness_writes_the_mcp_section_and_runner():
    config = HarnessConfig(
        kind="rethlas_command",
        executable="bash",
        codex_config=True,
        trajectory_format="plain_text",
        home_dir="/root",
        args=("/root/.psrl-harness/run_rethlas.sh",),
        env={
            "PSRL_RETHLAS_REPO": "/workspace/rethlas",
            "PSRL_RETHLAS_STATE": "/root/.psrl-harness",
        },
    )
    session = FakeSession()
    harness = RethlasCommandHarness(config, session)
    await harness.prepare(_runtime())

    config_toml = session.files["/root/.codex/config.toml"].decode()
    assert 'model_provider = "psrl"' in config_toml
    assert "[mcp_servers.reasoning_agent]" in config_toml
    assert "/workspace/rethlas/agents/generation/mcp/server.py" in config_toml
    assert "tool_timeout_sec = 3600" in config_toml
    runner = session.files["/root/.psrl-harness/run_rethlas.sh"].decode()
    assert "tests/run_example.sh" in runner
    assert "VERIFY_PROOF_URL" in runner
    assert ".psrl-wheelhouse" in runner
    assert "--no-index" in runner
    assert 'PROBLEM_NAME="$(cat "$STATE/problem_name"' in runner
    assert 'PROBLEM_FILE="data/psrl/${PROBLEM_NAME}.md"' in runner
    assert "{{" not in runner
    env = harness.build_env(_runtime())
    assert env["PSRL_RETHLAS_MODEL"] == "/models/test"


async def test_the_verification_tool_timeout_follows_the_episode_budget():
    config = HarnessConfig(
        kind="rethlas_command",
        executable="bash",
        codex_config=True,
        home_dir="/root",
        env={
            "PSRL_RETHLAS_REPO": "/workspace/rethlas",
            "PSRL_RETHLAS_STATE": "/root/.psrl-harness",
            "PSRL_RETHLAS_TOOL_TIMEOUT": "11100",
        },
    )
    session = FakeSession()
    harness = RethlasCommandHarness(config, session)
    await harness.prepare(_runtime())
    config_toml = session.files["/root/.codex/config.toml"].decode()
    assert "tool_timeout_sec = 11100" in config_toml
