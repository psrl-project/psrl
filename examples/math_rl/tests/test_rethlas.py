"""Tests for the Rethlas staging helpers and harness adapter."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from examples.math_rl.rethlas_harness import RethlasCommandHarness
from examples.math_rl.rethlas_loop import episode_id_for, pack_rethlas, patch_mcp_verifier_url, patch_web_search
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

    archive = pack_rethlas(str(repo))
    names = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz").getnames()

    assert "agents/generation/tests/run_example.sh" in names
    assert "agents/generation/mcp/server.py" in names
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
