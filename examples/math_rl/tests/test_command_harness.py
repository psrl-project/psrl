"""Tests for the generic command harness."""

from __future__ import annotations

from psrl.workers.agent_loop.harness.base import HarnessConfig, HarnessRuntime
from psrl.workers.agent_loop.harness.command import CommandHarness


class FakeSession:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []

    async def read_bytes(self, path: str) -> bytes:
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


def test_the_command_is_built_from_the_configured_args():
    harness = CommandHarness(
        HarnessConfig(kind="command", executable="bash", args=("run.sh", "--flag")),
        FakeSession(),
    )
    assert list(harness.build_command("prompt", _runtime())) == ["bash", "run.sh", "--flag"]


def test_without_args_the_prompt_is_passed_to_the_shell():
    harness = CommandHarness(HarnessConfig(kind="command", executable="bash"), FakeSession())
    assert list(harness.build_command("echo hi", _runtime())) == ["bash", "-c", "echo hi"]


async def test_preparation_creates_directories_without_probing_a_runtime_mount():
    session = FakeSession()
    harness = CommandHarness(
        HarnessConfig(kind="command", executable="bash", home_dir="/root"),
        session,
    )
    await harness.prepare(_runtime())
    assert len(session.commands) == 1
    assert session.commands[0].startswith("mkdir -p")
    assert "test -x" not in session.commands[0]


async def test_a_codex_backed_command_writes_the_session_config():
    session = FakeSession()
    harness = CommandHarness(
        HarnessConfig(kind="command", executable="bash", codex_config=True, home_dir="/root"),
        session,
    )
    await harness.prepare(_runtime())
    config = session.files["/root/.codex/config.toml"].decode()
    assert 'model_provider = "psrl"' in config
    assert 'base_url = "http://worker:8000/v1"' in config
    assert 'web_search = "disabled"' in config


def test_the_callback_host_stays_out_of_the_proxy():
    harness = CommandHarness(
        HarnessConfig(kind="command", executable="bash", env={"A": "1"}),
        FakeSession(),
    )
    env = harness.build_env(_runtime())
    assert env["A"] == "1"
    assert "worker" in env["no_proxy"]


def test_a_codex_backed_command_gets_the_session_identity():
    harness = CommandHarness(
        HarnessConfig(kind="command", executable="bash", codex_config=True, home_dir="/root"),
        FakeSession(),
    )
    env = harness.build_env(_runtime())
    assert env["OPENAI_API_KEY"] == "sid"
    assert env["CODEX_HOME"] == "/root/.codex"
    assert env["HOME"] == "/root"
