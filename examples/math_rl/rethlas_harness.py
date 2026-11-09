"""Command harness adapter for the Rethlas generation loop.

The adapter keeps the Rethlas workspace's own turn loop in charge. It writes the Codex
configuration the PSRL session needs, registers the generation agent's verification tool
as an MCP server that talks to the episode gateway, and installs the runner script the
command harness executes.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath

from psrl.sandbox import SandboxPathNotFoundError
from psrl.workers.agent_loop.harness.base import HarnessConfig, HarnessRuntime
from psrl.workers.agent_loop.harness.command import CommandHarness
from psrl.workers.agent_loop.harness.registry import register_harness

RETHLAS_REPO_ENV = "PSRL_RETHLAS_REPO"
RETHLAS_STATE_ENV = "PSRL_RETHLAS_STATE"
RETHLAS_TOOL_TIMEOUT_ENV = "PSRL_RETHLAS_TOOL_TIMEOUT"

_RUNNER_NAME = "run_rethlas.sh"

_MCP_SECTION = """
[mcp_servers.reasoning_agent]
command = {python}
args = [{server}]
cwd = {cwd}
tool_timeout_sec = {tool_timeout}
"""

_RUNNER = """#!/bin/bash
set -euo pipefail

REPO="${PSRL_RETHLAS_REPO:?PSRL_RETHLAS_REPO is not set}"
STATE="${PSRL_RETHLAS_STATE:?PSRL_RETHLAS_STATE is not set}"

if [ -s "$STATE/proxy" ]; then
    PROXY="$(cat "$STATE/proxy")"
    export HTTP_PROXY="$PROXY" HTTPS_PROXY="$PROXY" http_proxy="$PROXY" https_proxy="$PROXY"
fi
if [ -s "$STATE/gateway_host" ]; then
    GATEWAY_HOST="$(cat "$STATE/gateway_host")"
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$GATEWAY_HOST" no_proxy="${no_proxy:+$no_proxy,}$GATEWAY_HOST"
fi
export VERIFY_PROOF_URL="$(cat "$STATE/verifier_url")"
export VERIFIER_EPISODE_ID="$(cat "$STATE/episode_id")"
export PROBLEM_FILE="data/psrl/problem.md"
export MAX_ITERATIONS="${PSRL_RETHLAS_MAX_ITERATIONS:-3}"
export LLM_MODEL="${PSRL_RETHLAS_MODEL:?PSRL_RETHLAS_MODEL is not set}"
export LLM_REASONING_EFFORT="${PSRL_RETHLAS_EFFORT:-max}"
export LOG_ROOT="$REPO/.logs"
export SCRATCH_ROOT="$REPO/.scratch"
export MEMORY_ROOT="$REPO/.memory"
export HOME="${HOME:-/root}"
export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"

mkdir -p "$REPO/results/psrl/problem" "$LOG_ROOT" "$SCRATCH_ROOT" "$MEMORY_ROOT"

cd "$REPO/agents/generation"
WHEELHOUSE="$REPO/agents/generation/.psrl-wheelhouse"
if [ ! -x .venv/bin/python ]; then
    uv venv --python "${PSRL_RETHLAS_PYTHON:-3.11}" .venv
    if [ -d "$WHEELHOUSE" ]; then
        if ! uv pip install --python .venv/bin/python --no-index \
            --find-links "$WHEELHOUSE" -r mcp/requirements.txt; then
            echo "the staged wheelhouse did not satisfy mcp/requirements.txt; using the package proxy" >&2
            uv pip install --python .venv/bin/python -r mcp/requirements.txt
        fi
    else
        uv pip install --python .venv/bin/python -r mcp/requirements.txt
    fi
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# The package proxy is only for the install above. Codex talks to the in-cluster
# session endpoint and the agent's MCP tool talks to the in-cluster gateway, and a
# public proxy in the environment would send those calls to the wrong host.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

if bash tests/run_example.sh; then
    rc=0
else
    rc=$?
    echo "run_example.sh exited $rc"
    for log in "$LOG_ROOT/psrl/problem/iter"/*.md; do
        [ -f "$log" ] || continue
        echo "===== $log"
        tail -n 120 "$log"
    done
fi
exit "$rc"
"""


@register_harness("rethlas_command")
class RethlasCommandHarness(CommandHarness):
    """Run the Rethlas generation loop inside one sandboxed workspace."""

    def __init__(self, config: HarnessConfig, sandbox) -> None:
        super().__init__(config, sandbox)
        self.repo = str(config.env.get(RETHLAS_REPO_ENV) or "/workspace/rethlas")
        self.state_dir = str(config.env.get(RETHLAS_STATE_ENV) or config.state_dir())

    def build_env(self, runtime: HarnessRuntime) -> dict[str, str]:
        env = dict(super().build_env(runtime))
        env["PSRL_RETHLAS_MODEL"] = runtime.model
        env["PSRL_RETHLAS_EFFORT"] = str(self.config.reasoning_effort or "max")
        return env

    async def _prepare(self, runtime: HarnessRuntime) -> None:
        await super()._prepare(runtime)
        await self._append_mcp_server()
        await self._write_runner()

    async def _append_mcp_server(self) -> None:
        config_dir = PurePosixPath(self.config.home_dir) / ".codex"
        config_path = config_dir / "config.toml"
        try:
            existing = (await self.sandbox.read_bytes(str(config_path))).decode("utf-8")
        except SandboxPathNotFoundError:
            existing = ""
        timeout = str(self.config.env.get(RETHLAS_TOOL_TIMEOUT_ENV) or "3600")
        section = _MCP_SECTION.format(
            python=_toml_string(f"{self.repo}/agents/generation/.venv/bin/python"),
            server=_toml_string(f"{self.repo}/agents/generation/mcp/server.py"),
            cwd=_toml_string(f"{self.repo}/agents/generation"),
            tool_timeout=timeout,
        )
        if "[mcp_servers.reasoning_agent]" in existing:
            return
        await self.sandbox.write_bytes(str(config_path), (existing.rstrip("\n") + "\n" + section).encode("utf-8"))

    async def _write_runner(self) -> None:
        path = str(PurePosixPath(self.state_dir) / _RUNNER_NAME)
        await self.sandbox.write_bytes(path, _RUNNER.encode("utf-8"))
        await self.sandbox.exec(f"chmod +x {shlex.quote(path)}", timeout_s=30)


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
