"""Command harness: run a workspace script that drives its own agent runtime."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import PurePosixPath

from psrl.workers.agent_loop.harness.base import Harness, HarnessRuntime
from psrl.workers.agent_loop.harness.codex import CodexHarness


class CommandHarness(Harness):
    """Run one command inside the sandbox and let it own the agent loop.

    The harness serves runtimes that ship their own turn loop, prompt policy and model
    plumbing as a workspace script. The owning loop stages the task through the script's
    environment and files, so this adapter stays agent-agnostic: it prepares the Codex
    configuration when asked, starts the script, and leaves trajectory collection to the
    TITO session that every model call still goes through.
    """

    def config_dir(self) -> str | None:
        """Return the Codex config directory when this command drives Codex."""
        if not self.config.codex_config:
            return None
        return str(PurePosixPath(self.config.home_dir) / ".codex")

    async def prepare(self, runtime: HarnessRuntime) -> None:
        """Create harness directories and write adapter files.

        The base class probes an executable under a mounted runtime tree. A command
        harness runs an image binary against a workspace script, so the probe is skipped
        and the script reports its own failures with its own diagnostics.
        """
        directories = [self._log_dir, self.config_dir()]
        mkdir = "mkdir -p " + " ".join(shlex.quote(d) for d in directories if d)
        await self.sandbox.exec(mkdir, timeout_s=60)
        await self._prepare(runtime)

    async def _prepare(self, runtime: HarnessRuntime) -> None:
        if self.config.codex_config:
            codex = CodexHarness(replace(self.config, kind="codex"), self.sandbox)
            await codex._prepare(runtime)

    def build_command(self, prompt: str, runtime: HarnessRuntime) -> Sequence[str]:
        args = [str(item) for item in self.config.args]
        if args:
            return [self.config.executable, *args]
        return [self.config.executable, "-c", prompt]

    def build_env(self, runtime: HarnessRuntime) -> Mapping[str, str]:
        env = dict(self.config.env)
        env.update(self.inherited_proxy_env())
        if self.config.codex_config:
            # The script's Codex runs against the task-scoped session endpoint, so it
            # needs the same session identity the Codex harness hands its own CLI.
            env["HOME"] = self.config.home_dir
            env["CODEX_HOME"] = str(self.config_dir())
            env["OPENAI_API_KEY"] = runtime.session_id
        return self.callback_no_proxy(runtime, env)
