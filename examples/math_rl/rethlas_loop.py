"""Rethlas harness loop: stage the workspace, run its loop, grade the final proof."""

from __future__ import annotations

import io
import re
import shlex
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import examples.math_rl.rethlas_harness  # noqa: F401  (registers the harness kind)
from examples.math_rl.agent_loop import (
    _DEFAULT_SOLUTION_CAP,
    MathHarnessAgentLoop,
    MathHarnessArtifact,
    MathHarnessTaskState,
    last_assistant_message,
)
from examples.math_rl.verifier_gateway import VerifierGateway, get_verifier_gateway
from psrl.sandbox import SandboxPathNotFoundError
from psrl.workers.agent_loop.harness import HarnessResult, HarnessRuntime, HarnessTaskContext

_DEFAULT_VERIFIER_URL = "http://127.0.0.1:8091/verify"
_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        ".runs",
        "results",
        ".logs",
        ".scratch",
        ".memory",
        "node_modules",
        ".pytest_cache",
    }
)


@dataclass(frozen=True)
class RethlasSettings:
    """Where the Rethlas checkout lives and how long one episode may run."""

    repo_path: str
    sandbox_repo: str = "/workspace/rethlas"
    max_iterations: int = 3
    venv_python: str = "3.11"
    max_calls_per_episode: int = 4
    max_tokens_per_episode: int = 0
    problem_name: str = "problem"

    @classmethod
    def from_mapping(cls, mapping: Any = None) -> RethlasSettings:
        # The checkout is validated when an episode actually stages it, so a recipe that
        # never selects this loop can still list it without setting the host path.
        values = dict(mapping or {})
        repo_path = str(values.get("repo_path") or "").strip()
        return cls(
            repo_path=repo_path,
            sandbox_repo=str(values.get("sandbox_repo") or cls.sandbox_repo),
            max_iterations=int(values.get("max_iterations", 3)),
            venv_python=str(values.get("venv_python", "3.11")),
            max_calls_per_episode=int(values.get("max_calls_per_episode", 4)),
            max_tokens_per_episode=int(values.get("max_tokens_per_episode", 0)),
            problem_name=str(values.get("problem_name") or "problem"),
        )


def pack_rethlas(repo_path: str) -> bytes:
    """Pack a Rethlas checkout for staging, excluding runtime state and VCS data."""
    root = Path(repo_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in _EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_file():
                archive.add(path, arcname=str(relative), recursive=False)
    return buffer.getvalue()


def episode_id_for(value: Any) -> str:
    """Return a URL-safe episode id for the gateway route."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "")).strip("._-")
    return sanitized[:120] or "episode"


async def patch_web_search(session, repo: str) -> None:
    """Keep the Rethlas loop from enabling live web search on alternating rounds."""
    path = f"{repo}/agents/generation/tests/run_example.sh"
    try:
        raw = await session.read_bytes(path)
    except SandboxPathNotFoundError as error:
        raise RuntimeError(f"The staged Rethlas runner is missing at {path}.") from error
    text = raw.decode("utf-8")
    updated = text.replace(r'web_search=\"live\"', r'web_search=\"disabled\"')
    if updated != text:
        await session.write_bytes(path, updated.encode("utf-8"))


async def patch_mcp_verifier_url(session, repo: str, url: str) -> None:
    """Point the generation agent's verification tool at this episode's gateway."""
    path = f"{repo}/agents/generation/mcp/server.py"
    try:
        raw = await session.read_bytes(path)
    except SandboxPathNotFoundError as error:
        raise RuntimeError(f"The staged Rethlas MCP server is missing at {path}.") from error
    text = raw.decode("utf-8")
    if _DEFAULT_VERIFIER_URL not in text:
        raise RuntimeError(
            "The staged Rethlas MCP server no longer carries the expected local verifier URL; "
            "the adapter cannot redirect it to the gateway."
        )
    await session.write_bytes(path, text.replace(_DEFAULT_VERIFIER_URL, url, 1).encode("utf-8"))


class RethlasHarnessAgentLoop(MathHarnessAgentLoop):
    """Run the Rethlas generate/verify loop and train on its trajectory."""

    def __init__(
        self,
        context,
        harness,
        sandbox_config=None,
        verifier=None,
        rethlas=None,
        runtime_in_image: bool = True,
        solution_cap: int = _DEFAULT_SOLUTION_CAP,
        **kwargs,
    ) -> None:
        super().__init__(
            context=context,
            harness=harness,
            sandbox_config=sandbox_config,
            verifier=verifier,
            runtime_in_image=runtime_in_image,
            solution_cap=solution_cap,
            **kwargs,
        )
        self.rethlas = RethlasSettings.from_mapping(rethlas)
        self._episode_gateways: dict[str, VerifierGateway] = {}

    def _validate_repo(self) -> None:
        repo_path = self.rethlas.repo_path
        if not repo_path:
            raise ValueError("rethlas.repo_path is required when the Rethlas loop runs an episode.")
        if not (Path(repo_path) / "agents" / "generation" / "tests" / "run_example.sh").is_file():
            raise ValueError(
                f"rethlas.repo_path {repo_path!r} does not look like the Rethlas checkout: "
                "agents/generation/tests/run_example.sh is missing."
            )

    async def prepare_harness_sandbox(
        self,
        task: HarnessTaskContext[MathHarnessTaskState],
        session,
        timing: dict[str, float],
    ) -> None:
        self._validate_repo()
        started = time.perf_counter()
        repo = self.rethlas.sandbox_repo
        state_dir = self.harness_config.state_dir()
        problem_name = self.rethlas.problem_name
        await session.exec(f"mkdir -p {shlex.quote(state_dir)} {shlex.quote(repo)}", timeout_s=60)

        archive = pack_rethlas(self.rethlas.repo_path)
        archive_path = f"{state_dir}/rethlas.tar.gz"
        await session.write_bytes(archive_path, archive)
        extract = await session.exec(
            f"tar -xzf {shlex.quote(archive_path)} -C {shlex.quote(repo)}",
            timeout_s=180,
        )
        if extract.exit_code != 0:
            raise RuntimeError(
                f"Staging the Rethlas checkout into {repo} failed with exit {extract.exit_code}: "
                f"{(extract.stderr or '')[-500:]}"
            )

        await session.exec(
            f"mkdir -p {shlex.quote(repo)}/data/psrl {shlex.quote(repo)}/results/psrl/{shlex.quote(problem_name)}",
            timeout_s=60,
        )
        await session.write_bytes(f"{repo}/data/psrl/{problem_name}.md", task.state.problem.encode("utf-8"))

        gateway = await get_verifier_gateway(
            self.verifier_config,
            max_calls_per_episode=self.rethlas.max_calls_per_episode,
            max_tokens_per_episode=self.rethlas.max_tokens_per_episode,
        )
        episode_id = episode_id_for(task.state.uid)
        gateway.register_episode(episode_id, task.state.problem)
        self._episode_gateways[episode_id] = gateway
        url = gateway.episode_url(episode_id)
        await session.write_bytes(f"{state_dir}/verifier_url", url.encode("utf-8"))
        await session.write_bytes(f"{state_dir}/episode_id", episode_id.encode("utf-8"))
        host = urlsplit(url).hostname or ""
        if host:
            await session.write_bytes(f"{state_dir}/gateway_host", host.encode("utf-8"))
        if self.verifier_config.proxy:
            await session.write_bytes(f"{state_dir}/proxy", self.verifier_config.proxy.encode("utf-8"))

        await patch_mcp_verifier_url(session, repo, url)
        await patch_web_search(session, repo)
        timing["rethlas_stage_s"] = time.perf_counter() - started

    async def collect_harness_artifact(
        self,
        task: HarnessTaskContext[MathHarnessTaskState],
        sandbox,
        runtime: HarnessRuntime,
        harness_result: HarnessResult,
        *,
        harness_log: bytes | None = None,
    ) -> MathHarnessArtifact:
        base = f"{self.rethlas.sandbox_repo}/results/psrl/{self.rethlas.problem_name}"
        for name, source in (("blueprint_verified.md", "blueprint_verified"), ("blueprint.md", "blueprint")):
            try:
                raw = await sandbox.read_bytes(f"{base}/{name}")
            except SandboxPathNotFoundError:
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                if len(text) > self.solution_cap:
                    text = text[: self.solution_cap]
                    source = f"{source}_truncated"
                return MathHarnessArtifact(solution=text, source=source, chars=len(text))
        if harness_log:
            fallback = last_assistant_message(harness_log)
            if fallback:
                text = fallback[: self.solution_cap]
                return MathHarnessArtifact(solution=text, source="harness_log", chars=len(text))
        return MathHarnessArtifact(solution="", source="missing", chars=0)

    async def finalize_harness_task(
        self,
        task: HarnessTaskContext[MathHarnessTaskState],
        artifact: MathHarnessArtifact,
        clean_snapshot,
        timing: dict[str, float],
        grading_timeout_s: float | None,
    ) -> dict:
        try:
            return await super().finalize_harness_task(
                task,
                artifact,
                clean_snapshot,
                timing,
                grading_timeout_s,
            )
        finally:
            self._release_episode(task)

    async def close_harness_task(self, task: HarnessTaskContext[MathHarnessTaskState]) -> None:
        self._release_episode(task)

    def _release_episode(self, task: HarnessTaskContext[MathHarnessTaskState]) -> None:
        episode_id = episode_id_for(task.state.uid)
        gateway = self._episode_gateways.pop(episode_id, None)
        if gateway is not None:
            gateway.release_episode(episode_id)
