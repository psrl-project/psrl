"""Rethlas harness loop: stage the workspace, run its loop, grade the final proof."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import io
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
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
_DEFAULT_MCP_TIMEOUT = "timeout_seconds: int = 3600"
_RETHLAS_REQUIREMENTS = "agents/generation/mcp/requirements.txt"
_RETHLAS_WHEELHOUSE_PREFIX = "agents/generation/.psrl-wheelhouse"
_RETHLAS_WHEELHOUSE_CACHE_ENV = "PSRL_RETHLAS_WHEELHOUSE_CACHE"
_WORKSPACE_ROOT = "/workspace"
_WORKSPACE_LINK_NAMES = ("results", ".scratch", ".memory")
_WHEELHOUSE_PLATFORM = "manylinux2014_x86_64"
_UNSET = object()

psrl_logger = logging.getLogger("psrl.math_rl.rethlas_loop")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))
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
    # Outbound proxy for package installs inside the sandbox. Separate from the
    # verifier proxy: the verifier may be reachable without one, while PyPI is not.
    sandbox_proxy: str = ""

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
            sandbox_proxy=str(values.get("sandbox_proxy") or ""),
        )


def pack_rethlas(repo_path: str, wheelhouse: str | Path | None = None) -> bytes:
    """Pack a Rethlas checkout for staging, excluding runtime state and VCS data.

    ``wheelhouse`` is an optional directory of wheels for the generation agent's
    MCP runtime. Its files are added under ``agents/generation/.psrl-wheelhouse``
    so the sandbox can create the venv without reaching the package index.
    """
    root = Path(repo_path)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if any(part in _EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.is_file():
                archive.add(path, arcname=str(relative), recursive=False)
        if wheelhouse is not None:
            wheel_root = Path(wheelhouse)
            prefix = PurePosixPath(_RETHLAS_WHEELHOUSE_PREFIX)
            for path in sorted(wheel_root.rglob("*")):
                if path.is_file():
                    archive.add(
                        path,
                        arcname=str(prefix / path.relative_to(wheel_root)),
                        recursive=False,
                    )
    return buffer.getvalue()


def _wheelhouse_key(requirements: bytes, python_version: str) -> str:
    """Return a stable cache key for one requirements file and interpreter version."""
    digest = hashlib.sha256(requirements + b"\0" + python_version.encode("utf-8"))
    return digest.hexdigest()[:16]


def _wheelhouse_download_command(
    requirements: Path, destination: Path, python_version: str
) -> list[str]:
    """Build the pip command that downloads wheels for the sandbox interpreter."""
    abi = f"cp{python_version.replace('.', '')}"
    return [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--python-version",
        python_version,
        "--implementation",
        "cp",
        "--abi",
        abi,
        "--platform",
        _WHEELHOUSE_PLATFORM,
        "-d",
        str(destination),
        "-r",
        str(requirements),
    ]


def build_rethlas_wheelhouse(
    repo_path: str,
    python_version: str,
    cache_root: str | None = None,
) -> Path | None:
    """Download the MCP runtime wheels once per requirements hash.

    The result is a directory the runner can install from with
    ``--no-index --find-links``. ``None`` means the requirements file is missing
    or the download failed, in which case the runner keeps its online fallback.
    """
    repository = Path(repo_path)
    requirements = repository / _RETHLAS_REQUIREMENTS
    if not requirements.is_file():
        return None
    key = _wheelhouse_key(requirements.read_bytes(), python_version)
    root = Path(
        cache_root
        or os.environ.get(_RETHLAS_WHEELHOUSE_CACHE_ENV)
        or Path.home() / ".cache" / "psrl" / "rethlas-wheelhouse"
    )
    destination = root / key
    if (destination / "complete").is_file():
        return destination
    root.mkdir(parents=True, exist_ok=True)
    try:
        lock = open(root / f"{key}.lock", "w")
    except OSError as error:
        psrl_logger.warning(f"Could not open the Rethlas wheelhouse lock in {root}: {error!r}.")
        return None
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "complete").is_file():
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
        try:
            subprocess.run(
                _wheelhouse_download_command(requirements, temporary, python_version),
                check=True,
                capture_output=True,
                text=True,
            )
            (temporary / "complete").write_text("ok\n", encoding="utf-8")
            temporary.rename(destination)
        except (OSError, subprocess.CalledProcessError) as error:
            details = getattr(error, "stderr", "") or ""
            psrl_logger.warning(
                f"Could not build the Rethlas runtime wheelhouse for python {python_version}: "
                f"{error!r} {details[-500:]}"
            )
            shutil.rmtree(temporary, ignore_errors=True)
            return None
    return destination


async def link_workspace_paths(session, repo: str) -> None:
    """Resolve the workspace-root paths named by the Rethlas skills into the checkout.

    The skills write proofs to ``/workspace/results/{problem_id}/`` and keep state
    under ``/workspace/.scratch`` and ``/workspace/.memory``. The episode stages the
    checkout at ``/workspace/rethlas``, so without these links a proof written at
    the documented path lands outside the tree the artifact collector reads. A
    directory already present at a link path is moved aside once, never deleted.
    """
    quoted_repo = shlex.quote(repo)
    targets = " ".join(f"{quoted_repo}/{name}" for name in _WORKSPACE_LINK_NAMES)
    names = " ".join(_WORKSPACE_LINK_NAMES)
    command = (
        f"mkdir -p {targets} {shlex.quote(_WORKSPACE_ROOT)} && "
        f"for name in {names}; do "
        f'link="{_WORKSPACE_ROOT}/$name"; '
        f'target="{quoted_repo}/$name"; '
        'if [ -e "$link" ] && [ ! -L "$link" ]; then '
        'mv "$link" "$link.psrl-image" || exit 1; '
        "fi; "
        'ln -sfn "$target" "$link" || exit 1; '
        "done"
    )
    result = await session.exec(command, timeout_s=60)
    if result.exit_code != 0:
        raise RuntimeError(
            f"Linking the workspace paths into {repo} failed with exit {result.exit_code}: "
            f"{(result.stderr or '')[-500:]}"
        )


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


async def patch_mcp_verifier_url(session, repo: str, url: str, timeout_s: str | int) -> None:
    """Point the generation agent's verification tool at this episode's gateway.

    The tool's own request timeout is widened to the harness tool timeout. A slow
    deployment must not look like a tool failure to the agent while the gateway is
    still waiting on the same proof.
    """
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
    if _DEFAULT_MCP_TIMEOUT not in text:
        raise RuntimeError(
            "The staged Rethlas MCP server no longer carries the expected verification timeout; "
            "the adapter cannot widen it to the episode budget."
        )
    text = text.replace(_DEFAULT_VERIFIER_URL, url, 1)
    text = text.replace(_DEFAULT_MCP_TIMEOUT, f"timeout_seconds: int = {int(timeout_s)}", 1)
    await session.write_bytes(path, text.encode("utf-8"))


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
        self._wheelhouse: Path | None | object = _UNSET
        self._archive: bytes | None = None

    async def _staged_wheelhouse(self) -> Path | None:
        """Build the MCP wheelhouse once per worker and reuse it for every episode."""
        if self._wheelhouse is _UNSET:
            self._wheelhouse = await asyncio.to_thread(
                build_rethlas_wheelhouse,
                self.rethlas.repo_path,
                self.rethlas.venv_python,
            )
        return self._wheelhouse if isinstance(self._wheelhouse, Path) else None

    def _staged_archive(self, wheelhouse: Path | None) -> bytes:
        """Pack the checkout and its runtime once, then reuse the archive."""
        if self._archive is None:
            self._archive = pack_rethlas(self.rethlas.repo_path, wheelhouse=wheelhouse)
        return self._archive

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

        archive = self._staged_archive(await self._staged_wheelhouse())
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
        await link_workspace_paths(session, repo)
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
        if self.rethlas.sandbox_proxy:
            await session.write_bytes(f"{state_dir}/proxy", self.rethlas.sandbox_proxy.encode("utf-8"))

        tool_timeout = self.harness_config.env.get(
            examples.math_rl.rethlas_harness.RETHLAS_TOOL_TIMEOUT_ENV
        ) or "3600"
        await patch_mcp_verifier_url(session, repo, url, tool_timeout)
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
