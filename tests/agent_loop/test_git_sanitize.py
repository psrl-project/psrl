"""Unit tests for runtime/bake-time git leak sanitization."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.mini_swe.utils.git_sanitize import (
    PROBE_CLEAN_MARKER,
    PROBE_DIRTY_MARKER,
    PROBE_SCRIPT,
    PROBE_WORKTREE_MARKER,
    _build_purge_script,
    ensure_git_sanitized,
)

_REPO_ROOT = Path(__file__).parents[2]


# --- Shell helpers (run real git in a temp repo) ---


def _git(cwd: Path, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", command], cwd=cwd, capture_output=True, text=True)


def _make_repo(tmp_path: Path, *, leaked_branch: bool, remote: bool) -> Path:
    _git(tmp_path, "git init -q . && git config user.email a@b.c && git config user.name a")
    (tmp_path / "f").write_text("a\n")
    _git(tmp_path, "git add f && git commit -qm c1")
    if leaked_branch:
        _git(tmp_path, "git checkout -q -b future && echo b >> f && git commit -qam c2 && git checkout -q -")
    if remote:
        _git(tmp_path, "git remote add origin https://github.com/foo/bar.git")
    return tmp_path


def test_probe_and_purge_remove_leak_without_touching_worktree(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, leaked_branch=True, remote=True)
    head_before = _git(repo, "git rev-parse HEAD").stdout.strip()

    assert PROBE_DIRTY_MARKER in _git(repo, PROBE_SCRIPT).stdout

    purge = _git(repo, _build_purge_script(None))
    assert purge.returncode == 0

    assert PROBE_CLEAN_MARKER in _git(repo, PROBE_SCRIPT).stdout
    assert _git(repo, "git remote").stdout.strip() == ""
    assert _git(repo, "git for-each-ref --format='%(refname)'").stdout.strip() == ""
    # No reset, no clean: HEAD and the worktree are unchanged.
    assert _git(repo, "git rev-parse HEAD").stdout.strip() == head_before
    assert (repo / "f").read_text() == "a\n"


def test_probe_reports_clean_for_fresh_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, leaked_branch=False, remote=False)

    assert PROBE_CLEAN_MARKER in _git(repo, PROBE_SCRIPT).stdout


def test_probe_reports_not_a_worktree(tmp_path: Path) -> None:
    assert PROBE_WORKTREE_MARKER in _git(tmp_path, PROBE_SCRIPT).stdout


# --- Purge script shape ---


def test_purge_script_never_resets_or_cleans_worktree() -> None:
    script = _build_purge_script(None)

    assert "reset --hard" not in script
    assert "clean -fd" not in script
    assert "clean -ffd" not in script
    for expected in (
        "checkout --detach HEAD",
        "git remote remove",
        "git update-ref -d",
        "reflog expire",
        "gc --prune=now",
    ):
        assert expected in script


def test_purge_script_guards_non_ancestor_base_commit() -> None:
    assert "refs/psrl/base-commit" not in _build_purge_script(None)

    script = _build_purge_script("abc123")
    assert "merge-base --is-ancestor" in script
    assert "refs/psrl/base-commit" in script


# --- ensure_git_sanitized control flow ---


class _FakeExecSession:
    """Minimal async session returning scripted results per exec call."""

    def __init__(self, *results) -> None:
        self._results = list(results)
        self.commands: list[str] = []

    async def exec(self, command, cwd=None, env=None, timeout_s=None):  # noqa: ANN001
        self.commands.append(command)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _result(exit_code: int = 0, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(exit_code=exit_code, stdout=stdout, stderr=stderr)


@pytest.mark.asyncio
async def test_ensure_git_sanitized_skips_clean_repo() -> None:
    session = _FakeExecSession(_result(stdout=f"{PROBE_CLEAN_MARKER}\n"))

    metrics = await ensure_git_sanitized(session, "/testbed")

    assert len(session.commands) == 1
    assert metrics["git_leak_detected"] == 0.0
    assert metrics["git_sanitize_error"] == 0.0
    assert metrics["git_purge_s"] == 0.0


@pytest.mark.asyncio
async def test_ensure_git_sanitized_purges_dirty_repo() -> None:
    session = _FakeExecSession(_result(stdout=f"{PROBE_DIRTY_MARKER}\n"), _result(stdout=""))

    metrics = await ensure_git_sanitized(session, "/testbed", base_commit="deadbeef")

    assert len(session.commands) == 2
    assert "gc --prune=now" in session.commands[1]
    assert "deadbeef" in session.commands[1]
    assert metrics["git_leak_detected"] == 1.0
    assert metrics["git_sanitize_error"] == 0.0


@pytest.mark.asyncio
async def test_ensure_git_sanitized_skips_non_worktree() -> None:
    session = _FakeExecSession(_result(stdout=f"{PROBE_WORKTREE_MARKER}\n"))

    metrics = await ensure_git_sanitized(session, "/testbed")

    assert len(session.commands) == 1
    assert metrics["git_sanitize_error"] == 0.0
    assert metrics["git_leak_detected"] == 0.0


@pytest.mark.asyncio
async def test_ensure_git_sanitized_degrades_on_probe_exception() -> None:
    session = _FakeExecSession(RuntimeError("boom"))

    metrics = await ensure_git_sanitized(session, "/testbed")

    assert metrics["git_sanitize_error"] == 1.0
    assert metrics["git_leak_detected"] == 0.0


@pytest.mark.asyncio
async def test_ensure_git_sanitized_degrades_on_unexpected_marker() -> None:
    session = _FakeExecSession(_result(stdout="totally unexpected output\n"))

    metrics = await ensure_git_sanitized(session, "/testbed")

    assert metrics["git_sanitize_error"] == 1.0


@pytest.mark.asyncio
async def test_ensure_git_sanitized_degrades_on_purge_failure() -> None:
    session = _FakeExecSession(_result(stdout=f"{PROBE_DIRTY_MARKER}\n"), _result(exit_code=1, stderr="nope"))

    metrics = await ensure_git_sanitized(session, "/testbed")

    assert metrics["git_leak_detected"] == 1.0
    assert metrics["git_sanitize_error"] == 1.0


# --- Bake digest parity (bash script <-> runner.py) ---

_BAKE_SCRIPT = _REPO_ROOT / "examples/mini_swe/prepare/docker_scripts/bake_harness_image.sh"
_REBAKE_SCRIPT = _REPO_ROOT / "examples/mini_swe/prepare/docker_scripts/rebake_harness_image.sh"


def _extract_shell_function(path: Path, name: str) -> str:
    """Return the source of ``name() { ... }`` from a shell script."""
    lines = path.read_text().splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith(f"{name}() {{")), None)
    assert start is not None, f"{name}() not found in {path.name}"
    for i in range(start, len(lines)):
        if lines[i] == "}":
            return "\n".join(lines[start : i + 1])
    raise AssertionError(f"unterminated {name}() in {path.name}")


def _run_shell_function(body: str, name: str, argument: str) -> str:
    """Run a shell function definition with ``argument`` as ``$1``."""
    result = subprocess.run(
        ["bash", "-c", f'{body}\n{name} "$1"', "--", argument],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_derivative_digest_matches_bash() -> None:
    from examples.mini_swe.runner import _image_digest

    image = "swebench/swesmith.x86_64.foo:latest"
    # Run the baker's own digest() definition, so the two formulas cannot drift.
    digest_body = _extract_shell_function(_BAKE_SCRIPT, "digest")

    assert _image_digest(image) == _run_shell_function(digest_body, "digest", image)
    # The value is pinned independently of the baker.
    assert _image_digest(image) == hashlib.sha256(image.encode()).hexdigest()[:12]


def test_rebake_tag_matches_runner() -> None:
    from examples.mini_swe.runner import _image_digest

    image = "swebench/swesmith.x86_64.foo:latest"
    tag_body = _extract_shell_function(_REBAKE_SCRIPT, "derivative_tag")

    assert _run_shell_function(tag_body, "derivative_tag", image) == f"psrl/swebench-harness:{_image_digest(image)}"


# --- Trajectory time breakdown includes sandbox init / git keys ---


def _import_agent_loop_base():
    gen_utils = sys.modules.get("psrl.workers.gen.utils")
    if gen_utils is not None:
        for name in ("TokenInput", "TokenOutput"):
            if not hasattr(gen_utils, name):
                setattr(gen_utils, name, type(name, (), {}))
    from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase
    from psrl.workers.agent_loop.loops.utils import TerminateReason

    return AgentLoopBase, TerminateReason


try:
    _AGENT_LOOP_BASE, _TERMINATE_REASON = _import_agent_loop_base()
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover (environment dependent)
    _AGENT_LOOP_BASE, _TERMINATE_REASON, _IMPORT_ERROR = None, None, exc

requires_loop = pytest.mark.skipif(_AGENT_LOOP_BASE is None, reason=f"agent loop import unavailable: {_IMPORT_ERROR}")


@requires_loop
def test_time_breakdown_includes_sandbox_init_and_git_keys() -> None:
    # `_build_summary_text` does not touch `self`, so a bare stand-in is enough.
    loop = SimpleNamespace()
    out = SimpleNamespace(
        agent_reward_info={
            "patch": "",
            "num_turns": 3,
            "timing": {
                "prep_s": 20.0,
                "sandbox_init_s": 4.5,
                "git_probe_s": 0.2,
                "git_purge_s": 4.1,
                "harness_prepare_s": 3.0,
            },
        },
        extra_fields={},
        prompt_ids=[1, 2, 3],
        response_ids=[4],
        response_mask=[1],
        num_turns=3,
    )

    text = _AGENT_LOOP_BASE._build_summary_text(loop, out, _TERMINATE_REASON.FINISHED)

    assert "sandbox_init=4.5s" in text
    assert "git_probe=0.2s" in text
    assert "git_purge=4.1s" in text


@requires_loop
def test_host_mounted_repository_is_skipped() -> None:
    from psrl.workers.agent_loop.loops.mini_swe_harness_agent_loop import MiniSWEHarnessAgentLoop

    bind_mount_task = SimpleNamespace(
        sandbox_spec=SimpleNamespace(workdir="/testbed", mounts=[SimpleNamespace(target="/testbed")]),
    )
    runtime_mount_task = SimpleNamespace(
        sandbox_spec=SimpleNamespace(workdir="/testbed", mounts=[SimpleNamespace(target="/opt/harness")]),
    )

    assert MiniSWEHarnessAgentLoop._uses_host_mounted_repository(bind_mount_task, {}) is True
    assert MiniSWEHarnessAgentLoop._uses_host_mounted_repository(runtime_mount_task, {}) is False
    # Explicit host repo path takes precedence even without a matching mount target.
    assert MiniSWEHarnessAgentLoop._uses_host_mounted_repository(
        runtime_mount_task, {"repo_path": "/host/repo", "use_preexisting_repo": False}
    )
