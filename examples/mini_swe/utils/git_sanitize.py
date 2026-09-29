"""Runtime git-leak probe and fallback purge for SWE task sandboxes.

The primary defense is the bake step
(``examples/mini_swe/prepare/docker_scripts/bake_harness_image.sh``), which
purges leaked git metadata once per task image so every sandbox that starts
from the baked derivative is already clean. This module is the fallback for
images that were not baked: it runs a cheap probe and only pays for the purge
when the image actually leaks.

Nothing here resets the worktree or moves HEAD. It only removes the metadata
that could expose a future fix commit (remotes, refs, reflog, unreachable
objects).
"""

from __future__ import annotations

import logging
import os
import shlex
import time
from typing import Any

from psrl.sandbox import SandboxSession

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

PROBE_WORKTREE_MARKER = "PSRL_GIT_NOT_WORKTREE"
PROBE_CLEAN_MARKER = "PSRL_GIT_CLEAN"
PROBE_DIRTY_MARKER = "PSRL_GIT_DIRTY"

# Cheap, read-only probe. A repository is clean when it has no remotes and no
# commit reachable from any ref or reflog entry that HEAD cannot already reach.
#
# `--count` avoids materialising the commit list and is the single correct
# reachability check. Remotes are checked separately because a remote with no fetched refs is invisible to `rev-list`.
PROBE_SCRIPT = rf"""set -u
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "{PROBE_WORKTREE_MARKER}"; exit 0
fi
if [ -n "$(git remote 2>/dev/null)" ]; then
  echo "{PROBE_DIRTY_MARKER}"; exit 0
fi
leaked="$(git rev-list --count --all --reflog --not HEAD 2>/dev/null || echo 1)"
if [ "${{leaked:-1}}" = "0" ]; then
  echo "{PROBE_CLEAN_MARKER}"
else
  echo "{PROBE_DIRTY_MARKER}"
fi
"""

# Must stay in sync with the purge step in bake_harness_image.sh.
_PURGE_TEMPLATE = """set -euo pipefail
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  exit 0
fi
git -c advice.detachedHead=false checkout --detach HEAD
for remote in $(git remote); do
  git remote remove "$remote"
done
git for-each-ref --format='%(refname)' refs/heads refs/remotes refs/tags refs/stash refs/notes 2>/dev/null \\
  | while IFS= read -r ref; do
      [ -z "$ref" ] && continue
      git update-ref -d "$ref" || true
    done
{keep_base}
git reflog expire --expire=now --expire-unreachable=now --all
git gc --prune=now --quiet
"""


def _build_purge_script(base_commit: str | None) -> str:
    """Build the purge script, keeping a non-ancestor base commit reachable."""
    if base_commit:
        quoted = shlex.quote(base_commit)
        keep_base = (
            f"if ! git merge-base --is-ancestor {quoted} HEAD 2>/dev/null; then\n"
            f"  git update-ref refs/psrl/base-commit {quoted}\n"
            "fi\n"
        )
    else:
        keep_base = ""
    return _PURGE_TEMPLATE.format(keep_base=keep_base)


async def ensure_git_sanitized(
    session: SandboxSession,
    workdir: str,
    base_commit: str | None = None,
    probe_timeout_s: float = 60.0,
    purge_timeout_s: float = 600.0,
) -> dict[str, float]:
    """Probe a task sandbox for leaked git metadata and purge it when present.

    Returns float metrics merged into the rollout ``timing`` breakdown. A probe
    or purge failure is recorded via ``git_sanitize_error`` and degraded rather
    than raised, so a fallback failure never blocks training.
    """
    metrics = {
        "git_probe_s": 0.0,
        "git_purge_s": 0.0,
        "git_leak_detected": 0.0,
        "git_sanitize_error": 0.0,
    }
    if not workdir:
        metrics["git_sanitize_error"] = 1.0
        return metrics

    probe_started = time.perf_counter()
    try:
        probe = await session.exec(PROBE_SCRIPT, cwd=workdir, timeout_s=probe_timeout_s)
    except Exception:
        metrics["git_probe_s"] = time.perf_counter() - probe_started
        metrics["git_sanitize_error"] = 1.0
        psrl_logger.warning("Git leak probe failed for workdir %r, so skipping sanitization.", workdir, exc_info=True)
        return metrics
    metrics["git_probe_s"] = time.perf_counter() - probe_started

    marker = _last_marker(probe.stdout)
    if probe.exit_code != 0 or marker not in (PROBE_CLEAN_MARKER, PROBE_DIRTY_MARKER, PROBE_WORKTREE_MARKER):
        metrics["git_sanitize_error"] = 1.0
        psrl_logger.warning(
            f"Git leak probe returned an unexpected result for workdir {workdir!r} "
            f"(exit_code={probe.exit_code}, marker={marker!r})."
        )
        return metrics
    if marker != PROBE_DIRTY_MARKER:
        return metrics

    metrics["git_leak_detected"] = 1.0
    purge_started = time.perf_counter()
    try:
        purge = await session.exec(_build_purge_script(base_commit), cwd=workdir, timeout_s=purge_timeout_s)
    except Exception:
        metrics["git_purge_s"] = time.perf_counter() - purge_started
        metrics["git_sanitize_error"] = 1.0
        psrl_logger.warning("Git leak purge failed for workdir %r, so continuing unsanitized.", workdir, exc_info=True)
        return metrics
    metrics["git_purge_s"] = time.perf_counter() - purge_started
    if purge.exit_code != 0:
        metrics["git_sanitize_error"] = 1.0
        psrl_logger.warning(
            f"Git leak purge exited with {purge.exit_code} for workdir {workdir!r}: {purge.stderr.strip()!r}."
        )
    return metrics


def _last_marker(stdout: str) -> str:
    """Return the last non-empty stdout line, used as the probe's result marker."""
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def repository_mount_targets(mounts: Any) -> set[str]:
    """Collect bind-mount targets that could alias a host checkout."""
    targets: set[str] = set()
    for mount in mounts or ():
        target = getattr(mount, "target", None)
        if target:
            targets.add(str(target))
    return targets
