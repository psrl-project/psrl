"""Task-specific helpers shared by sandboxed coding harnesses."""

import logging
import os
import shlex
from pathlib import PurePosixPath

from psrl.sandbox import (
    SandboxCollectorError,
    SandboxCommandTimeout,
    SandboxExitReason,
    SandboxPathNotFoundError,
    SandboxSession,
    SandboxSessionLostError,
    SandboxStatus,
)

psrl_logger = logging.getLogger(__file__)

# Staged inside the sandbox, then fetched through the file API. The exec response is
# capped at `max_exec_output_bytes`, and a patch is user-controlled data that has no
# business going through a diagnostic channel: a truncated patch would grade an
# incomplete solution and report the result as if it were measured.
_HARNESS_PATCH_NAME = "psrl-harness.patch"
# Collection diagnostics, written by the script and read back through the file API.
# The sandbox exec channel now carries stderr on the p3b backend, but the
# self-report also names which tool or directory was missing at each step, which
# a stderr tail does not, and it survives on backends whose exec channel still
# carries no stderr.
_HARNESS_DIAG_NAME = "psrl-harness-collect.diag"
# A repository read by git, not an interactive command, so one generous call is enough
# and every retry cost is the agent's work being wasted.
_PATCH_COLLECT_TIMEOUT_S = 600.0
# Size at which a patch stops being a code diff. The patch is still collected in full,
# because only the grader may judge it, but a repository that the agent filled with
# build output is worth reporting before it reaches the grading container.
_PATCH_SIZE_WARNING_BYTES = 64 * 1024 * 1024
# The collector runs inside the sandbox, so an unbounded `git diff --binary` on a
# file the agent created can exhaust the sandbox's memory and kill the episode (and
# its whole group). Bound the work at the source: skip a file above this size, stop
# once the patch reaches the total budget, and report both in the diagnostics. A
# skipped file is build output or a data artifact far more often than a solution.
_MAX_UNTRACKED_FILE_BYTES = int(
    os.environ.get("PSRL_COLLECTOR_MAX_UNTRACKED_FILE_BYTES", 16 * 1024 * 1024)
)
_MAX_TRACKED_FILE_BYTES = int(
    os.environ.get("PSRL_COLLECTOR_MAX_TRACKED_FILE_BYTES", 32 * 1024 * 1024)
)
_MAX_PATCH_BYTES = int(
    os.environ.get("PSRL_COLLECTOR_MAX_PATCH_BYTES", _PATCH_SIZE_WARNING_BYTES)
)


def build_harness_prompt(problem_statement: str) -> str:
    """Render task and integrity constraints as the harness user prompt.

    The native ``problem_template`` starts the task with ``<pr_description>``.
    Keep that task boundary while leaving Claude Code's stock system prompt
    intact. Integrity constraints are part of the task instruction rather than
    a replacement system policy.
    """
    statement = problem_statement.strip()
    return (
        "<pr_description>\n"
        f"{statement}\n"
        "</pr_description>\n\n"
        "Implement the required changes in the current repository and verify the fix.\n\n"
        "Integrity rules:\n"
        "- Do not modify tests, pytest configuration, or evaluation harness files.\n"
        "- Do not retrieve a solution, patch, commit, or pull request from the task repository or its mirrors.\n"
        "- Do not create nested git repositories (directories containing a .git) inside the working directory; "
        "if you need a scratch repository to reproduce the issue, create it under /tmp instead."
    )


async def collect_git_patch(
    session: SandboxSession,
    workdir: str,
    base_commit: str | None = None,
    *,
    staging_dir: str,
    timeout_s: float = _PATCH_COLLECT_TIMEOUT_S,
) -> str:
    """Collect staged, unstaged, and untracked changes as one git patch.

    The patch is written to a file inside the sandbox and fetched with `read_bytes`,
    never returned as command output. Command output travels through a capped response
    channel, and the cap is a diagnostic safety limit: an oversized patch used to fail
    the episode and abort its whole training group, when the patch itself was complete
    and gradeable. Reading it back as a file keeps the full patch intact.

    ``staging_dir`` is where the patch and the collector's self-report are written. It is
    the harness's own state directory rather than a world-writable convention like
    ``/tmp``, for two reasons: the harness prompt tells the agent to use ``/tmp`` as its
    scratch space, so a framework payload staged there competes with agent-controlled
    files; and a sandbox whose ``/tmp`` is gone - which happens on a node under enough
    pressure that the container's filesystem is disturbed - reported a collectable patch
    as a failed collection, costing the episode and then its whole group. The script
    re-asserts the directory either way, so a directory the agent removed costs nothing.

    Robust against nested git repositories that an agent may have created inside
    the workdir (e.g. ``test_repo/`` used to reproduce a DVC bug): a nested repo
    without a checked-out commit makes ``git add -A`` fail with
    ``error: 'test_repo/' does not have a commit checked out``. This collector
    never stages anything. It emits ``git diff --cached`` (index vs base/HEAD)
    plus ``git diff`` (worktree vs index) for tracked changes, then adds
    untracked files as ``new file`` diffs while excluding nested repositories.
    """
    patch_path = str(PurePosixPath(staging_dir) / _HARNESS_PATCH_NAME)
    diag_path = str(PurePosixPath(staging_dir) / _HARNESS_DIAG_NAME)
    base_arg = shlex.quote(base_commit) if base_commit else ""
    patch_arg = shlex.quote(patch_path)
    diag_arg = shlex.quote(diag_path)
    staging_arg = shlex.quote(staging_dir)
    workdir_arg = shlex.quote(workdir)
    # Per-file bodies, executed by `sh` via `xargs -0`. Both bound the work before
    # invoking Git: an untracked artifact or a large modified file is skipped with
    # a diagnostic line, and the whole patch stops at the total budget. This is
    # what keeps a collector from exhausting a sandbox whose memory was priced for
    # the agent, and it makes the skipped files visible instead of an OOM.
    tracked_body = (
        "for f do\n"
        '    [ -z "$f" ] && continue\n'
        '    size=$(wc -c < "$f" 2>/dev/null || echo 0)\n'
        '    if [ "${size:-0}" -gt "$max_file" ]; then\n'
        '        echo "skipped_tracked=$f size=$size" >> "$diag"\n'
        '        continue\n'
        "    fi\n"
        '    out_size=$(wc -c < "$out" 2>/dev/null || echo 0)\n'
        '    if [ "$out_size" -ge "$max_patch" ]; then\n'
        '        echo "patch_capped=1" >> "$diag"\n'
        "        break\n"
        "    fi\n"
        '    if [ -n "$base" ]; then\n'
        '        git diff $cached --binary --submodule=diff --no-renames "$base" -- "$f" >> "$out" 2>>"$diag" || true\n'
        "    else\n"
        '        git diff $cached --binary --submodule=diff --no-renames -- "$f" >> "$out" 2>>"$diag" || true\n'
        "    fi\n"
        '    out_size=$(wc -c < "$out" 2>/dev/null || echo 0)\n'
        '    if [ "$out_size" -ge "$max_patch" ]; then\n'
        '        echo "patch_capped=1" >> "$diag"\n'
        "        break\n"
        "    fi\n"
        "done\n"
    )
    # Per-file body for the untracked pass. It is executed by `sh` (via `xargs -0`),
    # not by a `while read -d` loop: the sandbox runs commands with `/bin/sh`, and
    # dash's `read` has no `-d`, so the bash form silently skipped every untracked
    # file while still exiting 0.
    untracked_body = (
        "for f do\n"
        '    [ -z "$f" ] && continue\n'
        '    case "$f" in */) continue ;; esac\n'
        "    skip=0\n"
        '    d=$(dirname "$f")\n'
        '    while [ "$d" != "." ] && [ "$d" != "/" ]; do\n'
        '        if [ -e "$d/.git" ]; then skip=1; break; fi\n'
        '        d=$(dirname "$d")\n'
        "    done\n"
        '    if [ "$skip" = 1 ]; then continue; fi\n'
        '    size=$(wc -c < "$f" 2>/dev/null || echo 0)\n'
        '    if [ "${size:-0}" -gt "$max_file" ]; then\n'
        '        echo "skipped_untracked=$f size=$size" >> "$diag"\n'
        '        continue\n'
        "    fi\n"
        '    out_size=$(wc -c < "$out" 2>/dev/null || echo 0)\n'
        '    if [ "$out_size" -ge "$max_patch" ]; then\n'
        '        echo "patch_capped=1" >> "$diag"\n'
        "        break\n"
        "    fi\n"
        # `--no-index` exits 1 when the files differ, which is the normal case here.
        '    git diff --no-index --binary /dev/null -- "$f" >> "$out" 2>>"$diag" || true\n'
        '    out_size=$(wc -c < "$out" 2>/dev/null || echo 0)\n'
        '    if [ "$out_size" -ge "$max_patch" ]; then\n'
        '        echo "patch_capped=1" >> "$diag"\n'
        "        break\n"
        "    fi\n"
        "done\n"
    )
    script = (
        "set -u\n"
        f"out={patch_arg}\n"
        f"diag={diag_arg}\n"
        f"base={base_arg}\n"
        f"max_tracked={_MAX_TRACKED_FILE_BYTES}\n"
        f"max_untracked={_MAX_UNTRACKED_FILE_BYTES}\n"
        f"max_patch={_MAX_PATCH_BYTES}\n"
        "cached=--cached\n"
        "max_file=$max_tracked\n"
        "export out diag base cached max_file max_patch\n"
        # The staging directory belongs to the harness, but nothing here may assume the
        # step that created it still ran or still holds: the agent has a shell in this
        # container too. One builtin buys independence from any directory it removed.
        f"mkdir -p {staging_arg}\n"
        ': > "$out"\n'
        ': > "$diag"\n'
        # Self-report so a non-zero exit is diagnosable without an stderr channel.
        "{\n"
        '    echo "shell=$0"\n'
        '    echo "cwd=$(pwd)"\n'
        f'    echo "wanted_cwd={workdir}"\n'
        f'    echo "staging_dir={staging_dir}"\n'
        '    echo "base=${base:-<none>}"\n'
        '    echo "max_tracked=$max_tracked"\n'
        '    echo "max_untracked=$max_untracked"\n'
        '    echo "max_patch=$max_patch"\n'
        "    for tool in git wc dirname xargs tr grep; do\n"
        '        printf "%s=%s\\n" "$tool" "$(command -v "$tool" 2>/dev/null || echo MISSING)"\n'
        "    done\n"
        f"    ls -ld {workdir_arg} {staging_arg} 2>&1 || true\n"
        '} >> "$diag" 2>&1\n'
        # staged changes (index vs base / HEAD), one bounded diff per file
        'git diff $cached --name-only -z --no-renames --diff-filter=ACMRTUXB ${base:+$base --} '
        '2>>"$diag" | xargs -0 sh -c '
        f"{shlex.quote(tracked_body)} sh\n"
        # unstaged changes (worktree vs index), with no base commit
        "base=\n"
        "cached=\n"
        "export base cached\n"
        'git diff $cached --name-only -z --no-renames --diff-filter=ACMRTUXB '
        '2>>"$diag" | xargs -0 sh -c '
        f"{shlex.quote(tracked_body)} sh\n"
        'echo "untracked=$(git ls-files --others --exclude-standard -z | tr -cd "\\000" | wc -c)" '
        '>> "$diag"\n'
        # untracked files, excluding nested git repositories (agent scratch repos)
        "max_file=$max_untracked\n"
        "export max_file\n"
        'git ls-files --others --exclude-standard -z 2>>"$diag" '
        f"| xargs -0 sh -c {shlex.quote(untracked_body)} sh\n"
        'if grep -q "^patch_capped=1" "$diag" 2>/dev/null; then\n'
        '    echo "collector=capped" >> "$diag"\n'
        "    exit 3\n"
        "fi\n"
        'echo "collector=done" >> "$diag"\n'
        'wc -c < "$out"\n'
    )
    try:
        result = await session.exec(script, cwd=workdir, timeout_s=timeout_s)
    except SandboxSessionLostError as exc:
        # The sandbox is gone, so there is no patch to collect and never will be. Reported
        # as itself rather than as a patch-collection failure: the worker classifies it as
        # an infrastructure fault, which replaces the group without counting the episode
        # against the harness. Raising a bare RuntimeError here instead is what made a
        # released lease look like a broken collector.
        #
        # The runtime's evidence travels on the error: the exit code and finish
        # time, plus the daemon's state text or the inspect failure that made the
        # code unknown. The collector has nothing else to add once the sandbox is
        # gone, so this line is the post mortem's sandbox-side record.
        psrl_logger.warning(
            "Sandbox was lost before the harness patch could be collected "
            "(reason=%s, exit_code=%s, finished_at=%s, state_error=%s), so this "
            "episode has no patch.",
            getattr(exc.exit_reason, "value", exc.exit_reason),
            exc.exit_code if exc.exit_code is not None else "unknown",
            exc.finished_at or "<unknown>",
            exc.state_error or "<none>",
            exc_info=True,
        )
        raise
    except SandboxCommandTimeout as exc:
        # The collector hit its budget with part of the patch already written. Each
        # file's diff is complete on its own, so the partial patch grades the workspace
        # the harness actually left; discarding it would cost a finished episode to the
        # size of the repository rather than to the solution.
        diagnostics, _unavailable = await _read_collector_diagnostics(session, diag_path)
        try:
            partial = await session.read_bytes(patch_path)
        except SandboxPathNotFoundError:
            # The collector stopped before it created the patch file, so it wrote nothing
            # rather than something unreadable.
            partial = b""
        except Exception as read_error:  # noqa: BLE001 - the failure names both causes
            raise SandboxCollectorError(
                f"Harness patch collection timed out after {timeout_s:g}s and the partial "
                f"patch could not be read: {read_error!r}.",
                exit_code=None,
                diagnostics=diagnostics,
            ) from exc
        if not partial.strip():
            raise SandboxCollectorError(
                f"Harness patch collection timed out after {timeout_s:g}s before writing any patch.",
                exit_code=None,
                diagnostics=diagnostics,
            ) from exc
        psrl_logger.warning(
            "Harness patch collection for %r hit its %gs budget; grading the %d bytes written so far.",
            workdir,
            timeout_s,
            len(partial),
        )
        return partial.decode(errors="replace")
    if result.exit_code != 0:
        diagnostics, unavailable = await _read_collector_diagnostics(session, diag_path)
        status, status_error, lost = await _sandbox_status(session)
        probe = await _collector_environment_probe(session, workdir, staging_dir)
        sandbox_id = getattr(session.ref, "sandbox_id", "")
        node_id = getattr(session, "node_id", "")
        alive = status == SandboxStatus.RUNNING.value
        detail = (
            f"Harness patch collection failed (exit {result.exit_code}, cwd={workdir!r}, "
            f"sandbox={sandbox_id!r}, node={node_id!r}, status={status!r}"
            f"{f', status_error={status_error}' if status_error else ''}).\n"
            f"stdout (tail):\n{_tail(result.stdout)}\n"
            f"stderr (tail):\n{_tail(result.stderr)}\n"
            f"collector diagnostics (unavailable={unavailable}):\n{diagnostics}\n"
            f"environment probe:\n{probe}"
        )
        if not alive:
            # Only the sandbox's own liveness decides whether the episode is a
            # lost session. A collector that could not run -- including one whose
            # diagnostics file was never created -- is not evidence that the
            # sandbox is gone; the runtime's answer is. A status that cannot be
            # read is treated as gone, which is the safe direction when the
            # sandbox's state is unknown.
            #
            # If the status call itself reported the loss, its attribution (the
            # service's exit code and finish time) is stronger evidence than
            # anything reconstructed from the collector, so it is carried over.
            reason = (
                lost.exit_reason
                if lost is not None and lost.exit_reason != SandboxExitReason.UNKNOWN
                else (
                    SandboxExitReason.OOM
                    if result.exit_code == 137
                    else SandboxExitReason.CONTAINER_EXITED
                )
            )
            raise SandboxSessionLostError(
                detail,
                sandbox_id=sandbox_id or (lost.sandbox_id if lost is not None else ""),
                exit_reason=reason,
                exit_code=(
                    lost.exit_code
                    if lost is not None and lost.exit_code is not None
                    else result.exit_code
                ),
                state_error=lost.state_error if lost is not None else "",
                finished_at=lost.finished_at if lost is not None else "",
            )
        # The sandbox is alive: its execution environment could not run the
        # collector. This is a collector fault, not a lost session, and calling
        # it a session loss destroys a healthy sandbox and blames the wrong
        # layer. The probe above names the missing tool, directory, or limit.
        raise SandboxCollectorError(
            detail, exit_code=result.exit_code, diagnostics=diagnostics
        )
    try:
        payload = await session.read_bytes(patch_path)
    except Exception as exc:
        raise RuntimeError(f"Could not read the collected harness patch from {patch_path!r}: {exc!r}") from exc
    if len(payload) >= _PATCH_SIZE_WARNING_BYTES:
        psrl_logger.warning(
            "Harness patch for task %r is %d bytes. Grading it in full, but a patch this size is "
            "usually repository build output rather than a solution.",
            workdir,
            len(payload),
        )
    return payload.decode(errors="replace")


async def _sandbox_status(
    session: SandboxSession,
) -> tuple[str, str, SandboxSessionLostError | None]:
    """Return the sandbox's status, why it could not be read, and its loss.

    The status is the only evidence that decides whether a failed collector is
    a collector fault or a lost session. A status call that fails is reported as
    ``unreadable`` rather than guessed at; the caller treats it as gone, which is
    the safe direction when the sandbox's state cannot be established. A status
    call that reports a lost session is returned as that error so its exit code
    and finish time survive into the caller's attribution.
    """
    try:
        status = await session.status()
    except SandboxSessionLostError as exc:
        return "lost", str(exc), exc
    except Exception as exc:  # noqa: BLE001 - an unreadable status is diagnostic
        return "unreadable", repr(exc), None
    return getattr(status, "value", str(status)), "", None


def _tail(text: str | None, limit: int = 2000) -> str:
    """Keep the end of a command's output, where a failure is reported."""
    trimmed = (text or "").strip()
    return trimmed[-limit:] if trimmed else "<empty>"


async def _collector_environment_probe(
    session: SandboxSession, workdir: str, staging_dir: str
) -> str:
    """Report, from inside the sandbox, why the collector could not start.

    The collector writes its diagnostics in its first lines, so a missing file
    means it never reached them -- and its exit code alone does not say which
    precondition failed. This probe asks the sandbox directly: whether the shell
    can run, which tools are on ``PATH``, whether the work and staging
    directories exist, whether the filesystem has room, and what the cgroup and
    host memory pressure are. It is only run after a collector failure, so it
    costs one exec on the failure path and nothing on the success path.
    """
    probe = (
        'echo "cwd=$(pwd)"\n'
        'echo "path=$PATH"\n'
        "id 2>&1 || true\n"
        "ulimit -u 2>&1 || true\n"
        "for tool in sh git wc dirname xargs tr mkdir; do\n"
        '    printf "%s=%s\\n" "$tool" "$(command -v "$tool" 2>/dev/null || echo MISSING)"\n'
        "done\n"
        f"ls -ld {shlex.quote(workdir)} {shlex.quote(staging_dir)} 2>&1 || true\n"
        f"df -h {shlex.quote(workdir)} {shlex.quote(staging_dir)} 2>&1 || true\n"
        "for path in"
        " /sys/fs/cgroup/memory.max"
        " /sys/fs/cgroup/memory/memory.limit_in_bytes"
        " /sys/fs/cgroup/memory.current"
        " /sys/fs/cgroup/memory/memory.usage_in_bytes"
        " /sys/fs/cgroup/pids.current"
        " /sys/fs/cgroup/pids.max; do\n"
        '    [ -r "$path" ] && printf "%s=%s\\n" "$path" "$(cat "$path")"\n'
        "done\n"
        "head -3 /proc/meminfo 2>&1 || true\n"
    )
    try:
        result = await session.exec(probe, cwd=workdir, timeout_s=30)
    except Exception as exc:  # noqa: BLE001 - the probe's failure is the diagnosis
        return f"<probe could not run: {exc!r}>"
    return (
        f"exit={result.exit_code}\n"
        f"stdout:\n{_tail(result.stdout)}\n"
        f"stderr:\n{_tail(result.stderr)}"
    )


async def _read_collector_diagnostics(
    session: SandboxSession, diag_path: str
) -> tuple[str, bool]:
    """Read the collector's self-report after a non-zero exit.

    Returns the text and whether the report was *unavailable* rather than simply
    empty. The second value is diagnostic, not a classification: a missing file
    in a live sandbox says the collector never created it, while a missing
    sandbox is established by ``status()``, not by a file read.

    Read as a file rather than from the exec response because the sandbox exec
    channel does not carry stderr on every backend. Failure to read is itself
    diagnostic, so it is reported inline instead of masking the collection error.
    """
    try:
        payload = await session.read_bytes(diag_path)
    except SandboxPathNotFoundError as exc:
        # A missing file in a live sandbox. The collector creates this file in
        # its first lines, so its absence means the script never got that far.
        return f"<missing: the collector never created its diagnostics file: {exc!r}>", True
    except Exception as exc:  # noqa: BLE001 - any read failure is diagnostic here
        # The file is the only record of which tool the sandbox was missing, so losing it
        # loses the diagnosis. `read_bytes` is one way to reach it and not the only one:
        # fall back to the exec channel, which survives on backends whose file reads do
        # not, and report both failures inline rather than in a log line nobody reads.
        read_error = f"read_bytes failed: {exc!r}"
        try:
            shown = await session.exec(f"cat {shlex.quote(diag_path)}", timeout_s=30)
        except Exception as exec_exc:
            return f"<unavailable: {read_error}; cat failed: {exec_exc!r}>", True
        if shown.exit_code != 0:
            return f"<unavailable: {read_error}; cat exited {shown.exit_code}>", True
        text = (shown.stdout or "").strip()
        return (text or f"<empty; {read_error}>"), not bool(text)
    text = payload.decode(errors="replace").strip()
    return (text or "<empty>"), not bool(text)
