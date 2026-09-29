"""Task-specific helpers shared by sandboxed coding harnesses."""

import shlex

from psrl.sandbox import SandboxSession


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
    timeout_s: float = 60.0,
) -> str:
    """Collect staged, unstaged, and untracked changes as one git patch.

    Robust against nested git repositories that an agent may have created inside
    the workdir (e.g. ``test_repo/`` used to reproduce a DVC bug): a nested repo
    without a checked-out commit makes ``git add -A`` fail with
    ``error: 'test_repo/' does not have a commit checked out``. This collector
    never stages anything. It emits ``git diff --cached`` (index vs base/HEAD)
    plus ``git diff`` (worktree vs index) for tracked changes, then adds
    untracked files as ``new file`` diffs while excluding nested repositories.
    """
    base_arg = f"{shlex.quote(base_commit)} --" if base_commit else "--"
    script = (
        "set -u\n"
        'out="$(mktemp)"\n'
        # staged changes (index vs base / HEAD)
        f'git diff --cached --binary --submodule=diff {base_arg} >> "$out" 2>/dev/null || true\n'
        # unstaged changes (worktree vs index)
        'git diff --binary --submodule=diff -- >> "$out" 2>/dev/null || true\n'
        # untracked files, excluding nested git repositories (agent scratch repos)
        "git ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do\n"
        '    [ -z "$f" ] && continue\n'
        '    case "$f" in */) continue ;; esac\n'
        "    skip=0\n"
        '    d="$(dirname "$f")"\n'
        '    while [ "$d" != "." ] && [ "$d" != "/" ]; do\n'
        '        if [ -e "$d/.git" ]; then skip=1; break; fi\n'
        '        d="$(dirname "$d")"\n'
        "    done\n"
        '    [ "$skip" = 1 ] && continue\n'
        '    git diff --no-index --binary /dev/null "$f" >> "$out" 2>/dev/null || true\n'
        "done\n"
        'cat "$out"\n'
        'rm -f "$out"\n'
    )
    result = await session.exec(script, cwd=workdir, timeout_s=timeout_s)
    if result.exit_code != 0:
        raise RuntimeError(f"Could not collect harness patch: {result.stderr.strip()}")
    return result.stdout
