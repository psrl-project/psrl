"""
Conservative rewriting of redundant editable installs.
"""

from __future__ import annotations

import re
import shlex

# Characters that make a shell line more than one command. A line carrying any
# of them is never rewritten: splitting it correctly is not worth the risk.
_SHELL_METACHARACTERS = ";&|<>()`$\\\n"
# Flags safe to keep when the line is otherwise a single editable install of the
# project root. Anything else may also install dependencies, which must keep running verbatim.
_EDITABLE_INSTALL_BENIGN_FLAGS = frozenset(
    {
        "-v",
        "--verbose",
        "-q",
        "--quiet",
        "--no-deps",
        "--no-build-isolation",
        "--no-index",
        "--no-cache-dir",
        "--no-warn-script-location",
        "--use-pep517",
        "--no-use-pep517",
    }
)
_EDITABLE_FLAGS = ("-e", "--editable")


def _is_editable_flag(token: str) -> bool:
    """
    Whether a pip token selects editable mode (`-e`, `-ve`, ...).

    Only bare short-flag bundles qualify, so `-Ceditable-verbose=true` (a
    pip config setting) is not mistaken for one.
    """
    if token in _EDITABLE_FLAGS:
        return True
    if not token.startswith("-") or token.startswith("--"):
        return False
    bundle = token[1:]
    return bool(bundle) and bundle.isalpha() and "e" in bundle and set(bundle) <= set("veq")


def _local_project_target(token: str) -> bool:
    """
    Whether an editable target names the project root (`.` / `.[extra]`).
    """
    return token == "."


def standalone_editable_install(line: str) -> str | None:
    """
    Return the interpreter token when *line* is one editable install of `.`.

    Only a whole-line `python -m pip install [-flags] -e .` qualifies. Lines
    that chain commands (`;` / `&&`) or install anything else are rejected,
    so the rewrite can never drop a dependency install.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or any(ch in stripped for ch in _SHELL_METACHARACTERS):
        return None
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        return None
    if len(tokens) < 5 or re.fullmatch(r"python(?:3(?:\.\d+)?)?", tokens[0]) is None:
        return None
    if tokens[1] != "-m" or tokens[2] != "pip" or tokens[3] != "install":
        return None
    editable = False
    saw_target = False
    index = 4
    while index < len(tokens):
        token = tokens[index]
        if _is_editable_flag(token):
            editable = True
            index += 1
            if index >= len(tokens) or not _local_project_target(tokens[index]):
                return None
            saw_target = True
        elif token in _EDITABLE_INSTALL_BENIGN_FLAGS:
            pass
        else:
            return None
        index += 1
    if not (editable and saw_target):
        return None
    return tokens[0]


def render_eval_script(
    eval_script: str,
    *,
    skip_editable_install: bool,
    probe_path: str,
) -> str:
    """
    Guard an eligible install with the checkout probe.
    """
    if not skip_editable_install:
        return eval_script
    # The probe defaults to its own cwd, which is the directory the pip command
    # would install from, so a script that changes directory is handled without guessing.
    probe = shlex.quote(probe_path)
    rendered: list[str] = []
    for line in eval_script.split("\n"):
        interpreter = standalone_editable_install(line)
        if interpreter is None:
            rendered.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        rendered.append(
            f"{indent}if {interpreter} {probe}; then "
            f'echo "psrl: editable install already present; skipping"; '
            f"else {line.strip()}; fi"
        )
    return "\n".join(rendered)
