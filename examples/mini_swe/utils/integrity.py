"""Post-rollout integrity checks for sandboxed SWE coding harnesses."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Iterator
from pathlib import PurePosixPath
from typing import Any

_NETWORK_COMMAND_RE = re.compile(
    r"\b(?:curl|wget)\b|\bgit\s+(?:clone|fetch|pull|ls-remote|archive)\b|"
    r"\bgit\s+remote\s+add\b|\bgh\s+(?:api|repo|pr|issue)\b|"
    r"\b(?:pip(?:3)?|python(?:3)?\s+-m\s+pip)\s+install\b.*(?:https?://|git\+https?://)",
    flags=re.IGNORECASE,
)
_WEB_TOOL_MARKERS = (
    "browser",
    "web_search",
    "websearch",
    "search_query",
    "open_url",
    "fetch_url",
    "webfetch",
    "web_fetch",
)
_WRITE_TOOLS = {"edit", "write", "multiedit", "multi_edit", "apply_patch"}
_PROTECTED_BASENAMES = {
    "conftest.py",
    "pytest.ini",
    "tox.ini",
    "setup.cfg",
    "pyproject.toml",
}

# Trajectory output formats the scanner can dispatch on. These mirror the
# `HarnessConfig.trajectory_format` values. `plain_text` intentionally has no parser and is unscannable.
TRAJECTORY_FORMAT_CLAUDE_CODE = "claude_code_stream_json"
TRAJECTORY_FORMAT_CODEX = "codex_jsonl"
TRAJECTORY_FORMAT_PLAIN_TEXT = "plain_text"

TrajectoryParser = Callable[[Any], Iterator[tuple[str, dict[str, Any]]]]


def scan_trajectory_integrity(log_bytes: bytes, trajectory_format: str, repo: str) -> dict[str, Any]:
    """Scan a harness trajectory for repository downloads and protected writes.

    Dispatches on the trajectory output format rather than the harness kind so a
    new harness only needs a parser registered for its format. An unparsable
    trajectory (unknown format, or no valid JSON lines) is reported as
    unscannable but is NOT treated as a violation.
    """
    parser = _TRAJECTORY_PARSERS.get(trajectory_format)
    if parser is None:
        return _unscannable_result(f"No integrity parser for trajectory format {trajectory_format!r}.")

    tool_calls: list[tuple[str, dict[str, Any]]] = []
    parsed_lines = 0
    malformed_lines = 0
    for raw_line in log_bytes.splitlines():
        try:
            payload = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            malformed_lines += 1
            continue
        parsed_lines += 1
        tool_calls.extend(parser(payload))

    if parsed_lines == 0:
        return _unscannable_result(
            "Trajectory log contained no parseable JSON lines.",
            parsed_lines=parsed_lines,
            malformed_lines=malformed_lines,
        )
    return _evaluate_tool_calls(
        tool_calls,
        _normalize_repo_slug(repo),
        parsed_lines=parsed_lines,
        malformed_lines=malformed_lines,
    )


def scan_claude_code_integrity(log_bytes: bytes, repo: str) -> dict[str, Any]:
    """Backwards-compatible wrapper for the Claude Code stream-json scanner."""
    return scan_trajectory_integrity(log_bytes, TRAJECTORY_FORMAT_CLAUDE_CODE, repo)


def _unscannable_result(
    note: str,
    parsed_lines: int = 0,
    malformed_lines: int = 0,
) -> dict[str, Any]:
    """Return an all-clear, non-violating result for an unscannable trajectory."""
    return {
        "violated": False,
        "scannable": False,
        "reasons": [],
        "violations": [],
        "parsed_lines": parsed_lines,
        "malformed_lines": malformed_lines,
        "tool_calls": 0,
        "scan_note": note,
    }


def _evaluate_tool_calls(
    tool_calls: list[tuple[str, dict[str, Any]]],
    repo_slug: str,
    parsed_lines: int,
    malformed_lines: int,
) -> dict[str, Any]:
    """Apply the repository-access and protected-write checks to normalized calls."""
    violations: list[dict[str, str]] = []
    seen_calls: set[tuple[str, str]] = set()

    for name, arguments in tool_calls:
        serialized = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
        call_key = (name.lower(), serialized)
        if call_key in seen_calls:
            continue
        seen_calls.add(call_key)

        if repo_slug and _accesses_task_repository(name, arguments, repo_slug):
            violations.append(
                {
                    "kind": "invalid_tool_call",
                    "reason": "blocked_repo_web_access",
                    "tool": name,
                    "repo": repo_slug,
                    "snippet": _snippet(serialized),
                }
            )
        protected_path = _protected_write_path(name, arguments)
        if protected_path:
            violations.append(
                {
                    "kind": "invalid_protected_write",
                    "reason": "write_to_test_or_harness_path",
                    "tool": name,
                    "path": protected_path,
                    "snippet": _snippet(serialized),
                }
            )

    reasons = sorted({violation["reason"] for violation in violations})
    return {
        "violated": bool(violations),
        "scannable": True,
        "reasons": reasons,
        "violations": violations,
        "parsed_lines": parsed_lines,
        "malformed_lines": malformed_lines,
        "tool_calls": len(seen_calls),
    }


def _iter_claude_code_tool_calls(value: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield Anthropic or OpenAI-style tool calls from a nested JSON event."""
    if isinstance(value, dict):
        if value.get("type") == "tool_use" and value.get("name"):
            arguments = value.get("input")
            yield str(value["name"]), arguments if isinstance(arguments, dict) else {}

        function = value.get("function")
        if isinstance(function, dict) and function.get("name"):
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            yield str(function["name"]), arguments if isinstance(arguments, dict) else {}

        for nested in value.values():
            yield from _iter_claude_code_tool_calls(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_claude_code_tool_calls(nested)


def _iter_codex_tool_calls(value: Any) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield normalized tool calls from a Codex `exec --json` JSONL event.

    Codex emits `item.*` events whose `item.type` identifies the action. The
    parser maps those onto the same `(tool_name, arguments)` shape the shared
    checks consume, so it does not depend on Anthropic/OpenAI tool schemas.
    """
    if isinstance(value, dict):
        event_type = value.get("type")
        if event_type == "command_execution":
            command = value.get("command")
            if command:
                yield "bash", {"command": command}
        elif event_type == "file_change":
            for change in value.get("changes") or []:
                if isinstance(change, dict) and change.get("path"):
                    yield "edit", {"path": change["path"]}
        elif event_type == "web_search":
            yield "web_search", {"query": value.get("query")}
        elif event_type == "mcp_tool_call":
            yield str(value.get("tool") or "mcp_tool_call"), value.get("arguments") or {}

        for nested in value.values():
            yield from _iter_codex_tool_calls(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _iter_codex_tool_calls(nested)


_TRAJECTORY_PARSERS: dict[str, TrajectoryParser] = {
    TRAJECTORY_FORMAT_CLAUDE_CODE: _iter_claude_code_tool_calls,
    TRAJECTORY_FORMAT_CODEX: _iter_codex_tool_calls,
}


def _accesses_task_repository(name: str, arguments: dict[str, Any], repo_slug: str) -> bool:
    text = json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)
    lowered_name = name.lower()
    if lowered_name == "bash" and not _NETWORK_COMMAND_RE.search(text):
        return False
    if lowered_name != "bash" and not any(marker in lowered_name for marker in _WEB_TOOL_MARKERS):
        return False
    escaped = re.escape(repo_slug)
    return bool(
        re.search(
            rf"(?:github\.com[/:]|raw\.githubusercontent\.com/|api\.github\.com/repos/|"
            rf"codeload\.github\.com/|patch-diff\.githubusercontent\.com/raw/){escaped}(?:\.git)?(?:[/#?\s\"']|$)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _protected_write_path(name: str, arguments: dict[str, Any]) -> str:
    lowered_name = name.lower()
    if lowered_name in _WRITE_TOOLS:
        path = _argument_path(arguments)
        return path if _is_protected_path(path) else ""
    if lowered_name != "bash":
        return ""

    command = str(arguments.get("command") or arguments.get("cmd") or "")
    for path in _redirect_paths(command):
        if _is_protected_path(path):
            return path
    for path in _write_command_paths(command):
        if _is_protected_path(path):
            return path
    return ""


def _argument_path(arguments: dict[str, Any]) -> str:
    for key in ("filePath", "file_path", "path", "filepath"):
        if arguments.get(key):
            return str(arguments[key])
    return ""


def _redirect_paths(command: str) -> list[str]:
    paths = [match.group(2) for match in re.finditer(r"(?:^|[^>])>>?\s*(['\"]?)([^'\"\s;&|()<>]+)\1", command)]
    paths.extend(match.group(2) for match in re.finditer(r"\btee(?:\s+-a)?\s+(['\"]?)([^'\"\s;&|()<>]+)\1", command))
    paths.extend(
        match.group(1)
        for match in re.finditer(
            r"\b(?:open|Path)\s*\(\s*['\"]([^'\"]+)['\"].{0,80}"
            r"(?:['\"][wa+]|write_(?:text|bytes))",
            command,
        )
    )
    return paths


def _write_command_paths(command: str) -> list[str]:
    paths: list[str] = []
    for fragment in re.split(r"\s*(?:&&|\|\||;|\n|\|)\s*", command):
        try:
            tokens = shlex.split(fragment)
        except ValueError:
            continue
        if not tokens:
            continue
        executable = PurePosixPath(tokens[0]).name.lower()
        if executable in {"rm", "touch", "truncate"}:
            paths.extend(token for token in tokens[1:] if not token.startswith("-"))
        elif executable in {"cp", "mv"} and len(tokens) >= 3:
            paths.append(tokens[-1])
        elif executable in {"sed", "perl"} and any(token.startswith("-i") for token in tokens[1:]):
            paths.extend(token for token in tokens[1:] if not token.startswith("-"))
    return paths


def _is_protected_path(path: str) -> bool:
    normalized = path.replace("\\", "/").strip().strip("\"'")
    if not normalized or normalized.startswith("/tmp/"):
        return False
    if "/testbed/" in normalized:
        normalized = normalized.split("/testbed/", 1)[1]
    parsed = PurePosixPath(normalized.lstrip("./"))
    lowered_parts = tuple(part.lower() for part in parsed.parts)
    basename = parsed.name.lower()
    return bool(
        any(part in {"test", "tests", "testing", "r2e_tests"} for part in lowered_parts)
        or basename.startswith("test_")
        or basename.endswith("_test.py")
        or basename in _PROTECTED_BASENAMES
    )


def _normalize_repo_slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"^https?://(?:www\.)?github\.com/", "", text)
    text = text.removesuffix(".git")
    match = re.match(r"^([a-z0-9_.-]+)/([a-z0-9_.-]+)(?:[/#?].*)?$", text)
    return f"{match.group(1)}/{match.group(2)}" if match else ""


def _snippet(value: str, limit: int = 240) -> str:
    normalized = " ".join(value.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."
