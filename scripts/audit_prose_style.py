#!/usr/bin/env python3
"""
Audit human-readable source text against the PSRL prose rules.

The audit covers comments, Python docstrings, log messages, and assertion
messages. It intentionally ignores vendored patches, deprecated code, and
documentation because those trees have separate ownership or tooling.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {
    ".jinja",
    ".jinja2",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}
DEFAULT_PATHS = (
    "psrl",
    "tests",
    "unit_tests",
    "examples",
    "scripts",
    "hydra_plugins",
    ".github",
    "setup.py",
    "pyproject.toml",
    ".pre-commit-config.yaml",
)
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "_vendor",
    "deprecated",
    "external",
    "patch",
}
LOGGER_METHODS = {
    "critical",
    "debug",
    "error",
    "exception",
    "info",
    "log",
    "warning",
}
DIRECTIVE_PREFIXES = (
    "#!",
    "# -*-",
    "# coding:",
    "# fmt:",
    "# noqa",
    "# pyright:",
    "# ruff:",
    "# shellcheck",
    "# type:",
    "#SBATCH",
)
MARKER_RE = re.compile(r"^(NOTE|TODO|FIXME|HACK)\(([a-z][a-z0-9_-]*)\):\s+(.+)$")
MARKER_PREFIX_RE = re.compile(r"^(NOTE|TODO|FIXME|HACK)\(")
CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
DOUBLE_SPACE_RE = re.compile(r"\S {2,}\S")
CLAUSE_DASH_RE = re.compile(r"\s-\s")
SECTION_RE = re.compile(r"^[-=*_#]{3,}(?:\s+\S.*)?$")
INLINE_CODE_RE = re.compile(r"`+[^`\n]+`+")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    code: str
    message: str


@dataclass(frozen=True)
class Comment:
    line: int
    text: str
    full_line: bool


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def _requires_assert_message(path: Path) -> bool:
    try:
        relative_path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        relative_path = path
    return bool(
        relative_path.parts and relative_path.parts[0] == "psrl" and not relative_path.name.startswith("test_")
    )


def iter_source_files(paths: Iterable[Path]) -> Iterator[Path]:
    seen: set[Path] = set()
    for path in paths:
        if not path.exists() or _is_excluded(path):
            continue
        candidates = (path,) if path.is_file() else path.rglob("*")
        for candidate in candidates:
            if (
                candidate.is_file()
                and candidate.suffix in SUPPORTED_SUFFIXES
                and not _is_excluded(candidate)
                and candidate not in seen
            ):
                seen.add(candidate)
                yield candidate


def _is_directive(comment: str) -> bool:
    stripped = f"#{comment.lstrip()}"
    return stripped.startswith(DIRECTIVE_PREFIXES)


def _python_comments(source: str) -> list[Comment]:
    comments: list[Comment] = []
    readline = io.StringIO(source).readline
    try:
        tokens = tokenize.generate_tokens(readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            prefix = token.line[: token.start[1]]
            text = token.string[1:].strip()
            if not _is_directive(text):
                comments.append(
                    Comment(
                        line=token.start[0],
                        text=text,
                        full_line=not prefix.strip(),
                    )
                )
    except (IndentationError, tokenize.TokenError):
        return comments
    return comments


def _hash_comment(line: str) -> tuple[str, bool] | None:
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if quote is not None:
            if character == quote:
                quote = None
            continue
        if character in {'"', "'"}:
            quote = character
            continue
        if character == "#" and (index == 0 or line[index - 1].isspace()):
            text = line[index + 1 :].strip()
            if _is_directive(text):
                return None
            return text, not line[:index].strip()
    return None


def _hash_comments(source: str) -> list[Comment]:
    comments: list[Comment] = []
    for line_number, line in enumerate(source.splitlines(), start=1):
        match = _hash_comment(line)
        if match is not None:
            text, full_line = match
            comments.append(Comment(line_number, text, full_line))
    return comments


def _jinja_comments(source: str) -> list[Comment]:
    comments: list[Comment] = []
    for match in re.finditer(r"\{#-?(.*?)-?#\}", source, flags=re.DOTALL):
        start_line = source.count("\n", 0, match.start()) + 1
        for offset, text in enumerate(match.group(1).splitlines()):
            comments.append(Comment(start_line + offset, text.strip(), True))
    return comments


def _static_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(value.value if isinstance(value, ast.Constant) else "<value>" for value in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _docstring_nodes(tree: ast.AST) -> Iterator[tuple[int, str]]:
    node_types = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, node_types) or not node.body:
            continue
        expression = node.body[0]
        if not isinstance(expression, ast.Expr):
            continue
        text = _static_string(expression.value)
        if text is not None:
            yield expression.lineno, text


def _runtime_messages(tree: ast.AST) -> Iterator[tuple[int, str, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and node.msg is not None:
            text = _static_string(node.msg)
            if text is not None:
                yield node.msg.lineno, "assertion", text
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            for argument in node.args:
                text = _static_string(argument)
                if text is not None:
                    yield argument.lineno, "output", text
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in LOGGER_METHODS or not node.args:
            continue
        text = _static_string(node.args[0])
        if text is not None:
            yield node.args[0].lineno, "log", text


def _audit_text(
    path: Path,
    line: int,
    kind: str,
    text: str,
) -> Iterator[Finding]:
    for offset, prose_line in enumerate(text.splitlines()):
        stripped = INLINE_CODE_RE.sub("<code>", prose_line.strip())
        if not stripped:
            continue
        for character, label in ((";", "semicolon"), ("—", "em dash"), ("–", "en dash")):
            if character in stripped:
                yield Finding(path, line + offset, "PS001", f"{kind} contains a banned {label}.")
        if CLAUSE_DASH_RE.search(stripped) and not SECTION_RE.fullmatch(stripped):
            yield Finding(path, line + offset, "PS002", f"{kind} contains a clause-joining dash.")
        if kind != "output" and DOUBLE_SPACE_RE.search(stripped):
            yield Finding(path, line + offset, "PS003", f"{kind} contains repeated spaces.")
        if CHINESE_RE.search(stripped):
            yield Finding(path, line + offset, "PS004", f"{kind} contains Chinese text.")


def _comment_blocks(comments: list[Comment], source: str) -> Iterator[list[Comment]]:
    full_line_comments = [comment for comment in comments if comment.full_line and comment.text]
    lines = source.splitlines()
    block: list[Comment] = []
    for comment in full_line_comments:
        gap_has_code = block and any(line.strip() for line in lines[block[-1].line : comment.line - 1])
        if gap_has_code:
            yield block
            block = []
        block.append(comment)
    if block:
        yield block


def _is_exempt_block(block: list[Comment]) -> bool:
    text = " ".join(comment.text for comment in block)
    lowered = text.lower()
    if any(term in lowered for term in ("copyright", "licensed under", "spdx-license")):
        return True
    return all(SECTION_RE.fullmatch(comment.text) for comment in block)


def _audit_comment_structure(path: Path, comments: list[Comment], source: str) -> Iterator[Finding]:
    for comment in comments:
        if MARKER_PREFIX_RE.match(comment.text) and not MARKER_RE.match(comment.text):
            yield Finding(path, comment.line, "PS005", "Annotation marker has an invalid prefix or message.")

    for block in _comment_blocks(comments, source):
        if _is_exempt_block(block):
            continue
        marker = MARKER_RE.match(block[0].text)
        limit = 3 if marker and marker.group(1) == "NOTE" else 2
        if len(block) > limit:
            yield Finding(
                path,
                block[0].line,
                "PS006",
                f"Comment block has {len(block)} lines. The limit is {limit}.",
            )
        if marker:
            message = marker.group(3)
            first_letter = next((character for character in message if character.isalpha()), "")
            if first_letter and not first_letter.isupper():
                yield Finding(path, block[0].line, "PS005", "Annotation marker message must start uppercase.")
            if not block[-1].text.endswith("."):
                yield Finding(path, block[-1].line, "PS005", "Annotation marker must end with a period.")


def audit_python(path: Path, source: str) -> Iterator[Finding]:
    comments = _python_comments(source)
    for comment in comments:
        yield from _audit_text(path, comment.line, "comment", comment.text)
    yield from _audit_comment_structure(path, comments, source)

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as error:
        yield Finding(path, error.lineno or 1, "PS000", f"Could not parse Python source: {error.msg}.")
        return

    if _requires_assert_message(path):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert) and node.msg is None:
                yield Finding(path, node.lineno, "PS007", "Assertion must include a diagnostic message.")

    for line, text in _docstring_nodes(tree):
        yield from _audit_text(path, line, "docstring", text)
    for line, kind, text in _runtime_messages(tree):
        yield from _audit_text(path, line, kind, text)


def audit_non_python(path: Path, source: str) -> Iterator[Finding]:
    comments = _jinja_comments(source) if path.suffix in {".jinja", ".jinja2"} else _hash_comments(source)
    for comment in comments:
        yield from _audit_text(path, comment.line, "comment", comment.text)
    yield from _audit_comment_structure(path, comments, source)


def audit_file(path: Path) -> Iterator[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    if path.suffix == ".py":
        yield from audit_python(path, source)
    else:
        yield from audit_non_python(path, source)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Files or directories to audit.")
    parser.add_argument("--summary-only", action="store_true", help="Print only the final audit summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.paths or [Path(path) for path in DEFAULT_PATHS]
    findings = sorted(
        (finding for path in iter_source_files(paths) for finding in audit_file(path)),
        key=lambda finding: (str(finding.path), finding.line, finding.code),
    )
    if not args.summary_only:
        for finding in findings:
            print(f"{finding.path}:{finding.line}: {finding.code} {finding.message}")
    if findings:
        print(f"Found {len(findings)} prose-style violations in {len({item.path for item in findings})} files.")
        return 1
    print("Prose-style audit passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
