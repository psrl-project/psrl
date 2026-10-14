#!/usr/bin/env python3
"""
Normalize punctuation in Markdown prose without modifying code or math.

Protected regions:
  - Fenced code blocks with backtick or tilde delimiters of any length.
  - Inline code spans with backtick delimiters of any length.
  - Block and inline math.
  - YAML front matter at the top of the file.

Replacement rules for prose:
  - Convert en dashes to ASCII hyphens for ranges and compounds.
  - Convert em dashes after labels to colons and all remaining em dashes to commas.
  - Convert line-ending semicolons to periods and capitalize the next nonempty line.
  - Convert all other semicolons to commas.

Usage:
    python3 scripts/depunct_docs.py docs/ [docs2/ ...]

The script edits files in place. It prints a per-file change summary.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- Tokenization ---

FENCE_RE = re.compile(r"^(\s*)(```+|~~~+)(.*)$")
DIRECTIVE_INFO_RE = re.compile(r"^\s*\{([\w-]+)\}")

# Rewrite prose-body MyST directives while preserving their opening and closing fences.
PROSE_BODY_DIRECTIVES = {
    "admonition",
    "seealso",
    "note",
    "warning",
    "tip",
    "caution",
    "important",
    "attention",
    "danger",
    "error",
    "hint",
    "figure",  # caption text below options
    "tab-item",
    "tab-set",
    "grid",
    "grid-item",
    "grid-item-card",
}


def _is_prose_body_fence(info_string: str) -> bool:
    m = DIRECTIVE_INFO_RE.match(info_string)
    return bool(m and m.group(1).lower() in PROSE_BODY_DIRECTIVES)


def split_top_level(src: str) -> list[tuple[str, str]]:
    """Split a markdown file into a list of (kind, text) chunks at the
    line-block level. kind is one of: 'frontmatter', 'fence', 'prose'.

    For backtick-fenced MyST directives whose body is prose (admonition,
    seealso, figure, ...), the opening and closing fence markers are emitted
    as 'fence' chunks and the body as a 'prose' chunk so its punctuation
    gets rewritten too.
    """
    chunks: list[tuple[str, str]] = []
    lines = src.split("\n")
    n = len(lines)
    i = 0

    # Optional YAML front-matter at very top.
    if n > 0 and lines[0].rstrip() == "---":
        j = 1
        while j < n and lines[j].rstrip() != "---":
            j += 1
        if j < n:
            chunks.append(("frontmatter", "\n".join(lines[0 : j + 1])))
            i = j + 1

    prose_buf: list[str] = []
    while i < n:
        line = lines[i]
        m = FENCE_RE.match(line)
        if m:
            if prose_buf:
                chunks.append(("prose", "\n".join(prose_buf)))
                prose_buf = []
            _fence_indent, fence_marker, info = m.group(1), m.group(2), m.group(3)
            fence_char = fence_marker[0]
            fence_len = len(fence_marker)
            open_line = line
            body_lines: list[str] = []
            close_line: str | None = None
            i += 1
            while i < n:
                m_close = FENCE_RE.match(lines[i])
                if (
                    m_close
                    and m_close.group(2)[0] == fence_char
                    and len(m_close.group(2)) >= fence_len
                    and m_close.group(3).strip() == ""
                ):
                    close_line = lines[i]
                    i += 1
                    break
                body_lines.append(lines[i])
                i += 1
            if _is_prose_body_fence(info):
                chunks.append(("fence", open_line))
                if body_lines:
                    chunks.append(("prose", "\n".join(body_lines)))
                if close_line is not None:
                    chunks.append(("fence", close_line))
            else:
                all_lines = [open_line, *body_lines]
                if close_line is not None:
                    all_lines.append(close_line)
                chunks.append(("fence", "\n".join(all_lines)))
            continue
        prose_buf.append(line)
        i += 1

    if prose_buf:
        chunks.append(("prose", "\n".join(prose_buf)))
    return chunks


# --- Prose tokenization ---

# Order matters: longest/most specific first.
PROTECT_RE = re.compile(
    r"(?P<bmath>\$\$[\s\S]*?\$\$)"  # block math (multiline)
    r"|(?P<imath>(?<!\\)\$[^\$\n]+?\$)"  # inline math (single line)
    r"|(?P<code>`+[^`\n]+?`+)"  # inline code
)


_PH_RE = re.compile(r"\x00PH(\d+)\x00")


def transform_prose(text: str) -> str:
    """Apply punctuation rewrites to a chunk of prose, protecting math/code.

    Protected spans are replaced with opaque placeholders BEFORE rewriting
    so structural regexes (e.g. heading detection) can still match across
    inline-code spans, then restored verbatim after rewriting.
    """
    placeholders: list[str] = []

    def _save(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"\x00PH{len(placeholders) - 1}\x00"

    masked = PROTECT_RE.sub(_save, text)
    rewritten = _rewrite(masked)

    def _restore(m: re.Match[str]) -> str:
        return placeholders[int(m.group(1))]

    return _PH_RE.sub(_restore, rewritten)


# em-dash that follows a label-like token: heading, list bullet, closing
# inline-code/link/bold. Captured group is kept verbatim.
#
# Plain trailing words are not labels because this punctuation usually marks a parenthetical or contrast.
# A comma reads more naturally in those cases.
LABEL_EMDASH_RE = re.compile(
    r"(?P<label>"
    r"^\s{0,3}#{1,6}\s+[^\n—]*?"  # heading line content
    r"|^\s{0,3}[-*+]\s+[^\n—]*?"  # bullet list item content
    r"|^\s{0,3}\d+\.\s+[^\n—]*?"  # ordered list item content
    r"|^\s*:::+\s*\{[^}\n]+\}[^\n—]*?"  # MyST directive title line
    r"|`[^`\n]+`"  # closing inline code
    r"|\]\([^)\n]+\)"  # closing markdown link
    r"|\*\*[^*\n]+\*\*"  # closing bold
    r")\s+—\s+",
    re.MULTILINE,
)


def _replace_em_after_label(match: re.Match[str]) -> str:
    return f"{match.group('label')}: "


def _capitalize_first(line: str) -> str:
    for k, ch in enumerate(line):
        if ch.isalpha():
            return line[:k] + ch.upper() + line[k + 1 :]
        if ch not in " \t":
            return line
    return line


def _semicolon_newline(match: re.Match[str]) -> str:
    """Replace a line-ending semicolon with a period and capitalize the next nonempty line."""
    prefix_ws = match.group("pre") or ""
    next_line = match.group("next") or ""
    return f".\n{prefix_ws}{_capitalize_first(next_line)}"


def _rewrite(text: str) -> str:
    if not text:
        return text

    # 1. Replace en dashes with ASCII hyphens.
    text = text.replace("–", "-")

    # 2. Replace em dashes after labels with colons, including multiple occurrences per line.
    prev = None
    while prev != text:
        prev = text
        text = LABEL_EMDASH_RE.sub(_replace_em_after_label, text)

    # 3. Replace remaining em dashes with commas.
    text = re.sub(r"\s*—\s*", ", ", text)

    # 4a. Replace line-ending semicolons and capitalize the next paragraph.
    text = re.sub(
        r";[ \t]*\n(?P<pre>[ \t]*)(?P<next>[^\n]*)",
        _semicolon_newline,
        text,
    )
    # 4b. Replace mid-sentence semicolons with commas.
    text = re.sub(r";[ \t]+", ", ", text)
    # 4c. Replace bare semicolons with commas.
    text = text.replace(";", ",")

    # 5. Collapse redundant comma sequences from malformed input.
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r",\s*\.", ".", text)

    return text


# --- Driver ---


def process_file(path: Path) -> dict:
    original = path.read_text(encoding="utf-8")
    chunks = split_top_level(original)
    rebuilt = "\n".join(transform_prose(text) if kind == "prose" else text for kind, text in chunks)

    # Count what changed.
    stats = {
        "em_before": original.count("—"),
        "en_before": original.count("–"),
        "semi_before": original.count(";"),
        "em_after": rebuilt.count("—"),
        "en_after": rebuilt.count("–"),
        "semi_after": rebuilt.count(";"),
        "changed": rebuilt != original,
    }

    if stats["changed"]:
        path.write_text(rebuilt, encoding="utf-8")
    return stats


def iter_markdown(roots: list[Path]):
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".md":
            yield root
            continue
        yield from root.rglob("*.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes but do not write files.",
    )
    args = ap.parse_args()

    total = {"em": 0, "en": 0, "semi": 0, "files_changed": 0, "files_seen": 0}
    for p in sorted(iter_markdown(args.paths)):
        total["files_seen"] += 1
        if args.dry_run:
            original = p.read_text(encoding="utf-8")
            chunks = split_top_level(original)
            rebuilt = "\n".join(transform_prose(text) if k == "prose" else text for k, text in chunks)
            changed = rebuilt != original
            em_d = original.count("—") - rebuilt.count("—")
            en_d = original.count("–") - rebuilt.count("–")
            semi_d = original.count(";") - rebuilt.count(";")
        else:
            stats = process_file(p)
            changed = stats["changed"]
            em_d = stats["em_before"] - stats["em_after"]
            en_d = stats["en_before"] - stats["en_after"]
            semi_d = stats["semi_before"] - stats["semi_after"]

        if changed:
            total["files_changed"] += 1
            total["em"] += em_d
            total["en"] += en_d
            total["semi"] += semi_d
            print(f"  {p}: -{em_d} em-dash, -{en_d} en-dash, -{semi_d} semicolon")

    print()
    print(
        f"Summary: {total['files_changed']}/{total['files_seen']} files changed, "
        f"removed {total['em']} em-dash, {total['en']} en-dash, "
        f"{total['semi']} semicolon."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
