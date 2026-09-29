"""Vendored helper functions required by the official log parsers."""

from __future__ import annotations

import re

_ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def ansi_escape(text: str) -> str:
    """Remove ANSI escape sequences from ``text`` (swebench ``utils.ansi_escape``)."""
    return _ANSI_ESCAPE_RE.sub("", text)
