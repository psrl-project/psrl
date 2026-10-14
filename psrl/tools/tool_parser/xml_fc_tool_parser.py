"""
Parse XML function calls into executable shell commands.

The parser accepts SWE-agent-style `<function=NAME>...</function>` calls for
`bash`, `submit`, and `str_replace_editor`.

`parse_xml_fc_to_bash` supports direct mini-SWE-agent integration.
`XmlFcToolParser` returns structured `ToolCall` values.
"""

from __future__ import annotations

import logging
import os
import re

from psrl.tools.base import ToolCall
from psrl.tools.tool_parser.base import ToolParser

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# --- Constants ---

# Submission command injected when the model calls <function=submit>.
_SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"

# `_translate_create` adds a random suffix if this uncommon delimiter collides with file content.
_HEREDOC_DELIM = "PSRL_EOF"

# --- Regex Patterns ---

# Canonical: <function=NAME>\n...\n</function>
_FN_PATTERN = re.compile(r"<function=([^>]+)>\s*\n?(.*?)\s*</function>", re.DOTALL)

# Parameter extraction: <parameter=KEY>VALUE</parameter>
_PARAM_PATTERN = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL)

# Degraded patterns (fallback when the model doesn't produce canonical XML).
_DEGRADED_PATTERNS = [
    # <bash>CMD</bash>
    re.compile(r"<bash>\s*\n?(.*?)\n?\s*</bash>", re.DOTALL),
    # <bash command="CMD"...> or <bash\ncommand="CMD"\n</command>
    re.compile(r'<bash[^>]*command="(.*?)"', re.DOTALL),
    # ```bash\nCMD\n```
    re.compile(r"```(?:bash)?\s*\n(.*?)\n```", re.DOTALL),
]


# --- Parameter Extraction ---


def _extract_params(fn_body: str) -> dict[str, str]:
    """Extract ``<parameter=KEY>VALUE</parameter>`` pairs from function body."""
    return {m.group(1): m.group(2) for m in _PARAM_PATTERN.finditer(fn_body)}


# --- Function Name Classification ---

# Map function name variants produced during exploration back to canonical tools.
_BASH_NAMES = frozenset(
    {
        "bash",
        "execute_bash",
        "terminal",
        "shell",
        "run",
        "exec",
    }
)

_SUBMIT_NAMES = frozenset(
    {
        "submit",
        "finish",
        "done",
        "complete",
    }
)

_STR_REPLACE_NAMES = frozenset(
    {
        "str_replace_editor",
        "str_replace",
        "edit",
        "editor",
        "file_editor",
        "text_editor",
    }
)

# Patterns for fuzzy matching degraded names.
_STR_REPLACE_PREFIXES = ("str_replace", "str_edit", "str_write", "str_create", "str_view")
_FILE_OP_NAMES = frozenset(
    {
        "file",
        "read_file",
        "read",
        "write",
        "open",
        "view",
        "create_file",
        "create",
        "new_file",
        "write_file",
    }
)


def _classify_function_name(fn_name: str) -> str:
    """
    Classify a (possibly degraded) function name into a canonical tool.

    Returns one of: ``"bash"``, ``"submit"``, ``"str_replace_editor"``, ``"unknown"``.
    """
    name_lower = fn_name.lower().strip()

    # Exact match first.
    if name_lower in _BASH_NAMES:
        return "bash"
    if name_lower in _SUBMIT_NAMES:
        return "submit"
    if name_lower in _STR_REPLACE_NAMES:
        return "str_replace_editor"

    # File operation names → treat as str_replace_editor.
    if name_lower in _FILE_OP_NAMES:
        return "str_replace_editor"

    # Prefix match for str_replace variants (str_replace_text, str_view, etc.).
    for prefix in _STR_REPLACE_PREFIXES:
        if name_lower.startswith(prefix):
            return "str_replace_editor"

    return "unknown"


def _get_path(params: dict[str, str]) -> str:
    """Get file path from parameter dict, trying common key variants."""
    for key in ("path", "file_name", "file", "filename", "file_path"):
        if key in params:
            return params[key].strip()
    return ""


def _get_content(params: dict[str, str]) -> str:
    """Get file content from parameter dict, trying common key variants."""
    for key in ("file_content", "content", "file_text", "new_str", "filecontent"):
        if key in params:
            return params[key]
    return ""


def _get_old_str(params: dict[str, str]) -> str:
    """Get old string for replacement."""
    for key in ("old_str", "old_value", "old", "old_text"):
        if key in params:
            return params[key]
    return ""


def _get_new_str(params: dict[str, str]) -> str:
    """Get new string for replacement."""
    for key in ("new_str", "new_value", "new", "new_text", "new_content"):
        if key in params:
            return params[key]
    return ""


# --- `str_replace_editor` Translation ---


def _translate_create(path: str, content: str) -> str:
    """Translate ``create`` command to a heredoc bash command."""
    # Choose a delimiter that doesn't appear in the content.
    delim = _HEREDOC_DELIM
    if delim in content:
        import hashlib

        suffix = hashlib.md5(content.encode()).hexdigest()[:8]
        delim = f"{_HEREDOC_DELIM}_{suffix}"

    # Ensure parent directory exists.
    dir_part = os.path.dirname(path)
    mkdir_prefix = f"mkdir -p {dir_part} && " if dir_part else ""

    return f"{mkdir_prefix}cat << '{delim}' > {path}\n{content}\n{delim}"


def _translate_view(path: str, params: dict[str, str]) -> str:
    """Translate ``view`` command to cat/sed/find.

    Matches the official SWE-agent ``str_replace_editor`` behavior:
    - If ``path`` is a directory, list non-hidden files up to 2 levels deep.
    - If ``path`` is a file (with optional view_range), show with line numbers.
    """
    view_range = params.get("view_range", "").strip()
    if view_range:
        range_match = re.match(r"\[(\d+),\s*(-?\d+)\]", view_range)
        if range_match:
            start = int(range_match.group(1))
            end = range_match.group(2)
            if end == "-1":
                return f"sed -n '{start},$p' {path} | nl -ba -v {start}"
            else:
                return f"sed -n '{start},{end}p' {path} | nl -ba -v {start}"
    # For directories: list files up to 2 levels deep (matches official SWE-agent).
    # For files: show with line numbers.
    return f"if [ -d {path} ]; then find {path} -maxdepth 2 -not -path '*/\\.*' | head -100; else cat -n {path}; fi"


def _translate_str_replace(path: str, old_str: str, new_str: str) -> str:
    """Translate ``str_replace`` command to a Python one-liner."""
    # Use a Python script for reliable multi-line replacement.
    # repr() handles all escaping for us.
    old_repr = repr(old_str)
    new_repr = repr(new_str)
    return (
        f'python3 -c "\n'
        f"import pathlib, sys\n"
        f"p = pathlib.Path({repr(path)})\n"
        f"content = p.read_text()\n"
        f"old = {old_repr}\n"
        f"new = {new_repr}\n"
        f"if old not in content:\n"
        f"    print('ERROR: old_str not found in file'); sys.exit(1)\n"
        f"p.write_text(content.replace(old, new, 1))\n"
        f"print('Successfully applied edit to ' + {repr(path)})\n"
        f'"'
    )


def _translate_insert(path: str, insert_line: str, new_str: str) -> str:
    """Translate ``insert`` command to a Python one-liner."""
    new_repr = repr(new_str)
    return (
        f'python3 -c "\n'
        f"import pathlib\n"
        f"p = pathlib.Path({repr(path)})\n"
        f"lines = p.read_text().splitlines(True)\n"
        f"new_text = {new_repr}\n"
        f"if not new_text.endswith('\\n'):\n"
        f"    new_text += '\\n'\n"
        f"lines.insert({insert_line}, new_text)\n"
        f"p.write_text(''.join(lines))\n"
        f"print('Inserted text after line {insert_line} in ' + {repr(path)})\n"
        f'"'
    )


def _translate_str_replace_editor(fn_body: str) -> str | None:
    """
    Translate a str_replace_editor function call into a bash command.

    Returns None if the call cannot be translated (malformed parameters).
    """
    params = _extract_params(fn_body)
    command = params.get("command", "").strip()
    path = _get_path(params)

    # Handle a `command` parameter that contains a path.
    # Example: `<function=read_file><parameter=command>/testbed/file.py</parameter></function>`.
    if command.startswith("/") or command.startswith("./"):
        # Infer the operation from the remaining parameters.
        inferred_path = command
        content = _get_content(params)
        if content:
            return _translate_create(inferred_path, content)
        return _translate_view(inferred_path, params)

    if command == "create":
        content = _get_content(params)
        if not path:
            return None
        return _translate_create(path, content)

    if command == "view":
        if not path:
            return None
        return _translate_view(path, params)

    if command == "str_replace":
        old_str = _get_old_str(params)
        new_str = _get_new_str(params)
        if not path or not old_str:
            return None
        return _translate_str_replace(path, old_str, new_str)

    if command == "insert":
        insert_line = params.get("insert_line", "").strip()
        new_str = _get_new_str(params)
        if not path or not insert_line or not new_str:
            return None
        return _translate_insert(path, insert_line, new_str)

    if command == "undo_edit":
        if not path:
            return None
        return f"cd /testbed && git checkout -- {path}"

    # Fall back when path and content are present without a recognized command, as in
    # `<function=file><parameter=filename>X</parameter><parameter=file_content>Y</parameter></function>`.
    if not command and path:
        content = _get_content(params)
        if content:
            return _translate_create(path, content)
        return _translate_view(path, params)

    # Also handle: command is something like "cat" / the file path without /
    if path:
        content = _get_content(params)
        old_str = _get_old_str(params)
        if content:
            return _translate_create(path, content)
        if old_str:
            new_str = _get_new_str(params)
            return _translate_str_replace(path, old_str, new_str)
        # Default to view if we just have a path.
        return _translate_view(path, params)

    # Cannot translate.
    return None


# --- Main Entry Point ---


def parse_xml_fc_to_bash(text: str) -> str | None:
    """
    Parse an XML function-calling model response and return a bash command.

    Canonical `submit`, `bash`, and `str_replace` calls are preferred. Degraded
    patterns provide fallback extraction.

    Args:
        text: Raw model output text.

    Returns:
        A bash command string, or None if no valid action could be extracted.
    """
    # --- 1. Try canonical <function=NAME>...</function> patterns ---
    fn_matches = list(_FN_PATTERN.finditer(text))

    if fn_matches:
        # Take the LAST function call (same heuristic as SWE-agent).
        match = fn_matches[-1]
        fn_name = match.group(1).strip()
        fn_body = match.group(2)

        # Classify the function name into one of the 3 canonical tools.
        tool = _classify_function_name(fn_name)

        if tool == "submit":
            return _SUBMIT_COMMAND

        if tool == "bash":
            params = _extract_params(fn_body)
            cmd = params.get("command", "").strip()
            if cmd:
                return cmd
            # If no command parameter, treat the whole body as the command.
            body_stripped = fn_body.strip()
            if body_stripped:
                return body_stripped
            return None

        if tool == "str_replace_editor":
            result = _translate_str_replace_editor(fn_body)
            if result:
                return result
            # Fall through to degraded patterns.

        # Degraded names such as `<function=file>` may still contain valid parameters.
        if tool == "unknown":
            params = _extract_params(fn_body)
            # If it has a 'command' param that's a bash command, use it.
            cmd = params.get("command", "").strip()
            if cmd and cmd not in ("create", "view", "str_replace", "insert", "undo_edit"):
                return cmd
            # If it has file creation params, treat as create.
            path = _get_path(params)
            content = _get_content(params)
            if path and content:
                return _translate_create(path, content)
            # If it has just a path, treat as view.
            if path and not content:
                return _translate_view(path, params)

    # --- 2. Check for bare <function=submit> without closing tag ---
    if re.search(r"<function=submit>", text):
        return _SUBMIT_COMMAND

    # --- 3. Try degraded patterns ---
    for pattern in _DEGRADED_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            # Take the last match.
            cmd = matches[-1].strip()
            if cmd:
                return cmd

    # Nothing matched.
    return None


# --- Structured Tool Parser ---


@ToolParser.register("xml_fc")
class XmlFcToolParser(ToolParser):
    """
    XML function-calling parser for SWE-agent-LM models.

    Extracts ``<function=NAME>...</function>`` tool calls from model output.
    Unlike the text-level ``parse_xml_fc_to_bash``, this returns structured
    ``ToolCall`` objects for use with the PSRL tool execution pipeline.
    """

    def extract_tool_calls(self, responses_ids: list[int]) -> tuple[str, list[ToolCall]]:
        """Extract tool calls from token IDs."""
        response_str = self.tokenizer.decode(responses_ids)

        fn_matches = list(_FN_PATTERN.finditer(response_str))
        if not fn_matches:
            return response_str, []

        tool_calls = []
        for match in fn_matches:
            fn_name = match.group(1).strip()
            fn_body = match.group(2)
            params = _extract_params(fn_body)

            tool_calls.append(
                ToolCall(
                    name=fn_name,
                    arguments=params,
                )
            )

        # Remaining text (everything outside function tags).
        content = _FN_PATTERN.sub("", response_str).strip()
        return content, tool_calls
