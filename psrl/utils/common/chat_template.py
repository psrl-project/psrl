"""
Helpers for resolving custom chat-template values from PSRL configs.

The `gen_actor_rollout_ref.model.custom_chat_template` field can be either
an inline jinja string or a path to a `.jinja` file on disk. This module
centralizes the resolution logic so both the model worker (`HFModelConfig`)
and the agent-loop worker pick up the same patched template.
"""

import os


def resolve_chat_template_value(value: str | None) -> str | None:
    """
    Resolve a `custom_chat_template` config value to its jinja string.

    If `value` points at an existing file, return the file's contents.
    Otherwise return `value` unchanged.

    Args:
        value (str | None): Raw config value. None passes through.

    Returns:
        str | None: Resolved jinja template string, or None.
    """
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value) as f:
            return f.read()
    return value
