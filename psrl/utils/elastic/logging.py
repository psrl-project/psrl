"""
Collect structured diagnostics for one scaling decision.
"""

import logging
from collections.abc import Callable
from typing import Any


class CycleLog:
    """
    Small structured log adapter for ITL's per-decision diagnostics.
    """

    def __init__(self, emit: Callable[..., None]) -> None:
        self._emit = emit
        self._lines: list[str] = []

    def start(self) -> None:
        self._lines = []

    def section(self, title: str) -> None:
        self._lines.append(f"[{title}]")

    def kv(self, **pairs: Any) -> None:
        self._lines.append(" ".join(f"{key}={value}" for key, value in pairs.items()))

    def note(self, message: str) -> None:
        self._lines.append(message)

    def outcome(self, kind: str, reason: str, **extra: Any) -> None:
        self._lines.append(f"outcome={kind} reason={reason} {extra}")

    def flush(self) -> None:
        if self._lines:
            self._emit("itl_cycle", details=" | ".join(self._lines))


def log_policy_event(
    logger: logging.Logger,
    event: str,
    **kwargs: Any,
) -> None:
    """
    Emit a policy event with deterministically ordered fields.
    """
    fields = " ".join(f"{key}={kwargs[key]!r}" for key in sorted(kwargs))
    logger.info(f"elastic_rm_policy {event} | {fields}")


def log_policy_no_action(
    logger: logging.Logger,
    reason: str,
    diagnostics: list[str],
    detail: list[str],
) -> None:
    """
    Emit the reason and diagnostics for a decision without scaling actions.
    """
    logger.info(f"elastic_rm_policy decision | outcome=no_action reason={reason}")
    if diagnostics:
        fields = "|".join(sorted(diagnostics))
        logger.info(f"elastic_rm_policy no_action | diagnostics={fields}")
    for line in detail:
        logger.info(f"elastic_rm_policy no_action_detail | {line}")
