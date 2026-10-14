"""
mini-SWE-agent failure classification.

Unlike Harbor, mini-SWE-agent's runner already catches everything and reports an
`exit_status` string (see `examples/mini_swe/runner.py`), so classification here is a
lookup rather than text archaeology. The value of routing it through the shared base
class is that the abort-and-retry policy stays in one place.
"""

from __future__ import annotations

from psrl.utils.agent.exceptions import AgentExceptionClassifier
from psrl.workers.agent_loop.loops.utils import TerminateReason

# `exit_status` values produced by `examples/mini_swe/runner.py`.
_EXIT_STATUS_TO_REASON: dict[str, TerminateReason] = {
    # The runner returned normally. Turn/length caps are applied by the loop.
    "": TerminateReason.FINISHED,
    # `PromptOverflowError` was caught. The turns before the overflow are valid.
    "context_exceeded": TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
    # Any other exception. The worker retries, then the group is refilled.
    "error": TerminateReason.ROLLOUT_ERROR,
}


class MiniSweExceptionClassifier(AgentExceptionClassifier):
    """
    Map mini-SWE-agent `exit_status` values onto PSRL terminate reasons.
    """

    def classify_exit_status(self, exit_status: str | None) -> TerminateReason | None:
        """
        Classify a runner `exit_status`.

        Args:
            exit_status (str | None): Value from the runner's result dict.

        Returns:
            TerminateReason | None: Matching reason, or `None` when unrecognised so the
                caller can decide rather than silently retrying.
        """
        if exit_status is None:
            return None
        return _EXIT_STATUS_TO_REASON.get(exit_status)


__all__ = ["MiniSweExceptionClassifier"]
