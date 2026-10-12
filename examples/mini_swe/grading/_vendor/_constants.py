"""Minimal, dependency-free subset of ``swebench.harness.constants``.

Only the symbols the grading driver needs are reproduced here so the vendored
grader runs inside a task image without the ``swebench`` package installed.
Values are byte-for-byte identical to swebench 4.1.0.
"""

from __future__ import annotations

from enum import Enum

# Evaluation result keys (subset).
FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_FAIL = "PASS_TO_FAIL"
KEY_INSTANCE_ID = "instance_id"
KEY_PREDICTION = "model_patch"

# Log sentinels emitted by official SWE-bench eval scripts.
APPLY_PATCH_FAIL = ">>>>> Patch Apply Failed"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

# A log carrying any of these means the run cannot be graded.
BAD_LOG_MARKERS = (APPLY_PATCH_FAIL, RESET_FAILED, TESTS_ERROR, TESTS_TIMEOUT)


class ResolvedStatus(Enum):
    """Instance-level resolution outcome."""

    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


class TestStatus(Enum):
    """Per-test status values recognised by the official log parsers."""

    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


class EvalType(Enum):
    """Whether both pass->pass and fail->pass are graded, or fail-only."""

    PASS_AND_FAIL = "pass_and_fail"
    FAIL_ONLY = "fail_only"
