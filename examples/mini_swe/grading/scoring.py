"""
PSRL scoring rules adapted from `swebench.harness.grading`.

Only the pure resolution functions are reproduced (no `make_test_spec` /
`MAP_REPO_VERSION_TO_SPECS` dependency). `get_logs_eval` is adapted to take a
ready parser callable instead of a `TestSpec` and to support logs that carry no
`>>>>> Start/End Test Output` markers (e.g. SWE-smith and PSRL-generated
SWE-Gym scripts).
"""

from __future__ import annotations

from collections.abc import Callable

from ._vendor._constants import (
    BAD_LOG_MARKERS,
    END_TEST_OUTPUT,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    START_TEST_OUTPUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)

StatusMap = dict[str, str]
EvalReport = dict[str, dict[str, list[str]]]

RESOLVED_FULL = ResolvedStatus.FULL.value


def test_passed(case: str, status_map: StatusMap) -> bool:
    """
    Return whether `case` passed (PASSED or XFAIL), matching the official rule.
    """
    return case in status_map and status_map[case] in (TestStatus.PASSED.value, TestStatus.XFAIL.value)


def get_logs_eval(
    log_text: str,
    parser: Callable[..., StatusMap],
    *,
    markers: bool,
    test_spec: object | None = None,
) -> tuple[StatusMap, bool]:
    """
    Parse an eval log into a `{test_case: status}` map.

    Args:
        log_text: Full stdout/stderr captured from the eval script.
        parser: Repo-specific parser callable `(log, test_spec) -> status_map`.
        markers: When True, only the section between the official
            `Start/End Test Output` sentinels is parsed.
        test_spec: Optional object exposing `instance_id`/`repo` for the few
            parsers (JavaScript, Ruby) that inspect it.

    Returns:
        Tuple of (status map, ok). `ok` is False when the log carries a fatal
        marker or, in marker mode, when the sentinels are missing.
    """
    if any(marker in log_text for marker in BAD_LOG_MARKERS):
        return {}, False

    content = log_text
    if markers:
        if START_TEST_OUTPUT not in log_text or END_TEST_OUTPUT not in log_text:
            return {}, False
        content = log_text.split(START_TEST_OUTPUT, 1)[1].split(END_TEST_OUTPUT, 1)[0]

    status_map = parser(content, test_spec)
    # Mirror the official fallback: some runners emit the test section without
    # the surrounding sentinels captured between them.
    if not status_map and markers:
        status_map = parser(log_text, test_spec)
    return status_map, bool(status_map)


def get_eval_tests_report(
    eval_status_map: StatusMap,
    gold_results: dict[str, list[str]],
    eval_type: EvalType = EvalType.PASS_AND_FAIL,
) -> EvalReport:
    """
    Compare parsed statuses against F2P/P2P expectations (official semantics).
    """

    def check_pass_and_fail(test_case: str, success: list[str], failed: list[str]) -> None:
        if test_passed(test_case, eval_status_map):
            success.append(test_case)
        else:
            failed.append(test_case)

    def check_fail_only(test_case: str, success: list[str], failed: list[str]) -> None:
        if test_case in eval_status_map and eval_status_map[test_case] == TestStatus.FAILED.value:
            failed.append(test_case)
        else:
            success.append(test_case)

    eval_type = EvalType(eval_type)
    check = check_pass_and_fail if eval_type == EvalType.PASS_AND_FAIL else check_fail_only

    report: EvalReport = {
        FAIL_TO_PASS: {"success": [], "failure": []},
        PASS_TO_PASS: {"success": [], "failure": []},
    }
    for test_case in gold_results.get(FAIL_TO_PASS, []):
        check(test_case, report[FAIL_TO_PASS]["success"], report[FAIL_TO_PASS]["failure"])
    for test_case in gold_results.get(PASS_TO_PASS, []):
        check(test_case, report[PASS_TO_PASS]["success"], report[PASS_TO_PASS]["failure"])
    return report


def compute_fail_to_pass(report: EvalReport) -> float:
    """
    Fraction of fail-to-pass tests that passed (1.0 when the set is empty).
    """
    entry = report[FAIL_TO_PASS]
    total = len(entry["success"]) + len(entry["failure"])
    return 1 if total == 0 else len(entry["success"]) / total


def compute_pass_to_pass(report: EvalReport) -> float:
    """
    Fraction of pass-to-pass tests that were maintained (1.0 when the set is empty).
    """
    entry = report[PASS_TO_PASS]
    total = len(entry["success"]) + len(entry["failure"])
    return 1 if total == 0 else len(entry["success"]) / total


def get_resolution_status(report: EvalReport) -> str:
    """
    Return RESOLVED_FULL only when all F2P and all P2P tests hold.
    """
    f2p = compute_fail_to_pass(report)
    p2p = compute_pass_to_pass(report)
    if f2p == 1 and p2p == 1:
        return ResolvedStatus.FULL.value
    if 0 < f2p < 1 and p2p == 1:
        return ResolvedStatus.PARTIAL.value
    return ResolvedStatus.NO.value
