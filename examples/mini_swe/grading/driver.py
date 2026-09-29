"""
Execute the bundled log parser and write a validated scorecard inside the sandbox.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .parsers import resolve_parser
from .schema import GradingPlan, GradingResult
from .scoring import RESOLVED_FULL, get_eval_tests_report, get_logs_eval, get_resolution_status

_TAIL_CHARS = 2048


def grade_log(payload: dict, log_text: str) -> GradingResult:
    """
    Validate the input contract and grade a complete test log.
    """
    try:
        plan = GradingPlan(
            instance_id=payload["instance_id"],
            repo=payload["repo"],
            eval_script="",
            parser_name=payload["log_parser"],
            f2p=payload["f2p"],
            p2p=payload["p2p"],
            eval_type=payload["eval_type"],
            markers=payload["markers"],
        )
    except (ValueError, KeyError, TypeError) as exc:
        return GradingResult(failure_reason="invalid_plan", error=str(exc))
    expected = {"f2p_total": len(plan.f2p), "p2p_total": len(plan.p2p)}
    try:
        statuses, valid = get_logs_eval(
            log_text,
            resolve_parser(plan.parser_name),
            markers=payload["markers"],
            test_spec=SimpleNamespace(instance_id=plan.instance_id, repo=plan.repo),
        )
    except Exception as exc:
        return GradingResult(
            **expected,
            failure_reason="parser_error",
            parser_error=f"{type(exc).__name__}: {exc}",
            output_tail=log_text[-_TAIL_CHARS:],
        )
    if not valid:
        return GradingResult(**expected, failure_reason="log_unparseable", output_tail=log_text[-_TAIL_CHARS:])
    report = get_eval_tests_report(
        statuses,
        {"FAIL_TO_PASS": plan.f2p, "PASS_TO_PASS": plan.p2p},
        eval_type=plan.eval_type,
    )
    resolved = get_resolution_status(report) == RESOLVED_FULL
    return GradingResult(
        **expected,
        resolved=resolved,
        f2p_pass=len(report["FAIL_TO_PASS"]["success"]),
        p2p_pass=len(report["PASS_TO_PASS"]["success"]),
        output_tail="" if resolved else log_text[-_TAIL_CHARS:],
    )


def main() -> int:
    """
    Read explicit artifact paths supplied by the host command builder.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        log_text = args.log.read_text(encoding="utf-8", errors="replace")
        result = grade_log(payload, log_text)
    except (OSError, ValueError, TypeError) as exc:
        result = GradingResult(failure_reason="grading_incomplete", error=str(exc))
    args.output.write_text(json.dumps(result.to_dict(), separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
