"""Audit Open Problem Pool JSONL files for RL training readiness."""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import re
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

psrl_logger = logging.getLogger(__name__)

SOURCE_FIELDS = frozenset({"uid", "problem", "solution", "msc_tag", "references", "extra"})
COMBINED_FIELDS = frozenset(
    {
        "uid",
        "problem",
        "solution",
        "category",
        "references",
        "extra",
        "difficulty",
        "publicity",
        "status",
        "difficulty_reason",
        "publicity_reason",
        "status_reason",
        "literature_reviewed_at",
    }
)
LABEL_FIELDS = ("difficulty", "publicity", "status")
UID_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """
    Load JSONL records and annotate each with its source file.

    Args:
        paths (Sequence[Path]): JSONL files.

    Returns:
        list[dict[str, Any]]: Parsed records. Malformed lines are skipped.
    """
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    psrl_logger.warning(f"Skipping a malformed line in {path}.")
                    continue
                if isinstance(record, dict):
                    record["_source_file"] = str(path)
                    records.append(record)
    return records


def detect_schema(record: dict[str, Any]) -> str:
    """
    Classify one record by its exact field set.

    Args:
        record (dict[str, Any]): One JSONL record.

    Returns:
        str: ``source``, ``combined``, or ``unknown``.
    """
    keys = frozenset(key for key in record if not key.startswith("_"))
    # A record may carry extra fields the pool adds later, such as `subfield`.
    # Classification is by the required contract, not by an exact field set.
    if COMBINED_FIELDS <= keys:
        return "combined"
    if SOURCE_FIELDS <= keys:
        return "source"
    return "unknown"


def _strip_label(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _problem_key(problem: Any) -> str:
    return " ".join(str(problem or "").split())


def _is_iso_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _length_summary(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": ordered[int(len(ordered) * 0.9)],
        "p99": ordered[int(len(ordered) * 0.99)],
        "max": ordered[-1],
    }


def _issue(code: str, severity: str, count: int, detail: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "count": count, "detail": detail}


def audit(
    records: Sequence[dict[str, Any]],
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    """
    Summarize schema shape, label quality, and RL blockers.

    Args:
        records (Sequence[dict[str, Any]]): Records from one or more JSONL files.
        token_counter (Callable[[str], int] | None): Optional tokenizer-backed counter.

    Returns:
        dict[str, Any]: Audit counts and issues.
    """
    total = len(records)
    schemas = collections.Counter(detect_schema(record) for record in records)
    field_sets = collections.Counter(
        tuple(sorted(key for key in record if not key.startswith("_"))) for record in records
    )
    uids: list[str] = []
    problems: list[str] = []
    problem_lengths: list[int] = []
    token_lengths: list[int] = []
    labels = {field: collections.Counter() for field in LABEL_FIELDS}
    trailing_newlines = {field: 0 for field in LABEL_FIELDS}
    empty_solution = 0
    null_category = 0
    invalid_uid = 0
    invalid_references = 0
    invalid_extra = 0
    invalid_literature = 0

    for record in records:
        uid = record.get("uid")
        problem = record.get("problem")
        uids.append(str(uid) if uid is not None else "")
        problems.append(_problem_key(problem))
        problem_lengths.append(len(str(problem or "")))
        if token_counter is not None and problem:
            token_lengths.append(token_counter(str(problem)))

        if not str(record.get("solution") or "").strip():
            empty_solution += 1
        if detect_schema(record) == "combined" and not str(record.get("category") or "").strip():
            null_category += 1
        if not isinstance(uid, str) or not UID_PATTERN.fullmatch(uid):
            invalid_uid += 1
        references = record.get("references")
        if not isinstance(references, list) or any(not isinstance(item, str) or not item for item in references):
            invalid_references += 1
        if not isinstance(record.get("extra"), dict):
            invalid_extra += 1
        if detect_schema(record) == "combined" and not _is_iso_timestamp(record.get("literature_reviewed_at")):
            invalid_literature += 1

        for field in LABEL_FIELDS:
            value = record.get(field)
            if value is None:
                continue
            labels[field][_strip_label(value)] += 1
            if str(value).endswith("\n"):
                trailing_newlines[field] += 1

    duplicate_problems = sum(count - 1 for count in collections.Counter(problems).values() if count > 1)
    duplicate_uids = sum(count - 1 for count in collections.Counter(uids).values() if count > 1)
    issues = [
        _issue(
            "empty_solution",
            "design",
            empty_solution,
            "No ground truth solution is available, so reward must come from a judge.",
        ),
        _issue(
            "null_category",
            "error",
            null_category,
            "schema.json requires a non-empty category string.",
        ),
        _issue(
            "duplicate_problem",
            "warning",
            duplicate_problems,
            "Duplicate problem statements spend rollout budget twice.",
        ),
        _issue(
            "duplicate_uid",
            "error",
            duplicate_uids,
            "Each record must have a unique uid.",
        ),
        _issue(
            "invalid_uid",
            "error",
            invalid_uid,
            "uid must match the dataset-prefixed lowercase pattern.",
        ),
        _issue(
            "invalid_references",
            "error",
            invalid_references,
            "references must be a list of non-empty strings.",
        ),
        _issue(
            "invalid_extra",
            "error",
            invalid_extra,
            "extra must be an object.",
        ),
        _issue(
            "invalid_literature_timestamp",
            "error",
            invalid_literature,
            "literature_reviewed_at must be an ISO 8601 timestamp.",
        ),
        _issue(
            "trailing_newline_label",
            "info",
            sum(trailing_newlines.values()),
            "Label values carry a trailing newline. Strip them during projection.",
        ),
    ]
    issues = [issue for issue in issues if issue["count"] > 0]

    return {
        "total": total,
        "schemas": dict(schemas),
        "distinct_field_sets": len(field_sets),
        "field_sets": [list(fields) for fields in field_sets],
        "issues": issues,
        "labels": {field: dict(counter) for field, counter in labels.items()},
        "trailing_newlines": trailing_newlines,
        "problem_chars": _length_summary(problem_lengths),
        "problem_tokens": _length_summary(token_lengths) if token_lengths else None,
    }


def _token_counter(tokenizer_path: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def _format_human(report: dict[str, Any], source: str) -> str:
    lines = [
        f"source: {source}",
        f"records: {report['total']}",
        f"schemas: {report['schemas']}",
        f"distinct field sets: {report['distinct_field_sets']}",
        f"problem chars: {report['problem_chars']}",
    ]
    if report.get("problem_tokens"):
        lines.append(f"problem tokens: {report['problem_tokens']}")
    for field, values in report["labels"].items():
        lines.append(f"{field}: {values}")
    lines.append("issues:")
    for issue in report["issues"]:
        lines.append(f"  [{issue['severity']}] {issue['code']} x{issue['count']}: {issue['detail']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Open Problem Pool JSONL files.")
    parser.add_argument("--input", action="append", type=Path, default=[], help="JSONL file. Repeatable.")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer path for token lengths.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.input
    if not paths:
        root = Path.cwd()
        print(
            f"No --input given. Pass one or more JSONL files, or run the projection script against {root}.",
            file=sys.stderr,
        )
        return 2
    records = load_records(paths)
    counter = _token_counter(args.tokenizer) if args.tokenizer else None
    report = audit(records, token_counter=counter)
    source = ", ".join(str(path) for path in paths)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        print(_format_human(report, source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
