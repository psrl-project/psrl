#!/usr/bin/env python3
"""Compare versioned trainer, PS, and rollout weight fingerprints in PSRL logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

FINGERPRINT_MARKER = "[WEIGHT_FINGERPRINT]"
SLEEP_WAKE_MARKER = "[TRAINER_SLEEP_WAKE_WEIGHT_CHECK]"


def _json_after_marker(line: str, marker: str) -> dict[str, Any] | None:
    marker_index = line.find(marker)
    if marker_index < 0:
        return None
    json_index = line.find("{", marker_index + len(marker))
    if json_index < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(line[json_index:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def parse_weight_verification_logs(log_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fingerprints = []
    sleep_wake_checks = []
    for path in sorted(log_dir.rglob("*.log")):
        with path.open(encoding="utf-8", errors="replace") as log_file:
            for line_number, line in enumerate(log_file, start=1):
                if FINGERPRINT_MARKER in line:
                    record = _json_after_marker(line, FINGERPRINT_MARKER)
                    if record is not None:
                        record.update({"file": str(path.relative_to(log_dir)), "line": line_number})
                        fingerprints.append(record)
                if SLEEP_WAKE_MARKER in line:
                    record = _json_after_marker(line, SLEEP_WAKE_MARKER)
                    if record is not None:
                        record.update({"file": str(path.relative_to(log_dir)), "line": line_number})
                        sleep_wake_checks.append(record)
    return fingerprints, sleep_wake_checks


def _merge_tensor_digests(records: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    values: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for name, digest in record.get("tensor_digests", {}).items():
            values[str(name)].add(str(digest))
    conflicts = {name: sorted(digests) for name, digests in values.items() if len(digests) > 1}
    merged = {name: next(iter(digests)) for name, digests in values.items() if len(digests) == 1}
    return merged, conflicts


def _compare_record_groups(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    model_version: int,
    check: str,
    instance_id: int | None = None,
) -> dict[str, Any]:
    expected_digests, expected_conflicts = _merge_tensor_digests(expected)
    actual_digests, actual_conflicts = _merge_tensor_digests(actual)
    common = expected_digests.keys() & actual_digests.keys()
    differing = sorted(name for name in common if expected_digests[name] != actual_digests[name])
    missing = sorted(expected_digests.keys() - actual_digests.keys())
    extra = sorted(actual_digests.keys() - expected_digests.keys())
    comparable = bool(expected_digests) and bool(actual_digests)
    match = comparable and not (expected_conflicts or actual_conflicts or differing or missing or extra)
    return {
        "model_version": model_version,
        "check": check,
        "instance_id": instance_id,
        "match": match,
        "comparable": comparable,
        "expected_tensor_count": len(expected_digests),
        "actual_tensor_count": len(actual_digests),
        "differing_tensor_count": len(differing),
        "missing_tensor_count": len(missing),
        "extra_tensor_count": len(extra),
        "expected_conflict_count": len(expected_conflicts),
        "actual_conflict_count": len(actual_conflicts),
        "differing_tensors": differing[:20],
        "missing_tensors": missing[:20],
        "extra_tensors": extra[:20],
        "expected_conflicts": sorted(expected_conflicts)[:20],
        "actual_conflicts": sorted(actual_conflicts)[:20],
    }


def analyze_weight_fingerprints(
    fingerprints: list[dict[str, Any]],
    sleep_wake_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    by_version_stage: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    versions = set()
    for record in fingerprints:
        version = int(record["model_version"])
        versions.add(version)
        by_version_stage[(version, str(record["stage"]))].append(record)

    comparisons = []
    for version in sorted(versions):
        trainer = by_version_stage[(version, "trainer_before_push")]
        ps_train = by_version_stage[(version, "ps_after_receive")]
        ps_rollout = by_version_stage[(version, "ps_after_layout_copy")]
        if trainer and ps_train:
            comparisons.append(
                _compare_record_groups(
                    trainer,
                    ps_train,
                    model_version=version,
                    check="trainer_to_ps_train",
                )
            )
        if ps_train and ps_rollout:
            comparisons.append(
                _compare_record_groups(
                    ps_train,
                    ps_rollout,
                    model_version=version,
                    check="ps_train_to_ps_rollout",
                )
            )

        rollout_raw = by_version_stage[(version, "rollout_after_raw_pull")]
        rollout_final = by_version_stage[(version, "rollout_after_param_sync")]
        instance_ids = sorted(
            {
                int(record["instance_id"])
                for record in rollout_raw + rollout_final
                if record.get("instance_id") is not None
            }
        )
        for instance_id in instance_ids:
            raw_instance = [
                record
                for record in rollout_raw
                if record.get("instance_id") is not None and int(record["instance_id"]) == instance_id
            ]
            final_instance = [
                record
                for record in rollout_final
                if record.get("instance_id") is not None and int(record["instance_id"]) == instance_id
            ]
            if ps_rollout and raw_instance:
                comparisons.append(
                    _compare_record_groups(
                        ps_rollout,
                        raw_instance,
                        model_version=version,
                        check="ps_rollout_to_rollout_raw",
                        instance_id=instance_id,
                    )
                )
            if raw_instance and final_instance:
                comparisons.append(
                    _compare_record_groups(
                        raw_instance,
                        final_instance,
                        model_version=version,
                        check="rollout_raw_to_param_sync",
                        instance_id=instance_id,
                    )
                )

    comparable = [comparison for comparison in comparisons if comparison["comparable"]]
    mismatches = [comparison for comparison in comparable if not comparison["match"]]
    sleep_wake_mismatches = [record for record in sleep_wake_checks if record.get("status") != "match"]
    return {
        "summary": {
            "fingerprint_records": len(fingerprints),
            "versions": sorted(versions),
            "comparisons": len(comparisons),
            "comparable_checks": len(comparable),
            "mismatched_checks": len(mismatches),
            "trainer_sleep_wake_checks": len(sleep_wake_checks),
            "trainer_sleep_wake_mismatches": len(sleep_wake_mismatches),
        },
        "comparisons": comparisons,
        "trainer_sleep_wake_checks": sleep_wake_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="PSRL run log directory")
    parser.add_argument("--output", type=Path, help="defaults to LOG_DIR/weight_fingerprint_report.json")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    fingerprints, sleep_wake_checks = parse_weight_verification_logs(args.log_dir)
    report = analyze_weight_fingerprints(fingerprints, sleep_wake_checks)
    output = args.output or args.log_dir / "weight_fingerprint_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    if args.fail_on_mismatch and (
        report["summary"]["mismatched_checks"]
        or report["summary"]["trainer_sleep_wake_mismatches"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
