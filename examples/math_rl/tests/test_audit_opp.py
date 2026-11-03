"""Tests for the dataset audit."""

from __future__ import annotations

import json

from examples.math_rl.audit_opp import audit, detect_schema, load_records


def test_detect_schema_distinguishes_source_and_combined(source_record, combined_record):
    assert detect_schema(source_record) == "source"
    assert detect_schema(combined_record) == "combined"
    assert detect_schema({"uid": "x"}) == "unknown"


def test_audit_reports_rl_blockers(source_record, combined_record):
    report = audit([source_record, combined_record])
    issues = {issue["code"]: issue for issue in report["issues"]}
    assert report["total"] == 2
    assert report["schemas"] == {"source": 1, "combined": 1}
    assert issues["empty_solution"]["count"] == 2
    assert issues["null_category"]["count"] == 1
    assert issues["trailing_newline_label"]["count"] == 3


def test_audit_counts_duplicate_problems(source_record):
    report = audit([source_record, dict(source_record)])
    issues = {issue["code"]: issue for issue in report["issues"]}
    assert issues["duplicate_problem"]["count"] == 1
    assert issues["duplicate_uid"]["count"] == 1


def test_load_records_skips_malformed_lines(tmp_path, source_record):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(source_record) + "\nnot json\n" + json.dumps(source_record) + "\n",
        encoding="utf-8",
    )
    records = load_records([path])
    assert len(records) == 2
    assert records[0]["_source_file"] == str(path)
