"""Tests for the dataset projection."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from examples.math_rl.project_dataset import (
    COLUMNS,
    Filters,
    derive_category,
    derive_difficulty,
    derive_status,
    normalize_record,
    project,
    write_parquet,
)


def test_normalize_combined_record_fills_category_and_strips_labels(combined_record):
    normalized = normalize_record(combined_record)
    assert normalized["category"] == "Graph Theory"
    assert normalized["category_source"] == "direction"
    assert normalized["difficulty"] == "1"
    assert normalized["status"] == "resolved"
    assert normalized["reference_context"]
    assert normalized["schema"] == "combined"


def test_category_fallback_is_mathematics():
    record = {
        "uid": "x_1",
        "problem": "P",
        "solution": "",
        "msc_tag": None,
        "references": [],
        "extra": {},
    }
    category, source = derive_category(record)
    assert category == "Mathematics"
    assert source == "fallback"


def test_source_difficulty_and_status_derivation(source_record):
    assert derive_difficulty(source_record) == "2"
    assert derive_status(source_record) == "unknown"


def test_project_filters_dedupes_and_builds_messages(combined_record):
    duplicate = dict(combined_record)
    duplicate["uid"] = "danus_open_problems_m_002"
    rows, manifest = project(
        [combined_record, duplicate],
        Filters(min_difficulty="1", max_difficulty="1", status="resolved"),
    )
    assert len(rows) == 1
    assert manifest["dropped"]["duplicate_problem"] == 1
    row = rows[0]
    assert set(row) == set(COLUMNS)
    assert row["prompt"][0]["role"] == "user"
    assert row["extra_info"]["uid"] == combined_record["uid"]
    assert row["extra_info"]["solution_path"] == "/workspace/solution.md"


def test_project_drops_records_outside_the_filter(source_record):
    rows, manifest = project([source_record], Filters(min_difficulty="1", max_difficulty="1", status="resolved"))
    assert rows == []
    assert manifest["dropped"]["difficulty_out_of_range"] == 1


def test_project_applies_prompt_token_limit(combined_record, word_counter):
    rows, manifest = project(
        [combined_record],
        Filters(min_difficulty="1", max_difficulty="1", status="resolved", max_prompt_tokens=5),
        token_counter=word_counter,
    )
    assert rows == []
    assert manifest["dropped"]["prompt_too_long"] == 1
    assert manifest["prompt_tokens"]["count"] == 1


def test_write_parquet_splits_and_keeps_the_contract(tmp_path, combined_record):
    rows, _ = project([combined_record], Filters(min_difficulty="1", max_difficulty="1", status="resolved"))
    assert len(rows) == 1
    many = []
    for index in range(10):
        row = json.loads(json.dumps(rows[0]))
        row["extra_info"]["uid"] = f"uid_{index}"
        many.append(row)
    split = write_parquet(many, tmp_path, seed=7, train_fraction=0.8)
    assert split == {"train_rows": 8, "val_rows": 2}
    train = pd.read_parquet(tmp_path / "train.parquet")
    val = pd.read_parquet(tmp_path / "val.parquet")
    assert list(train.columns) == COLUMNS
    assert len(train) == 8
    assert len(val) == 2
    assert train["prompt"].iloc[0][0]["role"] == "user"


def test_write_parquet_rejects_empty_rows(tmp_path):
    with pytest.raises(ValueError, match="No rows"):
        write_parquet([], tmp_path, seed=1, train_fraction=0.9)


def test_difficulty_range_keeps_low_and_medium(combined_record):
    low = dict(combined_record)
    medium = dict(combined_record)
    medium["uid"] = "danus_open_problems_m_003"
    medium["difficulty"] = "2\n"
    medium["problem"] = "A second problem."
    invalid = dict(combined_record)
    invalid["uid"] = "danus_open_problems_m_004"
    invalid["difficulty"] = "4\n"
    invalid["problem"] = "A third problem."
    rows, manifest = project([low, medium, invalid], Filters())
    assert {row["extra_info"]["difficulty"] for row in rows} == {"1", "2"}
    assert manifest["dropped"]["difficulty_out_of_range"] == 1
