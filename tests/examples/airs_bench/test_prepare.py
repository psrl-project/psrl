import json

import pytest
import yaml
from examples.airs_bench.prepare.build_parquet import build_row, load_split
from examples.airs_bench.prepare.prepare_airs_data import (
    load_task_metadata,
    rewrite_dataset_yaml,
)

METADATA_FIXTURE = {
    "metric_lower_is_better": True,
    "logging_info": {
        "name": "GraphRegressionZincMae",
        "category": "Graph",
        "metric": "MAE",
        "estimated_worst_score": 9.699924,
        "optimal_score": 0.0,
        "sota": [{"sota_score": 0.017, "sota_paper_title": "Some Paper"}],
    },
}


@pytest.mark.cpu_test
def test_load_task_metadata_extracts_normalization_constants(tmp_path):
    task_dir = tmp_path / "GraphRegressionZincMae"
    task_dir.mkdir()
    (task_dir / "metadata.yaml").write_text(yaml.safe_dump(METADATA_FIXTURE))

    metadata = load_task_metadata(task_dir)

    assert metadata["metric"] == "MAE"
    assert metadata["metric_lower_is_better"] is True
    assert metadata["sota_score"] == 0.017
    assert metadata["estimated_worst_score"] == 9.699924
    assert metadata["optimal_score"] == 0.0
    assert metadata["category"] == "Graph"


@pytest.mark.cpu_test
def test_load_task_metadata_rejects_missing_sota(tmp_path):
    broken = {"metric_lower_is_better": True, "logging_info": {"metric": "MAE", "sota": []}}
    task_dir = tmp_path / "Broken"
    task_dir.mkdir()
    (task_dir / "metadata.yaml").write_text(yaml.safe_dump(broken))

    with pytest.raises(ValueError, match="sota"):
        load_task_metadata(task_dir)


@pytest.mark.cpu_test
def test_rewrite_dataset_yaml_replaces_only_the_data_path(tmp_path):
    src = tmp_path / "src.yaml"
    dst = tmp_path / "dst.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "data_path": "/checkpoint/maui/shared/datasets/airs_text_only_prepared/Foo",
                "is_local": True,
                "name": "ZINC",
                "description": {"y": {"_type": "Value"}},
            }
        )
    )

    rewrite_dataset_yaml(src, dst, "/shared/airs_prepared/Foo")

    written = yaml.safe_load(dst.read_text())
    assert written["data_path"] == "/shared/airs_prepared/Foo"
    assert written["is_local"] is True, "Unrelated keys must survive the rewrite."
    assert written["name"] == "ZINC"
    assert written["description"] == {"y": {"_type": "Value"}}


@pytest.mark.cpu_test
def test_load_split_returns_fourteen_train_and_six_val():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    split_path = repo_root / "examples" / "airs_bench" / "prepare" / "split.json"
    train, val = load_split(split_path)

    assert len(train) == 14, f"Expected 14 training tasks, got {len(train)}."
    assert len(val) == 6, f"Expected 6 validation tasks, got {len(val)}."
    assert set(train).isdisjoint(set(val)), "A task must never appear in both splits."
    assert len(set(train) | set(val)) == 20, "The split must cover all 20 tasks exactly once."


@pytest.mark.cpu_test
def test_split_is_stratified_across_categories():
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    split_path = repo_root / "examples" / "airs_bench" / "prepare" / "split.json"
    payload = json.loads(split_path.read_text())

    val_categories = {payload["categories"][task_id] for task_id in payload["val"]}
    assert len(val_categories) >= 3, f"Validation must span at least three categories, got {sorted(val_categories)}."


@pytest.mark.cpu_test
def test_build_row_embeds_every_constant_the_reward_needs():
    metadata = {
        "metric": "MAE",
        "metric_lower_is_better": True,
        "sota_score": 0.017,
        "estimated_worst_score": 9.699924,
        "optimal_score": 0.0,
        "category": "Graph",
    }

    row = build_row(
        task_id="GraphRegressionZincMae",
        metadata=metadata,
        task_config_path="tasks/GraphRegressionZincMae.yaml",
        dataset_data_path="/shared/airs_prepared/GraphRegressionZincMae",
    )

    assert row["data_source"] == "airs_bench"
    assert isinstance(row["prompt"], list), "prompt must be a chat message list."
    extra = row["extra_info"]
    for key in (
        "airs_task_id",
        "task_config_path",
        "dataset_data_path",
        "metric",
        "metric_lower_is_better",
        "sota_score",
        "estimated_worst_score",
        "optimal_score",
    ):
        assert key in extra, f"extra_info is missing the required key {key}."
    assert extra["airs_task_id"] == "GraphRegressionZincMae"


@pytest.mark.cpu_test
def test_build_row_extra_info_feeds_the_reward_function():
    """The row contract and the reward contract must agree."""
    from examples.airs_bench.reward import compute_score

    metadata = {
        "metric": "MAE",
        "metric_lower_is_better": True,
        "sota_score": 0.017,
        "estimated_worst_score": 9.699924,
        "optimal_score": 0.0,
        "category": "Graph",
    }
    row = build_row("GraphRegressionZincMae", metadata, "t.yaml", "/d")

    extra = dict(row["extra_info"])
    extra["airs_scores"] = [{"MAE": 0.5}]
    extra["airs_submit_count"] = 1
    result = compute_score("airs_bench", "", None, extra)

    assert 0.0 < result["score"] < 1.0, "A prepared row must score end to end without extra keys."
