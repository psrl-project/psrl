"""
Emit the train and validation parquet files for AIRS-Bench.

Each task becomes exactly one row. MLGym builds the real prompts from its own
templates, so the `prompt` column carries only a minimal seed message. Every
normalization constant the reward needs is embedded in `extra_info`, which keeps
the reward function pure at scoring time.

Usage:
    python -m examples.airs_bench.prepare.build_parquet \\
        --airs-repo /apdcephfs_zwfy10/share_303541817/lhy/science_infra/airs-bench \\
        --data-root /apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data \\
        --out-dir examples/airs_bench/data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from examples.airs_bench.prepare.prepare_airs_data import load_task_metadata

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

SEED_PROMPT = "You are an autonomous machine learning researcher. Solve the assigned task."


def load_split(split_path: Path) -> tuple[list[str], list[str]]:
    """
    Read the pinned train and validation task ids.

    The split is checked in rather than computed at load time, so that comparisons
    across runs stay valid.

    Args:
        split_path (Path): Path to `split.json`.

    Returns:
        tuple[list[str], list[str]]: Train task ids and validation task ids.
    """
    payload = json.loads(split_path.read_text())
    return list(payload["train"]), list(payload["val"])


def build_row(
    task_id: str,
    metadata: dict[str, Any],
    task_config_path: str,
    dataset_data_path: str,
) -> dict[str, Any]:
    """
    Build one dataset row for one AIRS-Bench task.

    Args:
        task_id (str): AIRS-Bench task identifier.
        metadata (dict[str, Any]): Output of `load_task_metadata`.
        task_config_path (str): MLGym task config path, relative to its config dir.
        dataset_data_path (str): Absolute path to this task's prepared data.

    Returns:
        dict[str, Any]: Row with prompt, data_source, reward_model, and extra_info.
    """
    return {
        "prompt": [{"role": "user", "content": SEED_PROMPT}],
        "data_source": "airs_bench",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "agent_name": "mlgym_agent",
        "extra_info": {
            "airs_task_id": task_id,
            "task_config_path": task_config_path,
            "dataset_data_path": dataset_data_path,
            "metric": metadata["metric"],
            "metric_lower_is_better": metadata["metric_lower_is_better"],
            "sota_score": metadata["sota_score"],
            "estimated_worst_score": metadata["estimated_worst_score"],
            "optimal_score": metadata["optimal_score"],
            "category": metadata["category"],
        },
    }


def main() -> None:
    """Write train.parquet and val.parquet."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build AIRS-Bench parquet files.")
    parser.add_argument("--airs-repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--split-path",
        type=Path,
        default=Path("examples/airs_bench/prepare/split.json"),
    )
    args = parser.parse_args()

    train_ids, val_ids = load_split(args.split_path)
    rad_root = args.airs_repo / "airsbench" / "tasks" / "rad"
    prepared_dir = args.data_root / "airs_prepared"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, task_ids in (("train", train_ids), ("val", val_ids)):
        rows = [
            build_row(
                task_id=task_id,
                metadata=load_task_metadata(rad_root / task_id),
                task_config_path=f"tasks/{task_id}.yaml",
                dataset_data_path=str(prepared_dir / task_id),
            )
            for task_id in task_ids
        ]
        out_path = args.out_dir / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(out_path)
        psrl_logger.info(f"Wrote rows to {out_path!s}. Count: {len(rows)}.")


if __name__ == "__main__":
    main()
