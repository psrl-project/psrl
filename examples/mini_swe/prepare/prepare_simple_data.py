"""
Generate simple mini-SWE-agent train and validation parquet files.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_PREPARE_DIR = Path(__file__).parent


# --- Helpers ---


def _make_minimal_prompt(problem_statement: str) -> list[dict[str, str]]:
    """
    Minimal prompt satisfying the framework's ``raw_prompt`` requirement.
    """
    return [{"role": "user", "content": problem_statement}]


def _load_simple_cases(filename: str) -> list[dict[str, Any]]:
    """
    Load simple test cases from a JSON file next to this script.
    """
    path = _PREPARE_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- Simple test data ---


def generate_simple_test_data(
    num_samples: int,
    split: str,
    agent_name: str = "mini_swe_agent",
) -> pd.DataFrame:
    """
    Generate simple test data for quick validation.

    Train and validation use completely disjoint problem pools.
    Cases are loaded from JSON files in the same directory.

    Repos are pre-baked into the Docker image by ``bake_simple_repos.sh``
    at ``/<split>_<case_idx>/``. Each sample uses preexisting repo mode
    so no repo creation happens at rollout time.
    """
    if split == "train":
        pool = _load_simple_cases("simple_cases_train.json")
        repo_prefix = "train"
    else:
        pool = _load_simple_cases("simple_cases_val.json")
        repo_prefix = "val"

    rows: list[dict[str, Any]] = []
    for idx in range(num_samples):
        case = pool[idx % len(pool)]
        case_idx = idx % len(pool)
        repo_name = f"{repo_prefix}_{case_idx}"

        rows.append(
            {
                "prompt": _make_minimal_prompt(case["problem_statement"]),
                "data_source": "mini_swe_agent_simple",
                "ability": "software_engineering",
                "reward_model": {
                    "style": "mini_swe_agent",
                    "ground_truth": case["expected_patch"],
                },
                "extra_info": {
                    "index": idx,
                    "split": split,
                    "expected_patch": case["expected_patch"],
                    "problem_statement": case["problem_statement"],
                    "sandbox_overrides": {
                        "use_preexisting_repo": True,
                        "preexisting_repo_name": repo_name,
                    },
                },
                "agent_name": agent_name,
            }
        )

    return pd.DataFrame(rows)


# --- CLI ---


def main() -> None:
    """
    CLI entry point for dataset generation.
    """
    parser = argparse.ArgumentParser(description="mini-SWE-agent Dataset Generator")
    parser.add_argument("--mode", choices=["simple"], default="simple", help="Data generation mode")
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--output_dir", default="data/mini_swe_agent")
    parser.add_argument("--agent_name", default="mini_swe_agent")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    psrl_logger.info("Generating simple test data...")
    train_df = generate_simple_test_data(args.train_size, "train", args.agent_name)
    test_df = generate_simple_test_data(args.test_size, "test", args.agent_name)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    psrl_logger.info("Dataset generation completed!")
    psrl_logger.info(f"Train sample count: {len(train_df)}. Path: {train_path!r}.")
    psrl_logger.info(f"Test sample count: {len(test_df)}. Path: {test_path!r}.")


if __name__ == "__main__":
    main()
