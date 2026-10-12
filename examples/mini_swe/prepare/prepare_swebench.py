"""
SWE-bench / SWE-smith-py Dataset Converter.

Converts Hugging Face dataset rows into the PSRL parquet format consumed by
`fsdp_qwen_7b_swe_smith.sh` and related training scripts.

Each output row contains `prompt` (a minimal user message, since the framework
appends agent templates), `data_source` (`swebench_verified` or `swe_smith_py`),
`reward_model` (grounding truth for reward computation), `extra_info` (per-problem
overrides and grading metadata), and `agent_name` (`mini_swe_agent`).

Usage::

    # SWE-smith-py, repo-balanced 1 000 training SWE problems
    python -m examples.mini_swe.prepare.prepare_swebench \\
        --dataset smith \\
        --split train \\
        --per-repo-k 10 \\
        --total 1000 \\
        --output-dir examples/mini_swe/data/swe_smith_py_1k

    # SWE-bench Verified, full 500
    python -m examples.mini_swe.prepare.prepare_swebench \\
        --dataset verified \\
        --split test \\
        --output-dir examples/mini_swe/data/swe_bench_verified

    # SWE-bench Verified, 80-problem lightweight validation subset
    python -m examples.mini_swe.prepare.prepare_swebench \\
        --dataset verified \\
        --split test \\
        --total 80 \\
        --repo-balanced \\
        --output-dir examples/mini_swe/data/verified_subset_80 \\
        --output-filename test.parquet

Grading metadata:
    Verified rows keep the ``eval_script`` / ``log_parser`` shipped by the HF
    dataset. SWE-smith rows carry neither, so this script freezes an
    official-shaped ``eval_script`` and the flattened SWE-smith parser name
    (requires the ``swesmith`` package). See
    ``examples/mini_swe/grading/freeze.py`` and
    ``examples/mini_swe/prepare/README.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from examples.mini_swe.grading.freeze import freeze_smith_grading, validate_prepared_problem
from examples.mini_swe.prepare.swebench_subsets import (
    filter_by_spec,
    get_swebench_image_name,
    repo_balanced_sample,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# --- Dataset mappings ---

_DATASET_HF_MAP: dict[str, str] = {
    "verified": "SWE-bench/SWE-bench_Verified",
    "lite": "SWE-bench/SWE-bench_Lite",
    "full": "SWE-bench/SWE-bench",
    "smith": "SWE-bench/SWE-smith-py",
}

_DATA_SOURCE_MAP: dict[str, str] = {
    "verified": "swebench_verified",
    "lite": "swebench_verified",
    "full": "swebench_verified",
    "smith": "swe_smith_py",
}

# SWE-smith uses `HEAD~1` to restore removed F2P tests.
# Other datasets use the image checkout directly.
_NEEDS_HEAD_MINUS_ONE: dict[str, bool] = {
    "verified": False,
    "lite": False,
    "full": False,
    "smith": True,
}

# --- Row conversion ---


def _ensure_list(value: Any) -> list[str]:
    """
    Coerce FAIL_TO_PASS / PASS_TO_PASS to a plain Python list of strings.

    SWE-bench rows may store these fields as JSON-encoded strings. SWE-smith
    rows use native HF `Sequence` (already a list).

    Args:
        value (Any): Raw field value from the dataset row.

    Returns:
        list[str]: Decoded list of test case identifiers.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return []


def _build_row(
    swe_problem: dict[str, Any],
    *,
    data_source: str,
    needs_head_minus_one: bool,
    agent_name: str = "mini_swe_agent",
) -> dict[str, Any]:
    """
    Convert one HF dataset row into the PSRL parquet row format.

    Args:
        swe_problem (dict[str, Any]): Single HF dataset row describing one
            SWE problem (the HF field is still called ``instance_id`` upstream).
        data_source (str): Target data_source tag.
        needs_head_minus_one (bool): True for SWE-smith rows.
        agent_name (str): Name of the agent loop class to use.

    Returns:
        dict[str, Any]: PSRL parquet row.
    """
    swe_problem_id: str = swe_problem["instance_id"]
    problem_statement: str = swe_problem.get("problem_statement", "") or ""
    image_name: str = get_swebench_image_name(swe_problem)

    swe_problem = dict(swe_problem)
    if needs_head_minus_one:
        swe_problem.update(freeze_smith_grading(swe_problem))
    validate_prepared_problem(swe_problem)

    f2p: list[str] = _ensure_list(swe_problem.get("FAIL_TO_PASS", []))
    p2p: list[str] = _ensure_list(swe_problem.get("PASS_TO_PASS", []))

    # Keep `instance_id` because downstream harnesses require that exact key.
    ground_truth: dict[str, Any] = {
        "instance_id": swe_problem_id,
        "repo": swe_problem.get("repo", ""),
        "image_name": image_name,
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
        # The gold patch supports offline analysis but is not used for RL reward.
        "gold_patch": swe_problem.get("patch", ""),
    }
    # Include Verified-only fields when present.
    for key in ("base_commit", "test_patch", "version", "environment_setup_commit"):
        if swe_problem.get(key):
            ground_truth[key] = swe_problem[key]

    # Sandbox overrides: swap image and set cwd to /testbed.
    sandbox_overrides: dict[str, Any] = {
        "environment": {
            "image": image_name,
            "cwd": "/testbed",
        },
    }

    # Store the complete SWE problem dict for the grader, freezing an official-shaped
    # eval script for SWE-smith rows that lack the HF dataset's `eval_script`/`log_parser`.
    swe_problem_plain: dict[str, Any] = {
        k: _ensure_list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v for k, v in swe_problem.items()
    }

    extra_info: dict[str, Any] = {
        "swe_problem_id": swe_problem_id,
        "problem_statement": problem_statement,
        "swe_problem": swe_problem_plain,
        "swe_problem_image": image_name,
        "swe_restore_tests": needs_head_minus_one,
        "swe_grader": "swebench_fresh_container",
        "sandbox_overrides": sandbox_overrides,
    }

    return {
        "prompt": [{"role": "user", "content": problem_statement}],
        "data_source": data_source,
        "ability": "software_engineering",
        "reward_model": {
            "style": "swebench_test_exec",
            "ground_truth": ground_truth,
        },
        "extra_info": extra_info,
        "agent_name": agent_name,
    }


# --- Main conversion ---


def convert_dataset(
    dataset_key: str,
    *,
    split: str,
    subset_spec: str = "",
    total: int | None = None,
    repo_balanced: bool = True,
    per_repo_k: int | None = None,
    seed: int = 42,
    agent_name: str = "mini_swe_agent",
) -> pd.DataFrame:
    """
    Download a SWE-bench / SWE-smith-py dataset and convert it to a DataFrame.

    Args:
        dataset_key (str): One of ``verified``, ``lite``, ``full``, ``smith``.
        split (str): HF split name (e.g. ``"test"`` for Verified, ``"train"``
            for SWE-smith-py).
        subset_spec (str): Optional slice (``"0:100"``) or regex filter applied
            before sampling.
        total (int | None): Maximum number of rows in the output. If None,
            use all rows that survive filtering and per-repo caps.
        repo_balanced (bool): If True, apply repo-balanced round-robin sampling
            to reach `total`. If False, simply truncate to `total`.
        per_repo_k (int | None): Per-repo hard cap applied before round-robin.
        seed (int): Random seed for deterministic shuffling.
        agent_name (str): Agent loop name tag written to each row.

    Returns:
        pd.DataFrame: Converted dataset, ready to write as parquet.
    """
    from datasets import load_dataset  # Avoid loading the heavy dependency at module import.

    hf_path = _DATASET_HF_MAP.get(dataset_key)
    assert hf_path is not None, f"Unknown dataset key {dataset_key!r}. Valid keys: {sorted(_DATASET_HF_MAP.keys())}."
    data_source = _DATA_SOURCE_MAP[dataset_key]
    needs_head_minus_one = _NEEDS_HEAD_MINUS_ONE[dataset_key]

    psrl_logger.info(f"Loading dataset={hf_path!r}, split={split!r}...")
    swe_problems: list[dict[str, Any]] = list(load_dataset(hf_path, split=split))
    psrl_logger.info(f"Loaded SWE problems. Count: {len(swe_problems)}.")

    # Apply spec filter first.
    if subset_spec:
        swe_problems = filter_by_spec(swe_problems, subset_spec)
        psrl_logger.info(f"SWE problems after spec filter: {len(swe_problems)}.")

    # Drop instances without a problem statement because they cannot guide an agent.
    n_before = len(swe_problems)
    swe_problems = [p for p in swe_problems if p.get("problem_statement", "")]
    n_dropped = n_before - len(swe_problems)
    if n_dropped:
        psrl_logger.info(
            f"Dropped instances with empty problem_statement: {n_dropped}. Remaining: {len(swe_problems)}."
        )
        print(f"[prepare_swebench] Dropped {n_dropped} / {n_before} instances with empty problem_statement.")

    # Sampling.
    if total is not None and repo_balanced:
        swe_problems = repo_balanced_sample(
            swe_problems,
            total=total,
            per_repo_k=per_repo_k,
            seed=seed,
        )
    elif total is not None:
        swe_problems = swe_problems[:total]

    psrl_logger.info(f"Converting SWE problems to PSRL format. Count: {len(swe_problems)}...")
    rows = [
        _build_row(prob, data_source=data_source, needs_head_minus_one=needs_head_minus_one, agent_name=agent_name)
        for prob in swe_problems
    ]
    df = pd.DataFrame(rows)
    psrl_logger.info(f"Conversion row count: {len(df)}.")
    return df


# --- CLI ---


def main() -> None:
    """
    CLI entry point for SWE-bench / SWE-smith-py dataset conversion.
    """
    parser = argparse.ArgumentParser(
        description="Convert SWE-bench / SWE-smith-py to PSRL parquet format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(_DATASET_HF_MAP.keys()),
        default="smith",
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--split",
        default="",
        help="HF split name.  Defaults to 'train' for smith and 'test' for others.",
    )
    parser.add_argument(
        "--subset-spec",
        default="",
        help="Slice (e.g. '0:100') or regex filter on the SWE problem's instance_id field.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Maximum number of output rows.",
    )
    parser.add_argument(
        "--repo-balanced",
        action="store_true",
        default=True,
        help="Apply repo-balanced round-robin sampling (default: enabled).",
    )
    parser.add_argument(
        "--no-repo-balanced",
        dest="repo_balanced",
        action="store_false",
        help="Disable repo-balanced sampling; truncate to --total instead.",
    )
    parser.add_argument(
        "--per-repo-k",
        type=int,
        default=None,
        help="Per-repo hard cap before round-robin (e.g. 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/mini_swe/data",
        help="Output directory.",
    )
    parser.add_argument(
        "--output-filename",
        default="train.parquet",
        help="Output parquet filename.",
    )
    parser.add_argument(
        "--agent-name",
        default="mini_swe_agent",
        help="Agent name tag written to each row.",
    )
    args = parser.parse_args()

    # Default split per dataset.
    split = args.split
    if not split:
        split = "train" if args.dataset == "smith" else "test"

    df = convert_dataset(
        args.dataset,
        split=split,
        subset_spec=args.subset_spec,
        total=args.total,
        repo_balanced=args.repo_balanced,
        per_repo_k=args.per_repo_k,
        seed=args.seed,
        agent_name=args.agent_name,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_filename
    df.to_parquet(out_path)
    print(f"Wrote {len(df)} rows to {out_path}.")

    # Print a brief schema summary.
    row0 = df.iloc[0].to_dict()
    print(f"data_source    : {row0['data_source']!r}")
    print(f"swe_problem_id : {row0['extra_info']['swe_problem_id']!r}")
    print(f"swe_image      : {row0['extra_info']['swe_problem_image']!r}")
    print(f"swe_grader     : {row0['extra_info']['swe_grader']!r}")
    f2p_count = len(row0["reward_model"]["ground_truth"]["FAIL_TO_PASS"])
    p2p_count = len(row0["reward_model"]["ground_truth"]["PASS_TO_PASS"])
    print(f"F2P / P2P      : {f2p_count} / {p2p_count}")


if __name__ == "__main__":
    main()
