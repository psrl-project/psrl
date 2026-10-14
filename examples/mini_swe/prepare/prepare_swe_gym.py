"""
Convert SWE-Gym datasets into PSRL parquet files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from examples.mini_swe.prepare.swebench_subsets import (
    repo_balanced_sample,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# --- Dataset mappings ---

_DATASET_HF_MAP: dict[str, str] = {
    "gym": "SWE-Gym/SWE-Gym",
    "gym-subset": "SumanthRH/SWE-Gym-Subset",
}

_DATASET_SPLIT_MAP: dict[str, str] = {
    "gym": "train",
    "gym-subset": "train",
}

# --- Image name helpers ---


def get_swegym_image_name(swe_problem: dict[str, Any]) -> str:
    """
    Compute the Docker image name for a SWE-Gym instance.

    Convention from OpenClaw-RL swe_utils.py:
      xingyaoww/sweb.eval.x86_64.{instance_id.replace("__", "_s_").lower()}:latest

    Args:
        swe_problem (dict[str, Any]): A single dataset row.

    Returns:
        str: Fully-qualified Docker image name.
    """
    instance_id: str = swe_problem["instance_id"]
    id_compat = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{id_compat}:latest"


# --- Evaluation script resolution ---


def _get_eval_script(row: dict[str, Any], has_eval_script_col: bool) -> str:
    """Get eval_script for a SWE-Gym instance.

    Priority:
    1. If the dataset has an eval_script column, use it directly.
    2. Otherwise, try swebench.harness.test_spec.make_test_spec
       (requires SWE-Bench-Fork 2.0.13).

    Args:
        row (dict[str, Any]): Single HF dataset row.
        has_eval_script_col (bool): Whether the HF dataset has eval_script.

    Returns:
        str: Bash eval script, or empty string if unavailable.
    """
    if has_eval_script_col:
        script = row.get("eval_script", "")
        if script and isinstance(script, str) and script.strip():
            return script.strip()

    # Fallback: generate via make_test_spec (needs SWE-Bench-Fork 2.0.13).
    try:
        from swebench.harness.test_spec import make_test_spec
    except ImportError:
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ImportError:
            return ""

    instance = {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "version": row.get("version", row.get("base_commit", "")),
        "FAIL_TO_PASS": _ensure_list(row.get("FAIL_TO_PASS")),
        "PASS_TO_PASS": _ensure_list(row.get("PASS_TO_PASS")),
        "patch": row.get("patch", ""),
        "test_patch": row.get("test_patch", ""),
        "problem_statement": row.get("problem_statement", ""),
        "hints_text": row.get("hints_text", ""),
        "created_at": row.get("created_at", ""),
    }
    try:
        ts = make_test_spec(instance)
        return ts.eval_script
    except Exception as e:
        psrl_logger.warning(f"make_test_spec failed for {row['instance_id']}: {e}")
        return ""


# --- Row conversion ---


def _ensure_list(value: Any) -> list[str]:
    """
    Coerce FAIL_TO_PASS / PASS_TO_PASS to a plain Python list of strings.

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
    # numpy array or other iterable
    try:
        return list(value)
    except TypeError:
        return []


def _build_row(
    swe_problem: dict[str, Any],
    *,
    eval_script: str,
    agent_name: str = "mini_swe_agent",
) -> dict[str, Any]:
    """
    Convert one HF dataset row into the PSRL parquet row format.

    Args:
        swe_problem (dict[str, Any]): Single HF dataset row.
        eval_script (str): Pre-computed eval script for grading.
        agent_name (str): Name of the agent loop class to use.

    Returns:
        dict[str, Any]: PSRL parquet row.
    """
    instance_id: str = swe_problem["instance_id"]
    problem_statement: str = swe_problem.get("problem_statement", "") or ""
    image_name: str = get_swegym_image_name(swe_problem)

    f2p: list[str] = _ensure_list(swe_problem.get("FAIL_TO_PASS", []))
    p2p: list[str] = _ensure_list(swe_problem.get("PASS_TO_PASS", []))

    # Ground truth for reward computation.
    ground_truth: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": swe_problem.get("repo", ""),
        "image_name": image_name,
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
        "gold_patch": swe_problem.get("patch", ""),
    }
    for key in ("base_commit", "test_patch", "version"):
        if swe_problem.get(key):
            ground_truth[key] = swe_problem[key]

    # Sandbox overrides: swap image and set cwd to /testbed.
    sandbox_overrides: dict[str, Any] = {
        "environment": {
            "image": image_name,
            "cwd": "/testbed",
        },
    }

    # Store the complete SWE problem dict + eval_script for the grader.
    swe_problem_plain: dict[str, Any] = {
        k: _ensure_list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v for k, v in swe_problem.items()
    }
    # Attach eval_script into the swe_problem dict for grading.
    swe_problem_plain["eval_script"] = eval_script

    extra_info: dict[str, Any] = {
        "swe_problem_id": instance_id,
        "problem_statement": problem_statement,
        "swe_problem": swe_problem_plain,
        "swe_problem_image": image_name,
        "swe_restore_tests": False,  # SWE-Gym does NOT need HEAD~1
        "swe_grader": "swebench_fresh_container",
        "sandbox_overrides": sandbox_overrides,
    }

    return {
        "prompt": [{"role": "user", "content": problem_statement}],
        "data_source": "swe_gym",
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
    split: str | None = None,
    total: int | None = None,
    repo_balanced: bool = True,
    per_repo_k: int | None = None,
    seed: int = 42,
    agent_name: str = "mini_swe_agent",
) -> pd.DataFrame:
    """
    Download a SWE-Gym dataset and convert it to a DataFrame.

    Args:
        dataset_key (str): One of ``gym``, ``gym-subset``.
        split (str | None): HF split name. Defaults per dataset.
        total (int | None): Maximum number of rows in the output.
        repo_balanced (bool): Apply repo-balanced round-robin sampling.
        per_repo_k (int | None): Per-repo hard cap before round-robin.
        seed (int): Random seed for deterministic shuffling.
        agent_name (str): Agent loop name tag.

    Returns:
        pd.DataFrame: Converted dataset, ready to write as parquet.
    """
    from datasets import load_dataset  # Avoid loading the heavy dependency at module import.

    hf_path = _DATASET_HF_MAP.get(dataset_key)
    assert hf_path is not None, f"Unknown dataset key {dataset_key!r}. Valid keys: {sorted(_DATASET_HF_MAP.keys())}."
    if split is None:
        split = _DATASET_SPLIT_MAP[dataset_key]

    print(f"[prepare_swe_gym] Loading {hf_path!r} split={split!r}...")
    swe_problems: list[dict[str, Any]] = list(load_dataset(hf_path, split=split))
    print(f"[prepare_swe_gym] Loaded {len(swe_problems)} instances.")

    has_eval_script_col = "eval_script" in (swe_problems[0].keys() if swe_problems else {})

    # Drop instances with empty problem_statement.
    n_before = len(swe_problems)
    swe_problems = [p for p in swe_problems if p.get("problem_statement", "")]
    if n_before - len(swe_problems):
        print(f"[prepare_swe_gym] Dropped {n_before - len(swe_problems)} instances with empty problem_statement.")

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

    # Convert each row.
    rows = []
    skipped = 0
    for idx, prob in enumerate(swe_problems):
        eval_script = _get_eval_script(prob, has_eval_script_col)
        if not eval_script:
            psrl_logger.warning(
                f"[{idx}] Skipping {prob['instance_id']}: no eval_script. "
                f"Install SWE-Bench-Fork to generate eval scripts."
            )
            skipped += 1
            continue
        rows.append(_build_row(prob, eval_script=eval_script, agent_name=agent_name))
        if (idx + 1) % 500 == 0:
            print(f"[prepare_swe_gym] Processed {idx + 1}/{len(swe_problems)}...")

    if skipped:
        print(f"[prepare_swe_gym] Skipped {skipped} instances (no eval_script).")

    df = pd.DataFrame(rows)
    print(f"[prepare_swe_gym] Conversion complete: {len(df)} rows.")
    return df


# --- CLI ---


def main() -> None:
    """CLI entry point for SWE-Gym dataset conversion."""
    parser = argparse.ArgumentParser(
        description="Convert SWE-Gym to PSRL parquet format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=list(_DATASET_HF_MAP.keys()),
        default="gym",
        help="Dataset to convert.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="HF split name. Defaults to 'train'.",
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
        help="Disable repo-balanced sampling.",
    )
    parser.add_argument(
        "--per-repo-k",
        type=int,
        default=None,
        help="Per-repo hard cap before round-robin.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/mini_swe/data/swe_gym_2438",
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
        help="Agent name tag.",
    )
    args = parser.parse_args()

    df = convert_dataset(
        args.dataset,
        split=args.split,
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

    # Repo distribution.
    repos = df["extra_info"].apply(lambda ei: ei["swe_problem"]["repo"])
    print(f"\nRepo distribution ({len(repos.unique())} repos):")
    for repo, count in repos.value_counts().items():
        print(f"  {repo}: {count}")


if __name__ == "__main__":
    main()
