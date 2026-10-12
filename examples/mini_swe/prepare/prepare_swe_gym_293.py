"""SkyRL-v0-293 dataset converter to the PSRL parquet format.

Dataset: ``NovaSky-AI/SkyRL-v0-293-data``, with 293 train and 23 validation
instances. It is a curated SWE-bench Verified subset that a Qwen3.5-4B model
can already solve about 32.6% of the time (non-zero reward signal).

Images: ``xingyaoww/sweb.eval.x86_64.<instance __->_s_ lower>:latest``. Legacy
repos use ``swebench/sweb.eval.x86_64.<owner>_1776_<repo>-<issue>:latest``.

Grading: ``swebench_fresh_container``. The per-instance ``eval_script`` is
generated with the SWE-Bench-Fork (the swegym fork of swebench), which supports
the SWE-Gym repos that swebench 4.x does not.

The output parquet matches the schema consumed by the mini-SWE harness pipeline
(``prompt / data_source / ability / reward_model / extra_info / agent_name``),
with ``extra_info.sandbox_overrides.environment.image`` set so that
``prefetch_images.sh`` can pull the per-task images.

Usage::

    # 1) Generate the parquet (creates a temp venv with the SWE-Bench-Fork if needed)
    python -m examples.mini_swe.prepare.prepare_swe_gym_293 \
--output-dir examples/mini_swe/data/swe_gym_293 \
--ensure-fork --fork-venv /tmp/swegym-fork-venv

    # 2) Prefetch / fan-out images (see prepare/docker_scripts/swe_gym_293.sh)
    bash examples/mini_swe/prepare/docker_scripts/swe_gym_293.sh

Environment notes:
    The eval_script generation requires the SWE-Bench-Fork
    (``git+https://github.com/SWE-Gym/SWE-Bench-Fork.git``), which conflicts with
    swebench 4.x, so it is installed inside an isolated ``--fork-venv`` and never
    touches the main Python environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from examples.mini_swe.grading.freeze import validate_prepared_problem
from examples.mini_swe.grading.parsers import DEFAULT_PARSER

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

DATASET_NAME = "NovaSky-AI/SkyRL-v0-293-data"
TRAIN_PARQUET_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/train.parquet"
VAL_PARQUET_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/validation.parquet"
SWE_BENCH_FORK_REQ = "git+https://github.com/SWE-Gym/SWE-Bench-Fork.git"

LEGACY_SWEBENCH_IMAGE_REPOS = {
    "marshmallow-code/marshmallow",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pvlib/pvlib-python",
    "pyvista/pyvista",
    "sqlfluff/sqlfluff",
}


# --- Image naming ---


def registry_image_for_instance(instance_id: str, repo: str) -> str:
    """Return the Docker image used by SWE-Gym-293 run."""
    owner, repo_with_issue = instance_id.split("__", 1)
    image_repo, issue_id = repo_with_issue.rsplit("-", 1)
    if repo in LEGACY_SWEBENCH_IMAGE_REPOS:
        return f"swebench/sweb.eval.x86_64.{owner}_1776_{image_repo}-{issue_id}:latest"
    suffix = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"


# --- SWE-Bench-Fork venv management (eval_script generation) ---


def ensure_fork_venv(venv: str, *, force: bool = False) -> str:
    """Create (if missing) an isolated venv with the SWE-Bench-Fork installed."""
    venv = os.path.abspath(venv)
    python_bin = os.path.join(venv, "bin", "python")
    if os.path.isfile(python_bin) and not force:
        return python_bin

    print(f"[prepare_swe_gym_293] Creating fork venv at {venv} ...")
    subprocess.run([sys.executable, "-m", "venv", *(["--clear"] if force else []), venv], check=True)
    print(f"[prepare_swe_gym_293] Installing SWE-Bench-Fork into {venv} ...")
    subprocess.run(
        [os.path.join(venv, "bin", "pip"), "install", "-q", SWE_BENCH_FORK_REQ],
        check=True,
    )
    return python_bin


def _make_eval_script(instance: dict[str, Any], fork_python: str) -> str:
    """Generate a SWE-Gym eval_script via the fork's ``make_test_spec``.

    Runs in a subprocess with the fork venv's interpreter so the main
    environment's swebench 4.x is never disturbed.
    """
    code = (
        "import json, sys\n"
        "from swebench.harness.test_spec import make_test_spec\n"
        "inst = json.load(sys.stdin)\n"
        "ts = make_test_spec(inst)\n"
        "print(ts.eval_script)\n"
    )
    proc = subprocess.run(
        [fork_python, "-c", code],
        input=json.dumps(instance),
        capture_output=True,
        text=True,
        check=True,
    )
    script = proc.stdout.strip()
    if not script:
        raise RuntimeError(f"make_test_spec produced an empty eval_script for {instance['instance_id']}.")
    return script


# --- Row conversion ---


def _ensure_list(value: Any) -> list[str]:
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
    try:
        return list(value)
    except TypeError:
        return []


def _normalize(value: Any) -> Any:
    """Convert numpy / non-JSON values to plain Python natives."""
    import numpy as np

    if isinstance(value, np.ndarray):
        return [_normalize(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(x) for x in value]
    return value


def _build_row(
    instance: dict[str, Any],
    *,
    eval_script: str,
    agent_name: str = "mini_swe_agent",
) -> dict[str, Any]:
    """Convert one SkyRL-v0-293 ``instance`` dict into a PSRL parquet row."""
    instance_id: str = instance["instance_id"]
    problem_statement: str = instance.get("problem_statement", "") or ""
    repo: str = instance.get("repo", "")
    instance = dict(instance)
    instance["eval_script"] = eval_script
    validate_prepared_problem(instance, default_parser=DEFAULT_PARSER)
    image_name: str = registry_image_for_instance(instance_id, repo)

    f2p: list[str] = _ensure_list(instance.get("FAIL_TO_PASS", []))
    p2p: list[str] = _ensure_list(instance.get("PASS_TO_PASS", []))

    ground_truth: dict[str, Any] = {
        "instance_id": instance_id,
        "repo": repo,
        "image_name": image_name,
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
        "gold_patch": instance.get("patch", ""),
    }
    for key in ("base_commit", "test_patch", "version"):
        if instance.get(key):
            ground_truth[key] = instance[key]

    sandbox_overrides: dict[str, Any] = {
        "environment": {
            "image": image_name,
            "cwd": "/testbed",
        },
    }

    swe_problem_plain: dict[str, Any] = {
        k: _ensure_list(v) if k in ("FAIL_TO_PASS", "PASS_TO_PASS") else v for k, v in instance.items()
    }
    swe_problem_plain["eval_script"] = eval_script
    validate_prepared_problem(swe_problem_plain, default_parser=DEFAULT_PARSER)

    extra_info: dict[str, Any] = {
        "swe_problem_id": instance_id,
        "problem_statement": problem_statement,
        "swe_problem": swe_problem_plain,
        "swe_problem_image": image_name,
        "swe_restore_tests": False,  # SWE-Gym instances do not need HEAD~1 restore
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


def _download_parquet(url: str) -> pd.DataFrame:
    print(f"[prepare_swe_gym_293] Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    import io

    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(resp.content))
    return table.to_pandas()


def convert_split(
    url: str,
    *,
    fork_python: str,
    agent_name: str = "mini_swe_agent",
    total: int | None = None,
) -> pd.DataFrame:
    """Download one SkyRL split, generate eval_scripts, and convert to PSRL rows."""
    df = _download_parquet(url)
    if total is not None:
        df = df.head(total)
    print(f"[prepare_swe_gym_293] Loaded {len(df)} instances.")

    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(df.to_dict("records")):
        instance = _normalize(row["instance"])
        if not isinstance(instance, dict):
            raise ValueError(f"row {idx} has non-dict instance field: {type(instance)}")
        if not instance.get("problem_statement", ""):
            print(f"[prepare_swe_gym_293] Skipping {instance.get('instance_id', '?')}: empty problem_statement.")
            continue
        eval_script = _make_eval_script(instance, fork_python)
        rows.append(_build_row(instance, eval_script=eval_script, agent_name=agent_name))
        if (idx + 1) % 50 == 0:
            print(f"[prepare_swe_gym_293] Processed {idx + 1}/{len(df)}...")

    out = pd.DataFrame(rows)
    print(f"[prepare_swe_gym_293] Converted {len(out)} rows.")
    return out


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SkyRL-v0-293 to PSRL parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", default="examples/mini_swe/data/swe_gym_293")
    parser.add_argument("--fork-venv", default="/tmp/swegym-fork-venv", help="SWE-Bench-Fork venv path.")
    parser.add_argument(
        "--ensure-fork",
        action="store_true",
        help="Create the fork venv and install the SWE-Bench-Fork if missing.",
    )
    parser.add_argument("--force-fork", action="store_true", help="Recreate the fork venv from scratch.")
    parser.add_argument("--agent-name", default="mini_swe_agent")
    parser.add_argument("--train-total", type=int, default=None, help="Cap the number of train rows.")
    parser.add_argument("--val-total", type=int, default=None, help="Cap the number of validation rows.")
    args = parser.parse_args()

    fork_python = os.path.join(os.path.abspath(args.fork_venv), "bin", "python")
    if args.ensure_fork or args.force_fork:
        fork_python = ensure_fork_venv(args.fork_venv, force=args.force_fork)
    elif not os.path.isfile(fork_python):
        raise SystemExit(f"fork venv python not found at {fork_python}. Re-run with --ensure-fork to create it.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = convert_split(
        TRAIN_PARQUET_URL, fork_python=fork_python, agent_name=args.agent_name, total=args.train_total
    )
    train.to_parquet(out_dir / "train.parquet")
    print(f"Wrote {len(train)} train rows to {out_dir / 'train.parquet'}.")

    try:
        val = convert_split(VAL_PARQUET_URL, fork_python=fork_python, agent_name=args.agent_name, total=args.val_total)
        val.to_parquet(out_dir / "val.parquet")
        print(f"Wrote {len(val)} validation rows to {out_dir / 'val.parquet'}.")
    except Exception as exc:  # noqa: BLE001 (validation split is optional)
        psrl_logger.warning(f"Validation split conversion failed (continuing): {exc}")

    # Summary
    for split_name in ("train", "val"):
        p = out_dir / f"{split_name}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            print(f"{split_name}: 0 rows (empty)")
            continue
        row0 = df.iloc[0]
        ei = row0["extra_info"]
        print(
            f"{split_name}: data_source={row0['data_source']!r} "
            f"image={ei['swe_problem_image']!r} "
            f"F2P={len(row0['reward_model']['ground_truth']['FAIL_TO_PASS'])} "
            f"P2P={len(row0['reward_model']['ground_truth']['PASS_TO_PASS'])}"
        )
        repos = df["extra_info"].apply(lambda e: e["swe_problem"]["repo"])
        print(
            f"  repos ({len(repos.unique())}): {', '.join(sorted(repos.unique())[:8])}"
            f"{' ...' if len(repos.unique()) > 8 else ''}"
        )


if __name__ == "__main__":
    main()
