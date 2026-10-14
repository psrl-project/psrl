"""
Verify every AIRS-Bench task can grade a reference submission.

For each task this:
1. Builds a task-appropriate trivial submission from the prepared test split.
2. Runs `evaluate_prepare.py` with that submission to produce `test_with_labels`.
3. Creates the hardcoded container path `/home/agent/workspace/data/test_with_labels`
   that `evaluate.py` reads (all scripts use this absolute path).
4. Runs `evaluate.py` and asserts the output is valid JSON with the expected metric key.

A task that fails here can never produce reward signal, so this must pass for all
20 tasks before any RL run.

Usage:
    python -m examples.airs_bench.prepare.verify_grading \\
        --data-root /apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data \\
        --airs-repo /apdcephfs_zwfy10/share_303541817/lhy/science_infra/airs-bench
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from examples.airs_bench.prepare.prepare_airs_data import load_task_metadata

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# All evaluate.py scripts hardcode this path, which mirrors the in-container layout.
AGENT_WORKSPACE = Path("/home/agent/workspace")


def build_submission(task_id: str, prepared: Path, workspace: Path) -> tuple[bool, str]:
    """
    Build a task-appropriate constant-prediction submission CSV.

    Returns:
        tuple[bool, str]: (success, error_message_or_empty).
    """
    submission = workspace / "submission.csv"

    # Tasks with special submission formats.
    if task_id == "QuestionAnsweringEli5RougeL":
        # Needs text predictions (not integers). Empty string as placeholder.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, csv\n"
                    "from datasets import load_from_disk\n"
                    f"ds = load_from_disk({str(prepared / 'test')!r})\n"
                    f"with open({str(submission)!r}, 'w', newline='') as handle:\n"
                    "    writer = csv.writer(handle)\n"
                    "    writer.writerow(['prediction'])\n"
                    "    for _ in range(len(ds)):\n"
                    "        writer.writerow(['placeholder'])\n"
                    "print(len(ds))\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Could not build ELI5 submission: {result.stderr[:300]}."
        return True, ""

    if task_id == "QuestionAnsweringDuoRCAccuracy":
        # Needs answer + has_answer columns. Build from test.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, csv\n"
                    "from datasets import load_from_disk\n"
                    f"ds = load_from_disk({str(prepared / 'test')!r})\n"
                    f"with open({str(submission)!r}, 'w', newline='') as handle:\n"
                    "    writer = csv.writer(handle)\n"
                    "    writer.writerow(['answer', 'has_answer'])\n"
                    "    for _ in range(len(ds)):\n"
                    "        writer.writerow(['', 'True'])\n"
                    "print(len(ds))\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Could not build DuoRC submission: {result.stderr[:300]}."
        return True, ""

    if task_id == "CodeRetrievalCodeXGlueMRR":
        # Needs query + rankings columns. Build from test/queries.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, csv, json\n"
                    "from datasets import load_from_disk\n"
                    f"ds = load_from_disk({str(prepared / 'test' / 'queries')!r})\n"
                    f"with open({str(submission)!r}, 'w', newline='') as handle:\n"
                    "    writer = csv.writer(handle)\n"
                    "    writer.writerow(['query', 'rankings'])\n"
                    "    for row in ds:\n"
                    "        writer.writerow([row['query'], json.dumps([])])\n"
                    "print(len(ds))\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Could not build CodeRetrieval submission: {result.stderr[:300]}."
        return True, ""

    if task_id == "CodeGenerationAPPSPassAt5":
        # Needs columns code1..code5. Build from test.
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, csv\n"
                    "from datasets import load_from_disk\n"
                    f"ds = load_from_disk({str(prepared / 'test')!r})\n"
                    f"with open({str(submission)!r}, 'w', newline='') as handle:\n"
                    "    writer = csv.writer(handle)\n"
                    "    writer.writerow(['code1','code2','code3','code4','code5'])\n"
                    "    for _ in range(len(ds)):\n"
                    "        writer.writerow(['# placeholder']*5)\n"
                    "print(len(ds))\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False, f"Could not build APPS submission: {result.stderr[:300]}."
        return True, ""

    # Generic single-column prediction submission for all other tasks.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, csv\n"
                "from datasets import load_from_disk\n"
                f"ds = load_from_disk({str(prepared / 'test')!r})\n"
                f"with open({str(submission)!r}, 'w', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['prediction'])\n"
                "    for _ in range(len(ds)):\n"
                "        writer.writerow([0])\n"
                "print(len(ds))\n"
            ),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, f"Could not build a reference submission: {result.stderr[:300]}."
    return True, ""


def verify_task(task_id: str, airs_repo: Path, data_root: Path) -> tuple[bool, str]:
    """
    Grade one reference submission for one task.

    Returns:
        tuple[bool, str]: Success flag and a human-readable detail string.
    """
    prepared = data_root / "airs_prepared" / task_id
    raw_dir = data_root / "airs_raw"
    metadata = load_task_metadata(airs_repo / "airsbench" / "tasks" / "rad" / task_id)
    eval_script = data_root / "airs_configs" / "data" / task_id / "evaluate.py"
    eval_prepare_script = airs_repo / "airsbench" / "tasks" / "rad" / task_id / "evaluate_prepare.py"

    if not eval_script.exists():
        return False, f"Missing evaluate.py at {eval_script}."
    if not eval_prepare_script.exists():
        return False, f"Missing evaluate_prepare.py at {eval_prepare_script}."

    workspace = data_root / "verify" / task_id
    workspace.mkdir(parents=True, exist_ok=True)
    log_dir = workspace / "logs"
    log_dir.mkdir(exist_ok=True)

    ok, err = build_submission(task_id, prepared, workspace)
    if not ok:
        return False, err

    submission = workspace / "submission.csv"

    # Copy submission to logs/ so evaluate_prepare.py can find it.
    shutil.copy(submission, log_dir / "submission.csv")

    # Run evaluate_prepare.py to produce test_with_labels in the verify workspace.
    eval_prepare = subprocess.run(
        [
            sys.executable,
            str(eval_prepare_script),
            "--global-shared-data-dir",
            str(raw_dir),
            "--agent-data-mount-dir",
            str(workspace),
            "--agent-log-dir",
            str(log_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(airs_repo / "airsbench" / "tasks" / "rad" / task_id),
    )
    if eval_prepare.returncode != 0:
        return False, f"evaluate_prepare.py failed: {eval_prepare.stderr[:300]}."

    test_with_labels = workspace / "test_with_labels"
    if not test_with_labels.exists():
        return False, "evaluate_prepare.py did not create test_with_labels."

    # Rebuild submissions when `test_with_labels` changes the row count.
    # Time series tasks also require JSON array predictions.
    ts_tasks = {
        "TimeSeriesForecastingKaggleWebTrafficMASE",
        "TimeSeriesForecastingRideshareMAE",
        "TimeSeriesForecastingSolarWeeklyMAE",
    }
    if task_id in ts_tasks:
        # Forecast horizons differ by task, and Rideshare nests 15 series per row.
        ts_forecast_horizon = {
            "TimeSeriesForecastingKaggleWebTrafficMASE": None,  # use label_len
            "TimeSeriesForecastingRideshareMAE": 48,
            "TimeSeriesForecastingSolarWeeklyMAE": 5,
        }
        horizon = ts_forecast_horizon[task_id]
        rebuild = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import csv, json\n"
                    "from datasets import load_from_disk\n"
                    f"twl = load_from_disk({str(test_with_labels)!r})\n"
                    "first_label = twl[0]['label_target']\n"
                    "is_nested = isinstance(first_label[0], list)\n"
                    f"horizon = {horizon!r}\n"
                    "if is_nested:\n"
                    "    flen = horizon if horizon else 48\n"
                    "    dummy = json.dumps([0] * flen)\n"
                    "    total = sum(len(item['label_target']) for item in twl)\n"
                    f"    with open({str(submission)!r}, 'w', newline='') as fh:\n"
                    "        writer = csv.writer(fh)\n"
                    "        writer.writerow(['prediction'])\n"
                    "        for _ in range(total):\n"
                    "            writer.writerow([dummy])\n"
                    "    print(f'Built nested TS submission: {total} rows x {flen}')\n"
                    "else:\n"
                    "    flen = horizon if horizon else len(first_label)\n"
                    "    dummy = json.dumps([0] * flen)\n"
                    "    needed = len(twl)\n"
                    f"    with open({str(submission)!r}, 'w', newline='') as fh:\n"
                    "        writer = csv.writer(fh)\n"
                    "        writer.writerow(['prediction'])\n"
                    "        for _ in range(needed):\n"
                    "            writer.writerow([dummy])\n"
                    "    print(f'Built flat TS submission: {needed} rows x {flen}')\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if rebuild.returncode != 0:
            return False, f"Could not build TS submission: {rebuild.stderr[:200]}."
        shutil.copy(submission, log_dir / "submission.csv")
    else:
        rebuild = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import csv\n"
                    "from datasets import load_from_disk\n"
                    f"twl = load_from_disk({str(test_with_labels)!r})\n"
                    f"current = sum(1 for _ in open({str(submission)!r})) - 1\n"
                    "needed = len(twl)\n"
                    "if current != needed:\n"
                    f"    with open({str(submission)!r}, 'w', newline='') as fh:\n"
                    "        writer = csv.writer(fh)\n"
                    "        writer.writerow(['prediction'])\n"
                    "        for _ in range(needed):\n"
                    "            writer.writerow([0])\n"
                    "    print(f'Rebuilt submission: {needed} rows')\n"
                    "else:\n"
                    "    print(f'Submission already correct: {current} rows')\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if rebuild.returncode != 0:
            return False, f"Could not rebuild submission: {rebuild.stderr[:200]}."
        shutil.copy(submission, log_dir / "submission.csv")

    # APPS evaluate.py runs actual code, which takes 20+ minutes for 5000 tests.
    # Truncate test_with_labels to 1 sample for verification speed.
    if task_id == "CodeGenerationAPPSPassAt5":
        truncate = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import shutil\n"
                    "from datasets import load_from_disk\n"
                    f"src = {str(test_with_labels)!r}\n"
                    f"tmp = {str(test_with_labels) + '_tmp'!r}\n"
                    "ds = load_from_disk(src)\n"
                    "ds.select([0]).save_to_disk(tmp)\n"
                    "shutil.rmtree(src)\n"
                    "shutil.move(tmp, src)\n"
                    "print('Truncated to 1 sample.')\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if truncate.returncode != 0:
            return False, f"Could not truncate APPS test_with_labels: {truncate.stderr[:200]}."
        # Also truncate the submission to match.
        trunc_sub = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import pandas as pd\n"
                    f"df = pd.read_csv({str(submission)!r}, header=0)\n"
                    f"df.head(1).to_csv({str(submission)!r}, index=False)\n"
                    "print('Truncated submission to 1 row.')\n"
                ),
            ],
            capture_output=True,
            text=True,
        )
        if trunc_sub.returncode != 0:
            return False, f"Could not truncate APPS submission: {trunc_sub.stderr[:200]}."

    # Create the hardcoded container path that all evaluate.py scripts read from.
    agent_data_dir = AGENT_WORKSPACE / "data"
    agent_data_dir.mkdir(parents=True, exist_ok=True)
    agent_twl = agent_data_dir / "test_with_labels"
    if agent_twl.is_symlink() or agent_twl.exists():
        agent_twl.unlink()
    agent_twl.symlink_to(test_with_labels.resolve())

    # evaluate.py uses --submission-file (dash, not underscore).
    # APPS needs cwd to be its data dir so it can import utils.py.
    if task_id == "CodeGenerationAPPSPassAt5":
        eval_cwd = str(eval_script.parent)
    else:
        eval_cwd = str(workspace)

    cmd = [sys.executable, str(eval_script), "--submission-file", str(submission)]

    completed = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=eval_cwd,
    )
    if completed.returncode != 0:
        return False, f"evaluate.py exited {completed.returncode}: {completed.stderr[:300]}."

    try:
        # Some evaluate.py scripts print multi-line JSON, others print it last.
        # Find the JSON block by scanning all lines.
        stdout = completed.stdout.strip()
        # Replace JavaScript NaN/Infinity with valid JSON equivalents before parsing.
        cleaned = stdout.replace(": NaN", ": null").replace(": Infinity", ": 1e308").replace(": -Infinity", ": -1e308")
        # Try the full output as JSON first.
        try:
            metrics = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find a JSON object in the output (the last { ... } block).
            start = cleaned.rfind("{")
            end = cleaned.rfind("}") + 1
            if start == -1 or end == 0:
                raise json.JSONDecodeError("No JSON object found", cleaned, 0) from None
            metrics = json.loads(cleaned[start:end])
    except (json.JSONDecodeError, IndexError):
        return False, f"evaluate.py output was not JSON: {completed.stdout[:300]}."

    if metadata["metric"] not in metrics:
        return False, f"Expected metric key {metadata['metric']!r}, got keys {sorted(metrics)}."
    return True, f"metric {metadata['metric']}={metrics[metadata['metric']]}"


def main() -> None:
    """Verify grading for every task and exit non-zero on any failure."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Verify AIRS-Bench grading.")
    parser.add_argument("--airs-repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    rad_root = args.airs_repo / "airsbench" / "tasks" / "rad"
    task_ids = sorted(path.name for path in rad_root.iterdir() if path.is_dir())

    failures: list[str] = []
    for task_id in task_ids:
        ok, detail = verify_task(task_id, args.airs_repo, args.data_root)
        status = "OK  " if ok else "FAIL"
        psrl_logger.info(f"[{status}] {task_id}: {detail}")
        if not ok:
            failures.append(task_id)

    psrl_logger.info(f"Verified {len(task_ids) - len(failures)}/{len(task_ids)} task(s).")
    if failures:
        psrl_logger.error(f"Tasks that cannot grade: {failures!r}.")
        raise SystemExit(1)
    psrl_logger.info("All AIRS-Bench tasks can grade a submission!")


if __name__ == "__main__":
    main()
