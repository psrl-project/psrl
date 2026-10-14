"""
Stage 1 and 2 of AIRS-Bench data preparation, plus dataset yaml rewriting.

Stage 1 downloads the raw HuggingFace datasets. Stage 2 runs each task's own
`prepare.py` and `evaluate_prepare.py`, which produce the train, test, and
test_with_labels directories that the in-sandbox `evaluate.py` reads. Both stages
are cached, so re-running only fills gaps.

Usage:
    python -m examples.airs_bench.prepare.prepare_airs_data \\
        --airs-repo /apdcephfs_zwfy10/share_303541817/lhy/science_infra/airs-bench \\
        --data-root /apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

META_INTERNAL_PREFIX = "/checkpoint/maui/shared/datasets/airs_text_only_prepared"


def load_task_metadata(rad_task_dir: Path) -> dict[str, Any]:
    """
    Read one task's normalization constants from its AIRS-Bench metadata.

    Args:
        rad_task_dir (Path): Directory holding the task's `metadata.yaml`.

    Returns:
        dict[str, Any]: Keys metric, metric_lower_is_better, sota_score,
            estimated_worst_score, optimal_score, and category.

    Raises:
        ValueError: If the metadata carries no SOTA entry, which would make the
            reward unnormalizable.
    """
    metadata = yaml.safe_load((rad_task_dir / "metadata.yaml").read_text())
    logging_info = metadata.get("logging_info", {})
    sota_entries = logging_info.get("sota") or []
    if not sota_entries or sota_entries[0].get("sota_score") is None:
        raise ValueError(f"Task {rad_task_dir.name} carries no sota score in its metadata.")

    return {
        "metric": logging_info["metric"],
        "metric_lower_is_better": bool(metadata["metric_lower_is_better"]),
        "sota_score": float(sota_entries[0]["sota_score"]),
        "estimated_worst_score": float(logging_info["estimated_worst_score"]),
        "optimal_score": float(logging_info["optimal_score"]),
        "category": logging_info.get("category", "Unknown"),
    }


def rewrite_dataset_yaml(src_yaml: Path, dst_yaml: Path, new_data_path: str) -> None:
    """
    Copy one dataset yaml, repointing `data_path` at our prepared data.

    Every other key is preserved verbatim, because the description block encodes the
    dataset schema MLGym shows the agent.

    Args:
        src_yaml (Path): Source dataset yaml from the AIRS-Bench repo.
        dst_yaml (Path): Destination path.
        new_data_path (str): Replacement value for `data_path`.
    """
    payload = yaml.safe_load(src_yaml.read_text())
    payload["data_path"] = new_data_path
    dst_yaml.parent.mkdir(parents=True, exist_ok=True)
    dst_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
    psrl_logger.info(f"Updated data_path={new_data_path!r} in {dst_yaml!s}.")


def run_stage_one(airs_repo: Path, raw_dir: Path) -> None:
    """
    Download the raw HuggingFace datasets, skipping work already cached.

    Two environment adjustments are needed:

    1. `HF_ENDPOINT` is set to `https://huggingface.co` because the default mirror
       (`hf-mirror.com`) redirects blob downloads to `huggingface.co`, which
       `huggingface_hub>=1.16` rejects as a security check.

    2. The airs-bench download script requires `datasets==3.6.0` (for `trust_remote_code`
       support on dataset scripts). The psrl env has `datasets==5.0.1`. We install
       `datasets==3.6.0` into a side directory with `--target` and prepend it to
       `PYTHONPATH` so the download subprocess picks it up without modifying the main
       environment.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    marker = raw_dir / ".download_complete"
    if marker.exists():
        psrl_logger.info(f"Stage 1 already complete at {raw_dir}, skipping.")
        return

    # Install datasets==3.6.0 to a side directory if needed.
    compat_dir = raw_dir.parent / "hf_compat_env"
    compat_marker = compat_dir / ".installed"
    if not compat_marker.exists():
        compat_dir.mkdir(parents=True, exist_ok=True)
        psrl_logger.info(f"Installing datasets==3.6.0 into {compat_dir}...")
        install = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--target",
                str(compat_dir),
                "datasets==3.6.0",
            ],
        )
        if install.returncode != 0:
            raise RuntimeError(f"Failed to install datasets==3.6.0 into {compat_dir}.")
        compat_marker.touch()

    script = airs_repo / "datasets" / "download_hf_datasets.sh"
    psrl_logger.info(f"Running stage 1 download into {raw_dir}...")
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = f"{compat_dir}:{existing_pythonpath}" if existing_pythonpath else str(compat_dir)
    env = {
        **os.environ,
        "HF_ENDPOINT": "https://huggingface.co",
        "PYTHONPATH": new_pythonpath,
    }
    completed = subprocess.run(
        ["bash", str(script), str(raw_dir)],
        cwd=str(airs_repo),
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Stage 1 download failed with exit code {completed.returncode}.")

    # The download script can report a partial failure despite complete data on disk.
    # Validate the expected directories before treating its failure log as fatal.
    failed_log = airs_repo / "failed_datasets.txt"
    if failed_log.exists() and failed_log.stat().st_size > 0:
        csv_path = airs_repo / "datasets" / "hf_datasets.csv"

        expected: dict[str, str] = {}
        with csv_path.open() as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            for row in reader:
                if len(row) >= 2:
                    expected[f"{row[0].strip()}/{row[1].strip()}"] = row[0].strip()

        missing = [name for name in expected if not (raw_dir / name).exists()]
        if missing:
            raise RuntimeError(
                f"Stage 1 download logged failures and {len(missing)} dataset(s) "
                f"are missing from disk: {missing!r}. Check {failed_log}."
            )
        psrl_logger.info("Stage 1 logged failures but all expected dataset directories exist on disk.")
    marker.touch()


def run_stage_two(
    airs_repo: Path,
    raw_dir: Path,
    prepared_dir: Path,
    task_ids: list[str],
) -> None:
    """
    Run each task's own `prepare.py` and `evaluate_prepare.py` to build the train,
    test, and test_with_labels splits.

    `evaluate_prepare.py` requires a submission file as input. A trivial dummy
    submission (one constant row per test row) is built to satisfy that requirement.
    The resulting `test_with_labels` directory is what the in-container `evaluate.py`
    reads at episode time.
    """
    for task_id in task_ids:
        task_out = prepared_dir / task_id
        marker = task_out / ".prepare_complete"
        if marker.exists():
            psrl_logger.info(f"Stage 2 already complete for {task_id}, skipping.")
            continue

        rad_dir = airs_repo / "airsbench" / "tasks" / "rad" / task_id
        task_out.mkdir(parents=True, exist_ok=True)
        log_dir = task_out / "logs"
        log_dir.mkdir(exist_ok=True)

        script = rad_dir / "prepare.py"
        if not script.exists():
            raise FileNotFoundError(f"Expected {script} to exist for task {task_id}.")
        psrl_logger.info(f"Running prepare.py for {task_id}...")
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--global-shared-data-dir",
                str(raw_dir),
                "--agent-data-mount-dir",
                str(task_out),
                "--agent-log-dir",
                str(log_dir),
            ],
            cwd=str(rad_dir),
        )
        if completed.returncode != 0:
            raise RuntimeError(f"prepare.py failed for task {task_id} with exit code {completed.returncode}.")

        # `evaluate_prepare.py` reads a placeholder submission from `agent_log_dir`
        # before producing the labels used by the in-container evaluator.
        dummy_sub = log_dir / "submission.csv"
        if not dummy_sub.exists():
            build_dummy_submission = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import csv, sys\n"
                        "from datasets import load_from_disk\n"
                        f"ds = load_from_disk({str(task_out / 'test')!r})\n"
                        f"handle = open({str(dummy_sub)!r}, 'w', newline='')\n"
                        "writer = csv.writer(handle)\n"
                        "writer.writerow(['prediction'])\n"
                        "[writer.writerow([0]) for _ in range(len(ds))]\n"
                        "handle.close()\n"
                    ),
                ],
                cwd=str(rad_dir),
                capture_output=True,
                text=True,
            )
            if build_dummy_submission.returncode != 0:
                psrl_logger.warning(
                    f"Could not build a dummy submission for {task_id}: {build_dummy_submission.stderr[:200]}."
                )

        eval_prepare = rad_dir / "evaluate_prepare.py"
        if eval_prepare.exists():
            psrl_logger.info(f"Running evaluate_prepare.py for {task_id}...")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(eval_prepare),
                    "--global-shared-data-dir",
                    str(raw_dir),
                    "--agent-data-mount-dir",
                    str(task_out),
                    "--agent-log-dir",
                    str(log_dir),
                ],
                cwd=str(rad_dir),
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                psrl_logger.warning(f"evaluate_prepare.py failed for {task_id}: {completed.stderr[:200]}.")

        for required in ("train", "test", "test_with_labels"):
            if not (task_out / required).exists():
                raise RuntimeError(f"Task {task_id} is missing the {required} directory after stage 2.")
        marker.touch()


def run_stage_three(
    airs_repo: Path,
    prepared_dir: Path,
    config_out: Path,
    task_ids: list[str],
) -> None:
    """Emit task and dataset yamls with data paths repointed at our prepared data."""
    for task_id in task_ids:
        mlgym_dir = airs_repo / "airsbench" / "tasks" / "mlgym" / task_id / "configs"

        task_src = mlgym_dir / "tasks" / f"{task_id}.yaml"
        task_dst = config_out / "tasks" / f"{task_id}.yaml"
        task_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(task_src, task_dst)

        for dataset_yaml in sorted((mlgym_dir / "datasets").glob("*.yaml")):
            rewrite_dataset_yaml(
                dataset_yaml,
                config_out / "datasets" / dataset_yaml.name,
                str(prepared_dir / task_id),
            )

        data_src = airs_repo / "airsbench" / "tasks" / "mlgym" / task_id / "data"
        data_dst = config_out / "data" / task_id
        if data_dst.exists():
            shutil.rmtree(data_dst)
        shutil.copytree(data_src, data_dst)

        # APPS requires pyext which does not support Python 3.12 (uses inspect.getargspec).
        # Provide a compatibility stub in the staged data directory.
        if task_id == "CodeGenerationAPPSPassAt5":
            (data_dst / "pyext.py").write_text(
                '"""Compatibility stub for pyext on Python 3.12+."""\n'
                "import types\n\n\n"
                "class RuntimeModule:\n"
                "    @staticmethod\n"
                "    def from_string(name: str, docstring: str, code: str) -> types.ModuleType:\n"
                "        module = types.ModuleType(name)\n"
                "        exec(compile(code, name, 'exec'), module.__dict__)  # noqa: S102\n"
                "        return module\n"
            )
            psrl_logger.info(f"Installed pyext compatibility stub for {task_id}.")

        psrl_logger.info(f"Staged configs and starter code for {task_id}.")


def main() -> None:
    """Run all three preparation stages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepare AIRS-Bench data for PSRL.")
    parser.add_argument("--airs-repo", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    raw_dir = args.data_root / "airs_raw"
    prepared_dir = args.data_root / "airs_prepared"
    config_out = args.data_root / "airs_configs"

    rad_root = args.airs_repo / "airsbench" / "tasks" / "rad"
    task_ids = sorted(path.name for path in rad_root.iterdir() if path.is_dir())
    psrl_logger.info(f"Preparing AIRS-Bench tasks. Count: {len(task_ids)}.")

    run_stage_one(args.airs_repo, raw_dir)
    run_stage_two(args.airs_repo, raw_dir, prepared_dir, task_ids)
    run_stage_three(args.airs_repo, prepared_dir, config_out, task_ids)
    psrl_logger.info("AIRS-Bench data preparation complete!")


if __name__ == "__main__":
    main()
