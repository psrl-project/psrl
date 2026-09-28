"""
Build a parquet dataset from SciBuddy Harbor task packages.

Each task package is a .tar.gz containing:
task.toml (task identity), instruction.md (task prompt), bindings.json
(data mounts), tests/ (verifier), environment/ (Dockerfile).

The parquet schema mirrors sciaccel_rl so it works with the same PSRL data
pipeline. The `prompt` column carries the instruction text, but Harbor
re-reads instruction.md from disk, so it is informational only. All
task-specific fields go in `extra_info`.

Usage:
    python examples/scibuddy_rollout/prepare/build_dataset.py \\
        --packages /path/to/job-*.tar.gz \\
        --output examples/scibuddy_rollout/data/scibuddy.parquet

The packages are extracted to a sibling `tasks/` directory next to the
output file so PSRL workers can reach them by absolute path.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd


def _task_name_from_toml(toml_text: str) -> str:
    """Extract task name from task.toml without a TOML parser dependency."""
    for line in toml_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("name"):
            parts = stripped.split("=", 1)
            if len(parts) == 2:
                return parts[1].strip().strip('"').strip("'")
    return "unknown"


def _qualified_name(name: str, org: str) -> str:
    """Return ``name`` in org/name form, adding ``org`` only when it is absent."""
    return name if "/" in name else f"{org}/{name}"


def _read_member(tf: tarfile.TarFile, name: str) -> str:
    """Read a text member from a tarfile."""
    member = tf.getmember(name)
    f = tf.extractfile(member)
    if f is None:
        raise ValueError(f"Cannot read {name}")
    return f.read().decode()


def _pkg_hash(path: Path) -> str:
    """Stable short identifier for a package by its filename stem."""
    return hashlib.sha256(path.name.encode()).hexdigest()[:12]


def _patch_task_toml(task_toml: Path, org: str) -> None:
    """Prefix task.name with org/ when it is not already in org/name format.

    Harbor 0.22.0 validates that task.name matches "org/name" pattern. Packages
    built by ScienceInfra use a plain slug, so this adds the org prefix in place
    after extraction.

    Args:
        task_toml: Path to the task.toml file to patch.
        org: Organisation prefix to prepend when the name lacks a slash.
    """
    if not task_toml.exists():
        return
    text = task_toml.read_text()
    lines = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("name") and "=" in stripped and "/" not in stripped:
            _, _, val = stripped.partition("=")
            name = val.strip().strip('"').strip("'")
            line = line.replace(val.strip(), f'"{_qualified_name(name, org)}"')
        lines.append(line)
    task_toml.write_text("".join(lines))


def build_row(package_path: Path, task_dir: Path) -> dict:
    """Build one dataset row from a package tarball.

    Args:
        package_path: Absolute path to the .tar.gz.
        task_dir: Directory where the package was extracted.

    Returns:
        A dict with columns matching the sciaccel_rl parquet schema.
    """
    with tarfile.open(package_path) as tf:
        task_toml = _read_member(tf, "package/task.toml")
        instruction = _read_member(tf, "package/instruction.md")
        bindings_raw = _read_member(tf, "package/bindings.json")

    # Read the name from the extracted copy, which `_patch_task_toml` has already
    # qualified. Reading the tarball instead would miss the prefix and double it.
    extracted_toml = task_dir / "package" / "task.toml"
    if extracted_toml.exists():
        task_toml = extracted_toml.read_text()
    task_name = _qualified_name(_task_name_from_toml(task_toml), "scibuddy")
    bindings = json.loads(bindings_raw)

    # Prompt carries the raw instruction for human review and offline analysis.
    # Harbor re-reads instruction.md from disk so this has no effect on rollout.
    prompt = [{"role": "user", "content": instruction}]

    extra_info = {
        "task_path": str((task_dir / "package").resolve()),
        "task_name": task_name,
        "reward_key": "reward",
        "bindings": bindings,
        # Package hash doubles as a stable uid for deduplication.
        "package_id": _pkg_hash(package_path),
    }

    return {
        "prompt": prompt,
        "data_source": "scibuddy_rollout",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "task_name": task_name,
        "extra_info": extra_info,
    }


def build_dataset(package_globs: list[str], output: Path) -> None:
    """Extract packages and build a parquet.

    Args:
        package_globs: Glob patterns that each expand to .tar.gz paths.
        output: Destination .parquet file.
    """
    packages: list[Path] = []
    for pattern in package_globs:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"WARNING: no files matched {pattern!r}")
        for p in matched:
            packages.append(Path(p).resolve())

    if not packages:
        raise SystemExit("No packages found. Check --packages.")

    tasks_dir = output.parent / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pkg in packages:
        pkg_id = _pkg_hash(pkg)
        task_dir = tasks_dir / pkg_id
        if not task_dir.exists():
            print(f"Extracting {pkg.name} -> {task_dir}")
            task_dir.mkdir(parents=True)
            with tarfile.open(pkg) as tf:
                tf.extractall(task_dir)
            # Harbor requires task.name in "org/name" format. Packages from the
            # ScienceInfra pipeline use a plain slug, so prefix it here.
            _patch_task_toml(task_dir / "package" / "task.toml", "scibuddy")
        else:
            print(f"Already extracted: {task_dir}")

        row = build_row(pkg, task_dir)
        rows.append(row)
        print(f"  task={row['task_name']}")

    df = pd.DataFrame(rows)
    df.to_parquet(output, index=False)
    print(f"\nWrote {len(df)} tasks to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        nargs="+",
        required=True,
        metavar="GLOB",
        help="Glob pattern(s) for Harbor package tarballs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/scibuddy_rollout/data/scibuddy.parquet"),
        help="Output parquet file.",
    )
    args = parser.parse_args()
    build_dataset(args.packages, args.output)


if __name__ == "__main__":
    main()
