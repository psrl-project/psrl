#!/usr/bin/env python3
"""Plot the two 70B runs' rollout, GenRM, and harmonic throughput CSVs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "logs/verl_deployment_modes/70b_harmonic_throughput"

RUNS = (
    (
        "Disaggregated",
        "disaggregated_harmonic_throughput.csv",
        "#2166ac",
    ),
    (
        "Elastic RL",
        "elastic_rl_harmonic_throughput.csv",
        "#b2182b",
    ),
)
PLOTS = (
    (
        "rollout_total_throughput",
        "Rollout total throughput",
        "rollout_throughput_vs_time.png",
    ),
    (
        "rm_total_throughput",
        "GenRM total throughput",
        "genrm_throughput_vs_time.png",
    ),
    (
        "harmonic_throughput",
        "Harmonic throughput",
        "harmonic_throughput_vs_time.png",
    ),
)


@dataclass(frozen=True)
class RunData:
    wall_time_s: list[float]
    steps: list[int]
    metrics: dict[str, list[float]]


def read_csv(path: Path) -> RunData:
    metrics = {column: [] for column, _, _ in PLOTS}
    wall_time_s: list[float] = []
    steps: list[int] = []
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"wall_time_s", "step", *metrics}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Missing columns {sorted(missing)} in {path}")
        for row in reader:
            wall_time_s.append(float(row["wall_time_s"]))
            steps.append(int(row["step"]))
            for column in metrics:
                metrics[column].append(float(row[column]))
    if not wall_time_s:
        raise RuntimeError(f"No data rows found in {path}")
    return RunData(wall_time_s=wall_time_s, steps=steps, metrics=metrics)


def select_steps(data: RunData, step_start: int, step_end: int) -> RunData:
    selected = [
        index
        for index, step in enumerate(data.steps)
        if step_start <= step <= step_end
    ]
    if not selected:
        raise RuntimeError(f"No samples found for steps {step_start}-{step_end}")
    time_origin = data.wall_time_s[selected[0]]
    return RunData(
        wall_time_s=[data.wall_time_s[index] - time_origin for index in selected],
        steps=[data.steps[index] for index in selected],
        metrics={
            metric: [values[index] for index in selected]
            for metric, values in data.metrics.items()
        },
    )


def plot_metric(
    run_data: list[tuple[str, RunData, str]],
    metric: str,
    title: str,
    output_path: Path,
    step_range: tuple[int, int] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    fig, axis = plt.subplots(figsize=(12, 4.8), layout="constrained")
    for label, data, color in run_data:
        axis.plot(
            data.wall_time_s,
            data.metrics[metric],
            color=color,
            linewidth=0.85,
            alpha=0.9,
            label=label,
        )

    step_suffix = "" if step_range is None else f" (steps {step_range[0]}-{step_range[1]})"
    axis.set_title(f"70B {title} vs time{step_suffix}")
    if step_range is None:
        axis.set_xlabel("Wall time since training start (s)")
    else:
        axis.set_xlabel(f"Elapsed time since step {step_range[0]} start (s)")
    axis.set_ylabel("Throughput (tokens/s)")
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)
    axis.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axis.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--step-start", type=int)
    parser.add_argument("--step-end", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    if (args.step_start is None) != (args.step_end is None):
        raise SystemExit("--step-start and --step-end must be specified together")
    if args.step_start is not None and (
        args.step_start <= 0 or args.step_start > args.step_end
    ):
        raise SystemExit("Step range must satisfy 1 <= --step-start <= --step-end")
    step_range = (
        None
        if args.step_start is None
        else (args.step_start, args.step_end)
    )
    filename_suffix = (
        "" if step_range is None else f"_steps_{step_range[0]}-{step_range[1]}"
    )
    run_data = []
    for label, filename, color in RUNS:
        path = input_dir / filename
        if not path.is_file():
            raise SystemExit(f"Input CSV not found: {path}")
        data = read_csv(path)
        if step_range is not None:
            data = select_steps(data, *step_range)
        run_data.append((label, data, color))

    for metric, title, filename in PLOTS:
        output_path = output_dir / filename.replace(".png", f"{filename_suffix}.png")
        plot_metric(run_data, metric, title, output_path, step_range)
        print(output_path)


if __name__ == "__main__":
    main()
