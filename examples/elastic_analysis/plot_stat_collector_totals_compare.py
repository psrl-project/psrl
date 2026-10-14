#!/usr/bin/env python3
"""Plot role totals from per-instance StatCollector logs.

This is the StatCollector counterpart of plot_elastic_monitor_totals_compare.py.
It overlays an elastic run with an optional non-elastic run, using the snapshot
timestamp embedded in each StatCollector record to align instances.  The five
subplots are total generation throughput, running requests, waiting requests,
active StatCollector instances, and resident tokens.  Router backlog is not
available in StatCollector logs, so it intentionally is not inferred here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Parse / plot tuning (edit here) ---
PLOT_WINDOW_START_MIN = 0.0
PLOT_WINDOW_END_MIN = 300.0
PLOT_ELASTIC_ONLY = False
# A collector is considered active only while it continues emitting snapshots.
# This prevents stopped/slept instances from being forward-filled indefinitely.
ACTIVE_SNAPSHOT_MAX_AGE_SEC = 3.0
AGGREGATE_TS_TO_SECOND = True

PLOT_YLIM_THROUGHPUT: tuple[float, float] | None = None
PLOT_YLIM_RUNNING: tuple[float, float] | None = None
PLOT_YLIM_WAITING: tuple[float, float] | None = None
PLOT_YLIM_ACTIVE: tuple[float, float] | None = None
PLOT_YLIM_TOKENS: tuple[float, float] | None = None

FIGSIZE = (13, 14)
SAVE_DPI = 400
LINE_WIDTH = 1.0

ROLES = ("RewardModel", "Rollout")
METRICS = ("throughput", "running", "waiting", "active", "tokens")

COLOR_MIN4_REWARDMODEL = "#f1948a"
COLOR_MIN4_ROLLOUT = "#7fb3d5"
COLOR_MIN0_REWARDMODEL = "#922b21"
COLOR_MIN0_ROLLOUT = "#1a5276"

NON_ELASTIC_LOG_DIR = (
    REPO_ROOT
    / "logs/verl_deployment_modes/mixed_test_none_request_level_candidate_evaluation_mode5_bs_128_share_32_elastic_rl_Qwen2.5-32B_Qwen3-30B-A3B-Thinking-2507"
)
ELASTIC_LOG_DIR = (
    REPO_ROOT
    / "logs/verl_deployment_modes/mixed_test_new_request_level_candidate_evaluation_mode5_bs_128_share_32_elastic_rl_Qwen2.5-32B_Qwen3-30B-A3B-Thinking-2507"
)
_OUT_TAG = "elastic_only" if PLOT_ELASTIC_ONLY else "compare_none_vs_new"
OUT_PATH = (
    REPO_ROOT
    / f"logs/verl_deployment_modes/{_OUT_TAG}_none_new/"
    f"StatCollector_none_new_share_32_totals_{PLOT_WINDOW_START_MIN}-{PLOT_WINDOW_END_MIN}_min.png"
)

RE_TIMESTAMP = re.compile(r"'timestamp':\s*'(?P<timestamp>[^']+)'")
RE_THROUGHPUT = re.compile(r"'generation_throughput':\s*(?:np\.float64\()?([\d.eE+-]+)")
RE_NUM_RUNNING = re.compile(r"'num_running_reqs':\s*(\d+)")
RE_NUM_WAITING = re.compile(r"'num_waiting_reqs':\s*(\d+)")
RE_PROMPT_SECTION = re.compile(
    r"'req_id_to_prompt_token_num':\s*\{(.*?)\},\s*'req_id_to_response_token_num'"
)
RE_RESP_SECTION = re.compile(
    r"'req_id_to_response_token_num':\s*\{(.*?)\},\s*'req_id_in_waiting'"
)
RE_DICT_INT_VALS = re.compile(r":\s*(\d+)")
RE_ROLLOUT = re.compile(r"^StatCollector_I(\d+)\.log$")
RE_RM = re.compile(r"^StatCollector_RM_.+_I(\d+)\.log$")


@dataclass(frozen=True)
class Snapshot:
    timestamp: datetime
    throughput: float
    running: float
    waiting: float
    tokens: float


def _normalize_ts(timestamp: datetime) -> datetime:
    return timestamp.replace(microsecond=0) if AGGREGATE_TS_TO_SECOND else timestamp


def _sum_dict_int_values(section: str) -> int:
    return sum(int(value) for value in RE_DICT_INT_VALS.findall(section))


def _total_tokens(line: str) -> int:
    total = 0
    if match := RE_PROMPT_SECTION.search(line):
        total += _sum_dict_int_values(match.group(1))
    if match := RE_RESP_SECTION.search(line):
        total += _sum_dict_int_values(match.group(1))
    return total


def _parse_snapshot(line: str) -> Snapshot | None:
    if "Snapshot (model version" not in line:
        return None
    timestamp = RE_TIMESTAMP.search(line)
    throughput = RE_THROUGHPUT.search(line)
    running = RE_NUM_RUNNING.search(line)
    waiting = RE_NUM_WAITING.search(line)
    if not (timestamp and throughput and running and waiting):
        return None
    try:
        snapshot = Snapshot(
            timestamp=_normalize_ts(datetime.fromisoformat(timestamp.group("timestamp"))),
            throughput=float(throughput.group(1)),
            running=float(running.group(1)),
            waiting=float(waiting.group(1)),
            tokens=float(_total_tokens(line)),
        )
    except ValueError:
        return None
    return snapshot if math.isfinite(snapshot.throughput) else None


def _classify_log(path: Path) -> tuple[str, int] | None:
    if match := RE_ROLLOUT.match(path.name):
        return "Rollout", int(match.group(1))
    if match := RE_RM.match(path.name):
        return "RewardModel", int(match.group(1))
    return None


def _parse_instance_log(path: Path) -> dict[datetime, Snapshot]:
    # Multiple high-frequency snapshots can share one second. Keep the latest.
    points: dict[datetime, Snapshot] = {}
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            snapshot = _parse_snapshot(line)
            if snapshot is not None:
                points[snapshot.timestamp] = snapshot
    return points


def prepare_series(log_dir: Path) -> tuple[list[float], dict[str, dict[str, list[float]]]]:
    """Aggregate all StatCollector instances in ``log_dir`` by role and timestamp."""
    log_dir = log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")

    instances: dict[str, dict[int, dict[datetime, Snapshot]]] = {role: {} for role in ROLES}
    all_times: set[datetime] = set()
    for path in sorted(log_dir.glob("StatCollector*.log")):
        metadata = _classify_log(path)
        if metadata is None:
            continue
        role, instance_id = metadata
        points = _parse_instance_log(path)
        if not points:
            print(f"Skip (no valid snapshots): {path.name}")
            continue
        instances[role][instance_id] = points
        all_times.update(points)

    if not all_times:
        raise SystemExit(f"No valid StatCollector snapshots found under {log_dir}")

    all_times_sorted = sorted(all_times)
    t0 = all_times_sorted[0]
    if not any(
        PLOT_WINDOW_START_MIN
        <= (timestamp - t0).total_seconds() / 60.0
        <= PLOT_WINDOW_END_MIN
        for timestamp in all_times_sorted
    ):
        raise SystemExit(f"No StatCollector samples in plot window: {log_dir}")

    data = {role: {metric: [] for metric in METRICS} for role in ROLES}
    latest: dict[str, dict[int, Snapshot]] = {role: {} for role in ROLES}
    x_values: list[float] = []
    for timestamp in all_times_sorted:
        for role in ROLES:
            for instance_id, points in instances[role].items():
                if timestamp in points:
                    latest[role][instance_id] = points[timestamp]
            active = [
                point
                for point in latest[role].values()
                if 0 <= (timestamp - point.timestamp).total_seconds() <= ACTIVE_SNAPSHOT_MAX_AGE_SEC
            ]
            elapsed_min = (timestamp - t0).total_seconds() / 60.0
            if PLOT_WINDOW_START_MIN <= elapsed_min <= PLOT_WINDOW_END_MIN:
                data[role]["throughput"].append(sum(point.throughput for point in active))
                data[role]["running"].append(sum(point.running for point in active))
                data[role]["waiting"].append(sum(point.waiting for point in active))
                data[role]["active"].append(float(len(active)))
                data[role]["tokens"].append(sum(point.tokens for point in active))
        if PLOT_WINDOW_START_MIN <= (timestamp - t0).total_seconds() / 60.0 <= PLOT_WINDOW_END_MIN:
            x_values.append((timestamp - t0).total_seconds() / 60.0)

    return x_values, data


def plot_merged(
    elastic: tuple[list[float], dict[str, dict[str, list[float]]]],
    out_path: Path,
    ylims: dict[str, tuple[float, float] | None],
    non_elastic: tuple[list[float], dict[str, dict[str, list[float]]]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    metric_specs = [
        ("throughput", "Total Generation Throughput (tokens/s)"),
        ("running", "Total Running Requests"),
        ("waiting", "Total Waiting Requests"),
        ("active", "Active StatCollector Instances"),
        ("tokens", "Total Resident Tokens (prompt + response)"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    curves = []
    if non_elastic is not None:
        curves.extend([
            ("min_4 (non-elastic)", non_elastic, "RewardModel", COLOR_MIN4_REWARDMODEL),
            ("min_4 (non-elastic)", non_elastic, "Rollout", COLOR_MIN4_ROLLOUT),
        ])
    curves.extend([
        ("min_0 (elastic)", elastic, "RewardModel", COLOR_MIN0_REWARDMODEL),
        ("min_0 (elastic)", elastic, "Rollout", COLOR_MIN0_ROLLOUT),
    ])
    for axis, (metric, ylabel) in zip(axes, metric_specs):
        for run_label, (x_values, values), role, color in curves:
            axis.plot(x_values, values[role][metric], label=f"{role} ({run_label})", color=color,
                      linewidth=LINE_WIDTH, alpha=0.95)
        axis.set_ylabel(ylabel)
        axis.set_xlim(PLOT_WINDOW_START_MIN, PLOT_WINDOW_END_MIN)
        if ylims[metric] is not None:
            axis.set_ylim(*ylims[metric])
        axis.grid(True, alpha=0.35)
        axis.legend(loc="best", fontsize=8, ncol=2 if non_elastic else 1)
    axes[-1].set_xlabel("Elapsed time from first StatCollector snapshot (min)")
    fig.suptitle("StatCollector role totals: red = RewardModel, blue = Rollout")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)


def main() -> None:
    if PLOT_WINDOW_START_MIN > PLOT_WINDOW_END_MIN:
        raise SystemExit("PLOT_WINDOW_START_MIN must be <= PLOT_WINDOW_END_MIN.")
    ylims = {
        "throughput": PLOT_YLIM_THROUGHPUT,
        "running": PLOT_YLIM_RUNNING,
        "waiting": PLOT_YLIM_WAITING,
        "active": PLOT_YLIM_ACTIVE,
        "tokens": PLOT_YLIM_TOKENS,
    }
    elastic = prepare_series(ELASTIC_LOG_DIR)
    non_elastic = None if PLOT_ELASTIC_ONLY else prepare_series(NON_ELASTIC_LOG_DIR)
    plot_merged(elastic, OUT_PATH.resolve(), ylims, non_elastic)
    print(f"Elastic points: {len(elastic[0])}")
    if non_elastic is not None:
        print(f"Non-elastic points: {len(non_elastic[0])}")
    print(f"Wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
