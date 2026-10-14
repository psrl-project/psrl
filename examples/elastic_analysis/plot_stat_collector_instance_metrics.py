#!/usr/bin/env python3
"""
Parse StatCollector logs (Rollout + RewardModel) and plot per-instance metrics over time.

Each (metric, role) pair produces one figure; all instances of that role are
overlaid on a single axes (one line per instance).

Metrics:
  - total_request_num  (num_running_reqs + num_waiting_reqs)
  - num_running_reqs
  - num_waiting_reqs
  - total_token_num    (sum of prompt + response tokens in scheduler_stats)
  - kv_cache_usage
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Config (edit here) ---
LOG_DIR = (
    REPO_ROOT
    / "logs/verl_deployment_modes/mode5_bs_256_share_24_elastic_rl_rollout7b_rm8b"
)
OUT_DIR = LOG_DIR / "stat_collector_plots"
# Inclusive plot range in minutes. None = no bound.
TIME_START_MIN: float | None = None
TIME_END_MIN: float | None = None
FIGSIZE = (18.0, 6.5)
SAVE_DPI = 400
LINE_WIDTH = 0.8
LEGEND_NCOL = 4

METRICS: tuple[tuple[str, str], ...] = (
    ("total_request_num", "Total Request Num"),
    ("num_running_reqs", "Running Request Num"),
    ("num_waiting_reqs", "Waiting Request Num"),
    ("total_token_num", "Total Token Num (prompt + response)"),
    ("kv_cache_usage", "KV Cache Usage"),
)

ROLE_OUTPUT_PREFIX = {
    "Rollout": "rollout",
    "RewardModel": "rm",
}

RE_ELAPSED = re.compile(r"'total_elapsed_time':\s*([\d.eE+-]+)")
RE_NUM_RUNNING = re.compile(r"'num_running_reqs':\s*(\d+)")
RE_NUM_WAITING = re.compile(r"'num_waiting_reqs':\s*(\d+)")
RE_KV_CACHE = re.compile(r"'kv_cache_usage':\s*([\d.eE+-]+)")
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
class SnapshotPoint:
    elapsed_sec: float
    total_request_num: int
    num_running_reqs: int
    num_waiting_reqs: int
    total_token_num: int
    kv_cache_usage: float


@dataclass(frozen=True)
class InstanceSeries:
    role: str
    instance_id: int
    label: str
    points: list[SnapshotPoint]


def _sum_dict_int_values(section: str) -> int:
    return sum(int(v) for v in RE_DICT_INT_VALS.findall(section))


def _compute_total_tokens(line: str) -> int:
    total = 0
    if m := RE_PROMPT_SECTION.search(line):
        total += _sum_dict_int_values(m.group(1))
    if m := RE_RESP_SECTION.search(line):
        total += _sum_dict_int_values(m.group(1))
    return total


def parse_snapshot_line(line: str) -> SnapshotPoint | None:
    if "Snapshot (model version" not in line:
        return None

    elapsed = RE_ELAPSED.search(line)
    num_running = RE_NUM_RUNNING.search(line)
    num_waiting = RE_NUM_WAITING.search(line)
    kv_cache = RE_KV_CACHE.search(line)
    if not (elapsed and num_running and num_waiting and kv_cache):
        return None

    try:
        elapsed_sec = float(elapsed.group(1))
        running = int(num_running.group(1))
        waiting = int(num_waiting.group(1))
        kv = float(kv_cache.group(1))
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for v in (elapsed_sec, kv)):
        return None

    return SnapshotPoint(
        elapsed_sec=elapsed_sec,
        total_request_num=running + waiting,
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        total_token_num=_compute_total_tokens(line),
        kv_cache_usage=kv,
    )


def parse_log_file(path: Path) -> list[SnapshotPoint]:
    points: list[SnapshotPoint] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            point = parse_snapshot_line(line)
            if point is not None:
                points.append(point)
    points.sort(key=lambda p: p.elapsed_sec)
    return points


def classify_log(path: Path) -> tuple[str, int] | None:
    name = path.name
    if m := RE_ROLLOUT.match(name):
        return "Rollout", int(m.group(1))
    if m := RE_RM.match(name):
        return "RewardModel", int(m.group(1))
    return None


def load_series(log_dir: Path) -> dict[str, list[InstanceSeries]]:
    grouped: dict[str, list[InstanceSeries]] = {"Rollout": [], "RewardModel": []}

    for path in sorted(log_dir.iterdir()):
        if not path.is_file() or path.suffix != ".log":
            continue
        meta = classify_log(path)
        if meta is None:
            continue
        role, instance_id = meta
        points = parse_log_file(path)
        if not points:
            print(f"Skip (no valid snapshots): {path.name}")
            continue

        if TIME_START_MIN is not None or TIME_END_MIN is not None:
            filtered: list[SnapshotPoint] = []
            for p in points:
                elapsed_min = p.elapsed_sec / 60.0
                if TIME_START_MIN is not None and elapsed_min < TIME_START_MIN:
                    continue
                if TIME_END_MIN is not None and elapsed_min > TIME_END_MIN:
                    continue
                filtered.append(p)
            points = filtered

        if not points:
            print(f"Skip (empty after time filter): {path.name}")
            continue

        short_role = "Rollout" if role == "Rollout" else "RM"
        grouped[role].append(
            InstanceSeries(
                role=role,
                instance_id=instance_id,
                label=f"{short_role} I{instance_id}",
                points=points,
            )
        )

    for role in grouped:
        grouped[role].sort(key=lambda s: s.instance_id)
    return grouped


def plot_role_metric_figure(
    instances: list[InstanceSeries],
    role: str,
    metric_key: str,
    metric_title: str,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if not instances:
        return

    role_label = "Rollout" if role == "Rollout" else "Reward Model"
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.set_title(f"{metric_title} — {role_label}", fontsize=14)

    for inst in instances:
        x = [p.elapsed_sec / 60.0 for p in inst.points]
        y = [getattr(p, metric_key) for p in inst.points]
        ax.plot(x, y, linewidth=LINE_WIDTH, label=inst.label)

    ax.set_xlabel("Elapsed Time (min)")
    ax.set_ylabel(metric_title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=LEGEND_NCOL, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    log_dir = LOG_DIR.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")

    out_dir = OUT_DIR.resolve()
    series_by_role = load_series(log_dir)
    if not any(series_by_role.values()):
        raise SystemExit(f"No valid StatCollector snapshots found under {log_dir}")

    for metric_key, metric_title in METRICS:
        for role in ("Rollout", "RewardModel"):
            instances = series_by_role[role]
            if not instances:
                continue
            prefix = ROLE_OUTPUT_PREFIX[role]
            out_path = out_dir / f"{prefix}_{metric_key}.png"
            plot_role_metric_figure(instances, role, metric_key, metric_title, out_path)
            print(f"Saved {out_path}")

    rollout_n = len(series_by_role["Rollout"])
    rm_n = len(series_by_role["RewardModel"])
    print(
        f"Done: {rollout_n} Rollout + {rm_n} RewardModel instances, "
        f"{len(METRICS) * 2} figures -> {out_dir}"
    )


if __name__ == "__main__":
    main()
