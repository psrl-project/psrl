#!/usr/bin/env python3
"""Export role totals and weighted harmonic throughput for the 70B runs.

When both roles are active, the calculation matches the logged ITL-harmonic
configuration:

    H = (w_rollout + w_rm) / (w_rollout / rollout_tp + w_rm / rm_tp)

where each raw request-count weight is ``running + router backlog`` and a zero
weight falls back to 1.  ElasticMonitor contains both terms for the elastic RL
run.  The disaggregated StatCollector logs contain only in-engine request
counts, so its unavailable router backlog is explicitly treated as zero.

When exactly one role has zero measured throughput, the exported metric falls
back to the non-zero role's throughput (the one-sided sum).  It is zero only
when both roles have zero throughput.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DISAGGREGATED_RUN = (
    REPO_ROOT
    / "logs/verl_deployment_modes/"
    "mixed_mode1_roll_4_rm_4_disaggregated_Qwen2.5-72B_Qwen3.5-122B-A10B"
)
DEFAULT_ELASTIC_RUN = (
    REPO_ROOT
    / "logs/verl_deployment_modes/"
    "mixed_mode5_bs_128_share_64_elastic_rl_Qwen2.5-72B_Qwen3.5-122B-A10B"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs/verl_deployment_modes/70b_harmonic_throughput"

ROLES = ("Rollout", "RewardModel")
ROLLOUT = "Rollout"
REWARD_MODEL = "RewardModel"
ACTIVE_SNAPSHOT_MAX_AGE_SEC = 3.0
CSV_COLUMNS = (
    "wall_time_s",
    "step",
    "rollout_total_throughput",
    "rm_total_throughput",
    "harmonic_throughput",
)

RE_TRAINING_BATCH = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*"
    r"\[Begin Event\] WAIT - Wait for training batch (?P<batch>\d+)\s*$"
)
RE_SNAPSHOT_TIMESTAMP = re.compile(r"'timestamp':\s*'(?P<timestamp>[^']+)'")
RE_THROUGHPUT = re.compile(
    r"'generation_throughput':\s*(?:np\.float64\()?([\d.eE+-]+)"
)
RE_NUM_RUNNING = re.compile(r"'num_running_reqs':\s*(\d+)")
RE_ROLLOUT_LOG = re.compile(r"^StatCollector_I(?P<instance>\d+)\.log$")
RE_RM_LOG = re.compile(r"^StatCollector_RM_.+_I(?P<instance>\d+)\.log$")
RE_STATUS_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Instance current Status:\s*(?P<status>\{.*\})\s*$"
)
RE_BACKLOG_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Router backlog summary: by_role=(?P<by_role>\{[^}]+\})"
)


@dataclass(frozen=True)
class RolePoint:
    timestamp: datetime
    throughput: float
    running: float


@dataclass(frozen=True)
class CombinedPoint:
    timestamp: datetime
    rollout_throughput: float
    rm_throughput: float
    rollout_pressure: float
    rm_pressure: float


def _parse_log_timestamp(raw: str) -> datetime:
    return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S,%f")


def _to_second(timestamp: datetime) -> datetime:
    return timestamp.replace(microsecond=0)


def parse_training_boundaries(log_path: Path) -> list[datetime]:
    """Return ordered starts of training batches 0, 1, ... from trainer logs."""
    by_batch: dict[int, datetime] = {}
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = RE_TRAINING_BATCH.match(line)
            if match is None:
                continue
            by_batch[int(match.group("batch"))] = _parse_log_timestamp(match.group("ts"))

    if not by_batch or 0 not in by_batch:
        raise RuntimeError(f"No training batch 0 boundary found in {log_path}")
    expected = list(range(max(by_batch) + 1))
    if sorted(by_batch) != expected:
        raise RuntimeError(f"Non-contiguous training batches in {log_path}: {sorted(by_batch)}")
    boundaries = [by_batch[batch] for batch in expected]
    if boundaries != sorted(boundaries):
        raise RuntimeError(f"Non-monotonic training batch timestamps in {log_path}")
    return boundaries


def _classify_stat_collector(path: Path) -> tuple[str, int] | None:
    if match := RE_ROLLOUT_LOG.match(path.name):
        return ROLLOUT, int(match.group("instance"))
    if match := RE_RM_LOG.match(path.name):
        return REWARD_MODEL, int(match.group("instance"))
    return None


def _parse_stat_collector(path: Path) -> dict[datetime, RolePoint]:
    """Keep the latest high-frequency snapshot for each wall-clock second."""
    points: dict[datetime, RolePoint] = {}
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if "Snapshot (model version" not in line:
                continue
            timestamp_match = RE_SNAPSHOT_TIMESTAMP.search(line)
            throughput_match = RE_THROUGHPUT.search(line)
            running_match = RE_NUM_RUNNING.search(line)
            if not (timestamp_match and throughput_match and running_match):
                continue
            try:
                timestamp = _to_second(
                    datetime.fromisoformat(timestamp_match.group("timestamp"))
                )
                throughput = float(throughput_match.group(1))
                running = float(running_match.group(1))
            except ValueError:
                continue
            if not math.isfinite(throughput):
                continue
            points[timestamp] = RolePoint(timestamp, throughput, running)
    return points


def parse_disaggregated_points(run_dir: Path) -> list[CombinedPoint]:
    """Aggregate fresh StatCollector snapshots; router backlog is unavailable."""
    instances: dict[str, dict[int, dict[datetime, RolePoint]]] = {
        role: {} for role in ROLES
    }
    all_times: set[datetime] = set()
    for path in sorted(run_dir.glob("StatCollector*.log")):
        metadata = _classify_stat_collector(path)
        if metadata is None:
            continue
        role, instance_id = metadata
        points = _parse_stat_collector(path)
        if not points:
            continue
        instances[role][instance_id] = points
        all_times.update(points)

    missing_roles = [role for role in ROLES if not instances[role]]
    if missing_roles:
        raise RuntimeError(f"Missing StatCollector data for {missing_roles} under {run_dir}")

    latest: dict[str, dict[int, RolePoint]] = {role: {} for role in ROLES}
    combined: list[CombinedPoint] = []
    for timestamp in sorted(all_times):
        totals: dict[str, tuple[float, float]] = {}
        for role in ROLES:
            for instance_id, points in instances[role].items():
                if point := points.get(timestamp):
                    latest[role][instance_id] = point
            active = [
                point
                for point in latest[role].values()
                if 0.0
                <= (timestamp - point.timestamp).total_seconds()
                <= ACTIVE_SNAPSHOT_MAX_AGE_SEC
            ]
            totals[role] = (
                sum(point.throughput for point in active),
                sum(point.running for point in active),
            )
        combined.append(
            CombinedPoint(
                timestamp=timestamp,
                rollout_throughput=totals[ROLLOUT][0],
                rm_throughput=totals[REWARD_MODEL][0],
                rollout_pressure=totals[ROLLOUT][1],
                rm_pressure=totals[REWARD_MODEL][1],
            )
        )
    return combined


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def parse_elastic_points(log_path: Path) -> list[CombinedPoint]:
    """Aggregate ElasticMonitor status and align its router backlog snapshots."""
    statuses: dict[datetime, dict[str, dict[int, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    backlogs: dict[datetime, dict[str, float]] = {}

    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if match := RE_STATUS_LINE.match(line):
                try:
                    status = ast.literal_eval(match.group("status"))
                except (SyntaxError, ValueError):
                    continue
                if not isinstance(status, dict) or status.get("role") not in ROLES:
                    continue
                try:
                    instance_id = int(status.get("instance"))
                except (TypeError, ValueError):
                    continue
                timestamp = _to_second(_parse_log_timestamp(match.group("ts")))
                statuses[timestamp][str(status["role"])][instance_id] = (
                    _safe_float(status.get("throughput", 0.0)),
                    _safe_float(status.get("running", 0.0)),
                )
                continue

            match = RE_BACKLOG_LINE.match(line)
            if match is None:
                continue
            try:
                by_role = ast.literal_eval(match.group("by_role"))
            except (SyntaxError, ValueError):
                continue
            if not isinstance(by_role, dict):
                continue
            timestamp = _to_second(_parse_log_timestamp(match.group("ts")))
            backlogs[timestamp] = {
                role: _safe_float(by_role.get(role, 0.0)) for role in ROLES
            }

    if not statuses:
        raise RuntimeError(f"No ElasticMonitor status samples found in {log_path}")

    backlog_times = sorted(backlogs)
    backlog_index = 0
    latest_backlog = {role: 0.0 for role in ROLES}
    combined: list[CombinedPoint] = []
    for timestamp in sorted(statuses):
        while backlog_index < len(backlog_times) and backlog_times[backlog_index] <= timestamp:
            latest_backlog = backlogs[backlog_times[backlog_index]]
            backlog_index += 1

        totals: dict[str, tuple[float, float]] = {}
        for role in ROLES:
            role_statuses = statuses[timestamp].get(role, {}).values()
            throughput = sum(value[0] for value in role_statuses)
            running = sum(value[1] for value in role_statuses)
            totals[role] = (throughput, running + latest_backlog[role])
        combined.append(
            CombinedPoint(
                timestamp=timestamp,
                rollout_throughput=totals[ROLLOUT][0],
                rm_throughput=totals[REWARD_MODEL][0],
                rollout_pressure=totals[ROLLOUT][1],
                rm_pressure=totals[REWARD_MODEL][1],
            )
        )
    return combined


def weighted_harmonic_throughput(point: CombinedPoint) -> float:
    """Return harmonic throughput, falling back to the active side when starved."""
    rollout_throughput = max(0.0, point.rollout_throughput)
    rm_throughput = max(0.0, point.rm_throughput)
    if rollout_throughput <= 0.0 or rm_throughput <= 0.0:
        return rollout_throughput + rm_throughput
    rollout_weight = point.rollout_pressure if point.rollout_pressure > 0.0 else 1.0
    rm_weight = point.rm_pressure if point.rm_pressure > 0.0 else 1.0
    denominator = (
        rollout_weight / rollout_throughput
        + rm_weight / rm_throughput
    )
    return (rollout_weight + rm_weight) / denominator


def export_csv(
    points: list[CombinedPoint], boundaries: list[datetime], output_path: Path
) -> int:
    training_start = boundaries[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(CSV_COLUMNS)
        for point in points:
            if point.timestamp < training_start:
                continue
            writer.writerow(
                (
                    f"{(point.timestamp - training_start).total_seconds():.3f}",
                    bisect_right(boundaries, point.timestamp),
                    f"{point.rollout_throughput:.6f}",
                    f"{point.rm_throughput:.6f}",
                    f"{weighted_harmonic_throughput(point):.6f}",
                )
            )
            row_count += 1
    return row_count


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--disaggregated-run-dir", type=Path, default=DEFAULT_DISAGGREGATED_RUN
    )
    parser.add_argument("--elastic-run-dir", type=Path, default=DEFAULT_ELASTIC_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    disaggregated_run = args.disaggregated_run_dir.resolve()
    elastic_run = args.elastic_run_dir.resolve()
    output_dir = args.output_dir.resolve()
    for run_dir in (disaggregated_run, elastic_run):
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory not found: {run_dir}")

    disaggregated_boundaries = parse_training_boundaries(
        disaggregated_run / "MainRayTrainer.log"
    )
    elastic_boundaries = parse_training_boundaries(elastic_run / "MainRayTrainer.log")

    print(f"Parsing disaggregated StatCollector logs under {disaggregated_run}")
    disaggregated_points = parse_disaggregated_points(disaggregated_run)
    disaggregated_output = output_dir / "disaggregated_harmonic_throughput.csv"
    disaggregated_rows = export_csv(
        disaggregated_points, disaggregated_boundaries, disaggregated_output
    )

    print(f"Parsing elastic monitor {elastic_run / 'ElasticMonitor.log'}")
    elastic_points = parse_elastic_points(elastic_run / "ElasticMonitor.log")
    elastic_output = output_dir / "elastic_rl_harmonic_throughput.csv"
    elastic_rows = export_csv(elastic_points, elastic_boundaries, elastic_output)

    print(
        f"disaggregated: {disaggregated_rows} rows, "
        f"training_start={disaggregated_boundaries[0].isoformat()}, "
        f"output={disaggregated_output}"
    )
    print(
        f"elastic_rl: {elastic_rows} rows, "
        f"training_start={elastic_boundaries[0].isoformat()}, "
        f"output={elastic_output}"
    )


if __name__ == "__main__":
    main()
