#!/usr/bin/env python3
"""Plot the 32-GPU Qwen2.5-7B + Qwen3-30B-A3B case study.

The script produces three figures:

1. Rollout, GenRM, and pressure-weighted harmonic throughput.
2. Elastic awake GPUs for Rollout, GenRM, and Trainer.
3. Engine-running and router-waiting sequences by role.

All x axes use minutes since training step 1 starts.  Long intervals without a
scaling decision or an awake-GPU state change are compressed with broken-axis
markers so that the transition windows remain readable.  Successful elastic
scaling actions are marked when execution completes, rather than when the
policy decision is accepted.  The selected disaggregated run did not
periodically log router backlog, so that series is shown as unavailable unless
``--disaggregated-router-backlog-csv`` is supplied.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/psrl-matplotlib-cache")

DEFAULT_ELASTIC_DIR = (
    REPO_ROOT / "logs/verl_deployment_modes/"
    "mixed_mode5_bs_128_share_24_elastic_Qwen2.5-7B_"
    "Qwen3-30B-A3B-Thinking-2507"
)
DEFAULT_DISAGGREGATED_DIR = (
    REPO_ROOT / "logs/verl_deployment_modes/"
    "mixed_mode1_bs_128_roll_12_rm_3_disaggregated_qwen7b_rm32b_"
    "Qwen2.5-7B_Qwen3-30B-A3B-Thinking-2507"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "logs/verl_deployment_modes/32gpu_7b_30b_case_study"

ROLLOUT = "Rollout"
GENRM = "RewardModel"
ROLES = (ROLLOUT, GENRM)
ROLE_LABEL = {ROLLOUT: "Rollout", GENRM: "GenRM"}
ROLE_COLOR = {ROLLOUT: "#2878B5", GENRM: "#D55E00"}
RUN_COLOR = {"Elastic": "#B2182B", "Disaggregated": "#2166AC"}
RUN_STYLE = {"Elastic": "-", "Disaggregated": "--"}
TP_SIZE = {ROLLOUT: 1, GENRM: 4}
TRAINER_GPUS = 8
ACTIVE_SNAPSHOT_MAX_AGE_S = 3.0
DEFAULT_IDLE_GAP_MINUTES = 2.5
DEFAULT_IDLE_CONTEXT_MINUTES = 0.35
DEFAULT_IDLE_COMPRESSION_RATIO = 0.08

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
RE_LINE_TIMESTAMP = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
RE_TRAINING_BATCH = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*"
    r"\[Begin Event\] WAIT - Wait for training batch (?P<batch>\d+)\s*$"
)
RE_FIRST_BUFFER_WAIT = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*"
    r"TRAINING Buffer 0 is not ready"
)
RE_SNAPSHOT_TIMESTAMP = re.compile(r"'timestamp':\s*'(?P<timestamp>[^']+)'")
RE_THROUGHPUT = re.compile(r"'generation_throughput':\s*(?:np\.float64\()?([\d.eE+-]+)")
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
RE_AWAKE_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Awake instances summary: by_role=(?P<by_role>\{[^}]*\})"
)
RE_SCALE_ACTION = re.compile(
    r"'action_type':\s*'(?P<action>scale_up|scale_down)'.*?"
    r"'role_name':\s*'(?P<role>Rollout|RewardModel)'.*?"
    r"'num_instances':\s*(?P<count>\d+)"
)
RE_DECISION_ID = re.compile(r"\bdecision_id=(?P<decision_id>\d+)")
RE_SCALING_COMPLETION = re.compile(
    r"\[ELASTIC_OVERHEAD\] operation=scaling_action\b.*?"
    r"\bdecision_id=(?P<decision_id>\d+).*?"
    r"\baction_type=(?P<action>scale_up|scale_down).*?"
    r"\bsuccess=(?P<success>True|False).*?"
    r"\bactual_actions=(?P<actual_actions>\d+)"
)


@dataclass(frozen=True)
class RolePoint:
    timestamp: datetime
    throughput: float
    running: float


@dataclass(frozen=True)
class Sample:
    timestamp: datetime
    step: int
    rollout_throughput: float
    genrm_throughput: float
    harmonic_throughput: float
    rollout_running: float
    genrm_running: float
    rollout_waiting: float
    genrm_waiting: float


@dataclass(frozen=True)
class RunData:
    label: str
    origin: datetime
    samples: tuple[Sample, ...]
    waiting_recorded: bool


@dataclass(frozen=True)
class ScalingEvent:
    timestamp: datetime
    role: str
    action: str
    count: int


@dataclass(frozen=True)
class AwakePoint:
    timestamp: datetime
    rollout_gpus: int
    genrm_gpus: int
    trainer_gpus: int


@dataclass(frozen=True)
class BrokenTimeAxis:
    """Piecewise-linear time scale that compresses idle intervals."""

    xlim: tuple[float, float]
    gaps: tuple[tuple[float, float], ...]
    compression_ratio: float

    def _knots(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        source = [self.xlim[0]]
        display = [self.xlim[0]]
        cursor = self.xlim[0]
        display_cursor = self.xlim[0]
        for gap_start, gap_end in self.gaps:
            if gap_start > cursor:
                display_cursor += gap_start - cursor
                source.append(gap_start)
                display.append(display_cursor)
            display_cursor += (gap_end - gap_start) * self.compression_ratio
            source.append(gap_end)
            display.append(display_cursor)
            cursor = gap_end
        if cursor < self.xlim[1]:
            display_cursor += self.xlim[1] - cursor
            source.append(self.xlim[1])
            display.append(display_cursor)
        return tuple(source), tuple(display)

    @staticmethod
    def _interpolate(values, source, target):
        import numpy as np

        array = np.asarray(values, dtype=float)
        flat = array.reshape(-1)
        mapped = np.interp(flat, source, target)
        mapped = np.where(flat < source[0], target[0] + flat - source[0], mapped)
        mapped = np.where(flat > source[-1], target[-1] + flat - source[-1], mapped)
        return mapped.reshape(array.shape)

    def forward(self, values):
        source, display = self._knots()
        return self._interpolate(values, source, display)

    def inverse(self, values):
        source, display = self._knots()
        return self._interpolate(values, display, source)


def _parse_timestamp(raw: str) -> datetime:
    return datetime.strptime(raw, TIMESTAMP_FORMAT)


def _to_second(timestamp: datetime) -> datetime:
    return timestamp.replace(microsecond=0)


def _safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def parse_first_step_start(run_dir: Path) -> datetime:
    """Return the batch-0 wait start, i.e. the beginning of training step 1."""
    trainer_path = run_dir / "MainRayTrainer.log"
    with trainer_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            match = RE_TRAINING_BATCH.match(line)
            if match is not None and int(match.group("batch")) == 0:
                return _parse_timestamp(match.group("ts"))

    agentloop_path = run_dir / "AgentLoopManager.log"
    with agentloop_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if match := RE_FIRST_BUFFER_WAIT.match(line):
                return _parse_timestamp(match.group("ts"))
    raise RuntimeError(f"No training step 1 start found under {run_dir}")


def parse_disaggregated_boundaries(path: Path) -> list[datetime]:
    by_batch: dict[int, datetime] = {}
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if match := RE_TRAINING_BATCH.match(line):
                by_batch[int(match.group("batch"))] = _parse_timestamp(
                    match.group("ts")
                )
    if 0 not in by_batch:
        raise RuntimeError(f"No training batch 0 boundary found in {path}")
    expected = list(range(max(by_batch) + 1))
    if sorted(by_batch) != expected:
        raise RuntimeError(f"Non-contiguous training batches in {path}")
    return [by_batch[index] for index in expected]


def parse_elastic_step_ends(path: Path) -> list[datetime]:
    ends: list[datetime] = []
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if "Released trainer-pool elastic instances from training" not in line:
                continue
            if match := RE_LINE_TIMESTAMP.match(line):
                ends.append(_parse_timestamp(match.group("ts")))
    if not ends:
        raise RuntimeError(f"No elastic trainer-release events found in {path}")
    return ends


def _weighted_harmonic(
    rollout_throughput: float,
    genrm_throughput: float,
    rollout_pressure: float,
    genrm_pressure: float,
) -> float:
    """Use the active side when the other side has zero measured throughput."""
    rollout_throughput = max(0.0, rollout_throughput)
    genrm_throughput = max(0.0, genrm_throughput)
    if rollout_throughput <= 0.0 or genrm_throughput <= 0.0:
        return rollout_throughput + genrm_throughput
    rollout_weight = rollout_pressure if rollout_pressure > 0.0 else 1.0
    genrm_weight = genrm_pressure if genrm_pressure > 0.0 else 1.0
    denominator = rollout_weight / rollout_throughput + genrm_weight / genrm_throughput
    return (rollout_weight + genrm_weight) / denominator


def _elastic_step(timestamp: datetime, step_ends: Sequence[datetime]) -> int:
    return bisect_right(step_ends, timestamp) + 1


def _disaggregated_step(timestamp: datetime, boundaries: Sequence[datetime]) -> int:
    return bisect_right(boundaries, timestamp)


def parse_elastic_samples(
    monitor_path: Path,
    origin: datetime,
    step_ends: Sequence[datetime],
) -> RunData:
    statuses: dict[datetime, dict[str, dict[int, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    backlogs: dict[datetime, dict[str, float]] = {}
    end = step_ends[-1]

    with monitor_path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if match := RE_STATUS_LINE.match(line):
                timestamp = _to_second(_parse_timestamp(match.group("ts")))
                if not (origin <= timestamp <= end):
                    continue
                try:
                    status = ast.literal_eval(match.group("status"))
                except (SyntaxError, ValueError):
                    continue
                if not isinstance(status, dict) or status.get("role") not in ROLES:
                    continue
                try:
                    instance_id = int(status["instance"])
                except (KeyError, TypeError, ValueError):
                    continue
                if status.get("status") == "AWAKEN":
                    statuses[timestamp][str(status["role"])][instance_id] = (
                        _safe_float(status.get("throughput")),
                        _safe_float(status.get("running")),
                    )
                continue

            if match := RE_BACKLOG_LINE.match(line):
                timestamp = _to_second(_parse_timestamp(match.group("ts")))
                if not (origin <= timestamp <= end):
                    continue
                try:
                    by_role = ast.literal_eval(match.group("by_role"))
                except (SyntaxError, ValueError):
                    continue
                if isinstance(by_role, dict):
                    backlogs[timestamp] = {
                        role: _safe_float(by_role.get(role)) for role in ROLES
                    }

    if not statuses:
        raise RuntimeError(f"No elastic status samples found in {monitor_path}")
    if not backlogs:
        raise RuntimeError(f"No elastic router backlog found in {monitor_path}")

    backlog_times = sorted(backlogs)
    backlog_index = 0
    latest_backlog = {role: 0.0 for role in ROLES}
    samples: list[Sample] = []
    for timestamp in sorted(statuses):
        while (
            backlog_index < len(backlog_times)
            and backlog_times[backlog_index] <= timestamp
        ):
            latest_backlog = backlogs[backlog_times[backlog_index]]
            backlog_index += 1
        totals = {
            role: (
                sum(value[0] for value in statuses[timestamp][role].values()),
                sum(value[1] for value in statuses[timestamp][role].values()),
            )
            for role in ROLES
        }
        harmonic = _weighted_harmonic(
            totals[ROLLOUT][0],
            totals[GENRM][0],
            totals[ROLLOUT][1] + latest_backlog[ROLLOUT],
            totals[GENRM][1] + latest_backlog[GENRM],
        )
        samples.append(
            Sample(
                timestamp=timestamp,
                step=_elastic_step(timestamp, step_ends),
                rollout_throughput=totals[ROLLOUT][0],
                genrm_throughput=totals[GENRM][0],
                harmonic_throughput=harmonic,
                rollout_running=totals[ROLLOUT][1],
                genrm_running=totals[GENRM][1],
                rollout_waiting=latest_backlog[ROLLOUT],
                genrm_waiting=latest_backlog[GENRM],
            )
        )
    return RunData("Elastic", origin, tuple(samples), waiting_recorded=True)


def _classify_stat_collector(path: Path) -> tuple[str, int] | None:
    if match := RE_ROLLOUT_LOG.match(path.name):
        return ROLLOUT, int(match.group("instance"))
    if match := RE_RM_LOG.match(path.name):
        return GENRM, int(match.group("instance"))
    return None


def _parse_stat_collector(
    path: Path, origin: datetime, end: datetime
) -> dict[datetime, RolePoint]:
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
            if not (origin <= timestamp < end) or not math.isfinite(throughput):
                continue
            points[timestamp] = RolePoint(timestamp, throughput, running)
    return points


def parse_backlog_csv(
    path: Path | None, origin: datetime
) -> dict[datetime, dict[str, float]]:
    if path is None:
        return {}
    points: dict[datetime, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"wall_time_s", "rollout_waiting", "genrm_waiting"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise RuntimeError(f"Missing columns {sorted(missing)} in {path}")
        for row in reader:
            timestamp = _to_second(
                origin + timedelta(seconds=float(row["wall_time_s"]))
            )
            points[timestamp] = {
                ROLLOUT: max(0.0, float(row["rollout_waiting"])),
                GENRM: max(0.0, float(row["genrm_waiting"])),
            }
    return points


def parse_disaggregated_samples(
    run_dir: Path,
    origin: datetime,
    boundaries: Sequence[datetime],
    backlog_csv: Path | None,
) -> RunData:
    if len(boundaries) < 2:
        raise RuntimeError("Need a next-step boundary to close the disaggregated run")
    end = boundaries[-1]
    instances: dict[str, dict[int, dict[datetime, RolePoint]]] = {
        role: {} for role in ROLES
    }
    all_times: set[datetime] = set()
    for path in sorted(run_dir.glob("StatCollector*.log")):
        metadata = _classify_stat_collector(path)
        if metadata is None:
            continue
        role, instance_id = metadata
        points = _parse_stat_collector(path, origin, end)
        if points:
            instances[role][instance_id] = points
            all_times.update(points)
    missing = [role for role in ROLES if not instances[role]]
    if missing:
        raise RuntimeError(f"Missing StatCollector data for {missing} under {run_dir}")

    backlogs = parse_backlog_csv(backlog_csv, origin)
    backlog_times = sorted(backlogs)
    backlog_index = 0
    latest_backlog = {role: 0.0 for role in ROLES}
    latest: dict[str, dict[int, RolePoint]] = {role: {} for role in ROLES}
    samples: list[Sample] = []
    for timestamp in sorted(all_times):
        while (
            backlog_index < len(backlog_times)
            and backlog_times[backlog_index] <= timestamp
        ):
            latest_backlog = backlogs[backlog_times[backlog_index]]
            backlog_index += 1
        totals: dict[str, tuple[float, float]] = {}
        for role in ROLES:
            for instance_id, points in instances[role].items():
                if point := points.get(timestamp):
                    latest[role][instance_id] = point
            fresh = [
                point
                for point in latest[role].values()
                if 0.0
                <= (timestamp - point.timestamp).total_seconds()
                <= ACTIVE_SNAPSHOT_MAX_AGE_S
            ]
            totals[role] = (
                sum(point.throughput for point in fresh),
                sum(point.running for point in fresh),
            )
        harmonic = _weighted_harmonic(
            totals[ROLLOUT][0],
            totals[GENRM][0],
            totals[ROLLOUT][1] + latest_backlog[ROLLOUT],
            totals[GENRM][1] + latest_backlog[GENRM],
        )
        samples.append(
            Sample(
                timestamp=timestamp,
                step=_disaggregated_step(timestamp, boundaries),
                rollout_throughput=totals[ROLLOUT][0],
                genrm_throughput=totals[GENRM][0],
                harmonic_throughput=harmonic,
                rollout_running=totals[ROLLOUT][1],
                genrm_running=totals[GENRM][1],
                rollout_waiting=(latest_backlog[ROLLOUT] if backlogs else math.nan),
                genrm_waiting=latest_backlog[GENRM] if backlogs else math.nan,
            )
        )
    return RunData(
        "Disaggregated", origin, tuple(samples), waiting_recorded=bool(backlogs)
    )


def parse_scaling_events(
    path: Path, origin: datetime, end: datetime
) -> list[ScalingEvent]:
    accepted_actions: dict[int, list[tuple[str, str, int]]] = defaultdict(list)
    completions: list[tuple[datetime, int, str, bool, int]] = []
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if "Policy decision accepted" in line:
                decision_match = RE_DECISION_ID.search(line)
                if decision_match is None:
                    continue
                decision_id = int(decision_match.group("decision_id"))
                for match in RE_SCALE_ACTION.finditer(line):
                    accepted_actions[decision_id].append(
                        (
                            match.group("role"),
                            match.group("action"),
                            int(match.group("count")),
                        )
                    )
                continue

            completion_match = RE_SCALING_COMPLETION.search(line)
            if completion_match is None:
                continue
            timestamp_match = RE_LINE_TIMESTAMP.match(line)
            if timestamp_match is None:
                continue
            timestamp = _parse_timestamp(timestamp_match.group("ts"))
            completions.append(
                (
                    timestamp,
                    int(completion_match.group("decision_id")),
                    completion_match.group("action"),
                    completion_match.group("success") == "True",
                    int(completion_match.group("actual_actions")),
                )
            )

    action_indices: dict[tuple[int, str], int] = defaultdict(int)
    events: list[ScalingEvent] = []
    for timestamp, decision_id, action, succeeded, actual_actions in completions:
        matching = [
            item for item in accepted_actions.get(decision_id, ()) if item[1] == action
        ]
        key = (decision_id, action)
        action_index = action_indices[key]
        action_indices[key] += 1
        if action_index >= len(matching):
            continue
        role, _, count = matching[action_index]
        if succeeded and actual_actions > 0 and origin <= timestamp <= end:
            events.append(ScalingEvent(timestamp, role, action, count))
    return events


def parse_awake_gpus(path: Path, origin: datetime, end: datetime) -> list[AwakePoint]:
    awake_events: list[tuple[datetime, dict[str, int]]] = []
    trainer_events: list[tuple[datetime, int]] = []
    last_before_origin: tuple[datetime, dict[str, int]] | None = None
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            timestamp_match = RE_LINE_TIMESTAMP.match(line)
            if timestamp_match is None:
                continue
            timestamp = _parse_timestamp(timestamp_match.group("ts"))
            if match := RE_AWAKE_LINE.match(line):
                try:
                    by_role = ast.literal_eval(match.group("by_role"))
                except (SyntaxError, ValueError):
                    continue
                if not isinstance(by_role, dict):
                    continue
                counts = {
                    role: int(_safe_float(by_role.get(role))) * TP_SIZE[role]
                    for role in ROLES
                }
                if timestamp < origin:
                    last_before_origin = (timestamp, counts)
                elif timestamp <= end:
                    awake_events.append((timestamp, counts))
                continue
            if not (origin <= timestamp <= end):
                continue
            if "Reserved trainer-pool elastic instances for training" in line:
                trainer_events.append((timestamp, TRAINER_GPUS))
            elif "Released trainer-pool elastic instances from training" in line:
                trainer_events.append((timestamp, 0))

    if not awake_events:
        raise RuntimeError(f"No awake-instance summaries found in {path}")
    current_awake = (
        last_before_origin[1]
        if last_before_origin is not None
        else {role: 0 for role in ROLES}
    )
    current_trainer = 0
    trainer_events.sort(key=lambda item: item[0])
    trainer_index = 0
    points = [
        AwakePoint(
            origin,
            current_awake[ROLLOUT],
            current_awake[GENRM],
            current_trainer,
        )
    ]
    for timestamp, counts in awake_events:
        while (
            trainer_index < len(trainer_events)
            and trainer_events[trainer_index][0] <= timestamp
        ):
            current_trainer = trainer_events[trainer_index][1]
            trainer_index += 1
        current_awake = counts
        points.append(
            AwakePoint(
                timestamp,
                current_awake[ROLLOUT],
                current_awake[GENRM],
                current_trainer,
            )
        )
    return points


def _select_samples(data: RunData, step_range: tuple[int, int] | None) -> RunData:
    if step_range is None:
        return data
    start, end = step_range
    samples = tuple(sample for sample in data.samples if start <= sample.step <= end)
    if not samples:
        raise RuntimeError(
            f"{data.label} has no samples in requested steps {start}-{end}"
        )
    return RunData(data.label, data.origin, samples, data.waiting_recorded)


def _time_minutes(origin: datetime, timestamps: Iterable[datetime]) -> list[float]:
    return [(timestamp - origin).total_seconds() / 60.0 for timestamp in timestamps]


def _selected_time_bounds(runs: Sequence[RunData]) -> tuple[float, float]:
    values = [
        (sample.timestamp - run.origin).total_seconds() / 60.0
        for run in runs
        for sample in run.samples
    ]
    left = min(values)
    if any(sample.step == 1 for run in runs for sample in run.samples):
        left = min(0.0, left)
    return left, max(values)


def _awake_change_minutes(
    points: Sequence[AwakePoint],
    origin: datetime,
    xlim: tuple[float, float],
) -> list[float]:
    changes: list[float] = []
    previous_state: tuple[int, int, int] | None = None
    for point in points:
        state = (point.rollout_gpus, point.genrm_gpus, point.trainer_gpus)
        if state != previous_state:
            minute = (point.timestamp - origin).total_seconds() / 60.0
            if xlim[0] <= minute <= xlim[1]:
                changes.append(minute)
            previous_state = state
    return changes


def _build_broken_time_axis(
    xlim: tuple[float, float],
    events: Sequence[ScalingEvent],
    awake_points: Sequence[AwakePoint],
    origin: datetime,
    idle_gap_minutes: float,
    idle_context_minutes: float,
    compression_ratio: float,
    activity_xlim: tuple[float, float] | None = None,
) -> BrokenTimeAxis:
    activity_xlim = activity_xlim or xlim
    activity = _awake_change_minutes(awake_points, origin, activity_xlim)
    activity.extend(
        minute
        for event in events
        if activity_xlim[0]
        <= (minute := (event.timestamp - origin).total_seconds() / 60.0)
        <= activity_xlim[1]
    )
    activity = sorted(set(activity))
    if not activity or compression_ratio >= 1.0:
        return BrokenTimeAxis(xlim, (), 1.0)

    anchors = [xlim[0], *activity, xlim[1]]
    gaps: list[tuple[float, float]] = []
    for left, right in zip(anchors, anchors[1:]):
        gap = (
            max(xlim[0], left + idle_context_minutes),
            min(xlim[1], right - idle_context_minutes),
        )
        if gap[1] > gap[0] and gap[1] - gap[0] >= idle_gap_minutes:
            gaps.append(gap)
    return BrokenTimeAxis(xlim, tuple(gaps), compression_ratio)


def _compressed_ticks(scale: BrokenTimeAxis, count: int = 11) -> list[float]:
    import numpy as np

    left, right = scale.xlim
    display_left, display_right = scale.forward([left, right])
    values = scale.inverse(np.linspace(display_left, display_right, count))
    span = right - left
    quantum = 5.0 if span > 240.0 else 1.0 if span > 80.0 else 0.5
    candidates = [
        left,
        *(round(float(value) / quantum) * quantum for value in values),
        right,
    ]
    ticks: list[float] = []
    for value in sorted(candidates):
        if not left <= value <= right:
            continue
        if any(start < value < end for start, end in scale.gaps):
            continue
        if ticks and abs(value - ticks[-1]) < quantum * 0.25:
            continue
        ticks.append(value)
    return ticks


def _configure_time_axis(axes, scale: BrokenTimeAxis) -> None:
    axes = list(axes)
    for axis in axes:
        if scale.gaps:
            axis.set_xscale(
                "function",
                functions=(scale.forward, scale.inverse),
            )
        axis.set_xlim(*scale.xlim)

    bottom_axis = axes[-1]
    if scale.gaps:
        ticks = _compressed_ticks(scale)
        span = scale.xlim[1] - scale.xlim[0]
        labels = [f"{tick:.0f}" if span > 80.0 else f"{tick:.1f}" for tick in ticks]
        bottom_axis.set_xticks(ticks, labels)
        for axis in axes:
            for gap_start, gap_end in scale.gaps:
                axis.text(
                    (gap_start + gap_end) / 2.0,
                    0.0,
                    "//",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#555555",
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1},
                    clip_on=False,
                    zorder=8,
                )


def _time_axis_label(scale: BrokenTimeAxis) -> str:
    label = "Minutes since step 1 start"
    return f"{label} (idle intervals compressed)" if scale.gaps else label


def _style_axis(axis, ylabel: str) -> None:
    from matplotlib.ticker import StrMethodFormatter

    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.75)
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _scaling_handles(events: Sequence[ScalingEvent]):
    from matplotlib.lines import Line2D

    handles = []
    seen: set[tuple[str, str]] = set()
    for event in events:
        key = (event.role, event.action)
        if key in seen:
            continue
        seen.add(key)
        handles.append(
            Line2D(
                [],
                [],
                color=ROLE_COLOR[event.role],
                marker="^" if event.action == "scale_up" else "v",
                linestyle="None",
                markersize=6,
                label=(
                    f"{ROLE_LABEL[event.role]} "
                    f"{'scale-up' if event.action == 'scale_up' else 'scale-down'} "
                    "completed"
                ),
            )
        )
    return handles


def _mark_scaling_events(
    axes,
    events: Sequence[ScalingEvent],
    origin: datetime,
    xlim: tuple[float, float],
) -> list[ScalingEvent]:
    visible = [
        event
        for event in events
        if xlim[0] <= (event.timestamp - origin).total_seconds() / 60.0 <= xlim[1]
    ]
    for event in visible:
        x = (event.timestamp - origin).total_seconds() / 60.0
        color = ROLE_COLOR[event.role]
        for axis in axes:
            axis.axvline(
                x,
                color=color,
                linewidth=0.9,
                linestyle="--",
                alpha=0.30,
                zorder=1,
            )
        axes[0].scatter(
            [x],
            [0.985],
            transform=axes[0].get_xaxis_transform(),
            marker="^" if event.action == "scale_up" else "v",
            color=color,
            s=32,
            alpha=0.98,
            edgecolors="white",
            linewidths=0.45,
            clip_on=False,
            zorder=5,
        )
    return visible


def _figure_suffix(step_range: tuple[int, int] | None) -> str:
    return "" if step_range is None else f"_steps_{step_range[0]}-{step_range[1]}"


def _save_figure(fig, output_stem: Path, formats: Sequence[str], dpi: int) -> None:
    for output_format in formats:
        path = output_stem.with_suffix(f".{output_format}")
        fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
        print(path)


def plot_throughput(
    runs: Sequence[RunData],
    events: Sequence[ScalingEvent],
    time_axis: BrokenTimeAxis,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    step_range: tuple[int, int] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("rollout_throughput", "Rollout throughput"),
        ("genrm_throughput", "GenRM throughput"),
        ("harmonic_throughput", "Pressure-weighted harmonic throughput"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(12.6, 9.0), sharex=True)
    for axis, (field, ylabel) in zip(axes, metrics):
        for run in runs:
            axis.plot(
                _time_minutes(run.origin, (sample.timestamp for sample in run.samples)),
                [getattr(sample, field) for sample in run.samples],
                color=RUN_COLOR[run.label],
                linewidth=0.9,
                alpha=0.9,
                label=run.label,
            )
        _style_axis(axis, f"{ylabel}\n(tokens/s)")

    elastic_event_xlim = _selected_time_bounds((runs[0],))
    visible_events = _mark_scaling_events(
        axes, events, runs[0].origin, elastic_event_xlim
    )
    _configure_time_axis(axes, time_axis)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend(_scaling_handles(visible_events))
    labels.extend(handle.get_label() for handle in handles[len(labels) :])
    axes[0].legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=8.5,
    )
    axes[-1].set_xlabel(_time_axis_label(time_axis))
    step_text = (
        "" if step_range is None else f" | steps {step_range[0]}-{step_range[1]}"
    )
    fig.suptitle(
        "32-GPU Case Study: Qwen2.5-7B + Qwen3-30B-A3B\n"
        f"Throughput vs time{step_text}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)


def _select_awake(
    points: Sequence[AwakePoint],
    origin: datetime,
    xlim: tuple[float, float],
) -> list[AwakePoint]:
    selected: list[AwakePoint] = []
    previous: AwakePoint | None = None
    for point in points:
        x = (point.timestamp - origin).total_seconds() / 60.0
        if x < xlim[0]:
            previous = point
            continue
        if x > xlim[1]:
            break
        if previous is not None and not selected:
            selected.append(
                AwakePoint(
                    timestamp=origin + timedelta(minutes=xlim[0]),
                    rollout_gpus=previous.rollout_gpus,
                    genrm_gpus=previous.genrm_gpus,
                    trainer_gpus=previous.trainer_gpus,
                )
            )
        selected.append(point)
    return selected


def plot_awake_gpus(
    points: Sequence[AwakePoint],
    origin: datetime,
    events: Sequence[ScalingEvent],
    time_axis: BrokenTimeAxis,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    step_range: tuple[int, int] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = _select_awake(points, origin, time_axis.xlim)
    if not selected:
        raise RuntimeError("No awake-GPU samples in selected time window")
    x = _time_minutes(origin, (point.timestamp for point in selected))
    fig, axis = plt.subplots(figsize=(12.6, 5.0))
    for label, field, color in (
        ("Rollout", "rollout_gpus", ROLE_COLOR[ROLLOUT]),
        ("GenRM", "genrm_gpus", ROLE_COLOR[GENRM]),
        ("Trainer", "trainer_gpus", "#2A9D6F"),
    ):
        axis.step(
            x,
            [getattr(point, field) for point in selected],
            where="post",
            color=color,
            linewidth=1.35,
            label=label,
        )
    _style_axis(axis, "Awake GPUs")
    axis.set_ylim(0, 34)
    visible_events = _mark_scaling_events([axis], events, origin, time_axis.xlim)
    _configure_time_axis([axis], time_axis)
    axis.set_xlabel(_time_axis_label(time_axis))
    handles, labels = axis.get_legend_handles_labels()
    scale_handles = _scaling_handles(visible_events)
    handles.extend(scale_handles)
    labels.extend(handle.get_label() for handle in scale_handles)
    axis.legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=8.5,
    )
    step_text = (
        "" if step_range is None else f" | steps {step_range[0]}-{step_range[1]}"
    )
    axis.set_title(f"Elastic awake GPUs by role vs time{step_text}")
    fig.tight_layout()
    _save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)


def plot_sequences(
    runs: Sequence[RunData],
    events: Sequence[ScalingEvent],
    time_axis: BrokenTimeAxis,
    output_stem: Path,
    formats: Sequence[str],
    dpi: int,
    step_range: tuple[int, int] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12.6, 7.3), sharex=True)
    for run in runs:
        x = _time_minutes(run.origin, (sample.timestamp for sample in run.samples))
        for role, running_field, waiting_field in (
            (ROLLOUT, "rollout_running", "rollout_waiting"),
            (GENRM, "genrm_running", "genrm_waiting"),
        ):
            label = f"{run.label} - {ROLE_LABEL[role]}"
            axes[0].plot(
                x,
                [getattr(sample, running_field) for sample in run.samples],
                color=ROLE_COLOR[role],
                linestyle=RUN_STYLE[run.label],
                linewidth=0.95,
                alpha=0.9,
                label=label,
            )
            if run.waiting_recorded:
                axes[1].plot(
                    x,
                    [getattr(sample, waiting_field) for sample in run.samples],
                    color=ROLE_COLOR[role],
                    linestyle=RUN_STYLE[run.label],
                    linewidth=0.95,
                    alpha=0.9,
                    label=label,
                )

    _style_axis(axes[0], "Running sequences\n(in engines)")
    _style_axis(axes[1], "Waiting sequences\n(in routers)")
    missing_waiting = [run.label for run in runs if not run.waiting_recorded]
    if missing_waiting:
        axes[1].text(
            0.012,
            0.95,
            f"{', '.join(missing_waiting)} router backlog: not recorded",
            transform=axes[1].transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#666666",
        )
    elastic_event_xlim = _selected_time_bounds((runs[0],))
    visible_events = _mark_scaling_events(
        axes, events, runs[0].origin, elastic_event_xlim
    )
    _configure_time_axis(axes, time_axis)
    handles, labels = axes[0].get_legend_handles_labels()
    scale_handles = _scaling_handles(visible_events)
    handles.extend(scale_handles)
    labels.extend(handle.get_label() for handle in scale_handles)
    axes[0].legend(
        handles,
        labels,
        loc="upper right",
        frameon=False,
        ncol=2,
        fontsize=8.2,
    )
    axes[-1].set_xlabel(_time_axis_label(time_axis))
    step_text = (
        "" if step_range is None else f" | steps {step_range[0]}-{step_range[1]}"
    )
    fig.suptitle(
        "32-GPU Case Study: sequence pressure vs time" + step_text,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save_figure(fig, output_stem, formats, dpi)
    plt.close(fig)


def export_run_csv(data: RunData, path: Path) -> None:
    columns = (
        "wall_time_s",
        "step",
        "rollout_throughput",
        "genrm_throughput",
        "harmonic_throughput",
        "rollout_running",
        "genrm_running",
        "rollout_router_waiting",
        "genrm_router_waiting",
    )
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        for sample in data.samples:
            writer.writerow(
                (
                    f"{(sample.timestamp - data.origin).total_seconds():.3f}",
                    sample.step,
                    f"{sample.rollout_throughput:.6f}",
                    f"{sample.genrm_throughput:.6f}",
                    f"{sample.harmonic_throughput:.6f}",
                    f"{sample.rollout_running:.6f}",
                    f"{sample.genrm_running:.6f}",
                    (
                        f"{sample.rollout_waiting:.6f}"
                        if math.isfinite(sample.rollout_waiting)
                        else ""
                    ),
                    (
                        f"{sample.genrm_waiting:.6f}"
                        if math.isfinite(sample.genrm_waiting)
                        else ""
                    ),
                )
            )


def export_awake_csv(
    points: Sequence[AwakePoint], origin: datetime, path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("wall_time_s", "rollout_gpus", "genrm_gpus", "trainer_gpus"))
        for point in points:
            writer.writerow(
                (
                    f"{(point.timestamp - origin).total_seconds():.3f}",
                    point.rollout_gpus,
                    point.genrm_gpus,
                    point.trainer_gpus,
                )
            )


def export_scaling_csv(
    events: Sequence[ScalingEvent], origin: datetime, path: Path
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("wall_time_s", "role", "action", "num_instances"))
        for event in events:
            writer.writerow(
                (
                    f"{(event.timestamp - origin).total_seconds():.3f}",
                    ROLE_LABEL[event.role],
                    event.action,
                    event.count,
                )
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elastic-run-dir", type=Path, default=DEFAULT_ELASTIC_DIR)
    parser.add_argument(
        "--disaggregated-run-dir", type=Path, default=DEFAULT_DISAGGREGATED_DIR
    )
    parser.add_argument(
        "--disaggregated-router-backlog-csv",
        type=Path,
        help=(
            "Optional CSV with wall_time_s, rollout_waiting, genrm_waiting. "
            "The selected disaggregated logs do not contain this metric."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-start", type=int)
    parser.add_argument("--step-end", type=int)
    parser.add_argument("--formats", default="png,pdf")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--idle-gap-minutes",
        type=float,
        default=DEFAULT_IDLE_GAP_MINUTES,
        help=(
            "Compress an interval when this many minutes remain between the "
            "context windows around consecutive scaling-related changes "
            f"(default: {DEFAULT_IDLE_GAP_MINUTES})."
        ),
    )
    parser.add_argument(
        "--idle-context-minutes",
        type=float,
        default=DEFAULT_IDLE_CONTEXT_MINUTES,
        help=(
            "Uncompressed context retained on each side of a scaling-related "
            f"change (default: {DEFAULT_IDLE_CONTEXT_MINUTES})."
        ),
    )
    parser.add_argument(
        "--idle-compression-ratio",
        type=float,
        default=DEFAULT_IDLE_COMPRESSION_RATIO,
        help=(
            "Displayed width / elapsed width for idle intervals "
            f"(default: {DEFAULT_IDLE_COMPRESSION_RATIO})."
        ),
    )
    parser.add_argument(
        "--no-broken-time-axis",
        action="store_true",
        help="Use the original linear elapsed-time axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (args.step_start is None) != (args.step_end is None):
        raise SystemExit("--step-start and --step-end must be specified together")
    if args.step_start is not None and (
        args.step_start <= 0 or args.step_start > args.step_end
    ):
        raise SystemExit("Step range must satisfy 1 <= --step-start <= --step-end")
    step_range = None if args.step_start is None else (args.step_start, args.step_end)
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item)
    unsupported = set(formats).difference({"png", "pdf", "svg"})
    if not formats or unsupported:
        raise SystemExit(f"Unsupported output formats: {sorted(unsupported)}")
    if args.idle_gap_minutes < 0.0 or args.idle_context_minutes < 0.0:
        raise SystemExit("Idle-gap and context minutes must be non-negative")
    if not 0.0 < args.idle_compression_ratio <= 1.0:
        raise SystemExit("--idle-compression-ratio must satisfy 0 < ratio <= 1")

    elastic_dir = args.elastic_run_dir.resolve()
    disaggregated_dir = args.disaggregated_run_dir.resolve()
    for run_dir in (elastic_dir, disaggregated_dir):
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory not found: {run_dir}")
    backlog_csv = (
        args.disaggregated_router_backlog_csv.resolve()
        if args.disaggregated_router_backlog_csv
        else None
    )
    if backlog_csv is not None and not backlog_csv.is_file():
        raise SystemExit(f"Router backlog CSV not found: {backlog_csv}")

    elastic_origin = parse_first_step_start(elastic_dir)
    disaggregated_origin = parse_first_step_start(disaggregated_dir)
    executor_path = elastic_dir / "ElasticExecutor.log"
    elastic_step_ends = parse_elastic_step_ends(executor_path)
    disaggregated_boundaries = parse_disaggregated_boundaries(
        disaggregated_dir / "MainRayTrainer.log"
    )
    elastic = parse_elastic_samples(
        elastic_dir / "ElasticMonitor.log", elastic_origin, elastic_step_ends
    )
    disaggregated = parse_disaggregated_samples(
        disaggregated_dir,
        disaggregated_origin,
        disaggregated_boundaries,
        backlog_csv,
    )
    events = parse_scaling_events(executor_path, elastic_origin, elastic_step_ends[-1])
    awake_points = parse_awake_gpus(
        executor_path, elastic_origin, elastic_step_ends[-1]
    )

    selected_runs = (
        _select_samples(elastic, step_range),
        _select_samples(disaggregated, step_range),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _figure_suffix(step_range)

    comparison_xlim = _selected_time_bounds(selected_runs)
    elastic_xlim = _selected_time_bounds((selected_runs[0],))
    compression_ratio = 1.0 if args.no_broken_time_axis else args.idle_compression_ratio
    comparison_time_axis = _build_broken_time_axis(
        comparison_xlim,
        events,
        awake_points,
        elastic_origin,
        args.idle_gap_minutes,
        args.idle_context_minutes,
        compression_ratio,
        activity_xlim=elastic_xlim,
    )
    elastic_time_axis = _build_broken_time_axis(
        elastic_xlim,
        events,
        awake_points,
        elastic_origin,
        args.idle_gap_minutes,
        args.idle_context_minutes,
        compression_ratio,
    )

    export_run_csv(elastic, output_dir / "elastic_timeseries.csv")
    export_run_csv(disaggregated, output_dir / "disaggregated_timeseries.csv")
    export_awake_csv(
        awake_points, elastic_origin, output_dir / "elastic_awake_gpus.csv"
    )
    export_scaling_csv(
        events, elastic_origin, output_dir / "elastic_scaling_events.csv"
    )

    plot_throughput(
        selected_runs,
        events,
        comparison_time_axis,
        output_dir / f"01_throughput_vs_time{suffix}",
        formats,
        args.dpi,
        step_range,
    )
    plot_awake_gpus(
        awake_points,
        elastic_origin,
        events,
        elastic_time_axis,
        output_dir / f"02_awake_gpus_vs_time{suffix}",
        formats,
        args.dpi,
        step_range,
    )
    plot_sequences(
        selected_runs,
        events,
        comparison_time_axis,
        output_dir / f"03_sequences_vs_time{suffix}",
        formats,
        args.dpi,
        step_range,
    )

    print(
        f"Elastic: step_1_start={elastic_origin.isoformat()}, "
        f"steps=1-{max(sample.step for sample in elastic.samples)}, "
        f"samples={len(elastic.samples)}, scaling_events={len(events)}, "
        f"compressed_idle_gaps={len(comparison_time_axis.gaps)}"
    )
    print(
        f"Disaggregated: step_1_start={disaggregated_origin.isoformat()}, "
        f"steps=1-{max(sample.step for sample in disaggregated.samples)}, "
        f"samples={len(disaggregated.samples)}, "
        f"router_backlog={'recorded' if disaggregated.waiting_recorded else 'not recorded'}"
    )


if __name__ == "__main__":
    main()
