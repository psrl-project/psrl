#!/usr/bin/env python3
"""Plot comparable per-instance heatmaps from StatCollector logs.

The input is a run directory containing ``StatCollector*.log`` files. Snapshot
timestamps embedded in the records are resampled onto one shared time grid, so
Rollout and RewardModel instances can be compared even though their collectors
write independently. Numeric cells are only shown while an instance has a
recent snapshot; stale periods are gray and exposed explicitly in the active
panel.

Examples:
    python scripts/plot_stat_collector_instance_load.py
    python scripts/plot_stat_collector_instance_load.py /path/to/run
    python scripts/plot_stat_collector_instance_load.py --start-min 60 --end-min 120
    python scripts/plot_stat_collector_instance_load.py /path/to/run --steps 10
    python scripts/plot_stat_collector_instance_load.py --roles Rollout --bin-sec 0.5
"""

from __future__ import annotations

import argparse
import bisect
import copy
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = (
    REPO_ROOT
    / "logs/verl_deployment_modes/"
    "mixed_mode1_roll_4_rm_4_disaggregated_Qwen2.5-72B_Qwen3.5-122B-A10B"
)

ROLE_ORDER = ("Rollout", "RewardModel")
NUMERIC_METRICS = ("running", "waiting", "kv_cache", "throughput", "tokens")
DEFAULT_METRICS = ("running", "waiting", "kv_cache", "throughput", "active")
ALL_METRICS = (*NUMERIC_METRICS, "active")
TRAINER_ACTIVE_EVENTS = (
    "Recompute log_prob on training side",
    "Update actor",
)
METRIC_SPECS = {
    "running": ("Running queue", "YlGnBu"),
    "waiting": ("Waiting queue", "YlOrRd"),
    "kv_cache": ("KV cache usage", "viridis"),
    "throughput": ("Generation throughput", "magma"),
    "tokens": ("Resident tokens", "cividis"),
}

RE_TIMESTAMP = re.compile(r"'timestamp':\s*'(?P<timestamp>[^']+)'")
RE_THROUGHPUT = re.compile(
    r"'generation_throughput':\s*(?:np\.float64\()?([\d.eE+-]+)"
)
RE_NUM_RUNNING = re.compile(r"'num_running_reqs':\s*(\d+)")
RE_NUM_WAITING = re.compile(r"'num_waiting_reqs':\s*(\d+)")
RE_KV_CACHE = re.compile(r"'kv_cache_usage':\s*([\d.eE+-]+)")
RE_PROMPT_SECTION = re.compile(
    r"'req_id_to_prompt_token_num':\s*\{(.*?)\},\s*'req_id_to_response_token_num'"
)
RE_RESPONSE_SECTION = re.compile(
    r"'req_id_to_response_token_num':\s*\{(.*?)\},\s*'req_id_in_waiting'"
)
RE_DICT_INT_VALUES = re.compile(r":\s*(\d+)")
RE_ROLLOUT_LOG = re.compile(r"^StatCollector_I(?P<instance>\d+)\.log$")
RE_RM_LOG = re.compile(
    r"^StatCollector_RM_(?P<model>.+)_I(?P<instance>\d+)\.log$"
)
RE_LOG_TIMESTAMP = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)
RE_TRAINING_BATCH_BEGIN = re.compile(
    r"\[Begin Event\] WAIT - Wait for training batch (?P<step>\d+)"
)


@dataclass(frozen=True)
class Snapshot:
    timestamp: datetime
    running: float
    waiting: float
    kv_cache: float
    throughput: float
    tokens: float


@dataclass(frozen=True)
class InstanceSeries:
    role: str
    model: str
    instance_id: int
    path: Path
    points: tuple[Snapshot, ...]


TrainerInterval = tuple[datetime, datetime | None]


def classify_log(path: Path) -> tuple[str, str, int] | None:
    if match := RE_ROLLOUT_LOG.match(path.name):
        return "Rollout", "rollout", int(match.group("instance"))
    if match := RE_RM_LOG.match(path.name):
        return (
            "RewardModel",
            match.group("model"),
            int(match.group("instance")),
        )
    return None


def _sum_dict_values(section: str) -> int:
    return sum(int(value) for value in RE_DICT_INT_VALUES.findall(section))


def _resident_tokens(line: str) -> int:
    total = 0
    if match := RE_PROMPT_SECTION.search(line):
        total += _sum_dict_values(match.group(1))
    if match := RE_RESPONSE_SECTION.search(line):
        total += _sum_dict_values(match.group(1))
    return total


def parse_snapshot_line(line: str, include_tokens: bool = True) -> Snapshot | None:
    if "Snapshot (model version" not in line:
        return None
    timestamp = RE_TIMESTAMP.search(line)
    running = RE_NUM_RUNNING.search(line)
    waiting = RE_NUM_WAITING.search(line)
    kv_cache = RE_KV_CACHE.search(line)
    throughput = RE_THROUGHPUT.search(line)
    if not (timestamp and running and waiting and kv_cache and throughput):
        return None

    try:
        point = Snapshot(
            timestamp=datetime.fromisoformat(timestamp.group("timestamp")),
            running=float(running.group(1)),
            waiting=float(waiting.group(1)),
            kv_cache=float(kv_cache.group(1)),
            throughput=float(throughput.group(1)),
            tokens=float(_resident_tokens(line)) if include_tokens else 0.0,
        )
    except ValueError:
        return None
    values = (
        point.running,
        point.waiting,
        point.kv_cache,
        point.throughput,
        point.tokens,
    )
    return point if all(math.isfinite(value) for value in values) else None


def parse_instance_log(
    path: Path, include_tokens: bool = True
) -> tuple[Snapshot, ...]:
    points: list[Snapshot] = []
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            point = parse_snapshot_line(line, include_tokens=include_tokens)
            if point is not None:
                points.append(point)
    points.sort(key=lambda point: point.timestamp)
    return tuple(points)


def load_instances(
    log_dir: Path, requested_roles: set[str], include_tokens: bool = True
) -> dict[str, list[InstanceSeries]]:
    instances: dict[str, list[InstanceSeries]] = {
        role: [] for role in ROLE_ORDER if role in requested_roles
    }
    skipped = 0
    for path in sorted(log_dir.glob("StatCollector*.log")):
        metadata = classify_log(path)
        if metadata is None:
            continue
        role, model, instance_id = metadata
        if role not in requested_roles:
            continue
        points = parse_instance_log(path, include_tokens=include_tokens)
        if not points:
            skipped += 1
            continue
        instances.setdefault(role, []).append(
            InstanceSeries(role, model, instance_id, path, points)
        )

    for role in instances:
        instances[role].sort(key=lambda instance: (instance.instance_id, instance.model))
    if skipped:
        print(f"Skipped {skipped} StatCollector logs without valid snapshots.")
    return {role: role_instances for role, role_instances in instances.items() if role_instances}


def parse_trainer_active_intervals(log_path: Path) -> list[TrainerInterval]:
    """Parse trainer GPU-compute intervals from MainRayTrainer event records."""
    if not log_path.is_file():
        return []

    starts: dict[str, datetime] = {}
    intervals: list[TrainerInterval] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            event = next(
                (name for name in TRAINER_ACTIVE_EVENTS if name in line), None
            )
            if event is None:
                continue
            timestamp_match = RE_LOG_TIMESTAMP.match(line)
            if timestamp_match is None:
                continue
            timestamp = datetime.strptime(
                timestamp_match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f"
            )
            if "[Begin Event]" in line:
                starts[event] = timestamp
            elif "[End Event]" in line and event in starts:
                start = starts.pop(event)
                if timestamp >= start:
                    intervals.append((start, timestamp))

    intervals.extend((start, None) for start in starts.values())
    intervals.sort(key=lambda interval: interval[0])
    return intervals


def first_steps_end_timestamp(log_path: Path, step_count: int) -> datetime:
    """Return the wall-clock boundary after the first ``step_count`` steps."""
    if step_count <= 0:
        raise SystemExit("--first-steps must be positive.")
    if not log_path.is_file():
        raise SystemExit(f"Trainer log not found: {log_path}")

    batch_begins: dict[int, datetime] = {}
    update_ends: list[datetime] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            timestamp_match = RE_LOG_TIMESTAMP.match(line)
            if timestamp_match is None:
                continue
            timestamp = datetime.strptime(
                timestamp_match.group("timestamp"), "%Y-%m-%d %H:%M:%S,%f"
            )
            if batch_match := RE_TRAINING_BATCH_BEGIN.search(line):
                batch_begins[int(batch_match.group("step"))] = timestamp
            if "[End Event] TRAIN - Update actor" in line:
                update_ends.append(timestamp)

    update_ends.sort()
    if len(update_ends) < step_count:
        raise SystemExit(
            f"Requested first {step_count} steps, but only {len(update_ends)} "
            f"completed actor updates were found in {log_path}."
        )

    update_end = update_ends[step_count - 1]
    next_batch_begin = batch_begins.get(step_count)
    if next_batch_begin is not None and next_batch_begin >= update_end:
        return next_batch_begin
    return update_end


def trainer_active_grid(
    grid: list[datetime], intervals: list[TrainerInterval]
) -> object:
    import numpy as np

    active = np.zeros(len(grid), dtype=float)
    for start, end in intervals:
        first = bisect.bisect_left(grid, start)
        last = len(grid) if end is None else bisect.bisect_right(grid, end)
        if first < len(grid) and last > 0:
            active[max(first, 0) : min(last, len(grid))] = 1.0
    return active


def make_time_grid(
    instances_by_role: dict[str, list[InstanceSeries]],
    bin_sec: float,
    start_min: float | None,
    end_min: float | None,
    end_timestamp: datetime | None = None,
) -> tuple[datetime, list[datetime]]:
    all_instances = [
        instance
        for role_instances in instances_by_role.values()
        for instance in role_instances
    ]
    origin = min(instance.points[0].timestamp for instance in all_instances)
    last_timestamp = max(instance.points[-1].timestamp for instance in all_instances)
    start = origin + timedelta(minutes=start_min or 0.0)
    end = (
        origin + timedelta(minutes=end_min)
        if end_min is not None
        else last_timestamp
    )
    if end_timestamp is not None:
        end = min(end, end_timestamp)
    if start > last_timestamp or end < origin or start > end:
        raise SystemExit("No StatCollector samples intersect the selected time window.")
    start = max(start, origin)
    end = min(end, last_timestamp)
    count = int(math.floor((end - start).total_seconds() / bin_sec)) + 1
    return origin, [start + timedelta(seconds=index * bin_sec) for index in range(count)]


def build_matrices(
    instances_by_role: dict[str, list[InstanceSeries]],
    grid: list[datetime],
    max_age_sec: float,
) -> dict[str, dict[str, object]]:
    import numpy as np

    matrices: dict[str, dict[str, object]] = {}
    for role, instances in instances_by_role.items():
        role_matrices = {
            metric: np.full((len(instances), len(grid)), np.nan, dtype=float)
            for metric in NUMERIC_METRICS
        }
        role_matrices["active"] = np.zeros(
            (len(instances), len(grid)), dtype=float
        )

        for row, instance in enumerate(instances):
            point_index = -1
            points = instance.points
            for column, timestamp in enumerate(grid):
                while (
                    point_index + 1 < len(points)
                    and points[point_index + 1].timestamp <= timestamp
                ):
                    point_index += 1
                if point_index < 0:
                    continue
                point = points[point_index]
                age_sec = (timestamp - point.timestamp).total_seconds()
                if age_sec < 0.0 or age_sec > max_age_sec:
                    continue
                role_matrices["active"][row, column] = 1.0
                for metric in NUMERIC_METRICS:
                    role_matrices[metric][row, column] = getattr(point, metric)
        matrices[role] = role_matrices
    return matrices


def _time_edges(x_minutes: object, bin_sec: float) -> object:
    import numpy as np

    x_values = np.asarray(x_minutes, dtype=float)
    half_bin_min = bin_sec / 120.0
    return np.concatenate(
        (x_values - half_bin_min, [x_values[-1] + half_bin_min])
    )


def _tick_rows(instance_count: int, max_ticks: int = 16) -> list[int]:
    if instance_count <= max_ticks:
        return list(range(instance_count))
    step = math.ceil(instance_count / max_ticks)
    return list(range(0, instance_count, step))


def _numeric_limit(
    matrices: dict[str, dict[str, object]],
    roles: list[str],
    metric: str,
    percentile: float,
) -> float:
    import numpy as np

    finite_parts = []
    for role in roles:
        values = matrices[role][metric]
        finite = values[np.isfinite(values)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        return 1.0
    limit = float(np.percentile(np.concatenate(finite_parts), percentile))
    return limit if limit > 0.0 else 1.0


def plot_heatmaps(
    origin: datetime,
    grid: list[datetime],
    instances_by_role: dict[str, list[InstanceSeries]],
    roles: list[str],
    metrics: list[str],
    bin_sec: float,
    max_age_sec: float,
    trainer_intervals: list[TrainerInterval],
    percentile: float,
    output_path: Path,
    dpi: int,
) -> dict[str, float]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    matrices = build_matrices(instances_by_role, grid, max_age_sec)
    x_minutes = np.array(
        [(timestamp - origin).total_seconds() / 60.0 for timestamp in grid]
    )
    x_edges = _time_edges(x_minutes, bin_sec)

    plot_rows = [*metrics, "trainer_active"]
    fig, axes = plt.subplots(
        len(plot_rows),
        len(roles),
        figsize=(max(10.0, 8.5 * len(roles)), max(4.0, 3.0 * len(plot_rows))),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    numeric_limits = {
        metric: _numeric_limit(matrices, roles, metric, percentile)
        for metric in metrics
        if metric in NUMERIC_METRICS
    }
    active_cmap = ListedColormap(["#d9d9d9", "#59a14f"])
    trainer_cmap = ListedColormap(["#d9d9d9", "#e15759"])
    active_norm = BoundaryNorm([-0.5, 0.5, 1.5], active_cmap.N)
    trainer_active = trainer_active_grid(grid, trainer_intervals)

    for row, metric in enumerate(plot_rows):
        row_images = []
        for column, role in enumerate(roles):
            ax = axes[row, column]
            instances = instances_by_role[role]
            if metric == "trainer_active":
                values = trainer_active[np.newaxis, :]
                image = ax.pcolormesh(
                    x_edges,
                    np.arange(2),
                    values,
                    cmap=trainer_cmap,
                    norm=active_norm,
                    shading="flat",
                )
                ax.set_ylim(1, 0)
                ax.set_yticks([0.5], labels=["Trainer"], fontsize=7)
                ax.set_ylabel("State")
                ax.set_title(f"Trainer compute aligned with {role}", fontsize=10)
            else:
                values = matrices[role][metric]
                y_edges = np.arange(len(instances) + 1)
                if metric == "active":
                    image = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        values,
                        cmap=active_cmap,
                        norm=active_norm,
                        shading="flat",
                    )
                else:
                    _, cmap_name = METRIC_SPECS[metric]
                    cmap = copy.copy(plt.get_cmap(cmap_name))
                    cmap.set_bad("#d9d9d9")
                    image = ax.pcolormesh(
                        x_edges,
                        y_edges,
                        values,
                        cmap=cmap,
                        vmin=0.0,
                        vmax=numeric_limits[metric],
                        shading="flat",
                    )

                ax.set_ylim(len(instances), 0)
                tick_rows = _tick_rows(len(instances))
                ax.set_yticks([tick + 0.5 for tick in tick_rows])
                ax.set_yticklabels(
                    [f"I{instances[tick].instance_id}" for tick in tick_rows],
                    fontsize=7,
                )
                models = ", ".join(
                    model
                    for model in dict.fromkeys(
                        instance.model for instance in instances
                    )
                    if model != "rollout"
                )
                model_suffix = f"; {models}" if models else ""
                ax.set_title(
                    f"{role} ({len(instances)} instances{model_suffix})",
                    fontsize=10,
                )
                ax.set_ylabel("Instance")

            if row == len(plot_rows) - 1:
                ax.set_xlabel("Elapsed time from first StatCollector snapshot (min)")
            row_images.append(image)

        if metric == "trainer_active":
            label = "Trainer active"
        elif metric == "active":
            label = "Active/reporting"
        else:
            label = METRIC_SPECS[metric][0]
        axes[row, 0].text(
            -0.17,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
        colorbar = fig.colorbar(row_images[-1], ax=list(axes[row, :]), pad=0.01)
        if metric == "trainer_active":
            colorbar.set_ticks([0, 1], labels=["INACTIVE", "ACTIVE"])
        elif metric == "active":
            colorbar.set_ticks([0, 1], labels=["STALE", "ACTIVE"])
        else:
            suffix = f" (p{percentile:g} cap)" if percentile < 100.0 else ""
            colorbar.set_label(f"{METRIC_SPECS[metric][0]}{suffix}")

    fig.suptitle(
        "StatCollector per-instance load\n"
        f"origin={origin.isoformat(sep=' ')}; bin={bin_sec:g}s; "
        f"active age <= {max_age_sec:g}s; trainer active = log_prob or actor update",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return numeric_limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing StatCollector logs (default: {DEFAULT_LOG_DIR})",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    parser.add_argument("--start-min", type=float, help="Inclusive elapsed-time start")
    parser.add_argument("--end-min", type=float, help="Inclusive elapsed-time end")
    parser.add_argument(
        "--first-steps",
        "--steps",
        dest="first_steps",
        type=int,
        help="Plot only the first N completed training steps",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=ROLE_ORDER,
        default=list(ROLE_ORDER),
        help="Role columns to draw",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=ALL_METRICS,
        default=list(DEFAULT_METRICS),
        help="Metric rows to draw; 'tokens' is available but not drawn by default",
    )
    parser.add_argument(
        "--bin-sec",
        type=float,
        default=1.0,
        help="Shared time-grid interval in seconds (default: 1)",
    )
    parser.add_argument(
        "--max-age-sec",
        type=float,
        default=3.0,
        help="Maximum snapshot age considered active (default: 3)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=100.0,
        help="Shared numeric color-scale cap percentile (default: 100)",
    )
    parser.add_argument("--dpi", type=int, default=250, help="Output DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.resolve()
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory not found: {log_dir}")
    if args.start_min is not None and args.start_min < 0.0:
        raise SystemExit("--start-min must be non-negative.")
    if args.end_min is not None and args.end_min < 0.0:
        raise SystemExit("--end-min must be non-negative.")
    if (
        args.start_min is not None
        and args.end_min is not None
        and args.start_min > args.end_min
    ):
        raise SystemExit("--start-min must be <= --end-min.")
    if args.first_steps is not None and args.first_steps <= 0:
        raise SystemExit("--first-steps must be positive.")
    if args.first_steps is not None and args.end_min is not None:
        raise SystemExit("--first-steps and --end-min cannot be used together.")
    if args.bin_sec <= 0.0:
        raise SystemExit("--bin-sec must be positive.")
    if args.max_age_sec < 0.0:
        raise SystemExit("--max-age-sec must be non-negative.")
    if not 0.0 < args.percentile <= 100.0:
        raise SystemExit("--percentile must be in (0, 100].")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive.")

    roles = list(dict.fromkeys(args.roles))
    metrics = list(dict.fromkeys(args.metrics))
    instances_by_role = load_instances(
        log_dir, set(roles), include_tokens="tokens" in metrics
    )
    roles = [role for role in roles if role in instances_by_role]
    if not roles:
        raise SystemExit(f"No valid StatCollector logs found under {log_dir}")
    trainer_log_path = log_dir / "MainRayTrainer.log"
    steps_end = (
        first_steps_end_timestamp(trainer_log_path, args.first_steps)
        if args.first_steps is not None
        else None
    )
    origin, grid = make_time_grid(
        instances_by_role,
        args.bin_sec,
        args.start_min,
        args.end_min,
        end_timestamp=steps_end,
    )
    trainer_intervals = parse_trainer_active_intervals(trainer_log_path)
    default_output_name = (
        f"StatCollector_instance_load_first_{args.first_steps}_steps.png"
        if args.first_steps is not None
        else "StatCollector_instance_load.png"
    )
    output_path = (
        args.output.resolve()
        if args.output
        else log_dir / default_output_name
    )
    limits = plot_heatmaps(
        origin=origin,
        grid=grid,
        instances_by_role=instances_by_role,
        roles=roles,
        metrics=metrics,
        bin_sec=args.bin_sec,
        max_age_sec=args.max_age_sec,
        trainer_intervals=trainer_intervals,
        percentile=args.percentile,
        output_path=output_path,
        dpi=args.dpi,
    )

    elapsed_start = (grid[0] - origin).total_seconds() / 60.0
    elapsed_end = (grid[-1] - origin).total_seconds() / 60.0
    role_summary = ", ".join(
        f"{role}={len(instances_by_role[role])}" for role in roles
    )
    limit_summary = ", ".join(
        f"{metric}={limit:g}" for metric, limit in limits.items()
    )
    print(
        f"Built {len(grid)} grid points over {elapsed_start:.2f}-{elapsed_end:.2f} min; "
        f"{role_summary}"
    )
    if limit_summary:
        print(f"Shared color-scale maxima: {limit_summary}")
    print(f"Trainer active intervals: {len(trainer_intervals)}")
    if args.first_steps is not None:
        print(f"Step window: first {args.first_steps} completed steps")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
