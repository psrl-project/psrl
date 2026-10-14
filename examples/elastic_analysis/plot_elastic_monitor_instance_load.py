#!/usr/bin/env python3
"""Plot per-instance load heatmaps from an ElasticMonitor.log file.

The monitor emits one ``Instance current Status`` record per instance and
monitor cycle.  This script aligns those records on the monitor timestamp and
plots one heatmap per (metric, role) pair.  Numeric panels only color AWAKEN
instances; ASLEEP/TRAINING periods are gray and are distinguished in the
categorical status panel.

Examples:
    python scripts/plot_elastic_monitor_instance_load.py
    python scripts/plot_elastic_monitor_instance_load.py /path/ElasticMonitor.log
    python scripts/plot_elastic_monitor_instance_load.py --start-min 60 --end-min 120
    python scripts/plot_elastic_monitor_instance_load.py --roles Rollout --pool train_pool
"""

from __future__ import annotations

import argparse
import ast
import copy
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_PATH = (
    REPO_ROOT
    / "logs/verl_deployment_modes/"
    "dapo_mode5_bs_128_share_16_elastic_rl_rollout7_ds_7b_rm8b/"
    "ElasticMonitor.log"
)

ROLE_ORDER = ("Rollout", "RewardModel")
NUMERIC_METRICS = ("running", "waiting", "kv_cache", "throughput")
DEFAULT_METRICS = (*NUMERIC_METRICS, "status")
STATUS_CODES = {"ASLEEP": 0.0, "TRAINING": 1.0, "AWAKEN": 2.0}
# Status lines from one cycle are emitted back-to-back, while monitor cycles in
# this log are roughly five seconds apart. This gap also handles cycles that
# happen to cross a wall-clock second boundary.
MONITOR_CYCLE_GAP_SEC = 2.0
METRIC_SPECS = {
    "running": ("Running queue", "YlGnBu"),
    "waiting": ("Waiting queue", "YlOrRd"),
    "kv_cache": ("KV cache usage", "viridis"),
    "throughput": ("Throughput", "magma"),
}

RE_STATUS_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Instance current Status:\s*(?P<payload>\{.*\})\s*$"
)


@dataclass(frozen=True)
class InstanceKey:
    role: str
    model: str
    instance_id: int
    pool_id: str


@dataclass(frozen=True)
class InstanceSample:
    status: str
    running: float
    waiting: float
    kv_cache: float
    throughput: float


SnapshotMap = dict[datetime, dict[InstanceKey, InstanceSample]]


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_status_line(
    line: str,
) -> tuple[datetime, InstanceKey, InstanceSample] | None:
    match = RE_STATUS_LINE.match(line)
    if not match:
        return None
    try:
        payload = ast.literal_eval(match.group("payload"))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    role = str(payload.get("role", ""))
    model = str(payload.get("model", ""))
    pool_id = str(payload.get("pool_id", "unknown"))
    status = str(payload.get("status", "UNKNOWN"))
    try:
        instance_id = int(payload["instance"])
    except (KeyError, TypeError, ValueError):
        return None
    if not role:
        return None

    timestamp = datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
    key = InstanceKey(role, model, instance_id, pool_id)
    sample = InstanceSample(
        status=status,
        running=_finite_float(payload.get("running")),
        waiting=_finite_float(payload.get("waiting")),
        kv_cache=_finite_float(payload.get("kv_cache")),
        throughput=_finite_float(payload.get("throughput")),
    )
    return timestamp, key, sample


def parse_monitor_log(log_path: Path) -> SnapshotMap:
    snapshots: SnapshotMap = {}
    cycle_timestamp: datetime | None = None
    previous_timestamp: datetime | None = None
    cycle_samples: dict[InstanceKey, InstanceSample] = {}

    def finish_cycle() -> None:
        if cycle_timestamp is not None and cycle_samples:
            snapshots[cycle_timestamp] = dict(cycle_samples)

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            parsed = parse_status_line(line)
            if parsed is None:
                continue
            timestamp, key, sample = parsed
            if previous_timestamp is not None:
                gap_sec = (timestamp - previous_timestamp).total_seconds()
                if gap_sec < 0.0 or gap_sec > MONITOR_CYCLE_GAP_SEC:
                    finish_cycle()
                    cycle_samples.clear()
                    cycle_timestamp = timestamp
            if cycle_timestamp is None:
                cycle_timestamp = timestamp
            cycle_samples[key] = sample
            previous_timestamp = timestamp
    finish_cycle()
    return snapshots


def select_snapshots(
    snapshots: SnapshotMap,
    roles: set[str],
    pools: set[str] | None,
    start_min: float | None,
    end_min: float | None,
) -> tuple[datetime, list[datetime], SnapshotMap]:
    if not snapshots:
        raise SystemExit("No valid 'Instance current Status' records were parsed.")

    origin = min(snapshots)
    selected: SnapshotMap = {}
    for timestamp in sorted(snapshots):
        elapsed_min = (timestamp - origin).total_seconds() / 60.0
        if start_min is not None and elapsed_min < start_min:
            continue
        if end_min is not None and elapsed_min > end_min:
            continue
        filtered = {
            key: sample
            for key, sample in snapshots[timestamp].items()
            if key.role in roles and (pools is None or key.pool_id in pools)
        }
        if filtered:
            selected[timestamp] = filtered

    if not selected:
        raise SystemExit("No monitor records remain after applying the filters.")
    return origin, sorted(selected), selected


def collect_instances(
    times: list[datetime], snapshots: SnapshotMap
) -> dict[str, list[InstanceKey]]:
    found: dict[str, set[InstanceKey]] = {role: set() for role in ROLE_ORDER}
    pool_order: dict[str, dict[str, int]] = {role: {} for role in ROLE_ORDER}
    for timestamp in times:
        for key in snapshots[timestamp]:
            found.setdefault(key.role, set()).add(key)
            role_pools = pool_order.setdefault(key.role, {})
            role_pools.setdefault(key.pool_id, len(role_pools))

    result: dict[str, list[InstanceKey]] = {}
    for role, keys in found.items():
        if not keys:
            continue
        order = pool_order[role]
        result[role] = sorted(
            keys,
            key=lambda key: (
                order[key.pool_id],
                key.instance_id,
                key.model,
            ),
        )
    return result


def build_matrices(
    times: list[datetime],
    snapshots: SnapshotMap,
    instances_by_role: dict[str, list[InstanceKey]],
) -> dict[str, dict[str, object]]:
    import numpy as np

    matrices: dict[str, dict[str, object]] = {}
    for role, instances in instances_by_role.items():
        row_by_key = {key: row for row, key in enumerate(instances)}
        role_matrices = {
            metric: np.full((len(instances), len(times)), np.nan, dtype=float)
            for metric in DEFAULT_METRICS
        }
        for column, timestamp in enumerate(times):
            for key, sample in snapshots[timestamp].items():
                if key.role != role or key not in row_by_key:
                    continue
                row = row_by_key[key]
                role_matrices["status"][row, column] = STATUS_CODES.get(
                    sample.status, np.nan
                )
                if sample.status != "AWAKEN":
                    continue
                for metric in NUMERIC_METRICS:
                    role_matrices[metric][row, column] = getattr(sample, metric)
        matrices[role] = role_matrices
    return matrices


def _time_edges(x_minutes: object) -> object:
    import numpy as np

    x = np.asarray(x_minutes, dtype=float)
    if len(x) == 1:
        return np.array([x[0] - 0.5 / 60.0, x[0] + 0.5 / 60.0])
    midpoints = (x[:-1] + x[1:]) / 2.0
    first = x[0] - (midpoints[0] - x[0])
    last = x[-1] + (x[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def _tick_rows(instances: list[InstanceKey], max_ticks: int = 16) -> list[int]:
    if len(instances) <= max_ticks:
        return list(range(len(instances)))
    step = math.ceil(len(instances) / max_ticks)
    ticks = set(range(0, len(instances), step))
    for row, key in enumerate(instances):
        if row == 0 or key.pool_id != instances[row - 1].pool_id:
            ticks.add(row)
    return sorted(ticks)


def _draw_pool_boundaries(ax: object, instances: list[InstanceKey]) -> None:
    for row in range(1, len(instances)):
        if instances[row].pool_id != instances[row - 1].pool_id:
            ax.axhline(row, color="white", linewidth=1.2, linestyle="--")


def _pool_summary(instances: list[InstanceKey]) -> str:
    counts = Counter(key.pool_id for key in instances)
    return ", ".join(f"{pool}={count}" for pool, count in counts.items())


def _numeric_limit(
    matrices: dict[str, dict[str, object]], roles: list[str], metric: str, percentile: float
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
    times: list[datetime],
    snapshots: SnapshotMap,
    roles: list[str],
    metrics: list[str],
    percentile: float,
    output_path: Path,
    dpi: int,
) -> dict[str, float]:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    instances_by_role = collect_instances(times, snapshots)
    roles = [role for role in roles if role in instances_by_role]
    matrices = build_matrices(times, snapshots, instances_by_role)
    x_minutes = np.array(
        [(timestamp - origin).total_seconds() / 60.0 for timestamp in times]
    )
    x_edges = _time_edges(x_minutes)

    fig_width = max(10.0, 8.5 * len(roles))
    fig_height = max(4.0, 3.0 * len(metrics))
    fig, axes = plt.subplots(
        len(metrics),
        len(roles),
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )

    numeric_limits: dict[str, float] = {}
    for metric in metrics:
        if metric in NUMERIC_METRICS:
            numeric_limits[metric] = _numeric_limit(
                matrices, roles, metric, percentile
            )

    status_cmap = ListedColormap(["#d9d9d9", "#4c78a8", "#59a14f"])
    status_cmap.set_bad("#ffffff")
    status_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], status_cmap.N)

    for row, metric in enumerate(metrics):
        row_images = []
        for column, role in enumerate(roles):
            ax = axes[row, column]
            instances = instances_by_role[role]
            y_edges = np.arange(len(instances) + 1)
            values = matrices[role][metric]

            if metric == "status":
                image = ax.pcolormesh(
                    x_edges,
                    y_edges,
                    values,
                    cmap=status_cmap,
                    norm=status_norm,
                    shading="flat",
                )
            else:
                label, cmap_name = METRIC_SPECS[metric]
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
            row_images.append(image)

            ax.set_ylim(len(instances), 0)
            tick_rows = _tick_rows(instances)
            ax.set_yticks([tick + 0.5 for tick in tick_rows])
            ax.set_yticklabels(
                [f"I{instances[tick].instance_id}" for tick in tick_rows], fontsize=7
            )
            ax.set_ylabel("Instance")
            _draw_pool_boundaries(ax, instances)
            ax.set_title(
                f"{role} ({len(instances)} instances; {_pool_summary(instances)})",
                fontsize=10,
            )
            if row == len(metrics) - 1:
                ax.set_xlabel("Elapsed time from first monitor snapshot (min)")

        label = "Status" if metric == "status" else METRIC_SPECS[metric][0]
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
        if metric == "status":
            colorbar.set_ticks([0, 1, 2], labels=["ASLEEP", "TRAINING", "AWAKEN"])
        else:
            suffix = f" (p{percentile:g} cap)" if percentile < 100.0 else ""
            colorbar.set_label(f"{METRIC_SPECS[metric][0]}{suffix}")

    fig.suptitle(
        "ElasticMonitor per-instance load\n"
        f"origin={origin.isoformat(sep=' ')}; gray numeric cells = not AWAKEN/missing",
        fontsize=13,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return numeric_limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"ElasticMonitor log (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    parser.add_argument("--start-min", type=float, help="Inclusive elapsed-time start")
    parser.add_argument("--end-min", type=float, help="Inclusive elapsed-time end")
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=ROLE_ORDER,
        default=list(ROLE_ORDER),
        help="Role columns to draw",
    )
    parser.add_argument(
        "--pool",
        action="append",
        dest="pools",
        help="Only include this pool; repeat to select multiple pools",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_METRICS,
        default=list(DEFAULT_METRICS),
        help="Metric rows to draw",
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
    log_path = args.log_path.resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")
    if args.start_min is not None and args.start_min < 0:
        raise SystemExit("--start-min must be non-negative.")
    if args.end_min is not None and args.end_min < 0:
        raise SystemExit("--end-min must be non-negative.")
    if (
        args.start_min is not None
        and args.end_min is not None
        and args.start_min > args.end_min
    ):
        raise SystemExit("--start-min must be <= --end-min.")
    if not 0.0 < args.percentile <= 100.0:
        raise SystemExit("--percentile must be in (0, 100].")
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive.")

    snapshots = parse_monitor_log(log_path)
    roles = list(dict.fromkeys(args.roles))
    pools = set(args.pools) if args.pools else None
    origin, times, selected = select_snapshots(
        snapshots, set(roles), pools, args.start_min, args.end_min
    )
    instances_by_role = collect_instances(times, selected)
    roles = [role for role in roles if role in instances_by_role]
    if not roles:
        raise SystemExit("No requested roles remain after applying the filters.")

    output_path = (
        args.output.resolve()
        if args.output
        else log_path.with_name(f"{log_path.stem}_instance_load.png")
    )
    limits = plot_heatmaps(
        origin=origin,
        times=times,
        snapshots=selected,
        roles=roles,
        metrics=list(dict.fromkeys(args.metrics)),
        percentile=args.percentile,
        output_path=output_path,
        dpi=args.dpi,
    )

    elapsed_start = (times[0] - origin).total_seconds() / 60.0
    elapsed_end = (times[-1] - origin).total_seconds() / 60.0
    role_summary = ", ".join(
        f"{role}={len(instances_by_role[role])}" for role in roles
    )
    limit_summary = ", ".join(
        f"{metric}={limit:g}" for metric, limit in limits.items()
    )
    print(
        f"Parsed {len(times)} monitor snapshots over "
        f"{elapsed_start:.2f}-{elapsed_end:.2f} min; {role_summary}"
    )
    if limit_summary:
        print(f"Shared color-scale maxima: {limit_summary}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
