#!/usr/bin/env python3
"""
Plot ElasticMonitor metrics: elastic alone, or non-elastic (min_4) vs elastic (min_0).

Self-contained: parses ElasticMonitor.log directly (no other plot_* scripts).

Metrics (5 subplots):
- total throughput / running / waiting (from Instance current Status lines)
- awake instances / router backlog (from ElasticExecutor summary lines in the same log)

X-axis: elapsed minutes from each log's first aggregated timestamp.

Set PLOT_ELASTIC_ONLY=True to draw only the elastic (min_0) curves.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Parse / plot tuning (edit here) ---
# Inclusive plot range in minutes from the first aggregated timestamp in each log.
PLOT_WINDOW_START_MIN = 0.0
PLOT_WINDOW_END_MIN = 100.0
# If True, only plot elastic (min_0); skip non-elastic overlay.
PLOT_ELASTIC_ONLY = False
# Optional y-axis ranges for the five subplots. None = auto-scale.
PLOT_YLIM_THROUGHPUT: tuple[float, float] | None = None
PLOT_YLIM_RUNNING: tuple[float, float] | None = None
PLOT_YLIM_WAITING: tuple[float, float] | None = None
PLOT_YLIM_AWAKE: tuple[float, float] | None = None
PLOT_YLIM_BACKLOG: tuple[float, float] | None = None
# Merge monitor-cycle timestamps to second resolution before aggregation.
AGGREGATE_TS_TO_SECOND = True
STALINESS = 2

FIGSIZE = (13, 14)
SAVE_DPI = 400
LINE_WIDTH = 1.0

ROLES = ("RewardModel", "Rollout")
TOTAL_METRICS = ("throughput", "running", "waiting")
SUMMARY_METRICS = ("awake", "backlog")

RE_STATUS_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Instance current Status:\s*(?P<status>\{.*\})\s*$"
)
RE_AWAKE_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Awake instances summary: by_role=(?P<br>\{[^}]+\})"
)
RE_BACKLOG_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    r".*Router backlog summary: by_role=(?P<br>\{[^}]+\})"
)

# min_4: light red / light blue; min_0 (elastic): dark red / dark blue; all solid.
COLOR_MIN4_REWARDMODEL = "#f1948a"
COLOR_MIN4_ROLLOUT = "#7fb3d5"
COLOR_MIN0_REWARDMODEL = "#922b21"
COLOR_MIN0_ROLLOUT = "#1a5276"

NON_ELASTIC_LOG_PATH = (
    REPO_ROOT
    / f"logs/verl_deployment_modes/mixed_test_none_request_level_candidate_evaluation_mode5_bs_128_share_32_elastic_rl_Qwen2.5-32B_Qwen3-30B-A3B-Thinking-2507/ElasticMonitor.log"
)
ELASTIC_LOG_PATH = (
    REPO_ROOT
    / f"logs/verl_deployment_modes/mixed_test_new_request_level_candidate_evaluation_mode5_bs_128_share_32_elastic_rl_Qwen2.5-32B_Qwen3-30B-A3B-Thinking-2507/ElasticMonitor.log"
)
_OUT_TAG = "elastic_only" if PLOT_ELASTIC_ONLY else "compare_none_vs_new"
OUT_PATH = (
    REPO_ROOT
    / f"logs/verl_deployment_modes/{_OUT_TAG}_none_new/"
    f"none_new_share_32_totals_merged_"
    f"{PLOT_WINDOW_START_MIN}-{PLOT_WINDOW_END_MIN}_min.png"
)

# NON_ELASTIC_LOG_PATH = (
#     REPO_ROOT
#     / f"logs/psrl_elastic_rm/none_elastic_min_4_share_8_train_8_rm_qwen_8b_rollout_qwen_7b_staleness_{STALINESS}/ElasticMonitor.log"
# )
# ELASTIC_LOG_PATH = (
#     REPO_ROOT
#     / f"logs/psrl_elastic_rm/debug_itl_policy_min_0_share_8_train_8_rm_qwen_8b_rollout_qwen_7b_staleness_{STALINESS}/ElasticMonitor.log"
# )
# OUT_PATH = (
#     REPO_ROOT
#     / f"logs/psrl_elastic_rm/compare_elastic_min0_vs_min4_itl_policy_min_0_share_8_train_8_rm_qwen_8b_rollout_qwen_7b_staleness_{STALINESS}/Initial_ElasticMonitor_rm_rollout_totals_merged.png"
# )


def _normalize_ts(dt: datetime) -> datetime:
    if AGGREGATE_TS_TO_SECOND:
        return dt.replace(microsecond=0)
    return dt


def _role_count(by_role: dict, role: str) -> float:
    value = by_role.get(role, 0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_by_role_dict(raw: str) -> dict | None:
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def parse_monitor_totals(
    log_path: Path,
) -> tuple[list[datetime], dict[str, dict[str, list[float]]]]:
    """
    Parse Instance current Status lines; aggregate throughput/running/waiting by role.

    Returns:
        times, data[role][metric] lists aligned to times.
    """
    grouped: dict[datetime, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"throughput": 0.0, "running": 0.0, "waiting": 0.0})
    )

    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_STATUS_LINE.match(line)
            if not m:
                continue
            dt = _normalize_ts(
                datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            )
            try:
                status_dict = ast.literal_eval(m.group("status"))
            except (SyntaxError, ValueError):
                continue
            role = status_dict.get("role")
            if role not in ROLES:
                continue
            try:
                grouped[dt][role]["throughput"] += float(
                    status_dict.get("throughput", 0.0)
                )
                grouped[dt][role]["running"] += float(status_dict.get("running", 0.0))
                grouped[dt][role]["waiting"] += float(status_dict.get("waiting", 0.0))
            except (TypeError, ValueError):
                continue

    if not grouped:
        return [], {}

    times = sorted(grouped.keys())
    data: dict[str, dict[str, list[float]]] = {
        role: {metric: [] for metric in TOTAL_METRICS} for role in ROLES
    }
    last_seen: dict[str, dict[str, float]] = {
        role: {metric: 0.0 for metric in TOTAL_METRICS} for role in ROLES
    }
    for dt in times:
        role_data = grouped[dt]
        for role in ROLES:
            for metric in TOTAL_METRICS:
                if role in role_data:
                    value = role_data[role][metric]
                else:
                    value = last_seen[role][metric]
                data[role][metric].append(value)
                last_seen[role][metric] = value
    return times, data


def parse_executor_summaries(
    log_path: Path,
) -> dict[datetime, dict[str, dict[str, float]]]:
    """
    Parse awake / router-backlog snapshots keyed by log timestamp.

    Returns:
        grouped[dt][role]["awake"|"backlog"] -> count
    """
    grouped: dict[datetime, dict[str, dict[str, float]]] = defaultdict(
        lambda: {role: {"awake": 0.0, "backlog": 0.0} for role in ROLES}
    )
    with log_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m_awake = RE_AWAKE_LINE.match(line)
            if m_awake:
                by_role = _parse_by_role_dict(m_awake.group("br"))
                if by_role is None:
                    continue
                dt = _normalize_ts(
                    datetime.strptime(m_awake.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
                )
                for role in ROLES:
                    grouped[dt][role]["awake"] = _role_count(by_role, role)
                continue

            m_backlog = RE_BACKLOG_LINE.match(line)
            if not m_backlog:
                continue
            by_role = _parse_by_role_dict(m_backlog.group("br"))
            if by_role is None:
                continue
            dt = _normalize_ts(
                datetime.strptime(m_backlog.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            )
            for role in ROLES:
                grouped[dt][role]["backlog"] = _role_count(by_role, role)
    return grouped


def filter_by_elapsed_minutes(
    times: list[datetime],
    data: dict[str, dict[str, list[float]]],
    t0: datetime,
    start_min: float,
    end_min: float,
) -> tuple[list[datetime], dict[str, dict[str, list[float]]]]:
    """Keep samples whose elapsed time since t0 is in [start_min, end_min] (inclusive)."""
    new_times: list[datetime] = []
    new_data: dict[str, dict[str, list[float]]] = {
        role: {metric: [] for metric in TOTAL_METRICS} for role in ROLES
    }
    for i, t in enumerate(times):
        elapsed_min = (t - t0).total_seconds() / 60.0
        if elapsed_min < start_min or elapsed_min > end_min:
            continue
        new_times.append(t)
        for role in ROLES:
            for metric in TOTAL_METRICS:
                new_data[role][metric].append(data[role][metric][i])
    return new_times, new_data


def align_summaries_to_times(
    times: list[datetime],
    summaries: dict[datetime, dict[str, dict[str, float]]],
) -> dict[str, dict[str, list[float]]]:
    """Align awake/backlog snapshots to the monitor time axis (forward-fill gaps)."""
    data: dict[str, dict[str, list[float]]] = {
        role: {metric: [] for metric in SUMMARY_METRICS} for role in ROLES
    }
    last_seen: dict[str, dict[str, float]] = {
        role: {metric: 0.0 for metric in SUMMARY_METRICS} for role in ROLES
    }
    for dt in times:
        snap = summaries.get(dt)
        for role in ROLES:
            for metric in SUMMARY_METRICS:
                if snap is not None:
                    value = snap[role][metric]
                    last_seen[role][metric] = value
                else:
                    value = last_seen[role][metric]
                data[role][metric].append(value)
    return data


def prepare_series(
    log_path: Path,
) -> tuple[list[float], dict[str, dict[str, list[float]]], dict[str, dict[str, list[float]]]]:
    log_path = log_path.resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log file not found: {log_path}")

    times, data = parse_monitor_totals(log_path)
    if not times:
        raise SystemExit(f"No valid monitor records parsed: {log_path}")

    summaries = parse_executor_summaries(log_path)
    t0 = times[0]
    times, data = filter_by_elapsed_minutes(
        times, data, t0, PLOT_WINDOW_START_MIN, PLOT_WINDOW_END_MIN
    )
    if not times:
        raise SystemExit(
            f"No samples in window for {log_path} (t0={t0}, "
            f"[{PLOT_WINDOW_START_MIN}, {PLOT_WINDOW_END_MIN}] min)."
        )
    summary_data = align_summaries_to_times(times, summaries)
    x_minutes = [(t - t0).total_seconds() / 60.0 for t in times]
    return x_minutes, data, summary_data


def plot_merged(
    series_elastic: tuple[
        list[float],
        dict[str, dict[str, list[float]]],
        dict[str, dict[str, list[float]]],
    ],
    out_path: Path,
    xlim_min: tuple[float, float],
    ylims: dict[str, tuple[float, float] | None],
    figsize: tuple[float, float],
    dpi: int,
    line_width: float,
    series_non_elastic: tuple[
        list[float],
        dict[str, dict[str, list[float]]],
        dict[str, dict[str, list[float]]],
    ]
    | None = None,
    label_elastic: str = "min_0 (elastic)",
    label_non_elastic: str = "min_4 (non-elastic)",
) -> None:
    """
    Plot ElasticMonitor totals; optionally overlay non-elastic curves.

    Args:
        series_elastic: Prepared (x, totals, summaries) for the elastic run.
        out_path (Path): Output PNG path.
        xlim_min (tuple[float, float]): X-axis window in minutes.
        ylims (dict[str, tuple[float, float] | None]): Per-metric y limits.
        figsize (tuple[float, float]): Figure size.
        dpi (int): Save DPI.
        line_width (float): Line width.
        series_non_elastic: Prepared series for non-elastic; None = elastic only.
        label_elastic (str): Legend label for elastic curves.
        label_non_elastic (str): Legend label for non-elastic curves.
    """
    import matplotlib.pyplot as plt

    (x_e, data_e, summary_e) = series_elastic

    fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True, layout="constrained")

    metric_specs = [
        ("throughput", "Total Throughput"),
        ("running", "Total Running Queue"),
        ("waiting", "Total Waiting Queue"),
        ("awake", "Awake Instances"),
        ("backlog", "Router Backlog"),
    ]
    curve_defs: list[
        tuple[
            str,
            list[float],
            dict[str, dict[str, list[float]]],
            dict[str, dict[str, list[float]]],
            str,
            str,
        ]
    ] = []
    if series_non_elastic is not None:
        x_ne, data_ne, summary_ne = series_non_elastic
        curve_defs.extend(
            [
                (
                    label_non_elastic,
                    x_ne,
                    data_ne,
                    summary_ne,
                    "RewardModel",
                    COLOR_MIN4_REWARDMODEL,
                ),
                (
                    label_non_elastic,
                    x_ne,
                    data_ne,
                    summary_ne,
                    "Rollout",
                    COLOR_MIN4_ROLLOUT,
                ),
            ]
        )
    curve_defs.extend(
        [
            (
                label_elastic,
                x_e,
                data_e,
                summary_e,
                "RewardModel",
                COLOR_MIN0_REWARDMODEL,
            ),
            (label_elastic, x_e, data_e, summary_e, "Rollout", COLOR_MIN0_ROLLOUT),
        ]
    )

    legend_ncol = 2 if series_non_elastic is not None else 1
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for run_label, x, totals, summaries, role, color in curve_defs:
            series = summaries[role] if metric in SUMMARY_METRICS else totals[role]
            ax.plot(
                x,
                series[metric],
                label=f"{role} ({run_label})",
                color=color,
                linestyle="-",
                linewidth=line_width,
                alpha=0.95,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim_min[0], xlim_min[1])
        if ylims[metric] is not None:
            ymin, ymax = ylims[metric]
            ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=8, ncol=legend_ncol)

    axes[-1].set_xlabel("Elapsed time from first log timestamp (min)")
    if series_non_elastic is None:
        fig.suptitle(
            "ElasticMonitor — elastic only: dark red (RM), dark blue (Rollout); solid lines"
        )
    else:
        fig.suptitle(
            "ElasticMonitor — min_4: light red (RM), light blue (Rollout); "
            "min_0 (elastic): dark red (RM), dark blue (Rollout); solid lines"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)


def main() -> None:
    if PLOT_WINDOW_START_MIN > PLOT_WINDOW_END_MIN:
        raise SystemExit("PLOT_WINDOW_START_MIN must be <= PLOT_WINDOW_END_MIN.")

    ylims: dict[str, tuple[float, float] | None] = {
        "throughput": PLOT_YLIM_THROUGHPUT,
        "running": PLOT_YLIM_RUNNING,
        "waiting": PLOT_YLIM_WAITING,
        "awake": PLOT_YLIM_AWAKE,
        "backlog": PLOT_YLIM_BACKLOG,
    }
    for metric, ylim in ylims.items():
        if ylim is not None and ylim[0] > ylim[1]:
            raise SystemExit(f"Invalid ylim for {metric}: ymin > ymax.")

    elastic = prepare_series(ELASTIC_LOG_PATH)
    non_elastic = None if PLOT_ELASTIC_ONLY else prepare_series(NON_ELASTIC_LOG_PATH)

    if non_elastic is None:
        print(
            f"Elastic only: {len(elastic[0])} points | "
            f"Window {PLOT_WINDOW_START_MIN}–{PLOT_WINDOW_END_MIN} min"
        )
    else:
        print(
            f"Non-elastic: {len(non_elastic[0])} points | "
            f"Elastic: {len(elastic[0])} points | "
            f"Window {PLOT_WINDOW_START_MIN}–{PLOT_WINDOW_END_MIN} min"
        )

    plot_merged(
        elastic,
        series_non_elastic=non_elastic,
        out_path=OUT_PATH.resolve(),
        xlim_min=(PLOT_WINDOW_START_MIN, PLOT_WINDOW_END_MIN),
        ylims=ylims,
        figsize=FIGSIZE,
        dpi=SAVE_DPI,
        line_width=LINE_WIDTH,
    )
    print(f"Wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
