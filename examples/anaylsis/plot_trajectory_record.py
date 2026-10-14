#!/usr/bin/env python3
"""
Load a trajectory profiling JSONL file and produce aggregate statistics and plots.

Reads every line as a ``TrajectoryProfilingData`` record, computes per-trajectory
and per-turn statistics (min / max / mean / std / p50 / p90 / p99), prints a
formatted summary table, and saves a multi-panel figure.

Usage::

    python plot_trajectory_record.py /path/to/trajectory_profiling.jsonl
    python plot_trajectory_record.py /path/to/trajectory_profiling.jsonl --out report.png
    python plot_trajectory_record.py /path/to/trajectory_profiling.jsonl --no-plot
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# --- Optional plotting dependency ---
try:
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

from psrl.utils.profiling.records import TrajectoryProfilingData

psrl_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# --- Input ---


def load_records(path: str | Path) -> list[TrajectoryProfilingData]:
    """
    Read all trajectory profiling records from a JSONL file.

    Each line must be a JSON object parseable as a `TrajectoryProfilingData`
    record. Malformed lines are skipped with a warning. A missing file returns
    an empty list.

    Args:
        path (str | Path): Path to the JSONL file.

    Returns:
        list[TrajectoryProfilingData]: Parsed records in file order.
    """
    path = Path(path)

    if not path.exists():
        psrl_logger.warning(f"File not found: {path}.")
        return []

    records: list[TrajectoryProfilingData] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                data = json.loads(raw_line)
                records.append(TrajectoryProfilingData.from_dict(data))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                psrl_logger.warning(f"Skipping malformed line {lineno}: {exc}.")

    psrl_logger.info(f"Loaded records from {path!s}. Count: {len(records)}.")
    return records


# --- Statistics ---


def _stat_dict(values: list[float], *, include_percentiles: bool = True) -> dict:
    """
    Compute summary statistics for a flat list of numeric values.

    Args:
        values (list[float]): Input values. Must be non-empty.
        include_percentiles (bool): Whether to include p50 / p90 / p99.

    Returns:
        dict: Minimum, maximum, mean, standard deviation, and optional percentiles.
    """
    arr = np.array(values, dtype=float)
    result = {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }
    if include_percentiles:
        result["p50"] = float(np.percentile(arr, 50))
        result["p90"] = float(np.percentile(arr, 90))
        result["p99"] = float(np.percentile(arr, 99))
    return result


def compute_stats(records: list[TrajectoryProfilingData]) -> dict:
    """
    Compute aggregate statistics across all trajectory profiling records.

    `compute_summary()` is called on records whose summary dict is empty so
    that all derived fields are available before aggregation.

    Args:
        records (list[TrajectoryProfilingData]): Records to analyze.

    Returns:
        dict: Nested statistics dict. Returns an empty dict when no records
            are provided.
    """
    if not records:
        return {}

    # Ensure summaries are populated.
    for r in records:
        if not r.summary:
            r.compute_summary()

    # --- trajectory-level series ---
    total_durations = [r.total_duration_s for r in records]
    total_turns = [r.total_turns for r in records]
    total_tokens = [r.summary.get("total_generated_tokens", 0) for r in records]
    cache_hit_rates = [r.summary.get("avg_cache_hit_rate", 0.0) for r in records]

    # --- time fraction series ---
    decode_fractions = [r.summary.get("decode_fraction", 0.0) for r in records]
    prefill_fractions = [r.summary.get("prefill_fraction", 0.0) for r in records]
    router_wait_fractions = [r.summary.get("router_wait_fraction", 0.0) for r in records]
    scheduler_wait_fractions = [r.summary.get("scheduler_wait_fraction", 0.0) for r in records]
    env_fractions = [r.summary.get("env_fraction", 0.0) for r in records]

    # --- turn-level series ---
    avg_turn_durations = [r.summary.get("avg_turn_duration_s", 0.0) for r in records]

    decode_throughputs: list[float] = []
    for r in records:
        for turn in r.turn_records:
            if turn.decode_time_s > 0:
                decode_throughputs.append(turn.num_generated_tokens / turn.decode_time_s)

    # --- prefill seqlen series ---
    prefill_seqlens: list[int] = []
    for r in records:
        for turn in r.turn_records:
            for pr in turn.prefill_records:
                prefill_seqlens.append(pr.num_prefill_tokens)

    # --- trigger breakdown ---
    trigger_totals: dict[str, int] = {}
    for r in records:
        breakdown = r.summary.get("prefill_trigger_breakdown", {})
        for trigger, count in breakdown.items():
            trigger_totals[trigger] = trigger_totals.get(trigger, 0) + count

    return {
        "num_trajectories": len(records),
        # trajectory-level
        "total_duration_s": _stat_dict(total_durations),
        "total_turns": _stat_dict(total_turns, include_percentiles=False),
        "total_generated_tokens": _stat_dict(total_tokens),
        "avg_cache_hit_rate": _stat_dict(cache_hit_rates),
        # time fractions
        "decode_fraction": _stat_dict(decode_fractions, include_percentiles=False),
        "prefill_fraction": _stat_dict(prefill_fractions, include_percentiles=False),
        "router_wait_fraction": _stat_dict(router_wait_fractions, include_percentiles=False),
        "scheduler_wait_fraction": _stat_dict(scheduler_wait_fractions, include_percentiles=False),
        "env_fraction": _stat_dict(env_fractions, include_percentiles=False),
        # turn-level
        "avg_turn_duration_s": _stat_dict(avg_turn_durations),
        "decode_throughput_tok_s": (_stat_dict(decode_throughputs) if decode_throughputs else {}),
        "prefill_seqlen": (_stat_dict(prefill_seqlens) if prefill_seqlens else {}),
        # trigger breakdown
        "prefill_trigger_breakdown": trigger_totals,
        # raw series (kept for plotting)
        "_series": {
            "total_duration_s": total_durations,
            "total_turns": total_turns,
            "total_generated_tokens": total_tokens,
            "avg_cache_hit_rate": cache_hit_rates,
            "decode_throughput_tok_s": decode_throughputs,
            "prefill_seqlen": prefill_seqlens,
            "decode_fraction": decode_fractions,
            "prefill_fraction": prefill_fractions,
            "router_wait_fraction": router_wait_fractions,
            "scheduler_wait_fraction": scheduler_wait_fractions,
            "env_fraction": env_fractions,
        },
    }


# --- Console output ---

_COL_W = 11  # column width for the stats table


def _fmt(v: float | None) -> str:
    """
    Format a float value for the stats table.

    Args:
        v (float | None): Value to format.

    Returns:
        str: Right-justified formatted string, or ``"-"`` when ``None``.
    """
    if v is None:
        return "-".rjust(_COL_W)
    return f"{v:.4g}".rjust(_COL_W)


def _row(label: str, s: dict) -> str:
    """
    Format one statistics row.

    Args:
        label (str): Row label.
        s (dict): Stats dict produced by `_stat_dict`.

    Returns:
        str: Formatted table row.
    """
    cols = [
        _fmt(s.get("min")),
        _fmt(s.get("max")),
        _fmt(s.get("mean")),
        _fmt(s.get("std")),
        _fmt(s.get("p50")),
        _fmt(s.get("p90")),
        _fmt(s.get("p99")),
    ]
    return f"  {label:<38}" + "".join(cols)


def print_stats(stats: dict) -> None:
    """
    Print a formatted statistics table to stdout.

    Args:
        stats (dict): Output of `compute_stats`.
    """
    if not stats:
        print("No records to report.")
        return

    header = (
        f"  {'Metric':<38}"
        + "        min"
        + "        max"
        + "       mean"
        + "        std"
        + "        p50"
        + "        p90"
        + "        p99"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print(f"Trajectory Profiling Statistics: {stats['num_trajectories']} trajectories")
    print("=" * len(header))

    print()
    print("  [Trajectory-level]")
    print(header)
    print(sep)
    print(_row("total_duration_s", stats["total_duration_s"]))
    print(_row("total_turns", stats["total_turns"]))
    print(_row("total_generated_tokens", stats["total_generated_tokens"]))
    print(_row("avg_cache_hit_rate", stats["avg_cache_hit_rate"]))

    print()
    print("  [Time Fractions]")
    print(header)
    print(sep)
    print(_row("decode_fraction", stats["decode_fraction"]))
    print(_row("prefill_fraction", stats["prefill_fraction"]))
    print(_row("router_wait_fraction", stats["router_wait_fraction"]))
    print(_row("scheduler_wait_fraction", stats["scheduler_wait_fraction"]))
    print(_row("env_fraction", stats["env_fraction"]))

    print()
    print("  [Turn-level]")
    print(header)
    print(sep)
    print(_row("avg_turn_duration_s", stats["avg_turn_duration_s"]))
    if stats["decode_throughput_tok_s"]:
        print(_row("decode_throughput_tok_s", stats["decode_throughput_tok_s"]))
    if stats["prefill_seqlen"]:
        print(_row("prefill_seqlen_tokens", stats["prefill_seqlen"]))

    breakdown = stats.get("prefill_trigger_breakdown", {})
    if breakdown:
        print()
        print("  [Prefill Trigger Breakdown]")
        total_triggers = sum(breakdown.values()) or 1
        for trigger, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * count / total_triggers
            print(f"    {trigger:<36} {count:>6}  ({pct:.1f}%)")

    print()


# --- Plots ---


def _hist_ax(
    ax,
    values: list[float],
    title: str,
    xlabel: str,
    color: str = "#4C72B0",
    bins: int = 30,
) -> None:
    """
    Draw a histogram with a mean/median annotation onto `ax`.

    Args:
        ax: Matplotlib `Axes` object.
        values (list[float]): Data to plot.
        title (str): Subplot title.
        xlabel (str): X-axis label.
        color (str): Bar fill color.
        bins (int): Number of histogram bins.
    """
    if not values:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    arr = np.array(values)
    ax.hist(arr, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    mean_v = float(np.mean(arr))
    median_v = float(np.median(arr))
    ax.axvline(mean_v, color="tomato", linewidth=1.4, linestyle="--", label=f"mean={mean_v:.3g}")
    ax.axvline(median_v, color="gold", linewidth=1.4, linestyle=":", label=f"p50={median_v:.3g}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _fraction_bar_ax(ax, stats: dict) -> None:
    """
    Draw a horizontal grouped bar chart showing mean ± std for each time fraction.

    Args:
        ax: Matplotlib `Axes` object.
        stats (dict): Output of `compute_stats`.
    """
    labels = ["decode", "router_wait", "env", "prefill", "scheduler_wait"]
    keys = [
        "decode_fraction",
        "router_wait_fraction",
        "env_fraction",
        "prefill_fraction",
        "scheduler_wait_fraction",
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    means = [stats[k]["mean"] * 100 for k in keys]
    stds = [stats[k]["std"] * 100 for k in keys]

    y_pos = np.arange(len(labels))
    bars = ax.barh(
        y_pos, means, xerr=stds, color=colors, edgecolor="white", linewidth=0.4, capsize=4, error_kw={"linewidth": 1.2}
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("fraction of total trajectory time (%)", fontsize=9)
    ax.set_title("Time Fraction Breakdown (mean ± std)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    for bar, mean_v in zip(bars, means):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_v:.1f}%",
            va="center",
            fontsize=8,
        )


def plot_stats(
    stats: dict,
    out_file: str | Path | None = None,
) -> None:
    """
    Produce a multi-panel figure summarising trajectory profiling statistics.

    The figure contains six panels:

    - Row 0: total duration histogram, total turns histogram, total tokens histogram
    - Row 1: cache hit rate histogram, decode throughput histogram, time fraction bar
    - Row 2 (single wide): prefill seqlen histogram

    Args:
        stats (dict): Output of `compute_stats`.
        out_file (str | Path | None): If provided, save the figure to this path.
            If ``None``, the figure is displayed interactively.
    """
    if not _MATPLOTLIB_AVAILABLE:
        psrl_logger.warning("Matplotlib is not available. Skipping plot.")
        return
    if not stats:
        psrl_logger.warning("No stats to plot.")
        return

    series = stats.get("_series", {})

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Trajectory Profiling Report  ({stats['num_trajectories']} trajectories)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # --- Row 0: three histograms ---
    ax00 = fig.add_subplot(3, 3, 1)
    ax01 = fig.add_subplot(3, 3, 2)
    ax02 = fig.add_subplot(3, 3, 3)

    _hist_ax(ax00, series.get("total_duration_s", []), "Trajectory Total Duration", "seconds", color="#4C72B0")
    _hist_ax(ax01, series.get("total_turns", []), "Trajectory Turn Count", "turns", color="#55A868", bins=20)
    _hist_ax(ax02, series.get("total_generated_tokens", []), "Total Generated Tokens", "tokens", color="#8172B2")

    # --- Row 1: cache hit rate, decode throughput, fraction bar ---
    ax10 = fig.add_subplot(3, 3, 4)
    ax11 = fig.add_subplot(3, 3, 5)
    ax12 = fig.add_subplot(3, 3, 6)

    _hist_ax(
        ax10,
        series.get("avg_cache_hit_rate", []),
        "Avg Cache Hit Rate (per trajectory)",
        "cache hit rate",
        color="#C44E52",
    )
    _hist_ax(
        ax11,
        series.get("decode_throughput_tok_s", []),
        "Decode Throughput (per turn)",
        "tokens / second",
        color="#DD8452",
    )
    _fraction_bar_ax(ax12, stats)

    # --- Row 2: prefill seqlen (wide) + trigger breakdown text ---
    ax20 = fig.add_subplot(3, 3, (7, 8))
    ax21 = fig.add_subplot(3, 3, 9)

    _hist_ax(
        ax20,
        series.get("prefill_seqlen", []),
        "Prefill Sequence Length (per prefill event)",
        "tokens",
        color="#64B5CD",
    )

    # Trigger breakdown as text panel.
    breakdown = stats.get("prefill_trigger_breakdown", {})
    ax21.axis("off")
    if breakdown:
        total = sum(breakdown.values()) or 1
        lines = ["Prefill Trigger Breakdown\n"]
        for trigger, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * count / total
            lines.append(f"{trigger}\n  {count}  ({pct:.1f}%)")
        ax21.text(
            0.05,
            0.95,
            "\n".join(lines),
            transform=ax21.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if out_file:
        out_file = Path(out_file)
        plt.savefig(out_file, dpi=150, bbox_inches="tight")
        psrl_logger.info(f"Figure saved to {out_file}.")
        print(f"Figure saved to: {out_file}")
    else:
        plt.show()

    plt.close(fig)


# --- Entry point ---


def main() -> None:
    """
    Parse command-line arguments, load records, compute statistics, and (optionally) plot.
    """
    ap = argparse.ArgumentParser(
        description=(
            "Analyze a trajectory profiling JSONL file: compute min/max/mean/std/percentile "
            "statistics and produce a multi-panel plot."
        )
    )
    ap.add_argument(
        "jsonl",
        help="path to the trajectory_profiling.jsonl file",
    )
    ap.add_argument(
        "--out",
        default=None,
        metavar="FILE",
        help="output figure path (png/pdf); default: show interactively",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="skip figure generation (only print statistics)",
    )
    args = ap.parse_args()

    records = load_records(args.jsonl)
    if not records:
        print("No records loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(records)
    print_stats(stats)

    if not args.no_plot:
        if not _MATPLOTLIB_AVAILABLE:
            print("Matplotlib is not installed. Skipping plot.", file=sys.stderr)
        else:
            out = args.out
            if out is None and not args.no_plot:
                # Default output path next to the input file.
                out = Path(args.jsonl).with_suffix(".png")
            plot_stats(stats, out_file=out)


if __name__ == "__main__":
    main()
