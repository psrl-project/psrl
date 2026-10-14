"""
Plot request route segments and per-instance KV cache use.

Both panels use minutes relative to the first event. Incomplete route segments are omitted.
"""

import argparse
import logging
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

psrl_logger = logging.getLogger(__file__)

# --- constants ---

_TS_FMT = "%Y-%m-%d %H:%M:%S"
_FIELD_RE = re.compile(r'(\w+)=("[^"]*"|\S+)')
_MIN_SEG_MIN = 1 / 60  # 1 second expressed in minutes (minimum visible width)


# --- helpers ---


def _parse_ts(line: str) -> datetime | None:
    """Parse leading YYYY-MM-DD HH:MM:SS from a log line."""
    if len(line) < 19:
        return None
    try:
        return datetime.strptime(line[:19], _TS_FMT)
    except ValueError:
        return None


def _parse_fields(line: str) -> dict[str, str]:
    """Extract all key=value pairs from a log line into a dict."""
    return {k: v.strip('"') for k, v in _FIELD_RE.findall(line)}


def _instance_colormap(n_instances: int):
    """
    Return a list of n_instances distinct colours drawn from qualitative colormaps.
    """
    pools = [
        plt.cm.tab20.colors,
        plt.cm.tab20b.colors,
        plt.cm.tab20c.colors,
        plt.cm.Set1.colors,
        plt.cm.Set2.colors,
        plt.cm.Set3.colors,
    ]
    palette = []
    for pool in pools:
        palette.extend(pool)
        if len(palette) >= n_instances:
            break
    return palette[:n_instances]


# --- parsing ---


def parse_log(
    log_path: str,
    num_requests: int,
    duration_min: float,
) -> tuple[
    datetime,
    dict[int, list[tuple[float, str, int]]],  # req_id -> [(t_min, kind, inst)]
    dict[int, list[tuple[float, float]]],  # inst -> [(t_min, kv_usage)]
]:
    """
    Stream the log file and collect events within the time window.

    Returns:
        t0 (datetime): Timestamp of the first parsed event.
        req_events: Per-request list of (t_min, kind, instance) tuples.
        kv_events: Per-instance list of (t_min, kv_usage) tuples.
    """
    t0: datetime | None = None
    window_end: float = duration_min

    req_events: dict[int, list[tuple[float, str, int]]] = {}
    kv_events: dict[int, list[tuple[float, float]]] = {}

    lines_read = 0
    events_captured = 0

    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            # Fast pre-filter: only lines with event= are interesting.
            if 'event="' not in line:
                continue

            ts = _parse_ts(line)
            if ts is None:
                continue

            if t0 is None:
                t0 = ts

            t_min = (ts - t0).total_seconds() / 60.0

            if t_min > window_end:
                break

            fields = _parse_fields(line)
            event = fields.get("event", "")

            if event == "route":
                req_id_s = fields.get("request_id")
                inst_s = fields.get("dst_instance")
                kv_s = fields.get("dst_kv_cache_usage")
                if req_id_s is None or inst_s is None:
                    continue
                req_id = int(req_id_s)
                inst = int(inst_s)
                if 0 <= req_id < num_requests:
                    req_events.setdefault(req_id, []).append((t_min, "route", inst))
                if kv_s is not None:
                    kv_events.setdefault(inst, []).append((t_min, float(kv_s)))

            elif event == "pop":
                req_id_s = fields.get("request_id")
                inst_s = fields.get("instance")
                kv_s = fields.get("kv_cache_usage")
                if req_id_s is None or inst_s is None:
                    continue
                req_id = int(req_id_s)
                inst = int(inst_s)
                if 0 <= req_id < num_requests:
                    req_events.setdefault(req_id, []).append((t_min, "pop", inst))
                if kv_s is not None:
                    kv_events.setdefault(inst, []).append((t_min, float(kv_s)))

            events_captured += 1
            lines_read += 1

    psrl_logger.info(f"Parsed relevant lines: {lines_read}. Captured route and pop events: {events_captured}.")

    if t0 is None:
        raise ValueError("No parseable events found in the log file.")

    return t0, req_events, kv_events


# --- segment pairing ---


def pair_segments(
    req_events: dict[int, list[tuple[float, str, int]]],
) -> dict[int, list[tuple[float, float, int]]]:
    """
    For each request, pair route→pop events into segments.

    Only complete route→pop pairs are kept. A route with no matching pop
    (orphaned mid-stream, or still open at window end) is dropped and not drawn.

    Returns:
        segments: req_id -> list of (t_start, t_end, instance).
    """
    segments: dict[int, list[tuple[float, float, int]]] = {}

    for req_id, events in req_events.items():
        segs: list[tuple[float, float, int]] = []
        open_route: tuple[float, int] | None = None

        for t_min, kind, inst in events:
            if kind == "route":
                # A new route supersedes any unmatched route.
                open_route = (t_min, inst)
            elif kind == "pop":
                if open_route is not None:
                    t_r, inst_r = open_route
                    t_end = max(t_min, t_r + _MIN_SEG_MIN)
                    segs.append((t_r, t_end, inst_r))
                    open_route = None
                # A pop without an open route predates the window.

        # Do not draw trailing routes without a pop in the window.
        if segs:
            segments[req_id] = segs

    return segments


# --- plotting ---


def plot(
    segments: dict[int, list[tuple[float, float, int]]],
    kv_events: dict[int, list[tuple[float, float]]],
    window_end_min: float,
    num_requests: int,
    n_instances: int,
    kv_max_points_per_instance: int,
    height_ratio: tuple[int, int],
    title: str | None,
    dpi: int,
    out_path: str,
) -> None:
    """Render and save the dual-panel figure."""
    colors = _instance_colormap(n_instances)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, (ax_top, ax_kv) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(20, 10),
            gridspec_kw={"height_ratios": list(height_ratio)},
        )

        # --- top panel: request timeline ---
        legend_handles: dict[int, matplotlib.lines.Line2D] = {}

        for req_id, segs in segments.items():
            for t_start, t_end, inst in segs:
                col = colors[inst % len(colors)]
                line = matplotlib.lines.Line2D(
                    [t_start, t_end],
                    [req_id, req_id],
                    color=col,
                    linewidth=1.5,
                    alpha=0.85,
                    linestyle="-",
                    solid_capstyle="butt",
                )
                ax_top.add_line(line)
                if inst not in legend_handles:
                    legend_handles[inst] = matplotlib.lines.Line2D(
                        [],
                        [],
                        color=col,
                        linewidth=2,
                        label=f"inst {inst}",
                    )

        ax_top.set_xlim(0, window_end_min)
        ax_top.set_ylim(-1, num_requests)
        ax_top.invert_yaxis()
        ax_top.set_ylabel("Request ID", fontsize=13)
        if title:
            ax_top.set_title(title, fontsize=14)

        # Sort the instance legend outside the right edge.
        if legend_handles:
            sorted_handles = [legend_handles[k] for k in sorted(legend_handles)]
            ax_top.legend(
                handles=sorted_handles,
                loc="upper left",
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
                fontsize=8,
                ncol=max(1, len(sorted_handles) // 20),
                frameon=True,
                title="Instance",
                title_fontsize=9,
            )

        # --- bottom panel: KV cache ---
        for inst in sorted(kv_events):
            pts = kv_events[inst]
            if not pts:
                continue
            col = colors[inst % len(colors)]
            ts_arr = np.array([p[0] for p in pts])
            kv_arr = np.array([p[1] for p in pts])
            # Stride subsample to avoid overplotting.
            n = len(ts_arr)
            if n > kv_max_points_per_instance:
                stride = n // kv_max_points_per_instance
                ts_arr = ts_arr[::stride]
                kv_arr = kv_arr[::stride]
            ax_kv.scatter(
                ts_arr,
                kv_arr,
                color=col,
                s=2,
                alpha=0.35,
                linewidths=0,
                label=f"inst {inst}",
            )

        ax_kv.set_xlim(0, window_end_min)
        ax_kv.set_ylim(-0.02, 1.05)
        ax_kv.set_ylabel("KV cache usage", fontsize=13)
        ax_kv.set_xlabel("Time (min)", fontsize=13)
        ax_kv.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))

        fig.tight_layout()
        psrl_logger.info(f"Saving figure to {out_path}.")
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        psrl_logger.info(f"Saved to {out_path}.")
        plt.close(fig)


# --- CLI ---


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Plot per-request route/pop timeline (top) and per-instance KV-cache "
            "scatter (bottom) from a PSRL route_trace.log file."
        ),
    )
    p.add_argument("log_path", help="Path to route_trace.log.")
    p.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=1024,
        help="Number of requests to plot (IDs 0..n-1). Default: 1024.",
    )
    p.add_argument(
        "-d",
        "--duration-min",
        type=float,
        required=True,
        help="Plot window length in minutes from the first log event.",
    )
    p.add_argument(
        "-o",
        "--out",
        default=str(Path(__file__).parent / "request_route_timeline.png"),
        help="Output PNG path. Default: request_route_timeline.png next to this script.",
    )
    p.add_argument(
        "--instances",
        type=int,
        default=None,
        help=("Number of instances (for colour map). Auto-detected from the log if omitted."),
    )
    p.add_argument(
        "--kv-max-points-per-instance",
        type=int,
        default=20_000,
        help="Max KV-scatter points per instance (stride-subsampled). Default: 20000.",
    )
    p.add_argument("--title", default=None, help="Optional figure title.")
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI. Default: 150.",
    )
    p.add_argument(
        "--height-ratio",
        type=int,
        nargs=2,
        default=[3, 1],
        metavar=("TOP", "BOTTOM"),
        help="Height ratio of top:bottom panels. Default: 3 1.",
    )
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _build_parser().parse_args()

    psrl_logger.info(
        f"Parsing log={args.log_path!r} for request range=0..{args.num_requests - 1} over minutes={args.duration_min}."
    )

    t0, req_events, kv_events = parse_log(
        args.log_path,
        args.num_requests,
        args.duration_min,
    )

    psrl_logger.info(
        f"Parsed window t0={t0!s}. Requests with events: {len(req_events)}. Instances with KV data: {len(kv_events)}."
    )

    segments = pair_segments(req_events)
    psrl_logger.info(f"Paired request segments. Count: {len(segments)}.")

    n_instances = args.instances
    if n_instances is None:
        # Infer from route dst_instance and pop instance observed in the window.
        all_insts = set(kv_events.keys())
        for segs in segments.values():
            for _, _, inst in segs:
                all_insts.add(inst)
        n_instances = max(all_insts) + 1 if all_insts else 8
        psrl_logger.info(f"Auto-detected instance count: {n_instances}.")

    plot(
        segments=segments,
        kv_events=kv_events,
        window_end_min=args.duration_min,
        num_requests=args.num_requests,
        n_instances=n_instances,
        kv_max_points_per_instance=args.kv_max_points_per_instance,
        height_ratio=tuple(args.height_ratio),
        title=args.title,
        dpi=args.dpi,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
