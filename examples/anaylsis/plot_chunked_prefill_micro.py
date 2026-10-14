#!/usr/bin/env python3
"""
Plot chunked prefill microbenchmark throughput by tensor parallel size.

Only rows whose decomposition is `multi` are plotted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_tp_file(spec: str) -> tuple[int, Path]:
    """Parse 'TP:/path/to/file.jsonl' into (tp, path)."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"expected TP:path, got {spec!r} (example: 1:/path/to/file.jsonl)")
    tp_str, path_str = spec.split(":", 1)
    try:
        tp = int(tp_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"TP must be an int, got {tp_str!r}") from e
    path = Path(path_str)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {path}")
    return tp, path


def load_multi_points(path: Path) -> list[tuple[int, float]]:
    """Load (m, throughput_tok_per_s) for decomposition == 'multi', sorted by m."""
    points: list[tuple[int, float]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e

            if obj.get("decomposition") != "multi":
                continue
            if obj.get("skipped"):
                continue
            if "m" not in obj or "throughput_tok_per_s" not in obj:
                continue
            points.append((int(obj["m"]), float(obj["throughput_tok_per_s"])))

    points.sort(key=lambda p: p[0])
    return points


def plot(
    series: list[tuple[int, list[tuple[int, float]]]],
    out: Path,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for tp, points in series:
        if not points:
            print(f"warning: TP={tp} has no multi points, skipping")
            continue
        xs = [m for m, _ in points]
        ys = [tput for _, tput in points]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"TP={tp}")

    ax.set_xlabel("m")
    ax.set_ylabel("throughput_tok_per_s")
    ax.set_title(title or "chunked prefill multi: throughput vs m")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot multi-decomposition throughput vs m for multiple TP jsonl files."
    )
    parser.add_argument(
        "--tp-file",
        action="append",
        required=True,
        type=parse_tp_file,
        metavar="TP:PATH",
        help="TP to jsonl mapping, e.g. 1:/path/to/micro_TP1.jsonl (repeatable)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("chunked_prefill_multi_throughput.png"),
        help="output image path",
    )
    parser.add_argument("--title", type=str, default=None, help="optional plot title")
    args = parser.parse_args()

    # Stable order by TP
    tp_files = sorted(args.tp_file, key=lambda x: x[0])
    series: list[tuple[int, list[tuple[int, float]]]] = []
    for tp, path in tp_files:
        points = load_multi_points(path)
        print(f"TP={tp}: {len(points)} multi points from {path}")
        series.append((tp, points))

    plot(series, args.out, title=args.title)


if __name__ == "__main__":
    main()
