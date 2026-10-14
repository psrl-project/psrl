#!/usr/bin/env python3
"""Plot ITL vs KV cache usage scatter plots from rollout benchmark JSONL details.

For each model, produces two figures:
  - disable attn (disable_attn=true)
  - w/ attn (disable_attn=false)

Each figure overlays all batch sizes with distinct colors.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

FILE_RE = re.compile(
    r"^(?P<model>.+)_Syn_TP(?P<tp>\d+)_PP(?P<pp>\d+)_B(?P<batch>\d+)_P(?P<prompt>\d+)_R(?P<response>\d+)_disable_attn_(?P<disable>true|false)\.jsonl$"
)

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
BATCH_COLORS = plt.cm.viridis([i / (len(BATCH_SIZES) - 1) for i in range(len(BATCH_SIZES))])


def parse_file_name(path: Path) -> dict | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    info["tp"] = int(info["tp"])
    info["pp"] = int(info["pp"])
    info["batch"] = int(info["batch"])
    info["prompt"] = int(info["prompt"])
    info["response"] = int(info["response"])
    info["disable"] = info["disable"] == "true"
    return info


def load_scatter_points(path: Path, expected_batch: int, max_itl: float = 0.1) -> list[tuple[float, float]]:
    """Return (kv_cache_usage, inter_token_latency) points from decode-phase stats."""
    points: list[tuple[float, float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") != "stats":
                continue

            iteration_stats = obj.get("iteration_stats", {})
            scheduler_stats = obj.get("scheduler_stats", {})

            if iteration_stats.get("time_to_first_tokens_avg", 0.0) != 0.0:
                continue
            if iteration_stats.get("num_finished_requests", 0) != 0:
                continue

            running_reqs = int(scheduler_stats.get("num_running_reqs", 0))
            if running_reqs != expected_batch:
                continue

            kv_cache = scheduler_stats.get("kv_cache_usage")
            itl = iteration_stats.get("inter_token_latencies_avg")
            if kv_cache is None or itl is None:
                continue

            kv_cache_val = float(kv_cache)
            itl_val = float(itl)
            if not math.isfinite(kv_cache_val) or not math.isfinite(itl_val):
                continue
            if itl_val <= 0.0 or itl_val > max_itl:
                continue
            if kv_cache_val < 0.0 or kv_cache_val > 1.0:
                continue

            points.append((kv_cache_val, itl_val))
    return points


def discover_models(detail_dir: Path) -> list[str]:
    models = set()
    for path in detail_dir.glob("*.jsonl"):
        info = parse_file_name(path)
        if info is not None:
            models.add(info["model"])
    return sorted(models)


def plot_model_attn_setting(
    detail_dir: Path,
    model: str,
    disable_attn: bool,
    output_path: Path,
    max_itl: float,
) -> int:
    """Plot one figure; return total point count."""
    attn_label = "disable attn" if disable_attn else "w/ attn"
    fig, ax = plt.subplots(figsize=(10, 6))

    total_points = 0
    for batch in BATCH_SIZES:
        matches = [
            path
            for path in detail_dir.glob("*.jsonl")
            if (info := parse_file_name(path)) is not None
            and info["model"] == model
            and info["batch"] == batch
            and info["disable"] == disable_attn
        ]
        if not matches:
            continue
        if len(matches) > 1:
            matches = sorted(matches, key=lambda p: p.name)

        path = matches[0]
        points = load_scatter_points(path, expected_batch=batch, max_itl=max_itl)
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        color = BATCH_COLORS[BATCH_SIZES.index(batch)]
        ax.scatter(
            xs,
            ys,
            s=8,
            alpha=0.35,
            color=color,
            label=f"batch={batch} (n={len(points)})",
            edgecolors="none",
        )
        total_points += len(points)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("KV cache usage")
    ax.set_ylabel("ITL (s)")
    ax.set_title(f"{model} — {attn_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9, markerscale=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path} ({total_points} points)")
    return total_points


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ITL vs KV cache scatter from rollout JSONL details")
    parser.add_argument(
        "detail_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "exp" / "details",
        help="Directory containing rollout JSONL detail files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output PNG files (default: <detail_dir>/../plots/itl_vs_kv_cache)",
    )
    parser.add_argument(
        "--max-itl",
        type=float,
        default=0.1,
        help="Drop ITL outliers above this threshold (seconds)",
    )
    args = parser.parse_args()

    detail_dir = args.detail_dir.resolve()
    if not detail_dir.is_dir():
        raise SystemExit(f"Detail directory not found: {detail_dir}")

    output_dir = args.output_dir.resolve() if args.output_dir else detail_dir.parent / "plots" / "itl_vs_kv_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = discover_models(detail_dir)
    if not models:
        raise SystemExit(f"No matching JSONL files found in {detail_dir}")

    print(f"Found {len(models)} models: {', '.join(models)}")
    for model in models:
        for disable_attn in (True, False):
            suffix = "disable_attn" if disable_attn else "with_attn"
            out_name = f"{sanitize_filename(model)}_{suffix}.png"
            plot_model_attn_setting(
                detail_dir=detail_dir,
                model=model,
                disable_attn=disable_attn,
                output_path=output_dir / out_name,
                max_itl=args.max_itl,
            )

    print(f"Done. Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
