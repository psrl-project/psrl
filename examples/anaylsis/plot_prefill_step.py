#!/usr/bin/env python3
"""
Plot prefill request tokens and host time from step logs.

The output combines a per-step scatter plot with grouped exponential-bin bars.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(
    r"step=(?P<step>\d+)\s+.*?"
    r"host_ms=(?P<host_ms>[\d.]+)\s+.*?"
    r"ctx_tokens=(?P<ctx_tokens>\d+)"
)

# Prefill request lines: "new" or "chunk" (not decode). Capture q= tokens.
REQ_RE = re.compile(r"^\s*\[\d+\]\s+rid=\S+\s+(?:new|chunk)\b.*?q=(?P<q>\d+)")


@dataclass
class StepRecord:
    step: int
    host_ms: float
    ctx_tokens: int
    prefill_qs: list[int] = field(default_factory=list)

    @property
    def prefill_sum(self) -> int:
        return int(sum(self.prefill_qs))


def parse_args():
    p = argparse.ArgumentParser(description="Plot prefill step composition and latency from Prefill_*.log")
    p.add_argument("input", help="Prefill log file path (e.g. Prefill_I1.log)")
    p.add_argument(
        "--out",
        "-o",
        default=None,
        help="Output PNG path (default: <input_basename>_prefill_step.png)",
    )
    p.add_argument("--show", action="store_true", help="Show the plot interactively")
    return p.parse_args()


def parse_prefill_log(path: str) -> list[StepRecord]:
    """Parse steps with ctx_tokens > 0 and attach per-request prefill q values."""
    steps: list[StepRecord] = []
    current: StepRecord | None = None

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if "step=" in line and "ctx_tokens=" in line:
                m = STEP_RE.search(line)
                if not m:
                    continue
                try:
                    step = int(m.group("step"))
                    host_ms = float(m.group("host_ms"))
                    ctx_tokens = int(m.group("ctx_tokens"))
                except ValueError as e:
                    print(
                        f"Warning: bad numeric fields on line {line_no}: {e}",
                        file=sys.stderr,
                    )
                    current = None
                    continue
                if ctx_tokens == 0:
                    current = None
                    continue
                current = StepRecord(step=step, host_ms=host_ms, ctx_tokens=ctx_tokens)
                steps.append(current)
                continue

            if current is None:
                continue
            rm = REQ_RE.search(line)
            if not rm:
                continue
            try:
                q = int(rm.group("q"))
            except ValueError:
                continue
            if q > 0:
                current.prefill_qs.append(q)

    return steps


def build_bin_edges(max_tokens: int) -> list[int]:
    """Edges: 0, 64, 128, 256, 512, ... until covering max_tokens."""
    edges = [0, 64]
    while edges[-1] <= max_tokens:
        edges.append(edges[-1] * 2)
    return edges


def bin_label(lo: int, hi: int) -> str:
    return f"{lo}-{hi}"


def main():
    args = parse_args()
    steps = parse_prefill_log(args.input)
    if not steps:
        print("No prefill steps with ctx_tokens > 0 found. Exiting.", file=sys.stderr)
        sys.exit(1)

    ctx_tokens = np.array([s.ctx_tokens for s in steps], dtype=np.int64)
    host_ms = np.array([s.host_ms for s in steps], dtype=np.float64)
    step_ids = np.array([s.step for s in steps], dtype=np.int64)
    prefill_sums = np.array([s.prefill_sum for s in steps], dtype=np.int64)

    # Flatten per-request prefill q points: (step, q)
    req_steps: list[int] = []
    req_qs: list[int] = []
    for s in steps:
        for q in s.prefill_qs:
            req_steps.append(s.step)
            req_qs.append(q)
    req_steps_arr = np.array(req_steps, dtype=np.int64)
    req_qs_arr = np.array(req_qs, dtype=np.int64)

    edges = build_bin_edges(int(ctx_tokens.max()))
    n_bins = len(edges) - 1
    counts = np.zeros(n_bins, dtype=np.int64)
    sum_ms = np.zeros(n_bins, dtype=np.float64)
    labels = [bin_label(edges[i], edges[i + 1]) for i in range(n_bins)]

    bin_idx = np.digitize(ctx_tokens, edges[1:], right=False)
    # digitize with edges[1:] maps: <64 -> 0, [64,128) -> 1, [128,256) -> 2, ...
    for i, ms in zip(bin_idx, host_ms):
        if 0 <= i < n_bins:
            counts[i] += 1
            sum_ms[i] += ms

    print(f"Parsed {len(steps)} steps from {args.input}")
    print(f"prefill requests (q>0): {len(req_qs)}")
    print(f"ctx_tokens range: [{ctx_tokens.min()}, {ctx_tokens.max()}]")
    print(f"host_ms range: [{host_ms.min():.1f}, {host_ms.max():.1f}]")
    print("\nSummary (bins):")
    print(f"{'bin':>14} | {'steps':>8} | {'sum_ms':>12}")
    print("-" * 42)
    for label, c, s in zip(labels, counts, sum_ms):
        print(f"{label:>14} | {c:8d} | {s:12.1f}")
    print("-" * 42)
    print(f"{'total':>14} | {int(counts.sum()):8d} | {float(sum_ms.sum()):12.1f}")

    out = args.out
    if out is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out = f"{base}_prefill_step.png"

    fig, (ax_scatter, ax_bar) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [1.0, 1.1]})

    if len(req_qs_arr) > 0:
        ax_scatter.scatter(
            req_steps_arr,
            req_qs_arr,
            s=10,
            alpha=0.45,
            c="C0",
            edgecolors="none",
            label="per-request prefill q",
            zorder=2,
        )
    ax_scatter.scatter(
        step_ids,
        prefill_sums,
        s=28,
        c="red",
        marker="o",
        edgecolors="darkred",
        linewidths=0.4,
        label="step sum (prefill tokens)",
        zorder=3,
    )
    ax_scatter.set_xlabel("step")
    ax_scatter.set_ylabel("prefill compute tokens")
    ax_scatter.set_title("Per-step prefill compute tokens (per request + sum)")
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend(loc="upper right")

    x = np.arange(n_bins)
    width = 0.38
    bars1 = ax_bar.bar(x - width / 2, counts, width, label="step count", color="C0")
    ax_bar.set_ylabel("step count")
    ax_bar.set_xlabel("ctx_tokens bins")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=30, ha="right")
    ax_bar.set_title("Prefill steps and latency sum by ctx_tokens bin")
    ax_bar.grid(True, axis="y", alpha=0.3)

    ax_bar2 = ax_bar.twinx()
    bars2 = ax_bar2.bar(x + width / 2, sum_ms, width, label="sum host_ms", color="C1")
    ax_bar2.set_ylabel("sum host_ms")

    # Combined legend
    handles = [bars1, bars2]
    legend_labels = ["step count", "sum host_ms"]
    ax_bar.legend(handles, legend_labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nSaved plot to {out}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
