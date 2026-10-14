#!/usr/bin/env python3
"""
plot_seqlen_bins.py

Reads a JSONL file where lines may contain "request_id", "prompt_length", "generated_length".
For each request_id (expected to appear twice, typically start/end),
compute seq_len = prompt_length + generated_length.
Bin seq_len into ranges: <1k, 1k-2k, 2k-4k, 4k-8k, 8k-16k, 16k-32k.
Plot two bars per bin:
 - % of sequences in the bin
 - % of total tokens (sum of seq_len) that fall in the bin

Annotations and labels are in English.
"""

import argparse
import json
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Plot seq length distribution by bins from JSONL logs.")
    p.add_argument("input", help="Input JSONL file path")
    p.add_argument("--out", "-o", default="seqlen_bins.png", help="Output figure file path (png)")
    p.add_argument("--show", action="store_true", help="Show the plot interactively")
    return p.parse_args()


# Define bins as left-inclusive, right-exclusive: [low, high)
BIN_EDGES = [0, 1000, 2000, 4000, 8000, 16000, 32000]
BIN_LABELS = ["<1k", "1k-2k", "2k-4k", "4k-8k", "8k-16k", "16k-32k"]


def find_bin_index(seq_len):
    # clamp seq_len >= 0
    if seq_len < 0:
        return None
    for i in range(len(BIN_EDGES) - 1):
        lo = BIN_EDGES[i]
        hi = BIN_EDGES[i + 1]
        if lo <= seq_len < hi:
            return i
    # seq_len >= last edge
    return None


def main():
    args = parse_args()
    groups = defaultdict(list)

    # Read JSONL
    with open(args.input, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(
                    f"Warning: failed to parse json on line {line_no}: {e}",
                    file=sys.stderr,
                )
                continue
            if "request_id" not in obj:
                continue
            rid = obj["request_id"]
            # if int(rid[6:]) >= 512:
            #    continue
            groups[rid].append(obj)

    seq_lens = []
    skipped = 0
    for rid, entries in groups.items():
        if len(entries) != 2:
            # Salvage available lengths when a request has an unexpected entry count.
            print(
                f"Warning: request_id {rid} has {len(entries)} entries (expected 2). Attempting to salvage.",
                file=sys.stderr,
            )
        # take sum of prompt_length and generated_length found across entries
        prompt = 0
        gen = 0
        for e in entries:
            if "prompt_length" in e and isinstance(e["prompt_length"], (int, float)):
                prompt = max(prompt, int(e["prompt_length"]))  # prefer max if both present
            if "generated_length" in e and isinstance(e["generated_length"], (int, float)):
                gen = max(gen, int(e["generated_length"]))
        if prompt == 0 and gen == 0:
            # nothing meaningful
            print(
                f"Warning: request_id {rid} has no prompt_length or generated_length. Skipping.",
                file=sys.stderr,
            )
            skipped += 1
            continue
        seq_len = int(prompt) + int(gen)
        seq_lens.append(seq_len)

    if not seq_lens:
        print("No sequences found after parsing. Exiting.", file=sys.stderr)
        return

    total_seqs = len(seq_lens)
    total_tokens = sum(seq_lens)

    # Bin counts and token sums
    counts = [0] * (len(BIN_EDGES) - 1)
    token_sums = [0] * (len(BIN_EDGES) - 1)
    out_of_range_count = 0
    out_of_range_tokens = 0

    for s in seq_lens:
        idx = find_bin_index(s)
        if idx is None:
            out_of_range_count += 1
            out_of_range_tokens += s
        else:
            counts[idx] += 1
            token_sums[idx] += s

    # If any out-of-range, report and ignore from plotted bins (alternatively could make an '>=32k' bin)
    if out_of_range_count > 0:
        print(
            f"Note: {out_of_range_count} sequences (tokens={out_of_range_tokens}) "
            f"fall outside defined bins and will be ignored in the plot.",
            file=sys.stderr,
        )
        # Percentages include values outside the plotted bins.

    # Compute percentages (relative to overall totals)
    seq_pct = [(c / total_seqs) * 100.0 for c in counts]
    token_pct = [(t / total_tokens) * 100.0 for t in token_sums]

    # Print summary table
    print("\nSummary (bins):")
    print(f"{'bin':>10} | {'count':>8} | {'% of seqs':>9} | {'tokens':>12} | {'% of tokens':>12}")
    print("-" * 65)
    for i, label in enumerate(BIN_LABELS):
        print(f"{label:>10} | {counts[i]:8d} | {seq_pct[i]:9.2f}% | {token_sums[i]:12d} | {token_pct[i]:12.2f}%")
    if out_of_range_count > 0:
        print("-" * 65)
        print(
            f"{'>=32k or <0':>10} | "
            f"{out_of_range_count:8d} | "
            f"{(out_of_range_count / total_seqs) * 100:9.2f}% | "
            f"{out_of_range_tokens:12d} | "
            f"{(out_of_range_tokens / total_tokens) * 100:12.2f}%"
        )
    print("\nTotal sequences:", total_seqs)
    print("Total tokens:", total_tokens)
    if skipped:
        print(f"Skipped {skipped} request(s) due to missing lengths.", file=sys.stderr)

    # Plot: two bars per bin
    x = np.arange(len(BIN_LABELS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, seq_pct, width, label="% of sequences")
    bars2 = ax.bar(x + width / 2, token_pct, width, label="% of tokens")

    # Labels and title (in English per request)
    ax.set_xlabel("Sequence length bins (tokens)", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Distribution of sequences and tokens by sequence-length bins", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.legend()

    # Annotate bars with percentage values
    def autolabel(bars, fmt="{:.2f}%"):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                fmt.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved plot to {args.out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
