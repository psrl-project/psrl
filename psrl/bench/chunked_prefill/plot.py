"""
Plot micro-benchmark results from JSONL files.

The E1 figure shows throughput and MFU by token count. E2 shows mixed-batch
overhead, E3 compares activation memory with KV capacity, and E_chunked shows
per-step and total sequence latency.

Usage:

    python -m psrl.bench.chunked_prefill.plot --results-dir ./results --out plots/

    # Only E1:
    python -m psrl.bench.chunked_prefill.plot \\
        --results-dir ./results --experiments e1 --out plots/

    # Include E3b JSON files from memory sweep:
    python -m psrl.bench.chunked_prefill.plot \\
        --results-dir ./results --e3b-dir ./results/e3b --out plots/
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# --- Data Loading ---


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all JSON records from a JSONL file, skipping malformed lines."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Warning: skipped malformed line {lineno} in {path}: {exc}.")
    return records


def _load_results_dir(results_dir: Path) -> list[dict[str, Any]]:
    """Load all JSONL records from a directory (non-recursive)."""
    all_records: list[dict[str, Any]] = []
    for p in sorted(results_dir.glob("*.jsonl")):
        all_records.extend(_load_jsonl(p))
    return all_records


def _load_e3b_dir(e3b_dir: Path) -> list[dict[str, Any]]:
    """Load all JSON records from E3b per-N probe files."""
    records = []
    for p in sorted(e3b_dir.glob("*.json")):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"Warning: could not load {p}: {exc}.")
    return records


# --- CSV Export ---


def _write_csv(records: list[dict[str, Any]], out_path: Path, fields: list[str]) -> None:
    """Write a flat list of dicts as a CSV, keeping only ``fields`` columns."""
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


# --- Plot Helpers ---

_FIG_WIDTH = 9
_FIG_HEIGHT_PER_ROW = 4

# Default budget visible as a dashed reference line on E1 plots.
_DEFAULT_BUDGET_TOKENS = 16384


def _label(rec: dict[str, Any]) -> str:
    tp = rec.get("tensor_parallel_size", "?")
    decomp = rec.get("decomposition", "")
    return f"TP={tp}" + (f" ({decomp})" if decomp else "")


def _style_for_label(label: str) -> dict[str, Any]:
    """Assign consistent marker/linestyle per label."""
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]
    idx = hash(label) % len(markers)
    lsidx = (hash(label) // len(markers)) % len(linestyles)
    return {"marker": markers[idx], "linestyle": linestyles[lsidx], "markersize": 5}


# --- E1 Plot ---


def plot_e1(records: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot throughput(M) and MFU(M) from E1 records."""
    e1 = [r for r in records if r.get("experiment") == "e1" and not r.get("skipped")]
    if not e1:
        print("No E1 records found, skipping E1 plot.")
        return

    # Group by label.
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in e1:
        by_label[_label(rec)].append(rec)

    fig, (ax_thr, ax_mfu) = plt.subplots(2, 1, figsize=(_FIG_WIDTH, 2 * _FIG_HEIGHT_PER_ROW), sharex=True)

    for lbl, recs in sorted(by_label.items()):
        recs_sorted = sorted(recs, key=lambda r: r.get("m", 0))
        ms = [r["m"] for r in recs_sorted]
        thrs = [r.get("throughput_tok_per_s") or 0 for r in recs_sorted]
        mfus = [r.get("mfu") for r in recs_sorted]
        style = _style_for_label(lbl)
        ax_thr.plot(ms, thrs, label=lbl, **style)
        valid_mfu = [(m, v) for m, v in zip(ms, mfus) if v is not None]
        if valid_mfu:
            mx, vy = zip(*valid_mfu)
            ax_mfu.plot(mx, vy, label=lbl, **style)

    for ax in (ax_thr, ax_mfu):
        ax.axvline(
            _DEFAULT_BUDGET_TOKENS,
            color="grey",
            linestyle=":",
            linewidth=1,
            label=f"default budget ({_DEFAULT_BUDGET_TOKENS})",
        )
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    ax_thr.set_ylabel("Throughput (tokens/s)")
    ax_thr.set_title("E1: Prefill throughput vs step token count M")
    ax_mfu.set_ylabel("MFU")
    ax_mfu.set_xlabel("Total query tokens M (log₂ scale)")
    ax_mfu.set_title("E1: Model FLOPs utilisation vs M")

    fig.tight_layout()
    out_path = out_dir / "e1_mfu.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved E1 plot to {out_path}.")

    # CSV export.
    csv_fields = [
        "run_id",
        "tensor_parallel_size",
        "decomposition",
        "m",
        "num_reqs",
        "total_q_tokens",
        "latency_ms_median",
        "latency_ms_p10",
        "latency_ms_p90",
        "throughput_tok_per_s",
        "mfu",
        "flops",
    ]
    _write_csv(e1, out_dir / "e1_results.csv", csv_fields)
    print(f"Saved E1 CSV to {out_dir / 'e1_results.csv'}.")


# --- E2 Plot ---


def plot_e2(records: list[dict[str, Any]], out_dir: Path) -> None:
    """Plot E2 overhead heat-maps (one subplot per context length)."""
    e2 = [r for r in records if r.get("experiment") == "e2" and not r.get("skipped")]
    if not e2:
        print("No E2 records found, skipping E2 plot.")
        return

    # Build latency lookup: (decode_count, prefill_chunk, context_len, variant) → latency_ms.
    lat: dict[tuple, float] = {}
    for rec in e2:
        key = (
            rec.get("decode_count", 0),
            rec.get("prefill_chunk", 0),
            rec.get("context_len", 0),
            rec.get("variant", ""),
        )
        lat[key] = rec.get("latency_ms_median") or 0.0

    context_lens = sorted({r.get("context_len", 0) for r in e2 if r.get("context_len")})
    decode_counts = sorted({r.get("decode_count", 0) for r in e2 if r.get("decode_count") is not None})
    prefill_chunks = sorted({r.get("prefill_chunk", 0) for r in e2 if r.get("prefill_chunk")})

    if not context_lens or not decode_counts or not prefill_chunks:
        print("Insufficient E2 data for heat-map, skipping.")
        return

    ncols = len(context_lens)
    fig, axes = plt.subplots(1, ncols, figsize=(_FIG_WIDTH * ncols // 2, _FIG_HEIGHT_PER_ROW))
    if ncols == 1:
        axes = [axes]

    for ax, l_ctx in zip(axes, context_lens):
        # Build matrix: rows = decode_counts, cols = prefill_chunks.
        matrix = np.full((len(decode_counts), len(prefill_chunks)), np.nan)
        for i, d in enumerate(decode_counts):
            for j, c in enumerate(prefill_chunks):
                t_mixed = lat.get((d, c, l_ctx, "mixed"), 0.0)
                t_dec = lat.get((d, c, l_ctx, "decode_only"), 0.0)
                t_pf = lat.get((d, c, l_ctx, "prefill_only"), 0.0)
                denom = t_dec + t_pf
                if denom > 0 and t_mixed > 0:
                    matrix[i, j] = t_mixed / denom

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0.9, vmax=1.5)
        ax.set_xticks(range(len(prefill_chunks)))
        ax.set_xticklabels([str(c) for c in prefill_chunks], rotation=45, ha="right")
        ax.set_yticks(range(len(decode_counts)))
        ax.set_yticklabels([str(d) for d in decode_counts])
        ax.set_xlabel("Prefill chunk C (tokens)")
        ax.set_ylabel("Decode count D")
        ax.set_title(f"E2 overhead: ctx={l_ctx}")
        plt.colorbar(im, ax=ax, label="overhead (1=no overhead)")

        # Annotate cells.
        for i in range(len(decode_counts)):
            for j in range(len(prefill_chunks)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    out_path = out_dir / "e2_overhead.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved E2 plot to {out_path}.")

    csv_fields = [
        "run_id",
        "tensor_parallel_size",
        "decode_count",
        "prefill_chunk",
        "context_len",
        "variant",
        "num_reqs",
        "total_q_tokens",
        "latency_ms_median",
        "latency_ms_p10",
        "latency_ms_p90",
    ]
    _write_csv(e2, out_dir / "e2_results.csv", csv_fields)
    print(f"Saved E2 CSV to {out_dir / 'e2_results.csv'}.")


# --- E3 Plot ---


def plot_e3(
    records: list[dict[str, Any]],
    e3b_records: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    """Plot activation increment (E3a) and memory reservation (E3b) vs M."""
    e3a = [r for r in records if r.get("experiment") == "e3a" and not r.get("skipped")]
    e3b = e3b_records

    if not e3a and not e3b:
        print("No E3 records found, skipping E3 plot.")
        return

    fig, ax_act = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT_PER_ROW))
    ax_kv = ax_act.twinx()

    # E3a: per-step activation increment.
    if e3a:
        by_tp: dict[str, list] = defaultdict(list)
        for rec in e3a:
            by_tp[str(rec.get("tensor_parallel_size", "?"))].append(rec)
        for tp_lbl, recs in sorted(by_tp.items()):
            recs_sorted = sorted(recs, key=lambda r: r.get("m", 0))
            ms = [r["m"] for r in recs_sorted]
            acts_gib = [((r.get("activation_bytes_median") or 0) / 1024**3) for r in recs_sorted]
            ax_act.plot(ms, acts_gib, marker="o", linestyle="-", label=f"TP={tp_lbl} act. increment (GiB)")

    # E3b: reserved activation and KV token capacity vs N.
    if e3b:
        by_util: dict[str, list] = defaultdict(list)
        for rec in e3b:
            by_util[str(rec.get("gpu_memory_utilization", "?"))].append(rec)
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for idx, (util_lbl, recs) in enumerate(sorted(by_util.items())):
            recs_sorted = sorted(recs, key=lambda r: r.get("max_num_batched_tokens", 0))
            ns = [r["max_num_batched_tokens"] for r in recs_sorted]
            resv_gib = [r.get("peak_activation_bytes", 0) / 1024**3 for r in recs_sorted]
            kv_ktok = [r.get("kv_token_capacity", 0) / 1000 for r in recs_sorted]
            col = colors[idx % len(colors)]
            ax_act.plot(
                ns,
                resv_gib,
                marker="s",
                linestyle="--",
                color=col,
                label=f"util={util_lbl} reserved act. (GiB)",
            )
            ax_kv.plot(
                ns,
                kv_ktok,
                marker="^",
                linestyle=":",
                color=col,
                alpha=0.7,
                label=f"util={util_lbl} KV capacity (k tokens)",
            )

    ax_act.axvline(
        _DEFAULT_BUDGET_TOKENS,
        color="grey",
        linestyle=":",
        linewidth=1,
        label=f"default budget ({_DEFAULT_BUDGET_TOKENS})",
    )
    ax_act.set_xscale("log", base=2)
    ax_act.set_xlabel("max_num_batched_tokens N (log₂ scale)")
    ax_act.set_ylabel("Activation (GiB)")
    ax_kv.set_ylabel("KV token capacity (k tokens)")
    ax_act.set_title("E3: Activation footprint and KV capacity vs N")

    lines1, labels1 = ax_act.get_legend_handles_labels()
    lines2, labels2 = ax_kv.get_legend_handles_labels()
    ax_act.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax_act.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "e3_activation_kv.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved E3 plot to {out_path}.")

    if e3b:
        csv_fields = [
            "timestamp",
            "model_path",
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_num_batched_tokens",
            "total_gpu_bytes",
            "requested_bytes",
            "weights_bytes",
            "peak_activation_bytes",
            "non_torch_bytes",
            "available_kv_bytes",
            "num_gpu_blocks",
            "block_size_tokens",
            "kv_token_capacity",
        ]
        _write_csv(e3b, out_dir / "e3b_results.csv", csv_fields)
        print(f"Saved E3b CSV to {out_dir / 'e3b_results.csv'}.")


# --- E_chunked Plot ---


def plot_e_chunked(records: list[dict[str, Any]], out_dir: Path) -> None:
    """
    Plot E_chunked per-step and total sequence latency.

    Produce one figure per `(total_len, prefix_len, num_decode_reqs)` combination.
    """
    ec = [r for r in records if r.get("experiment") == "e_chunked" and not r.get("skipped")]
    if not ec:
        print("No E_chunked records found, skipping E_chunked plot.")
        return

    # Group by (total_len, prefix_len, num_decode_reqs).
    combos: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for rec in ec:
        key = (
            rec.get("total_len", 0),
            rec.get("prefix_len", 0),
            rec.get("num_decode_reqs", 0),
        )
        combos[key].append(rec)

    for (total_len, prefix_len, num_dec), recs in sorted(combos.items()):
        recs_sorted = sorted(recs, key=lambda r: r.get("chunk_size", 0))
        if not recs_sorted:
            continue

        # ---- Figure 1: per-step latency breakdown (stacked bar per chunk_size) ----
        chunk_sizes = [r["chunk_size"] for r in recs_sorted]
        max_steps = max(len(r.get("step_latency_ms_median") or []) for r in recs_sorted)
        if max_steps == 0:
            continue

        # Colour map: each step gets a colour.
        cmap = plt.get_cmap("tab10")
        step_colors = [cmap(s % 10) for s in range(max_steps)]

        fig, (ax_bar, ax_total) = plt.subplots(1, 2, figsize=(_FIG_WIDTH, _FIG_HEIGHT_PER_ROW))
        x_pos = np.arange(len(chunk_sizes))
        bar_width = 0.6

        bottoms = np.zeros(len(chunk_sizes))
        for step_idx in range(max_steps):
            heights = []
            for rec in recs_sorted:
                lats = rec.get("step_latency_ms_median") or []
                heights.append(lats[step_idx] if step_idx < len(lats) else 0.0)
            heights = np.array(heights, dtype=float)
            ax_bar.bar(
                x_pos,
                heights,
                bar_width,
                bottom=bottoms,
                color=step_colors[step_idx],
                label=f"step {step_idx}",
                alpha=0.85,
            )
            bottoms += heights

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([str(c) for c in chunk_sizes])
        ax_bar.set_xlabel("Chunk size (tokens per step)")
        ax_bar.set_ylabel("Latency (ms)")
        ax_bar.set_title(f"E_chunked step breakdown\ntotal={total_len} prefix={prefix_len} dec={num_dec}")
        ax_bar.legend(fontsize=7, ncol=2)
        ax_bar.grid(True, axis="y", alpha=0.3)

        # ---- Figure 2: total sequence latency vs chunk_size ----
        total_lats = [r.get("sequence_latency_ms_median") or 0 for r in recs_sorted]
        p10_lats = [r.get("sequence_latency_ms_p10") or 0 for r in recs_sorted]
        p90_lats = [r.get("sequence_latency_ms_p90") or 0 for r in recs_sorted]
        ax_total.errorbar(
            chunk_sizes,
            total_lats,
            yerr=[
                [max(0, m - p) for m, p in zip(total_lats, p10_lats)],
                [max(0, p - m) for m, p in zip(total_lats, p90_lats)],
            ],
            marker="o",
            capsize=4,
            label="sequence latency",
        )
        ax_total.set_xlabel("Chunk size (tokens per step)")
        ax_total.set_ylabel("Total sequence latency (ms)")
        ax_total.set_title("E_chunked: total latency vs chunk size")
        ax_total.grid(True, alpha=0.3)
        ax_total.legend(fontsize=8)

        fig.tight_layout()
        out_path = out_dir / f"e_chunked_T{total_len}_P{prefix_len}_D{num_dec}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved E_chunked plot to {out_path}.")

    # CSV export.
    csv_fields = [
        "run_id",
        "tensor_parallel_size",
        "total_len",
        "chunk_size",
        "prefix_len",
        "num_decode_reqs",
        "num_steps",
        "sequence_latency_ms_median",
        "sequence_latency_ms_p10",
        "sequence_latency_ms_p90",
        "sequence_throughput_tok_per_s",
    ]
    _write_csv(ec, out_dir / "e_chunked_results.csv", csv_fields)
    print(f"Saved E_chunked CSV to {out_dir / 'e_chunked_results.csv'}.")


# --- Entry Point ---


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot chunked prefill micro-benchmark results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory containing .jsonl result files.",
    )
    parser.add_argument(
        "--e3b-dir",
        type=Path,
        default=None,
        help="Directory containing E3b per-N JSON probe files (optional).",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["e1", "e2", "e3", "e_chunked"],
        help="Which experiments to plot. Choices: e1, e2, e3.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./plots"),
        help="Output directory for PNG files and CSV summaries.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    records = _load_results_dir(args.results_dir)
    print(f"Loaded {len(records)} records from {args.results_dir}.")

    e3b_records: list[dict[str, Any]] = []
    if args.e3b_dir and args.e3b_dir.exists():
        e3b_records = _load_e3b_dir(args.e3b_dir)
        print(f"Loaded {len(e3b_records)} E3b probe records from {args.e3b_dir}.")

    if "e1" in args.experiments:
        plot_e1(records, args.out)
    if "e2" in args.experiments:
        plot_e2(records, args.out)
    if "e3" in args.experiments:
        plot_e3(records, e3b_records, args.out)

    if "e_chunked" in args.experiments:
        plot_e_chunked(records, args.out)

    print("Done.")


if __name__ == "__main__":
    main()
