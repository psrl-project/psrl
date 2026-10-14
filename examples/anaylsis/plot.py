#!/usr/bin/env python3
"""
Plot processed time series from one or more JSONL files.

A caller supplies a processor that selects rows and returns labels plus an x value.
"""

import argparse
import ast
import glob
import json
import math
import os
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
from processor import build_processor


def natural_sort_key(text: str):
    """
    Generate a sort key that treats numbers in strings as numbers (natural sort).
    Example: ['file1', 'file10', 'file2'] -> ['file1', 'file2', 'file10']
    """

    def convert(text_part):
        return int(text_part) if text_part.isdigit() else text_part.lower()

    return [convert(c) for c in re.split(r"(\d+)", text)]


def read_jsonl_lines(path: str) -> Iterable[dict[str, Any]]:
    """
    Yield JSON objects from JSONL or prefixed log lines.

    Supports JSON and Python dictionary syntax.
    """
    with open(path, encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                # First try parsing the entire line as JSON (for normal JSONL files)
                yield json.loads(raw)
                continue
            except Exception:
                pass

            # If that fails, try to extract dict/JSON from log lines
            # Find the first occurrence of '{' or '[' which likely starts the object
            json_start = -1
            for char in ["{", "["]:
                pos = raw.find(char)
                if pos != -1 and (json_start == -1 or pos < json_start):
                    json_start = pos

            if json_start != -1:
                json_part = raw[json_start:]
                json_part = json_part.replace("np.float64(", "").replace(")", "")
                # Try JSON parsing first (standard format with double quotes)
                try:
                    yield json.loads(json_part)
                    continue
                except Exception:
                    pass

                # If JSON parsing fails, try Python literal eval (handles single quotes)
                try:
                    result = ast.literal_eval(json_part)
                    if isinstance(result, dict):
                        yield result
                        continue
                except Exception:
                    pass

                # If both fail, report error
                print(f"[warn] skip line {i}: could not extract/parse json/dict")
                continue
            else:
                print(f'[warn] skip line {i}: no dict/json object found (no "{{" or "[")')
                continue


def collect_by_processor(
    path: str,
    processor_name: str,
    file_label: str = None,
    use_filename_as_label: bool = True,
) -> dict[str, list[tuple[Any, float]]]:
    """
    Iterate file, call processor for each JSON object.
    Return: dict[label] -> list of (x_value, y_value) preserving input order.

    Args:
        path: Path to the jsonl file
        processor_name: Processor name
        file_label: If provided and use_filename_as_label=False, use this as the complete label.
                    If provided and use_filename_as_label=True, prefix labels with this.
        use_filename_as_label: Whether to prefix labels or replace them completely
    """
    series: dict[str, list[tuple[Any, float]]] = {}
    processor = build_processor(processor_name)
    for obj in read_jsonl_lines(path):
        try:
            keep, values, x = processor(obj)
        except Exception as e:
            print(f"[warn]: Processor raised an error. Skipping line: {e}.")
            raise e
            continue

        if not keep:
            continue
        if not isinstance(values, dict):
            print("[warn]: Processor returned non-dict values. Skipping line.")
            continue

        for label, raw_v in values.items():
            try:
                v = float(raw_v)
                if math.isfinite(v):
                    # Choose label strategy
                    if file_label and not use_filename_as_label:
                        # Use file_label as complete label (custom labels mode)
                        full_label = file_label
                    elif file_label and use_filename_as_label:
                        # Prefix label with file_label (default multi-file mode)
                        full_label = f"{file_label}::{label}"
                    else:
                        # No file_label, use original label
                        full_label = label
                    series.setdefault(full_label, []).append((x, v))
            except Exception:
                # skip non-numeric label/value for this row
                pass

    return series


def find_matching_jsonl_files(directory: str, substring: str, extensions: list[str] | None = None) -> list[str]:
    """
    Find all files in directory whose filename contains substring and has matching extension.
    Returns sorted list of file paths (natural sort).

    Args:
        directory: Directory to search
        substring: Substring to match in filenames
        extensions: List of file extensions to match (e.g., ['.jsonl', '.log']).
                   If None, defaults to ['.jsonl', '.log'].
                   If empty list [], matches all files regardless of extension.
    """
    if extensions is None:
        extensions = [".jsonl", ".log"]

    all_files = []
    if extensions:
        for ext in extensions:
            pattern = os.path.join(directory, f"*{ext}")
            all_files.extend(glob.glob(pattern))
    else:
        # If extensions is empty list, match all files
        pattern = os.path.join(directory, "*")
        all_files = glob.glob(pattern)
        # Filter out directories
        all_files = [f for f in all_files if os.path.isfile(f)]

    matching = [f for f in all_files if substring in os.path.basename(f)]
    return sorted(matching, key=lambda f: natural_sort_key(os.path.basename(f)))


def collect_multiple_files(
    directory: str,
    substring: str,
    processor_name: str,
    custom_labels: list[str] | None = None,
    extensions: list[str] | None = None,
    return_separate: bool = False,
) -> dict[str, list[tuple[Any, float]]] | list[dict[str, list[tuple[Any, float]]]]:
    """
    Find matching files, process each with the processor, and merge results.

    Args:
        directory: Directory to search
        substring: Substring to match in filenames
        processor_name: Processor name
        custom_labels: Optional list of custom labels for files (must match number of files)
                      If provided, these labels completely replace the original metric labels.
        extensions: Optional list of file extensions to match (e.g., ['.jsonl', '.log']).
                   Defaults to ['.jsonl', '.log'] if None.
        return_separate: If True, return a list of series dicts (one per file).
                        If False, return a single merged series dict.

    Returns:
        If return_separate=False: Merged series dict with all data
        If return_separate=True: List of series dicts, one per file

    Raises:
        AssertionError: If custom_labels is provided but doesn't match file count
    """
    files = find_matching_jsonl_files(directory, substring, extensions)
    if not files:
        ext_desc = "matching files" if extensions else "files"
        print(f"No {ext_desc} found in {directory} matching substring '{substring}'")
        return [] if return_separate else {}

    print(f"Found {len(files)} matching files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # Assert custom labels match file count if provided
    if custom_labels is not None:
        assert len(custom_labels) == len(files), (
            f"custom_labels count ({len(custom_labels)}) must match files count ({len(files)})"
        )

    all_series: dict[str, list[tuple[Any, float]]] = {}
    separate_series_list: list[dict[str, list[tuple[Any, float]]]] = []

    for i, filepath in enumerate(files):
        # Use custom label if available, otherwise use basename without extension
        if custom_labels:
            file_label = custom_labels[i]
            # Use custom label as complete label (not prefix)
            use_filename_as_label = False
        else:
            file_label = os.path.splitext(os.path.basename(filepath))[0]
            # Prefix original labels with filename
            use_filename_as_label = True

        print(f"Processing {filepath}...")
        series = collect_by_processor(filepath, processor_name, file_label, use_filename_as_label)

        if return_separate:
            separate_series_list.append(series)
        else:
            # Merge into all_series
            for label, points in series.items():
                all_series.setdefault(label, []).extend(points)

    return separate_series_list if return_separate else all_series


def plot_series(
    series: dict[str, list[tuple[Any, float]]],
    title: str = "Plot",
    xlabel: str = "x",
    ylabel: str = "value",
    out_png: str = None,
    show: bool = True,
):
    """
    Plot each label's points in the order they were collected (no sorting).
    If x-values are datetime objects, format the x-axis accordingly.
    """
    plt.figure(figsize=(10, 6))
    any_datetime = False

    for label, points in series.items():
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if isinstance(xs[0], datetime):
            any_datetime = True
        plt.scatter(xs, ys, marker=".", label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc="best")

    if any_datetime:
        try:
            plt.gcf().autofmt_xdate()
        except Exception:
            pass

    if out_png:
        plt.savefig(out_png, bbox_inches="tight")
        print(f"Saved plot to {out_png}")
    if show:
        plt.show()
    plt.close()


def plot_series_subplots(
    series_list: list[dict[str, list[tuple[Any, float]]]],
    title: str = "Plot",
    xlabel: str = "x",
    ylabel: str = "value",
    out_png: str = None,
    show: bool = True,
):
    """
    Plot each series dict in a separate subplot, arranged vertically in one column.
    Each subplot shows all labels from the corresponding series dict.

    Args:
        series_list: List of series dicts, one per file/subplot
        title: Overall plot title
        xlabel: X axis label (applied to all subplots)
        ylabel: Y axis label (applied to all subplots)
        out_png: Output PNG path
        show: Whether to display the plot
    """
    num_subplots = len(series_list)
    if num_subplots == 0:
        print("No data to plot")
        return

    # Create subplots: num_subplots rows, 1 column
    fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 4 * num_subplots), sharex=True)

    # Handle single subplot case (axes is not a list)
    if num_subplots == 1:
        axes = [axes]

    any_datetime = False

    for idx, series in enumerate(series_list):
        ax = axes[idx]

        for label, points in series.items():
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            if xs and isinstance(xs[0], datetime):
                any_datetime = True
            ax.scatter(xs, ys, marker=".", label=label)

        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend(loc="best")
        # Set subplot title from first label or use index
        if series:
            first_label = next(iter(series.keys()))
            ax.set_title(first_label, fontsize=10)

    # Set common labels
    axes[-1].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=12)

    # Adjust spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle

    if any_datetime:
        try:
            fig.autofmt_xdate()
        except Exception:
            pass

    if out_png:
        plt.savefig(out_png, bbox_inches="tight")
        print(f"Saved plot to {out_png}")
    if show:
        plt.show()
    plt.close()


def sum_series_by_x(
    series_list: list[dict[str, list[tuple[Any, float]]]],
) -> dict[str, list[tuple[Any, float]]]:
    """
    Sum y-values across multiple series by aligning x-coordinates.
    Uses the first series' x-coordinates as the baseline.
    For each x in the baseline, finds the nearest x in other series and sums their y-values.

    Handles labels with "::" separator (filename::metric) by extracting the metric part
    and summing across all files for the same metric.

    Args:
        series_list: List of series dicts, one per file

    Returns:
        A single series dict with summed y-values, using the first series' x-coordinates
    """
    if not series_list:
        return {}

    # Get the first series as baseline
    baseline_series = series_list[0]
    if not baseline_series:
        return {}

    # Group labels by metric (extract part after "::" if present)
    # This allows summing across files even when labels are "filename::metric"
    metric_to_labels: dict[str, list[str]] = {}
    all_labels = set()
    for series in series_list:
        all_labels.update(series.keys())

    for label in all_labels:
        # Remove any filename prefix from the metric label.
        if "::" in label:
            metric = label.split("::", 1)[1]
        else:
            metric = label
        metric_to_labels.setdefault(metric, []).append(label)

    # For each metric, sum across all series that have labels matching this metric
    result: dict[str, list[tuple[Any, float]]] = {}

    for metric, matching_labels in metric_to_labels.items():
        # Get all series points for labels matching this metric
        metric_series_points = []
        for series in series_list:
            for label in matching_labels:
                if label in series and series[label]:
                    metric_series_points.append(series[label])
                    break  # Use first matching label from each series

        if not metric_series_points:
            continue

        # Use the first series' x-coordinates as baseline
        baseline_points = metric_series_points[0]
        if not baseline_points:
            continue

        summed_points = []

        for base_x, base_y in baseline_points:
            # Start with the baseline y-value
            total_y = base_y

            # For each subsequent series, find the nearest x and add its y-value
            for other_series_points in metric_series_points[1:]:
                if not other_series_points:
                    continue

                # Find the point with the nearest x-coordinate
                min_dist = None
                nearest_y = None

                for other_x, other_y in other_series_points:
                    # Calculate distance (handle both numeric and datetime)
                    if isinstance(base_x, datetime) and isinstance(other_x, datetime):
                        dist = abs((base_x - other_x).total_seconds())
                    elif isinstance(base_x, (int, float)) and isinstance(other_x, (int, float)):
                        dist = abs(base_x - other_x)
                    else:
                        # Skip if types don't match
                        continue

                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        nearest_y = other_y

                # Add the nearest y-value if found
                if nearest_y is not None:
                    total_y += nearest_y

            summed_points.append((base_x, total_y))

        if summed_points:
            # Use metric as the label for the result
            result[metric] = summed_points

    return result


# --- Example processor ---


def _get_by_dotted(d: dict[str, Any], path: str, default=None):
    """Get nested value by dotted path (e.g. 'throughput_stats.total_throughput')."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def processor_example(obj: dict[str, Any]) -> tuple[bool, dict[str, float], Any]:
    """
    Extract throughput and KV cache usage with elapsed time as the x value.
    """
    # choose x
    x = obj.get("elapsed_time")
    if x is None:
        ts = obj.get("timestamp")
        if isinstance(ts, str):
            try:
                x = datetime.fromisoformat(ts)
            except Exception:
                x = ts  # keep raw string if can't parse
        else:
            x = None

    values: dict[str, float] = {}
    ttp = _get_by_dotted(obj, "throughput_stats.total_throughput", None)
    if ttp is not None:
        try:
            values["total_throughput"] = float(ttp)
        except Exception:
            pass
    kv = _get_by_dotted(obj, "scheduler_stats.kv_cache_usage", None)
    if kv is not None:
        try:
            values["kv_cache_usage"] = float(kv)
        except Exception:
            pass

    # keep only if we have at least one numeric value
    keep = len(values) > 0
    return keep, values, x


# --- CLI ---
def main():
    ap = argparse.ArgumentParser(description="Simple jsonl -> multi-line plot using a user processor.")

    # Input: either a single file OR directory + substring
    ap.add_argument("jsonl", help=".jsonl input path OR directory path")
    ap.add_argument(
        "--substring",
        help="substring to match in filenames (for directory mode)",
        default=None,
    )
    ap.add_argument(
        "--custom-labels",
        help="comma-separated custom labels for files (must match file count)",
        default=None,
    )

    ap.add_argument("--out", help="output PNG path", default="output.png")
    ap.add_argument("--title", help="plot title", default="Plot")
    ap.add_argument("--xlabel", help="x axis label", default="x")
    ap.add_argument("--ylabel", help="y axis label", default="value")
    ap.add_argument(
        "--mode",
        choices=["merge", "subplot", "sum"],
        default="merge",
        help="Multi-file plotting mode: 'merge' (all files in one figure, default), "
        "'subplot' (each file in separate subplot), "
        "'sum' (sum y-values by aligning x-coordinates using first file as baseline)",
    )
    ap.add_argument("--processor", help="processor name", default="intertoken_indexed")
    args = ap.parse_args()

    # Replace `processor_example` with your own `processor` function if desired.
    # processor = processor_example

    if os.path.isfile(args.jsonl):
        # Single file mode
        print(f"Processing single file: {args.jsonl}")
        series = collect_by_processor(args.jsonl, args.processor)
        if not series:
            print("No data collected with current processor.")
            return
        plot_series(
            series,
            title=args.title,
            xlabel=args.xlabel,
            ylabel=args.ylabel,
            out_png=args.out,
            show=True,
        )
    elif os.path.isdir(args.jsonl):
        # Directory mode
        if args.substring is None:
            print("Error: --substring is required when providing a directory")
            return
        print(f"Processing directory: {args.jsonl} (matching '{args.substring}')")

        # Parse custom labels if provided
        custom_labels = None
        if args.custom_labels:
            custom_labels = [label.strip() for label in args.custom_labels.split(",")]

        # Determine mode and collect data accordingly
        if args.mode == "subplot" or args.mode == "sum":
            # Need separate series for subplot or sum mode
            result = collect_multiple_files(
                args.jsonl,
                args.substring,
                args.processor,
                custom_labels,
                return_separate=True,
            )
            series_list = result
            if not series_list or all(not s for s in series_list):
                print("No data collected with current processor.")
                return

            if args.mode == "subplot":
                plot_series_subplots(
                    series_list,
                    title=args.title,
                    xlabel=args.xlabel,
                    ylabel=args.ylabel,
                    out_png=args.out,
                    show=True,
                )
            elif args.mode == "sum":
                # Sum y-values by aligning x-coordinates
                summed_series = sum_series_by_x(series_list)
                if not summed_series:
                    print("No data after summing.")
                    return
                plot_series(
                    summed_series,
                    title=args.title,
                    xlabel=args.xlabel,
                    ylabel=args.ylabel,
                    out_png=args.out,
                    show=True,
                )
        else:
            # merge mode (default): merge all files into one figure
            result = collect_multiple_files(
                args.jsonl,
                args.substring,
                args.processor,
                custom_labels,
                return_separate=False,
            )
            series = result
            if not series:
                print("No data collected with current processor.")
                return
            plot_series(
                series,
                title=args.title,
                xlabel=args.xlabel,
                ylabel=args.ylabel,
                out_png=args.out,
                show=True,
            )
    else:
        print("Error: path does not exist:", args.jsonl)
        return


if __name__ == "__main__":
    main()
