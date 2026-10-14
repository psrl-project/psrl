import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Set style for better-looking plots
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
# mpl.rcParams['text.usetex'] = True
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["font.size"] = 20


def parse_log_file(file_path: str, target_key: str) -> list[float]:
    """
    Parse log file and extract values for specified key

    Args:
        file_path: Path to log file
        target_key: Metric key to extract

    Returns:
        List of extracted values
    """
    values = []

    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                # Match patterns: key:value or key=value
                pattern1 = rf"{re.escape(target_key)}:([-\d.]+)"
                pattern2 = rf"{re.escape(target_key)}=([-\d.]+)"

                match1 = re.search(pattern1, line)
                match2 = re.search(pattern2, line)

                if match1:
                    try:
                        value = float(match1.group(1))
                        values.append(value)
                    except ValueError:
                        continue
                elif match2:
                    try:
                        value = float(match2.group(1))
                        values.append(value)
                    except ValueError:
                        continue

    except FileNotFoundError:
        print(f"Warning: File {file_path} not found")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")

    return values


def plot_metrics_from_files(
    file_paths: list[str],
    target_key: str,
    labels: list[str] | None = None,
    max_points: int | None = None,
    figsize: tuple = (9, 9),
    title: str | None = None,
    xlabel: str = "Step",
    ylabel: str | None = None,
    save_path: str | None = None,
):
    """
    Extract metrics from multiple log files and plot curves

    Args:
        file_paths: List of log file paths
        target_key: Metric key to extract
        labels: Optional list of labels for each file path (must match length of file_paths)
        max_points: Optional maximum number of data points to plot (takes first N points)
        figsize: Figure size
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the plot
    """
    # Validate labels
    if labels is not None:
        if len(labels) != len(file_paths):
            raise ValueError(f"Length of labels ({len(labels)}) must match length of file_paths ({len(file_paths)})")
    else:
        # Use filename as default label
        labels = [os.path.basename(fp) for fp in file_paths]

    # bright color palette
    color_palette = [
        "#FF6B6B",  # Bright Coral Red
        "#4ECDC4",  # Bright Turquoise
        "#45B7D1",  # Bright Sky Blue
        "#FFA07A",  # Light Salmon
        "#98D8C8",  # Mint Green
        "#F7DC6F",  # Bright Yellow
        "#BB8FCE",  # Soft Purple
        "#85C1E2",  # Light Blue
        "#F8B739",  # Golden Yellow
        "#52BE80",  # Emerald Green
        "#EC7063",  # Soft Red
        "#5DADE2",  # Bright Blue
    ]
    color_palette = color_palette[::-1]

    # Line styles for variety
    line_styles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=figsize)

    all_data = {}

    for idx, file_path in enumerate(file_paths):
        label = labels[idx]
        values = parse_log_file(file_path, target_key)

        if values:
            # Limit data points if max_points is specified
            if max_points is not None and max_points > 0:
                original_count = len(values)
                values = values[:max_points]
                if len(values) < original_count:
                    print(f"File {label}: Truncated from {original_count} to {len(values)} data points")

            all_data[label] = values
            # Plot curve with color and style cycling
            x_values = range(len(values))
            color = color_palette[idx % len(color_palette)]
            linestyle = line_styles[(idx // len(color_palette)) % len(line_styles)]

            ax.plot(
                x_values,
                values,
                marker="o",
                markersize=4,
                linewidth=3,
                label=label,
                color=color,
                linestyle=linestyle,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                alpha=0.9,
            )
            print(f"File {label}: Found {len(values)} data points")
        else:
            print(f"File {label}: No data found for metric '{target_key}'")

    if not all_data:
        print("No data found, cannot create plot")
        return

    # Configure plot appearance with modern styling
    if title is None:
        title = f"Metric Trend: {target_key}"
    # Set title to overlay on top border
    ax.set_title(title, fontsize=20, fontweight="bold", pad=5, y=1.02)

    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20, fontweight="medium")
    ax.set_xlabel(xlabel, fontsize=20, fontweight="medium")

    # horizontal legend
    ax.legend(
        bbox_to_anchor=(0.5, -0.2),
        loc="upper center",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=20,
        framealpha=0.95,
        ncol=len(all_data),
    )

    # horizontal grid at y ticks
    ax.grid(True, axis="y", alpha=0.4, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)

    # Increase tick label font size
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=18)

    # visible plot borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        # spine.set_color('#000000')
        spine.set_linewidth(3)  # Make borders slightly thicker

    # Adjust layout to accommodate horizontal legend at bottom
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Save plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"Plot saved to: {save_path}")

    plt.show()

    # Print statistics
    print("\nStatistics:")
    for label, values in all_data.items():
        if values:
            print(
                f"{label}: Points={len(values)}, Mean={np.mean(values):.4f}, "
                f"Std={np.std(values):.4f}, Range=[{min(values):.4f}, {max(values):.4f}]"
            )


# Example usage
if __name__ == "__main__":
    # Define your file paths
    file_paths = [
        "${PSRL_WORKSPACE}/psrl/examples/paper_exp/background/staleness/7b_staleness_1.log",
        "${PSRL_WORKSPACE}/psrl/examples/paper_exp/background/staleness/7b_staleness_2.log",
        "${PSRL_WORKSPACE}/psrl/examples/paper_exp/background/staleness/7b_staleness_3.log",
    ]

    # Optional: Define custom labels for each file path
    labels = ["ŋ = 1", "ŋ = 2", "ŋ = 3"]

    # Specify the metric you want to extract
    target_key = "timing_s/wait_for_gen"

    # Create the plot
    plot_metrics_from_files(
        file_paths=file_paths,
        target_key=target_key,
        labels=labels,  # Optional: if not provided, uses filenames
        max_points=200,  # Optional: limit to first 100 data points
        title="Training Phase Waiting for Batch Time (s)",
        # ylabel="Time (s)",
        save_path="wait_for_rollout.svg",
    )

    # Define your file paths
    file_paths = [
        "${PSRL_WORKSPACE}/psrl/examples/paper_exp/background/staleness/7b_staleness_1.log",
        "${PSRL_WORKSPACE}/psrl/examples/paper_exp/background/staleness/7b_staleness_3.log",
        "${PSRL_WORKSPACE}/psrl/examples/paper_exp/background/staleness/7b_staleness_2.log",
    ]

    # Optional: Define custom labels for each file path
    labels = ["ŋ = 1", "ŋ = 2", "ŋ = 3"]

    # Specify the metric you want to extract
    target_key = "critic/rewards/mean"

    # Create the plot
    plot_metrics_from_files(
        file_paths=file_paths,
        target_key=target_key,
        labels=labels,  # Optional: if not provided, uses filenames
        max_points=200,  # Optional: limit to first 100 data points
        title="Reward",
        # ylabel="Time (s)",
        save_path="reward.svg",
    )
