import argparse
import ast

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def parse_log_file(file_path):
    """
    Parse log file and extract request num, token num and actual generation throughput
    """
    data = []

    with open(file_path, encoding="utf-8") as file:
        for line in file:
            # Find lines containing "Snapshot"
            if "Snapshot" not in line:
                continue

            try:
                # Extract JSON portion
                json_start = line.find("{")
                json_end = line.rfind("}") + 1
                if json_start == -1 or json_end == 0:
                    continue

                json_str = line[json_start:json_end]

                # Use ast.literal_eval to parse Python-style dictionaries with single quotes
                log_data = ast.literal_eval(json_str)

                # Extract required data
                iteration_stats = log_data.get("iteration_stats", {})
                scheduler_stats = log_data.get("scheduler_stats", {})

                # Skip if num_prompt_reqs is not zero
                num_prompt_reqs = iteration_stats.get("num_prompt_reqs", 0)
                if num_prompt_reqs != 0:
                    continue

                # request_num corresponds to num_generation_reqs
                request_num = iteration_stats.get("num_generation_reqs", 0)

                # token_num is the sum of all values in prompt_token_num and response_token_num dictionaries
                prompt_tokens_dict = scheduler_stats.get("req_id_to_prompt_token_num", {})
                response_tokens_dict = scheduler_stats.get("req_id_to_response_token_num", {})

                prompt_token_num = sum(prompt_tokens_dict.values())
                response_token_num = sum(response_tokens_dict.values())
                token_num = prompt_token_num + response_token_num

                # throughput is calculated as num_generation_reqs / avg_inter_token_latencies
                avg_inter_token_latencies = iteration_stats.get("avg_inter_token_latencies", 0)
                throughput = request_num / avg_inter_token_latencies if avg_inter_token_latencies > 0 else 0

                # Skip records where request_num is 0
                if request_num == 0:
                    continue

                data.append(
                    {
                        "request_num": request_num,
                        "token_num": token_num,
                        "throughput": throughput,
                    }
                )

            except (SyntaxError, KeyError, ValueError) as e:
                # Skip this line if parsing fails
                print(f"Error parsing line: {e}")
                continue

    return data


def calculate_latency(request_num, throughput):
    """
    Calculate latency from throughput
    latency = request_num / throughput
    """
    return request_num / throughput


def latency_model(request_num, k, b, threshold):
    """
    Latency model: latency = max(threshold, k * request_num + b)
    """
    linear_part = k * request_num + b
    return np.maximum(threshold, linear_part)


def attn_latency_model(token_num, k, b):
    """
    Attention latency model: latency = k * token_num + b
    """
    return k * token_num + b


def throughput_model(request_num, k, b, threshold):
    """
    Throughput model: throughput = request_num / latency
    where latency = max(threshold, k * request_num + b)
    """
    latency = latency_model(request_num, k, b, threshold)
    return request_num / latency


def fit_model(data):
    """
    Fit model parameters using non-linear least squares
    """
    # Prepare data arrays
    request_nums = np.array([d["request_num"] for d in data])
    actual_throughputs = np.array([d["throughput"] for d in data])
    _ = calculate_latency(request_nums, actual_throughputs)  # actual_latencies unused

    # Initial parameter guesses based on data observation
    initial_k = 0.0001  # Slope coefficient for request_num
    initial_b = 0.001  # Intercept for latency
    initial_threshold = 0.005  # Minimum latency threshold
    initial_params = [initial_k, initial_b, initial_threshold]

    # Define function for curve fitting
    def fit_function(x, k, b, threshold):
        return throughput_model(x, k, b, threshold)

    try:
        # Set parameter bounds to ensure physical meaning
        # All parameters should be non-negative
        lower_bounds = [0, 0, 0]  # Minimum values for [k, b, threshold]
        upper_bounds = [0.01, 0.1, 0.1]  # Maximum values for [k, b, threshold]

        # Perform curve fitting
        popt, pcov = curve_fit(
            fit_function,
            request_nums,
            actual_throughputs,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,  # Maximum function evaluations
        )

        # Extract optimized parameters
        k_opt, b_opt, threshold_opt = popt

        # Calculate R-squared value for goodness of fit
        predicted_throughputs = throughput_model(request_nums, k_opt, b_opt, threshold_opt)
        ss_res = np.sum((actual_throughputs - predicted_throughputs) ** 2)  # Residual sum of squares
        ss_tot = np.sum((actual_throughputs - np.mean(actual_throughputs)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot)  # Coefficient of determination

        return {
            "other_threshold": round(threshold_opt, 7),
            "other_latency_b": round(b_opt, 7),
            "other_latency_k": round(k_opt, 7),
            "r_squared": r_squared,
            "parameters": popt,
            "covariance": pcov,
        }

    except Exception as e:
        print(f"Error in fitting process: {e}")
        return None


def fit_attention_model(data, k, b, threshold):
    """
    Fit attention model parameters: attention_latencies = attn_k * token_nums + attn_b
    """
    # Prepare data arrays
    request_nums = np.array([d["request_num"] for d in data])
    token_nums = np.array([d["token_num"] for d in data])
    actual_throughputs = np.array([d["throughput"] for d in data])
    actual_latencies = calculate_latency(request_nums, actual_throughputs)

    # Calculate attention latencies by subtracting other latency components
    other_latencies = latency_model(request_nums, k, b, threshold)
    attention_latencies = actual_latencies - other_latencies

    # Filter out negative or invalid attention latencies
    valid_mask = attention_latencies > 0
    if not np.any(valid_mask):
        print("Warning: No valid attention latencies found after subtracting other latencies")
        return None

    valid_token_nums = token_nums[valid_mask]
    valid_attention_latencies = attention_latencies[valid_mask]

    # Initial parameter guesses
    initial_attn_k = np.mean(valid_attention_latencies / valid_token_nums) if np.any(valid_token_nums > 0) else 0.0001
    initial_attn_b = np.mean(valid_attention_latencies) * 0.1

    initial_params = [initial_attn_k, initial_attn_b]

    # Define function for curve fitting
    def fit_function(x, attn_k, attn_b):
        return attn_latency_model(x, attn_k, attn_b)

    try:
        # Set parameter bounds to ensure physical meaning
        # All parameters should be non-negative
        lower_bounds = [1e-10, 1e-10]  # Minimum values for [attn_k, attn_b]
        upper_bounds = [0.1, 0.1]  # Maximum values for [attn_k, attn_b]

        # First fitting pass to identify outliers
        popt_initial, _ = curve_fit(
            fit_function,
            valid_token_nums,
            valid_attention_latencies,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,
        )

        # Calculate residuals from initial fit
        predicted_initial = attn_latency_model(valid_token_nums, popt_initial[0], popt_initial[1])
        residuals = np.abs(valid_attention_latencies - predicted_initial)

        # Use IQR method to filter outliers
        q1 = np.percentile(residuals, 25)
        q3 = np.percentile(residuals, 75)
        iqr = q3 - q1
        # Use a stricter threshold (2.5 * IQR) to remove more outliers
        outlier_threshold = q3 + 2.5 * iqr

        # Filter out outliers
        inlier_mask = residuals <= outlier_threshold
        filtered_token_nums = valid_token_nums[inlier_mask]
        filtered_attention_latencies = valid_attention_latencies[inlier_mask]

        if len(filtered_token_nums) < 10:  # Need at least some points for fitting
            print(
                f"Warning: Too few points after outlier removal ({len(filtered_token_nums)}), using all valid points"
            )
            filtered_token_nums = valid_token_nums
            filtered_attention_latencies = valid_attention_latencies
            final_mask = valid_mask
        else:
            # Update the valid_mask to include outlier filtering
            final_valid_mask = np.zeros_like(valid_mask, dtype=bool)
            valid_indices = np.where(valid_mask)[0]
            final_valid_mask[valid_indices[inlier_mask]] = True
            final_mask = final_valid_mask
            print(
                f"Outlier removal: {len(valid_token_nums)} -> {len(filtered_token_nums)} points "
                f"({len(filtered_token_nums) / len(valid_token_nums) * 100:.1f}% retained)"
            )

        # Refit with filtered data
        popt, pcov = curve_fit(
            fit_function,
            filtered_token_nums,
            filtered_attention_latencies,
            p0=popt_initial,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000,  # Maximum function evaluations
        )

        # Extract optimized parameters
        attn_k_opt, attn_b_opt = popt

        # Calculate R-squared value for goodness of fit using filtered data
        predicted_attention_latencies = attn_latency_model(filtered_token_nums, attn_k_opt, attn_b_opt)
        ss_res = np.sum((filtered_attention_latencies - predicted_attention_latencies) ** 2)  # Residual sum of squares
        ss_tot = np.sum(
            (filtered_attention_latencies - np.mean(filtered_attention_latencies)) ** 2
        )  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0  # Coefficient of determination

        return {
            "attn_latency_k": round(attn_k_opt, 10),
            "attn_latency_b": round(attn_b_opt, 10),
            "r_squared": r_squared,
            "parameters": popt,
            "covariance": pcov,
            "valid_mask": final_mask,
        }

    except Exception as e:
        print(f"Error in attention fitting process: {e}")
        return None


def plot_results(data, fit_result):
    """
    Plot fitting results for visualization
    """
    if not fit_result:
        return

    request_nums = np.array([d["request_num"] for d in data])
    actual_throughputs = np.array([d["throughput"] for d in data])

    k, b, threshold = fit_result["parameters"]

    # Generate smooth curve for plotting
    x_smooth = np.linspace(min(request_nums), max(request_nums), 100)
    y_pred_smooth = throughput_model(x_smooth, k, b, threshold)

    plt.figure(figsize=(10, 6))

    # Plot original data points
    plt.scatter(request_nums, actual_throughputs, alpha=0.7, label="Actual Data", color="blue")

    # Plot fitted curve
    plt.plot(
        x_smooth,
        y_pred_smooth,
        "r-",
        label=f"Fitted Curve (R² = {fit_result['r_squared']:.4f})",
    )

    plt.xlabel("Request Number")
    plt.ylabel("Throughput")
    plt.title("Throughput Model Fitting Results")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("throughput_model_fitting_results.png")


def plot_attn_results(data, attn_fit_result, k, b, threshold):
    """
    Plot the attention latency fit after removing invalid and outlier points.
    """
    if not attn_fit_result:
        return

    request_nums = np.array([d["request_num"] for d in data])
    token_nums = np.array([d["token_num"] for d in data])
    actual_throughputs = np.array([d["throughput"] for d in data])

    attn_k, attn_b = attn_fit_result["parameters"]
    valid_mask = attn_fit_result["valid_mask"]

    # Calculate actual latencies
    actual_latencies = calculate_latency(request_nums, actual_throughputs)

    # Calculate other latencies using latency_model
    other_latencies = latency_model(request_nums, k, b, threshold)

    # Subtract modeled nonattention latency from actual latency.
    attention_latencies = actual_latencies - other_latencies

    # Filter out negative values first (basic validation)
    basic_valid_mask = attention_latencies > 0
    # Then apply the outlier-filtered mask
    valid_mask = valid_mask & basic_valid_mask

    # Use only valid data points (after outlier filtering)
    valid_token_nums = token_nums[valid_mask]
    valid_attention_latencies = attention_latencies[valid_mask]

    if len(valid_token_nums) == 0:
        print("Warning: No valid data points to plot after filtering")
        return

    # Generate smooth curve for plotting
    token_nums_smooth = np.linspace(min(valid_token_nums), max(valid_token_nums), 100)
    attn_latencies_smooth = attn_latency_model(token_nums_smooth, attn_k, attn_b)

    plt.figure(figsize=(10, 6))

    # Plot only filtered data points (outliers already removed)
    plt.scatter(
        valid_token_nums,
        valid_attention_latencies,
        alpha=0.7,
        label="Filtered Attention Latency",
        color="blue",
        s=20,
    )

    # Plot fitted attention latency curve
    plt.plot(
        token_nums_smooth,
        attn_latencies_smooth,
        "r-",
        label=f"Fitted Curve (R² = {attn_fit_result['r_squared']:.4f})",
        linewidth=2,
    )

    plt.xlabel("Token Number")
    plt.ylabel("Attention Latency")
    plt.title("Attention Latency Model Fitting Results (Outliers Removed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("attention_latency_fitting_results.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit cost model from log file")
    parser.add_argument("file_path", type=str, help="Path to the log file")
    parser.add_argument(
        "--fit-attn-mode",
        action="store_true",
        help="Fit attention model (requires --other-k, --other-b, --other-threshold)",
    )
    parser.add_argument(
        "--other-k",
        type=float,
        default=None,
        help="k parameter for other model (required if --fit-attn-mode is set)",
    )
    parser.add_argument(
        "--other-b",
        type=float,
        default=None,
        help="b parameter for other model (required if --fit-attn-mode is set)",
    )
    parser.add_argument(
        "--other-threshold",
        type=float,
        default=None,
        help="threshold parameter for other model (required if --fit-attn-mode is set)",
    )

    args = parser.parse_args()

    try:
        print("Parsing log file...")
        data = parse_log_file(args.file_path)

        if not data:
            print("No valid data found. Please check file path and format.")
            exit(1)

        print(f"Successfully extracted {len(data)} data records")
        print("\nFirst 5 data points:")
        for i, d in enumerate(data[:5]):
            print(
                f"  {i + 1}. Request: {d['request_num']}, Token: {d['token_num']}, Throughput: {d['throughput']:.2f}"
            )

        if args.fit_attn_mode:
            # Fit attention model mode
            if args.other_k is None or args.other_b is None or args.other_threshold is None:
                parser.error("--fit-attn-mode requires --other-k, --other-b, and --other-threshold")

            print("\nFitting attention model...")
            print(f"Using provided parameters: k={args.other_k}, b={args.other_b}, threshold={args.other_threshold}")

            # Fit attention model
            attn_fit_result = fit_attention_model(data, args.other_k, args.other_b, args.other_threshold)

            if attn_fit_result:
                print("\nAttention Model Fitting Results:")
                print(f"  attn_latency_k: {attn_fit_result['attn_latency_k']}")
                print(f"  attn_latency_b: {attn_fit_result['attn_latency_b']}")
                print(f"\nGoodness of Fit (R²): {attn_fit_result['r_squared']:.6f}")

                # Calculate additional accuracy metrics for attention model
                request_nums = np.array([d["request_num"] for d in data])
                token_nums = np.array([d["token_num"] for d in data])
                actual_throughputs = np.array([d["throughput"] for d in data])

                attn_k_opt, attn_b_opt = attn_fit_result["parameters"]
                valid_mask = attn_fit_result["valid_mask"]

                # Calculate predicted attention latencies
                valid_token_nums = token_nums[valid_mask]
                valid_attention_latencies_pred = attn_latency_model(valid_token_nums, attn_k_opt, attn_b_opt)

                # Calculate actual attention latencies for valid data
                actual_latencies = calculate_latency(request_nums, actual_throughputs)
                other_latencies = latency_model(request_nums, args.other_k, args.other_b, args.other_threshold)
                actual_attention_latencies = actual_latencies - other_latencies
                valid_attention_latencies_actual = actual_attention_latencies[valid_mask]

                # Calculate mean relative error
                relative_errors = np.abs(
                    (valid_attention_latencies_actual - valid_attention_latencies_pred)
                    / valid_attention_latencies_actual
                )
                mean_relative_error = np.mean(relative_errors) * 100
                print(f"Mean Relative Error: {mean_relative_error:.2f}%")

                # Calculate root mean square error
                rmse = np.sqrt(np.mean((valid_attention_latencies_actual - valid_attention_latencies_pred) ** 2))
                print(f"Root Mean Square Error: {rmse:.6f}")

                # Generate final output in required format
                final_result = {
                    "attn_latency_k": attn_fit_result["attn_latency_k"],
                    "attn_latency_b": attn_fit_result["attn_latency_b"],
                }

                print("\nFinal Parameters:")
                print(final_result)

                # Plot results for visual inspection
                plot_attn_results(
                    data,
                    attn_fit_result,
                    args.other_k,
                    args.other_b,
                    args.other_threshold,
                )
            else:
                print("Attention model fitting failed.")
        else:
            # Original fitting mode
            print("\nFitting model...")
            fit_result = fit_model(data)

            if fit_result:
                print("\nFitting Results:")
                print(f"  other_threshold: {fit_result['other_threshold']}")
                print(f"  other_latency_b: {fit_result['other_latency_b']}")
                print(f"  other_latency_k: {fit_result['other_latency_k']}")
                print(f"\nGoodness of Fit (R²): {fit_result['r_squared']:.6f}")

                # Calculate additional accuracy metrics
                request_nums = np.array([d["request_num"] for d in data])
                actual_throughputs = np.array([d["throughput"] for d in data])
                k, b, threshold = fit_result["parameters"]
                predicted_throughputs = throughput_model(request_nums, k, b, threshold)

                # Calculate mean relative error
                relative_errors = np.abs((actual_throughputs - predicted_throughputs) / actual_throughputs)
                mean_relative_error = np.mean(relative_errors) * 100
                print(f"Mean Relative Error: {mean_relative_error:.2f}%")

                # Calculate root mean square error
                rmse = np.sqrt(np.mean((actual_throughputs - predicted_throughputs) ** 2))
                print(f"Root Mean Square Error: {rmse:.2f}")

                # Generate final output in required format
                final_result = {
                    "other_threshold": fit_result["other_threshold"],
                    "other_latency_b": fit_result["other_latency_b"],
                    "other_latency_k": fit_result["other_latency_k"],
                }

                print("\nFinal Parameters:")
                print(final_result)

                # Plot results for visual inspection
                plot_results(data, fit_result)
            else:
                print("Model fitting failed.")
    except FileNotFoundError:
        print(f"File not found: {args.file_path}")
        exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        exit(1)
