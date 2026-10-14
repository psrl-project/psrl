import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

FILE_RE = re.compile(
    r"^(?P<model>.+)_Syn_TP(?P<tp>\d+)_PP(?P<pp>\d+)(?:_EP(?P<ep>\d+))?_B(?P<batch>\d+)_P(?P<prompt>\d+)_R(?P<response>\d+)_disable_attn_(?P<disable>true|false)\.jsonl$"
)


def parse_file_name(path: Path) -> dict | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    info["tp"] = int(info["tp"])
    info["pp"] = int(info["pp"])
    info["ep"] = int(info["ep"]) if info.get("ep") is not None else 1
    info["batch"] = int(info["batch"])
    info["prompt"] = int(info["prompt"])
    info["response"] = int(info["response"])
    info["disable"] = info["disable"] == "true"
    return info


def latency_model(request_num, other_k, other_b, other_threshold):
    linear_part = other_k * request_num + other_b
    return np.maximum(other_threshold, linear_part)


def attn_latency_model(token_num, attn_k, attn_b):
    return attn_k * token_num + attn_b


def load_decode_points(path: Path, prompt_len: int, expected_batch: int) -> list[dict]:
    decode_points = []
    decode_step_idx = 0
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
            inter_token_latency = float(iteration_stats.get("inter_token_latencies_avg", 0.0))
            if running_reqs <= 0 or inter_token_latency <= 0.0:
                continue

            decode_step_idx += 1
            if running_reqs != expected_batch:
                # Synthetic benchmark keeps all requests alive until the very end.
                # Ignore any malformed points that do not match the target batch size.
                continue

            token_num = running_reqs * (prompt_len + decode_step_idx)
            decode_points.append(
                {
                    "request_num": running_reqs,
                    "token_num": token_num,
                    "latency": inter_token_latency,
                }
            )
    return decode_points


def fit_other_params(points: list[dict]) -> tuple[dict, np.ndarray]:
    request_nums = np.array([p["request_num"] for p in points], dtype=float)
    latencies = np.array([p["latency"] for p in points], dtype=float)

    initial_threshold = float(np.min(latencies))
    initial_b = float(np.min(latencies))
    initial_k = max(
        float((np.max(latencies) - np.min(latencies)) / max(np.max(request_nums) - np.min(request_nums), 1.0)), 1e-8
    )

    popt, _ = curve_fit(
        latency_model,
        request_nums,
        latencies,
        p0=[initial_k, initial_b, initial_threshold],
        bounds=([0.0, 0.0, 0.0], [0.1, 0.1, 0.1]),
        maxfev=10000,
    )

    predicted = latency_model(request_nums, *popt)
    ss_res = float(np.sum((latencies - predicted) ** 2))
    ss_tot = float(np.sum((latencies - np.mean(latencies)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    result = {
        "other_threshold": round(float(popt[2]), 7),
        "other_latency_b": round(float(popt[1]), 7),
        "other_latency_k": round(float(popt[0]), 7),
        "other_r_squared": r_squared,
    }
    return result, predicted


def fit_attn_params(points: list[dict], other_params: dict) -> dict:
    request_nums = np.array([p["request_num"] for p in points], dtype=float)
    token_nums = np.array([p["token_num"] for p in points], dtype=float)
    latencies = np.array([p["latency"] for p in points], dtype=float)

    other_latencies = latency_model(
        request_nums,
        other_params["other_latency_k"],
        other_params["other_latency_b"],
        other_params["other_threshold"],
    )
    attention_latencies = latencies - other_latencies
    valid_mask = np.isfinite(attention_latencies) & (attention_latencies > 0)
    valid_token_nums = token_nums[valid_mask]
    valid_attention_latencies = attention_latencies[valid_mask]

    if len(valid_token_nums) == 0:
        raise ValueError("No valid attention points after subtracting other latency")

    initial_attn_k = max(float(np.mean(valid_attention_latencies / valid_token_nums)), 1e-10)
    initial_attn_b = max(float(np.percentile(valid_attention_latencies, 10)), 1e-10)

    popt_initial, _ = curve_fit(
        attn_latency_model,
        valid_token_nums,
        valid_attention_latencies,
        p0=[initial_attn_k, initial_attn_b],
        bounds=([1e-12, 1e-12], [0.1, 0.1]),
        maxfev=10000,
    )

    predicted_initial = attn_latency_model(valid_token_nums, *popt_initial)
    residuals = np.abs(valid_attention_latencies - predicted_initial)
    q1 = float(np.percentile(residuals, 25))
    q3 = float(np.percentile(residuals, 75))
    iqr = q3 - q1
    threshold = q3 + 2.5 * iqr
    inlier_mask = residuals <= threshold

    filtered_token_nums = valid_token_nums[inlier_mask]
    filtered_attention_latencies = valid_attention_latencies[inlier_mask]
    if len(filtered_token_nums) < 10:
        filtered_token_nums = valid_token_nums
        filtered_attention_latencies = valid_attention_latencies

    popt, _ = curve_fit(
        attn_latency_model,
        filtered_token_nums,
        filtered_attention_latencies,
        p0=popt_initial,
        bounds=([1e-12, 1e-12], [0.1, 0.1]),
        maxfev=10000,
    )

    predicted = attn_latency_model(filtered_token_nums, *popt)
    ss_res = float(np.sum((filtered_attention_latencies - predicted) ** 2))
    ss_tot = float(np.sum((filtered_attention_latencies - np.mean(filtered_attention_latencies)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "attn_latency_k": round(float(popt[0]), 10),
        "attn_latency_b": round(float(popt[1]), 10),
        "attn_r_squared": r_squared,
        "attn_points_total": int(len(valid_token_nums)),
        "attn_points_used": int(len(filtered_token_nums)),
    }


def summarize_points(points: list[dict]) -> dict:
    latencies = np.array([p["latency"] for p in points], dtype=float)
    return {
        "mean_latency": float(np.mean(latencies)),
        "median_latency": float(np.median(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "count": int(len(points)),
    }


def build_model_points(detail_dir: Path) -> dict:
    grouped = defaultdict(lambda: {"disable_true": [], "disable_false": []})

    for path in sorted(detail_dir.glob("*.jsonl")):
        info = parse_file_name(path)
        if info is None:
            continue
        tp_pp = f"TP{info['tp']}_PP{info['pp']}"
        model_key = (info["model"], tp_pp)
        decode_points = load_decode_points(path, info["prompt"], info["batch"])
        if not decode_points:
            continue

        if info["disable"]:
            grouped[model_key]["disable_true"].append(
                {
                    "batch": info["batch"],
                    "path": str(path),
                    "summary": summarize_points(decode_points),
                    "representative_latency": float(np.median([p["latency"] for p in decode_points])),
                }
            )
        else:
            grouped[model_key]["disable_false"].extend(decode_points)

    return grouped


def fit_models(detail_dir: Path) -> dict:
    grouped = build_model_points(detail_dir)
    results = {}

    for (model_name, tp_pp), payload in sorted(grouped.items()):
        other_points = [
            {"request_num": point["batch"], "latency": point["representative_latency"]}
            for point in sorted(payload["disable_true"], key=lambda x: x["batch"])
        ]
        if len(other_points) < 3:
            continue

        other_params, _ = fit_other_params(other_points)
        attn_params = fit_attn_params(payload["disable_false"], other_params)

        results.setdefault(model_name, {})
        results[model_name][tp_pp] = {
            **other_params,
            **attn_params,
            "other_fit_points": payload["disable_true"],
            "attn_fit_point_count": len(payload["disable_false"]),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Fit cost models from rollout JSONL detail directory")
    parser.add_argument("detail_dir", type=Path, help="Directory containing rollout JSONL detail files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write per-model fitted JSON files. Defaults to sibling summary/ directory.",
    )
    args = parser.parse_args()

    detail_dir = args.detail_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else detail_dir.parent / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = fit_models(detail_dir)
    combined_path = output_dir / "fitted_cost_models.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    for model_name, model_result in results.items():
        out_path = output_dir / f"{model_name}_cost_model_fit.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(model_result, f, indent=2)
            f.write("\n")

    print(json.dumps(results, indent=2))
    print(f"Wrote {combined_path}")


if __name__ == "__main__":
    main()
