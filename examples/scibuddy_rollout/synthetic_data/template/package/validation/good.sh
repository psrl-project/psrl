#!/bin/bash
# Reference solution: compute the summary from the bound input.
set -euo pipefail
mkdir -p /logs/artifacts

python3 - <<'PY'
import csv, json

days, temps = [], []
with open("/data/series.csv", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        days.append(float(row["day"]))
        temps.append(float(row["temp_c"]))

n = len(temps)
mean_temp = sum(temps) / n
mean_day = sum(days) / n
cov = sum((d - mean_day) * (t - mean_temp) for d, t in zip(days, temps))
var_day = sum((d - mean_day) ** 2 for d in days)
slope = cov / var_day
trend = "warming" if slope > 0.01 else "cooling" if slope < -0.01 else "flat"

json.dump(
    {
        "status": "complete",
        "n": n,
        "mean_temp_c": mean_temp,
        "min_temp_c": min(temps),
        "max_temp_c": max(temps),
        "slope_per_day": slope,
        "trend": trend,
        "provenance": {"input_targets": ["/data/series.csv"]},
    },
    open("/logs/artifacts/results.json", "w"),
)
PY
