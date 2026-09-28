#!/bin/bash
# Negative control: correct schema, wrong arithmetic. Must be rejected.
set -euo pipefail
mkdir -p /logs/artifacts

python3 - <<'PY'
import csv, json

temps = []
with open("/data/series.csv", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        temps.append(float(row["temp_c"]))

json.dump(
    {
        "status": "complete",
        "n": len(temps),
        # Deliberately shifted mean and a slope of the wrong sign.
        "mean_temp_c": sum(temps) / len(temps) + 3.0,
        "min_temp_c": min(temps),
        "max_temp_c": max(temps),
        "slope_per_day": -0.5,
        "trend": "cooling",
        "provenance": {"input_targets": ["/data/series.csv"]},
    },
    open("/logs/artifacts/results.json", "w"),
)
PY
