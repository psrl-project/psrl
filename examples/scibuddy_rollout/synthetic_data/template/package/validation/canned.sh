#!/bin/bash
# Negative control: a plausible answer that never reads the input. Must be rejected.
set -euo pipefail
mkdir -p /logs/artifacts

cat > /logs/artifacts/results.json <<'JSON'
{
  "status": "complete",
  "n": 60,
  "mean_temp_c": 9.5,
  "min_temp_c": 7.5,
  "max_temp_c": 11.0,
  "slope_per_day": 0.045,
  "trend": "warming",
  "provenance": {"input_targets": ["/data/series.csv"]}
}
JSON
