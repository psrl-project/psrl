# Synthetic temperature-series summary

This is a synthetic SciBuddy-shaped task. It exists to exercise the batch-rollout
integration end to end, so the science is deliberately trivial while the output
contract matches the real task family: a schema-checked `results.json` graded to a
binary reward.

## Objective

Summarise the bound temperature series and classify its trend.

## Inputs

One read-only input is mounted at `/data`:

- `/data/series.csv`, two columns `day,temp_c`, with a header row.

## Required computation

From `/data/series.csv`, newly compute:

1. `n`, the number of data rows (excluding the header).
2. `mean_temp_c`, the arithmetic mean of `temp_c`.
3. `min_temp_c` and `max_temp_c`.
4. `slope_per_day`, the ordinary least-squares slope of `temp_c` against `day`.
5. `trend`, which is `"warming"` when the slope is above `+0.01`, `"cooling"`
   when it is below `-0.01`, and `"flat"` otherwise.

Do not hard-code these values. Read the file and compute them.

## Output contract

Write `/logs/artifacts/results.json` as a JSON object with exactly these fields.
That directory is the graded publish location, so a file left anywhere else
(including `/workspace`) is not collected and scores zero:

```json
{
  "status": "complete",
  "n": 0,
  "mean_temp_c": 0.0,
  "min_temp_c": 0.0,
  "max_temp_c": 0.0,
  "slope_per_day": 0.0,
  "trend": "flat",
  "provenance": {
    "input_targets": ["/data/series.csv"]
  }
}
```

Every number must be finite. `min_temp_c` must not exceed `mean_temp_c`, and
`mean_temp_c` must not exceed `max_temp_c`. `trend` must agree with the sign of
the slope you report. The verifier recomputes all of it from the same input and
rejects a submission whose numbers disagree beyond a small tolerance, so a
guessed or canned answer fails.

Report `"status": "fail_closed"` instead, naming the problem in a `failed_gate`
field, if the input cannot be read or parsed.
