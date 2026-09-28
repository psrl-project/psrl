"""Grade the synthetic tempstats submission.

Mirrors the real SciBuddy verifier shape: validate the schema, recompute every
reported quantity from the bound input, and write a binary reward to
/logs/verifier/reward.txt.
"""

import csv
import json
import math
from pathlib import Path

TOL = 1e-6


class InvalidSubmission(ValueError):
    pass


def finite(x):
    return type(x) in (int, float) and math.isfinite(float(x))


def require(condition, message):
    if not condition:
        raise InvalidSubmission(message)


def reference(series_path):
    """Recompute the expected summary from the input series."""
    days, temps = [], []
    with open(series_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            days.append(float(row["day"]))
            temps.append(float(row["temp_c"]))

    n = len(temps)
    mean_temp = sum(temps) / n
    mean_day = sum(days) / n
    var_day = sum((d - mean_day) ** 2 for d in days)
    cov = sum((d - mean_day) * (t - mean_temp) for d, t in zip(days, temps))
    slope = cov / var_day
    if slope > 0.01:
        trend = "warming"
    elif slope < -0.01:
        trend = "cooling"
    else:
        trend = "flat"
    return {
        "n": n,
        "mean_temp_c": mean_temp,
        "min_temp_c": min(temps),
        "max_temp_c": max(temps),
        "slope_per_day": slope,
        "trend": trend,
    }


def _submission_path(workspace):
    """Locate the submission in the publish dir, falling back to the workspace.

    With `environment_mode = "separate"` Harbor transfers only the agent's
    `/logs/artifacts` publish dir into the verifier, so that is the graded
    location. The workspace fallback keeps the grader usable when it is run by
    hand against a trial directory.
    """
    for candidate in (
        Path("/logs/artifacts/results.json"),
        workspace / "results.json",
    ):
        if candidate.exists():
            return candidate
    return Path("/logs/artifacts/results.json")


def check(workspace, series_path):
    try:
        value = json.loads(_submission_path(workspace).read_text())
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidSubmission("Missing or malformed results.json") from exc

    require(isinstance(value, dict), "results.json must contain an object")
    status = value.get("status")
    require(status in ("complete", "fail_closed"), "status must be complete or fail_closed")

    if status == "fail_closed":
        require(bool(value.get("failed_gate")), "fail_closed must name a failed gate")
        # A fail_closed certificate is only honest when the input really is unusable.
        require(not Path("/data/series.csv").exists(), "fail_closed claimed but the input is readable")
        return

    expected = reference(series_path)

    require(value.get("n") == expected["n"], f"n must be {expected['n']}")
    for key in ("mean_temp_c", "min_temp_c", "max_temp_c", "slope_per_day"):
        got = value.get(key)
        require(finite(got), f"{key} must be a finite number")
        require(
            math.isclose(float(got), expected[key], rel_tol=1e-4, abs_tol=TOL),
            f"{key} disagrees with the recomputed value",
        )

    require(
        float(value["min_temp_c"]) <= float(value["mean_temp_c"]) <= float(value["max_temp_c"]),
        "min <= mean <= max is violated",
    )
    require(value.get("trend") == expected["trend"], f"trend must be {expected['trend']!r}")

    provenance = value.get("provenance")
    require(isinstance(provenance, dict), "missing provenance")
    require(
        provenance.get("input_targets") == ["/data/series.csv"],
        "provenance must name the bound input target",
    )


def _series_path():
    """Locate the reference series, preferring the verifier's own baked copy."""
    for candidate in (Path("/tests/series.csv"), Path("/data/series.csv")):
        if candidate.exists():
            return candidate
    return Path("/data/series.csv")


def main(workspace=Path("/workspace"), logs=Path("/logs/verifier")):
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "reward.txt").unlink(missing_ok=True)
    try:
        check(workspace, _series_path())
    except InvalidSubmission as exc:
        report = {"reward": 0, "outcome": "rejected", "reason": str(exc)}
    else:
        report = {"reward": 1, "outcome": "accepted"}
    (logs / "report.json").write_text(json.dumps(report))
    (logs / "reward.txt").write_text(str(report["reward"]))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
