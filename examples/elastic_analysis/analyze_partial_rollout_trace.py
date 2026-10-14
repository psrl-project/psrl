#!/usr/bin/env python3
"""Summarize request-level partial-rollout trace events from a PSRL log directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TRACE_MARKER = "[PARTIAL_ROLLOUT_TRACE]"
FIELD_RE = re.compile(r"(?P<key>[A-Za-z_]+)=(?P<value>[^\s]+)")
TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")


def _parse_value(value: str) -> Any:
    if value == "none":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def parse_trace_events(log_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    trace_paths = sorted(log_dir.glob("GenWorker_I*_R*.log"))
    router_log = log_dir / "RolloutRouter.log"
    if router_log.is_file():
        trace_paths.append(router_log)
    for path in trace_paths:
        with path.open(encoding="utf-8", errors="replace") as log_file:
            for line_number, line in enumerate(log_file, start=1):
                if TRACE_MARKER not in line:
                    continue
                fields = {match["key"]: _parse_value(match["value"]) for match in FIELD_RE.finditer(line)}
                if "stage" not in fields or "uid" not in fields:
                    continue
                timestamp_match = TIMESTAMP_RE.match(line)
                fields.update(
                    {
                        "file": path.name,
                        "line": line_number,
                        "timestamp": timestamp_match["timestamp"] if timestamp_match else "",
                    }
                )
                events.append(fields)
    return sorted(events, key=lambda event: (event["timestamp"], event["file"], event["line"]))


def analyze_events(
    events: list[dict[str, Any]],
    require_logprob_alignment: bool,
    staleness_limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_uid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_uid[int(event["uid"])].append(event)

    violations: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for uid, uid_events in sorted(by_uid.items()):
        starts = [event for event in uid_events if event["stage"] == "generate_start"]
        results = [event for event in uid_events if event["stage"] == "generate_result"]
        continuation_starts = [event for event in starts if event.get("partial") == 1]
        cross_version = [
            event
            for event in continuation_starts
            if event.get("request_version") != event.get("loaded_version")
        ]
        counters["partial_uids"] += int(bool(continuation_starts))
        counters["cross_version_uids"] += int(bool(cross_version))
        counters["cross_version_chunks"] += len(cross_version)
        counters["same_version_continuation_chunks"] += len(continuation_starts) - len(cross_version)
        for start in starts:
            for field in ("request_staleness", "loaded_staleness"):
                value = start.get(field)
                if value is None or value < 0:
                    continue
                counters[f"max_{field}"] = max(counters[f"max_{field}"], int(value))
                if staleness_limit is not None and value > staleness_limit:
                    violations.append(
                        {
                            "uid": uid,
                            "kind": field,
                            "chunk": start.get("chunk"),
                            "detail": f"{field}={value} exceeds limit={staleness_limit}",
                            "file": start["file"],
                            "line": start["line"],
                        }
                    )

        results_by_chunk = {event.get("chunk"): event for event in results}
        for start in starts:
            if start.get("partial") != 1:
                continue
            previous = results_by_chunk.get(int(start.get("chunk", 0)) - 1)
            if previous is not None and (
                start.get("prefix_tokens") != previous.get("response_tokens")
                or start.get("prefix_sha256") != previous.get("response_sha256")
            ):
                violations.append(
                    {
                        "uid": uid,
                        "kind": "prefix_continuity",
                        "chunk": start.get("chunk"),
                        "detail": "resumed prefix differs from the preceding partial result",
                        "file": start["file"],
                        "line": start["line"],
                    }
                )
            if require_logprob_alignment and start.get("prior_logprob_tokens") != start.get("prefix_tokens"):
                violations.append(
                    {
                        "uid": uid,
                        "kind": "prefix_logprob_alignment",
                        "chunk": start.get("chunk"),
                        "detail": "prefix token count differs from accumulated rollout logprob count",
                        "file": start["file"],
                        "line": start["line"],
                    }
                )

        for result in results:
            if result.get("generated_tokens") != result.get("response_tokens", 0) - result.get("prefix_tokens", 0):
                violations.append(
                    {
                        "uid": uid,
                        "kind": "response_length",
                        "chunk": result.get("chunk"),
                        "detail": "response_tokens - prefix_tokens differs from generated_tokens",
                        "file": result["file"],
                        "line": result["line"],
                    }
                )
            if require_logprob_alignment and result.get("rollout_logprob_tokens") != result.get("response_tokens"):
                violations.append(
                    {
                        "uid": uid,
                        "kind": "result_logprob_alignment",
                        "chunk": result.get("chunk"),
                        "detail": "response token count differs from accumulated rollout logprob count",
                        "file": result["file"],
                        "line": result["line"],
                    }
                )

        versions = sorted(
            {
                int(event["loaded_version"])
                for event in starts
                if event.get("loaded_version") is not None
            }
        )
        request_rows.append(
            {
                "uid": uid,
                "start_chunks": len(starts),
                "result_chunks": len(results),
                "continuation_chunks": len(continuation_starts),
                "cross_version_chunks": len(cross_version),
                "loaded_versions": ",".join(map(str, versions)),
                "interrupted_chunks": sum(int(event.get("interrupted", 0)) for event in results),
                "scheduler_interrupted_chunks": sum(
                    int(event.get("interrupted_by_scheduler", 0)) for event in results
                ),
            }
        )

    counters["trace_events"] = len(events)
    counters["traced_uids"] = len(by_uid)
    counters["violations"] = len(violations)
    counters["router_partial_routes"] = sum(event["stage"] == "route" for event in events)
    routed_events = [
        event
        for event in events
        if event["stage"] == "route" and event.get("selected_instance") is not None
    ]
    counters["router_failed_partial_routes"] = counters["router_partial_routes"] - len(routed_events)
    counters["router_cross_instance_chunks"] = sum(
        event.get("previous_instance") != event.get("selected_instance") for event in routed_events
    )
    counters["router_cross_instance_uids"] = len(
        {
            int(event["uid"])
            for event in routed_events
            if event.get("previous_instance") != event.get("selected_instance")
        }
    )
    counters["router_same_instance_chunks"] = (
        len(routed_events) - counters["router_cross_instance_chunks"]
    )
    counters["router_higher_version_routes"] = sum(
        event.get("selected_version") is not None
        and event.get("selected_version") > event.get("requested_version", event.get("selected_version"))
        for event in routed_events
    )
    return {"summary": dict(sorted(counters.items())), "requests": request_rows}, violations


def write_report(output_dir: Path, report: dict[str, Any], violations: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "partial_rollout_trace_summary.json").write_text(
        json.dumps(report["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, rows, fieldnames in (
        (
            "partial_rollout_trace_requests.csv",
            report["requests"],
            [
                "uid",
                "start_chunks",
                "result_chunks",
                "continuation_chunks",
                "cross_version_chunks",
                "loaded_versions",
                "interrupted_chunks",
                "scheduler_interrupted_chunks",
            ],
        ),
        ("partial_rollout_trace_violations.csv", violations, ["uid", "kind", "chunk", "detail", "file", "line"]),
    ):
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="PSRL run log directory")
    parser.add_argument("--output-dir", type=Path, help="defaults to LOG_DIR/partial_rollout_trace")
    parser.add_argument(
        "--require-logprob-alignment",
        action="store_true",
        help="treat accumulated rollout logprob length mismatches as violations",
    )
    parser.add_argument(
        "--staleness-limit",
        type=int,
        help="treat request or loaded staleness above this value as a violation",
    )
    args = parser.parse_args()
    events = parse_trace_events(args.log_dir)
    report, violations = analyze_events(
        events,
        args.require_logprob_alignment,
        staleness_limit=args.staleness_limit,
    )
    output_dir = args.output_dir or args.log_dir / "partial_rollout_trace"
    write_report(output_dir, report, violations)
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
