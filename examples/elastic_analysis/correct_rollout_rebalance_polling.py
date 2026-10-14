#!/usr/bin/env python3
"""Estimate Rollout rebalance latency after removing the 500 ms router polls.

The exclusive Rollout migration queue dispatches one request at a time. Older
logs used the ordinary routing loop to release the next FIFO item, adding up to
``routing_strategy.check_interval_in_ms`` after each settled dispatch. This
script removes those between-request waits. For the complete executor-side
wall time it also estimates the final completion-poll wait from the gap between
the Router completion and coordinator-command completion timestamps. Interrupt
and re-prefill time are left unchanged. Raw logs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")
START_RE = re.compile(r"Exclusive rollout rebalance started: migration_id=(?P<migration_id>\S+)")
QUEUED_RE = re.compile(r"Queued interrupted rollout request .* request=(?P<request_id>\S+)")
SETTLED_RE = re.compile(r"Exclusive rollout rebalance request settled: request=(?P<request_id>\S+)")
COMPLETED_TEXT = "Exclusive rollout rebalance completed; resuming normal routing."
EXECUTOR_START_TEXT = "coordinator_cmd START stage=ABORT_planned_request_migrations role=Rollout"
EXECUTOR_END_TEXT = "coordinator_cmd END stage=ABORT_planned_request_migrations role=Rollout"
MIGRATION_CONTEXT_RE = re.compile(r"migration_id['\"]?:\s*['\"](?P<migration_id>[^'\"]+)")


@dataclass
class RequestRecord:
    migration_id: str
    request_id: str
    queued_at: datetime | None = None
    dispatched_at: datetime | None = None
    settled_at: datetime | None = None
    interrupt_s: float | None = None
    network_s: float | None = None
    abort_to_redispatch_s: float | None = None
    reprefill_s: float | None = None
    migration_s: float | None = None
    polling_removed_s: float = 0.0


@dataclass
class BatchRecord:
    migration_id: str
    started_at: datetime
    completed_at: datetime | None = None
    requests: dict[str, RequestRecord] = field(default_factory=dict)
    dispatch_order: list[str] = field(default_factory=list)
    polling_removed_s: float = 0.0
    executor_elapsed_s: float | None = None
    executor_completed_at: datetime | None = None
    completion_polling_removed_s: float = 0.0


def _timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    return datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT) if match else None


def _fields(line: str) -> dict[str, str]:
    return {match.group("key"): match.group("value") for match in KEY_VALUE_RE.finditer(line)}


def _number(fields: dict[str, str], key: str) -> float | None:
    value = fields.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_log(path: Path) -> list[BatchRecord]:
    batches: list[BatchRecord] = []
    by_id: dict[str, BatchRecord] = {}
    active: BatchRecord | None = None
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            timestamp = _timestamp(line)
            if timestamp is None:
                continue
            start = START_RE.search(line)
            if start:
                migration_id = start.group("migration_id")
                active = BatchRecord(migration_id=migration_id, started_at=timestamp)
                batches.append(active)
                by_id[migration_id] = active
                continue

            queued = QUEUED_RE.search(line)
            if queued and active is not None:
                request_id = queued.group("request_id")
                record = active.requests.setdefault(
                    request_id,
                    RequestRecord(active.migration_id, request_id),
                )
                record.queued_at = timestamp
                continue

            fields = _fields(line)
            operation = fields.get("operation")
            if operation in {"request_migration_dispatch", "post_scale_up_rebalance"}:
                migration_id = fields.get("migration_id", "")
                request_id = fields.get("request_id", "")
                batch = by_id.get(migration_id)
                if batch is None or not request_id:
                    continue
                record = batch.requests.setdefault(
                    request_id,
                    RequestRecord(migration_id, request_id),
                )
                if operation == "request_migration_dispatch":
                    record.dispatched_at = timestamp
                    record.interrupt_s = _number(fields, "interrupt_s")
                    record.network_s = _number(fields, "network_s")
                    record.abort_to_redispatch_s = _number(fields, "abort_to_redispatch_s")
                    if request_id not in batch.dispatch_order:
                        batch.dispatch_order.append(request_id)
                else:
                    record.reprefill_s = _number(fields, "reprefill_s")
                    record.migration_s = _number(fields, "migration_s")
                continue

            settled = SETTLED_RE.search(line)
            if settled and active is not None:
                request_id = settled.group("request_id")
                record = active.requests.setdefault(
                    request_id,
                    RequestRecord(active.migration_id, request_id),
                )
                record.settled_at = timestamp
                continue

            if COMPLETED_TEXT in line and active is not None:
                active.completed_at = timestamp
                active = None
    return batches


def parse_executor_log(path: Path) -> dict[str, tuple[float, datetime]]:
    """Match each Rollout migration id to its complete coordinator-command wall time."""
    results: dict[str, tuple[float, datetime]] = {}
    active_migration_id: str | None = None
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if EXECUTOR_START_TEXT in line:
                match = MIGRATION_CONTEXT_RE.search(line)
                active_migration_id = match.group("migration_id") if match else None
                continue
            if EXECUTOR_END_TEXT not in line or active_migration_id is None:
                continue
            timestamp = _timestamp(line)
            elapsed_s = _number(_fields(line), "elapsed_s")
            if timestamp is not None and elapsed_s is not None:
                results[active_migration_id] = (elapsed_s, timestamp)
            active_migration_id = None
    return results


def apply_correction(batch: BatchRecord, poll_s: float) -> None:
    removable_intervals: list[tuple[datetime, datetime]] = []
    ordered = [batch.requests[request_id] for request_id in batch.dispatch_order]
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.dispatched_at is None:
            continue
        previous_settled = previous.settled_at or previous.dispatched_at
        if previous_settled is None:
            continue
        ready_at = current.queued_at
        if ready_at is None and current.network_s is not None:
            ready_at = current.dispatched_at - timedelta(seconds=current.network_s)
        if ready_at is None:
            continue
        wait_started = max(previous_settled, ready_at)
        removable_s = min(poll_s, max(0.0, (current.dispatched_at - wait_started).total_seconds()))
        if removable_s <= 0.0:
            continue
        removable_intervals.append(
            (current.dispatched_at - timedelta(seconds=removable_s), current.dispatched_at)
        )
        batch.polling_removed_s += removable_s

    for record in ordered:
        if record.dispatched_at is None or record.network_s is None:
            continue
        ready_at = record.queued_at or record.dispatched_at - timedelta(seconds=record.network_s)
        correction_s = 0.0
        for interval_start, interval_end in removable_intervals:
            overlap_start = max(interval_start, ready_at)
            overlap_end = min(interval_end, record.dispatched_at)
            if overlap_end > overlap_start:
                correction_s += (overlap_end - overlap_start).total_seconds()
        record.polling_removed_s = min(record.network_s, correction_s)

    if batch.completed_at is not None and batch.executor_completed_at is not None:
        completion_gap_s = (batch.executor_completed_at - batch.completed_at).total_seconds()
        batch.completion_polling_removed_s = min(poll_s, max(0.0, completion_gap_s))


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(label: str, batches: list[BatchRecord]) -> dict[str, object]:
    completed_batches = [batch for batch in batches if batch.completed_at is not None]
    raw_batch_s = [(batch.completed_at - batch.started_at).total_seconds() for batch in completed_batches]
    corrected_batch_s = [
        max(0.0, raw - batch.polling_removed_s)
        for batch, raw in zip(completed_batches, raw_batch_s, strict=True)
    ]
    executor_batches = [batch for batch in completed_batches if batch.executor_elapsed_s is not None]
    raw_execution_s = [batch.executor_elapsed_s for batch in executor_batches]
    corrected_execution_s = [
        max(
            0.0,
            batch.executor_elapsed_s - batch.polling_removed_s - batch.completion_polling_removed_s,
        )
        for batch in executor_batches
        if batch.executor_elapsed_s is not None
    ]
    requests = [
        request
        for batch in batches
        for request in batch.requests.values()
        if request.migration_s is not None and request.network_s is not None
    ]
    raw_network_s = [request.network_s for request in requests if request.network_s is not None]
    corrected_network_s = [
        max(0.0, request.network_s - request.polling_removed_s)
        for request in requests
        if request.network_s is not None
    ]
    raw_migration_s = [request.migration_s for request in requests if request.migration_s is not None]
    corrected_migration_s = [
        max(0.0, request.migration_s - request.polling_removed_s)
        for request in requests
        if request.migration_s is not None
    ]
    return {
        "run": label,
        "completed_batches": len(completed_batches),
        "completed_requests": len(requests),
        "raw_batch": _stats(raw_batch_s),
        "corrected_batch": _stats(corrected_batch_s),
        "batch_polling_removed": _stats([batch.polling_removed_s for batch in completed_batches]),
        "raw_execution": _stats(raw_execution_s),
        "corrected_execution": _stats(corrected_execution_s),
        "execution_polling_removed": _stats(
            [batch.polling_removed_s + batch.completion_polling_removed_s for batch in executor_batches]
        ),
        "completion_polling_removed": _stats(
            [batch.completion_polling_removed_s for batch in executor_batches]
        ),
        "raw_network": _stats(raw_network_s),
        "corrected_network": _stats(corrected_network_s),
        "raw_migration": _stats(raw_migration_s),
        "corrected_migration": _stats(corrected_migration_s),
        "request_polling_removed": _stats([request.polling_removed_s for request in requests]),
    }


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not label or not (path / "RolloutRouter.log").is_file():
        raise argparse.ArgumentTypeError(f"invalid run or missing RolloutRouter.log: {value}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=_parse_run, metavar="LABEL=RUN_DIR")
    parser.add_argument("--poll-ms", type=float, default=500.0)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--batch-output-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    batch_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    for label, run_dir in args.run:
        batches = parse_log(run_dir / "RolloutRouter.log")
        executor_results = parse_executor_log(run_dir / "ElasticExecutor.log")
        for batch in batches:
            executor_result = executor_results.get(batch.migration_id)
            if executor_result is not None:
                batch.executor_elapsed_s, batch.executor_completed_at = executor_result
            apply_correction(batch, args.poll_ms / 1000.0)
            raw_router_s = (
                (batch.completed_at - batch.started_at).total_seconds()
                if batch.completed_at is not None
                else None
            )
            total_polling_removed_s = batch.polling_removed_s + batch.completion_polling_removed_s
            batch_rows.append(
                {
                    "run": label,
                    "migration_id": batch.migration_id,
                    "request_count": len(batch.dispatch_order),
                    "raw_router_batch_s": raw_router_s,
                    "between_request_polling_removed_s": batch.polling_removed_s,
                    "corrected_router_batch_s": (
                        max(0.0, raw_router_s - batch.polling_removed_s)
                        if raw_router_s is not None
                        else None
                    ),
                    "raw_execution_s": batch.executor_elapsed_s,
                    "completion_polling_removed_s": batch.completion_polling_removed_s,
                    "total_polling_removed_s": total_polling_removed_s,
                    "corrected_execution_s": (
                        max(0.0, batch.executor_elapsed_s - total_polling_removed_s)
                        if batch.executor_elapsed_s is not None
                        else None
                    ),
                }
            )
            for request in batch.requests.values():
                if request.migration_s is None or request.network_s is None:
                    continue
                request_rows.append(
                    {
                        "run": label,
                        "migration_id": batch.migration_id,
                        "request_id": request.request_id,
                        "raw_network_s": request.network_s,
                        "polling_removed_s": request.polling_removed_s,
                        "corrected_network_s": max(0.0, request.network_s - request.polling_removed_s),
                        "raw_migration_s": request.migration_s,
                        "corrected_migration_s": max(0.0, request.migration_s - request.polling_removed_s),
                    }
                )
        summary = summarize(label, batches)
        row: dict[str, object] = {
            "run": label,
            "completed_batches": summary["completed_batches"],
            "completed_requests": summary["completed_requests"],
        }
        for metric in (
            "raw_batch",
            "corrected_batch",
            "batch_polling_removed",
            "raw_execution",
            "corrected_execution",
            "execution_polling_removed",
            "completion_polling_removed",
            "raw_network",
            "corrected_network",
            "raw_migration",
            "corrected_migration",
            "request_polling_removed",
        ):
            for statistic, value in summary[metric].items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)

    columns = list(rows[0])
    print(",".join(columns))
    for row in rows:
        print(",".join(str(row[column]) for column in columns))
    if args.summary_csv:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    if args.batch_output_csv:
        args.batch_output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.batch_output_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(batch_rows[0]) if batch_rows else [])
            if batch_rows:
                writer.writeheader()
                writer.writerows(batch_rows)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(request_rows[0]) if request_rows else [])
            if request_rows:
                writer.writeheader()
                writer.writerows(request_rows)


if __name__ == "__main__":
    main()
