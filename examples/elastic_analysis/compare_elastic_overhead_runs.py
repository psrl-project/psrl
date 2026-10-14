#!/usr/bin/env python3
"""Extract and compare elastic scaling overheads from two PSRL runs.

The parser reads executor, router, and coordinator logs from each run
directory. It prints a compact comparison
and can write five CSV files:

* overhead_events.csv: one row per extracted numeric metric.
* overhead_summary.csv: descriptive statistics for each metric.
* rebalance_effectiveness.csv: planned/accepted/interrupted/routed counts.
* scaling_actions_per_step.csv: completed policy actions and instance transitions.
* run_windows.csv: source-log timestamps and the selected relative-time window.

``--usable-only`` drops request-migration interrupt/dispatch records that do
not have a matching terminal ``post_scale_up_rebalance`` record.  This keeps
aborted or incomplete requests out of latency statistics while preserving the
raw parser behavior by default.

Percentiles use linear interpolation over sorted samples. In current logs,
``network_s`` spans requeue to redispatch and is not pure network transfer;
``abort_to_redispatch_s`` preserves the full interruption-plus-redispatch span.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
OVERHEAD_RE = re.compile(r"\[ELASTIC_OVERHEAD\]\s+(?P<body>.*)$")
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
PREPARED_RE = re.compile(
    r"Prepared (?P<role>RM|rollout) request migrations: "
    r"planned=(?P<planned>\d+) accepted=(?P<accepted>\d+) skipped=(?P<skipped>\d+)"
)

FAST_POLICY_REASONS = {"cooldown", "decision_execution_in_progress"}
_STEP_SCALING_METRICS = (
    "actual_actions",
    "scale_up_actions",
    "scale_down_actions",
    "sleep_instances",
    "wakeup_instances",
    "instance_transitions",
)
LOG_FILES = (
    "ElasticExecutor.log",
    "RolloutRouter.log",
    "RewardModelRouter.log",
    "RolloutCoordinator.log",
    "RewardModelCoordinator.log",
)
TIME_METRICS = {
    "planner_s",
    "state_analysis_s",
    "candidate_ordering_s",
    "candidate_set_construction_s",
    "simulation_input_preparation_s",
    "candidate_evaluation_wall_s",
    "rebalance_simulation_s",
    "router_simulation_s",
    "simulation_wall_s",
    "rebalance_router_overlap_s",
    "candidate_scoring_s",
    "best_candidate_selection_s",
    "other_s",
    "planner_other_s",
    "sleep_s",
    "wakeup_s",
    "post_scale_up_rebalance_trigger_s",
    "handler_other_s",
    "execution_s",
    "total_s",
    "network_trigger_s",
    "planner_batch_s",
    "planner_share_s",
    "interrupt_s",
    "network_s",
    "abort_to_redispatch_s",
    "reprefill_s",
    "migration_s",
    "router_pause_s",
    "state_probe_s",
    "engine_sleep_s",
    "engine_wakeup_s",
    "router_update_s",
    "model_sync_s",
    "router_resume_s",
    "planner_candidate_set_construction_share_s",
    "planner_rebalance_simulation_share_s",
    "planner_router_simulation_share_s",
    "planner_best_candidate_selection_share_s",
    "planner_other_share_s",
    "engine_status_s",
    "router_backlog_s",
    "request_snapshot_s",
    "trainer_hint_s",
    "signal_build_s",
    "signal_logging_s",
    "input_other_s",
    "input_total_s",
}


@dataclass(frozen=True)
class MetricEvent:
    run: str
    run_path: str
    timestamp: str
    elapsed_min: float
    source: str
    operation: str
    cohort: str
    role: str
    method: str
    step: str
    decision_id: str
    migration_id: str
    request_id: str
    reason: str
    metric: str
    value: float


@dataclass(frozen=True)
class MetricSummary:
    run: str
    operation: str
    cohort: str
    role: str
    metric: str
    count: int
    mean: float
    stddev: float
    minimum: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: float
    total: float


@dataclass(frozen=True)
class ScalingActionStepSummary:
    run: str
    step: int
    actual_actions: int
    scale_up_actions: int
    scale_down_actions: int
    sleep_instances: int
    wakeup_instances: int
    instance_transitions: int


@dataclass
class RebalanceCounts:
    run: str
    role: str
    method: str = "unknown"
    trigger_events: int = 0
    planned: int = 0
    accepted: int = 0
    interrupted: int = 0
    skipped: int = 0
    forced: int = 0
    fallback: int = 0
    redispatched_overhead: int = 0
    completed_overhead: int = 0

    def as_row(self) -> dict[str, object]:
        row = asdict(self)
        request_level = self.method == "request_level"
        row["accepted_available"] = request_level
        row["accepted_ratio"] = _safe_ratio(self.accepted, self.planned) if request_level else float("nan")
        row["interrupted_ratio"] = _safe_ratio(self.interrupted, self.planned)
        row["forced_ratio"] = _safe_ratio(self.forced, self.forced + self.fallback)
        return row


@dataclass(frozen=True)
class ParsedOverhead:
    timestamp: datetime
    source: str
    fields: dict[str, str]


@dataclass(frozen=True)
class RunWindow:
    run: str
    run_path: str
    first_timestamp: str
    last_timestamp: str
    observed_minutes: float
    selected_start_min: float
    selected_end_min: float


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def _parse_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    normalized = raw.rstrip(",")
    if not NUMBER_RE.fullmatch(normalized):
        return None
    value = float(normalized)
    return value if math.isfinite(value) else None


def _parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT)


def _role_for_source(source: str) -> str:
    if source == "RolloutRouter.log":
        return "Rollout"
    if source == "RewardModelRouter.log":
        return "RewardModel"
    return "all"


def _role_for_overhead(parsed: ParsedOverhead) -> str:
    explicit_role = parsed.fields.get("role")
    if explicit_role:
        return explicit_role
    migration_parts = parsed.fields.get("migration_id", "").split(":")
    if len(migration_parts) >= 3 and migration_parts[2] in {"RewardModel", "Rollout"}:
        return migration_parts[2]
    return _role_for_source(parsed.source)


def _normalize_overhead(parsed: ParsedOverhead) -> tuple[str, str, str, str, dict[str, float]] | None:
    fields = parsed.fields
    raw_operation = fields.get("operation", "")
    role = _role_for_overhead(parsed)
    method = "unknown"
    cohort = "all"
    metrics: dict[str, float] = {}

    if raw_operation == "policy_planner":
        operation = "policy_planner"
        reason = fields.get("reason", "")
        actions = int(_parse_number(fields.get("actions")) or 0)
        if actions > 0:
            cohort = "action"
            if reason.endswith("RewardModel"):
                role = "RewardModel"
            elif reason.endswith("Rollout"):
                role = "Rollout"
        elif reason in FAST_POLICY_REASONS:
            cohort = "fast_gate"
        else:
            cohort = "no_action_evaluation"
        for metric in (
            "planner_s",
            "state_analysis_s",
            "candidate_ordering_s",
            "candidate_set_construction_s",
            "simulation_input_preparation_s",
            "candidate_evaluation_wall_s",
            "rebalance_simulation_s",
            "router_simulation_s",
            "simulation_wall_s",
            "rebalance_router_overlap_s",
            "candidate_scoring_s",
            "best_candidate_selection_s",
            "other_s",
        ):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation == "policy_input":
        operation = "policy_input"
        for metric in (
            "engine_status_s",
            "router_backlog_s",
            "request_snapshot_s",
            "trainer_hint_s",
            "signal_build_s",
            "signal_logging_s",
            "input_other_s",
            "input_total_s",
        ):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation in {"scale_up", "scale_down"}:
        operation = raw_operation
        for metric in TIME_METRICS:
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation in {"sleep", "wakeup"}:
        operation = raw_operation
        for metric in TIME_METRICS:
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation == "post_scale_up_rebalance_trigger":
        operation = "rebalance_trigger"
        method = "legacy"
        for metric in ("planner_s", "network_trigger_s"):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
        planned = _parse_number(fields.get("selected"))
        interrupted = _parse_number(fields.get("interrupted"))
        if planned is not None:
            metrics["planned_requests"] = planned
        if interrupted is not None:
            metrics["interrupted_requests"] = interrupted
        if planned and interrupted is not None:
            metrics["interrupt_ratio"] = interrupted / planned
    elif raw_operation == "planned_request_rebalance":
        operation = "rebalance_trigger"
        method = "request_level"
        planner_s = _parse_number(fields.get("planner_s"))
        planned = _parse_number(fields.get("planned"))
        interrupted = _parse_number(fields.get("interrupted"))
        if planner_s is not None:
            metrics["planner_s"] = planner_s
        if planned is not None:
            metrics["planned_requests"] = planned
        if interrupted is not None:
            metrics["interrupted_requests"] = interrupted
        if planned and interrupted is not None:
            metrics["interrupt_ratio"] = interrupted / planned
    elif raw_operation == "post_scale_up_rebalance":
        operation = "request_migration"
        migration_id = fields.get("migration_id", "")
        method = "request_level" if migration_id.endswith(":planned") else "legacy"
        for metric in (
            "planner_batch_s",
            "planner_share_s",
            "planner_candidate_set_construction_share_s",
            "planner_rebalance_simulation_share_s",
            "planner_router_simulation_share_s",
            "planner_best_candidate_selection_share_s",
            "planner_other_share_s",
            "interrupt_s",
            "network_s",
            "abort_to_redispatch_s",
            "reprefill_s",
            "migration_s",
        ):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation == "request_migration_dispatch":
        operation = "request_migration_dispatch"
        migration_id = fields.get("migration_id", "")
        method = "request_level" if migration_id.endswith(":planned") else "legacy"
        for metric in (
            "planner_batch_s",
            "planner_share_s",
            "planner_candidate_set_construction_share_s",
            "planner_rebalance_simulation_share_s",
            "planner_router_simulation_share_s",
            "planner_best_candidate_selection_share_s",
            "planner_other_share_s",
            "interrupt_s",
            "network_s",
            "abort_to_redispatch_s",
        ):
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    elif raw_operation == "request_migration_interrupt":
        operation = "request_migration_interrupt"
        migration_id = fields.get("migration_id", "")
        method = "request_level" if migration_id.endswith(":planned") else "legacy"
        interrupt_s = _parse_number(fields.get("interrupt_s"))
        if interrupt_s is not None:
            metrics["interrupt_s"] = interrupt_s
    elif raw_operation == "scaling_action":
        operation = "scaling_action"
        for metric in _STEP_SCALING_METRICS:
            value = _parse_number(fields.get(metric))
            if value is not None:
                metrics[metric] = value
    else:
        return None

    if not metrics:
        return None
    return operation, cohort, role, method, metrics


def parse_overhead_lines(run_dir: Path) -> list[ParsedOverhead]:
    parsed: list[ParsedOverhead] = []
    for filename in LOG_FILES:
        log_path = run_dir / filename
        if not log_path.is_file():
            continue
        with log_path.open(encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                overhead_match = OVERHEAD_RE.search(line)
                if overhead_match is None:
                    continue
                timestamp = _parse_timestamp(line)
                if timestamp is None:
                    continue
                fields = {
                    match.group("key"): match.group("value")
                    for match in KEY_VALUE_RE.finditer(overhead_match.group("body"))
                }
                parsed.append(ParsedOverhead(timestamp, filename, fields))
    return parsed


def collect_metric_events(
    label: str,
    run_dir: Path,
    *,
    start_min: float,
    end_min: float | None,
) -> list[MetricEvent]:
    parsed = parse_overhead_lines(run_dir)
    if not parsed:
        return []
    origin = min(item.timestamp for item in parsed)
    events: list[MetricEvent] = []
    for item in parsed:
        elapsed_min = (item.timestamp - origin).total_seconds() / 60.0
        if elapsed_min < start_min or (end_min is not None and elapsed_min > end_min):
            continue
        normalized = _normalize_overhead(item)
        if normalized is None:
            continue
        operation, cohort, role, method, metrics = normalized
        for metric, value in metrics.items():
            events.append(
                MetricEvent(
                    run=label,
                    run_path=str(run_dir),
                    timestamp=item.timestamp.isoformat(sep=" ", timespec="milliseconds"),
                    elapsed_min=elapsed_min,
                    source=item.source,
                    operation=operation,
                    cohort=cohort,
                    role=role,
                    method=method,
                    step=item.fields.get("step", ""),
                    decision_id=item.fields.get("decision_id", ""),
                    migration_id=item.fields.get("migration_id", ""),
                    request_id=item.fields.get("request_id", ""),
                    reason=item.fields.get("reason", ""),
                    metric=metric,
                    value=value,
                )
            )
    return events


def filter_usable_migration_events(events: Iterable[MetricEvent]) -> list[MetricEvent]:
    """Keep migration records with an observed terminal completion.

    The selective-abort path can emit an interrupt or dispatch timing even
    when the request never returns a final result.  A terminal record carrying
    Both ``migration_s`` and ``reprefill_s`` on the terminal record are the
    evidence that all three timing boundaries belong to one completed
    migration. Match by request ID when available and fall back to migration
    ID for older log formats.
    """
    event_list = list(events)
    complete_exact_metrics: dict[tuple[str, str, str, str], set[str]] = {}
    legacy_metrics: dict[tuple[str, str, str], set[str]] = {}
    for event in event_list:
        if event.operation != "request_migration":
            continue
        migration_key = (event.run, event.role, event.migration_id)
        if event.request_id:
            complete_exact_metrics.setdefault((*migration_key, event.request_id), set()).add(event.metric)
        else:
            legacy_metrics.setdefault(migration_key, set()).add(event.metric)

    complete_exact = {
        request_key
        for request_key, metrics in complete_exact_metrics.items()
        if {"migration_s", "reprefill_s"}.issubset(metrics)
    }
    usable_migrations = {
        migration_key
        for migration_key, metrics in legacy_metrics.items()
        if {"migration_s", "reprefill_s"}.issubset(metrics)
    }

    migration_operations = {
        "request_migration",
        "request_migration_dispatch",
        "request_migration_interrupt",
    }
    usable: list[MetricEvent] = []
    for event in event_list:
        if event.operation not in migration_operations:
            usable.append(event)
            continue
        migration_key = (event.run, event.role, event.migration_id)
        if event.request_id:
            if (*migration_key, event.request_id) not in complete_exact:
                continue
        elif migration_key not in usable_migrations:
            continue
        usable.append(event)
    return usable


def describe_run_window(
    label: str,
    run_dir: Path,
    parsed: Sequence[ParsedOverhead],
    *,
    start_min: float,
    end_min: float | None,
) -> RunWindow:
    first = min(item.timestamp for item in parsed)
    last = max(item.timestamp for item in parsed)
    observed_minutes = (last - first).total_seconds() / 60.0
    return RunWindow(
        run=label,
        run_path=str(run_dir),
        first_timestamp=first.isoformat(sep=" ", timespec="milliseconds"),
        last_timestamp=last.isoformat(sep=" ", timespec="milliseconds"),
        observed_minutes=observed_minutes,
        selected_start_min=start_min,
        selected_end_min=min(end_min, observed_minutes) if end_min is not None else observed_minutes,
    )


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        return float("nan")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize_metric_events(events: Iterable[MetricEvent]) -> list[MetricSummary]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for event in events:
        grouped[(event.run, event.operation, event.cohort, event.role, event.metric)].append(event.value)

    summaries: list[MetricSummary] = []
    for (run, operation, cohort, role, metric), raw_values in sorted(grouped.items()):
        values = sorted(raw_values)
        summaries.append(
            MetricSummary(
                run=run,
                operation=operation,
                cohort=cohort,
                role=role,
                metric=metric,
                count=len(values),
                mean=statistics.fmean(values),
                stddev=statistics.pstdev(values),
                minimum=values[0],
                p50=_percentile(values, 0.50),
                p90=_percentile(values, 0.90),
                p95=_percentile(values, 0.95),
                p99=_percentile(values, 0.99),
                maximum=values[-1],
                total=sum(values),
            )
        )
    return summaries


def summarize_scaling_actions_per_step(
    events: Iterable[MetricEvent],
) -> list[ScalingActionStepSummary]:
    grouped: dict[tuple[str, int], dict[str, int]] = {}
    for event in events:
        if event.operation != "scaling_action" or event.metric not in _STEP_SCALING_METRICS:
            continue
        try:
            step = int(event.step)
        except (TypeError, ValueError):
            continue
        counts = grouped.setdefault(
            (event.run, step),
            {metric: 0 for metric in _STEP_SCALING_METRICS},
        )
        counts[event.metric] += int(event.value)
    return [
        ScalingActionStepSummary(run=run, step=step, **counts)
        for (run, step), counts in sorted(grouped.items())
    ]


def _in_window(timestamp: datetime | None, origin: datetime | None, start_min: float, end_min: float | None) -> bool:
    if timestamp is None or origin is None:
        return False
    elapsed_min = (timestamp - origin).total_seconds() / 60.0
    return elapsed_min >= start_min and (end_min is None or elapsed_min <= end_min)


def collect_rebalance_counts(
    label: str,
    run_dir: Path,
    parsed_overheads: Sequence[ParsedOverhead],
    *,
    start_min: float,
    end_min: float | None,
    usable_only: bool = False,
) -> list[RebalanceCounts]:
    origin = min((item.timestamp for item in parsed_overheads), default=None)
    counts = {role: RebalanceCounts(run=label, role=role) for role in ("RewardModel", "Rollout")}

    usable_records: set[tuple[str, str, str]] | None = None
    if usable_only:
        parsed_events = collect_metric_events(
            label,
            run_dir,
            start_min=start_min,
            end_min=end_min,
        )
        usable_records = {
            (event.role, event.migration_id, event.request_id)
            for event in filter_usable_migration_events(parsed_events)
            if event.operation in {"request_migration", "request_migration_dispatch"}
        }

    for item in parsed_overheads:
        if not _in_window(item.timestamp, origin, start_min, end_min):
            continue
        operation = item.fields.get("operation")
        role = _role_for_overhead(item)
        if role not in counts:
            continue
        row = counts[role]
        if operation == "post_scale_up_rebalance_trigger":
            row.method = "legacy"
            row.trigger_events += 1
            row.planned += int(_parse_number(item.fields.get("selected")) or 0)
            row.interrupted += int(_parse_number(item.fields.get("interrupted")) or 0)
        elif operation == "planned_request_rebalance":
            row.method = "request_level"
            row.trigger_events += 1
            row.planned += int(_parse_number(item.fields.get("planned")) or 0)
            row.interrupted += int(_parse_number(item.fields.get("interrupted")) or 0)
        elif operation == "post_scale_up_rebalance":
            request_key = (role, item.fields.get("migration_id", ""), item.fields.get("request_id", ""))
            if not usable_only or request_key in (usable_records or set()):
                row.completed_overhead += 1
        elif operation == "request_migration_dispatch":
            request_key = (role, item.fields.get("migration_id", ""), item.fields.get("request_id", ""))
            if not usable_only or request_key in (usable_records or set()):
                row.redispatched_overhead += 1

    for filename in ("RolloutRouter.log", "RewardModelRouter.log"):
        log_path = run_dir / filename
        if not log_path.is_file():
            continue
        role = _role_for_source(filename)
        row = counts[role]
        with log_path.open(encoding="utf-8", errors="replace") as log_file:
            for line in log_file:
                timestamp = _parse_timestamp(line)
                if not _in_window(timestamp, origin, start_min, end_min):
                    continue
                prepared = PREPARED_RE.search(line)
                if prepared is not None:
                    row.accepted += int(prepared.group("accepted"))
                    row.skipped += int(prepared.group("skipped"))
                if "Forced planned " in line and " migration:" in line:
                    row.forced += 1
                elif "migration target invalid" in line:
                    row.fallback += 1
    return [counts["RewardModel"], counts["Rollout"]]


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=RUN_DIR")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(raw_path).expanduser().resolve()
    if not label:
        raise argparse.ArgumentTypeError("run label must not be empty")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"run directory does not exist: {path}")
    return label, path


def _format_stat(summary: MetricSummary | None) -> str:
    if summary is None:
        return "-"
    return (
        f"mean={summary.mean:.4f} p50={summary.p50:.4f} "
        f"p95={summary.p95:.4f} max={summary.maximum:.4f} n={summary.count}"
    )


def print_comparison(summaries: Sequence[MetricSummary], labels: Sequence[str]) -> None:
    by_key = {(item.run, item.operation, item.cohort, item.role, item.metric): item for item in summaries}
    metric_keys = sorted({(item.operation, item.cohort, item.role, item.metric) for item in summaries})
    print("\nOverhead statistics (seconds for *_s metrics):")
    print(f"| operation/cohort/role/metric | {labels[0]} | {labels[1]} |")
    print("|---|---|---|")
    for operation, cohort, role, metric in metric_keys:
        left = by_key.get((labels[0], operation, cohort, role, metric))
        right = by_key.get((labels[1], operation, cohort, role, metric))
        key = f"{operation}/{cohort}/{role}/{metric}"
        print(f"| {key} | {_format_stat(left)} | {_format_stat(right)} |")


def print_rebalance_effectiveness(rows: Sequence[RebalanceCounts]) -> None:
    print("\nRebalance effectiveness:")
    print(
        "| run | role | events | planned | accepted | interrupted | skipped | "
        "forced | fallback | redispatched | completed |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        accepted = str(row.accepted) if row.method == "request_level" else "-"
        skipped = str(row.skipped) if row.method == "request_level" else "-"
        forced = str(row.forced) if row.method == "request_level" else "-"
        fallback = str(row.fallback) if row.method == "request_level" else "-"
        print(
            f"| {row.run} | {row.role} | {row.trigger_events} | {row.planned} | "
            f"{accepted} | {row.interrupted} | {skipped} | {forced} | "
            f"{fallback} | {row.redispatched_overhead} | {row.completed_overhead} |"
        )


def print_scaling_actions_per_step(
    rows: Sequence[ScalingActionStepSummary],
) -> None:
    if not rows:
        return
    print("\nActual scaling actions per trainer step:")
    print(
        "| run | step | actions | scale up | scale down | slept instances | "
        "woken instances | instance transitions |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row.run} | {row.step} | {row.actual_actions} | "
            f"{row.scale_up_actions} | {row.scale_down_actions} | "
            f"{row.sleep_instances} | {row.wakeup_instances} | "
            f"{row.instance_transitions} |"
        )


def print_run_windows(windows: Sequence[RunWindow]) -> None:
    print("Input log windows:")
    print("| run | first overhead | last overhead | observed min | selected min |")
    print("|---|---|---|---:|---:|")
    for window in windows:
        print(
            f"| {window.run} | {window.first_timestamp} | {window.last_timestamp} | "
            f"{window.observed_minutes:.2f} | "
            f"{window.selected_start_min:.2f}-{window.selected_end_min:.2f} |"
        )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run,
        metavar="LABEL=RUN_DIR",
        help="Run label and directory; pass exactly twice.",
    )
    parser.add_argument(
        "--start-min",
        type=float,
        default=0.0,
        help="Start of each run's relative-time window (default: 0).",
    )
    parser.add_argument(
        "--end-min",
        type=float,
        default=None,
        help="End of each run's relative-time window (default: full run).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for extracted events and summary CSV files.",
    )
    parser.add_argument(
        "--usable-only",
        action="store_true",
        help="Keep only migration samples with a matching terminal completion record.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.run) != 2:
        raise SystemExit("--run must be passed exactly twice")
    labels = [label for label, _ in args.run]
    if labels[0] == labels[1]:
        raise SystemExit("run labels must be unique")
    if args.start_min < 0:
        raise SystemExit("--start-min must be non-negative")
    if args.end_min is not None and args.end_min < args.start_min:
        raise SystemExit("--end-min must be >= --start-min")

    all_events: list[MetricEvent] = []
    effectiveness: list[RebalanceCounts] = []
    windows: list[RunWindow] = []
    for label, run_dir in args.run:
        parsed = parse_overhead_lines(run_dir)
        events = collect_metric_events(
            label,
            run_dir,
            start_min=args.start_min,
            end_min=args.end_min,
        )
        if args.usable_only:
            events = filter_usable_migration_events(events)
        if not events:
            raise SystemExit(f"no supported overhead events found in {run_dir}")
        windows.append(
            describe_run_window(
                label,
                run_dir,
                parsed,
                start_min=args.start_min,
                end_min=args.end_min,
            )
        )
        all_events.extend(events)
        effectiveness.extend(
            collect_rebalance_counts(
                label,
                run_dir,
                parsed,
                start_min=args.start_min,
                end_min=args.end_min,
                usable_only=args.usable_only,
            )
        )

    summaries = summarize_metric_events(all_events)
    scaling_actions_per_step = summarize_scaling_actions_per_step(all_events)
    print_run_windows(windows)
    print_comparison(summaries, labels)
    print_rebalance_effectiveness(effectiveness)
    print_scaling_actions_per_step(scaling_actions_per_step)

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        _write_csv(output_dir / "overhead_events.csv", [asdict(item) for item in all_events])
        _write_csv(output_dir / "overhead_summary.csv", [asdict(item) for item in summaries])
        _write_csv(output_dir / "run_windows.csv", [asdict(item) for item in windows])
        _write_csv(
            output_dir / "scaling_actions_per_step.csv",
            [asdict(item) for item in scaling_actions_per_step],
        )
        _write_csv(
            output_dir / "rebalance_effectiveness.csv",
            [item.as_row() for item in effectiveness],
        )
        print(f"\nWrote CSV files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
