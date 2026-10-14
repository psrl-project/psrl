#!/usr/bin/env python3
"""Summarize actor, rollout, and GenRM sleep/wakeup latency.

The four default run specifications reproduce the model combinations used in
the elastic deployment comparison.  One completed command is one sample; TP
ranks are not counted separately.

Rollout and GenRM samples use the executor-side wall time spent awaiting the
coordinator command (``sleep_s`` or ``wakeup_s``).  Actor samples use trainer
``total_s``.  Initialization is excluded by selecting the interval from the
first training-batch wait through the next wait after the final completed actor
update, when that next wait exists.
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
TIMESTAMP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)
WAIT_RE = re.compile(r"\[Begin Event\] WAIT - Wait for training batch (?P<batch>\d+)")
UPDATE_END_RE = re.compile(r"\[End Event\] TRAIN - Update actor")
TRAINER_RE = re.compile(
    r"\[ELASTIC_OVERHEAD\] operation=(?P<operation>sleep|wakeup) "
    r"role=Trainer\b(?P<body>.*)"
)
HANDLER_RE = re.compile(
    r"elastic_rm (?P<action>scale_up|scale_down)_handler "
    r"decision_id=(?P<decision_id>\d+) begin task=.*?"
    r"'role_name': <PSRL_Role\.(?P<role>Rollout|RewardModel):"
)
ACTION_OVERHEAD_RE = re.compile(
    r"\[ELASTIC_OVERHEAD\] operation=(?P<action>scale_up|scale_down)\b(?P<body>.*)"
)
SCALING_ACTION_RE = re.compile(
    r"\[ELASTIC_OVERHEAD\] operation=scaling_action\b(?P<body>.*)"
)
PRE_SLEEP_RE = re.compile(
    r"elastic_rm scale_up_handler decision_id=(?P<decision_id>\d+) "
    r"pre_sleep_other count=(?P<count>\d+)"
)
WAKE_TARGETS_RE = re.compile(
    r"elastic_rm scale_up_handler decision_id=(?P<decision_id>\d+) "
    r"combined_wake .* wake_targets=(?P<targets>\[.*\])"
)
SLEEP_TARGETS_RE = re.compile(
    r"elastic_rm scale_down_handler decision_id=(?P<decision_id>\d+) "
    r"sleep_targets=(?P<targets>\[.*\])"
)
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[^\s]+)")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass(frozen=True)
class RunSpec:
    case: str
    policy_model: str
    policy_tp: int
    genrm_model: str
    genrm_tp: int
    genrm_ep: int | None
    run_dir: str


DEFAULT_RUNS = (
    RunSpec(
        case="1",
        policy_model="Qwen2.5-1.5B",
        policy_tp=1,
        genrm_model="GLM-Z1-9B",
        genrm_tp=1,
        genrm_ep=None,
        run_dir="mixed_mode5_overhead_bs_128_share_32_elastic_rl_"
        "Qwen2.5-1.5B_GLM-Z1-9B-0414",
    ),
    RunSpec(
        case="2",
        policy_model="Qwen2.5-7B",
        policy_tp=1,
        genrm_model="Qwen3-30B-A3B",
        genrm_tp=4,
        genrm_ep=4,
        run_dir="mixed_mode5_OVERHEAD_bs_128_share_16_elastic_"
        "Qwen2.5-7B_Qwen3-30B-A3B-Thinking-2507",
    ),
    RunSpec(
        case="3",
        policy_model="Qwen2.5-32B",
        policy_tp=4,
        genrm_model="Qwen3-30B-A3B",
        genrm_tp=4,
        genrm_ep=4,
        run_dir="mixed_mode5_bs_128_share_32_elastic_rl_"
        "Qwen2.5-32B_Qwen3-30B-A3B-Thinking-2507",
    ),
    RunSpec(
        case="4",
        policy_model="Qwen2.5-72B",
        policy_tp=8,
        genrm_model="Qwen3.5-122B-A10B",
        genrm_tp=8,
        genrm_ep=8,
        run_dir="mixed_mode5_overhead_bs_128_share_32_elastic_rl_"
        "Qwen2.5-72B_Qwen3.5-122B-A10B",
    ),
)


@dataclass(frozen=True)
class RunWindow:
    completed_steps: int
    start: datetime
    end: datetime
    start_batch: int
    terminal_batch: int | None


@dataclass(frozen=True)
class TargetInfo:
    role: str
    requested_instances: int | None
    training_step: int | None
    has_pre_wake_other: bool


@dataclass(frozen=True)
class ActionStatus:
    success: bool
    sleep_instances: int
    wakeup_instances: int


@dataclass(frozen=True)
class LatencyEvent:
    case: str
    policy_model: str
    genrm_model: str
    completed_steps: int
    timestamp: str
    component: str
    operation: str
    seconds: float
    instance_count: int | None
    source: str
    scaling_action: str
    decision_id: str


@dataclass(frozen=True)
class LatencySummary:
    case: str
    policy_model: str
    policy_tp: int
    genrm_model: str
    genrm_tp: int
    genrm_ep: int | None
    completed_steps: int
    component: str
    operation: str
    count: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    ddof: int


@dataclass(frozen=True)
class SelectedRun:
    case: str
    policy_model: str
    policy_tp: int
    genrm_model: str
    genrm_tp: int
    genrm_ep: int | None
    completed_steps: int
    window_start: str
    window_end: str
    start_batch: int
    terminal_batch: int | None
    run_path: str


def _timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group("timestamp"), TIMESTAMP_FORMAT)


def _fields(body: str) -> dict[str, str]:
    return {
        match.group("key"): match.group("value").rstrip(",")
        for match in KEY_VALUE_RE.finditer(body)
    }


def _finite_float(raw: str | None) -> float | None:
    if raw is None or not NUMBER_RE.fullmatch(raw):
        return None
    value = float(raw)
    return value if math.isfinite(value) else None


def _integer(raw: str | None) -> int | None:
    value = _finite_float(raw)
    if value is None or not value.is_integer():
        return None
    return int(value)


def parse_run_window(main_log: Path) -> RunWindow:
    waits: list[tuple[datetime, int]] = []
    completed_updates: list[datetime] = []
    with main_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp = _timestamp(line)
            if timestamp is None:
                continue
            wait_match = WAIT_RE.search(line)
            if wait_match is not None:
                waits.append((timestamp, int(wait_match.group("batch"))))
            if UPDATE_END_RE.search(line):
                completed_updates.append(timestamp)

    if not waits:
        raise ValueError(f"no training-batch wait found in {main_log}")
    if not completed_updates:
        raise ValueError(f"no completed actor update found in {main_log}")

    start, start_batch = waits[0]
    last_update = completed_updates[-1]
    terminal_wait = next(
        ((timestamp, batch) for timestamp, batch in waits if timestamp > last_update),
        None,
    )
    end, terminal_batch = terminal_wait or (last_update, None)
    return RunWindow(
        completed_steps=len(completed_updates),
        start=start,
        end=end,
        start_batch=start_batch,
        terminal_batch=terminal_batch,
    )


def _in_window(timestamp: datetime, window: RunWindow) -> bool:
    return window.start <= timestamp <= window.end


def _event(
    spec: RunSpec,
    window: RunWindow,
    timestamp: datetime,
    component: str,
    operation: str,
    seconds: float,
    instance_count: int | None,
    source: str,
    scaling_action: str = "",
    decision_id: str = "",
) -> LatencyEvent:
    return LatencyEvent(
        case=spec.case,
        policy_model=spec.policy_model,
        genrm_model=spec.genrm_model,
        completed_steps=window.completed_steps,
        timestamp=timestamp.strftime(TIMESTAMP_FORMAT),
        component=component,
        operation=operation,
        seconds=seconds,
        instance_count=instance_count,
        source=source,
        scaling_action=scaling_action,
        decision_id=decision_id,
    )


def parse_actor_events(spec: RunSpec, run_path: Path, window: RunWindow) -> list[LatencyEvent]:
    events: list[LatencyEvent] = []
    main_log = run_path / "MainRayTrainer.log"
    with main_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp = _timestamp(line)
            match = TRAINER_RE.search(line)
            if timestamp is None or match is None or not _in_window(timestamp, window):
                continue
            values = _fields(match.group("body"))
            seconds = _finite_float(values.get("total_s"))
            if seconds is None:
                continue
            events.append(
                _event(
                    spec,
                    window,
                    timestamp,
                    component="policy_actor",
                    operation=match.group("operation"),
                    seconds=seconds,
                    instance_count=_integer(values.get("world_size")),
                    source="MainRayTrainer.log:total_s",
                )
            )
    return events


def _parse_executor_metadata(
    executor_log: Path,
) -> tuple[
    dict[tuple[str, str], TargetInfo],
    dict[tuple[str, str], ActionStatus],
    dict[tuple[str, str, str], int],
]:
    targets: dict[tuple[str, str], TargetInfo] = {}
    statuses: dict[tuple[str, str], ActionStatus] = {}
    observed_counts: dict[tuple[str, str, str], int] = {}
    with executor_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            handler_match = HANDLER_RE.search(line)
            if handler_match is not None:
                task_text = line[handler_match.start():]
                num_match = re.search(r"'num_instances': (?P<count>\d+)", task_text)
                step_match = re.search(r"'training_step': (?P<step>-?\d+)", task_text)
                has_pre_wake_other = not bool(
                    re.search(r"'pre_wake_other_preferred': \[\]", task_text)
                )
                targets[(handler_match.group("action"), handler_match.group("decision_id"))] = TargetInfo(
                    role=handler_match.group("role"),
                    requested_instances=(int(num_match.group("count")) if num_match else None),
                    training_step=(int(step_match.group("step")) if step_match else None),
                    has_pre_wake_other=has_pre_wake_other,
                )

            status_match = SCALING_ACTION_RE.search(line)
            if status_match is not None:
                values = _fields(status_match.group("body"))
                action = values.get("action_type")
                decision_id = values.get("decision_id")
                sleep_instances = _integer(values.get("sleep_instances"))
                wakeup_instances = _integer(values.get("wakeup_instances"))
                if (
                    action in {"scale_up", "scale_down"}
                    and decision_id is not None
                    and sleep_instances is not None
                    and wakeup_instances is not None
                ):
                    statuses[(action, decision_id)] = ActionStatus(
                        success=values.get("success") == "True",
                        sleep_instances=sleep_instances,
                        wakeup_instances=wakeup_instances,
                    )

            pre_sleep_match = PRE_SLEEP_RE.search(line)
            if pre_sleep_match is not None:
                observed_counts[("scale_up", pre_sleep_match.group("decision_id"), "sleep")] = int(
                    pre_sleep_match.group("count")
                )
            wake_targets_match = WAKE_TARGETS_RE.search(line)
            if wake_targets_match is not None:
                observed_counts[("scale_up", wake_targets_match.group("decision_id"), "wakeup")] = (
                    wake_targets_match.group("targets").count("'instance_id':")
                )
            sleep_targets_match = SLEEP_TARGETS_RE.search(line)
            if sleep_targets_match is not None:
                observed_counts[("scale_down", sleep_targets_match.group("decision_id"), "sleep")] = (
                    sleep_targets_match.group("targets").count("'instance_id':")
                )
    return targets, statuses, observed_counts


def parse_executor_events(spec: RunSpec, run_path: Path, window: RunWindow) -> list[LatencyEvent]:
    executor_log = run_path / "ElasticExecutor.log"
    targets, statuses, observed_counts = _parse_executor_metadata(executor_log)
    events: list[LatencyEvent] = []
    with executor_log.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            timestamp = _timestamp(line)
            match = ACTION_OVERHEAD_RE.search(line)
            if timestamp is None or match is None or not _in_window(timestamp, window):
                continue
            values = _fields(match.group("body"))
            action = match.group("action")
            decision_id = values.get("decision_id", "")
            key = (action, decision_id)
            target = targets.get(key)
            if target is None:
                raise ValueError(f"missing target-role metadata for {key} in {executor_log}")
            if target.has_pre_wake_other:
                raise ValueError(
                    f"cannot attribute combined multi-role wakeup for {key} in {executor_log}"
                )
            status = statuses.get(key)
            if status is not None and not status.success:
                continue
            target_component = "policy_rollout" if target.role == "Rollout" else "genrm"
            other_component = "genrm" if target.role == "Rollout" else "policy_rollout"

            if action == "scale_up":
                sleep_s = _finite_float(values.get("sleep_s"))
                wakeup_s = _finite_float(values.get("wakeup_s"))
                sleep_count = (
                    status.sleep_instances
                    if status is not None
                    else observed_counts.get((action, decision_id, "sleep"), target.requested_instances)
                )
                wake_count = (
                    status.wakeup_instances
                    if status is not None
                    else observed_counts.get((action, decision_id, "wakeup"), target.requested_instances)
                )
                if sleep_s is not None and sleep_s > 0 and (status is None or sleep_count > 0):
                    events.append(
                        _event(
                            spec, window, timestamp, other_component, "sleep", sleep_s,
                            sleep_count, "ElasticExecutor.log:sleep_s", action, decision_id,
                        )
                    )
                if wakeup_s is not None and wakeup_s > 0 and (status is None or wake_count > 0):
                    events.append(
                        _event(
                            spec, window, timestamp, target_component, "wakeup", wakeup_s,
                            wake_count, "ElasticExecutor.log:wakeup_s", action, decision_id,
                        )
                    )
            else:
                sleep_s = _finite_float(values.get("sleep_s"))
                sleep_count = (
                    status.sleep_instances
                    if status is not None
                    else observed_counts.get((action, decision_id, "sleep"), target.requested_instances)
                )
                if sleep_s is not None and sleep_s > 0 and (status is None or sleep_count > 0):
                    events.append(
                        _event(
                            spec, window, timestamp, target_component, "sleep", sleep_s,
                            sleep_count, "ElasticExecutor.log:sleep_s", action, decision_id,
                        )
                    )
    return events


def collect_run(
    spec: RunSpec,
    logs_root: Path,
    min_steps: int,
) -> tuple[SelectedRun, list[LatencyEvent]]:
    run_path = logs_root / spec.run_dir
    required = ("MainRayTrainer.log", "ElasticExecutor.log")
    missing = [name for name in required if not (run_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing {missing} under {run_path}")
    window = parse_run_window(run_path / "MainRayTrainer.log")
    if window.completed_steps <= min_steps:
        raise ValueError(
            f"case {spec.case} has {window.completed_steps} completed steps; "
            f"expected > {min_steps}: {run_path}"
        )
    events = parse_actor_events(spec, run_path, window)
    events.extend(parse_executor_events(spec, run_path, window))
    events.sort(key=lambda item: (item.timestamp, item.component, item.operation))
    selected = SelectedRun(
        case=spec.case,
        policy_model=spec.policy_model,
        policy_tp=spec.policy_tp,
        genrm_model=spec.genrm_model,
        genrm_tp=spec.genrm_tp,
        genrm_ep=spec.genrm_ep,
        completed_steps=window.completed_steps,
        window_start=window.start.strftime(TIMESTAMP_FORMAT),
        window_end=window.end.strftime(TIMESTAMP_FORMAT),
        start_batch=window.start_batch,
        terminal_batch=window.terminal_batch,
        run_path=str(run_path.resolve()),
    )
    return selected, events


def summarize_events(
    specs: Sequence[RunSpec],
    selected_runs: Sequence[SelectedRun],
    events: Iterable[LatencyEvent],
    ddof: int,
) -> list[LatencySummary]:
    if ddof not in {0, 1}:
        raise ValueError("ddof must be 0 or 1")
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for event in events:
        grouped[(event.case, event.component, event.operation)].append(event.seconds)
    spec_by_case = {spec.case: spec for spec in specs}
    run_by_case = {run.case: run for run in selected_runs}
    summaries: list[LatencySummary] = []
    for case in sorted(spec_by_case, key=int):
        spec = spec_by_case[case]
        selected = run_by_case[case]
        for component in ("policy_rollout", "policy_actor", "genrm"):
            for operation in ("wakeup", "sleep"):
                values = grouped.get((case, component, operation), [])
                if not values:
                    raise ValueError(f"no {component} {operation} events found for case {case}")
                if ddof == 1 and len(values) < 2:
                    raise ValueError(f"ddof=1 requires at least two samples for case {case}")
                std_s = statistics.pstdev(values) if ddof == 0 else statistics.stdev(values)
                summaries.append(
                    LatencySummary(
                        case=case,
                        policy_model=spec.policy_model,
                        policy_tp=spec.policy_tp,
                        genrm_model=spec.genrm_model,
                        genrm_tp=spec.genrm_tp,
                        genrm_ep=spec.genrm_ep,
                        completed_steps=selected.completed_steps,
                        component=component,
                        operation=operation,
                        count=len(values),
                        mean_s=statistics.fmean(values),
                        std_s=std_s,
                        min_s=min(values),
                        max_s=max(values),
                        ddof=ddof,
                    )
                )
    return summaries


def _write_csv(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(asdict(rows[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _format_cell(summary: LatencySummary) -> str:
    return f"{summary.mean_s:.3f} +/- {summary.std_s:.3f} (n={summary.count})"


def render_markdown(
    specs: Sequence[RunSpec],
    selected_runs: Sequence[SelectedRun],
    summaries: Sequence[LatencySummary],
) -> str:
    ddof = summaries[0].ddof
    std_label = "population" if ddof == 0 else "sample"
    by_key = {
        (item.case, item.component, item.operation): item
        for item in summaries
    }
    selected_by_case = {item.case: item for item in selected_runs}
    lines = [
        "# Elastic sleep/wakeup latency",
        "",
        f"Times are seconds and shown as mean +/- {std_label} standard deviation "
        f"(ddof={ddof}). One completed command is one sample.",
        "",
        "| Case | Policy model | GenRM model | Completed steps | Component | Wakeup (s) | Sleep (s) |",
        "|---:|---|---|---:|---|---:|---:|",
    ]
    component_labels = {
        "policy_rollout": "Policy rollout",
        "policy_actor": "Policy actor",
        "genrm": "GenRM",
    }
    for spec in specs:
        steps = selected_by_case[spec.case].completed_steps
        for component in ("policy_rollout", "policy_actor", "genrm"):
            wakeup = by_key[(spec.case, component, "wakeup")]
            sleep = by_key[(spec.case, component, "sleep")]
            lines.append(
                f"| {spec.case} | {spec.policy_model} (TP={spec.policy_tp}) | "
                f"{spec.genrm_model} (TP={spec.genrm_tp}"
                f"{f', EP={spec.genrm_ep}' if spec.genrm_ep is not None else ''}) | "
                f"{steps} | {component_labels[component]} | "
                f"{_format_cell(wakeup)} | {_format_cell(sleep)} |"
            )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Window: first training-batch wait through the next wait after the final completed actor update.",
            "- Policy rollout and GenRM: executor wall time awaiting coordinator "
            "lifecycle calls (`sleep_s`/`wakeup_s`).",
            "- Policy actor: trainer lifecycle wall time (`total_s`).",
            "- Exactly zero-duration fields, explicitly logged zero-transition actions, "
            "and initialization events are excluded.",
        ]
    )
    return "\n".join(lines) + "\n"


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=repo_root / "logs" / "verl_deployment_modes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "analysis" / "elastic_sleep_wake",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=20,
        help="Require strictly more than this many completed actor updates.",
    )
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_runs: list[SelectedRun] = []
    events: list[LatencyEvent] = []
    for spec in DEFAULT_RUNS:
        selected, run_events = collect_run(spec, args.logs_root, args.min_steps)
        selected_runs.append(selected)
        events.extend(run_events)
    summaries = summarize_events(DEFAULT_RUNS, selected_runs, events, args.ddof)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "selected_runs.csv", selected_runs)
    _write_csv(args.output_dir / "events.csv", events)
    _write_csv(args.output_dir / "summary.csv", summaries)
    report = render_markdown(DEFAULT_RUNS, selected_runs, summaries)
    (args.output_dir / "SUMMARY.md").write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Wrote {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
