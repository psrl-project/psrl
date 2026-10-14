#!/usr/bin/env python3
"""Extract JSONL inputs for offline elastic request-level simulation.

The extractor joins three existing log families at the candidate-evaluation timestamp:

* ``ScalingPolicy.log`` supplies cycle ids, candidate actions, per-instance
  request/token aggregates, router backlog aggregates, policy values, and the
  reference simulation timings. Deterministic request lengths are constructed
  from those aggregates so timing complexity matches the logged cycle.
* ``StatCollector*.log`` supplies fallback request rows plus model versions and
  engine metadata. Its independently sampled request state is not used when
  the policy log contains the cycle's current-state aggregates.
* ``ElasticMonitor.log`` supplies awake/asleep status and pool metadata.

The output is deliberately plain JSONL.  It contains no Python reprs or enum
values and is intended to be consumed by a future C++ simulator.  Router-only
request metadata that is not present in the old logs is marked as synthetic:
all active instances are eligible, request order is deterministic, and token
totals are distributed as evenly as possible across the logged request count.

Examples:

    python scripts/extract_elastic_simulation_inputs.py \
        logs/verl_deployment_modes/<run> --max-cycles 20 \
        -o analysis/<run>/simulation_inputs.jsonl

    python scripts/extract_elastic_simulation_inputs.py \
        logs/verl_deployment_modes/<run> --start-cycle 100 --end-cycle 200

    python scripts/extract_elastic_simulation_inputs.py \
        logs/verl_deployment_modes/<run> --cycles 101,205,309

The file format is schema version 1.  Numeric fields use JSON integers or
finite JSON numbers; missing timestamps are represented by ``null``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = (
    REPO_ROOT / "logs/verl_deployment_modes/"
    "dapo_mode5_bs_128_share_16_elastic_staleness_2_enable_trainer_pool_"
    "Qwen2.5-7B_Qwen3-30B-A3B-Thinking-2507_val"
)
DEFAULT_ROLLOUT_COST_MODEL = REPO_ROOT / "psrl/trainer/config/cost_model/qwen2.5_7b.json"
DEFAULT_RM_COST_MODEL = REPO_ROOT / "psrl/trainer/config/cost_model/qwen3_30b_a3b_thinking_2507.json"

ROLE_ORDER = ("Rollout", "RewardModel")
LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S,%f"
LOG_TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
ISO_TIMESTAMP_RE = re.compile(r"'timestamp':\s*'(?P<timestamp>[^']+)'|\"timestamp\":\s*\"(?P<json_timestamp>[^\"]+)\"")
SNAPSHOT_MARKER = "Snapshot (model version"
MODEL_VERSION_RE = re.compile(r"Snapshot \(model version\s+(?P<version>-?\d+)\)")
FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
PROMPT_SECTION_RE = re.compile(r"'req_id_to_prompt_token_num':\s*\{(.*?)\},\s*'req_id_to_response_token_num'")
RESPONSE_SECTION_RE = re.compile(r"'req_id_to_response_token_num':\s*\{(.*?)\},\s*'req_id_in_waiting'")
WAITING_SECTION_RE = re.compile(r"'req_id_in_waiting':\s*\[(.*?)\],\s*'num_running_reqs'")
MAP_ENTRY_RE = re.compile(r"['\"](?P<key>[^'\"]+)['\"]\s*:\s*(?P<value>-?\d+)")
WAITING_ID_RE = re.compile(r"['\"](?P<key>[^'\"]+)['\"]")
NUM_RUNNING_RE = re.compile(r"'num_running_reqs':\s*(?P<value>\d+)")
NUM_WAITING_RE = re.compile(r"'num_waiting_reqs':\s*(?P<value>\d+)")
KV_CACHE_RE = re.compile(rf"'kv_cache_usage':\s*(?:np\.float64\()?\s*(?P<value>{FLOAT_RE})")
THROUGHPUT_RE = re.compile(rf"'generation_throughput':\s*(?:np\.float64\()?\s*(?P<value>{FLOAT_RE})")

ROLLOUT_STAT_RE = re.compile(r"^StatCollector_I(?P<instance>\d+)\.log$")
RM_STAT_RE = re.compile(r"^StatCollector_RM_(?P<model>.+)_I(?P<instance>\d+)\.log$")

CYCLE_BEGIN_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*"
    r"elastic_rm_policy ========== cycle (?P<cycle>\d+) BEGIN =========="
)
CYCLE_BODY_RE = re.compile(r"elastic_rm_policy cycle=(?P<cycle>\d+) \| (?P<body>.*)$")
EVALUATION_TIMING_RE = re.compile(r"elastic_rm_policy request_level_candidate_evaluation \| (?P<body>.*)$")
CANDIDATE_RE = re.compile(
    r"\* candidate\[(?P<index>\d+)\]\s+"
    r"(?P<action_type>scale_up|scale_down)\s+"
    r"(?P<role_name>[^/\s]+)/(?P<model_name>\S+)\s+"
    r"num_instances=(?P<num_instances>\d+)\s+"
    r"preferred=(?P<preferred>\[[^\]]*\])\s+"
    r"pre_wake=(?P<pre_wake>.*?)\s+pre_sleep=(?P<pre_sleep>.*)$"
)
ROLE_SUMMARY_RE = re.compile(
    rf"(?P<role>rollout|rm)_n=(?P<n>\d+)\s+"
    rf"(?P=role)_total_req=(?P<total_req>{FLOAT_RE})\s+"
    rf"(?P=role)_total_tok=(?P<total_tok>\d+)\s+"
    rf"(?P=role)_role_tp=(?P<role_tp>{FLOAT_RE})\s+"
    rf"(?P=role)_router_req=(?P<router_req>{FLOAT_RE})\s+"
    rf"(?P=role)_router_tok=(?P<router_tok>\d+)"
)
ROLE_INSTANCES_RE = re.compile(r"\*\s+(?P<role>rollout|rm)_instances=\[(?P<body>.*)\]\s*$")
ACCEPTED_ACTION_RE = re.compile(
    r">>> OUTCOME: action \| reason=itl_best_(?:scale_up|scale_down)_"
    r"(?P<role>Rollout|RewardModel)"
)
ROLE_INSTANCE_RE = re.compile(
    rf"(?:^|;\s*)id(?P<instance>\d+):"
    rf"req=(?P<requests>{FLOAT_RE}),"
    rf"tok=(?P<tokens>\d+),"
    rf"tp=(?P<throughput>{FLOAT_RE})"
)

MONITOR_MARKER = "Instance current Status:"

KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>'[^']*'|\"[^\"]*\"|[^\s]+)")


def parse_log_timestamp(raw: str) -> datetime:
    return datetime.strptime(raw, LOG_TIMESTAMP_FORMAT)


def parse_iso_timestamp(raw: str) -> datetime | None:
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1]
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def timestamp_from_snapshot_line(line: str, fallback: datetime) -> datetime:
    match = ISO_TIMESTAMP_RE.search(line)
    if match is None:
        return fallback
    raw = match.group("timestamp") or match.group("json_timestamp")
    return parse_iso_timestamp(raw) or fallback


def parse_number(raw: str) -> int | float | str | None:
    value = raw.strip().strip("'\"")
    if value.lower() in {"none", "null"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if re.fullmatch(r"[+-]?\d+", value):
            return int(value)
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    except ValueError:
        return value


def parse_kv_fields(body: str) -> dict[str, int | float | str | bool | None]:
    return {match.group("key"): parse_number(match.group("value")) for match in KV_RE.finditer(body)}


def parse_int_map(section: str | None) -> dict[str, int]:
    if not section:
        return {}
    result: dict[str, int] = {}
    for match in MAP_ENTRY_RE.finditer(section):
        result[match.group("key")] = int(match.group("value"))
    return result


def parse_waiting_ids(section: str | None) -> frozenset[str]:
    if not section:
        return frozenset()
    return frozenset(match.group("key") for match in WAITING_ID_RE.finditer(section))


@dataclass(frozen=True)
class StatSnapshot:
    timestamp: datetime
    model_version: int
    prompt_tokens: dict[str, int]
    response_tokens: dict[str, int]
    waiting_ids: frozenset[str]
    running_count: int
    waiting_count: int
    kv_cache_usage: float
    generation_throughput: float

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.prompt_tokens) | set(self.response_tokens)))

    def request_length(self, request_id: str) -> int:
        return max(0, self.prompt_tokens.get(request_id, 0) + self.response_tokens.get(request_id, 0))


def parse_stat_snapshot_line(line: str) -> StatSnapshot | None:
    if SNAPSHOT_MARKER not in line:
        return None
    log_match = LOG_TIMESTAMP_RE.match(line)
    model_match = MODEL_VERSION_RE.search(line)
    if log_match is None or model_match is None:
        return None
    prompt_match = PROMPT_SECTION_RE.search(line)
    response_match = RESPONSE_SECTION_RE.search(line)
    waiting_match = WAITING_SECTION_RE.search(line)
    running_match = NUM_RUNNING_RE.search(line)
    waiting_count_match = NUM_WAITING_RE.search(line)
    kv_match = KV_CACHE_RE.search(line)
    throughput_match = THROUGHPUT_RE.search(line)
    if not (running_match and waiting_count_match and kv_match and throughput_match):
        return None
    try:
        timestamp = timestamp_from_snapshot_line(line, parse_log_timestamp(log_match.group("timestamp")))
        return StatSnapshot(
            timestamp=timestamp,
            model_version=int(model_match.group("version")),
            prompt_tokens=parse_int_map(prompt_match.group(1) if prompt_match else None),
            response_tokens=parse_int_map(response_match.group(1) if response_match else None),
            waiting_ids=parse_waiting_ids(waiting_match.group(1) if waiting_match else None),
            running_count=int(running_match.group("value")),
            waiting_count=int(waiting_count_match.group("value")),
            kv_cache_usage=float(kv_match.group("value")),
            generation_throughput=float(throughput_match.group("value")),
        )
    except (TypeError, ValueError):
        return None


class StatCursor:
    """Stream one StatCollector file and retain only the latest point."""

    def __init__(self, path: Path, role: str, model: str, instance_id: int):
        self.path = path
        self.role = role
        self.model = model
        self.instance_id = instance_id
        self._file: TextIO = path.open("r", encoding="utf-8", errors="replace")
        self.current: StatSnapshot | None = None
        self._next: StatSnapshot | None = self._read_next()

    def _read_next(self) -> StatSnapshot | None:
        for line in self._file:
            point = parse_stat_snapshot_line(line)
            if point is not None:
                return point
        return None

    def advance_to(self, timestamp: datetime) -> StatSnapshot | None:
        while self._next is not None and self._next.timestamp <= timestamp:
            self.current = self._next
            self._next = self._read_next()
        return self.current

    def close(self) -> None:
        self._file.close()


@dataclass(frozen=True)
class MonitorPoint:
    log_timestamp: datetime
    snapshot_timestamp: datetime | None
    role: str
    model: str
    instance_id: int
    status: str
    pool_id: str | None
    running_count: int
    waiting_count: int
    kv_cache_usage: float
    generation_throughput: float


class MonitorCursor:
    """Stream ElasticMonitor records in log order."""

    def __init__(self, path: Path):
        self.path = path
        self._file: TextIO | None = path.open("r", encoding="utf-8", errors="replace") if path.is_file() else None
        self._next: MonitorPoint | None = self._read_next()
        self.latest: dict[tuple[str, str, int], MonitorPoint] = {}

    def _read_next(self) -> MonitorPoint | None:
        if self._file is None:
            return None
        for line in self._file:
            if MONITOR_MARKER not in line:
                continue
            log_match = LOG_TIMESTAMP_RE.match(line)
            if log_match is None:
                continue
            raw = line.split(MONITOR_MARKER, 1)[1].strip()
            try:
                value = eval_literal_dict(raw)
            except (SyntaxError, ValueError):
                continue
            try:
                role = str(value["role"])
                model = str(value["model"])
                instance_id = int(value["instance"])
                status = str(value.get("status", "UNKNOWN"))
                pool_id = value.get("pool_id")
                point = MonitorPoint(
                    log_timestamp=parse_log_timestamp(log_match.group("timestamp")),
                    snapshot_timestamp=parse_iso_timestamp(str(value.get("ts", ""))),
                    role=role,
                    model=model,
                    instance_id=instance_id,
                    status=status,
                    pool_id=None if pool_id is None else str(pool_id),
                    running_count=int(value.get("running", 0)),
                    waiting_count=int(value.get("waiting", 0)),
                    kv_cache_usage=float(value.get("kv_cache", 0.0)),
                    generation_throughput=float(value.get("throughput", 0.0)),
                )
            except (KeyError, TypeError, ValueError):
                continue
            return point
        return None

    def advance_to(self, timestamp: datetime) -> None:
        while self._next is not None and self._next.log_timestamp <= timestamp:
            point = self._next
            self.latest[(point.role, point.model, point.instance_id)] = point
            self._next = self._read_next()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def eval_literal_dict(raw: str) -> dict[str, object]:
    """Parse the primitive status dict without importing application modules."""
    import ast

    value = ast.literal_eval(raw)
    if not isinstance(value, dict):
        raise ValueError("monitor record is not a dict")
    return value


@dataclass(frozen=True)
class Candidate:
    index: int
    action_type: str
    role_name: str
    model_name: str
    num_instances: int
    preferred_instance_ids: tuple[int, ...]
    pre_wake: tuple[dict[str, int], ...]
    pre_sleep: tuple[dict[str, int], ...]
    role_plans: dict[str, dict[str, object]]


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in re.findall(r"-?\d+", raw))


def parse_role_refs(raw: str) -> tuple[dict[str, int], ...]:
    if raw.strip() in {"None", "null", ""}:
        return ()
    refs: list[dict[str, int]] = []
    patterns = (
        re.compile(
            r"role_name['\"]?\s*:\s*<PSRL_Role\.(?P<role>[A-Za-z_][A-Za-z0-9_]*)[^>]*>.*?"
            r"instance_id['\"]?\s*:\s*(?P<instance>\d+)"
        ),
        re.compile(
            r"role_name['\"]?\s*:\s*['\"](?P<role>[A-Za-z_][A-Za-z0-9_]*)['\"].*?"
            r"instance_id['\"]?\s*:\s*(?P<instance>\d+)"
        ),
    )
    for pattern in patterns:
        refs.extend(
            {
                "role_name": match.group("role"),
                "instance_id": int(match.group("instance")),
            }
            for match in pattern.finditer(raw)
        )
        if refs:
            break
    return tuple(refs)


def build_role_plans(
    action_type: str,
    role_name: str,
    preferred: tuple[int, ...],
    pre_wake: tuple[dict[str, int], ...],
    pre_sleep: tuple[dict[str, int], ...],
) -> dict[str, dict[str, object]]:
    wake: dict[str, set[int]] = {role: set() for role in ROLE_ORDER}
    sleep: dict[str, set[int]] = {role: set() for role in ROLE_ORDER}
    if role_name in wake:
        if action_type == "scale_up":
            wake[role_name].update(preferred)
        else:
            sleep[role_name].update(preferred)
    for ref in pre_wake:
        if ref["role_name"] in wake:
            wake[ref["role_name"]].add(ref["instance_id"])
    for ref in pre_sleep:
        if ref["role_name"] in sleep:
            sleep[ref["role_name"]].add(ref["instance_id"])
    return {
        role: {
            "wake_instance_ids": sorted(wake[role]),
            "sleep_instance_ids": sorted(sleep[role]),
            "primary_scale_up": bool(action_type == "scale_up" and role == role_name),
        }
        for role in ROLE_ORDER
    }


def parse_candidate(body: str) -> Candidate | None:
    match = CANDIDATE_RE.search(body)
    if match is None:
        return None
    pre_wake = parse_role_refs(match.group("pre_wake"))
    pre_sleep = parse_role_refs(match.group("pre_sleep"))
    action_type = match.group("action_type")
    role_name = match.group("role_name")
    preferred = parse_int_list(match.group("preferred"))
    return Candidate(
        index=int(match.group("index")),
        action_type=action_type,
        role_name=role_name,
        model_name=match.group("model_name"),
        num_instances=int(match.group("num_instances")),
        preferred_instance_ids=preferred,
        pre_wake=pre_wake,
        pre_sleep=pre_sleep,
        role_plans=build_role_plans(action_type, role_name, preferred, pre_wake, pre_sleep),
    )


@dataclass
class Cycle:
    cycle_id: int
    timestamp: datetime
    candidates: list[Candidate] = field(default_factory=list)
    policy_fields: dict[str, int | float | str | bool | None] = field(default_factory=dict)
    role_summary: dict[str, dict[str, int | float]] = field(default_factory=dict)
    role_instance_ids: dict[str, tuple[int, ...]] = field(default_factory=dict)
    role_instance_loads: dict[str, tuple[RoleInstanceLoad, ...]] = field(default_factory=dict)
    accepted_action_role: str | None = None
    evaluation_timing: dict[str, float] | None = None
    evaluation_timing_timestamp: datetime | None = None

    @property
    def simulation_timestamp(self) -> datetime:
        """Best available timestamp for the request-level simulation input."""
        if self.evaluation_timing_timestamp is None:
            return self.timestamp
        elapsed_s = max(0.0, float((self.evaluation_timing or {}).get("elapsed_s", 0.0)))
        return self.evaluation_timing_timestamp - timedelta(seconds=elapsed_s)


def numeric_fields(fields: dict[str, int | float | str | bool | None]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(float(value)):
            result[key] = float(value)
    return result


def parse_evaluation_timing(line: str) -> tuple[datetime, dict[str, float]] | None:
    match = EVALUATION_TIMING_RE.search(line)
    timestamp_match = LOG_TIMESTAMP_RE.match(line)
    if match is None or timestamp_match is None:
        return None
    fields = numeric_fields(parse_kv_fields(match.group("body")))
    wanted = {
        key: fields[key]
        for key in (
            "candidate_scoring_s",
            "rebalance_simulation_s",
            "router_simulation_s",
            "simulation_input_preparation_s",
            "elapsed_s",
        )
        if key in fields
    }
    return parse_log_timestamp(timestamp_match.group("timestamp")), wanted


def parse_role_summary(body: str) -> tuple[str, dict[str, int | float]] | None:
    match = ROLE_SUMMARY_RE.search(body)
    if match is None:
        return None
    role = "Rollout" if match.group("role") == "rollout" else "RewardModel"
    return role, {
        "instance_count": int(match.group("n")),
        "total_request_count": float(match.group("total_req")),
        "total_token_count": int(match.group("total_tok")),
        "role_throughput": float(match.group("role_tp")),
        "router_request_count": float(match.group("router_req")),
        "router_token_count": int(match.group("router_tok")),
    }


@dataclass(frozen=True)
class RoleInstanceLoad:
    instance_id: int
    request_count: float
    token_count: int
    throughput: float


def parse_role_instance_loads(body: str) -> tuple[str, tuple[RoleInstanceLoad, ...]] | None:
    match = ROLE_INSTANCES_RE.search(body)
    if match is None:
        return None
    role = "Rollout" if match.group("role") == "rollout" else "RewardModel"
    return role, tuple(
        RoleInstanceLoad(
            instance_id=int(item.group("instance")),
            request_count=float(item.group("requests")),
            token_count=int(item.group("tokens")),
            throughput=float(item.group("throughput")),
        )
        for item in ROLE_INSTANCE_RE.finditer(match.group("body"))
    )


def parse_role_instance_ids(body: str) -> tuple[str, tuple[int, ...]] | None:
    parsed = parse_role_instance_loads(body)
    if parsed is None:
        return None
    return parsed[0], tuple(load.instance_id for load in parsed[1])


def iter_policy_cycles(path: Path) -> Iterator[Cycle]:
    pending_timing: tuple[datetime, dict[str, float]] | None = None
    current: Cycle | None = None
    with path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line in log_file:
            if timing := parse_evaluation_timing(line):
                pending_timing = timing
                continue
            begin = CYCLE_BEGIN_RE.search(line)
            if begin is not None:
                if current is not None:
                    yield current
                cycle_timing = pending_timing
                pending_timing = None
                current = Cycle(
                    cycle_id=int(begin.group("cycle")),
                    timestamp=parse_log_timestamp(begin.group("timestamp")),
                    evaluation_timing=(cycle_timing[1] if cycle_timing else None),
                    evaluation_timing_timestamp=(cycle_timing[0] if cycle_timing else None),
                )
                continue
            if current is None:
                continue
            body_match = CYCLE_BODY_RE.search(line)
            if body_match is None:
                continue
            body = body_match.group("body")
            if candidate := parse_candidate(body):
                current.candidates.append(candidate)
                continue
            if role_summary := parse_role_summary(body):
                current.role_summary[role_summary[0]] = role_summary[1]
                continue
            if role_instances := parse_role_instance_loads(body):
                role, loads = role_instances
                current.role_instance_loads[role] = loads
                current.role_instance_ids[role] = tuple(load.instance_id for load in loads)
                continue
            if accepted_action := ACCEPTED_ACTION_RE.search(body):
                current.accepted_action_role = accepted_action.group("role")
                continue
            if body.startswith("current_throughput=") or "vllm_current_queue_scope=" in body:
                current.policy_fields.update(parse_kv_fields(body))
        if current is not None:
            yield current


@dataclass(frozen=True)
class CostModel:
    path: str
    tp_pp: str
    throughput_params: tuple[float, float, float, float]
    route_cost_params: tuple[float, float, float, float, float]
    source: str


def load_cost_model(path: Path, tp_pp: str) -> CostModel:
    default_throughput = (0.0, 1.0, 1.0, 0.0)
    default_route = (0.0, 0.0, 1.0, 0.0, 0.0)
    if not path.is_file():
        return CostModel(str(path), tp_pp, default_throughput, default_route, "default_missing_file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CostModel(str(path), tp_pp, default_throughput, default_route, "default_invalid_file")
    entry = payload.get(tp_pp) if isinstance(payload, dict) else None
    if not isinstance(entry, dict) and isinstance(payload, dict):
        entry = next((value for value in payload.values() if isinstance(value, dict)), None)
    if not isinstance(entry, dict):
        return CostModel(str(path), tp_pp, default_throughput, default_route, "default_missing_entry")

    def value(name: str, fallback: float) -> float:
        raw = entry.get(name, fallback)
        try:
            parsed = float(raw)
            return parsed if math.isfinite(parsed) else fallback
        except (TypeError, ValueError):
            return fallback

    route = (
        value("other_threshold", 0.0),
        value("other_latency_b", 0.0),
        value("other_latency_k", 1.0),
        value("attn_latency_b", 0.0),
        value("attn_latency_k", 0.0),
    )
    throughput = (
        value("A", route[4]),
        value("B", route[0]),
        value("C", route[2]),
        value("D", route[3]),
    )
    return CostModel(str(path), tp_pp, throughput, route, "cost_model")


def classify_stat_file(path: Path, rollout_model: str) -> tuple[str, str, int] | None:
    if match := ROLLOUT_STAT_RE.match(path.name):
        return "Rollout", rollout_model, int(match.group("instance"))
    if match := RM_STAT_RE.match(path.name):
        return "RewardModel", match.group("model"), int(match.group("instance"))
    return None


def discover_stat_cursors(log_dir: Path, rollout_model: str) -> list[StatCursor]:
    cursors: list[StatCursor] = []
    for path in sorted(log_dir.glob("StatCollector*.log")):
        metadata = classify_stat_file(path, rollout_model)
        if metadata is None:
            continue
        cursors.append(StatCursor(path, *metadata))
    return cursors


def format_timestamp(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat(timespec="milliseconds")


def safe_float(value: float | int) -> float:
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def age_seconds(reference: datetime, timestamp: datetime | None) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, (reference - timestamp).total_seconds())


def make_request_row(
    request_id: str,
    seq_len: int,
    source_instance_id: int | None,
    is_waiting: bool,
    route_order: int,
) -> dict[str, object]:
    return {
        "request_id": str(request_id),
        "seq_len": max(0, int(seq_len)),
        "source_instance_id": source_instance_id,
        "is_waiting": bool(is_waiting),
        "route_order": int(route_order),
        "routing_priority": [int(route_order)],
        # null means all instances in the evaluator; this is intentional for
        # the timing benchmark and avoids Python-specific placeholder values.
        "eligible_instance_ids": None,
        "fallback_instance_ids": None,
        "candidate_priorities": [],
        "synthetic_routing_metadata": True,
    }


def distribute_token_lengths(total_tokens: int, request_count: int, default_length: int) -> list[int]:
    """Construct deterministic lengths while preserving the logged aggregate."""
    if request_count <= 0:
        return []
    if total_tokens <= 0:
        return [max(0, default_length)] * request_count
    base, remainder = divmod(total_tokens, request_count)
    return [base + (1 if index < remainder else 0) for index in range(request_count)]


def role_limits(args: argparse.Namespace, role: str) -> dict[str, object]:
    if role == "Rollout":
        return {
            "max_concurrent_requests": args.rollout_max_concurrent_requests,
            "waiting_admission_cap": None,
            "delta_throughput_threshold": args.rollout_delta_threshold,
            "itl_max_itl": None,
        }
    return {
        "max_concurrent_requests": args.rm_max_concurrent_requests,
        "waiting_admission_cap": args.rm_waiting_admission_cap,
        "delta_throughput_threshold": args.rm_delta_threshold,
        "itl_max_itl": args.rm_max_itl,
    }


def build_role_snapshot(
    cycle: Cycle,
    role: str,
    role_cursors: list[StatCursor],
    monitor: MonitorCursor,
    cost_model: CostModel,
    args: argparse.Namespace,
) -> dict[str, object]:
    simulation_timestamp = cycle.simulation_timestamp
    monitor.advance_to(simulation_timestamp)
    policy_awake_ids = (
        frozenset(cycle.role_instance_ids[role]) if role in cycle.role_instance_ids else None
    )
    policy_loads = {
        load.instance_id: load for load in cycle.role_instance_loads.get(role, ())
    }
    role_cursors = sorted(role_cursors, key=lambda cursor: cursor.instance_id)
    # ElasticExecutor deliberately excludes validation-only rollout workers.
    # When ElasticMonitor is present, its instance set is the authoritative
    # policy pool; StatCollector also contains files for validation workers.
    monitored_ids = {
        instance_id
        for (point_role, point_model, instance_id) in monitor.latest
        if point_role == role and (not role_cursors or point_model == role_cursors[0].model)
    }
    if monitored_ids:
        role_cursors = [cursor for cursor in role_cursors if cursor.instance_id in monitored_ids]

    raw_instances: list[tuple[StatCursor, StatSnapshot | None, MonitorPoint | None]] = []
    for cursor in role_cursors:
        raw_point = cursor.advance_to(simulation_timestamp)
        monitor_point = monitor.latest.get((role, cursor.model, cursor.instance_id))
        # Once an instance is asleep/training, its scheduler state is not part
        # of the request-level policy snapshot.  Do not reuse an old
        # StatCollector point from before the transition.
        point = raw_point
        if policy_awake_ids is not None:
            if cursor.instance_id not in policy_awake_ids:
                point = None
        elif monitor_point is not None and monitor_point.status not in {"AWAKEN", "AWAKE", "RUNNING"}:
            point = None
        raw_instances.append((cursor, point, monitor_point))

    summary = cycle.role_summary.get(role, {})
    pending_count = max(0, int(round(float(summary.get("router_request_count", 0.0)))))
    pending_tokens = max(0, int(summary.get("router_token_count", 0)))
    if args.max_pending_requests is not None:
        pending_count = min(pending_count, args.max_pending_requests)

    pending_lengths = distribute_token_lengths(
        pending_tokens,
        pending_count,
        args.default_pending_seq_len,
    )
    pending_length_source = (
        "scaling_policy_router_aggregate"
        if pending_count and pending_tokens
        else "default_pending_seq_len"
    )

    pending_requests = [
        make_request_row(
            request_id=f"synthetic-{role.lower()}-c{cycle.cycle_id}-p{index}",
            seq_len=seq_len,
            source_instance_id=None,
            is_waiting=True,
            route_order=index,
        )
        for index, seq_len in enumerate(pending_lengths)
    ]

    instances: list[dict[str, object]] = []
    next_route_order = len(pending_requests)
    for cursor, point, monitor_point in raw_instances:
        request_rows: list[dict[str, object]] = []
        policy_load = policy_loads.get(cursor.instance_id)
        if policy_load is not None:
            policy_request_count = max(0, int(round(policy_load.request_count)))
            policy_lengths = distribute_token_lengths(
                policy_load.token_count,
                policy_request_count,
                args.default_pending_seq_len,
            )
            for index, seq_len in enumerate(policy_lengths):
                request_rows.append(
                    make_request_row(
                        request_id=(
                            f"synthetic-{role.lower()}-c{cycle.cycle_id}-"
                            f"i{cursor.instance_id}-r{index}"
                        ),
                        seq_len=seq_len,
                        source_instance_id=cursor.instance_id,
                        is_waiting=False,
                        route_order=next_route_order,
                    )
                )
                next_route_order += 1
            request_source = "scaling_policy_instance_aggregate"
        elif point is not None:
            for request_id in point.request_ids:
                request_rows.append(
                    make_request_row(
                        request_id=request_id,
                        seq_len=point.request_length(request_id),
                        source_instance_id=cursor.instance_id,
                        is_waiting=request_id in point.waiting_ids,
                        route_order=next_route_order,
                    )
                )
                next_route_order += 1
            request_source = "stat_collector"
        else:
            request_source = "none"
        status = monitor_point.status if monitor_point is not None else "UNKNOWN"
        if policy_awake_ids is not None:
            is_awake = cursor.instance_id in policy_awake_ids
            status_source = "scaling_policy_current_state"
        elif monitor_point is not None:
            is_awake = status in {"AWAKEN", "AWAKE", "RUNNING"}
            status_source = "elastic_monitor"
        else:
            # StatCollector files exist for sleeping instances too.  Unknown
            # status is kept explicit, while true is a conservative benchmark
            # default that avoids silently removing an instance from the pool.
            is_awake = True
            status_source = "unknown_assume_awake"
        route_request_count = len(request_rows)
        if policy_load is not None:
            running_count = route_request_count
            waiting_count = 0
            token_count = policy_load.token_count
            model_version = point.model_version if point is not None else 0
            snapshot_timestamp = point.timestamp if point is not None else None
            generation_throughput = policy_load.throughput
            kv_cache_usage = (
                point.kv_cache_usage
                if point is not None
                else (monitor_point.kv_cache_usage if monitor_point else 0.0)
            )
        elif point is not None:
            running_count = point.running_count
            waiting_count = point.waiting_count
            token_count = sum(point.request_length(request_id) for request_id in point.request_ids)
            model_version = point.model_version
            snapshot_timestamp = point.timestamp
            generation_throughput = point.generation_throughput
            kv_cache_usage = point.kv_cache_usage
        else:
            running_count = monitor_point.running_count if monitor_point else 0
            waiting_count = monitor_point.waiting_count if monitor_point else 0
            token_count = 0
            model_version = 0
            snapshot_timestamp = None
            generation_throughput = monitor_point.generation_throughput if monitor_point else 0.0
            kv_cache_usage = monitor_point.kv_cache_usage if monitor_point else 0.0
        instances.append(
            {
                "instance_id": cursor.instance_id,
                "is_awake": is_awake,
                "status": status,
                "status_source": status_source,
                "request_source": request_source,
                "pool_id": monitor_point.pool_id if monitor_point else None,
                "model_version": model_version,
                "requests": request_rows,
                "route_request_count": route_request_count,
                "running_count": max(0, int(running_count)),
                "waiting_count": max(0, int(waiting_count)),
                "token_count": max(0, int(token_count)),
                "max_model_len": args.rollout_max_model_len if role == "Rollout" else args.rm_max_model_len,
                "throughput_params": list(cost_model.throughput_params),
                "route_cost_params": list(cost_model.route_cost_params),
                "observed_generation_throughput": safe_float(generation_throughput),
                "observed_kv_cache_usage": safe_float(kv_cache_usage),
                "snapshot_timestamp": format_timestamp(snapshot_timestamp),
                "snapshot_age_s": age_seconds(simulation_timestamp, snapshot_timestamp),
                "monitor_timestamp": format_timestamp(monitor_point.snapshot_timestamp if monitor_point else None),
                "monitor_age_s": age_seconds(
                    simulation_timestamp,
                    monitor_point.snapshot_timestamp if monitor_point else None,
                ),
                "stat_source_file": cursor.path.name,
            }
        )

    limits = role_limits(args, role)
    return {
        "role": role,
        "model": role_cursors[0].model if role_cursors else (args.rollout_model if role == "Rollout" else ""),
        "strategy": "throughput_optimal" if role == "Rollout" else "itl",
        "instances": instances,
        "pending_requests": pending_requests,
        "pending_count_reported": int(round(float(summary.get("router_request_count", 0.0)))),
        "pending_token_count_reported": pending_tokens,
        "queue_scope": str(cycle.policy_fields.get("vllm_current_queue_scope", args.queue_scope)),
        **limits,
        "cost_model": {
            "path": cost_model.path,
            "tp_pp": cost_model.tp_pp,
            "source": cost_model.source,
            "throughput_params": list(cost_model.throughput_params),
            "throughput_param_order": ["A", "B", "C", "D"],
            "route_cost_params": list(cost_model.route_cost_params),
            "route_cost_param_order": [
                "other_threshold",
                "other_latency_b",
                "other_latency_k",
                "attn_latency_b",
                "attn_latency_k",
            ],
        },
        "synthetic_pending_length_source": pending_length_source,
        "inflight_request_source": (
            "scaling_policy_instance_aggregate"
            if policy_loads
            else "stat_collector_fallback"
        ),
    }


def candidate_to_json(candidate: Candidate) -> dict[str, object]:
    return {
        "index": candidate.index,
        "action_type": candidate.action_type,
        "role_name": candidate.role_name,
        "model_name": candidate.model_name,
        "num_instances": candidate.num_instances,
        "preferred_instance_ids": list(candidate.preferred_instance_ids),
        "pre_wake": [dict(item) for item in candidate.pre_wake],
        "pre_sleep": [dict(item) for item in candidate.pre_sleep],
        "role_plans": candidate.role_plans,
    }


def cycle_to_json(
    cycle: Cycle,
    roles: dict[str, dict[str, object]],
    cursors: list[StatCursor],
    args: argparse.Namespace,
) -> dict[str, object]:
    simulation_timestamp = cycle.simulation_timestamp
    timing = None
    if cycle.evaluation_timing is not None:
        timing = {
            **cycle.evaluation_timing,
            "source_timestamp": format_timestamp(cycle.evaluation_timing_timestamp),
        }
    max_stat_age = 0.0
    stat_age_values: list[float] = []
    monitor_age_values: list[float] = []
    for role in roles.values():
        for instance in role["instances"]:
            snapshot_timestamp = instance.get("snapshot_timestamp")
            if snapshot_timestamp:
                parsed = parse_iso_timestamp(str(snapshot_timestamp))
                if parsed:
                    age = max(0.0, (simulation_timestamp - parsed).total_seconds())
                    stat_age_values.append(age)
            monitor_timestamp = instance.get("monitor_timestamp")
            if monitor_timestamp:
                parsed = parse_iso_timestamp(str(monitor_timestamp))
                if parsed:
                    monitor_age_values.append(max(0.0, (simulation_timestamp - parsed).total_seconds()))
    if stat_age_values:
        max_stat_age = max(stat_age_values)
    return {
        "schema_version": 1,
        "record_type": "elastic_simulation_input",
        "run_dir": str(args.log_dir),
        "source": {
            "policy_log": "ScalingPolicy.log",
            "monitor_log": "ElasticMonitor.log",
            "stat_logs": "StatCollector*.log",
            "cycle_log_timestamp": format_timestamp(cycle.timestamp),
            "candidate_evaluation_completed_timestamp": format_timestamp(
                cycle.evaluation_timing_timestamp
            ),
        },
        "cycle_id": cycle.cycle_id,
        "accepted_action_role": cycle.accepted_action_role,
        "decision_timestamp": format_timestamp(simulation_timestamp),
        "roles": roles,
        "candidates": [
            candidate_to_json(candidate) for candidate in sorted(cycle.candidates, key=lambda item: item.index)
        ],
        "candidate_count": len(cycle.candidates),
        "policy": cycle.policy_fields,
        "cycle_role_summary": cycle.role_summary,
        "cycle_role_instance_ids": {
            role: list(instance_ids) for role, instance_ids in cycle.role_instance_ids.items()
        },
        "cycle_role_instance_loads": {
            role: [
                {
                    "instance_id": load.instance_id,
                    "request_count": load.request_count,
                    "token_count": load.token_count,
                    "throughput": load.throughput,
                }
                for load in loads
            ]
            for role, loads in cycle.role_instance_loads.items()
        },
        "reference_timing": timing,
        "alignment": {
            "method": "latest_snapshot_not_after_candidate_evaluation_start_timestamp",
            "cycle_log_delay_s": max(0.0, (cycle.timestamp - simulation_timestamp).total_seconds()),
            "max_stat_snapshot_age_s": safe_float(max_stat_age),
            "max_monitor_snapshot_age_s": safe_float(max(monitor_age_values) if monitor_age_values else 0.0),
            "stat_snapshot_count": sum(1 for cursor in cursors if cursor.current is not None),
            "stat_file_count": len(cursors),
        },
        "assumptions": {
            "purpose": "timing_benchmark_not_exact_policy_replay",
            "routing_metadata": "all_active_instances_are_eligible",
            "pending_order": "synthetic_deterministic_order",
            "pending_lengths": "scaling_policy_router_aggregate_evenly_distributed",
            "inflight_lengths": "scaling_policy_instance_aggregate_evenly_distributed",
            "unknown_monitor_status": "treat_as_awake",
            "instance_status": "scaling_policy_current_state_then_elastic_monitor",
            "route_request_count": "scaling_policy_instance_request_count_rounded_to_integer",
            "time_alignment": "evaluation_end_minus_elapsed; metadata uses nearest_previous_snapshot",
            "cxx_contract": "UTF-8 JSONL, schema_version=1, null for unavailable optional values",
        },
    }


def parse_cycle_ids(raw: str) -> frozenset[int]:
    try:
        values = frozenset(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("cycle ids must be comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one cycle id is required")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("cycle ids must be non-negative")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", nargs="?", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--start-cycle", type=int, default=None)
    parser.add_argument("--end-cycle", type=int, default=None)
    parser.add_argument(
        "--cycles",
        type=parse_cycle_ids,
        default=None,
        metavar="ID[,ID...]",
        help="emit only these cycle ids (may be combined with start/end filters)",
    )
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--cycle-stride", type=int, default=1)
    parser.add_argument(
        "--include-empty-cycles",
        action="store_true",
        help="also emit policy cycles that did not build a candidate set",
    )
    parser.add_argument(
        "--accepted-action-role",
        choices=ROLE_ORDER,
        default=None,
        help="emit only cycles whose selected action targets this role",
    )
    parser.add_argument("--max-pending-requests", type=int, default=None)
    parser.add_argument("--default-pending-seq-len", type=int, default=1024)
    parser.add_argument("--queue-scope", choices=("running", "running_waiting"), default="running")
    parser.add_argument("--rollout-model", default="Qwen2.5-7B")
    parser.add_argument("--rollout-cost-model", type=Path, default=DEFAULT_ROLLOUT_COST_MODEL)
    parser.add_argument("--rm-cost-model", type=Path, default=DEFAULT_RM_COST_MODEL)
    parser.add_argument("--rollout-tp-pp", default="TP1_PP1")
    parser.add_argument("--rm-tp-pp", default="TP4_PP1")
    parser.add_argument("--rollout-max-model-len", type=int, default=2**63 - 1)
    parser.add_argument("--rm-max-model-len", type=int, default=2**63 - 1)
    parser.add_argument("--rollout-max-concurrent-requests", type=int, default=None)
    parser.add_argument("--rm-max-concurrent-requests", type=int, default=128)
    parser.add_argument("--rm-waiting-admission-cap", type=int, default=3)
    parser.add_argument("--rollout-delta-threshold", type=float, default=0.5)
    parser.add_argument("--rm-delta-threshold", type=float, default=0.005)
    parser.add_argument("--rm-max-itl", type=float, default=None)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.log_dir.is_dir():
        raise SystemExit(f"log directory not found: {args.log_dir}")
    if args.cycle_stride <= 0:
        raise SystemExit("--cycle-stride must be positive")
    for name in ("max_cycles", "max_pending_requests"):
        value = getattr(args, name)
        if value is not None and value < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.start_cycle is not None and args.end_cycle is not None and args.end_cycle < args.start_cycle:
        raise SystemExit("--end-cycle must be >= --start-cycle")


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    scaling_policy_path = args.log_dir / "ScalingPolicy.log"
    if not scaling_policy_path.is_file():
        raise SystemExit(f"ScalingPolicy.log not found: {scaling_policy_path}")

    stat_cursors = discover_stat_cursors(args.log_dir, args.rollout_model)
    if not stat_cursors:
        raise SystemExit(f"no StatCollector*.log files found in {args.log_dir}")
    monitor = MonitorCursor(args.log_dir / "ElasticMonitor.log")
    cost_models = {
        "Rollout": load_cost_model(args.rollout_cost_model, args.rollout_tp_pp),
        "RewardModel": load_cost_model(args.rm_cost_model, args.rm_tp_pp),
    }
    for role, cost_model in cost_models.items():
        if cost_model.source != "cost_model":
            print(
                f"warning: {role} cost model fallback={cost_model.source} path={cost_model.path}",
                file=sys.stderr,
            )

    output: TextIO
    output_path = args.output
    if output_path is None:
        output = sys.stdout
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = output_path.open("w", encoding="utf-8")

    cursors_by_role: dict[str, list[StatCursor]] = {role: [] for role in ROLE_ORDER}
    for cursor in stat_cursors:
        cursors_by_role.setdefault(cursor.role, []).append(cursor)

    emitted = 0
    seen = 0
    try:
        for cycle in iter_policy_cycles(scaling_policy_path):
            if args.start_cycle is not None and cycle.cycle_id < args.start_cycle:
                continue
            if args.end_cycle is not None and cycle.cycle_id > args.end_cycle:
                break
            if args.cycles is not None and cycle.cycle_id > max(args.cycles):
                break
            if args.cycles is not None and cycle.cycle_id not in args.cycles:
                continue
            if (
                args.accepted_action_role is not None
                and cycle.accepted_action_role != args.accepted_action_role
            ):
                continue
            if not cycle.candidates and not args.include_empty_cycles:
                continue
            if seen % args.cycle_stride != 0:
                seen += 1
                continue
            seen += 1
            roles = {
                role: build_role_snapshot(
                    cycle,
                    role,
                    cursors_by_role.get(role, []),
                    monitor,
                    cost_models[role],
                    args,
                )
                for role in ROLE_ORDER
            }
            record = cycle_to_json(cycle, roles, stat_cursors, args)
            output.write(json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
            output.write("\n")
            emitted += 1
            if args.max_cycles is not None and emitted >= args.max_cycles:
                break
    finally:
        for cursor in stat_cursors:
            cursor.close()
        monitor.close()
        if output is not sys.stdout:
            output.close()

    print(f"wrote {emitted} cycle records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
