import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

# Track initialized paths so each process truncates once, then appends.
_jsonl_initialized_paths: set[str] = set()

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class _NumpySafeEncoder(json.JSONEncoder):
    """
    JSON encoder that converts numpy scalar types to native Python types.

    Profiling data flows through numpy arrays (non_tensor_batch, scheduler
    arithmetic) and may retain numpy dtypes such as int64 / float64. This
    encoder transparently converts them so that json.dumps never fails.
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class PrefillTrigger(StrEnum):
    """
    Reason a request was (re-)scheduled for prefill within a turn.

    Used as `PrefillRecord.trigger`. StrEnum ensures values serialize to plain
    strings in JSON and that typos are caught at construction time.

    RESUME is a transient placeholder written by the scheduler for any
    re-queued WAITING request. It is overwritten by vllm_rollout with the
    router-supplied trigger (PREEMPT_RESUME or PARTIAL_ROLLOUT_RESUME)
    before the record is finalized into a `ModelTurnRecord`.
    """

    INITIAL = "initial"
    RESUME = "resume"  # transient placeholder
    PREEMPT_RESUME = "preempt_resume"
    INTERNAL_PREEMPT_RESUME = "internal_preempt_resume"
    PARTIAL_ROLLOUT_RESUME = "partial_rollout_resume"


@dataclass
class PrefillRecord:
    """
    Records prefix cache hit and timing information for a single scheduling event.

    Produced each time the scheduler moves a request from WAITING to RUNNING.
    A single generation turn may contain multiple records if the request is
    preempted and rescheduled, or interrupted and rerouted within that turn.

    Timing fields form the per-segment budget:
        router_wait_s + scheduler_wait_s + prefill_duration_s
    together with the paired `DecodeRecord.decode_duration_s` cover the full
    wall-clock time for one continuous execution segment.
    """

    instance_id: int = 0
    trigger: PrefillTrigger = PrefillTrigger.INITIAL
    total_seq_len: int = 0
    num_computed_tokens: int = 0
    num_prefill_tokens: int = 0
    cache_hit_rate: float = 0.0

    router_wait_s: float = 0.0
    scheduler_wait_s: float = 0.0
    prefill_duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PrefillRecord":
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "trigger" in kwargs:
            kwargs["trigger"] = PrefillTrigger(kwargs["trigger"])
        return cls(**kwargs)


@dataclass
class DecodeRecord:
    """
    Records decode-phase timing for a contiguous decode segment.

    Produced for each uninterrupted decode run on a single engine instance.
    A turn may accumulate multiple `DecodeRecord`s if the request migrates between
    instances or is preempted mid-decode.

    `decode_duration_s` is pure GPU decode time, measured from the first output
    token to the end of the decode segment (i.e., prefill time is excluded).
    """

    instance_id: int = 0
    num_decode_tokens: int = 0
    decode_duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DecodeRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EnvTurnRecord:
    """
    Profiling record for a single environment step between two model turns.

    Captures the wall-clock time spent in the environment (tool calls, reward
    computation, etc.) between the end of model turn N-1 and the start of
    model turn N.
    """

    turn_index: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EnvTurnRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelTurnRecord:
    """
    Profiling record for a single model generation turn in the multi-turn agent loop.

    Corresponds to one call to `rollout_router.generate_async()`. Aggregates all
    scheduling events (`PrefillRecord`s) and decode segments (`DecodeRecord`s) that
    occurred during this turn, including any preempt-resume or reroute cycles.

    The total turn duration decomposes cleanly into four additive components:

        total_duration_s = router_wait_time_s
                         + scheduler_wait_time_s
                         + prefill_time_s
                         + decode_time_s

    Each component is a non-negative sum, so negative values are impossible.
    """

    turn_index: int = 0
    prefill_records: list[PrefillRecord] = field(default_factory=list)
    decode_records: list[DecodeRecord] = field(default_factory=list)
    total_seq_len: int = 0

    @property
    def num_generated_tokens(self) -> int:
        return sum(dr.num_decode_tokens for dr in self.decode_records)

    @property
    def router_wait_time_s(self) -> float:
        """Total routing overhead across all segments (outside any vLLM instance)."""
        return sum(pr.router_wait_s for pr in self.prefill_records)

    @property
    def scheduler_wait_time_s(self) -> float:
        """Total time spent waiting in vLLM scheduler queues across all segments."""
        return sum(pr.scheduler_wait_s for pr in self.prefill_records)

    @property
    def prefill_time_s(self) -> float:
        """Pure prefill GPU compute across all segments."""
        return sum(pr.prefill_duration_s for pr in self.prefill_records)

    @property
    def decode_time_s(self) -> float:
        """Pure decode GPU compute across all segments."""
        return sum(dr.decode_duration_s for dr in self.decode_records)

    @property
    def total_duration_s(self) -> float:
        """
        Total wall-clock time for this turn.

        Equals the sum of all four components:
        router_wait_time_s + scheduler_wait_time_s + prefill_time_s + decode_time_s.
        """
        return self.router_wait_time_s + self.scheduler_wait_time_s + self.prefill_time_s + self.decode_time_s

    @property
    def initial_cache_hit_rate(self) -> float:
        if not self.prefill_records:
            return 0.0
        return self.prefill_records[0].cache_hit_rate

    @property
    def avg_cache_hit_rate(self) -> float:
        """Weighted average cache hit rate across all prefill scheduling events."""
        if not self.prefill_records:
            return 0.0
        total_tokens = sum(pr.total_seq_len for pr in self.prefill_records)
        if total_tokens == 0:
            return 0.0
        return sum(pr.cache_hit_rate * pr.total_seq_len for pr in self.prefill_records) / total_tokens

    @property
    def num_preempt_resumes_in_turn(self) -> int:
        """Number of (re-)prefill events caused by preemption."""
        return sum(
            1
            for pr in self.prefill_records
            if pr.trigger in (PrefillTrigger.PREEMPT_RESUME, PrefillTrigger.INTERNAL_PREEMPT_RESUME)
        )

    @property
    def num_partial_rollout_resumes_in_turn(self) -> int:
        """Number of reroute events where the router sent the request to a different instance."""
        return sum(1 for pr in self.prefill_records if pr.trigger == PrefillTrigger.PARTIAL_ROLLOUT_RESUME)

    def to_dict(self) -> dict:
        return {
            "turn_index": self.turn_index,
            "total_seq_len": self.total_seq_len,
            "prefill_records": [r.to_dict() for r in self.prefill_records],
            "decode_records": [r.to_dict() for r in self.decode_records],
            "num_generated_tokens": self.num_generated_tokens,
            "router_wait_time_s": self.router_wait_time_s,
            "scheduler_wait_time_s": self.scheduler_wait_time_s,
            "prefill_time_s": self.prefill_time_s,
            "decode_time_s": self.decode_time_s,
            "total_duration_s": self.total_duration_s,
            "initial_cache_hit_rate": self.initial_cache_hit_rate,
            "avg_cache_hit_rate": self.avg_cache_hit_rate,
            "num_preempt_resumes_in_turn": self.num_preempt_resumes_in_turn,
            "num_partial_rollout_resumes_in_turn": self.num_partial_rollout_resumes_in_turn,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelTurnRecord":
        data = dict(data)
        prefill_records = [PrefillRecord.from_dict(r) for r in data.pop("prefill_records", [])]
        decode_records = [DecodeRecord.from_dict(r) for r in data.pop("decode_records", [])]
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered, prefill_records=prefill_records, decode_records=decode_records)


@dataclass
class TrajectoryProfilingData:
    """
    Complete profiling data for a trajectory across all turns.

    Produced at trajectory finalization time. Contains per-turn records and
    aggregated summary statistics for offline analysis.
    """

    request_id: int = 0
    total_turns: int = 0
    total_duration_s: float = 0.0

    turn_records: list[ModelTurnRecord] = field(default_factory=list)
    env_records: list[EnvTurnRecord] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def compute_summary(self) -> dict:
        total_prefill_time = sum(r.prefill_time_s for r in self.turn_records)
        total_decode_time = sum(r.decode_time_s for r in self.turn_records)
        total_scheduler_wait = sum(r.scheduler_wait_time_s for r in self.turn_records)
        total_router_wait = sum(r.router_wait_time_s for r in self.turn_records)
        total_env = sum(r.duration_s for r in self.env_records)
        total_turn_duration = sum(r.total_duration_s for r in self.turn_records)
        total_generated_tokens = sum(r.num_generated_tokens for r in self.turn_records)

        # Average cache hit rate across turns (each turn already weighted internally).
        avg_cache_hit_rate = (
            sum(r.avg_cache_hit_rate for r in self.turn_records) / len(self.turn_records) if self.turn_records else 0.0
        )

        # Trigger breakdown across all PrefillRecords.
        trigger_breakdown = {t.value: 0 for t in PrefillTrigger}
        for turn in self.turn_records:
            for pr in turn.prefill_records:
                trigger_breakdown[pr.trigger] += 1

        n_turns = len(self.turn_records) or 1
        total_dur = self.total_duration_s if self.total_duration_s > 0 else 1.0
        self.summary = {
            "total_prefill_time_s": total_prefill_time,
            "total_decode_time_s": total_decode_time,
            "total_scheduler_wait_time_s": total_scheduler_wait,
            "total_router_wait_time_s": total_router_wait,
            "total_env_time_s": total_env,
            "total_turn_duration_s": total_turn_duration,
            "avg_turn_duration_s": total_turn_duration / n_turns,
            "avg_num_generated_tokens": total_generated_tokens / n_turns,
            "prefill_fraction": total_prefill_time / total_dur,
            "decode_fraction": total_decode_time / total_dur,
            "scheduler_wait_fraction": total_scheduler_wait / total_dur,
            "router_wait_fraction": total_router_wait / total_dur,
            "env_fraction": total_env / total_dur,
            "avg_cache_hit_rate": avg_cache_hit_rate,
            "prefill_trigger_breakdown": trigger_breakdown,
            "total_generated_tokens": total_generated_tokens,
        }
        return self.summary

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "total_turns": self.total_turns,
            "total_duration_s": self.total_duration_s,
            "turn_records": [r.to_dict() for r in self.turn_records],
            "env_records": [r.to_dict() for r in self.env_records],
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrajectoryProfilingData":
        turns = [ModelTurnRecord.from_dict(r) for r in data.get("turn_records", [])]
        env_records = [EnvTurnRecord.from_dict(r) for r in data.get("env_records", [])]
        return cls(
            request_id=data.get("request_id", 0),
            total_turns=data.get("total_turns", 0),
            total_duration_s=data.get("total_duration_s", 0.0),
            turn_records=turns,
            env_records=env_records,
            summary=data.get("summary", {}),
        )

    def write_jsonl(self, path: str | Path) -> None:
        """
        Write this trajectory's profiling data to a JSONL file.

        On the first call within the current process for a given `path`, any
        existing file at that path is truncated so that each run starts fresh.
        Subsequent calls within the same process append to the file.

        Args:
            path (str | Path): Path to the JSONL file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Truncate on first write per process so each run starts with a clean
        # file rather than appending to data from a previous run.
        resolved = str(path.resolve())
        if resolved not in _jsonl_initialized_paths:
            _jsonl_initialized_paths.add(resolved)
            open_mode = "w"
        else:
            open_mode = "a"

        with open(path, open_mode) as f:
            f.write(json.dumps(self.to_dict(), cls=_NumpySafeEncoder) + "\n")
