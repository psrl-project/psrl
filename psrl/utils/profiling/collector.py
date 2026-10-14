import logging
import os
import time

from verl import DataProto

from psrl.utils.profiling.records import (
    DecodeRecord,
    EnvTurnRecord,
    ModelTurnRecord,
    PrefillRecord,
    TrajectoryProfilingData,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class TurnProfilingCollector:
    """
    Encapsulate turn-level profiling state for one trajectory.
    """

    def __init__(self):
        self._turn_records: list[ModelTurnRecord] = []
        self._env_records: list[EnvTurnRecord] = []
        self._trajectory_start_ts: float = 0.0
        self._turn_submit_ts: float = 0.0
        self._last_generation_end_ts: float = 0.0
        self._turn_index: int = 0

    def on_turn_submit(self) -> None:
        """
        Record the timestamp when a generation request is submitted.

        Call from AgentData.prepare_generation_request().
        """
        now = time.time()
        self._turn_submit_ts = now
        if self._turn_index == 0:
            self._trajectory_start_ts = now

    def on_turn_complete(self, output: DataProto) -> None:
        """
        Build and store a `ModelTurnRecord` from generation output.

        Extracts profiling records from `output.non_tensor_batch`, computes
        `router_wait_s` for the first segment using wall-clock timestamps, and
        appends the turn record.

        Call from `AgentData.update_trajectory_state_from_output()`.

        Args:
            output (DataProto): Model output containing profiling fields
                in `non_tensor_batch`.
        """
        ntb = output.non_tensor_batch

        generation_start_wall_ts = float(ntb.get("profiling_generation_start_wall_ts", [0.0])[0])
        generation_end_wall_ts = float(ntb.get("profiling_generation_end_wall_ts", [0.0])[0])

        # Skip if no profiling data was attached.
        if generation_start_wall_ts == 0.0 and generation_end_wall_ts == 0.0:
            psrl_logger.warning(f"on_turn_complete: No profiling data for turn {self._turn_index}, skipping.")
            self._turn_index += 1
            return

        # Router wait spans submission to engine execution for the first segment.
        router_wait_s = 0.0
        if self._turn_submit_ts > 0 and generation_start_wall_ts > 0:
            router_wait_s = generation_start_wall_ts - self._turn_submit_ts

        if self._turn_index > 0 and self._last_generation_end_ts > 0:
            env_duration_s = self._turn_submit_ts - self._last_generation_end_ts
            self._env_records.append(
                EnvTurnRecord(
                    turn_index=self._turn_index,
                    duration_s=max(env_duration_s, 0.0),
                )
            )

        if generation_end_wall_ts > 0:
            self._last_generation_end_ts = generation_end_wall_ts

        raw_prefill = ntb.get("profiling_prefill_records", [None])[0]
        raw_decode = ntb.get("profiling_decode_records", [None])[0]

        prefill_records = []
        if raw_prefill is not None and len(raw_prefill) > 0:
            prefill_records = [PrefillRecord.from_dict(r) for r in raw_prefill]

        decode_records = []
        if raw_decode is not None and len(raw_decode) > 0:
            decode_records = [DecodeRecord.from_dict(r) for r in raw_decode]

        # Set router_wait_s on the first PrefillRecord.
        # Only applies when the request went through the router (not an internal preempt).
        if prefill_records and prefill_records[0].trigger != "internal_preempt_resume":
            prefill_records[0].router_wait_s = max(router_wait_s, 0.0)

        instance_id = int(ntb.get("rollout_instance_id", [0])[0] or 0)
        total_seq_len = prefill_records[-1].total_seq_len if prefill_records else 0

        for pr in prefill_records:
            if pr.instance_id == 0:
                pr.instance_id = instance_id
        for dr in decode_records:
            if dr.instance_id == 0:
                dr.instance_id = instance_id

        record = ModelTurnRecord(
            turn_index=self._turn_index,
            prefill_records=prefill_records,
            decode_records=decode_records,
            total_seq_len=total_seq_len,
        )
        self._turn_records.append(record)
        self._turn_index += 1

    @property
    def trajectory_start_ts(self) -> float:
        """Timestamp when the first generation turn was submitted (0.0 if not yet set)."""
        return self._trajectory_start_ts

    def get_timing_breakdown(self) -> dict:
        """
        Return LLM-side timing summary for use in trajectory summary text.

        Returns a dict with keys:
            assistant_s: total wall-clock time across all model turns (router_wait + scheduler_wait + prefill + decode)
            env_s: total environment execution time between turns
        """
        return {
            "assistant_s": sum(r.total_duration_s for r in self._turn_records),
            "env_s": sum(r.duration_s for r in self._env_records),
        }

    def finalize(self, request_id: int) -> TrajectoryProfilingData | None:
        """
        Build and return TrajectoryProfilingData for the entire trajectory.

        Args:
            request_id (int): The trajectory's request ID.

        Returns:
            TrajectoryProfilingData | None: Profiling data, or None if no
                turns were recorded.
        """
        if not self._turn_records:
            return None

        profiling_end_ts = time.time()
        total_duration_s = profiling_end_ts - self._trajectory_start_ts if self._trajectory_start_ts > 0 else 0.0

        profiling_data = TrajectoryProfilingData(
            request_id=request_id,
            total_turns=len(self._turn_records),
            total_duration_s=total_duration_s,
            turn_records=self._turn_records,
            env_records=self._env_records,
        )
        profiling_data.compute_summary()
        return profiling_data
