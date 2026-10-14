import json
import logging
import os
from datetime import datetime, timezone
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from psrl.workers.gen.stats_collector import EngineStats
    from psrl.workers.gen.utils import RolloutInstanceId

logger = logging.getLogger(__name__)


class StatsRecorder:
    """
    Writes periodic per-replica stats snapshots to JSONL files.

    One file per (replica_idx, dp_rank):
        {logging_path}/stats_r{replica_idx}_dp{dp_rank}.jsonl

    A run-level config file is written once at startup:
        {logging_path}/stats_config.json

    This class is NOT a Ray actor. It is instantiated inside RolloutCoordinator
    and called from a background asyncio task.
    """

    def __init__(self, config: "DictConfig", logging_path: str) -> None:
        self._config = config
        self._logging_path = logging_path
        self._file_handles: dict[str, IO[str]] = {}  # filename -> open file handle
        os.makedirs(logging_path, exist_ok=True)

    def write_config(self, routing_strategy: str, partial_rollout: bool) -> None:
        """Write stats_config.json once at coordinator startup."""
        config_path = os.path.join(self._logging_path, "stats_config.json")
        payload = {
            "routing_strategy": routing_strategy,
            "partial_rollout": partial_rollout,
            "interval_in_s": self._config.status_collection.stats_recorder.interval_in_s,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }
        with open(config_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"StatsRecorder: wrote config to {config_path}")

    def record(self, instance_to_engine_status: "dict[RolloutInstanceId, EngineStats]") -> None:
        """
        Append one JSONL row per instance to its file.

        Uses the wall-clock time of this call (not the vLLM snapshot timestamp)
        so that rows from different replicas share a common reference time for
        cross-replica analysis.
        """
        if not instance_to_engine_status:
            return

        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        for (_replica_id, dp_rank), engine_stats in instance_to_engine_status.items():
            snapshot = engine_stats.snapshot
            sched = snapshot.get("scheduler_stats", {})
            iter_stats = snapshot.get("iteration_stats", {})

            row = {
                "ts": ts,
                "total_elapsed_time": snapshot.get("total_elapsed_time", 0.0),
                "model_version": engine_stats.model_version,
                "num_running_reqs": sched.get("num_running_reqs", 0),
                "num_waiting_reqs": sched.get("num_waiting_reqs", 0),
                "kv_cache_usage": sched.get("kv_cache_usage", 0.0),
                "generation_throughput": snapshot.get("generation_throughput", 0.0),
                "avg_ttft": iter_stats.get("avg_time_to_first_tokens") if iter_stats else None,
                "avg_itl": iter_stats.get("avg_inter_token_latencies") if iter_stats else None,
            }

            filename = self._get_filename(engine_stats.replica_idx, dp_rank)
            try:
                fh = self._get_or_open(filename)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            except OSError as e:
                logger.warning(f"StatsRecorder: failed to write {filename}: {e}")

    def record_smg_routing_status(self, routing_loop_status: dict, workers_stats) -> None:
        """
        Append one JSONL row with SMG routing-loop queue state.

        The row is intentionally separate from per-instance engine stats because
        it describes gateway-level routing pressure and SMG's current
        prompt-group pinning view.
        """
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "queue_len": routing_loop_status.get("queue_len"),
            "pending_request_num": routing_loop_status.get("pending_request_num"),
            "running_request_num": routing_loop_status.get("running_request_num"),
            "running_tasks": routing_loop_status.get("running_tasks"),
            "paused": routing_loop_status.get("paused"),
            "selecting": routing_loop_status.get("selecting"),
            "queue_keys": routing_loop_status.get("queue_keys", []),
            "partition_queue_lens": routing_loop_status.get("partition_queue_lens", {}),
            "workers": workers_stats,
            "routing_loop_status": routing_loop_status,
        }

        filename = self._get_smg_routing_status_filename()
        try:
            fh = self._get_or_open(filename)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        except OSError as e:
            logger.warning(f"StatsRecorder: failed to write {filename}: {e}")

    def close(self) -> None:
        """Flush and close all open file handles."""
        for filename, fh in self._file_handles.items():
            try:
                fh.flush()
                fh.close()
            except OSError as e:
                logger.warning(f"StatsRecorder: error closing {filename}: {e}")
        self._file_handles.clear()

    # --- Internal Helpers ---

    def _get_filename(self, replica_idx: int, dp_rank: int) -> str:
        return os.path.join(self._logging_path, f"stats_r{replica_idx}_dp{dp_rank}.jsonl")

    def _get_smg_routing_status_filename(self) -> str:
        return os.path.join(self._logging_path, "smg_routing_status.jsonl")

    def _get_or_open(self, filename: str):
        """Open filename in 'w' mode on first call per run, then 'a'."""
        if filename not in self._file_handles:
            self._file_handles[filename] = open(filename, "w")
        return self._file_handles[filename]
