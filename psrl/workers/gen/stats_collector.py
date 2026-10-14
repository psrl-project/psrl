import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

import numpy as np
from omegaconf import DictConfig
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

from psrl.utils.logger import FileOnlyHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class EngineStats:
    # initial properties
    replica_idx: Final[int]
    data_parallel_rank: Final[int]
    model_version: Final[int]
    snapshot: Final[dict]

    @staticmethod
    def get_default_snapshot() -> dict:
        return {
            "timestamp": None,
            "total_elapsed_time": 0.0,
            "elapsed_time_since_last_record": 0.0,
            "scheduler_stats": {
                "req_id_to_prompt_token_num": {},
                "req_id_to_response_token_num": {},
                "req_id_in_waiting": [],
                "num_running_reqs": 0,
                "num_waiting_reqs": 0,
                "kv_cache_usage": 0.0,
            },
            "generation_throughput": 0.0,
        }

    def get_waiting_queue_size(self) -> int:
        return self.snapshot.get("scheduler_stats", {}).get("num_waiting_reqs", 0)

    def get_running_queue_size(self) -> int:
        return self.snapshot.get("scheduler_stats", {}).get("num_running_reqs", 0)

    def get_waiting_and_running_queue_size(self) -> int:
        return self.get_waiting_queue_size() + self.get_running_queue_size()

    def get_generation_throughput(self) -> float:
        return self.snapshot.get("generation_throughput", 0.0)

    def get_kv_cache_utilization(self) -> float:
        return self.snapshot.get("scheduler_stats", {}).get("kv_cache_usage", 0.0)

    def get_req_id_to_prompt_token_num(self) -> dict[str, int]:
        return self.snapshot.get("scheduler_stats", {}).get("req_id_to_prompt_token_num", {})

    def get_req_id_to_response_token_num(self) -> dict[str, int]:
        return self.snapshot.get("scheduler_stats", {}).get("req_id_to_response_token_num", {})

    def get_total_token_num(self) -> int:
        return sum(self.get_req_id_to_prompt_token_num().values()) + sum(
            self.get_req_id_to_response_token_num().values()
        )


class DPLBStatCollector(StatLoggerBase):
    """
    Collects and reports statistics from vLLM engine to the rollout coordinator.

    This class extends vLLM's StatLoggerBase to:
    - Monitor scheduler statistics (waiting/running request counts)
    - Detect changes in request queue status
    - Send status updates to output queue for coordinator consumption
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        psrl_config: DictConfig,
        replica_idx: int,
        role: str,
    ):
        """
        Initialize the StatCollector.

        Args:
            vllm_config: vLLM configuration object
            replica_idx: Unique identifier for this AsyncLLM replica
        """
        self.vllm_config = vllm_config
        self.psrl_config = psrl_config
        self.role = role

        self.replica_idx = replica_idx

        self.model_version = 0
        self._begin_record = False
        self.start_time = None
        self.last_record_time = None
        self.last_dump_to_file_time = None
        self.last_push_to_queue_time = None
        self.output_queue = None

        # Cumulative prefill/decode time tracking.
        self._cumulative_prefill_time: float = 0.0
        self._cumulative_decode_time: float = 0.0
        self._cumulative_prefill_tokens: int = 0  # total prompt tokens (including cache hit)
        self._cumulative_prefill_computed_tokens: int = 0  # actual computed (cache miss)
        self._cumulative_decode_tokens: int = 0
        self._last_time_split_log_time: float | None = None
        self._time_split_enable: bool = bool(self.psrl_config.profile.time_split.enable)
        self._time_split_interval_s: float = float(self.psrl_config.profile.time_split.interval_in_s)
        if self._time_split_enable:
            self._time_split_logger = logging.getLogger(f"psrl.time_split.I{self.replica_idx}")
            self._time_split_logger.propagate = False
            self._time_split_logger.setLevel(logging.INFO)
            self._time_split_logger.addHandler(
                FileOnlyHandler(self.psrl_config.logging_path, f"TimeSplit_I{self.replica_idx}")
            )

        # Build logger
        if self.psrl_config.status_collection.dump_logging_to_file_level != "none":
            self.log_prefix = f"DPLBStatCollector_{self.role}_I{self.replica_idx}"
            psrl_logger.propagate = False
            psrl_logger.addHandler(FileOnlyHandler(self.psrl_config.logging_path, self.log_prefix))
            psrl_logger.info(f"Initialized DPLBStatCollector: replica={self.replica_idx}, role={self.role!r}.")

    def begin_record(self):
        """
        Begin recording statistics.
        """
        self._begin_record = True
        self.start_time = time.time()
        self.last_record_time = self.start_time
        self.last_dump_to_file_time = self.start_time
        self.last_push_to_queue_time = self.start_time

    def _make_snapshot_after_model_version_update(self) -> dict:
        curr_time = time.time()
        snapshot = EngineStats.get_default_snapshot()
        snapshot["total_elapsed_time"] = curr_time - self.start_time
        snapshot["elapsed_time_since_last_record"] = curr_time - self.last_record_time
        return snapshot

    def init_output_queue(self, output_queue):
        """
        Initialize the output queue for sending status updates.

        Args:
            output_queue: Ray queue for sending status updates to coordinator
        """
        self.output_queue = output_queue

    def record_model_version_update(self, model_version: int, engine_index: int):
        """
        Set the model version for this engine instance.

        Args:
            model_version: Model version
            engine_index: Engine instance identifier
        """
        curr_time = time.time()
        self.model_version = model_version
        snapshot = self._make_snapshot_after_model_version_update()
        # We force a record to the output queue to ensure the coordinator knows the model version update immediately
        self.output_queue.put_nowait(
            EngineStats(
                replica_idx=self.replica_idx,
                data_parallel_rank=engine_index,
                model_version=self.model_version,
                snapshot=snapshot,
            )
        )
        # Dump logging to file if enabled
        if self.psrl_config.status_collection.dump_logging_to_file_level != "none":
            psrl_logger.debug(
                f"Snapshot (model version {self.model_version}, "
                f"instance_id {(self.replica_idx, engine_index)}): {snapshot}"
            )
        self.last_record_time = curr_time
        self.last_push_to_queue_time = curr_time

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """
        Record and process scheduler statistics from vLLM engine.

        This method is called by vLLM during inference to report statistics.
        When request counts change, it sends updates to the coordinator via output queue.

        Args:
            scheduler_stats: Statistics from vLLM scheduler (request counts, etc.)
            iteration_stats: Statistics from vLLM iteration
            mm_cache_stats: Multi-modal cache statistics (not currently used)
            engine_idx: Engine index
        """
        assert self.output_queue is not None, "Output queue is not initialized"

        curr_time = time.time()

        if scheduler_stats is not None:
            scheduler_stats = {
                "req_id_to_prompt_token_num": scheduler_stats.req_id_to_prompt_token_num
                if scheduler_stats.req_id_to_prompt_token_num
                else {},
                "req_id_to_response_token_num": scheduler_stats.req_id_to_response_token_num
                if scheduler_stats.req_id_to_response_token_num
                else {},
                "req_id_in_waiting": list(scheduler_stats.req_id_in_waiting),
                "num_running_reqs": scheduler_stats.num_running_reqs,
                "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
                "kv_cache_usage": scheduler_stats.kv_cache_usage,
            }
        else:
            scheduler_stats = {
                "req_id_to_prompt_token_num": {},
                "req_id_to_response_token_num": {},
                "req_id_in_waiting": [],
                "num_running_reqs": 0,
                "num_waiting_reqs": 0,
                "kv_cache_usage": 0.0,
            }

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_elapsed_time": curr_time - self.start_time,
            "elapsed_time_since_last_record": curr_time - self.last_record_time,
            "scheduler_stats": scheduler_stats,
            "generation_throughput": 0.0,
        }

        if iteration_stats:
            time_to_first_tokens_iter = getattr(iteration_stats, "time_to_first_tokens_iter", [])
            num_prompt_reqs = len(time_to_first_tokens_iter)
            if len(time_to_first_tokens_iter) == 0:
                time_to_first_tokens_iter = [0.0]
            inter_token_latencies_iter = getattr(iteration_stats, "inter_token_latencies_iter", [])
            num_generation_reqs = len(inter_token_latencies_iter)
            if len(inter_token_latencies_iter) == 0:
                inter_token_latencies_iter = [0.0]
            iteration_stats_entry = {
                "num_prompt_tokens": getattr(iteration_stats, "num_prompt_tokens", 0),
                "num_generation_tokens": getattr(iteration_stats, "num_generation_tokens", 0),
                "num_prompt_reqs": num_prompt_reqs,
                "num_generation_reqs": num_generation_reqs,
                "num_preempted_reqs": getattr(iteration_stats, "num_preempted_reqs", 0),
                "num_finished_reqs": len(getattr(iteration_stats, "finished_requests", [])),
                # "max_time_to_first_tokens": np.max(time_to_first_tokens_iter),
                # "max_inter_token_latencies": np.max(inter_token_latencies_iter),
                "avg_time_to_first_tokens": np.mean(time_to_first_tokens_iter).item(),
                "avg_inter_token_latencies": np.mean(inter_token_latencies_iter).item(),
            }
            avg_itl = iteration_stats_entry["avg_inter_token_latencies"]
            snapshot["generation_throughput"] = num_generation_reqs / avg_itl if avg_itl > 0 else 0.0
            snapshot["iteration_stats"] = iteration_stats_entry

            # Ignore abnormally long intervals when accumulating prefill and decode
            # wall time.
            step_elapsed = snapshot["elapsed_time_since_last_record"]
            num_pt = iteration_stats_entry["num_prompt_tokens"]
            num_gt = iteration_stats_entry["num_generation_tokens"]
            # Actual computed prefill tokens (excluding cache hit).
            prompt_stats = getattr(iteration_stats, "prompt_token_stats", None)
            num_pt_computed = getattr(prompt_stats, "computed", 0) if prompt_stats else 0

            # Accumulate token counts unconditionally.
            self._cumulative_prefill_tokens += num_pt
            self._cumulative_prefill_computed_tokens += num_pt_computed
            self._cumulative_decode_tokens += num_gt

            # Accumulate time only for sane step durations.
            if 0 < step_elapsed < 5.0:
                if num_pt > 0 and num_gt == 0:
                    self._cumulative_prefill_time += step_elapsed
                elif num_pt == 0 and num_gt > 0:
                    self._cumulative_decode_time += step_elapsed
                elif num_pt > 0 and num_gt > 0:
                    # Mixed step: attribute proportionally by token count.
                    total_tokens = num_pt + num_gt
                    self._cumulative_prefill_time += step_elapsed * (num_pt / total_tokens)
                    self._cumulative_decode_time += step_elapsed * (num_gt / total_tokens)

            # Log cumulative prefill/decode time at configured interval.
            if self._time_split_enable:
                if self._last_time_split_log_time is None:
                    self._last_time_split_log_time = curr_time
                if curr_time - self._last_time_split_log_time >= self._time_split_interval_s:
                    total_tracked = self._cumulative_prefill_time + self._cumulative_decode_time
                    self._time_split_logger.info(
                        f"[I{self.replica_idx}] "
                        f"cumulative_prefill_s={self._cumulative_prefill_time:.2f}, "
                        f"cumulative_decode_s={self._cumulative_decode_time:.2f}, "
                        f"prefill_frac={self._cumulative_prefill_time / max(total_tracked, 1e-9):.4f}, "
                        f"prefill_tokens={self._cumulative_prefill_tokens}, "
                        f"prefill_computed_tokens={self._cumulative_prefill_computed_tokens}, "
                        f"decode_tokens={self._cumulative_decode_tokens}, "
                        f"wall_time_s={curr_time - self.start_time:.1f}"
                    )
                    self._last_time_split_log_time = curr_time

        if self.psrl_config.status_collection.dump_logging_to_file_level != "none":
            if (
                curr_time - self.last_dump_to_file_time
                >= self.psrl_config.status_collection.dump_logging_to_file_interval_in_ms / 1000.0
            ):
                if self.psrl_config.status_collection.dump_logging_to_file_level == "prompt":
                    if iteration_stats and snapshot["iteration_stats"]["num_prompt_reqs"] > 0:
                        psrl_logger.debug(f"Snapshot (model version {self.model_version}): {snapshot}")
                elif self.psrl_config.status_collection.dump_logging_to_file_level == "generation":
                    if (
                        iteration_stats
                        and snapshot["iteration_stats"]["num_prompt_reqs"] == 0
                        and snapshot["iteration_stats"]["num_generation_reqs"] > 0
                    ):
                        psrl_logger.debug(f"Snapshot (model version {self.model_version}): {snapshot}")
                elif self.psrl_config.status_collection.dump_logging_to_file_level == "all":
                    psrl_logger.debug(f"Snapshot (model version {self.model_version}): {snapshot}")
                else:
                    raise ValueError(
                        f"Invalid dump logging to file level: "
                        f"{self.psrl_config.status_collection.dump_logging_to_file_level}"
                    )
                self.last_dump_to_file_time = curr_time
        self.last_record_time = curr_time

        waiting_and_running_queue_size = (
            snapshot["scheduler_stats"]["num_waiting_reqs"] + snapshot["scheduler_stats"]["num_running_reqs"]
        )
        if (
            waiting_and_running_queue_size == 0
            or curr_time - self.last_push_to_queue_time
            >= self.psrl_config.status_collection.engine_sync_interval_in_ms / 1000.0
        ):
            self.output_queue.put_nowait(
                EngineStats(
                    replica_idx=self.replica_idx,
                    data_parallel_rank=engine_idx,
                    model_version=self.model_version,
                    snapshot=snapshot,
                )
            )
            self.last_push_to_queue_time = curr_time

    def log_engine_initialized(self):
        """
        Log engine initialization information including cache configuration.

        This method is called when the vLLM engine completes initialization
        and reports the number of GPU blocks available for caching.
        """
        if self.vllm_config.cache_config.num_gpu_blocks:
            psrl_logger.info(
                f"Engine {self.replica_idx}: vllm cache_config_info with initialization "
                f"after num_gpu_blocks is: {self.vllm_config.cache_config.num_gpu_blocks}"
            )
