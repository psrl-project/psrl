import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class StatCollector(StatLoggerBase):
    """
    Enhanced statistics collector for detailed profiling of vLLM engine.

    This class extends vLLM's StatLoggerBase to collect comprehensive metrics including:
    - Request queue statistics (running, waiting, corrupted requests)
    - KV cache usage and scheduler state
    - Prefix cache hit rates and statistics
    - Token throughput (prompt and generation)
    - Preemption counts and timing metrics
    - Memory usage statistics
    - Inter-token latencies and time-to-first-token metrics

    Features:
    - Auto-detects rank from environment variables (RANK)
    - Handles TP/PP scenarios where each rank records its own statistics
    - Robust error handling for None values in vllm_config
    - Comprehensive debugging information for troubleshooting
    - Safe attribute access with fallback values
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_index: int = 0,
        log_dir: str = "./profile_logs",
        log_file: str = "profile",
    ):
        """
        Initialize the Enhanced StatCollector.

        Args:
            vllm_config: vLLM configuration object
            engine_index: Unique identifier for this engine instance
            log_dir: Directory to save profile logs
        """
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize statistics tracking
        self.last_scheduler_stats = SchedulerStats()

        # Throughput tracking (similar to LoggingStatLogger)
        self.last_log_time = time.time()
        self.num_prompt_tokens = 0
        self.num_generation_tokens = 0
        self.last_prompt_throughput = 0.0
        self.last_generation_throughput = 0.0

        # Create rank-specific log file (timestamp not used currently)
        # _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{log_file}.jsonl"

        # Clear existing log file to ensure fresh start
        self._clear_log_file()

        # Statistics history
        self.stats_history = []
        self.start_time = time.time()
        self._warmup_completed = False

        # Initialize log file
        self._write_log_header()

        psrl_logger.info(f"Enhanced StatCollector initialized, engine {self.engine_index}")
        psrl_logger.info(f"Profile logs will be saved to: {self.log_file}")

    def _clear_log_file(self):
        """Clear the existing log file to ensure fresh start on each run."""
        try:
            if self.log_file.exists():
                # Remove existing file to ensure clean start
                self.log_file.unlink()
                psrl_logger.info(f"Cleared existing log file: {self.log_file}")
        except Exception as e:
            psrl_logger.warning(f"Failed to clear log file {self.log_file}: {e}")

    def clear_log_file(self):
        """Public method to manually clear the log file."""
        self._clear_log_file()
        # Re-write header after clearing
        self._write_log_header()

    def _write_log_header(self):
        """Write header information to the log file."""
        # Safely get cache config values (not currently used)
        # _cache_config = getattr(self.vllm_config, "cache_config", None)
        header = {
            "timestamp": datetime.now().isoformat(),
            "type": "header",
            "engine_index": self.engine_index,
        }
        self._write_log_entry(header)

    def _write_log_entry(self, entry: dict[str, Any]):
        """Write a log entry to the JSONL file."""
        try:
            # Use 'a' mode for appending, since file is cleared during initialization
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            psrl_logger.error(f"Failed to write log entry: {e}")

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        """
        Record and process detailed statistics from vLLM engine.

        Args:
            scheduler_stats: Statistics from vLLM scheduler
            iteration_stats: Statistics from vLLM iteration
            mm_cache_stats: Multi-modal cache statistics (not currently used)
            engine_idx: Engine index
        """
        if not self._warmup_completed:
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Track iteration stats for throughput calculation
        if iteration_stats is not None:
            self._track_iteration_stats(iteration_stats)

        if scheduler_stats is not None:
            # Calculate throughput
            prompt_throughput = self._get_throughput(self.num_prompt_tokens, current_time)
            generation_throughput = self._get_throughput(self.num_generation_tokens, current_time)

            # Reset counters for next interval
            self._reset(current_time)

            # Update last throughput values
            self.last_prompt_throughput = prompt_throughput
            self.last_generation_throughput = generation_throughput

            # Create detailed log entry
            time_to_first_tokens_iter = getattr(iteration_stats, "time_to_first_tokens_iter", [])
            if len(time_to_first_tokens_iter) == 0:
                time_to_first_tokens_iter = [0.0]
            inter_token_latencies_iter = getattr(iteration_stats, "inter_token_latencies_iter", [])
            if len(inter_token_latencies_iter) == 0:
                inter_token_latencies_iter = [0.0]
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "elapsed_time": elapsed_time,
                "type": "stats",
                "engine_index": self.engine_index,
                "scheduler_stats": {
                    "num_running_reqs": scheduler_stats.num_running_reqs,
                    "num_waiting_reqs": scheduler_stats.num_waiting_reqs,
                    "kv_cache_usage": scheduler_stats.kv_cache_usage,
                    "step_counter": getattr(scheduler_stats, "step_counter", 0),
                    "current_wave": getattr(scheduler_stats, "current_wave", 0),
                    "num_corrupted_reqs": getattr(scheduler_stats, "num_corrupted_reqs", 0),
                },
                "iteration_stats": {
                    "num_prompt_tokens": (getattr(iteration_stats, "num_prompt_tokens", 0) if iteration_stats else 0),
                    "num_generation_tokens": (
                        getattr(iteration_stats, "num_generation_tokens", 0) if iteration_stats else 0
                    ),
                    "num_preempted_reqs": (
                        getattr(iteration_stats, "num_preempted_reqs", 0) if iteration_stats else 0
                    ),
                    "num_finished_requests": (
                        len(getattr(iteration_stats, "finished_requests", [])) if iteration_stats else 0
                    ),
                    "time_to_first_tokens_avg": (np.mean(time_to_first_tokens_iter) if iteration_stats else 0),
                    "time_to_first_tokens_max": (np.max(time_to_first_tokens_iter) if iteration_stats else 0),
                    "inter_token_latencies_avg": (np.mean(inter_token_latencies_iter) if iteration_stats else 0),
                    "inter_token_latencies_max": (np.max(inter_token_latencies_iter) if iteration_stats else 0),
                },
                "throughput_stats": {
                    "prompt_throughput": prompt_throughput,
                    "generation_throughput": generation_throughput,
                    "total_throughput": prompt_throughput + generation_throughput,
                },
                "prefix_cache_stats": {
                    "reset": getattr(scheduler_stats.prefix_cache_stats, "reset", False),
                    "requests": getattr(scheduler_stats.prefix_cache_stats, "requests", 0),
                    "queries": getattr(scheduler_stats.prefix_cache_stats, "queries", 0),
                    "hits": getattr(scheduler_stats.prefix_cache_stats, "hits", 0),
                    "hit_rate": self._calculate_hit_rate(scheduler_stats.prefix_cache_stats),
                },
            }

            # Write to log file
            self._write_log_entry(log_entry)

            # Update last stats
            self.last_scheduler_stats = scheduler_stats

            # Store in history for analysis
            self.stats_history.append(log_entry)

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        """Track iteration stats for throughput calculation (similar to LoggingStatLogger)."""
        self.num_prompt_tokens += iteration_stats.num_prompt_tokens
        self.num_generation_tokens += iteration_stats.num_generation_tokens

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        """Calculate throughput for tracked stats (similar to LoggingStatLogger)."""
        delta_time = now - self.last_log_time
        if delta_time <= 0.0:
            return 0.0
        return float(tracked_stats / delta_time)

    def _reset(self, now: float):
        """Reset throughput tracking counters (similar to LoggingStatLogger)."""
        self.last_log_time = now
        self.num_prompt_tokens = 0
        self.num_generation_tokens = 0

    def _calculate_hit_rate(self, prefix_cache_stats) -> float:
        """Calculate prefix cache hit rate."""
        if not hasattr(prefix_cache_stats, "queries") or not hasattr(prefix_cache_stats, "hits"):
            return 0.0
        queries = getattr(prefix_cache_stats, "queries", 0)
        hits = getattr(prefix_cache_stats, "hits", 0)
        if queries == 0:
            return 0.0
        return float(hits / queries)

    def log_engine_initialized(self):
        """Log engine initialization with detailed cache information."""
        # Safely get cache config values
        cache_config = getattr(self.vllm_config, "cache_config", None)
        # num_gpu_blocks unused currently
        # _num_gpu_blocks = getattr(cache_config, "num_gpu_blocks", None) if cache_config else None

        cache_info = {
            "timestamp": datetime.now().isoformat(),
            "type": "engine_init",
            "engine_index": self.engine_index,
            "cache_config": {
                "block_size": (getattr(cache_config, "block_size", None) if cache_config else None),
                "gpu_memory_utilization": (
                    getattr(cache_config, "gpu_memory_utilization", None) if cache_config else None
                ),
            },
        }

        self._write_log_entry(cache_info)

    def log_generation_start(self, request_id: str, prompt_length: int, max_tokens: int):
        """Log the start of a generation request."""
        if not self._warmup_completed:
            return
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "generation_start",
            "engine_index": self.engine_index,
            "request_id": request_id,
            "prompt_length": prompt_length,
            "max_tokens": max_tokens,
        }
        self._write_log_entry(log_entry)

    def log_generation_end(
        self,
        request_id: str,
        generated_length: int,
        finish_reason: str,
        generation_time: float,
    ):
        """Log the end of a generation request."""
        if not self._warmup_completed:
            return
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "generation_end",
            "engine_index": self.engine_index,
            "request_id": request_id,
            "generated_length": generated_length,
            "finish_reason": finish_reason,
            "generation_time": generation_time,
        }
        self._write_log_entry(log_entry)

    def log_preempt_event(self, request_id: str, reason: str = "memory_pressure"):
        """Log a preemption event."""
        if not self._warmup_completed:
            return
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "preempt_event",
            "engine_index": self.engine_index,
            "request_id": request_id,
            "reason": reason,
        }
        self._write_log_entry(log_entry)

    def warmup_completion(self, warmup_iterations: int):
        """Log warmup phase completion."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "warmup_completion",
            "engine_index": self.engine_index,
            "warmup_iterations": warmup_iterations,
            "message": f"Completed {warmup_iterations} warmup iterations, starting performance test phase",
        }
        self._write_log_entry(log_entry)
        self._warmup_completed = True
        self.start_time = time.time()

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics from the collected data."""
        if not self.stats_history:
            return {}

        # Calculate summary statistics
        total_preempts = sum(
            entry["iteration_stats"]["num_preempted_reqs"]
            for entry in self.stats_history
            if "iteration_stats" in entry
        )

        avg_waiting = np.mean([entry["scheduler_stats"]["num_waiting_reqs"] for entry in self.stats_history])
        max_waiting = np.max([entry["scheduler_stats"]["num_waiting_reqs"] for entry in self.stats_history])
        avg_running = np.mean([entry["scheduler_stats"]["num_running_reqs"] for entry in self.stats_history])
        max_running = np.max([entry["scheduler_stats"]["num_running_reqs"] for entry in self.stats_history])
        avg_kv_cache_usage = np.mean([entry["scheduler_stats"]["kv_cache_usage"] for entry in self.stats_history])
        max_kv_cache_usage = np.max([entry["scheduler_stats"]["kv_cache_usage"] for entry in self.stats_history])

        # Calculate throughput statistics
        prompt_throughputs = [
            entry["throughput_stats"]["prompt_throughput"]
            for entry in self.stats_history
            if "throughput_stats" in entry
        ]
        generation_throughputs = [
            entry["throughput_stats"]["generation_throughput"]
            for entry in self.stats_history
            if "throughput_stats" in entry
        ]
        pure_generation_latencies = [
            entry["iteration_stats"]["inter_token_latencies_avg"]
            for entry in self.stats_history
            if "iteration_stats" in entry
            and entry["iteration_stats"]["time_to_first_tokens_avg"] == 0
            and entry["iteration_stats"]["inter_token_latencies_avg"] != 0
        ]

        avg_prompt_throughput = np.mean(prompt_throughputs) if prompt_throughputs else 0.0
        avg_generation_throughput = np.mean(generation_throughputs) if generation_throughputs else 0.0
        avg_pure_generation_latency = np.mean(pure_generation_latencies) if pure_generation_latencies else 0.0

        # Calculate prefix cache hit rate statistics
        hit_rates = [
            entry["prefix_cache_stats"]["hit_rate"] for entry in self.stats_history if "prefix_cache_stats" in entry
        ]
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0

        return {
            "total_preempts": total_preempts,
            "avg_waiting_requests": avg_waiting,
            "max_waiting_requests": max_waiting,
            "avg_running_requests": avg_running,
            "max_running_requests": max_running,
            "avg_kv_cache_usage": avg_kv_cache_usage,
            "max_kv_cache_usage": max_kv_cache_usage,
            "avg_pure_generation_latency": avg_pure_generation_latency,
            "avg_prompt_throughput": avg_prompt_throughput,
            "avg_generation_throughput": avg_generation_throughput,
            "avg_total_throughput": avg_prompt_throughput + avg_generation_throughput,
            "avg_prefix_cache_hit_rate": avg_hit_rate,
            "total_measurements": len(self.stats_history),
            "duration": time.time() - self.start_time,
        }
