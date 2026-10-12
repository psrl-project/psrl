import json
from dataclasses import dataclass
from typing import Final
from unittest.mock import MagicMock

import pytest

# --- Minimal Stubs Without vLLM or Ray Imports ---


@dataclass
class EngineStats:
    replica_idx: Final[int]
    data_parallel_rank: Final[int]
    model_version: Final[int]
    snapshot: Final[dict]


RolloutInstanceId = tuple  # (str, int)


# --- Tests ---


@pytest.mark.unit
def test_write_config_creates_file(tmp_path):
    """write_config() writes stats_config.json with correct fields."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    recorder = StatsRecorder(cfg, str(tmp_path))
    recorder.write_config(routing_strategy="request_num_balance", partial_rollout=False)

    config_path = tmp_path / "stats_config.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["routing_strategy"] == "request_num_balance"
    assert data["partial_rollout"] is False
    assert data["interval_in_s"] == 5.0
    assert "started_at" in data


@pytest.mark.unit
def test_record_creates_per_instance_files(tmp_path):
    """record() creates one JSONL file per (replica_id, dp_rank)."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    recorder = StatsRecorder(cfg, str(tmp_path))

    instance_to_engine_status = {
        ("rollout-0", 0): EngineStats(
            replica_idx=0,
            data_parallel_rank=0,
            model_version=1,
            snapshot={
                "scheduler_stats": {"num_running_reqs": 4, "num_waiting_reqs": 1, "kv_cache_usage": 0.5},
                "generation_throughput": 200.0,
            },
        ),
        ("rollout-1", 0): EngineStats(
            replica_idx=1,
            data_parallel_rank=0,
            model_version=1,
            snapshot={
                "scheduler_stats": {"num_running_reqs": 6, "num_waiting_reqs": 0, "kv_cache_usage": 0.7},
                "generation_throughput": 300.0,
            },
        ),
    }
    recorder.record(instance_to_engine_status)
    recorder.close()

    assert (tmp_path / "stats_r0_dp0.jsonl").exists()
    assert (tmp_path / "stats_r1_dp0.jsonl").exists()


@pytest.mark.unit
def test_record_row_schema(tmp_path):
    """Each JSONL row has the expected fields and types."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    recorder = StatsRecorder(cfg, str(tmp_path))

    instance_to_engine_status = {
        ("rollout-0", 0): EngineStats(
            replica_idx=0,
            data_parallel_rank=0,
            model_version=42,
            snapshot={
                "scheduler_stats": {"num_running_reqs": 8, "num_waiting_reqs": 2, "kv_cache_usage": 0.63},
                "generation_throughput": 412.5,
                "iteration_stats": {
                    "avg_time_to_first_tokens": 0.034,
                    "avg_inter_token_latencies": 0.009,
                },
            },
        ),
    }
    recorder.record(instance_to_engine_status)
    recorder.close()

    rows = [json.loads(line) for line in (tmp_path / "stats_r0_dp0.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["model_version"] == 42
    assert row["num_running_reqs"] == 8
    assert row["num_waiting_reqs"] == 2
    assert abs(row["kv_cache_usage"] - 0.63) < 1e-6
    assert abs(row["generation_throughput"] - 412.5) < 1e-6
    assert abs(row["avg_ttft"] - 0.034) < 1e-6
    assert abs(row["avg_itl"] - 0.009) < 1e-6
    assert "ts" in row


@pytest.mark.unit
def test_record_nullable_iteration_stats(tmp_path):
    """avg_ttft and avg_itl are null when iteration_stats is absent."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    recorder = StatsRecorder(cfg, str(tmp_path))

    instance_to_engine_status = {
        ("rollout-0", 0): EngineStats(
            replica_idx=0,
            data_parallel_rank=0,
            model_version=1,
            snapshot={
                "scheduler_stats": {"num_running_reqs": 0, "num_waiting_reqs": 0, "kv_cache_usage": 0.0},
                "generation_throughput": 0.0,
            },
        ),
    }
    recorder.record(instance_to_engine_status)
    recorder.close()

    rows = [json.loads(line) for line in (tmp_path / "stats_r0_dp0.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["avg_ttft"] is None
    assert row["avg_itl"] is None


@pytest.mark.unit
def test_record_empty_dict_skips_silently(tmp_path):
    """record() with empty dict creates no files."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    recorder = StatsRecorder(cfg, str(tmp_path))
    recorder.record({})
    recorder.close()

    jsonl_files = list(tmp_path.glob("stats_*.jsonl"))
    assert jsonl_files == []


@pytest.mark.unit
def test_record_multiple_ticks_appends(tmp_path):
    """Calling record() twice appends two rows to the same file."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    recorder = StatsRecorder(cfg, str(tmp_path))

    status = {
        ("rollout-0", 0): EngineStats(
            replica_idx=0,
            data_parallel_rank=0,
            model_version=1,
            snapshot={
                "scheduler_stats": {"num_running_reqs": 1, "num_waiting_reqs": 0, "kv_cache_usage": 0.1},
                "generation_throughput": 100.0,
            },
        ),
    }
    recorder.record(status)
    recorder.record(status)
    recorder.close()

    rows = (tmp_path / "stats_r0_dp0.jsonl").read_text().strip().splitlines()
    assert len(rows) == 2


@pytest.mark.unit
def test_makedirs_creates_logging_path(tmp_path):
    """StatsRecorder.__init__ creates logging_path if it doesn't exist."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    new_dir = tmp_path / "nested" / "logs"
    assert not new_dir.exists()

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    StatsRecorder(cfg, str(new_dir))
    assert new_dir.exists()


@pytest.mark.unit
def test_record_smg_routing_status_uses_selecting(tmp_path):
    """SMG routing-loop snapshots record the worker-selection barrier state."""
    from psrl.workers.gen.stats_recorder import StatsRecorder

    cfg = MagicMock()
    cfg.status_collection.stats_recorder.interval_in_s = 5.0
    recorder = StatsRecorder(cfg, str(tmp_path))

    routing_loop_status = {
        "queue_len": 2,
        "pending_request_num": 2,
        "running_request_num": 1,
        "running_tasks": 3,
        "paused": True,
        "selecting": False,
        "queue_keys": [1],
        "partition_queue_lens": {"1": 2},
    }
    recorder.record_smg_routing_status(routing_loop_status, workers_stats=[])
    recorder.close()

    rows = [json.loads(line) for line in (tmp_path / "smg_routing_status.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["selecting"] is False
    assert "routing" not in row
