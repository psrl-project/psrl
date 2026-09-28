"""Streaming JSONL writer for collected rollout trajectories.

Companion to `TrajectoryWriter` (readable per-trajectory text) and
`TurnOutputWriter` (per-turn request/response), which the agent loops and the
SessionRouter already populate. This writer owns the machine-readable index: one
JSON record per trajectory, appended and flushed as each episode lands, so a run
killed halfway leaves every completed episode on disk and a rerun can skip them.
"""

import json
import logging
import os
from collections import Counter

import ray

from psrl.utils.common.serialization import json_encode_default
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

ROLLOUT_FILE = "rollout.jsonl"
SUMMARY_FILE = "summary.json"


@ray.remote
class RolloutOutputWriter:
    """Append rollout records to one JSONL file, and report what is already done.

    A single actor owns the file, so no locking is needed and no two writers can
    interleave a partial line. Each record is flushed on write: an episode can cost
    an hour of container time, so losing a completed one to a buffered write is
    worse than the syscall.
    """

    def __init__(self, output_dir: str, resume: bool = True, logging_path: str = "") -> None:
        """Open the output file and index any prior run.

        Args:
            output_dir (str): Directory holding `rollout.jsonl` and `summary.json`.
            resume (bool): Whether to keep an existing `rollout.jsonl` and skip the
                uids in it. `False` truncates the file.
            logging_path (str): Directory for this actor's own log file.
        """
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.rollout_path = os.path.join(self.output_dir, ROLLOUT_FILE)

        if logging_path:
            psrl_logger.addHandler(DualOutputHandler(logging_path, "RolloutOutputWriter"))

        self.terminate_reasons: Counter = Counter()
        self.record_count = 0
        self._completed_uids: set = set()

        if resume:
            self._completed_uids = self._index_existing()
            psrl_logger.info(
                "Resuming into %s: %d record(s) across %d uid(s) already present.",
                self.rollout_path,
                self.record_count,
                len(self._completed_uids),
            )
        elif os.path.exists(self.rollout_path):
            os.remove(self.rollout_path)
            psrl_logger.info("Truncated %s because resume is disabled.", self.rollout_path)

        self._file = open(self.rollout_path, "a", encoding="utf-8")

    def _index_existing(self) -> set:
        """Read a prior run's records, tolerating a truncated final line.

        A run killed mid-write leaves a partial last line. It is dropped rather
        than parsed, because a half-written record names an episode whose result
        was never fully captured and which therefore must run again.

        Returns:
            set: The `uid` of every intact record found.
        """
        if not os.path.exists(self.rollout_path):
            return set()

        completed = set()
        malformed = 0
        with open(self.rollout_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                uid = record.get("uid")
                if uid is not None:
                    completed.add(uid)
                self.record_count += 1
                reason = record.get("terminate_reason")
                if reason:
                    self.terminate_reasons[reason] += 1
        if malformed:
            psrl_logger.warning(
                "Discarded %d unparseable line(s) in %s, most likely a run killed mid-write. "
                "Their episodes will be collected again.",
                malformed,
                self.rollout_path,
            )
        return completed

    def completed_uids(self) -> set:
        """Return the uids already recorded, so the feed can skip them."""
        return set(self._completed_uids)

    def append(self, records: list[dict]) -> int:
        """Append one episode's records and flush them.

        Args:
            records (list[dict]): One record per trajectory of a single episode.

        Returns:
            int: Number of records written.
        """
        if not records:
            return 0
        for record in records:
            self._file.write(json.dumps(record, ensure_ascii=False, default=json_encode_default) + "\n")
            uid = record.get("uid")
            if uid is not None:
                self._completed_uids.add(uid)
            self.record_count += 1
            reason = record.get("terminate_reason")
            if reason:
                self.terminate_reasons[reason] += 1
        self._file.flush()
        return len(records)

    def progress(self) -> dict:
        """Return the live counters, for periodic logging by the driver."""
        return {
            "records": self.record_count,
            "uids": len(self._completed_uids),
            "terminate_reasons": dict(self.terminate_reasons),
        }

    def finalize(self, extra: dict | None = None) -> dict:
        """Close the file and write `summary.json`.

        Args:
            extra (dict | None): Run metadata to record alongside the counters.

        Returns:
            dict: The summary that was written.
        """
        summary = {
            "rollout_file": self.rollout_path,
            "records": self.record_count,
            "uids": len(self._completed_uids),
            "terminate_reasons": dict(self.terminate_reasons),
            **(extra or {}),
        }
        summary_path = os.path.join(self.output_dir, SUMMARY_FILE)
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=json_encode_default)

        if not self._file.closed:
            self._file.flush()
            self._file.close()

        psrl_logger.info(
            "Wrote %d record(s) to %s. Termination histogram: %s.",
            self.record_count,
            self.rollout_path,
            dict(self.terminate_reasons),
        )
        return summary
