"""
Per-turn session output writer used by the SessionRouter.

Companion to `TrajectoryWriter` (`psrl.utils.rollout.trajectory_writer`), which
writes one text file per trajectory *after* the episode completes. This writer
instead appends one JSON record per turn as it happens, so a crashed, hung, or
context-overflowed episode still leaves every turn up to the failure on disk.

All turns of a TITO session land in a single `<output_dir>/{session_id}.jsonl`.
The file is truncated the first time a given session writes to it, so a rerun
never appends onto a previous experiment's records.
"""

import json
import logging
import os
import threading

from omegaconf import DictConfig

psrl_logger = logging.getLogger(__file__)


class TurnOutputWriter:
    """
    Appends per-turn request/response records to `<output_dir>/{session_id}.jsonl`.

    Owned by the SessionRouter process and initialized from
    `config.psrl.agentic_rl.turn_output` via `from_config`.
    """

    def __init__(self, output_dir: str, enable: bool, include_response: bool = True) -> None:
        """
        Initialize the writer.

        Args:
            output_dir (str): Base directory for per-session output files.
            enable (bool): Whether writing is active.
            include_response (bool): Whether to record the response body too.
        """
        self.enable = enable
        self.output_dir = output_dir
        self.include_response = include_response
        # Sessions already truncated in this process, so the first write of a run
        # clears any file left by an earlier experiment while later turns append.
        self._started: set[str] = set()
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config: DictConfig) -> "TurnOutputWriter":
        """
        Build a `TurnOutputWriter` from `config.psrl.agentic_rl.turn_output`.

        Falls back to `<psrl.logging_path>/session_turns` when `dir` is empty.

        Args:
            config (DictConfig): Top-level training configuration object.

        Returns:
            TurnOutputWriter: Configured writer instance.
        """
        turn_cfg = config.psrl.agentic_rl.get("turn_output", {})
        enable = bool(turn_cfg.get("enable", False))
        include_response = bool(turn_cfg.get("include_response", True))
        dir_ = str(turn_cfg.get("dir", "") or "")
        if not dir_:
            logging_path = str(getattr(config.psrl, "logging_path", "") or "")
            base = logging_path if logging_path else os.getcwd()
            dir_ = os.path.join(base, "session_turns")
        dir_ = os.path.abspath(os.path.expanduser(dir_))
        return cls(output_dir=dir_, enable=enable, include_response=include_response)

    def write_turn(
        self,
        session_id: str,
        turn: int,
        request_body: bytes,
        response_body: bytes | None = None,
        trajectory_id: int | None = None,
        status: int | None = None,
    ) -> None:
        """
        Append one turn record for `session_id`.

        Bodies are stored as parsed JSON when they decode cleanly, else as raw
        text, so a malformed or error response is still recorded rather than
        dropped. Never raises: an output failure must not fail the rollout.

        Args:
            session_id (str): TITO session ID that selects the output file.
            turn (int): Zero-based turn index within the session.
            request_body (bytes): Raw chat-completion request body.
            response_body (bytes | None): Raw upstream response body, if any.
            trajectory_id (int | None): Trajectory ID when the caller tracks one.
            status (int | None): Upstream HTTP status code.
        """
        if not self.enable:
            return

        record: dict = {"turn": turn, "session_id": session_id}
        if trajectory_id is not None:
            record["trajectory_id"] = trajectory_id
        if status is not None:
            record["status"] = status
        record["request"] = _decode_body(request_body)
        if self.include_response and response_body is not None:
            record["response"] = _decode_body(response_body)

        path = os.path.join(self.output_dir, f"{session_id}.jsonl")
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with self._lock:
                # Truncate on this run's first write for the session, then append.
                mode = "a" if session_id in self._started else "w"
                self._started.add(session_id)
                with open(path, mode) as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError) as e:
            psrl_logger.warning(f"Failed to write turn={turn}, session={session_id!r}, path={path!r}: {e}.")


def _decode_body(body: bytes) -> object:
    """Return `body` as parsed JSON, falling back to text then a byte count."""
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        try:
            return body.decode("utf-8", errors="replace")
        except Exception:
            return f"<{len(body)} undecodable bytes>"
