"""
Per-trajectory text file writer shared by all agent loops.
"""

import logging
import os

from omegaconf import DictConfig

psrl_logger = logging.getLogger(__file__)


class TrajectoryWriter:
    """
    Writes per-trajectory text files under `<output_dir>/v{version}/{traj_id}.txt`.

    Shared by all agent loops. Initialized once per loop instance from
    `config.psrl.agentic_rl.trajectory_output` via `from_config`.
    """

    def __init__(self, output_dir: str, enable: bool) -> None:
        """
        Initialize the writer with an output directory and enable flag.

        Args:
            output_dir (str): Base directory for trajectory files.
            enable (bool): Whether writing is active.
        """
        self.enable = enable
        self.output_dir = output_dir

    @classmethod
    def from_config(cls, config: DictConfig) -> "TrajectoryWriter":
        """
        Build a `TrajectoryWriter` from `config.psrl.agentic_rl.trajectory_output`.

        Falls back to `<psrl.logging_path>/trajectories` when `dir` is empty.

        Args:
            config (DictConfig): Top-level training configuration object.

        Returns:
            TrajectoryWriter: Configured writer instance.
        """
        traj_cfg = config.psrl.agentic_rl.get("trajectory_output", {})
        enable = bool(traj_cfg.get("enable", False))
        dir_ = str(traj_cfg.get("dir", "") or "")
        if not dir_:
            logging_path = str(getattr(config.psrl, "logging_path", "") or "")
            base = logging_path if logging_path else os.getcwd()
            dir_ = os.path.join(base, "trajectories")
        dir_ = os.path.abspath(os.path.expanduser(dir_))
        return cls(output_dir=dir_, enable=enable)

    def write(self, version: int, traj_id: str, text: str) -> str:
        """
        Write `text` to `<output_dir>/v{version}/{traj_id}.txt`.

        Args:
            version (int): Model version tag used as the sub-directory name.
            traj_id (str): Trajectory identifier used as the file stem.
            text (str): Full trajectory text to write.

        Returns:
            str: The written file path, or empty string if disabled or on error.
        """
        if not self.enable:
            return ""
        version_dir = os.path.join(self.output_dir, f"v{version}")
        os.makedirs(version_dir, exist_ok=True)
        path = os.path.join(version_dir, f"{traj_id}.txt")
        try:
            with open(path, "w") as f:
                f.write(text)
            return path
        except OSError as e:
            psrl_logger.warning(f"Failed to write trajectory to {path!r}: {e}.")
            return ""

    def append(self, path: str, text: str) -> None:
        """
        Append `text` to an existing trajectory file at `path`.

        No-op when disabled or `path` is empty.

        Args:
            path (str): Absolute path to the trajectory file.
            text (str): Text to append.
        """
        if not self.enable or not path:
            return
        try:
            with open(path, "a") as f:
                f.write(text)
        except OSError as e:
            psrl_logger.warning(f"Failed to append to trajectory {path!r}: {e}.")
