"""Log GPU memory periodically and on demand."""

import logging
import os
import threading
from collections.abc import Callable
from typing import TypeVar

import torch

from .ray_logger import DualOutputHandler

F = TypeVar("F", bound=Callable)


def _get_device_memory_info(device_id: int, unit: str = "GB", precision: int = 2) -> dict:
    """Get memory info for one GPU device.

    Args:
        device_id: CUDA device index.
        unit: "GB", "MB", or "KB".
        precision: Decimal places for numeric strings.

    Returns:
        Dict with keys: allocated, reserved, free, total, used (device-level).
        All values are formatted strings in the given unit.
    """
    assert unit in ("GB", "MB", "KB"), f"Expected memory unit GB, MB, or KB, got {unit!r}."
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024

    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)

    try:
        free_mem, total_mem = torch.cuda.mem_get_info(device_id)
        used_device = total_mem - free_mem
    except Exception:
        free_mem = total_mem = used_device = 0

    return {
        "allocated": f"{allocated / divisor:.{precision}f}",
        "reserved": f"{reserved / divisor:.{precision}f}",
        "free": f"{free_mem / divisor:.{precision}f}",
        "total": f"{total_mem / divisor:.{precision}f}",
        "used": f"{used_device / divisor:.{precision}f}",
    }


def get_all_gpu_memory_info(unit: str = "GB", precision: int = 2) -> list[dict]:
    """Get memory info for all visible CUDA devices.

    Returns:
        List of dicts, one per device, each from _get_device_memory_info.
    """
    dev_count = torch.cuda.device_count()
    return [_get_device_memory_info(i, unit=unit, precision=precision) for i in range(dev_count)]


def format_memory_message(device_infos: list[dict], prefix: str = "", unit: str = "GB") -> str:
    """Format device memory infos into a single log message."""
    parts = []
    for i, info in enumerate(device_infos):
        part = (
            f"Device {i}: allocated={info['allocated']} {unit}, reserved={info['reserved']} {unit}, "
            f"device used/total={info['used']}/{info['total']} {unit}"
        )
        parts.append(part)
    body = "; ".join(parts)
    if prefix:
        return f"{prefix} | {body}"
    return body


def log_gpu_memory_now(
    prefix: str = "",
    logger: logging.Logger | None = None,
    level: int = logging.INFO,
    unit: str = "GB",
) -> str:
    """Log current GPU memory for all devices at this moment.

    Args:
        prefix: Optional prefix string for the log line.
        logger: If given, log via this logger at `level`. Otherwise only return the message.
        level: Log level when logger is provided.
        unit: "GB", "MB", or "KB".

    Returns:
        The formatted message string.
    """
    assert torch.cuda.is_available(), "log_gpu_memory_now requires GPU (torch.cuda.is_available())."
    device_infos = get_all_gpu_memory_info(unit=unit)
    message = format_memory_message(device_infos, prefix=prefix, unit=unit)
    if logger is not None:
        logger.log(level, message)
    return message


class MemoryLogger:
    """Log GPU memory periodically and on demand.

    Memory is written to `logging_path / (log_prefix + "_memory.log")`.

    Example:
        logger = MemoryLogger("/path/to/logs", "TrainWorker_R0", interval_seconds=60.0)
        logger.start()
        # ... later, at a specific point:
        logger.log_now(prefix="after_forward")
        logger.stop()
    """

    def __init__(
        self,
        logging_path: str,
        log_prefix: str,
        interval_seconds: float,
        level: int = logging.INFO,
        unit: str = "GB",
    ):
        assert torch.cuda.is_available(), "MemoryLogger requires GPU (torch.cuda.is_available())."
        assert interval_seconds > 0, "interval_seconds must be positive."

        self.logging_path = os.path.expanduser(logging_path)
        self.log_prefix = log_prefix
        self.interval_seconds = interval_seconds
        self.level = level
        self.unit = unit

        self._memory_logger = logging.getLogger(f"{__name__}.MemoryLogger.{log_prefix}")
        self._memory_logger.setLevel(logging.DEBUG)
        self._memory_logger.propagate = False
        memory_prefix = f"{log_prefix}_memory"
        self._memory_logger.addHandler(DualOutputHandler(self.logging_path, memory_prefix))

        self._memory_log_path = os.path.join(self.logging_path, f"{memory_prefix}.log")

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _periodic_log(self) -> None:
        while not self._stop_event.wait(timeout=self.interval_seconds):
            log_gpu_memory_now(
                prefix=f"[periodic {self.log_prefix}]",
                logger=self._memory_logger,
                level=self.level,
                unit=self.unit,
            )

    def start(self) -> None:
        """Start the background thread that logs GPU memory every interval_seconds."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._periodic_log, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the periodic logging thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 2)
            self._thread = None

    def log_now(self, prefix: str = "") -> str:
        """Log current GPU memory at this moment (for external call). Uses this instance's file and level.

        Args:
            prefix: Optional prefix for this log line (e.g. "after_backward", "step_100").

        Returns:
            The formatted message string.
        """
        return log_gpu_memory_now(
            prefix=prefix or self.log_prefix,
            logger=self._memory_logger,
            level=self.level,
            unit=self.unit,
        )

    @property
    def memory_log_path(self) -> str:
        """Path to the file where memory logs are written."""
        return self._memory_log_path


def gpu_memory_logger_decorator(log_only_rank_0: bool = True) -> Callable[[F], F]:
    """Log GPU memory before and after the wrapped method.

    No logger is passed in. At call time the decorator checks self.memory_logger. If it is None
    (e.g. no GPU), the method runs without logging. Otherwise it logs "Before {name}" and
    "After {name}" via self.memory_logger.log_now(prefix=...).

    Intended for worker methods (self has .memory_logger and optionally .rank).

    Example:
        @gpu_memory_logger_decorator(role="megatron actor", log_only_rank_0=False)
        def update_actor(self, data: DataProto):
            ...
    """

    def decorator(func: F) -> F:
        def wrapper(self, *args, **kwargs):
            memory_logger = getattr(self, "memory_logger", None)
            if memory_logger is None:
                return func(self, *args, **kwargs)
            if log_only_rank_0:
                if hasattr(self, "rank"):
                    if self.rank != 0:
                        return func(self, *args, **kwargs)
                else:
                    try:
                        import torch.distributed as dist

                        if dist.is_initialized() and dist.get_rank() != 0:
                            return func(self, *args, **kwargs)
                    except Exception:
                        pass
            name = func.__name__
            prefix_before = f"Before {name}"
            prefix_after = f"After {name}"
            memory_logger.log_now(prefix=prefix_before)
            try:
                return func(self, *args, **kwargs)
            finally:
                memory_logger.log_now(prefix=prefix_after)

        return wrapper  # type: ignore[return-value]

    return decorator
