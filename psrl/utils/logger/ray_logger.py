import inspect
import logging
import os
import socket
import sys
import time
from contextlib import contextmanager
from enum import Enum


def get_worker_info():
    """Get the worker info from the environment variables."""
    import torch

    worker_ip = os.getenv("LOCAL_IP", None)
    if worker_ip is None:
        worker_ip = socket.gethostbyname(socket.gethostname())
    worker_gpu = None
    if torch.cuda.is_available():
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_devices = list(map(int, visible_devices.split(",")))
            current_logical = torch.cuda.current_device()
            current_physical = visible_devices[current_logical]
            worker_gpu = f"GPU {current_physical}"
    return worker_ip, worker_gpu


class EventType(Enum):
    PULL = "PULL"
    PUSH = "PUSH"
    BUFFER_READY = "BUFFER_READY"
    INIT = "INIT"
    TRAIN = "TRAIN"
    GEN = "GEN"
    VAL = "VAL"
    WAIT = "WAIT"
    SWITCH = "SWITCH"
    OTHER = "OTHER"


def _log_with_caller_info(psrl_logger: logging.Logger, level: int, message: str):
    """Log a message with the caller's file and line information."""
    if not psrl_logger.isEnabledFor(level):
        return

    try:
        psrl_logger.log(level, message, stacklevel=3)
    except TypeError:
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back.f_back
            pathname = caller_frame.f_code.co_filename
            lineno = caller_frame.f_lineno
            record = psrl_logger.makeRecord(psrl_logger.name, level, pathname, lineno, message, (), None)
            psrl_logger.handle(record)
        finally:
            del frame


@contextmanager
def log_dual_events(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
):
    start_time = time.time()
    log_begin_event(message, psrl_logger, level, event_type)
    try:
        yield
    finally:
        end_time = time.time()
        log_end_event(message, psrl_logger, level, event_type, end_time - start_time)


def log_single_event(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
):
    _log_with_caller_info(psrl_logger, level, f"[Single Event] {event_type.value} - {message}")


def log_begin_event(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
):
    _log_with_caller_info(psrl_logger, level, f"[Begin Event] {event_type.value} - {message}")


def log_end_event(
    message: str,
    psrl_logger: logging.Logger,
    level: int = logging.INFO,
    event_type: EventType = EventType.OTHER,
    duration: float = None,
):
    if duration is None:
        _log_with_caller_info(psrl_logger, level, f"[End Event] {event_type.value} - {message}")
    else:
        _log_with_caller_info(
            psrl_logger,
            level,
            f"[End Event] {event_type.value} - {message} - Time taken: {duration:.2f} seconds",
        )


class FileOnlyHandler(logging.Handler):
    """
    Write formatted records to a file this handler owns.

    Nested `logging.FileHandler` objects are closed by `logging.shutdown` and
    `dictConfig`. CPython then drops later writes when the mode is `w`.
    """

    terminator = "\n"

    def __init__(self, log_dir: str, log_prefix: str):
        super().__init__()
        self.log_prefix = log_prefix
        log_dir = os.path.expanduser(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, log_prefix + ".log")
        self.setFormatter(logging.Formatter("%(asctime)s - %(filename)s - %(lineno)d - %(message)s"))
        self._stream = None
        # Truncate at construction so a new process replaces the previous run.
        self._open("w")

    def _open(self, mode: str) -> None:
        self._stream = open(self.file_path, mode, encoding="utf-8")

    def _ensure_stream(self) -> None:
        if self._stream is None or self._stream.closed:
            self._open("a")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_stream()
            stream = self._stream
            assert stream is not None, "Log file stream is not open."
            stream.write(self.format(record) + self.terminator)
            stream.flush()
        except Exception:
            self.handleError(record)

    def flush(self) -> None:
        self.acquire()
        try:
            if self._stream is not None and not self._stream.closed:
                self._stream.flush()
        finally:
            self.release()

    def close(self) -> None:
        self.acquire()
        try:
            if self._stream is not None and not self._stream.closed:
                self._stream.flush()
                self._stream.close()
            self._stream = None
        finally:
            self.release()
            super().close()


class DualOutputHandler(FileOnlyHandler):
    """
    Write formatted records to a file this handler owns and to stderr.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            sys.stderr.write(f"<{self.log_prefix}> - {record.getMessage()}{self.terminator}")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)
