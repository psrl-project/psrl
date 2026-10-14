"""Limit cross-process concurrency with `fcntl`-locked slot files."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


class SlotManager:
    """Manage a fixed pool of cross-process run slots via file locks."""

    @staticmethod
    def slot_lock_dir(output_dir: str, prefix: str = "psrl_slots") -> str:
        """Return the lock directory for a given output directory.

        The directory is placed under the system temp dir, keyed by a hash
        of *output_dir* so that different output dirs get independent pools.
        """
        digest = hashlib.sha1(os.path.abspath(output_dir).encode("utf-8")).hexdigest()[:12]
        return os.path.join(tempfile.gettempdir(), f"{prefix}_{digest}")

    @classmethod
    async def acquire(
        cls,
        max_slots: int,
        output_dir: str,
        *,
        prefix: str = "psrl_slots",
        poll_interval: float = 0.2,
    ) -> tuple[int, int] | None:
        """Acquire one cross-process run slot via fcntl file lock.

        Args:
            max_slots: Maximum number of parallel slots. If <= 0,
                returns ``None`` immediately (no limiting).
            output_dir: Directory used to derive the lock file location.
            prefix: Prefix for the lock directory name.
            poll_interval: Seconds between retries when all slots are busy.

        Returns:
            A ``(fd, slot_index)`` tuple on success, or ``None`` when
            *max_slots* <= 0 (slot limiting disabled).
        """
        if max_slots <= 0:
            return None

        lock_dir = cls.slot_lock_dir(output_dir, prefix=prefix)
        logger.info(f"Slot lock dir: {lock_dir!r}, max_slots={max_slots}.")
        os.makedirs(lock_dir, exist_ok=True)

        while True:
            for slot_idx in range(max_slots):
                lock_path = os.path.join(lock_dir, f"slot_{slot_idx}.lock")
                fd = os.open(
                    lock_path,
                    os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
                    0o666,
                )
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.ftruncate(fd, 0)
                    os.write(fd, f"pid={os.getpid()}\n".encode())
                    return fd, slot_idx
                except BlockingIOError:
                    os.close(fd)

            await asyncio.sleep(poll_interval)

    @staticmethod
    def release(run_slot: tuple[int, int] | None) -> None:
        """Release a previously acquired run slot.

        Safe to call with ``None`` (no-op).
        """
        if run_slot is None:
            return
        fd, _ = run_slot
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
