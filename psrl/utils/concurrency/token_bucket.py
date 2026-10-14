from __future__ import annotations

import asyncio
import threading
import time


class TokenBucket:
    """Limit synchronous and asynchronous callers with a thread-safe token bucket.

    Args:
        rate: tokens replenished per second.
        capacity: maximum burst size. Defaults to `rate`.
        init_tokens: initial tokens. Defaults to `capacity`.

    Notes:
        - Uses `time.monotonic()` to avoid wall-clock jumps.
        - Internal state is protected by a `threading.Lock`.
    """

    def __init__(self, rate: float, capacity: float | None = None, init_tokens: float | None = None):
        if rate <= 0:
            raise ValueError("rate must be > 0")

        self._rate = rate
        self._capacity = capacity if capacity is not None else rate

        self._tokens = init_tokens if init_tokens is not None else self._capacity
        self._tokens = max(0.0, min(self._capacity, self._tokens))

        self._updated_at = time.monotonic()

        self._lock = threading.Lock()

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    def _refill_locked(self, now: float) -> None:
        """Refill tokens based on elapsed time since last update.

        IMPORTANT: Callers must hold `_lock`.
        """
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._updated_at = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

    def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens immediately.

        Returns True if successful, False otherwise.
        """
        if tokens <= 0:
            return True

        now = time.monotonic()
        with self._lock:
            self._refill_locked(now)
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def time_to_availability(self, tokens: float = 1.0) -> float:
        """Return minimal seconds to wait until `tokens` can be acquired."""
        if tokens <= 0:
            return 0.0

        now = time.monotonic()
        with self._lock:
            self._refill_locked(now)
            missing = tokens - self._tokens
            if missing <= 0:
                return 0.0
            return missing / self._rate

    async def async_acquire(self, tokens: float = 1.0, *, max_sleep: float = 0.1) -> None:
        """Wait until tokens are available and then acquire.

        Args:
            tokens: number of tokens to acquire.
            max_sleep: cap the sleep time to keep latency/jitter under control.
        """
        if tokens <= 0:
            return

        if self.acquire(tokens):
            return

        while True:
            wait_s = self.time_to_availability(tokens)
            await asyncio.sleep(min(max_sleep, max(0.0, wait_s)))

            if self.acquire(tokens):
                return
