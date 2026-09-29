"""Utilities for bridging synchronous callers and asyncio event loops."""

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


def run_coroutine_on_loop(
    event_loop: asyncio.AbstractEventLoop,
    coroutine: Coroutine[Any, Any, _T],
    *,
    timeout_s: float | None = None,
) -> _T:
    """Submit a coroutine to an event loop and wait synchronously for its result."""
    future = asyncio.run_coroutine_threadsafe(coroutine, event_loop)
    try:
        return future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        future.cancel()
        raise TimeoutError("Operation timed out while waiting for its owner event loop.") from None
