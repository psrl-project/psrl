"""Run agent-loop HTTP connections on a dedicated I/O thread.

The Ray actor's event loop sees only lightweight `asyncio.Future` completions
and no socket callbacks.
"""

import asyncio
import logging
import threading
from typing import Any

import aiohttp

from psrl.utils.common.http_utils import (
    JsonHttpResponse,
    RequestAbortedByGatewayError,
    _classify_http_error,
    _parse_body,
    _raise_for_status,
    filter_http_headers,
)

psrl_logger = logging.getLogger(__name__)

__all__ = ["HttpIOThread", "get_http_io_thread", "init_http_io_thread"]


class HttpIOThread:
    """Background thread with an independent asyncio event loop for HTTP I/O."""

    def __init__(self, max_concurrency: int = 2048, dns_ttl: int = 300):
        self._max_concurrency = max_concurrency
        self._dns_ttl = dns_ttl
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._started = threading.Event()

    def start(self) -> None:
        """Start the background I/O thread and block until its loop is ready."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="psrl-http-io", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=30):
            raise RuntimeError("HttpIOThread failed to start within 30 s.")
        psrl_logger.info(
            "HttpIOThread started: max_concurrency=%d, thread=%s.",
            self._max_concurrency,
            self._thread.name,
        )

    def stop(self) -> None:
        """Gracefully stop the I/O thread."""
        if self._loop is None:
            return

        async def _cleanup() -> None:
            if self._session and not self._session.closed:
                await self._session.close()

        asyncio.run_coroutine_threadsafe(_cleanup(), self._loop).result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        psrl_logger.info("HttpIOThread stopped.")

    def _run_loop(self) -> None:
        """Thread entry-point: create event loop and run forever."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._init_session())
        self._started.set()
        self._loop.run_forever()

    async def _init_session(self) -> None:
        """Initialize aiohttp session and semaphore on the I/O loop."""
        connector = aiohttp.TCPConnector(
            limit=self._max_concurrency,
            ttl_dns_cache=self._dns_ttl,
            enable_cleanup_closed=True,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None),
        )
        self._semaphore = asyncio.Semaphore(self._max_concurrency)

    async def _do_request(
        self,
        method: str,
        url: str,
        payload: Any | None,
        headers: dict[str, str] | None,
        max_retries: int,
    ) -> JsonHttpResponse:
        """Execute one HTTP request on the I/O thread's event loop."""
        async with self._semaphore:
            retry_count = 0
            while True:
                try:
                    kwargs: dict[str, Any] = {"headers": headers}
                    if payload is not None:
                        kwargs["json"] = payload
                    async with self._session.request(method, url, **kwargs) as resp:
                        body = await resp.read()
                        status = resp.status
                        resp_headers = filter_http_headers(resp.headers)
                    _raise_for_status(status, body, resp)
                    data, text = _parse_body(body)
                    return JsonHttpResponse(status=status, data=data, headers=resp_headers, text=text)
                except Exception as e:
                    http_error = _classify_http_error(e, headers)
                    if isinstance(http_error, RequestAbortedByGatewayError):
                        raise http_error from e
                    retry_count += 1
                    if retry_count >= max_retries:
                        raise
                    psrl_logger.debug(
                        "HttpIOThread retry %d/%d for %s: %s.",
                        retry_count,
                        max_retries,
                        url,
                        e,
                    )
                    await asyncio.sleep(1)

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        payload: Any | None = None,
        headers: dict[str, str] | None = None,
        max_retries: int = 5,
    ) -> JsonHttpResponse:
        """Submit an HTTP request and await the result from the caller's event loop.

        The actual socket I/O executes on the dedicated thread. The caller
        awaits only a lightweight `Future` completion.
        """
        assert self._loop is not None, "HttpIOThread not started."
        thread_future = asyncio.run_coroutine_threadsafe(
            self._do_request(method, url, payload, headers, max_retries),
            self._loop,
        )
        return await asyncio.wrap_future(thread_future)


_http_io_thread: HttpIOThread | None = None


def get_http_io_thread() -> HttpIOThread:
    """Return the global ``HttpIOThread`` singleton."""
    if _http_io_thread is None:
        raise RuntimeError("HttpIOThread not initialized. Call init_http_io_thread() first.")
    return _http_io_thread


def init_http_io_thread(
    server_concurrency: int,
    rollout_engine_num: int,
    *,
    producer_count: int = 1,
    producer_index: int = 0,
) -> HttpIOThread:
    """Initialize the global ``HttpIOThread`` singleton.

    effective = server_concurrency * rollout_engine_num / producer_count

    Args:
        server_concurrency: ``psrl.rollout_gateway.server_max_concurrency`` (e.g. 256).
        rollout_engine_num: Total engine count (``n_rollout + n_validate``).
        producer_count: Number of workers sharing the budget.
        producer_index: This worker's index (for logging).
    """
    global _http_io_thread
    if _http_io_thread is not None:
        return _http_io_thread

    total_concurrency = server_concurrency * rollout_engine_num
    max_concurrency = (total_concurrency + producer_count - 1) // producer_count
    _http_io_thread = HttpIOThread(max_concurrency=max_concurrency)
    _http_io_thread.start()
    psrl_logger.info(
        "[Worker %d] HttpIOThread: max_concurrency=%d (server=%d * engines=%d / producers=%d).",
        producer_index,
        max_concurrency,
        server_concurrency,
        rollout_engine_num,
        producer_count,
    )
    return _http_io_thread
