"""Shared primitives for session hang/continue scheduling.

The session router duplicates these constants because it runs in a separate
Uvicorn process.
"""

from __future__ import annotations

import json

import aiohttp

from psrl.workers.gen.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT


class SessionSchedulingBase:
    """
    Provide generic session-router HTTP helpers.

    The host class must initialize `session_router_url` and `_session_client`.
    """

    async def _ensure_session_client(self) -> aiohttp.ClientSession:
        if self._session_client is None or self._session_client.closed:
            connector = aiohttp.TCPConnector(
                limit=DEFAULT_MAX_CONNECTIONS,
                limit_per_host=DEFAULT_MAX_CONNECTIONS,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
            self._session_client = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session_client

    async def _session_get_json(self, path: str):
        client = await self._ensure_session_client()
        url = f"{self.session_router_url}{path}"
        async with client.get(url) as resp:
            resp.raise_for_status()
            text = await resp.text()
            return json.loads(text) if text.strip() else {}

    async def _session_post_json(self, path: str, payload):
        client = await self._ensure_session_client()
        url = f"{self.session_router_url}{path}"
        async with client.post(url, json=payload) as resp:
            resp.raise_for_status()
            text = await resp.text()
            return json.loads(text) if text.strip() else {}
