"""Shared primitives for session hang/continue scheduling.

The session router duplicates these constants because it runs in a separate
Uvicorn process.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import aiohttp

from psrl.workers.gen.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT, RolloutInstanceId

# Session-status values (mirror SessionState.status in session_router.py).
STATUS_GENERATE = "generate"
STATUS_ENV = "env"

# Hang-state values (mirror SessionState.hang_state in session_router.py).
SESSION_RUNNING = "running"
SESSION_HUNG = "hung"


@dataclass
class InstanceCapacity:
    """KV-cache capacity snapshot for one (replica_id, dp_rank) vLLM instance."""

    instance_id: RolloutInstanceId
    total_kv_tokens: int
    used_tokens: int


@dataclass
class SessionInfo:
    """One live TITO session as seen by the scheduler."""

    session_id: str
    instance_id: RolloutInstanceId | None
    status: str  # STATUS_GENERATE | STATUS_ENV
    hang_state: str  # SESSION_RUNNING | SESSION_HUNG
    total_tokens: int


class SessionScheduler(ABC):
    """Abstract base for session hang/continue decision logic."""

    @abstractmethod
    def decide(
        self,
        instances: list[InstanceCapacity],
        sessions: list[SessionInfo],
    ) -> tuple[list[str], list[tuple[str, RolloutInstanceId]]]:
        """Return ``(session_ids_to_hang, [(session_id, continue_instance), ...])``.

        The continue list pairs each readmitted session with its target instance.
        `None` instances are not permitted here because a strategy that defers
        routing to SMG should simply not emit the pair.
        """


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
