"""Per-worker HTTP gateway between sandboxed harnesses and the external verifier.

The Rethlas generation agent calls its verification tool from inside the sandbox, so the
tool needs a URL it can reach. This gateway is that URL. It owns the dataset problem, the
per-episode call budget, the single-flight map and the verifier client, and it returns the
Rethlas verification schema the agent's repair loop already understands.

The episode finalizer runs in the same process and shares the client's proof cache, so a
final proof that the agent already had verified costs nothing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web
from examples.math_rl.verifier import Verdict, VerifierConfig, VerifierError, evaluate

psrl_logger = logging.getLogger(__name__)

_PASSING_VERDICTS = frozenset({"7", "6"})


@dataclass
class EpisodeState:
    """One episode's problem, budget counters and in-flight verifications."""

    episode_id: str
    problem: str
    calls: int = 0
    tokens: int = 0
    inflight: dict[str, asyncio.Future] = field(default_factory=dict)


def _usage_tokens(usage: Mapping[str, Any]) -> int:
    total = 0
    for key in ("input_tokens", "output_tokens", "reasoning_output_tokens", "cached_input_tokens"):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            total += int(value)
    return total


def rethlas_response(config: VerifierConfig, statement: str, verdict: Verdict) -> dict[str, Any]:
    """Render one external verdict as the Rethlas verification payload."""
    correct = verdict.verdict in _PASSING_VERDICTS
    reason = verdict.reason or ""
    report = {
        "summary": reason or ("The proof was verified." if correct else "The proof was rejected."),
        "critical_errors": [],
        "gaps": [],
        "notes": reason,
    }
    return {
        "statement": statement,
        "verification_report": report,
        "verdict": "correct" if correct else "wrong",
        "repair_hints": "" if correct else reason,
        "endpoint": config.endpoint,
        "verdict_code": verdict.verdict,
        "verdict_score": verdict.score(config),
        "run_id": verdict.run_id,
        "cached": verdict.cached,
    }


class VerifierGateway:
    """One aiohttp service per agent-loop worker process."""

    def __init__(
        self,
        config: VerifierConfig,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        advertised_host: str | None = None,
        max_calls_per_episode: int = 4,
        max_tokens_per_episode: int = 0,
    ) -> None:
        if max_calls_per_episode < 1:
            raise ValueError("max_calls_per_episode must be at least one.")
        if max_tokens_per_episode < 0:
            raise ValueError("max_tokens_per_episode cannot be negative.")
        self.config = config
        self.host = host
        self.port = port
        self.advertised_host = advertised_host
        self.max_calls_per_episode = max_calls_per_episode
        self.max_tokens_per_episode = max_tokens_per_episode
        self._episodes: dict[str, EpisodeState] = {}
        self._runner: web.AppRunner | None = None
        self._base_url = ""
        self._port = 0
        self.total_calls = 0
        self.total_tokens = 0
        self.total_budget_errors = 0
        self.total_errors = 0
        self._verdict_counts: dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------

    def _advertised_host(self) -> str:
        """The address a sandbox on another node can reach."""
        if self.advertised_host:
            return self.advertised_host
        try:
            import ray

            return str(ray.util.get_node_ip_address())
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    @property
    def base_url(self) -> str:
        return self._base_url

    def episode_url(self, episode_id: str) -> str:
        return f"{self._base_url}/verify/{episode_id}"

    @property
    def local_url(self) -> str:
        """The loopback URL, for tests and same-host callers."""
        return f"http://127.0.0.1:{self._port}"

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/verify/{episode}", self._handle_verify)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/metrics", self._handle_metrics)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        sockets = site._server.sockets
        bound = sockets[0].getsockname()[1]
        self._runner = runner
        self._port = int(bound)
        self._base_url = f"http://{self._advertised_host()}:{bound}"
        psrl_logger.info("Verifier gateway listening at %s.", self._base_url)
        return self._base_url

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._episodes.clear()

    # -- episodes ----------------------------------------------------------

    def register_episode(self, episode_id: str, problem: str) -> str:
        if not problem.strip():
            raise ValueError("A registered episode needs a non-empty problem.")
        self._episodes[episode_id] = EpisodeState(episode_id=episode_id, problem=problem)
        return self.episode_url(episode_id)

    def release_episode(self, episode_id: str) -> None:
        self._episodes.pop(episode_id, None)

    # -- routes ------------------------------------------------------------

    async def _handle_verify(self, request: web.Request) -> web.Response:
        episode_id = request.match_info["episode"]
        state = self._episodes.get(episode_id)
        if state is None:
            return web.json_response({"error": "unknown_episode"}, status=404)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid_json"}, status=422)
        if not isinstance(body, dict):
            return web.json_response({"error": "invalid_body"}, status=422)
        proof = str(body.get("proof") or "")
        statement = str(body.get("statement") or "")
        if not proof.strip():
            return web.json_response({"error": "empty_proof"}, status=422)

        if state.calls >= self.max_calls_per_episode:
            self.total_budget_errors += 1
            return web.json_response(
                {
                    "error": "budget_exceeded",
                    "detail": (
                        f"episode {episode_id} used {state.calls} of "
                        f"{self.max_calls_per_episode} verification calls"
                    ),
                },
                status=429,
            )
        if self.max_tokens_per_episode and state.tokens >= self.max_tokens_per_episode:
            self.total_budget_errors += 1
            return web.json_response(
                {"error": "budget_exceeded", "detail": f"episode {episode_id} exhausted its token budget"},
                status=429,
            )

        state.calls += 1
        self.total_calls += 1
        key = proof
        future = state.inflight.get(key)
        if future is None:
            future = asyncio.ensure_future(self._evaluate(state, proof))
            state.inflight[key] = future

            def _clear(_: asyncio.Future, episode: EpisodeState = state, inflight_key: str = key) -> None:
                episode.inflight.pop(inflight_key, None)

            future.add_done_callback(_clear)

        try:
            verdict = await future
        except VerifierError as error:
            self.total_errors += 1
            psrl_logger.warning("Verifier gateway failed for episode %s: %r.", episode_id, error)
            return web.json_response(
                {"error": error.kind, "detail": str(error), "retryable": error.retryable},
                status=503,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            self.total_errors += 1
            psrl_logger.exception("Verifier gateway crashed for episode %s.", episode_id)
            return web.json_response({"error": "internal", "detail": str(error)}, status=500)

        tokens = _usage_tokens(verdict.usage)
        state.tokens += tokens
        self.total_tokens += tokens
        self._verdict_counts[verdict.verdict] = self._verdict_counts.get(verdict.verdict, 0) + 1
        return web.json_response(rethlas_response(self.config, statement, verdict))

    async def _evaluate(self, state: EpisodeState, proof: str) -> Verdict:
        return await evaluate(self.config, state.problem, proof)

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "status": "ok",
                "endpoint": self.config.endpoint,
                "model": self.config.model or self.config.profile,
                "verifier_harness_version": self.config.verifier_harness_version,
                "episodes": len(self._episodes),
                "total_calls": self.total_calls,
                "total_tokens": self.total_tokens,
            }
        )

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        episodes = {
            episode_id: {
                "calls": state.calls,
                "tokens": state.tokens,
                "inflight": len(state.inflight),
            }
            for episode_id, state in self._episodes.items()
        }
        return web.json_response(
            {
                "total_calls": self.total_calls,
                "total_tokens": self.total_tokens,
                "total_errors": self.total_errors,
                "budget_errors": self.total_budget_errors,
                "verdict_counts": dict(self._verdict_counts),
                "episodes": episodes,
            }
        )


_GATEWAYS: dict[tuple[str, str, str, str, str], VerifierGateway] = {}


async def get_verifier_gateway(
    config: VerifierConfig,
    *,
    max_calls_per_episode: int = 4,
    max_tokens_per_episode: int = 0,
) -> VerifierGateway:
    """Return the process-wide gateway for one external verifier configuration."""
    key = (
        config.endpoint,
        config.verifier_harness_version,
        json.dumps(config.model_config(), sort_keys=True, separators=(",", ":")),
        config.reward_mode,
        config.proxy,
    )
    gateway = _GATEWAYS.get(key)
    if gateway is None:
        gateway = VerifierGateway(
            config,
            max_calls_per_episode=max_calls_per_episode,
            max_tokens_per_episode=max_tokens_per_episode,
        )
        await gateway.start()
        _GATEWAYS[key] = gateway
    return gateway


async def reset_verifier_gateways() -> None:
    """Stop every process-wide gateway. Used by tests and shutdown paths."""
    for gateway in list(_GATEWAYS.values()):
        await gateway.stop()
    _GATEWAYS.clear()
