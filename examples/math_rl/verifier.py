"""Client for the external math verifier.

One call grades a proof against a problem. The service answers a blocking POST
with a verdict and returns its receipt in response headers. Transport and
service failures are infrastructure failures, never a zero reward.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import aiohttp

psrl_logger = logging.getLogger(__name__)

VERDICTS = ("7", "6", "1", "0")
DEFAULT_GRADED_MAPPING = {"7": 1.0, "6": 0.8, "1": 0.3, "0": 0.0}
DEFAULT_BINARY_MAPPING = {"7": 1.0, "6": 1.0, "1": 0.0, "0": 0.0}
MAX_REASON_CHARS = 16_000
# The service rejects a request body above 16 MiB, so the client keeps a
# margin for the problem, the model block and the JSON envelope.
MAX_REQUEST_BYTES = 16 * 1024 * 1024 - 8192
DEFAULT_SOLUTION_BYTES = 8 * 1024 * 1024


class VerifierError(RuntimeError):
    """A verification that produced no verdict.

    The episode is ungraded. ``kind`` names the class for metrics and for the
    decision to retry.
    """

    def __init__(self, message: str, *, kind: str = "error", retryable: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass(frozen=True)
class VerifierConfig:
    """Where the verifier lives and how it is asked."""

    endpoint: str = ""
    verifier_harness_version: str = ""
    model: str = ""
    profile: str = ""
    reasoning_effort: str = ""
    proxy: str = ""
    transport: str = "stream"
    stage_timeout_s: float = 600.0
    max_retries: int = 2
    max_concurrency: int = 64
    max_solution_bytes: int = DEFAULT_SOLUTION_BYTES
    max_request_bytes: int = MAX_REQUEST_BYTES
    cache_size: int = 256
    reward_mode: str = "graded"
    graded_mapping: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_GRADED_MAPPING))

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("The verifier endpoint is required.")
        if not self.verifier_harness_version:
            raise ValueError("The verifier harness version is required.")
        if not self.model and not self.profile:
            raise ValueError("The verifier needs a model or a profile.")
        if self.transport not in ("stream", "blocking"):
            raise ValueError(f"Verifier transport must be stream or blocking, got {self.transport!r}.")
        if self.stage_timeout_s <= 0:
            raise ValueError("The verifier stage timeout must be greater than zero.")
        if self.max_retries < 1:
            raise ValueError("The verifier max_retries must be at least one.")
        if self.max_concurrency < 1:
            raise ValueError("The verifier max_concurrency must be at least one.")
        if self.max_solution_bytes < 1 or self.max_request_bytes < 1:
            raise ValueError("The verifier byte limits must be greater than zero.")
        if self.max_solution_bytes >= self.max_request_bytes:
            raise ValueError("The solution limit must be below the request body limit.")
        if self.reward_mode not in ("graded", "binary"):
            raise ValueError(f"Reward mode must be graded or binary, got {self.reward_mode!r}.")
        mapping = dict(DEFAULT_BINARY_MAPPING if self.reward_mode == "binary" else self.graded_mapping)
        missing = [verdict for verdict in VERDICTS if verdict not in mapping]
        if missing:
            raise ValueError(f"Reward mapping is missing verdicts {missing}.")
        if any(not 0.0 <= float(score) <= 1.0 for score in mapping.values()):
            raise ValueError("Reward mapping scores must be between zero and one.")
        object.__setattr__(self, "graded_mapping", {key: float(value) for key, value in mapping.items()})

    @property
    def client_timeout_s(self) -> float:
        """The service runs contract, audits and grading in sequence."""
        return 3 * self.stage_timeout_s + 180.0

    @property
    def stream_read_timeout_s(self) -> float:
        """The longest silence between events is one stage.

        The stream emits only on stage transitions, so a gap longer than one
        stage limit plus margin means the stage is wedged rather than slow.
        """
        return self.stage_timeout_s + 120.0

    def model_config(self) -> dict[str, Any]:
        """Render the model block the service accepts.

        The service owns model selection. A caller cannot widen it beyond the
        configured profile or model.
        """
        config: dict[str, Any] = {}
        if self.model:
            config["model"] = self.model
        elif self.profile:
            config["profile"] = self.profile
        if self.reasoning_effort:
            config["reasoning_effort"] = self.reasoning_effort
        config["timeout_seconds"] = self.stage_timeout_s
        return config

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> VerifierConfig:
        """Build a config from the resolved Hydra mapping."""
        return cls(
            endpoint=str(mapping.get("endpoint") or "").strip(),
            verifier_harness_version=str(mapping.get("verifier_harness_version") or "").strip(),
            model=str(mapping.get("model") or "").strip(),
            profile=str(mapping.get("profile") or "").strip(),
            reasoning_effort=str(mapping.get("reasoning_effort") or "").strip(),
            proxy=str(mapping.get("proxy") or "").strip(),
            transport=str(mapping.get("transport") or "stream").strip().lower(),
            stage_timeout_s=float(mapping.get("stage_timeout_s", 600.0)),
            max_retries=int(mapping.get("max_retries", 2)),
            max_concurrency=int(mapping.get("max_concurrency", 64)),
            max_solution_bytes=int(mapping.get("max_solution_bytes", DEFAULT_SOLUTION_BYTES)),
            max_request_bytes=int(mapping.get("max_request_bytes", MAX_REQUEST_BYTES)),
            cache_size=int(mapping.get("cache_size", 256)),
            reward_mode=str(mapping.get("reward_mode", "graded")).strip().lower() or "graded",
            graded_mapping=dict(mapping.get("graded_mapping") or DEFAULT_GRADED_MAPPING),
        )

    @classmethod
    def from_env(cls) -> VerifierConfig:
        """Build a config from the environment, for scripts and tests."""
        return cls(
            endpoint=os.environ.get("MATH_RL_VERIFIER_URL", "").strip(),
            verifier_harness_version=os.environ.get("MATH_RL_VERIFIER_VERSION", "").strip(),
            model=os.environ.get("MATH_RL_VERIFIER_MODEL", "").strip(),
            profile=os.environ.get("MATH_RL_VERIFIER_PROFILE", "").strip(),
            reasoning_effort=os.environ.get("MATH_RL_VERIFIER_EFFORT", "").strip(),
            proxy=os.environ.get("MATH_RL_VERIFIER_PROXY", "").strip(),
            transport=os.environ.get("MATH_RL_VERIFIER_TRANSPORT", "stream").strip().lower(),
            stage_timeout_s=float(os.environ.get("MATH_RL_VERIFIER_STAGE_TIMEOUT_S", "600")),
            max_retries=int(os.environ.get("MATH_RL_VERIFIER_MAX_RETRIES", "2")),
            max_concurrency=int(os.environ.get("MATH_RL_VERIFIER_MAX_CONCURRENCY", "64")),
            max_solution_bytes=int(
                os.environ.get("MATH_RL_VERIFIER_MAX_SOLUTION_BYTES", str(DEFAULT_SOLUTION_BYTES))
            ),
            max_request_bytes=int(os.environ.get("MATH_RL_VERIFIER_MAX_REQUEST_BYTES", str(MAX_REQUEST_BYTES))),
            cache_size=int(os.environ.get("MATH_RL_VERIFIER_CACHE_SIZE", "256")),
            reward_mode=os.environ.get("MATH_RL_REWARD_MODE", "graded"),
        )


@dataclass(frozen=True)
class Verdict:
    """One graded verdict and the receipt that produced it."""

    verdict: str
    reason: str = ""
    run_id: str = ""
    harness_sha256: str = ""
    prompt_sha256: str = ""
    skill_sha256: str = ""
    duration_s: float | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    cached: bool = False
    proof_sha256: str = ""

    def score(self, config: VerifierConfig) -> float:
        """Map the verdict to the configured reward."""
        return float(config.graded_mapping[self.verdict])


class _VerdictCache:
    """A bounded, process-local cache.

    The cache freezes a verdict for one exact proof. It is a cost control, not
    an authority, and it is never written for an infrastructure failure.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._entries: OrderedDict[str, Verdict] = OrderedDict()

    def get(self, key: str) -> Verdict | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def put(self, key: str, verdict: Verdict) -> None:
        self._entries[key] = verdict
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_key(config: VerifierConfig, problem: str, solution: str) -> str:
    model = json.dumps(config.model_config(), sort_keys=True, separators=(",", ":"))
    return "|".join((_sha256(problem), _sha256(solution), model, config.verifier_harness_version))


def _parse_usage(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw[:512]}
    return parsed if isinstance(parsed, dict) else {"raw": raw[:512]}


def _receipt(headers: Mapping[str, str]) -> dict[str, Any]:
    duration = headers.get("x-verifier-duration-seconds")
    try:
        duration_s = float(duration) if duration is not None else None
    except ValueError:
        duration_s = None
    return {
        "run_id": headers.get("x-verifier-receipt", ""),
        "harness_sha256": headers.get("x-verifier-harness-sha256", ""),
        "prompt_sha256": headers.get("x-verifier-prompt-sha256", ""),
        "skill_sha256": headers.get("x-verifier-skill-sha256", ""),
        "duration_s": duration_s,
        "usage": _parse_usage(headers.get("x-verifier-usage")),
    }


class VerifierClient:
    """One aiohttp session with a bounded concurrency window."""

    def __init__(self, config: VerifierConfig) -> None:
        self.config = config
        self._session: aiohttp.ClientSession | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._cache = _VerdictCache(config.cache_size)

    def _checkout(self) -> tuple[aiohttp.ClientSession, asyncio.Semaphore]:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.client_timeout_s))
            self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        return self._session, self._semaphore

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
        self._semaphore = None
        self._cache.clear()

    async def verify(self, problem: str, solution: str, timeout_s: float | None = None) -> Verdict:
        """Grade one proof, or raise VerifierError.

        Args:
            problem: The dataset problem, byte identical to the staged file.
            solution: The final proof.
            timeout_s: Per-call wall clock. The configured client timeout when None.

        Returns:
            Verdict: The verdict and its receipt.

        Raises:
            VerifierError: When no verdict was produced.
        """
        if not problem.strip():
            raise VerifierError("The problem statement is empty.", kind="payload")
        encoded = solution.encode("utf-8")
        if len(encoded) > self.config.max_solution_bytes:
            raise VerifierError(
                f"The solution is {len(encoded)} bytes, above the {self.config.max_solution_bytes} byte limit.",
                kind="payload",
            )
        key = _cache_key(self.config, problem, solution)
        cached = self._cache.get(key)
        if cached is not None:
            psrl_logger.info(f"Verifier cache hit for proof {cached.proof_sha256[:12]} verdict {cached.verdict}.")
            return replace(cached, cached=True)

        payload = {
            "problem": problem,
            "solution_to_verify": solution,
            "model_config": self.config.model_config(),
            "verifier_harness_version": self.config.verifier_harness_version,
        }
        body_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if body_bytes > self.config.max_request_bytes:
            raise VerifierError(
                f"The verifier request body is {body_bytes} bytes, above the "
                f"{self.config.max_request_bytes} byte limit.",
                kind="payload",
            )
        verdict = await self._post(payload, timeout_s)
        self._cache.put(key, verdict)
        return verdict

    async def _post(self, payload: Mapping[str, Any], timeout_s: float | None) -> Verdict:
        """Send one request, retrying only failures that produced no verdict."""
        last_error: VerifierError | None = None
        for attempt in range(self.config.max_retries):
            try:
                if self.config.transport == "stream":
                    return await self._post_stream(payload, timeout_s)
                return await self._post_blocking(payload, timeout_s)
            except VerifierError as error:
                last_error = error
                if not error.retryable or attempt + 1 >= self.config.max_retries:
                    break
                delay = 2**attempt
                psrl_logger.warning(
                    f"Verifier call failed on attempt {attempt + 1}, retrying in {delay}s: {error!r}."
                )
                await asyncio.sleep(delay)
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                last_error = VerifierError(f"Verifier transport failure: {error!r}.", kind="transport", retryable=True)
                if attempt + 1 >= self.config.max_retries:
                    break
                delay = 2**attempt
                psrl_logger.warning(
                    f"Verifier transport failed on attempt {attempt + 1}, retrying in {delay}s: {error!r}."
                )
                await asyncio.sleep(delay)
        raise last_error or VerifierError("The verifier produced no result.", kind="transport")

    async def _post_blocking(self, payload: Mapping[str, Any], timeout_s: float | None) -> Verdict:
        """Post once and read the receipt from the response headers."""
        session, semaphore = self._checkout()
        effective_timeout = timeout_s or self.config.client_timeout_s
        started = time.perf_counter()
        async with semaphore:
            async with session.post(
                self.config.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=effective_timeout),
                proxy=self.config.proxy or None,
            ) as response:
                body = await response.text()
                _raise_for_status(response.status, body)
                verdict = _parse_verdict(body)
                receipt = _receipt(response.headers)
        result = Verdict(
            verdict=verdict["verdict"],
            reason=str(verdict.get("reason") or "")[:MAX_REASON_CHARS],
            proof_sha256=_sha256(str(payload["solution_to_verify"])),
            **receipt,
        )
        psrl_logger.info(
            f"Verifier verdict {result.verdict} in {time.perf_counter() - started:.2f}s "
            f"run_id {result.run_id or 'unknown'}."
        )
        return result

    async def _post_stream(self, payload: Mapping[str, Any], timeout_s: float | None) -> Verdict:
        """Follow the stage events and take the verdict from the terminal event."""
        session, semaphore = self._checkout()
        effective_timeout = timeout_s or self.config.client_timeout_s
        url = _stream_url(self.config.endpoint)
        started = time.perf_counter()
        stage = "connect"
        saw_event = False
        async with semaphore:
            timeout = aiohttp.ClientTimeout(total=effective_timeout, sock_read=self.config.stream_read_timeout_s)
            try:
                async with session.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                    timeout=timeout,
                    proxy=self.config.proxy or None,
                ) as response:
                    if response.status >= 400:
                        _raise_for_status(response.status, await response.text())
                    run_id_header = response.headers.get("x-verifier-receipt", "")
                    async for event, data in _iter_sse(response.content):
                        saw_event = True
                        if event == "progress":
                            next_stage = str(data.get("stage") or "")
                            if next_stage and next_stage != stage:
                                stage = next_stage
                                psrl_logger.info(
                                    f"Verifier stage {stage} at {time.perf_counter() - started:.1f}s "
                                    f"run_id {data.get('run_id') or run_id_header or 'unknown'}."
                                )
                        elif event == "result":
                            body = data.get("result")
                            if not isinstance(body, dict):
                                raise VerifierError("The verifier stream result has no body.", kind="malformed")
                            verdict = _validate_verdict(body)
                            receipt = _receipt_from_event(data.get("receipt"), run_id_header)
                            result = Verdict(
                                verdict=verdict,
                                reason=str(body.get("reason") or "")[:MAX_REASON_CHARS],
                                proof_sha256=_sha256(str(payload["solution_to_verify"])),
                                **receipt,
                            )
                            psrl_logger.info(
                                f"Verifier verdict {result.verdict} in {time.perf_counter() - started:.2f}s "
                                f"run_id {result.run_id or 'unknown'}."
                            )
                            return result
                        elif event == "error":
                            failed_stage = str(data.get("failed_stage") or stage)
                            raise VerifierError(
                                f"The verifier failed during stage {failed_stage}.",
                                kind="upstream",
                                retryable=False,
                            )
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                raise VerifierError(
                    f"Verifier stream failure during stage {stage}: {error!r}.",
                    kind="transport",
                    retryable=not saw_event,
                ) from error
        raise VerifierError(
            f"The verifier stream ended without a verdict during stage {stage}.",
            kind="malformed",
            retryable=not saw_event,
        )


def _raise_for_status(status: int, body: str) -> None:
    if status >= 500:
        raise VerifierError(
            f"The verifier returned HTTP {status}: {body[:300]!r}.",
            kind="upstream",
            retryable=True,
        )
    if status >= 400:
        raise VerifierError(
            f"The verifier rejected the request with HTTP {status}: {body[:300]!r}.",
            kind="rejected",
        )


def _validate_verdict(data: Mapping[str, Any]) -> str:
    verdict = str(data.get("verdict") or "").strip()
    if verdict not in VERDICTS:
        raise VerifierError(f"The verifier returned an unknown verdict {verdict!r}.", kind="malformed")
    return verdict


def _parse_verdict(body: str) -> dict[str, Any]:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as error:
        raise VerifierError(f"The verifier response is not JSON: {error!r}.", kind="malformed") from error
    if not isinstance(data, dict):
        raise VerifierError("The verifier response is not an object.", kind="malformed")
    _validate_verdict(data)
    return data


def _stream_url(endpoint: str) -> str:
    """Derive the streaming endpoint from the configured verify URL."""
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/verify/stream"):
        return trimmed
    if trimmed.endswith("/verify"):
        return trimmed + "/stream"
    return trimmed


def _receipt_from_event(receipt: Any, fallback_run_id: str = "") -> dict[str, Any]:
    entry = receipt if isinstance(receipt, Mapping) else {}
    duration = entry.get("duration_seconds")
    try:
        duration_s = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_s = None
    usage = entry.get("usage")
    return {
        "run_id": str(entry.get("run_id") or fallback_run_id or ""),
        "harness_sha256": str(entry.get("harness_sha256") or ""),
        "prompt_sha256": str(entry.get("prompt_sha256") or ""),
        "skill_sha256": str(entry.get("skill_sha256") or ""),
        "duration_s": duration_s,
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
    }


def _sse_payload(lines: list[str], event: str) -> dict[str, Any]:
    try:
        data = json.loads("\n".join(lines)) if lines else {}
    except json.JSONDecodeError as error:
        raise VerifierError(f"The verifier {event} event is not JSON: {error!r}.", kind="malformed") from error
    if not isinstance(data, dict):
        raise VerifierError(f"The verifier {event} event is not an object.", kind="malformed")
    return data


async def _iter_sse(content: Any) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Yield one parsed record per blank-line terminated Server-Sent Event."""
    event = ""
    data_lines: list[str] = []
    async for raw_line in content:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if event:
                yield event, _sse_payload(data_lines, event)
            event, data_lines = "", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if event:
        yield event, _sse_payload(data_lines, event)


_CLIENTS: dict[tuple[str, str, str, str, str], VerifierClient] = {}


def get_verifier_client(config: VerifierConfig) -> VerifierClient:
    """Return a cached client for one endpoint and model configuration."""
    key = (
        config.endpoint,
        config.verifier_harness_version,
        json.dumps(config.model_config(), sort_keys=True, separators=(",", ":")),
        config.reward_mode,
        config.transport,
    )
    client = _CLIENTS.get(key)
    if client is None:
        client = VerifierClient(config)
        _CLIENTS[key] = client
    return client


async def evaluate(config: VerifierConfig, problem: str, solution: str, timeout_s: float | None = None) -> Verdict:
    """Grade one proof through the cached client."""
    return await get_verifier_client(config).verify(problem, solution, timeout_s=timeout_s)


async def reset_verifier_clients() -> None:
    """Close every cached client. Used by tests and shutdown paths."""
    for client in list(_CLIENTS.values()):
        await client.close()
    _CLIENTS.clear()
