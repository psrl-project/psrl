# Adapted from slime/slime/utils/http_utils.py
import asyncio
import json
import logging
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp
import ray

psrl_logger = logging.getLogger(__name__)


def find_available_port(base_port: int):
    """Find an available port starting from base_port."""
    port = base_port
    while not is_port_available(port):
        port += 1
    return port


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_host_info():
    """Get the hostname and local IP address of the current machine."""
    hostname = socket.gethostname()

    # try DNS
    try:
        return hostname, socket.gethostbyname(hostname)
    except socket.gaierror:
        pass

    # try IPv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            udp_sock.connect(("8.8.8.8", 80))  # Google DNS
            return hostname, udp_sock.getsockname()[0]
    except OSError:
        pass

    # try IPv6
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as s6:
            s6.connect(("2001:4860:4860::8888", 80))
            return hostname, s6.getsockname()[0]
    except OSError:
        pass

    # hostname -I
    try:
        local_ip = os.popen("hostname -I | awk '{print $1}'").read().strip()
        return hostname, local_ip or "::1"
    except Exception:
        return hostname, "::1"


DEFAULT_HTTP_CONCURRENCY = 256


@dataclass(frozen=True, slots=True)
class HttpRuntimeConfig:
    """Runtime HTTP client configuration.

    ``total_concurrency`` is the shared budget for all producers. Each producer
    receives a fair share via ``producer_count`` so that multiple Ray workers do
    not multiply the intended gateway concurrency.
    """

    total_concurrency: int = DEFAULT_HTTP_CONCURRENCY
    producer_count: int = 1
    producer_index: int = 0
    dns_ttl_secs: int = 300
    total_timeout_secs: float | None = None

    @property
    def effective_concurrency(self) -> int:
        return max(1, (self.total_concurrency + self.producer_count - 1) // self.producer_count)


_http_client: aiohttp.ClientSession | None = None

# Maximum concurrency for the global HTTP client. Kept for compatibility with
# callers that use create_aiohttp_client(concurrency=None).
_client_concurrency: int = DEFAULT_HTTP_CONCURRENCY
_runtime_config = HttpRuntimeConfig()

# Optional Ray-based distributed POST dispatch. The actor pool is created by
# AgentLoopManager and installed into each AgentLoopWorker process.
_distributed_post_enabled: bool = False
_post_actors: list[Any] = []
_post_actor_idx: int = 0

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass(slots=True)
class HttpResponse:
    """Fully buffered HTTP response for proxy-style callers."""

    status: int
    body: bytes
    headers: dict[str, str]

    def json(self) -> Any:
        return json.loads(self.body)


@dataclass(slots=True)
class JsonHttpResponse:
    """Parsed HTTP response with headers preserved for routing metadata."""

    status: int
    data: Any
    headers: dict[str, str]
    text: str | None = None


def _next_post_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


def configure_distributed_post(
    actors: list[Any] | tuple[Any, ...] | None,
    *,
    enabled: bool,
    start_index: int = 0,
) -> None:
    """Install distributed POST actor handles in the current process."""
    global _distributed_post_enabled, _post_actors, _post_actor_idx

    _post_actors = list(actors or [])
    _distributed_post_enabled = bool(enabled and _post_actors)
    _post_actor_idx = max(0, int(start_index)) % max(1, len(_post_actors))
    psrl_logger.info(
        "Distributed POST configured: enabled=%s actors=%d start_index=%d.",
        _distributed_post_enabled,
        len(_post_actors),
        _post_actor_idx,
    )


def is_distributed_post_enabled() -> bool:
    return _distributed_post_enabled and bool(_post_actors)


def filter_http_headers(
    headers: Mapping[str, str],
    *,
    excluded: set[str] | None = None,
) -> dict[str, str]:
    """Remove hop-by-hop headers before forwarding requests or responses."""
    blocked = HOP_BY_HOP_HEADERS if excluded is None else HOP_BY_HOP_HEADERS | excluded
    return {key: value for key, value in headers.items() if key.lower() not in blocked}


def create_aiohttp_client(
    *,
    concurrency: int | None = None,
    timeout: aiohttp.ClientTimeout | None = None,
) -> aiohttp.ClientSession:
    """Create an aiohttp client configured for long-running rollout requests."""
    connector = aiohttp.TCPConnector(
        limit=_client_concurrency if concurrency is None else concurrency,
        ttl_dns_cache=_runtime_config.dns_ttl_secs,
        enable_cleanup_closed=True,
    )
    if timeout is None:
        total = _runtime_config.total_timeout_secs
        timeout = aiohttp.ClientTimeout(total=total)
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
    )


async def _ensure_http_client() -> aiohttp.ClientSession:
    global _http_client
    if _http_client is None or _http_client.closed:
        _http_client = create_aiohttp_client()
    return _http_client


async def close_http_client() -> None:
    """Close the global aiohttp client if it has been created."""
    global _http_client
    if _http_client is not None and not _http_client.closed:
        await _http_client.close()
    _http_client = None


def _extract_worker_headers(headers: Mapping[str, str]) -> dict[str, str | None]:
    return {
        "base_worker_id": headers.get("x-base-worker-id"),
        "target_dp_rank": headers.get("x-target-dp-rank"),
    }


def _with_header_info(data: Any, headers: Mapping[str, str]) -> Any:
    if isinstance(data, dict):
        data["header_info"] = _extract_worker_headers(headers)
    return data


async def _read_aiohttp_response(response) -> tuple[int, bytes, dict[str, str], Any]:
    body = await response.read()
    return response.status, body, filter_http_headers(response.headers), response


async def _read_generic_response(response) -> tuple[int, bytes, dict[str, str], Any]:
    """Read an aiohttp or httpx-style response.

    Tests inject httpx.AsyncClient transports into SessionRouter. Production
    code uses aiohttp, but supporting both here keeps the proxy helpers simple
    without adding a second HTTP utility path.
    """
    if hasattr(response, "read") and hasattr(response, "status"):
        return await _read_aiohttp_response(response)

    if hasattr(response, "aread"):
        body = await response.aread()
    else:
        body = getattr(response, "content", b"")
    status = getattr(response, "status", getattr(response, "status_code", 0))
    return int(status), body, filter_http_headers(response.headers), response


async def _request_once(
    session,
    method: str,
    url: str,
    *,
    content: bytes | None = None,
    json_payload: Any = None,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
) -> tuple[int, bytes, dict[str, str], Any]:
    kwargs: dict[str, Any] = {
        "headers": headers,
        "params": params,
    }
    if json_payload is not None:
        kwargs["json"] = json_payload
    else:
        kwargs["data"] = content

    request_ctx = session.request(method, url, **kwargs)
    if hasattr(request_ctx, "__aenter__"):
        async with request_ctx as response:
            return await _read_generic_response(response)

    response = await request_ctx
    try:
        return await _read_generic_response(response)
    finally:
        aclose = getattr(response, "aclose", None)
        if aclose is not None:
            await aclose()


def _raise_for_status(status: int, body: bytes, response: Any) -> None:
    if status < 400:
        return

    message = body.decode(errors="replace")
    request_info = getattr(response, "request_info", None)
    history = getattr(response, "history", ())
    headers = getattr(response, "headers", None)
    raise aiohttp.ClientResponseError(
        request_info=request_info,
        history=history,
        status=status,
        message=message,
        headers=headers,
    )


class RequestAbortedByGatewayError(Exception):
    """SMG returned the `request_aborted` 400 sentinel for the request.

    Callers should treat this as a deliberate termination and propagate it
    up to the agent loop boundary as `TerminateReason.ABORTED` rather than
    retry or log it as a transport error.
    """

    def __init__(self, request_id: str, message: str):
        super().__init__(f"Request {request_id!r} aborted by gateway: {message}")
        self.request_id = request_id
        self.message = message


_OVERLONG_MARKERS = (
    "longer than the maximum model length",
    "exceeds the model's maximum context length",
)

# Must match SMG `PROMPT_OVERFLOW_ERROR_CODE` / servicer trailing metadata.
_PROMPT_OVERFLOW_ERROR_CODE = "prompt_overflow"


class PromptOverflowError(Exception):
    """vLLM / SMG rejected a prompt that exceeds max_model_len.

    Raised in two paths:
    - SMG HTTP path: detected in `_classify_http_error()` via
      `x-smg-error-code: prompt_overflow` (preferred) or 400 body markers.
    - LiteLLM path: raised by overflow.py wrappers to abort the tenacity retry loop.
    Callers should treat this as a deliberate termination and map it to
    TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED.
    """


def _smg_error_code(response_headers: Mapping[str, str] | None) -> str | None:
    """Read `x-smg-error-code` case-insensitively from response headers."""
    if not response_headers:
        return None
    code = response_headers.get("x-smg-error-code")
    if code is not None:
        return code
    # Plain `dict` mocks are case-sensitive. Scan manually.
    if not hasattr(response_headers, "getall"):
        for key, value in response_headers.items():
            if key.lower() == "x-smg-error-code":
                return value
    return None


def _classify_http_error(
    exc: BaseException,
    request_headers: Mapping[str, str] | None = None,
) -> BaseException:
    """Return a more specific exception when `exc` is a classifiable SMG 400.

    Otherwise returns `exc` unchanged. Callers should `raise` the returned
    exception. Chained `__cause__` is preserved when a translation occurs.

    Only known sentinel codes (`request_aborted`, `prompt_overflow`) are
    translated. Other 400s stay as transport errors so real client mistakes
    are not silently swallowed.
    """
    if not isinstance(exc, aiohttp.ClientResponseError):
        return exc
    if exc.status != 400:
        return exc
    code = _smg_error_code(exc.headers)
    if code == "request_aborted":
        request_id = (request_headers or {}).get("x-request-id", "")
        translated = RequestAbortedByGatewayError(request_id=request_id, message=str(exc.message))
        translated.__cause__ = exc
        return translated
    body_text = str(exc.message)
    if code == _PROMPT_OVERFLOW_ERROR_CODE or any(marker in body_text for marker in _OVERLONG_MARKERS):
        translated = PromptOverflowError(f"Prompt exceeds max_model_len: {body_text}")
        translated.__cause__ = exc
        return translated
    return exc


def _parse_body(body: bytes) -> tuple[Any, str | None]:
    text: str | None = None
    try:
        return json.loads(body), None
    except (TypeError, json.JSONDecodeError, UnicodeDecodeError):
        text = body.decode(errors="replace")
        return text, text


async def request_raw(
    method: str,
    url: str,
    *,
    content: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    client: aiohttp.ClientSession | None = None,
    max_retries: int = 1,
    raise_for_status: bool = False,
) -> HttpResponse:
    """Send one HTTP request and return a fully buffered raw response."""
    retry_count = 0
    session = client or await _ensure_http_client()
    attempts = max(1, max_retries)
    while True:
        response = None
        try:
            status, body, response_headers, response = await _request_once(
                session,
                method,
                url,
                content=content,
                headers=headers,
                params=params,
            )
            if raise_for_status:
                _raise_for_status(status, body, response)
            return HttpResponse(status=status, body=body, headers=response_headers)
        except Exception as e:
            translated = _classify_http_error(e, headers)
            if isinstance(translated, RequestAbortedByGatewayError):
                raise translated from e
            if isinstance(translated, PromptOverflowError):
                raise translated from e
            retry_count += 1
            if retry_count >= attempts:
                raise
            psrl_logger.info(
                "Error: %s, retrying... (attempt %s/%s, url=%s)",
                e,
                retry_count,
                attempts,
                url,
            )
            await asyncio.sleep(1)


async def request_json(
    method: str,
    url: str,
    *,
    payload: Any = None,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    client: aiohttp.ClientSession | None = None,
    max_retries: int = 5,
    raise_for_status: bool = True,
) -> JsonHttpResponse:
    """Send one JSON request and return parsed data plus response headers."""
    retry_count = 0
    session = client or await _ensure_http_client()
    attempts = max(1, max_retries)
    while True:
        response = None
        try:
            status, body, response_headers, response = await _request_once(
                session,
                method,
                url,
                json_payload=payload,
                headers=headers,
                params=params,
            )
            if raise_for_status:
                _raise_for_status(status, body, response)
            data, text = _parse_body(body)
            return JsonHttpResponse(status=status, data=data, headers=response_headers, text=text)
        except Exception as e:
            translated = _classify_http_error(e, headers)
            if isinstance(translated, RequestAbortedByGatewayError):
                raise translated from e
            if isinstance(translated, PromptOverflowError):
                raise translated from e
            retry_count += 1
            psrl_logger.info(
                "Error: %s, retrying... (attempt %s/%s, url=%s)",
                e,
                retry_count,
                attempts,
                url,
            )
            if retry_count >= attempts:
                psrl_logger.info("Max retries (%s) reached, failing... (url=%s)", attempts, url)
                raise
            await asyncio.sleep(1)


async def raw_request(
    method: str,
    url: str,
    *,
    content: bytes | None = None,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    client: aiohttp.ClientSession | None = None,
    max_retries: int = 1,
) -> HttpResponse:
    """Send one HTTP request and return a fully buffered response.

    This helper is intentionally status-code agnostic. It is suitable for
    proxy paths where 4xx/5xx responses should be forwarded instead of raised.
    """
    return await request_raw(
        method,
        url,
        content=content,
        headers=headers,
        params=params,
        client=client,
        max_retries=max_retries,
        raise_for_status=False,
    )


async def _post(client, url, payload, max_retries=5, headers: dict[str, str] | None = None):
    """POST JSON payload with retries.

    Args:
        client: aiohttp.ClientSession instance.
        url: URL to POST to.
        payload: JSON-serializable payload to send.
        max_retries: Maximum number of retries on failure.
    """
    response = await request_json(
        "POST",
        url,
        payload=payload or {},
        headers=headers,
        client=client,
        max_retries=max_retries,
    )
    return _with_header_info(response.data, response.headers)


async def _get(client, url, params=None, max_retries=5, headers: dict[str, str] | None = None):
    """GET JSON payload with retries."""
    response = await request_json(
        "GET",
        url,
        params=params,
        headers=headers,
        client=client,
        max_retries=max_retries,
    )
    return _with_header_info(response.data, response.headers)


def init_http_client(
    server_concurrency: int,
    rollout_engine_num: int,
    *,
    producer_count: int = 1,
    producer_index: int = 0,
) -> None:
    """Configure the process-local HTTP client budget.

    ``server_concurrency * rollout_engine_num`` is the shared budget. Dividing
    by ``producer_count`` prevents each AgentLoopWorker from allocating a full
    copy of the gateway concurrency.
    """
    global _client_concurrency, _runtime_config

    total_concurrency = server_concurrency * rollout_engine_num
    _runtime_config = HttpRuntimeConfig(
        total_concurrency=total_concurrency,
        producer_count=max(1, int(producer_count)),
        producer_index=max(0, int(producer_index)),
    )
    _client_concurrency = _runtime_config.effective_concurrency


def init_distributed_post_pool(
    *,
    total_concurrency: int,
    post_actor_num_per_node: int,
) -> list[Any]:
    """Create node-affine Ray HTTP poster actors and split the total budget."""
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote
    class HttpPosterActor:
        def __init__(self, concurrency: int):
            self._concurrency = max(1, int(concurrency))
            self._client: aiohttp.ClientSession | None = None

        async def _ensure_client(self) -> aiohttp.ClientSession:
            if self._client is None or self._client.closed:
                self._client = create_aiohttp_client(concurrency=self._concurrency)
            return self._client

        async def request_json(
            self,
            method: str,
            url: str,
            payload: Any = None,
            headers: dict[str, str] | None = None,
            params: dict[str, str] | None = None,
            max_retries: int = 5,
            raise_for_status: bool = True,
        ) -> JsonHttpResponse:
            return await request_json(
                method,
                url,
                payload=payload,
                headers=headers,
                params=params,
                client=await self._ensure_client(),
                max_retries=max_retries,
                raise_for_status=raise_for_status,
            )

        async def aclose(self) -> None:
            if self._client is not None and not self._client.closed:
                await self._client.close()

    total_concurrency = max(1, int(total_concurrency))
    post_actor_num_per_node = max(1, int(post_actor_num_per_node))
    nodes = [node for node in ray.nodes() if node.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    actor_count = len(nodes) * post_actor_num_per_node
    per_actor_concurrency = (total_concurrency + actor_count - 1) // actor_count
    actors = []
    for node in nodes:
        scheduling = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        for _ in range(post_actor_num_per_node):
            actors.append(
                HttpPosterActor.options(
                    scheduling_strategy=scheduling,
                    max_concurrency=per_actor_concurrency,
                    num_cpus=0.001,
                ).remote(per_actor_concurrency)
            )
    psrl_logger.info(
        "Initialized distributed POST pool: actors=%d nodes=%d actors_per_node=%d "
        "total_concurrency=%d per_actor_concurrency=%d.",
        actor_count,
        len(nodes),
        post_actor_num_per_node,
        total_concurrency,
        per_actor_concurrency,
    )
    return actors


async def request_json_maybe_distributed(
    method: str,
    url: str,
    *,
    payload: Any = None,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    client: aiohttp.ClientSession | None = None,
    max_retries: int = 1,
    raise_for_status: bool = True,
) -> JsonHttpResponse:
    """Send JSON requests, routing POSTs through the distributed actor pool when enabled."""
    if client is None and method.upper() == "POST" and is_distributed_post_enabled():
        actor = _next_post_actor()
        if actor is not None:
            try:
                return await actor.request_json.remote(
                    method,
                    url,
                    payload,
                    dict(headers) if headers is not None else None,
                    dict(params) if params is not None else None,
                    max_retries,
                    raise_for_status,
                )
            except Exception as exc:
                psrl_logger.info(
                    "Distributed POST failed, falling back to local request_json: %s (url=%s).",
                    exc,
                    url,
                )

    return await request_json(
        method,
        url,
        payload=payload,
        headers=headers,
        params=params,
        client=client,
        max_retries=max_retries,
        raise_for_status=raise_for_status,
    )


async def post(
    url,
    payload,
    max_retries=5,
    headers: dict[str, str] | None = None,
    *,
    return_response: bool = False,
):
    """POST JSON payload using the global HTTP client."""
    response = await request_json_maybe_distributed(
        "POST",
        url,
        payload=payload or {},
        headers=headers,
        max_retries=max_retries,
    )
    if return_response:
        return response
    return _with_header_info(response.data, response.headers)


async def get(
    url,
    params: dict[str, Any] | None = None,
    max_retries=5,
    headers: dict[str, str] | None = None,
    *,
    return_response: bool = False,
):
    """GET JSON payload using the global HTTP client."""
    response = await request_json(
        "GET",
        url,
        params=params,
        headers=headers,
        max_retries=max_retries,
    )
    if return_response:
        return response
    return _with_header_info(response.data, response.headers)


async def delete(url, max_retries=1, headers: dict[str, str] | None = None) -> HttpResponse:
    """DELETE a URL and return the fully buffered response."""
    return await raw_request("DELETE", url, headers=headers, max_retries=max_retries)


# --- HTTP Concurrency Profiler ---


@dataclass
class HttpConcurrencyProfiler:
    """Track in-flight HTTP request concurrency over time.

    Records snapshots of concurrent request count to a JSONL file for
    offline analysis. Each record contains a timestamp, the number of
    concurrent requests at that moment, and event metadata.
    """

    _inflight: int = field(default=0, init=False, repr=False)
    _log_file: Any = field(default=None, init=False, repr=False)
    _log_path: str | None = field(default=None, init=False, repr=False)
    _enabled: bool = field(default=False, init=False, repr=False)
    _peak_inflight: int = field(default=0, init=False, repr=False)
    _total_requests: int = field(default=0, init=False, repr=False)

    def enable(self, log_path: str) -> None:
        """Enable profiling and set the output JSONL file path."""
        self._log_path = log_path
        self._enabled = True
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._log_file = open(log_path, "a")
        psrl_logger.info("[HTTP_PROFILER] Enabled, writing to %s", log_path)

    def _write_record(self, event: str, **extra) -> None:
        if not self._enabled or self._log_file is None:
            return
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) + f".{int(time.time() * 1000) % 1000:03d}",
            "epoch_ms": int(time.time() * 1000),
            "event": event,
            "inflight": self._inflight,
            "peak_inflight": self._peak_inflight,
            "total_requests": self._total_requests,
            **extra,
        }
        self._log_file.write(json.dumps(record) + "\n")
        self._log_file.flush()

    def on_request_start(self, url: str = "", request_id: str = "") -> None:
        """Call when an HTTP request begins."""
        self._inflight += 1
        self._total_requests += 1
        if self._inflight > self._peak_inflight:
            self._peak_inflight = self._inflight
        self._write_record("start", url=url, request_id=request_id)

    def on_request_end(self, url: str = "", request_id: str = "", elapsed_ms: float = 0.0) -> None:
        """Call when an HTTP request completes."""
        self._inflight -= 1
        self._write_record("end", url=url, request_id=request_id, elapsed_ms=round(elapsed_ms, 1))

    @property
    def inflight(self) -> int:
        """Current number of in-flight HTTP requests."""
        return self._inflight

    def close(self) -> None:
        """Close the log file."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


# Global profiler instance (activated by init_http_profiler).
_http_profiler = HttpConcurrencyProfiler()


def get_http_profiler() -> HttpConcurrencyProfiler:
    """Get the global HTTP concurrency profiler instance."""
    return _http_profiler


def init_http_profiler(log_path: str) -> None:
    """Initialize and enable the global HTTP concurrency profiler.

    Args:
        log_path: Path to the JSONL output file.
    """
    _http_profiler.enable(log_path)
