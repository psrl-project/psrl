"""
Verify gateway startup rejects unavailable native elastic APIs.
"""

from unittest.mock import Mock

import httpx
import psrl.workers.gen.smg_control as control
import pytest

pytestmark = pytest.mark.cpu_test


def _client(monkeypatch, responses):
    requests = []

    def respond(request):
        requests.append(request)
        result = responses[request.url.path]
        if isinstance(result, Exception):
            raise result
        status, body = result
        if body is None:
            return httpx.Response(status, content=b"null")
        return httpx.Response(status, json=body)

    client_cls = httpx.Client
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(control.httpx, "Client", lambda **kwargs: client_cls(transport=transport, **kwargs))
    return requests


def _responses():
    return {
        "/health": (200, {}),
        "/routing_loop/status": (200, {"paused": False}),
        "/routing_loop/snapshot": (200, []),
        "/routing_loop/migration": (200, None),
    }


@pytest.mark.parametrize("elastic", [False, True])
def test_readiness_checks_required_control_apis(monkeypatch, elastic):
    requests = _client(monkeypatch, _responses())
    control.wait_for_router_ready(
        "http://127.0.0.1:8100", Mock(is_alive=Mock(return_value=True)), require_elastic_runtime=elastic
    )
    expected = ["/health", "/routing_loop/status"]
    if elastic:
        expected += ["/routing_loop/snapshot", "/routing_loop/migration"]
        assert requests[-1].url.params["request_id"] == "psrl-startup-check", "Migration probe must be read only."
    assert [request.url.path for request in requests] == expected, "All required APIs must be checked."


@pytest.mark.parametrize("path", ["/routing_loop/status", "/routing_loop/snapshot", "/routing_loop/migration"])
def test_missing_native_api_fails_with_rebuild_instruction(monkeypatch, path):
    responses = _responses()
    responses[path] = (404, {"error": "Not found"})
    requests = _client(monkeypatch, responses)
    with pytest.raises(RuntimeError, match="reinstall_smg.sh") as error:
        control.wait_for_router_ready(
            "http://127.0.0.1:8100", Mock(is_alive=Mock(return_value=True)), require_elastic_runtime=True
        )
    assert path in str(error.value), "The error must name the unavailable API."
    assert requests[-1].url.path == path, "An unavailable API must fail initialization immediately."


def test_invalid_snapshot_shape_fails_readiness(monkeypatch):
    responses = _responses()
    responses["/routing_loop/snapshot"] = (200, {"queue_len": 0})
    _client(monkeypatch, responses)
    with pytest.raises(TypeError, match="must return list"):
        control.wait_for_router_ready(
            "http://127.0.0.1:8100", Mock(is_alive=Mock(return_value=True)), require_elastic_runtime=True
        )


def test_exited_router_fails_readiness(monkeypatch):
    requests = _client(monkeypatch, _responses())
    with pytest.raises(RuntimeError, match="exited during initialization"):
        control.wait_for_router_ready(
            "http://127.0.0.1:8100", Mock(is_alive=Mock(return_value=False)), require_elastic_runtime=True
        )
    assert not requests, "An exited router must fail before an HTTP request."


def test_router_startup_timeout_is_bounded(monkeypatch):
    requests = _client(monkeypatch, _responses())
    times = iter([0.0, 121.0])
    monkeypatch.setattr(control.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="initialization timed out"):
        control.wait_for_router_ready(
            "http://127.0.0.1:8100",
            Mock(is_alive=Mock(return_value=True)),
            require_elastic_runtime=True,
        )
    assert not requests, "An expired startup deadline must fail before an HTTP request."


def test_readiness_retries_connection_failure_until_http_is_ready(monkeypatch):
    client_cls = httpx.Client
    attempts = []

    def respond(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            raise httpx.ConnectError("Router is starting.", request=request)
        return httpx.Response(200, json={"paused": False})

    monkeypatch.setattr(
        control.httpx,
        "Client",
        lambda **kwargs: client_cls(
            transport=httpx.MockTransport(respond),
            **kwargs,
        ),
    )
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    control.wait_for_router_ready(
        "http://127.0.0.1:8100", Mock(is_alive=Mock(return_value=True)), require_elastic_runtime=False
    )
    assert attempts == ["/health", "/health", "/routing_loop/status"], "Connection failures must be retried."
