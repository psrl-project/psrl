"""Tests for the external verifier client."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from examples.math_rl.mock_verifier_server import build_app
from examples.math_rl.verifier import (
    VerifierConfig,
    VerifierError,
    evaluate,
    reset_verifier_clients,
)


def _config(endpoint: str, **overrides) -> VerifierConfig:
    values = {
        "endpoint": endpoint,
        "verifier_harness_version": "V3",
        "model": "mock-verifier",
        "reasoning_effort": "max",
        "transport": "blocking",
        "stage_timeout_s": 600.0,
        "max_retries": 2,
    }
    values.update(overrides)
    return VerifierConfig(**values)


@asynccontextmanager
async def serve(handler):
    app = web.Application()
    app.router.add_post("/verify", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/verify"))
    finally:
        await reset_verifier_clients()
        await server.close()


@asynccontextmanager
async def serve_stream(handler):
    """Serve only the streaming route, so the derived URL is exercised."""
    app = web.Application()
    app.router.add_post("/verify/stream", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/verify"))
    finally:
        await reset_verifier_clients()
        await server.close()


async def _sse(request, events) -> web.StreamResponse:
    response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "x-verifier-receipt": "run-9"})
    await response.prepare(request)
    for event, payload in events:
        await response.write(f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode())
    await response.write_eof()
    return response


async def test_verify_sends_exactly_four_fields_and_reads_the_receipt():
    seen = {}

    async def handler(request):
        seen["body"] = await request.json()
        return web.json_response(
            {"verdict": "6", "reason": "one gap"},
            headers={
                "x-verifier-receipt": "run-1",
                "x-verifier-harness-sha256": "h",
                "x-verifier-prompt-sha256": "p",
                "x-verifier-skill-sha256": "s",
                "x-verifier-duration-seconds": "1.5",
                "x-verifier-usage": '{"input_tokens": 12}',
            },
        )

    async with serve(handler) as endpoint:
        verdict = await evaluate(_config(endpoint), "Prove P.", "A proof.")

    assert set(seen["body"]) == {
        "problem",
        "solution_to_verify",
        "model_config",
        "verifier_harness_version",
    }
    assert seen["body"]["model_config"] == {
        "model": "mock-verifier",
        "reasoning_effort": "max",
        "timeout_seconds": 600.0,
    }
    assert verdict.verdict == "6"
    assert verdict.run_id == "run-1"
    assert verdict.duration_s == 1.5
    assert verdict.usage == {"input_tokens": 12}
    assert verdict.proof_sha256
    assert verdict.cached is False


async def test_a_verdict_maps_through_the_configured_reward():
    async def handler(request):
        return web.json_response({"verdict": "6", "reason": ""})

    async with serve(handler) as endpoint:
        graded = await evaluate(_config(endpoint), "P", "S")
        binary = await evaluate(_config(endpoint, reward_mode="binary"), "P", "S")

    assert graded.score(_config("http://unused/verify")) == 0.8
    assert binary.score(_config("http://unused/verify", reward_mode="binary")) == 1.0


async def test_an_identical_proof_is_served_from_the_cache():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        return web.json_response({"verdict": "7", "reason": ""})

    async with serve(handler) as endpoint:
        config = _config(endpoint)
        first = await evaluate(config, "P", "S")
        second = await evaluate(config, "P", "S")
        third = await evaluate(config, "P", "S2")

    assert calls["count"] == 2
    assert first.cached is False
    assert second.cached is True
    assert third.cached is False


async def test_a_5xx_is_retried():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        if calls["count"] == 1:
            return web.json_response({"detail": "busy"}, status=503)
        return web.json_response({"verdict": "7", "reason": ""})

    async with serve(handler) as endpoint:
        verdict = await evaluate(_config(endpoint), "P", "S")

    assert calls["count"] == 2
    assert verdict.verdict == "7"


async def test_a_rejected_request_is_not_retried():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        return web.json_response({"detail": "bad"}, status=422)

    async with serve(handler) as endpoint:
        with pytest.raises(VerifierError) as caught:
            await evaluate(_config(endpoint), "P", "S")

    assert calls["count"] == 1
    assert caught.value.kind == "rejected"


async def test_an_unknown_verdict_is_ungraded():
    async def handler(request):
        return web.json_response({"verdict": "9", "reason": "?"})

    async with serve(handler) as endpoint:
        with pytest.raises(VerifierError) as caught:
            await evaluate(_config(endpoint), "P", "S")

    assert caught.value.kind == "malformed"


async def test_an_oversized_solution_never_reaches_the_service():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        return web.json_response({"verdict": "7", "reason": ""})

    async with serve(handler) as endpoint:
        with pytest.raises(VerifierError) as caught:
            await evaluate(_config(endpoint, max_solution_bytes=4), "P", "12345")

    assert calls["count"] == 0
    assert caught.value.kind == "payload"


def test_the_config_needs_an_endpoint_and_a_model():
    with pytest.raises(ValueError, match="endpoint"):
        VerifierConfig(verifier_harness_version="V3", model="m")
    with pytest.raises(ValueError, match="model or a profile"):
        VerifierConfig(endpoint="http://unused/verify", verifier_harness_version="V3")
    with pytest.raises(ValueError, match="Reward mode"):
        VerifierConfig(
            endpoint="http://unused/verify",
            verifier_harness_version="V3",
            model="m",
            reward_mode="stars",
        )


async def test_the_mock_server_answers_the_contract():
    app = build_app("mock-v1", "hashed")
    server = TestServer(app)
    await server.start_server()
    try:
        endpoint = str(server.make_url("/verify"))
        config = _config(endpoint, verifier_harness_version="mock-v1")
        first = await evaluate(config, "P", "S")
        second = await evaluate(config, "P", "S2")
        assert first.verdict in {"0", "1", "6", "7"}
        assert second.verdict in {"0", "1", "6", "7"}
        assert first.proof_sha256 != second.proof_sha256
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json={"problem": "P"}) as response:
                assert response.status == 422
            async with session.get(str(server.make_url("/health"))) as response:
                assert (await response.json())["verifier_harness_version"] == "mock-v1"
    finally:
        await reset_verifier_clients()
        await server.close()


async def test_the_stream_transport_reads_the_terminal_event():
    async def handler(request):
        return await _sse(
            request,
            [
                ("progress", {"run_id": "run-9", "stage": "accepted"}),
                ("progress", {"run_id": "run-9", "stage": "contract", "elapsed_seconds": 1.0}),
                ("progress", {"run_id": "run-9", "stage": "audit"}),
                ("progress", {"run_id": "run-9", "stage": "grading"}),
                (
                    "result",
                    {
                        "run_id": "run-9",
                        "stage": "completed",
                        "result": {"verdict": "6", "reason": "minor gap"},
                        "receipt": {
                            "run_id": "run-9",
                            "harness_sha256": "h",
                            "prompt_sha256": "p",
                            "skill_sha256": "s",
                            "duration_seconds": 12.5,
                            "usage": {"input_tokens": 30},
                        },
                    },
                ),
            ],
        )

    async with serve_stream(handler) as endpoint:
        verdict = await evaluate(_config(endpoint, transport="stream"), "P", "S")

    assert verdict.verdict == "6"
    assert verdict.reason == "minor gap"
    assert verdict.run_id == "run-9"
    assert verdict.duration_s == 12.5
    assert verdict.usage == {"input_tokens": 30}
    assert verdict.cached is False


async def test_a_stream_error_event_is_not_retried():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        return await _sse(
            request,
            [
                ("progress", {"stage": "accepted"}),
                ("error", {"stage": "failed", "failed_stage": "audit"}),
            ],
        )

    async with serve_stream(handler) as endpoint:
        with pytest.raises(VerifierError) as caught:
            await evaluate(_config(endpoint, transport="stream"), "P", "S")

    assert calls["count"] == 1
    assert caught.value.kind == "upstream"
    assert caught.value.retryable is False
    assert "audit" in str(caught.value)


async def test_a_stream_failure_before_the_first_event_is_retried():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        if calls["count"] == 1:
            return web.json_response({"detail": "busy"}, status=503)
        return await _sse(
            request,
            [("result", {"result": {"verdict": "7", "reason": ""}, "receipt": {"run_id": "r", "usage": {}}})],
        )

    async with serve_stream(handler) as endpoint:
        verdict = await evaluate(_config(endpoint, transport="stream"), "P", "S")

    assert calls["count"] == 2
    assert verdict.verdict == "7"


async def test_a_stream_without_a_result_is_ungraded():
    async def handler(request):
        return await _sse(request, [("progress", {"stage": "accepted"})])

    async with serve_stream(handler) as endpoint:
        with pytest.raises(VerifierError) as caught:
            await evaluate(_config(endpoint, transport="stream"), "P", "S")

    assert caught.value.kind == "malformed"


async def test_the_whole_request_body_is_bounded():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        return web.json_response({"verdict": "7", "reason": ""})

    async with serve(handler) as endpoint:
        with pytest.raises(VerifierError) as caught:
            await evaluate(
                _config(endpoint, max_solution_bytes=100, max_request_bytes=200),
                "P" * 200,
                "S",
            )

    assert calls["count"] == 0
    assert caught.value.kind == "payload"


async def test_the_mock_server_streams_the_same_contract():
    app = build_app("mock-v1", "hashed")
    server = TestServer(app)
    await server.start_server()
    try:
        endpoint = str(server.make_url("/verify"))
        streamed = await evaluate(
            _config(endpoint, verifier_harness_version="mock-v1", transport="stream"), "P", "S"
        )
        blocking = await evaluate(
            _config(endpoint, verifier_harness_version="mock-v1", transport="blocking"), "P", "S"
        )
    finally:
        await reset_verifier_clients()
        await server.close()

    assert streamed.verdict in {"0", "1", "6", "7"}
    assert streamed.run_id.startswith("mock_")
    assert streamed.usage
    assert streamed.verdict == blocking.verdict


def test_the_config_carries_the_outbound_proxy():
    config = VerifierConfig(
        endpoint="http://unused/verify",
        verifier_harness_version="V3",
        model="m",
        proxy="http://proxy:3128",
    )
    assert config.proxy == "http://proxy:3128"


async def test_a_stage_timeout_is_not_retried():
    calls = {"count": 0}

    async def handler(request):
        calls["count"] += 1
        return web.json_response(
            {"detail": "codex exec timed out", "error_code": "verification_timeout"},
            status=504,
        )

    async with serve(handler) as endpoint:
        with pytest.raises(VerifierError) as caught:
            await evaluate(_config(endpoint), "P", "S")

    assert calls["count"] == 1
    assert caught.value.kind == "timeout"
    assert caught.value.retryable is False
