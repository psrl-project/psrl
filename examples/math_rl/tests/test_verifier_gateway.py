"""Tests for the per-worker verifier gateway."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from examples.math_rl.verifier import VerifierConfig, reset_verifier_clients
from examples.math_rl.verifier_gateway import VerifierGateway, reset_verifier_gateways


def _config(endpoint: str) -> VerifierConfig:
    return VerifierConfig(
        endpoint=endpoint,
        verifier_harness_version="V3",
        model="mock-verifier",
        transport="blocking",
        stage_timeout_s=60.0,
        max_retries=1,
    )


@asynccontextmanager
async def mock_verifier(verdict: str = "7", *, delay: float = 0.0, counter: list[int] | None = None):
    async def handler(request):
        if counter is not None:
            counter[0] += 1
        if delay:
            await asyncio.sleep(delay)
        return web.json_response(
            {"verdict": verdict, "reason": f"reason for {verdict}"},
            headers={
                "x-verifier-receipt": "run-1",
                "x-verifier-usage": '{"input_tokens": 10, "output_tokens": 5}',
            },
        )

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
async def gateway_for(endpoint: str, **kwargs):
    gateway = VerifierGateway(
        _config(endpoint),
        host="127.0.0.1",
        advertised_host="127.0.0.1",
        **kwargs,
    )
    await gateway.start()
    try:
        yield gateway
    finally:
        await gateway.stop()
        await reset_verifier_gateways()
        await reset_verifier_clients()


async def test_a_valid_verdict_maps_to_the_rethlas_schema():
    async with mock_verifier("7") as endpoint, gateway_for(endpoint) as gateway:
        url = gateway.register_episode("ep-1", "Prove P.")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"statement": "P", "proof": "A proof."}) as response:
                body = await response.json()
    assert response.status == 200
    assert body["verdict"] == "correct"
    assert body["repair_hints"] == ""
    assert body["verdict_code"] == "7"
    assert body["verification_report"]["summary"] == "reason for 7"
    assert body["verification_report"]["critical_errors"] == []


async def test_a_rejected_verdict_becomes_repair_hints():
    async with mock_verifier("0") as endpoint, gateway_for(endpoint) as gateway:
        url = gateway.register_episode("ep-1", "Prove P.")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"statement": "P", "proof": "A proof."}) as response:
                body = await response.json()
    assert response.status == 200
    assert body["verdict"] == "wrong"
    assert body["repair_hints"] == "reason for 0"


async def test_an_unknown_episode_is_rejected():
    async with mock_verifier("7") as endpoint, gateway_for(endpoint) as gateway:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{gateway.local_url}/verify/missing",
                json={"statement": "P", "proof": "A proof."},
            ) as response:
                await response.json()
    assert response.status == 404


async def test_concurrent_identical_proofs_collapse_to_one_call():
    counter = [0]
    async with mock_verifier("7", delay=0.3, counter=counter) as endpoint, gateway_for(endpoint) as gateway:
        url = gateway.register_episode("ep-1", "Prove P.")
        async with aiohttp.ClientSession() as session:
            responses = await asyncio.gather(
                *[
                    session.post(url, json={"statement": "P", "proof": "A proof."})
                    for _ in range(2)
                ]
            )
            bodies = [await response.json() for response in responses]
        # A third call is served by the client's proof cache.
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"statement": "P", "proof": "A proof."}) as response:
                await response.json()
    assert [body["verdict"] for body in bodies] == ["correct", "correct"]
    assert counter[0] == 1


async def test_the_per_episode_call_cap_is_enforced():
    async with mock_verifier("7") as endpoint, gateway_for(endpoint, max_calls_per_episode=1) as gateway:
        url = gateway.register_episode("ep-1", "Prove P.")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"statement": "P", "proof": "A proof."}) as response:
                assert response.status == 200
            async with session.post(url, json={"statement": "P", "proof": "A proof."}) as response:
                body = await response.json()
    assert response.status == 429
    assert body["error"] == "budget_exceeded"


async def test_health_and_metrics_report_the_episode():
    async with mock_verifier("7") as endpoint, gateway_for(endpoint) as gateway:
        url = gateway.register_episode("ep-1", "Prove P.")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json={"statement": "P", "proof": "A proof."}):
                pass
            async with session.get(f"{gateway.local_url}/health") as response:
                health = await response.json()
            async with session.get(f"{gateway.local_url}/metrics") as response:
                metrics = await response.json()
    assert health["episodes"] == 1
    assert metrics["total_calls"] == 1
    assert metrics["verdict_counts"] == {"7": 1}
    assert metrics["episodes"]["ep-1"]["calls"] == 1


def test_the_gateway_config_rejects_an_empty_call_budget():
    with pytest.raises(ValueError, match="max_calls_per_episode"):
        VerifierGateway(_config("http://unused/verify"), max_calls_per_episode=0)
