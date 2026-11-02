"""The client keeps the service's own classification of a failure.

A command timeout is a typed result rather than a transport fault: the sandbox
survives and the episode continues with a fresh shell. It is also a TimeoutError,
because callers reasonably treat "this took too long" that way, so a client that
catches asyncio.TimeoutError before the typed error rewrites it -- and the caller
abandons work the sandbox could still finish.
"""

from __future__ import annotations

import asyncio
import json
import socket

import pytest

from sandboxd.client import Transport
from sandboxd.errors import SandboxCommandTimeout, SandboxTransportError


async def serve_one(reply: dict | None) -> tuple[asyncio.AbstractServer, int, asyncio.Future[dict]]:
    """Serve one framed request, answering with `reply`, or never answering."""
    received: asyncio.Future[dict] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = await reader.readexactly(4)
            body = await reader.readexactly(int.from_bytes(header, "big"))
            if not received.done():
                received.set_result(json.loads(body))
            if reply is not None:
                frame = json.dumps(reply).encode()
                writer.write(len(frame).to_bytes(4, "big") + frame)
                await writer.drain()
            await reader.read()
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, int(server.sockets[0].getsockname()[1]), received


def call_once(reply: dict | None, *, timeout_s: float, expect):
    """Make one call against a fresh fake service and return what it received."""

    async def run():
        server, port, received = await serve_one(reply)
        transport = Transport(f"127.0.0.1:{port}", pool_size=1)
        try:
            with pytest.raises(expect) as caught:
                await transport.call("exec", {"sandbox_id": "sb-1", "command": "sleep 30"}, timeout_s=timeout_s)
        finally:
            await transport.close()
            server.close()
            await server.wait_closed()
        return caught.value, await asyncio.wait_for(received, timeout=1)

    return asyncio.run(run())


def test_a_command_timeout_is_not_rewritten_as_a_transport_fault():
    error, request = call_once(
        {"error": {"code": "command_timeout", "message": "stopped at its budget", "preserved": True}},
        timeout_s=30,
        expect=SandboxCommandTimeout,
    )
    if request["method"] != "exec":
        pytest.fail(f"the fake service received {request!r}")
    if isinstance(error, SandboxTransportError):
        raise AssertionError("a typed command timeout was reported as a transport fault")
    if not error.sandbox_preserved:
        raise AssertionError("the preserved flag from the service was dropped")


def test_a_silent_service_is_still_a_transport_fault():
    call_once(None, timeout_s=0.2, expect=SandboxTransportError)


def test_an_unreachable_service_is_a_transport_fault():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    # The port is closed again, so nothing is listening on it.

    async def run():
        transport = Transport(f"127.0.0.1:{port}", pool_size=1)
        try:
            with pytest.raises(SandboxTransportError):
                await transport.call("exec", {"sandbox_id": "sb-1"}, timeout_s=1)
        finally:
            await transport.close()

    asyncio.run(run())
