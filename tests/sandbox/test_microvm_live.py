"""Opt-in AgentEnv/CubeSandbox control- and state-plane conformance test."""

from __future__ import annotations

import os
import uuid

import pytest
from psrl.sandbox import SandboxManager, SandboxSource, SandboxSpec, SandboxStatePolicy, SnapshotKind
from psrl.sandbox.backends.e2b import AgentEnvBackend, CubeSandboxBackend

_BACKEND = os.getenv("PSRL_LIVE_MICROVM_BACKEND", "")
pytestmark = pytest.mark.skipif(
    _BACKEND not in {"agentenv", "cubesandbox"},
    reason="set PSRL_LIVE_MICROVM_BACKEND=agentenv or cubesandbox",
)


@pytest.mark.asyncio
async def test_live_microvm_snapshot_restore_and_transport_refresh() -> None:
    api_url = os.environ["PSRL_LIVE_MICROVM_API_URL"]
    api_key = os.getenv("PSRL_LIVE_MICROVM_API_KEY")
    source_ref = os.environ["PSRL_LIVE_MICROVM_SOURCE"]
    backend = (
        AgentEnvBackend(api_url=api_url, api_key=api_key)
        if _BACKEND == "agentenv"
        else CubeSandboxBackend(api_url=api_url, api_key=api_key)
    )
    manager = SandboxManager({_BACKEND: backend}, _BACKEND)
    source = (
        SandboxSource.image(source_ref)
        if _BACKEND == "agentenv" and os.getenv("PSRL_LIVE_MICROVM_SOURCE_KIND", "template") == "image"
        else SandboxSource.template(source_ref)
    )
    policy = SandboxStatePolicy(enabled=True)
    snapshot = None
    try:
        lease = await manager.acquire(
            SandboxSpec(
                source,
                idempotency_key=f"microvm-live-{uuid.uuid4().hex}",
                idle_timeout_s=300,
                state_policy=policy,
            )
        )
        await lease.session.write_bytes("/tmp/psrl-baseline", b"clean")
        snapshot = await manager.checkpoint(lease.session, SnapshotKind.FULL_STATE)
        restored = await manager.restore(snapshot, state_policy=policy)
        branched = await manager.branch(lease.session, state_policy=policy)

        assert await restored.session.read_bytes("/tmp/psrl-baseline") == b"clean"
        assert await branched.session.read_bytes("/tmp/psrl-baseline") == b"clean"
        assert (await lease.session.exec("printf parent-alive")).stdout == "parent-alive"
        assert (await restored.session.exec("printf child-alive")).stdout == "child-alive"
    finally:
        try:
            if snapshot is not None:
                await manager.delete_snapshot(snapshot)
        finally:
            await manager.shutdown()
