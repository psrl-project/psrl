from __future__ import annotations

import asyncio

import pytest
from psrl.sandbox.capacity import (
    ResourceQuantity,
    SandboxCapacityConfig,
    SandboxCapacityCoordinator,
    parse_memory_mb,
    resolve_sandbox_capacity,
)
from psrl.sandbox.core import ResourceSpec


@pytest.mark.parametrize(
    ("value", "expected"),
    [("8g", 8192), ("512m", 512), (1073741824, 1024), (None, None)],
)
def test_parse_memory_mb(value, expected) -> None:
    assert parse_memory_mb(value) == expected


def test_resource_quantity_uses_actual_container_request() -> None:
    quantity = ResourceQuantity.from_spec(ResourceSpec(cpu_count=1.25, memory_mb=3072))

    assert quantity == ResourceQuantity(memory_mb=3072, cpu_millis=1250)


def test_resolve_capacity_has_one_shared_utilization_margin() -> None:
    resolved = resolve_sandbox_capacity(
        SandboxCapacityConfig(utilization=0.75),
        detected_memory_mb=10000,
        detected_cpu_cores=8,
    )

    assert resolved.resources == ResourceQuantity(memory_mb=7500, cpu_millis=6000)
    assert resolved.memory_source == "detected"
    assert resolved.cpu_source == "detected"


def test_explicit_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError, match="memory_mb"):
        SandboxCapacityConfig(memory_mb=0)
    with pytest.raises(ValueError, match="cpu_cores"):
        SandboxCapacityConfig(cpu_cores=0)


@pytest.mark.asyncio
async def test_multi_resource_admission_blocks_until_complete_request_fits() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    await coordinator.acquire("large", "worker-a", 80, 1)
    waiting = asyncio.create_task(coordinator.acquire("cpu-heavy", "worker-b", 10, 4))
    await asyncio.sleep(0)

    assert not waiting.done()
    snapshot = await coordinator.snapshot()
    assert snapshot["available_capacity"] == {"memory_mb": 20, "cpu_millis": 3000}
    assert snapshot["waiters"] == 1

    await coordinator.release("large")
    await asyncio.wait_for(waiting, timeout=1)
    assert (await coordinator.snapshot())["allocations"] == 1
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_different_request_sizes_share_capacity_without_static_pools() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=10, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    requests = [("rollout-a", 20, 2), ("grader", 50, 4), ("rollout-b", 30, 3)]

    await asyncio.gather(*(coordinator.acquire(lease_id, "worker", memory, cpu) for lease_id, memory, cpu in requests))

    snapshot = await coordinator.snapshot()
    assert snapshot["available_capacity"] == {"memory_mb": 0, "cpu_millis": 1000}
    assert snapshot["allocations"] == 3
    await coordinator.release_owner("worker")
    assert (await coordinator.snapshot())["available_capacity"] == {"memory_mb": 100, "cpu_millis": 10000}
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_admission_prefers_lower_dominant_resource_weight() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=10, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    await coordinator.acquire("blocker", "owner", 100, 10)
    large = asyncio.create_task(coordinator.acquire("large", "owner", 80, 8))
    small = asyncio.create_task(coordinator.acquire("small", "owner", 30, 3))
    await asyncio.sleep(0)

    await coordinator.release("blocker")
    await asyncio.sleep(0)

    assert small.done()
    assert not large.done()
    await coordinator.release("small")
    await large
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_oldest_request_reserves_capacity_after_bounded_bypasses() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=10, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    await coordinator.acquire("blocker", "owner", 100, 10)
    large = asyncio.create_task(coordinator.acquire("large", "owner", 100, 10))
    small = [asyncio.create_task(coordinator.acquire(f"small-{index}", "owner", 10, 1)) for index in range(10)]
    await asyncio.sleep(0)

    await coordinator.release("blocker")
    await asyncio.sleep(0)

    admitted_small = [task for task in small if task.done()]
    assert len(admitted_small) == 8
    assert not large.done()

    for index in range(8):
        await coordinator.release(f"small-{index}")
    await large
    assert not small[8].done()
    assert not small[9].done()

    await coordinator.release("large")
    await asyncio.gather(*small)
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_oversized_request_fails_immediately() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )

    with pytest.raises(ValueError, match="exceeds node envelope"):
        await coordinator.acquire("too-large", "worker", 101, 1)

    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_cancel_removes_queued_request() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(memory_mb=100, cpu_cores=4, utilization=1, lease_ttl_s=10, heartbeat_interval_s=1)
    )
    await coordinator.acquire("active", "worker-a", 100, 4)
    waiting = asyncio.create_task(coordinator.acquire("waiting", "worker-b", 100, 4))
    await asyncio.sleep(0)

    await coordinator.cancel("waiting")

    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert (await coordinator.snapshot())["waiters"] == 0
    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_expired_worker_lease_returns_capacity() -> None:
    coordinator = SandboxCapacityCoordinator(
        SandboxCapacityConfig(
            memory_mb=100,
            cpu_cores=4,
            utilization=1,
            lease_ttl_s=0.08,
            heartbeat_interval_s=0.02,
        )
    )
    await coordinator.acquire("abandoned", "dead-worker", 100, 4)
    waiting = asyncio.create_task(coordinator.acquire("next", "live-worker", 100, 4))

    async def heartbeat() -> None:
        while not waiting.done():
            await asyncio.sleep(0.02)
            await coordinator.renew_owner("live-worker")

    heartbeat_task = asyncio.create_task(heartbeat())

    await asyncio.wait_for(waiting, timeout=0.5)
    await heartbeat_task

    snapshot = await coordinator.snapshot()
    assert snapshot["expired"] == 1
    assert snapshot["allocations"] == 1
    await coordinator.shutdown()
