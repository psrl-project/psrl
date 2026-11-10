"""Verify shared GPU ownership through concurrent role admissions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from psrl.workers.elastic.sleep_wake_orchestrator import SleepWakeOrchestrator, inference_lease

pytestmark = pytest.mark.cpu_test


def _orchestrator():
    events = []

    def coordinator(name):
        async def sleep(tag="all"):
            events.append((name, "sleep", tag))
            return True

        async def wake(tag="all"):
            events.append((name, "wake", tag))
            return True

        return SimpleNamespace(
            sleep=SimpleNamespace(remote=AsyncMock(side_effect=sleep)),
            wake_up=SimpleNamespace(remote=AsyncMock(side_effect=wake)),
        )

    owner = SleepWakeOrchestrator.__ray_metadata__.modified_class(
        "colocated", coordinator("policy"), {name: coordinator(name) for name in ("judge-a", "judge-b")}
    )
    return owner, events


async def test_conflicting_roles_wait_for_every_active_user_and_prevent_starvation():
    owner, events = _orchestrator()
    await owner.start()
    await owner.acquire("r1", "rollout")
    await owner.acquire("r2", "rollout")
    reward = asyncio.create_task(owner.acquire("rm", "reward", "judge-a"))
    await asyncio.sleep(0)
    next_rollout = asyncio.create_task(owner.acquire("r3", "rollout"))
    await asyncio.sleep(0)
    await owner.release("r1")
    assert not reward.done() and not next_rollout.done(), "Every current user must drain before a switch."
    await owner.release("r2")
    await reward
    assert not next_rollout.done(), "New rollout work must wait behind the already queued reward role."
    assert events[-2:] == [("policy", "sleep", "rollout"), ("judge-a", "wake", "all")], (
        "Outgoing memory must release before incoming memory is restored."
    )
    await owner.release("rm")
    await next_rollout
    await owner.release("r3")


async def test_validation_reward_and_training_use_the_same_storage_gate():
    owner, events = _orchestrator()
    await owner.start()
    await owner.acquire("v", "validate")
    reward = asyncio.create_task(owner.acquire("a", "reward", "judge-a"))
    other_reward = asyncio.create_task(owner.acquire("b", "reward", "judge-b"))
    train = asyncio.create_task(owner.acquire("t", "train"))
    await asyncio.sleep(0)
    await owner.release("v")
    await reward
    assert not other_reward.done() and not train.done(), "Different reward models must not overlap shared storage."
    await owner.release("a")
    await other_reward
    await owner.release("b")
    await train
    assert events[-1] == ("judge-b", "sleep", "all"), "Training admission must leave all inference memory asleep."
    await owner.release("t")


async def test_cancelled_waiter_cannot_leak_a_lease_or_block_the_next_role():
    owner, _ = _orchestrator()
    await owner.start()
    await owner.acquire("rollout", "rollout")
    waiter = asyncio.create_task(owner.acquire("cancelled", "reward", "judge-a"))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    await owner.release("cancelled")
    await owner.release("rollout")
    await owner.acquire("train", "train")
    await owner.release("train")


async def test_cancelled_caller_waits_for_admission_acknowledgement_before_release():
    owner, _ = _orchestrator()
    await owner.start()
    await owner.acquire("rollout", "rollout")
    client = SimpleNamespace(
        acquire=SimpleNamespace(remote=owner.acquire),
        release=SimpleNamespace(remote=owner.release),
    )

    async def caller():
        async with inference_lease(client, "reward", "judge-a"):
            raise AssertionError("Cancelled callers cannot use model memory.")

    task = asyncio.create_task(caller())
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    await owner.release("rollout")
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.wait_for(owner.acquire("train", "train"), timeout=1)
    await owner.release("train")


async def test_failed_transition_rejects_every_waiter_and_never_admits_training():
    owner, _ = _orchestrator()
    await owner.start()
    await owner.acquire("rollout", "rollout")
    owner.reward_model_coordinators["judge-a"].wake_up.remote.side_effect = RuntimeError("Restore failed.")
    reward = asyncio.create_task(owner.acquire("reward", "reward", "judge-a"))
    train = asyncio.create_task(owner.acquire("train", "train"))
    await asyncio.sleep(0)
    await owner.release("rollout")
    results = await asyncio.gather(reward, train, return_exceptions=True)
    assert all(isinstance(result, RuntimeError) for result in results), (
        "A failed transition cannot hand off ownership."
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        await owner.acquire("retry", "train")
