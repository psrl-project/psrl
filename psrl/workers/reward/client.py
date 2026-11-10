"""
Bound reward RPC submissions independently of rollout buffering.
"""

import asyncio
from typing import Any

import ray
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

REWARD_MANAGER_CONTROL_CONCURRENCY = 32


def reward_pool_concurrency(
    worker_config,
) -> int:
    """
    Return the number of reward requests the worker pool can execute concurrently.
    """
    worker_count = int(worker_config.num_workers)
    worker_concurrency = int(worker_config.max_concurrency_per_worker)
    if worker_count <= 0 or worker_concurrency <= 0:
        raise ValueError("Reward worker count and concurrency must be positive.")
    return worker_count * worker_concurrency


def reward_manager_concurrency(
    worker_config,
) -> int:
    """
    Reserve actor slots for request waiters, result callbacks, and control RPCs.
    """
    return 2 * reward_pool_concurrency(worker_config) + REWARD_MANAGER_CONTROL_CONCURRENCY


def reward_producer_concurrency(
    worker_config,
    producer_count: int,
    producer_index: int,
) -> int:
    """
    Partition the pool request budget across agent workers without exceeding it.
    """
    capacity = reward_pool_concurrency(worker_config)
    if not 0 < producer_count <= capacity or not 0 <= producer_index < producer_count:
        raise ValueError("Reward producer count and index must fit the worker pool capacity.")
    quotient, remainder = divmod(capacity, producer_count)
    return quotient + int(producer_index < remainder)


class RewardManagerClient:
    """
    Keep a bounded number of RPCs active until Ray confirms their completion.
    """

    def __init__(
        self,
        manager: ray.actor.ActorHandle,
        max_pending_requests: int,
    ):
        if max_pending_requests <= 0:
            raise ValueError("Reward RPC capacity must be positive.")
        self._manager = manager
        self._semaphore = asyncio.Semaphore(max_pending_requests)
        self._pending: set[asyncio.Future] = set()

    async def _call(
        self,
        method: str,
        payload: Any,
    ) -> Any:
        await self._semaphore.acquire()
        try:
            future = asyncio.ensure_future(getattr(self._manager, method).remote(payload))
        except BaseException:
            self._semaphore.release()
            raise
        self._pending.add(future)
        future.add_done_callback(self._finish_request)
        # Caller cancellation must not free capacity while the Ray RPC is still active.
        try:
            return await asyncio.shield(future)
        finally:
            if future.done():
                self._finish_request(future)

    def _finish_request(
        self,
        future: asyncio.Future,
    ) -> None:
        if future not in self._pending:
            return
        self._pending.remove(future)
        self._semaphore.release()
        if not future.cancelled():
            future.exception()

    async def compute_score(
        self,
        reward_inputs: TensorDict,
    ) -> dict[str, Any] | None:
        """
        Submit reward inputs while applying backpressure to the agent worker.
        """
        return await self._call("compute_score", reward_inputs)

    async def put_reward_result(
        self,
        result: dict[str, Any],
    ) -> None:
        """
        Wait for the manager to accept a completed reward result.
        """
        await self._call("put_reward_result", result)


class RewardSubmissionClient:
    """
    Route agent rewards to independently bounded training and validation clients.
    """

    def __init__(
        self,
        training: RewardManagerClient,
        validation: RewardManagerClient,
    ):
        self._training = training
        self._validation = validation

    async def compute_score(
        self,
        reward_inputs: TensorDict,
    ) -> dict[str, Any] | None:
        """
        Select the reward pool using the request's `validate` metadata.
        """
        client = self._validation if tu.get(reward_inputs, "validate", False) else self._training
        return await client.compute_score(reward_inputs)
