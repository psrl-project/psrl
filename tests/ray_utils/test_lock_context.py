import asyncio
import time
from collections import deque

import ray


def add_lock(cls):
    """Add asynchronous lock methods to a class."""
    original_init = getattr(cls, "__init__", None)

    def __init__(self, *args, **kwargs):
        if original_init is not None:
            original_init(self, *args, **kwargs)
        else:
            super(cls, self).__init__(*args, **kwargs)
        self._locked = False
        self._waiters = deque()

    async def acquire(self):
        if not self._locked:
            self._locked = True
            return
        fut = asyncio.get_event_loop().create_future()
        self._waiters.append(fut)
        await fut

    async def release(self):
        if self._waiters:
            fut = self._waiters.popleft()
            fut.set_result(None)
        else:
            self._locked = False

    cls.__init__ = __init__
    cls.acquire = acquire
    cls.release = release
    return cls


class RayLock:
    """Synchronously manage a remote actor lock from the driver."""

    def __init__(self, actor_handle):
        self._actor = actor_handle

    def __enter__(self):
        ray.get(self._actor.acquire.remote())

    def __exit__(self, exc_type, exc, tb):
        ray.get(self._actor.release.remote())


@ray.remote
@add_lock
class CounterActor:
    """Counter actor used to exercise locked and unlocked updates."""

    def __init__(self):
        self.count = 0

    def read(self) -> int:
        return self.count

    def write(self, value: int):
        self.count = value


@ray.remote
def worker_no_lock(counter: ray.actor.ActorHandle, work_id: int):
    """Increment without locking, allowing concurrent updates to collide."""
    curr = ray.get(counter.read.remote())
    time.sleep(0.1)
    ray.get(counter.write.remote(curr + 1))
    return f"worker_no_lock-{work_id} done"


@ray.remote
def worker_with_lock(counter: ray.actor.ActorHandle, work_id: int):
    """Increment under `RayLock` to serialize updates."""
    with RayLock(counter):
        curr = ray.get(counter.read.remote())
        time.sleep(0.1)
        ray.get(counter.write.remote(curr + 1))
    return f"worker_with_lock-{work_id} done"


NUM_WORKERS = 5


def test_no_lock_loses_updates(ray_cluster):
    """Allow concurrent unlocked updates to collide."""
    counter1 = CounterActor.remote()

    futures_no_lock = [worker_no_lock.remote(counter1, i) for i in range(NUM_WORKERS)]
    ray.get(futures_no_lock)

    final_no_lock = ray.get(counter1.read.remote())
    assert final_no_lock <= NUM_WORKERS, f"Expected final_no_lock <= {NUM_WORKERS}, got {final_no_lock}"


def test_lock_serializes_counter(ray_cluster):
    """Serialize locked updates to the worker count."""
    counter2 = CounterActor.remote()

    futures_with_lock = [worker_with_lock.remote(counter2, i) for i in range(NUM_WORKERS)]
    ray.get(futures_with_lock)

    final_with_lock = ray.get(counter2.read.remote())
    assert final_with_lock == NUM_WORKERS, f"Expected count={NUM_WORKERS} with lock, got {final_with_lock}"
