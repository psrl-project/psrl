import asyncio
import time

import ray


@ray.remote
class Worker:
    async def compute(self, x: int) -> int:
        """Return `x * x` after an asynchronous delay."""
        print(f"[{time.strftime('%X')}] Worker.compute({x}) start")
        await asyncio.sleep(2)
        print(f"[{time.strftime('%X')}] Worker.compute({x}) end")
        return x * x


@ray.remote
class MixedActor:
    def __init__(self, worker_handle):
        self.worker = worker_handle

    async def async_method(self, x: int) -> int:
        """Await a remote computation without blocking the actor event loop."""
        print(f"[{time.strftime('%X')}] MixedActor.async_method({x}) start")
        result = await self.worker.compute.remote(x)
        print(f"[{time.strftime('%X')}] MixedActor.async_method({x}) got result = {result}")
        return result

    def sync_method(self, x: int) -> int:
        """Block the actor event loop while waiting for a remote computation."""
        print(f"[{time.strftime('%X')}] MixedActor.sync_method({x}) start (blocking ray.get)")
        result = ray.get(self.worker.compute.remote(x))
        print(f"[{time.strftime('%X')}] MixedActor.sync_method({x}) got result = {result}")
        return result


def test_mixed_actor_sync_method(ray_cluster):
    """Call the synchronous actor method."""
    w = Worker.remote()
    a = MixedActor.remote(w)

    print(f"[{time.strftime('%X')}] --> Calling MixedActor.sync_method(3)")
    ref_sync = a.sync_method.remote(3)
    res_sync = ray.get(ref_sync)
    print(f"[{time.strftime('%X')}] <-- MixedActor.sync_method(3) returned: {res_sync}")
    assert res_sync == 9


def test_mixed_actor_async_method(ray_cluster):
    """Call the asynchronous actor method."""
    w = Worker.remote()
    a = MixedActor.remote(w)

    print(f"[{time.strftime('%X')}] --> Calling MixedActor.async_method(5)")
    ref_async = a.async_method.remote(5)
    res_async = ray.get(ref_async)
    print(f"[{time.strftime('%X')}] <-- MixedActor.async_method(5) returned: {res_async}")
    assert res_async == 25


def test_mixed_concurrent_calls(ray_cluster):
    """Issue synchronous and asynchronous actor calls concurrently."""
    w = Worker.remote()
    a = MixedActor.remote(w)

    print(f"[{time.strftime('%X')}] --> Calling MixedActor.async_method(7) and MixedActor.sync_method(9)")
    ref1 = a.sync_method.remote(9)
    ref2 = a.async_method.remote(7)
    res1 = ray.get(ref1)
    print(f"[{time.strftime('%X')}] <-- MixedActor.sync_method(9) returned: {res1}")
    res2 = ray.get(ref2)
    print(f"[{time.strftime('%X')}] <-- MixedActor.async_method(7) returned: {res2}")
    assert res1 == 81  # 9 * 9
    assert res2 == 49  # 7 * 7
