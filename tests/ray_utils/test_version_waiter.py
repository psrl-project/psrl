import asyncio
import time

import ray


@ray.remote
class VersionedActor:
    def __init__(self):
        self.version = 0
        self._version_waiters = {}

    async def wait_for_version(self, target_version: int):
        if self.version == target_version:
            return

        fut = asyncio.get_event_loop().create_future()
        if target_version not in self._version_waiters:
            self._version_waiters[target_version] = []
        self._version_waiters[target_version].append(fut)
        await fut

    def set_version(self, new_version: int):
        self.version = new_version
        if new_version in self._version_waiters:
            for fut in self._version_waiters[new_version]:
                if not fut.done():
                    fut.set_result(None)
            del self._version_waiters[new_version]

    def get_version(self) -> int:
        return self.version


@ray.remote
def version_waiter(actor_handle, target_ver: int, worker_id: int):
    ray.get(actor_handle.wait_for_version.remote(target_ver))
    return f"waiter {worker_id} unblocked at version {target_ver}"


@ray.remote
def version_setter(actor_handle, new_version: int, delay_s: float):
    time.sleep(delay_s)
    ray.get(actor_handle.set_version.remote(new_version))
    return f"setter set version to {new_version}"


def test_version_waiter_already_satisfied(ray_cluster):
    """Waiter for version 0 unblocks immediately since initial version is 0."""
    versioned = VersionedActor.remote()
    waiter = version_waiter.remote(versioned, 0, worker_id=3)
    result = ray.get(waiter)
    assert "waiter 3 unblocked at version 0" in result


def test_version_waiter_blocks_until_set(ray_cluster):
    """Waiters for version 1 and 2 unblock only after setter calls set_version."""
    versioned = VersionedActor.remote()

    waiter_futures = [
        version_waiter.remote(versioned, 1, worker_id=1),
        version_waiter.remote(versioned, 2, worker_id=2),
    ]
    # Let both waiters register before any version is set so the test exercises
    # the blocking path rather than the "already satisfied" shortcut.
    time.sleep(1.0)

    # Apply the versions in order, waiting for each setter, so the final version
    # is deterministic regardless of cluster scheduling jitter.
    ray.get(version_setter.remote(versioned, 1, delay_s=0.0))
    ray.get(version_setter.remote(versioned, 2, delay_s=0.0))

    waiter_results = ray.get(waiter_futures)

    assert any("waiter 1 unblocked at version 1" in r for r in waiter_results)
    assert any("waiter 2 unblocked at version 2" in r for r in waiter_results)

    final_version = ray.get(versioned.get_version.remote())
    assert final_version == 2
