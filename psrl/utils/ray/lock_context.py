import asyncio
import time
from collections import deque

import ray


def add_lock(cls):
    """
    A general class decorator that injects _locked, _waiters, and acquire/release methods into any class.
    """
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


def add_busy_polling_lock(cls):
    """
    A general class decorator that injects _locked and synchronous acquire/release methods into any class.
    Since Ray actor calls are serialized, we can use a simple attribute without threading primitives.
    """
    original_init = getattr(cls, "__init__", None)

    def __init__(self, *args, **kwargs):
        if original_init is not None:
            original_init(self, *args, **kwargs)
        else:
            super(cls, self).__init__(*args, **kwargs)

        self._locked = False

    def acquire(self):
        """Acquire the lock synchronously. Returns True if acquired, False if already locked."""
        if self._locked:
            return False
        self._locked = True
        return True

    def release(self):
        """Release the lock synchronously."""
        assert self._locked, "Lock is not locked"
        self._locked = False

    cls.__init__ = __init__
    cls.acquire = acquire
    cls.release = release
    return cls


def exclusive_push_model_context(ps_manager):
    """
    A context manager that ensures exclusive access during model push.
    Ensures no other push or pull operations are in progress when entering.
    """
    return _ExclusivePushModelContext(ps_manager)


def shared_pull_model_context(ps_manager):
    """
    A context manager that ensures shared access during model pull.
    Allows multiple pull operations to run concurrently, but blocks if a push is in progress.
    """
    return _SharedPullModelContext(ps_manager)


def exclusive_push_model_context_async(ps_manager, poll_interval=0.01):
    """
    An async context manager that ensures exclusive access during model push.
    Ensures no other push or pull operations are in progress when entering.
    """
    return _AsyncExclusivePushModelContext(ps_manager, poll_interval)


def shared_pull_model_context_async(ps_manager, poll_interval=0.01):
    """
    An async context manager that ensures shared access during model pull.
    Allows multiple pull operations to run concurrently, but blocks if a push is in progress.
    """
    return _AsyncSharedPullModelContext(ps_manager, poll_interval)


class RayLock:
    """Synchronous context manager around a LockActor."""

    def __init__(self, lock_actor):
        self._lock = lock_actor

    def acquire(self):
        """Acquire the lock, blocking until it is available."""
        ray.get(self._lock.acquire.remote())

    def release(self):
        """Release the lock."""
        ray.get(self._lock.release.remote())

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc, tb):
        self.release()


class AsyncRayLock:
    """Asynchronous context manager around a LockActor."""

    def __init__(self, lock_actor):
        self._lock = lock_actor

    async def acquire(self):
        """Acquire the lock, blocking until it is available."""
        await self._lock.acquire.remote()

    async def release(self):
        """Release the lock."""
        await self._lock.release.remote()

    async def __aenter__(self):
        await self.acquire()

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()


class BusyPollingRayLock:
    """
    Synchronous context manager around a LockActor using busy polling.
    Uses busy polling with a specified interval to check if the lock is available.
    """

    def __init__(self, lock_actor, poll_interval=0.01):
        """
        Initialize the BusyPollingRayLock.

        Args:
            lock_actor: The actor handle with lock methods (acquire, release).
                The actor should have acquire() method to check lock state.
            poll_interval: The interval (in seconds) between polling checks.
                Default is 0.01 seconds.
        """
        self._lock = lock_actor
        self._poll_interval = poll_interval

    def acquire(self):
        """
        Acquire the lock using busy polling, blocking until it is available.
        Continuously polls the acquire() method at the specified interval.
        """
        while True:
            got_locked = ray.get(self._lock.acquire.remote())
            if got_locked:
                return
            time.sleep(self._poll_interval)

    def release(self):
        """Release the lock."""
        ray.get(self._lock.release.remote())

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


class AsyncBusyPollingRayLock:
    """
    Asynchronous context manager around a LockActor using busy polling.
    Uses busy polling with a specified interval to check if the lock is available.
    """

    def __init__(self, lock_actor, poll_interval=0.01):
        """
        Initialize the AsyncBusyPollingRayLock.

        Args:
            lock_actor: The actor handle with lock methods (acquire, release).
                The actor should have acquire() method to check lock state.
            poll_interval: The interval (in seconds) between polling checks.
                Default is 0.01 seconds.
        """
        self._lock = lock_actor
        self._poll_interval = poll_interval

    async def acquire(self):
        """
        Acquire the lock using busy polling, blocking until it is available.
        Continuously polls the acquire() method at the specified interval.
        """
        while True:
            got_locked = await self._lock.acquire.remote()
            if got_locked:
                return
            await asyncio.sleep(self._poll_interval)

    async def release(self):
        """Release the lock."""
        await self._lock.release.remote()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release()


class _ExclusivePushModelContext:
    """
    Context manager for exclusive push model operations.
    Ensures no other push or pull operations are in progress when entering.
    """

    def __init__(self, ps_manager, poll_interval=0.01):
        """
        Initialize the exclusive push model context.

        Args:
            ps_manager: The PSManager instance
            poll_interval: The interval (in seconds) between polling checks. Default is 0.01 seconds.
        """
        self.ps_manager = ps_manager
        self.poll_interval = poll_interval

    def __enter__(self):
        """Enter the context, acquiring exclusive lock."""
        # Acquire only when neither a push nor a pull is active.
        while True:
            can_acquire = ray.get(self.ps_manager._try_acquire_exclusive_push_lock.remote())
            if can_acquire:
                return self
            time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, exc_tb):
        """Exit the context, releasing exclusive lock."""
        ray.get(self.ps_manager._release_exclusive_push_lock.remote())
        return False


class _SharedPullModelContext:
    """
    Context manager for shared pull model operations.
    Allows multiple pull operations to run concurrently, but blocks if a push is in progress.
    """

    def __init__(self, ps_manager, poll_interval=0.01):
        """
        Initialize the shared pull model context.

        Args:
            ps_manager: The PSManager instance
            poll_interval: The interval (in seconds) between polling checks. Default is 0.01 seconds.
        """
        self.ps_manager = ps_manager
        self.poll_interval = poll_interval

    def __enter__(self):
        """Enter the context, acquiring shared lock."""
        while True:
            can_acquire = ray.get(self.ps_manager._try_acquire_shared_pull_lock.remote())
            if can_acquire:
                return self
            time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, exc_tb):
        """Exit the context, releasing shared lock."""
        ray.get(self.ps_manager._release_shared_pull_lock.remote())
        return False


class _AsyncExclusivePushModelContext:
    """
    Async context manager for exclusive push model operations.
    Ensures no other push or pull operations are in progress when entering.
    """

    def __init__(self, ps_manager, poll_interval=0.01):
        """
        Initialize the async exclusive push model context.

        Args:
            ps_manager: The PSManager instance
            poll_interval: The interval (in seconds) between polling checks. Default is 0.01 seconds.
        """
        self.ps_manager = ps_manager
        self.poll_interval = poll_interval

    async def __aenter__(self):
        """Enter the context, acquiring exclusive lock."""
        # Acquire only when neither a push nor a pull is active.
        while True:
            can_acquire = await self.ps_manager._try_acquire_exclusive_push_lock.remote()
            if can_acquire:
                return self
            await asyncio.sleep(self.poll_interval)

    async def __aexit__(self, exc_type, exc, exc_tb):
        """Exit the context, releasing exclusive lock."""
        await self.ps_manager._release_exclusive_push_lock.remote()
        return False


class _AsyncSharedPullModelContext:
    """
    Async context manager for shared pull model operations.
    Allows multiple pull operations to run concurrently, but blocks if a push is in progress.
    """

    def __init__(self, ps_manager, poll_interval=0.01):
        """
        Initialize the async shared pull model context.

        Args:
            ps_manager: The PSManager instance
            poll_interval: The interval (in seconds) between polling checks. Default is 0.01 seconds.
        """
        self.ps_manager = ps_manager
        self.poll_interval = poll_interval

    async def __aenter__(self):
        """Enter the context, acquiring shared lock."""
        while True:
            can_acquire = await self.ps_manager._try_acquire_shared_pull_lock.remote()
            if can_acquire:
                return self
            await asyncio.sleep(self.poll_interval)

    async def __aexit__(self, exc_type, exc, exc_tb):
        """Exit the context, releasing shared lock."""
        await self.ps_manager._release_shared_pull_lock.remote()
        return False
