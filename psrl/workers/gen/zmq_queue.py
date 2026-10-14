import asyncio
import logging
import os

import zmq

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class ZMQPushQueue:
    def __init__(self, endpoint: str, sndhwm: int = 10000, drop_on_full: bool = True):
        self.endpoint = endpoint
        self._drop_on_full = drop_on_full
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.SNDHWM, sndhwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)

    def put_nowait(self, item):
        if self._drop_on_full:
            try:
                self._socket.send_pyobj(item, flags=zmq.NOBLOCK)
            except zmq.Again:
                # drop latest status snapshot when sender buffer is full
                pass
            return

        # Reliable mode: block until sent (no silent drop).
        self._socket.send_pyobj(item)

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


class ZMQPullQueue:
    def __init__(self, endpoint: str, rcvhwm: int = 10000):
        self.endpoint = endpoint
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PULL)
        self._socket.setsockopt(zmq.RCVHWM, rcvhwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(endpoint)
        self._fd = self._socket.getsockopt(zmq.FD)

    def _has_pending_data(self) -> bool:
        return self._socket.poll(timeout=0) != 0

    def _recv_nowait(self):
        if not self._has_pending_data():
            return False, None
        try:
            return True, self._socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            # Defensive fallback: readiness may change between poll and recv.
            return False, None

    async def _wait_readable(self):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        def _on_readable():
            if not fut.done():
                fut.set_result(None)

        loop.add_reader(self._fd, _on_readable)
        try:
            # Fast-path check in case data is already pending.
            if self._has_pending_data() and not fut.done():
                fut.set_result(None)
            await fut
        finally:
            loop.remove_reader(self._fd)

    async def get_async(self, block: bool = True, timeout: float | None = None):
        ok, msg = self._recv_nowait()
        if ok:
            return msg
        if not block:
            return None

        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout

        if timeout is None:
            while True:
                await self._wait_readable()
                ok, msg = self._recv_nowait()
                if ok:
                    return msg

        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"Timeout waiting for message on {self.endpoint}")
            try:
                await asyncio.wait_for(self._wait_readable(), timeout=remaining)
            except asyncio.TimeoutError as e:
                raise TimeoutError(f"Timeout waiting for message on {self.endpoint}") from e
            ok, msg = self._recv_nowait()
            if ok:
                return msg

    def empty(self) -> bool:
        return self._socket.poll(timeout=0) == 0

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
