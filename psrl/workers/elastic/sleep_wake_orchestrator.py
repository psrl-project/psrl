"""Serialize ownership of GPU memory shared by inference and training roles."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from enum import StrEnum
from uuid import uuid4

import ray


class OrchestratorPhase(StrEnum):
    ROLLOUT = "rollout"
    VALIDATE = "validate"
    REWARD = "reward"
    TRAIN = "train"


@asynccontextmanager
async def inference_lease(orchestrator, phase: str, model: str = ""):
    """Hold shared storage until the caller finishes using its current weights."""
    if orchestrator is None:
        yield
        return
    lease_id = uuid4().hex
    admission = asyncio.ensure_future(orchestrator.acquire.remote(lease_id, phase, model))

    async def release():
        try:
            await admission
        except BaseException:
            pass
        await orchestrator.release.remote(lease_id)

    try:
        await asyncio.shield(admission)
        yield
    finally:
        # Admission must settle before cancellation can release its identity.
        await asyncio.shield(release())


@ray.remote
class SleepWakeOrchestrator:
    """Batch concurrent users of one role and fairly drain them before switching."""

    def __init__(self, mode: str, rollout_coordinator, reward_model_coordinators: dict):
        if mode not in ("colocated", "rollout_rm_colocated"):
            raise ValueError(f"Unsupported phased deployment mode: {mode!r}.")
        self.mode = mode
        self.rollout_coordinator = rollout_coordinator
        self.reward_model_coordinators = dict(reward_model_coordinators)
        self._current: tuple[OrchestratorPhase, str] | None = None
        self._active: set[str] = set()
        self._pending: deque = deque()
        self._lock = asyncio.Lock()
        self._failure: BaseException | None = None
        self._initialized = False

    async def start(self) -> None:
        """Require every inference role to release storage before admitting work."""
        async with self._lock:
            await self.rollout_coordinator.sleep.remote("all")
            for coordinator in self.reward_model_coordinators.values():
                await coordinator.sleep.remote()
            self._initialized = True

    async def _apply(self, target: tuple[OrchestratorPhase, str]) -> None:
        if self._current == target:
            return
        if self._current is not None:
            phase, model = self._current
            if phase == OrchestratorPhase.REWARD:
                await self.reward_model_coordinators[model].sleep.remote()
            elif phase != OrchestratorPhase.TRAIN:
                await self.rollout_coordinator.sleep.remote(phase.value)
        phase, model = target
        if phase == OrchestratorPhase.REWARD:
            await self.reward_model_coordinators[model].wake_up.remote()
        elif phase != OrchestratorPhase.TRAIN:
            await self.rollout_coordinator.wake_up.remote(phase.value)
        self._current = target

    async def _advance(self) -> None:
        if self._active or not self._pending:
            return
        target = self._pending[0][1]
        try:
            await self._apply(target)
        except BaseException as error:
            self._failure = error
            while self._pending:
                _, _, future = self._pending.popleft()
                if not future.done():
                    future.set_exception(RuntimeError(f"Shared GPU transition failed: {error!r}."))
            return
        while self._pending and self._pending[0][1] == target:
            lease_id, _, future = self._pending.popleft()
            if not future.cancelled():
                self._active.add(lease_id)
                future.set_result(None)

    async def acquire(self, lease_id: str, phase: str, model: str = "") -> None:
        """Wait for conflicting users without admitting new work ahead of them."""
        target = (OrchestratorPhase(phase), model)
        if target[0] == OrchestratorPhase.REWARD and model not in self.reward_model_coordinators:
            raise ValueError(f"Unknown reward model: {model!r}.")
        async with self._lock:
            if not self._initialized or self._failure is not None:
                raise RuntimeError(f"Shared GPU admission is unavailable: {self._failure!r}.")
            if lease_id in self._active or any(item[0] == lease_id for item in self._pending):
                raise ValueError(f"GPU lease is already registered: {lease_id!r}.")
            if self._current == target and not self._pending:
                self._active.add(lease_id)
                return
            future = asyncio.get_running_loop().create_future()
            self._pending.append((lease_id, target, future))
            await self._advance()
        await future

    async def release(self, lease_id: str) -> None:
        """Release an active or cancelled admission before handing storage onward."""
        async with self._lock:
            self._active.discard(lease_id)
            remaining = deque()
            for item in self._pending:
                if item[0] == lease_id:
                    if not item[2].done():
                        item[2].cancel()
                else:
                    remaining.append(item)
            self._pending = remaining
            await self._advance()

    async def get_phase(self) -> str | None:
        return self._current[0].value if self._current else None
