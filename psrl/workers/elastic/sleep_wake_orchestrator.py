"""
Phased sleep/wake orchestrator for colocated deployment modes.

The SleepWakeOrchestrator drives time-multiplexed sleep/wake of rollout and
reward-model roles on a shared pool, used by deployment modes `colocated`
(mode 2) and `rollout_rm_colocated` (mode 3).

It exposes a small phase state machine (ROLLOUT | REWARD | TRAIN) and gates the
existing async pipeline (AgentLoopManager rollout dispatch, RewardManager rm
dispatch) via ``wait_phase``. Role-level SLEEP/WAKE_UP is issued through the
existing RolloutCoordinator / RewardModelCoordinator ``exec_command`` API with
``instance_ids=[-1]`` (all instances of the role).

Actor (trainer) sleep/wake for mode 2 is performed by the trainer driver
(``ray_trainer``) directly via NIXL, since the actor worker group is owned by
the driver process. the orchestrator only manages rollout/rm.
"""

import asyncio
import enum
import logging
import os

import ray

from psrl.utils.server.command import Command, CommandType

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class OrchestratorPhase(str, enum.Enum):
    ROLLOUT = "rollout"
    REWARD = "reward"
    TRAIN = "train"


@ray.remote
class SleepWakeOrchestrator:
    """
    Phased sleep/wake orchestrator for colocated deployment modes.

    Args:
        mode: deployment mode string ("colocated" or "rollout_rm_colocated").
        rollout_coordinator: Ray actor handle for the RolloutCoordinator.
        reward_model_coordinators: dict mapping reward_model_name -> RM
            coordinator Ray actor handle.
        rollout_model_name: model name string for the rollout role.
    """

    def __init__(
        self,
        mode: str,
        rollout_coordinator,
        reward_model_coordinators: dict,
        rollout_model_name: str,
    ):
        self.mode = str(mode)
        self.rollout_coordinator = rollout_coordinator
        self.reward_model_coordinators = dict(reward_model_coordinators or {})
        self.rollout_model_name = rollout_model_name

        self._phase: OrchestratorPhase = OrchestratorPhase.ROLLOUT
        # The trainer driver sleeps both roles before the orchestrator starts.
        self._rollout_awake: bool = False
        self._rm_awake: bool = False
        # Futures for wait_phase callers, keyed by the phase they are waiting for.
        self._phase_waiters: dict[OrchestratorPhase, list[asyncio.Future]] = {p: [] for p in OrchestratorPhase}
        # Serialize set_phase transitions to avoid overlapping sleep/wake waves.
        self._transition_lock = asyncio.Lock()
        self._initialized = False

    async def _sleep_rollout(self):
        if not self._rollout_awake or self.rollout_coordinator is None:
            return
        result = await self.rollout_coordinator.exec_command.remote(
            Command(type=CommandType.SLEEP, instance_ids=[-1]),
            blocking=True,
        )
        if result is not True:
            raise RuntimeError(f"Failed to sleep rollout instances: {result!r}")
        self._rollout_awake = False

    async def _wake_rollout(self):
        if self._rollout_awake or self.rollout_coordinator is None:
            return
        result = await self.rollout_coordinator.exec_command.remote(
            Command(type=CommandType.WAKE_UP, instance_ids=[-1]),
            blocking=True,
        )
        if result is not True:
            raise RuntimeError(f"Failed to wake rollout instances: {result!r}")
        self._rollout_awake = True

    async def _sleep_reward(self):
        if not self._rm_awake:
            return
        slept = []
        for rm_name, coord in self.reward_model_coordinators.items():
            result = await coord.exec_command.remote(
                Command(type=CommandType.SLEEP, instance_ids=[-1]),
                blocking=True,
            )
            if result is not True:
                for _, slept_coord in reversed(slept):
                    await slept_coord.exec_command.remote(
                        Command(type=CommandType.WAKE_UP, instance_ids=[-1]), blocking=True
                    )
                raise RuntimeError(f"Failed to sleep reward model {rm_name}: {result!r}")
            slept.append((rm_name, coord))
        self._rm_awake = False

    async def _wake_reward(self):
        if self._rm_awake:
            return
        woken = []
        for rm_name, coord in self.reward_model_coordinators.items():
            result = await coord.exec_command.remote(
                Command(type=CommandType.WAKE_UP, instance_ids=[-1]),
                blocking=True,
            )
            if result is not True:
                for _, woken_coord in reversed(woken):
                    await woken_coord.exec_command.remote(
                        Command(type=CommandType.SLEEP, instance_ids=[-1]), blocking=True
                    )
                raise RuntimeError(f"Failed to wake reward model {rm_name}: {result!r}")
            woken.append((rm_name, coord))
        self._rm_awake = True

    async def _apply_phase(self, phase: OrchestratorPhase):
        """
        Issue sleep/wake commands to realize ``phase`` (diff-based, idempotent).

        Ordering: always sleep the outgoing role before waking the incoming role
        so that two roles never hold the shared pool's GPUs simultaneously.
        """
        if phase == OrchestratorPhase.ROLLOUT:
            # Wake rollout, sleep rm: sleep rm first, then wake rollout.
            await self._sleep_reward()
            await self._wake_rollout()
        elif phase == OrchestratorPhase.REWARD:
            # Sleep rollout, wake rm: sleep rollout first, then wake rm.
            await self._sleep_rollout()
            await self._wake_reward()
        elif phase == OrchestratorPhase.TRAIN:
            # Mode 2 only: both rollout and rm sleep while the actor trains.
            await self._sleep_rollout()
            await self._sleep_reward()

    def _set_phase_and_release(self, phase: OrchestratorPhase):
        self._phase = phase
        waiters = self._phase_waiters.get(phase, [])
        self._phase_waiters[phase] = []
        for fut in waiters:
            if not fut.done():
                fut.set_result(phase.value)

    async def set_phase(self, phase: str) -> str:
        """
        Transition to ``phase`` and perform the corresponding sleep/wake.

        Returns the previous phase value. No-op if already in ``phase``.
        """
        target = OrchestratorPhase(str(phase))
        async with self._transition_lock:
            if self._phase == target:
                return self._phase.value
            previous = self._phase
            psrl_logger.info(
                "SleepWakeOrchestrator: phase %s -> %s (mode=%s)",
                previous.value,
                target.value,
                self.mode,
            )
            await self._apply_phase(target)
            self._set_phase_and_release(target)
            return previous.value

    async def wait_phase(self, phase: str) -> str:
        """
        Block until the orchestrator reaches ``phase``, then return the phase.
        """
        target = OrchestratorPhase(str(phase))
        if self._phase == target:
            return self._phase.value
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._phase_waiters[target].append(fut)
        return await fut

    async def get_phase(self) -> str:
        return self._phase.value

    async def notify_rollout_complete(self) -> None:
        """
        Signal that the current rollout cycle/buffer is fully generated.

        Mode 3 (rollout_rm_colocated): auto-transition ROLLOUT -> REWARD so rm can
        score the completed buffer. Mode 2 (colocated): the trainer driver owns
        phase transitions, so this is a no-op here.
        """
        if self.mode == "rollout_rm_colocated":
            await self.set_phase(OrchestratorPhase.REWARD.value)

    async def notify_reward_complete(self) -> None:
        """
        Signal that reward scoring for the current buffer is done.

        Mode 3: auto-transition REWARD -> ROLLOUT to start the next buffer.
        """
        if self.mode == "rollout_rm_colocated":
            await self.set_phase(OrchestratorPhase.ROLLOUT.value)

    async def start(self) -> None:
        """
        Initialize the orchestrator to the ROLLOUT phase.

        Assumes rollout/rm instances were already put to SLEEP by the trainer
        driver during init (so ``_apply_phase(ROLLOUT)`` only needs to wake
        rollout). We still sleep rm defensively (idempotent) then wake rollout.
        """
        async with self._transition_lock:
            await self._apply_phase(OrchestratorPhase.ROLLOUT)
            self._set_phase_and_release(OrchestratorPhase.ROLLOUT)
            self._initialized = True
            psrl_logger.info("SleepWakeOrchestrator started at phase=rollout (mode=%s).", self.mode)
