import asyncio
import logging
import os

from psrl.workers.gen.rollout_coordination import RolloutCoordinator
from psrl.workers.gen.smg_adapter import ROUTING_LOOP_STATUS_PATH

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RewardModelCoordinator(RolloutCoordinator):
    """Coordinate static reward model replicas without parameter-server synchronization."""

    def __init__(
        self,
        config,
        rm_config,
        rollout_gateway_url: str,
    ) -> None:
        super().__init__(
            config=config,
            ps_manager=None,
            rollout_gateway_url=rollout_gateway_url,
        )
        self.rm_config = rm_config
        self.reward_model_name = rm_config.reward_model_name
        psrl_logger.info("RewardModelCoordinator initialized for model_name=%s", self.reward_model_name)

    async def start_busy_loop(self) -> None:
        """Start command and optional status-processing tasks."""
        if self.command_handler_task is not None and not self.command_handler_task.done():
            return

        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(lambda f: f.result())

        if self.config.psrl.status_collection.enable:
            self.process_status_queue_task = self.running_loop.create_task(self._process_status_queue())
            self.process_status_queue_task.add_done_callback(lambda f: f.result())
            self.sync_status_to_router_task = self.running_loop.create_task(self._sync_status_to_router())
            self.sync_status_to_router_task.add_done_callback(lambda f: f.result())

        psrl_logger.info("RewardModelCoordinator busy loop started for model_name=%s", self.reward_model_name)

    async def sync_model(self, *args, **kwargs) -> None:
        """Skip weight synchronization because reward model weights are static."""

    async def update_model_version(self, model_version: int, *args, **kwargs) -> None:
        """No model version tracking for static reward models."""

    async def _greedy_sync_and_migrate_loop(self) -> None:
        """No sync/migration loop for static reward models."""

    async def _status_based_sync_and_migrate_loop(self) -> None:
        """No sync/migration loop for static reward models."""

    def _get_sleep_level(self) -> int:
        """Use level-2 sleep so reward-model weights are released from GPU memory."""
        return 2

    async def _do_sleep_instance(self, replica_id: str) -> None:
        """Sleep a reward server without NIXL deregistration."""
        await self.server_handles[replica_id].sleep.remote(
            level=self._get_sleep_level(),
        )

    async def _do_wake_up_instance(self, replica_id: str) -> None:
        """Reward model wake_up: calls server.wake_up() (non-nixl path)."""
        await self.server_handles[replica_id].wake_up.remote()

    async def get_router_backlog_size(self) -> int:
        """
        Return requests waiting in the SMG routing queue.

        Gateway errors propagate so the elastic executor retains the last valid count.
        """
        if not self.rollout_gateway_url:
            return 0
        status = await self._gateway_get_json(ROUTING_LOOP_STATUS_PATH)
        return int(status["pending_request_num"])

    async def get_router_backlog_summary(self, top_t: int | None = None) -> dict[str, int]:
        """
        Return the available aggregate waiting-queue metrics from SMG.

        SMG does not expose queued request token lengths. The summary therefore
        reports an exact request count and zero tokens instead of fabricating load.
        """
        pending = await self.get_router_backlog_size()
        count = pending if top_t is None else min(pending, max(0, int(top_t)))
        return {
            "pending": pending,
            "count": count,
            "total_tokens": 0,
        }
