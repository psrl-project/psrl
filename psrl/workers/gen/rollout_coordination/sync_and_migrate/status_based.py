"""StatusBasedSyncMixin: status-based sync-and-migrate background loop."""

from __future__ import annotations

import asyncio
import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class StatusBasedSyncMixin:
    """Provides ``_status_based_sync_and_migrate_loop`` for RolloutCoordinator.

    Expects ``self`` to carry: ``stop_sync_and_migrate``, ``config``,
    ``instance_to_engine_status``, ``tag_to_replica_ids``,
    ``instance_to_model_version``,
    ``ps_model_version``, ``ready_buffers``, ``replica_sync_tasks``, and the ``_get_awake_instance_ids`` /
    ``sync_with_ps`` / ``check_should_sync`` / ``check_and_migrate`` / ``_is_selecting`` helpers.
    """

    async def _status_based_sync_and_migrate_loop(self):
        """
        Background loop to collect engine status and decide whether to synchronize with PS based on the engine status.

        This method:
        1. Analyze the instance status (engine waiting & running request counts, etc.).
        2. Decide whether to synchronize with PS for each instance.
        """
        psrl_logger.info("Starting status based model synchronization and rollout migration loop.")

        while not self.stop_sync_and_migrate:
            # Sleep for a period of time and analyze the instance status
            await asyncio.sleep(
                self.config.psrl.rollout_coordination.sync_and_mig_strategy.check_interval_in_ms / 1000
            )

            awake_instance_ids = await self._get_awake_instance_ids(
                [
                    instance_id
                    for instance_id in self.instance_to_engine_status
                    if instance_id[0] in self.tag_to_replica_ids["rollout"]
                ]
            )
            have_syncing_instance = bool(self.replica_sync_tasks)
            sync_instance_ids = []
            for instance_id in awake_instance_ids:
                engine_stats = self.instance_to_engine_status[instance_id]
                if (
                    instance_id[0] in self.replica_sync_tasks
                    or engine_stats.model_version != self.instance_to_model_version[instance_id]
                ):
                    have_syncing_instance = True
                    continue
                # We do not synchronize with PS if the router is currently selecting workers.
                if await self._is_selecting():
                    # psrl_logger.info(f"Skipping synchronization with PS for instance
                    # {instance_id} because the router is currently selecting workers")
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) >= self.ps_model_version:
                    continue
                # Check whether current instance workload is empty (forbid partial rollout)
                # or satisfies the partial rollout policy
                if self.config.psrl.rollout_coordination.partial_rollout.enable:
                    if not (await self.check_should_sync(instance_id)):
                        continue
                else:
                    if engine_stats.get_waiting_and_running_queue_size() > 0:
                        continue
                # Check whether the training side can seamlessly continue to train after the synchronization
                if (
                    self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync.seamless_train_version
                    >= self.ps_model_version
                ):
                    if self.ps_model_version not in self.ready_buffers:
                        continue
                # Add the instance to the sync list
                sync_instance_ids.append(instance_id)
                """
                # NOTE(lhy): currently, we only synchronize with PS for one instance at a time
                # But the model pulling time can be overlapped
                break
                """

            if sync_instance_ids:
                await self.sync_with_ps(sync_instance_ids)
            elif not have_syncing_instance and self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.enable:
                # No instance is syncing with PS, check if migration is needed
                await self.check_and_migrate()

        psrl_logger.info("Status based model synchronization and rollout migration loop stopped.")
