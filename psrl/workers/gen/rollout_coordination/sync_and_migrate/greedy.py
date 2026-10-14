"""GreedySyncMixin: greedy sync-and-migrate background loop."""

from __future__ import annotations

import asyncio
import logging
import os

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class GreedySyncMixin:
    """Provides ``_greedy_sync_and_migrate_loop`` for RolloutCoordinator.

    Expects ``self`` to carry: ``stop_sync_and_migrate``, ``config``,
    ``instance_ids``, ``tag_to_replica_ids``, ``instance_to_model_version``,
    ``ps_model_version``,
    ``ready_buffers``, ``replica_sync_tasks``, and the ``_get_awake_instance_ids`` /
    ``sync_with_ps`` / ``check_no_activate_tasks`` / ``check_and_migrate`` helpers.
    """

    async def _greedy_sync_and_migrate_loop(self):
        """
        Background loop to synchronize with PS based on the greedy algorithm.

        This method:
        1. Greedily synchronize with PS for rollout that lags behind PS version.
        2. Check whether the instance has no active tasks if forbid partial rollout.
        3. Check if any instance is starving and do migration if necessary.
        """
        psrl_logger.info("Starting greedy model synchronization and rollout migration loop.")

        while not self.stop_sync_and_migrate:
            # Sleep for a period of time
            await asyncio.sleep(
                self.config.psrl.rollout_coordination.sync_and_mig_strategy.check_interval_in_ms / 1000
            )

            awake_instance_ids = await self._get_awake_instance_ids([
                instance_id for instance_id in self.instance_ids
                if instance_id[0] in self.tag_to_replica_ids["rollout"]
            ])
            have_syncing_instance = bool(self.replica_sync_tasks)
            sync_instance_ids = []
            for instance_id in awake_instance_ids:
                if instance_id[0] in self.replica_sync_tasks:
                    have_syncing_instance = True
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) >= self.ps_model_version:
                    continue
                # Check whether current instance workload is empty if forbid partial rollout
                if not self.config.psrl.rollout_coordination.partial_rollout.enable:
                    if not await self.check_no_activate_tasks(instance_id):
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

            if sync_instance_ids:
                await self.sync_with_ps(sync_instance_ids)
            elif not have_syncing_instance and self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.enable:
                # No instance is syncing with PS, check if migration is needed
                await self.check_and_migrate()

        psrl_logger.info("Greedy model synchronization and rollout migration loop stopped.")
