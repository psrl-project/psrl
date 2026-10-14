"""CommandHandlerMixin: periodic background loop for ABORT/SYNC/SLEEP/WAKE_UP commands."""

from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np

from psrl.utils.server.command import Command, CommandType
from psrl.workers.gen.smg_adapter import ReplicaRoutingStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class CommandHandlerMixin:
    """Provides ``_command_handler_loop`` for RolloutCoordinator.

    Expects ``self`` to carry: ``command_queue``, ``stop_command_handler``,
    ``server_handles`` and the
    ``_set_replica_routing_status`` / ``_do_sleep_instance`` / ``_do_wake_up_instance``
    methods provided by ``CoordinatorBase`` / ``RolloutCoordinator``.
    """

    @staticmethod
    def _replica_ids_from_instance_ids(instance_ids: list[tuple[str, int]]) -> list[str]:
        """Return unique replica IDs in deterministic order."""
        return sorted({instance_id[0] for instance_id in instance_ids})

    async def execute_request_migrations(self, moves: list[dict]) -> None:
        """Settle each simulation selected redispatch before releasing the next request."""
        timeout = float(self.config.psrl.deployment.elastic.get("request_migration_timeout_s", 60.0))
        for move in moves:
            engine_request_id = str(move["request_id"])
            source = move["source_instance_id"]
            destination = move["destination_instance_id"]
            async with await self._acquire_replica_operations([source[0], destination[0]]):
                request_id = await self.server_handles[source[0]].get_external_request_id.remote(engine_request_id)
                if request_id is None:
                    continue
                await self._gateway_post_json(
                    "/routing_loop/migration",
                    {
                        "request_id": request_id,
                        "destination": list(destination),
                    },
                )
                try:
                    aborted = await self.server_handles[source[0]].abort_requests.remote([engine_request_id])
                    if not aborted:
                        continue
                    deadline = time.monotonic() + timeout
                    while True:
                        status = await self._gateway_get_json(
                            "/routing_loop/migration",
                            params={"request_id": request_id},
                        )
                        if status and status["dispatched"]:
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Request migration did not settle: request_id={request_id!r}.")
                        await asyncio.sleep(0.01)
                finally:
                    await self._gateway_post_json("/routing_loop/migration/finish", {"request_id": request_id})

    async def _sleep_replicas(self, instance_ids: list[tuple[str, int]]) -> None:
        """Remove selected replicas from routing before sleeping them."""
        replica_ids = self._replica_ids_from_instance_ids(instance_ids)
        async with await self._acquire_replica_operations(replica_ids):
            await self._sleep_replicas_locked(replica_ids)

    async def _sleep_replicas_locked(self, replica_ids: list[str]) -> None:
        """Sleep replicas while their lifecycle reservations are held."""
        sleeping_flags = await asyncio.gather(
            *[self.server_handles[replica_id].is_sleeping.remote() for replica_id in replica_ids]
        )
        awake_replica_ids = [
            replica_id
            for replica_id, is_sleeping in zip(replica_ids, sleeping_flags, strict=True)
            if not is_sleeping and replica_id not in self.sleeping_replica_ids
        ]
        already_sleeping = [
            replica_id
            for replica_id, is_sleeping in zip(replica_ids, sleeping_flags, strict=True)
            if is_sleeping or replica_id in self.sleeping_replica_ids
        ]
        if already_sleeping:
            psrl_logger.info(f"Skipping sleep for already sleeping replicas: {already_sleeping!r}.")
            self.sleeping_replica_ids.update(already_sleeping)
        await self._pause_replica_dispatch(replica_ids)
        if not awake_replica_ids:
            return

        results = await asyncio.gather(
            *[self._do_sleep_instance(replica_id) for replica_id in awake_replica_ids],
            return_exceptions=True,
        )
        failures = [
            (replica_id, result)
            for replica_id, result in zip(awake_replica_ids, results, strict=True)
            if isinstance(result, BaseException)
        ]
        self.sleeping_replica_ids.update(
            replica_id
            for replica_id, result in zip(awake_replica_ids, results, strict=True)
            if not isinstance(result, BaseException)
        )
        if failures:
            failed_replica_ids = [replica_id for replica_id, _ in failures]
            psrl_logger.error(f"Keeping failed sleep replicas paused: replicas={failed_replica_ids!r}.")
            failure_details = [(replica_id, repr(error)) for replica_id, error in failures]
            raise RuntimeError(f"Failed to sleep replicas: {failure_details}.") from failures[0][1]

    async def _wake_up_replicas(self, instance_ids: list[tuple[str, int]]) -> None:
        """Wake selected replicas before making them eligible for routing."""
        replica_ids = self._replica_ids_from_instance_ids(instance_ids)
        async with await self._acquire_replica_operations(replica_ids):
            await self._wake_up_replicas_locked(replica_ids)

    async def _wake_up_replicas_locked(self, replica_ids: list[str]) -> None:
        """Restore replicas while their lifecycle reservations are held."""
        sleeping = await asyncio.gather(
            *(self.server_handles[replica_id].is_sleeping.remote() for replica_id in replica_ids)
        )
        wake_ids = [
            replica_id
            for replica_id, asleep in zip(replica_ids, sleeping, strict=True)
            if asleep or replica_id in self.sleeping_replica_ids
        ]
        if wake_ids:
            await self._pause_replica_dispatch(wake_ids)
        results = await asyncio.gather(
            *(self._do_wake_up_instance(replica_id) for replica_id in wake_ids), return_exceptions=True
        )
        failures = [
            (replica_id, result)
            for replica_id, result in zip(wake_ids, results, strict=True)
            if isinstance(result, BaseException)
        ]
        failed_ids = {replica_id for replica_id, _ in failures}
        ready_ids = [replica_id for replica_id in replica_ids if replica_id not in failed_ids]
        if ready_ids:
            await self._set_replica_routing_status(ready_ids, ReplicaRoutingStatus.ACTIVE)
            self.sleeping_replica_ids.difference_update(ready_ids)
        if failures:
            failure_details = [(replica_id, repr(error)) for replica_id, error in failures]
            raise RuntimeError(f"Failed to wake replicas: {failure_details!r}.") from failures[0][1]

    async def _command_handler_loop(self):
        """
        Background loop for processing commands from the command queue.

        This method continuously processes different types of commands:
        - ABORT: Interrupt specific requests on instances
        - SYNC: Interrupt instance, pull new model weights, and resume generation

        The loop runs until stop_command_handler is set to True.
        """
        while not self.stop_command_handler:
            if self.command_queue.empty():
                await asyncio.sleep(0.01)
                continue
            command = self.command_queue.get_nowait()
            command_id = command.get_kwargs()["id"]
            self._start_command(command_id)
            try:
                await self._execute_queued_command(command)
            except Exception as error:
                self._fail_command(command_id, error)
                psrl_logger.exception(f"Coordinator command {command_id} failed: type={command.type.name!r}.")
        psrl_logger.info("Background command handler of rollout coordinator has finished.")

    async def _execute_queued_command(self, command: Command) -> None:
        """
        Execute one command and notify its original waiter.
        """
        assert isinstance(command, Command), f"Expected Command type, got {type(command)!r}."

        # Unpack command attributes
        command_type = command.type
        command_id = command.get_kwargs()["id"]
        command_args = command.get_args()
        psrl_logger.debug(
            f"Receive command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}"
        )

        result = None
        # Process the command based on its type
        if command_type == CommandType.ABORT:
            instance_to_uids = command_args.get("instance_to_uids", None)
            instance_ids = command_args.get("instance_ids", None)
            if instance_to_uids is None and instance_ids is None:
                raise ValueError("ABORT command must contain 'instance_to_uids' or 'instance_ids' in args.")

            psrl_logger.info(
                f"Received ABORT command: instance_to_uids={instance_to_uids!r}, instance_ids={instance_ids!r}."
            )
            futures = []

            if instance_to_uids is not None:
                replica_to_uids = {}
                for instance_id, uids in instance_to_uids.items():
                    if not uids:
                        continue
                    replica_id = instance_id[0]
                    if not isinstance(uids, (list, set)):
                        uids = [uids]
                    abort_requests = set(uids)  # Ensure uniqueness
                    replica_to_uids.setdefault(replica_id, set()).update(abort_requests)
                for replica_id, abort_requests in replica_to_uids.items():
                    abort_requests = [str(req_id) for req_id in abort_requests]
                    futures.append(self.server_handles[replica_id].abort_requests.remote(abort_requests))
            if instance_ids is not None:
                for instance_id in instance_ids:
                    # TODO(linsh): Merge DP ranks of the same replica to reduce RPC calls.
                    replica_id, data_parallel_rank = instance_id
                    futures.append(self.server_handles[replica_id].abort_all_requests.remote())

            if not futures:
                interrupted_request_num = 0
            else:
                interrupted_request_status = await asyncio.gather(*futures)
                interrupted_request_nums = [
                    status if isinstance(status, int) else status.get("aborted_count", 0)
                    for status in interrupted_request_status
                ]
                interrupted_request_num = np.sum(interrupted_request_nums)

            result = interrupted_request_num
            psrl_logger.info(f"Completed ABORT command: interrupted={interrupted_request_num}.")
            # Post process the command result
            self._complete_command(command_id, result)

        elif command_type == CommandType.SYNC:
            # Interrupt the instance, pull the model weights from PS and resume generation.
            instance_ids = command_args.get("instance_ids", None)
            curr_ps_model_version = command_args.get("curr_ps_model_version", None)
            if instance_ids is None or curr_ps_model_version is None:
                raise ValueError("SYNC command must contain 'instance_ids' and 'curr_ps_model_version' in args.")
            if not isinstance(instance_ids, list):
                instance_ids = [instance_ids]
            instance_ids = await self._get_awake_instance_ids(instance_ids)
            if not instance_ids:
                self._complete_command(command_id, None)
                return
            replica_ids = self._replica_ids_from_instance_ids(instance_ids)
            async with await self._acquire_replica_operations(replica_ids):
                sync_instance_ids = await self._get_awake_instance_ids(instance_ids)
                if not sync_instance_ids:
                    self._complete_command(command_id, None)
                else:
                    psrl_logger.info(
                        f"Received SYNC command: instance_ids={sync_instance_ids!r}, "
                        f"ps_model_version={curr_ps_model_version}."
                    )
                    # Sync with PS (interrupt, pull model, and resume generation)
                    sync_futures = []

                    for replica_id in self._replica_ids_from_instance_ids(sync_instance_ids):
                        sync_future = self.server_handles[replica_id].sync_with_ps.remote(
                            curr_ps_model_version,
                            pause_generation=True,
                        )
                        sync_futures.append(sync_future)

                    results = await asyncio.gather(*sync_futures, return_exceptions=True)
                    failures = [result for result in results if isinstance(result, BaseException)]
                    if failures:
                        raise RuntimeError(f"Replica sync commands failed: {failures!r}.") from failures[0]
                    self._complete_command(command_id, None)
        elif command_type == CommandType.SLEEP:
            instance_ids = command_args.get("instance_ids", None)
            if instance_ids is None:
                raise ValueError("SLEEP command must contain 'instance_ids' in args.")
            if not isinstance(instance_ids, list):
                instance_ids = [instance_ids]
            await self._sleep_replicas(instance_ids)
            self._complete_command(command_id, True)
        elif command_type == CommandType.WAKE_UP:
            instance_ids = command_args.get("instance_ids", None)
            if instance_ids is None:
                raise ValueError("WAKE_UP command must contain 'instance_ids' in args.")
            if not isinstance(instance_ids, list):
                instance_ids = [instance_ids]
            replica_ids = self._replica_ids_from_instance_ids(instance_ids)
            psrl_logger.info(f"Received WAKE_UP command for replicas {replica_ids!r}.")

            await self._wake_up_replicas(instance_ids)
            self._complete_command(command_id, True)
        else:
            raise ValueError(f"Unknown command type: {command_type}")
