"""
Execute scoped inference commands with independently observable completion.
"""

from __future__ import annotations

import asyncio
import time

from psrl.utils.server.command import Command, CommandType

from .lifecycle import GatewayClient, ReplicaLifecycle
from .state import ModelState


class CommandService:
    def __init__(self, config, gateway: GatewayClient, lifecycle: ReplicaLifecycle) -> None:
        self.config = config
        self.gateway = gateway
        self.lifecycle = lifecycle

    async def run(self, model: ModelState) -> None:
        commands = model.commands
        while not model.closing.is_set() or not commands.command_queue.empty():
            try:
                command = await asyncio.wait_for(commands.command_queue.get(), timeout=0.1)
            except TimeoutError:
                continue
            command_id = command.get_kwargs()["id"]
            commands._start_command(command_id)
            try:
                result = await self.execute(model, command)
            except Exception as error:
                commands._fail_command(command_id, error)
            else:
                commands._complete_command(command_id, result)

    async def abort(self, model: ModelState, instance_ids=None, instance_to_uids=None) -> int:
        if instance_ids is None and instance_to_uids is None:
            raise ValueError("ABORT requires instance_ids or instance_to_uids.")
        futures = []
        if instance_to_uids is not None:
            grouped = {}
            model.expand_instances(list(instance_to_uids))
            for iid, uids in instance_to_uids.items():
                if uids:
                    grouped.setdefault(iid[0], set()).update(uids if isinstance(uids, (list, set)) else [uids])
            futures.extend(
                model.replicas[rid].server.abort_requests.remote([str(uid) for uid in uids])
                for rid, uids in grouped.items()
            )
        if instance_ids is not None:
            replicas = {iid[0] for iid in model.expand_instances(instance_ids)}
            futures.extend(model.replicas[rid].server.abort_all_requests.remote() for rid in sorted(replicas))
        results = await asyncio.gather(*futures)
        return sum(result if isinstance(result, int) else result.get("aborted_count", 0) for result in results)

    async def execute(self, model: ModelState, command: Command):
        args = command.get_args()
        if command.type == CommandType.ABORT:
            return await self.abort(model, args.get("instance_ids"), args.get("instance_to_uids"))
        instances = args.get("instance_ids")
        if instances is None:
            raise ValueError(f"Command {command.type!r} requires instance_ids.")
        if not isinstance(instances, list):
            instances = [instances]
        if command.type == CommandType.SLEEP:
            return await self.lifecycle.sleep(model, instances)
        if command.type == CommandType.WAKE_UP:
            return await self.lifecycle.wake(model, instances)
        if command.type == CommandType.SYNC:
            self.lifecycle.require_dynamic(model)
            version = args.get("curr_ps_model_version")
            if version is None:
                raise ValueError("SYNC requires curr_ps_model_version.")
            instances = await self.lifecycle.awake(model, instances)
            async with await model.reserve([iid[0] for iid in instances]):
                instances = await self.lifecycle.awake(model, instances)
                await asyncio.gather(
                    *(
                        model.replicas[rid].server.sync_with_ps.remote(version, pause_generation=True)
                        for rid in sorted({iid[0] for iid in instances})
                    )
                )
            return None
        raise ValueError(f"Unsupported inference command: {command.type!r}.")

    async def migrate_requests(self, model: ModelState, moves: list[dict]) -> None:
        timeout = float(self.config.psrl.deployment.elastic.get("request_migration_timeout_s", 60.0))
        for move in moves:
            source, destination = move["source_instance_id"], move["destination_instance_id"]
            model.expand_instances([tuple(source), tuple(destination)])
            async with await model.reserve([source[0], destination[0]]):
                server = model.replicas[source[0]].server
                engine_id = str(move["request_id"])
                request_id = await server.get_external_request_id.remote(engine_id)
                if request_id is None:
                    continue
                await self.gateway.post(
                    model,
                    "/routing_loop/migration",
                    {
                        "request_id": request_id,
                        "destination": list(destination),
                    },
                )
                try:
                    if not await server.abort_requests.remote([engine_id]):
                        continue
                    deadline = time.monotonic() + timeout
                    while True:
                        status = await self.gateway.get(model, "/routing_loop/migration", {"request_id": request_id})
                        if status and (status["dispatched"] or status["completed"]):
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(f"Request migration did not settle: request_id={request_id!r}.")
                        await asyncio.sleep(0.01)
                finally:
                    await self.gateway.post(model, "/routing_loop/migration/finish", {"request_id": request_id})
