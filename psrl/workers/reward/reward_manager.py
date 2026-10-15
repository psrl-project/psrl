import asyncio
import logging
import os
from typing import Any

import numpy as np
import ray
import torch
import transfer_queue as tq
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RewardLoopManager(CommandExtension):
    def __init__(
        self,
        config,
        ps_manager_handle,
        reward_loop_workers: list[ray.actor.ActorHandle],
        *,
        is_validate: bool,
    ):
        """Initialize reward processing state and worker coordination."""
        super().__init__()

        self.config = config
        self.is_validate = is_validate

        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n={self.rollout_n} must be greater than or equal to alg_rollout_n={self.alg_rollout_n}."
        )
        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n

        self.request_id_to_n_trajectory = {}
        self.request_key_to_future: dict[str, asyncio.Future] = {}
        self.request_key_to_reward = {}
        self.request_id_to_data_source = {}
        self.request_id_to_request_keys: dict[Any, list[str]] = {}
        self.request_key_to_worker: dict[str, int] = {}
        self.request_key_to_attempt: dict[str, int] = {}

        self.reward_normalization = self.config.reward.reward_normalization
        self.request_id_to_group = {}

        self.running_loop = None
        self.command_loop_task = None
        self.collect_task = None
        self.dispatch_task: asyncio.Task | None = None
        self.stop_command_loop_task = False
        self.stop_collect_task = False
        self.stop_dispatch_task = False

        worker_cfg = (
            self.config.reward.validation_reward_loop_worker if is_validate else self.config.reward.reward_loop_worker
        )
        if worker_cfg.result_queue_size <= 0 or worker_cfg.result_drain_batch_size <= 0:
            raise ValueError("Reward result queue size and drain batch size must be positive.")
        self.result_queue = asyncio.Queue(maxsize=worker_cfg.result_queue_size)
        self.result_drain_batch_size = worker_cfg.result_drain_batch_size
        assert reward_loop_workers, "Reward loop workers are required."
        self.reward_loop_workers = reward_loop_workers
        self.dispatch_idx = 0
        self.attempt_counter = 0
        self.dispatch_queue: asyncio.Queue = asyncio.Queue()

        self.ps_manager_handle = ps_manager_handle

        self.log_prefix = "ValidationRewardLoopManager" if is_validate else "RewardLoopManager"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized {self.log_prefix}.")

    def set_sleep_wake_orchestrator(self, orchestrator):
        """Inject the SleepWakeOrchestrator (colocated modes 2/3)."""
        self.sleep_wake_orchestrator = orchestrator

    def start_busy_loop(self):
        """Start reward request processing tasks."""
        if self.command_loop_task is not None and not self.command_loop_task.done():
            return

        for worker in self.reward_loop_workers:
            worker.start_busy_loop.remote()

        self.stop_command_loop_task = False
        self.stop_collect_task = False
        self.running_loop = asyncio.get_running_loop()
        self.command_loop_task = self.running_loop.create_task(self._command_event_handler())
        self.command_loop_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        self.collect_task = self.running_loop.create_task(self._collect_results())
        self.collect_task.add_done_callback(lambda f: f.result())
        self.stop_dispatch_task = False
        self.dispatch_task = self.running_loop.create_task(self._dispatch_loop())
        self.dispatch_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        """Stop and await reward request processing tasks."""
        if (
            (not self.command_loop_task or self.command_loop_task.done())
            and (not self.collect_task or self.collect_task.done())
            and (not self.dispatch_task or self.dispatch_task.done())
        ):
            return

        self.stop_command_loop_task = True
        self.stop_collect_task = True
        self.stop_dispatch_task = True
        tasks = [task for task in (self.command_loop_task, self.collect_task, self.dispatch_task) if task is not None]
        await asyncio.gather(*tasks)
        await asyncio.gather(*[worker.stop_busy_loop.remote() for worker in self.reward_loop_workers])

    async def _command_event_handler(self):
        """Process reward manager commands until shutdown."""
        while not self.stop_command_loop_task:
            if not self.command_queue.empty():
                command = self.command_queue.get_nowait()

                assert isinstance(command, Command), f"Expected Command, got {type(command)}"

                command_type = command.type
                command_id = command.get_kwargs()["id"]
                command_args = command.get_args()
                psrl_logger.debug(
                    f"Receive command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}"
                )

                result = None

                if command_type == CommandType.ABORT:
                    assert "parent_ids" in command_args or "uids" in command_args, (
                        "Abort command must contain either 'parent_ids' or 'uids' in args."
                    )
                    parent_ids = command_args.get("parent_ids", None)
                    uids = command_args.get("uids", None)
                    is_validate = command_args.get("is_validate", False)

                    assert not is_validate and not self.is_validate, "Validation rewards must not be aborted."

                    if parent_ids is None and uids is None:
                        raise ValueError("Abort command must contain either 'parent_ids' or 'uids' in args.")

                    psrl_logger.debug(f"Received ABORT command with parent_ids: {parent_ids}, uids: {uids}")
                    if not isinstance(parent_ids, (list, type(None))):
                        parent_ids = [parent_ids]
                    if not isinstance(uids, (list, type(None))):
                        uids = [uids]

                    abort_request_uids = set()
                    if parent_ids is not None:
                        parent_ids = set(parent_ids)  # Ensure uniqueness
                        psrl_logger.debug(f"Getting child requests. Parent count={len(parent_ids)}.")
                        child_uids = await self.ps_manager_handle.get_recorded_child_requests.remote(
                            list(parent_ids), is_validate
                        )
                        psrl_logger.debug(f"Found child requests. Child count={len(child_uids)}.")
                        abort_request_uids.update(child_uids)
                    if uids is not None:
                        uids = set(uids)
                        abort_request_uids.update(uids)

                    psrl_logger.debug(f"Requests to abort={len(abort_request_uids)}.")
                    for abort_request_id in abort_request_uids:
                        self.request_id_to_group.pop(abort_request_id, None)
                        self.request_id_to_data_source.pop(abort_request_id, None)

                    aborted_count = 0
                    worker_to_abort_keys: dict[int, list[str]] = {}
                    for abort_request_id in abort_request_uids:
                        n_trajectory = self.request_id_to_n_trajectory.pop(abort_request_id, 1)
                        if n_trajectory > 1:
                            reward_keys = [f"{abort_request_id}_{i}" for i in range(n_trajectory)]
                        else:
                            reward_keys = [str(abort_request_id)]

                        for key in reward_keys:
                            self.request_key_to_attempt.pop(key, None)
                            worker_index = self.request_key_to_worker.pop(key, None)
                            if worker_index is not None:
                                worker_to_abort_keys.setdefault(worker_index, []).append(key)

                            fut = self.request_key_to_future.pop(key, None)
                            if fut is not None and not fut.done():
                                fut.set_result(None)
                                aborted_count += 1
                            self.request_key_to_reward[key] = None

                    if worker_to_abort_keys:
                        await asyncio.gather(
                            *[
                                self.reward_loop_workers[worker_index].abort_requests.remote(keys)
                                for worker_index, keys in worker_to_abort_keys.items()
                            ],
                            return_exceptions=True,
                        )

                    psrl_logger.debug(f"Aborted reward computations={aborted_count}.")
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(
                        list(abort_request_uids),
                        PSRL_RequestStatus.REWARD_COMPLETED,
                        is_validate=is_validate,
                    )
                    # `update_request_status` returns a bool for one request and a list for multiple requests.
                    if isinstance(update_status_success, bool):
                        update_status_success = [update_status_success]
                    assert all(not status for status in update_status_success), (
                        "Update status should not be successful for aborted requests."
                    )
                    result = aborted_count
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                psrl_logger.debug(f"Completing command={command_id!r} with result={result!r}.")

            await asyncio.sleep(0)
        psrl_logger.info("Command event handler of reward manager has finished.")

    async def normalize_reward(self, reward_batch: KVBatchMeta) -> KVBatchMeta:
        """Normalize rewards for real trajectories without consuming padding metadata."""
        real_keys = [key for key, tag in zip(reward_batch.keys, reward_batch.tags) if not tag.get("is_padding", False)]
        if not real_keys:
            return reward_batch

        fields = ["uid", "reward_score", "reward_extra_info"]
        data = await tq.async_kv_batch_get(
            keys=real_keys, partition_id=reward_batch.partition_id, select_fields=fields
        )

        uids = tu.get(data, "uid")
        reward_extra_infos = tu.get(data, "reward_extra_info")
        uid_to_data_source = {uid: self.request_id_to_data_source[uid] for uid in set(uids)}
        for uid in uid_to_data_source:
            self.request_id_to_data_source.pop(uid)
        data_sources = [uid_to_data_source[uid] for uid in uids]
        reward_scores = np.asarray(tu.get(data, "reward_score"))

        for reward_extra_info, data_source, reward_score in zip(reward_extra_infos, data_sources, reward_scores):
            reward_extra_info["data_source"] = data_source
            reward_extra_info["original_reward_score"] = reward_score

        if self.reward_normalization != "batch" and self.reward_normalization != "group":
            data["reward_extra_info"] = reward_extra_infos
            await tq.async_kv_batch_put(
                keys=real_keys,
                partition_id=reward_batch.partition_id,
                fields=data.select("reward_extra_info"),
            )
            return reward_batch

        # Only each UID's final trajectory contributes to normalization.
        final_trajectories: dict[str, tuple[int, int]] = {}
        row_session_keys = []
        for i, key in enumerate(real_keys):
            fields = key.rsplit("_", 1)
            if len(fields) == 2:
                uid_key, index = fields[0], int(fields[1])
                if uid_key not in final_trajectories or final_trajectories[uid_key][0] < index:
                    final_trajectories[uid_key] = (index, i)
            else:
                uid_key, index = key, 0
                final_trajectories[uid_key] = (index, i)
            row_session_keys.append(uid_key)

        final_indices = []
        uid_key_to_local_index = {}
        for uid, (_, row_index) in final_trajectories.items():
            final_indices.append(row_index)
            uid_key_to_local_index[uid] = len(final_indices) - 1
        row_to_local_index = [uid_key_to_local_index[uid_key] for uid_key in row_session_keys]

        group_rewards_dicts = {}
        for local_index, row_index in enumerate(final_indices):
            uid = uids[row_index]
            if uid not in self.request_id_to_group:
                continue
            group_id = self.request_id_to_group[uid]
            if group_id not in group_rewards_dicts:
                group_rewards_dicts[group_id] = {
                    "local_indices": [],
                    "rewards": [],
                }
            group_rewards_dicts[group_id]["local_indices"].append(local_index)
            group_rewards_dicts[group_id]["rewards"].append(reward_scores[row_index])
            self.request_id_to_group.pop(uid, None)

        final_reward_scores = [0.0] * len(final_indices)
        for group_id, group_rewards_dict in group_rewards_dicts.items():
            local_indices = group_rewards_dict["local_indices"]
            rewards = np.array(group_rewards_dict["rewards"])
            psrl_logger.info(f"Rewards for group {group_id!r}: {len(rewards)}.")
            norm_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            for local_index, norm_reward in zip(local_indices, norm_rewards):
                final_reward_scores[local_index] = float(norm_reward)

        # Share each normalized reward across all trajectories for that UID.
        normalized_reward_scores = [final_reward_scores[local_index] for local_index in row_to_local_index]
        reward_scores = np.asarray(normalized_reward_scores, dtype=reward_scores.dtype)

        data["reward_extra_info"] = reward_extra_infos
        data["reward_score"] = reward_scores

        response_data = await tq.async_kv_batch_get(
            keys=real_keys,
            partition_id=reward_batch.partition_id,
            select_fields=["response_mask"],
        )
        response_mask = response_data["response_mask"]
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        rm_scores[:, -1] = torch.as_tensor(reward_scores, dtype=rm_scores.dtype, device=rm_scores.device)
        data["rm_scores"] = rm_scores

        await tq.async_kv_batch_put(
            keys=real_keys,
            partition_id=reward_batch.partition_id,
            fields=data.select("reward_extra_info", "reward_score", "rm_scores"),
        )
        return reward_batch

    @staticmethod
    def _default_reward_result() -> dict[str, Any]:
        return {
            "reward_score": 0.0,
            "reward_extra_info": {},
            "reward_metrics": {},
            "response_len": 1,
        }

    def _next_attempt_id(self) -> int:
        self.attempt_counter += 1
        return self.attempt_counter

    async def _dispatch_loop(self):
        """Drain the dispatch queue and send requests to workers."""
        while not self.stop_dispatch_task:
            batch: list[TensorDict] = []
            while not self.dispatch_queue.empty():
                try:
                    batch.append(self.dispatch_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if not batch:
                await asyncio.sleep(0)
                continue

            await self._dispatch_reward_requests(batch)
        psrl_logger.info("Dispatch loop has finished.")

    async def _dispatch_reward_requests(self, requests: list[TensorDict]) -> None:
        requests_by_worker: dict[int, list[TensorDict]] = {}
        worker_num = len(self.reward_loop_workers)
        for request in requests:
            worker_index = self.dispatch_idx % worker_num
            self.dispatch_idx += 1
            requests_by_worker.setdefault(worker_index, []).append(request)
            request_keys = tu.get(request, "request_keys")[0]
            for key in request_keys:
                self.request_key_to_worker[key] = worker_index

        tasks = []
        for worker_index, worker_requests in requests_by_worker.items():
            tasks.append(self.reward_loop_workers[worker_index].add_reward_requests.remote(worker_requests))
        await asyncio.gather(*tasks)

    async def put_reward_result(self, result: dict):
        if bool(result["is_validate"]) != self.is_validate:
            raise ValueError("Reward result does not belong to this manager's worker pool.")
        await self.result_queue.put(result)

    async def _collect_results(self):
        while not self.stop_collect_task:
            results: list[dict] = []
            while len(results) < self.result_drain_batch_size and not self.result_queue.empty():
                results.append(self.result_queue.get_nowait())

            if not results:
                await asyncio.sleep(0)
                continue

            await self._handle_reward_results(results)

        psrl_logger.info("Reward result collector has finished.")

    async def _handle_reward_results(self, results: list[dict]):
        valid_results = []
        for result in results:
            request_keys = result.get("request_keys")
            if not request_keys:
                continue
            expected_attempt = self.request_key_to_attempt.get(request_keys[0])
            if expected_attempt != result["attempt_id"]:
                psrl_logger.debug(
                    "Drop stale reward result uid=%s attempt=%s expected=%s",
                    result["uid"],
                    result["attempt_id"],
                    expected_attempt,
                )
                continue
            valid_results.append(result)

        if not valid_results:
            return

        await self.ps_manager_handle.update_request_status.remote(
            [result["uid"] for result in valid_results],
            PSRL_RequestStatus.REWARD_COMPLETED,
            is_validate=self.is_validate,
        )
        for result in valid_results:
            reward = result["result"] if result.get("error") is None else None
            if result.get("error") is not None:
                psrl_logger.error(
                    f"Reward worker {result['worker_id']} failed for uid={result['uid']!r}, "
                    f"keys={result['request_keys']!r}. Error: {result['error']!r}."
                )
            if reward is not None:
                reward["response_len"] = result["response_len"]
            await self._set_reward_for_request_keys(result["request_keys"], reward)

    async def _set_reward_for_request_keys(self, request_keys: list[str], reward: dict[str, Any] | None):
        for request_key in request_keys:
            per_key_reward = reward.copy() if reward is not None else None
            self.request_key_to_attempt.pop(request_key, None)
            self.request_key_to_worker.pop(request_key, None)
            if request_key in self.request_key_to_future:
                fut = self.request_key_to_future[request_key]
                if not fut.done():
                    fut.set_result(per_key_reward)
            else:
                self.request_key_to_reward[request_key] = per_key_reward

    async def compute_score(
        self,
        reward_inputs: TensorDict,
    ) -> dict[str, Any] | None:
        """Dispatch reward computation to reward loop workers."""
        is_validate = tu.get(reward_inputs, "validate", False)
        if bool(is_validate) != self.is_validate:
            raise ValueError("Reward request does not belong to this manager's worker pool.")
        request_ids = tu.get(reward_inputs, "uid")
        n_trajectories = tu.get(reward_inputs, "n_trajectory")

        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        request_keys = []
        dispatch_items: list[TensorDict] = []

        with log_dual_events(
            f"Compute reward for requests {request_ids}",
            psrl_logger,
            level=logging.DEBUG,
            event_type=EventType.OTHER,
        ):
            update_status_success = await self.ps_manager_handle.update_request_status.remote(
                request_ids,
                PSRL_RequestStatus.REWARD_RUNNING,
                is_validate=is_validate,
            )
            if not isinstance(update_status_success, list):
                update_status_success = [update_status_success]

            for i, (request_id, n_trajectory, status_ok) in enumerate(
                zip(request_ids, n_trajectories, update_status_success)
            ):
                reward_input = reward_inputs[i : i + 1]
                prompt_key = "parent_id" if rollout_n > 1 else "uid"
                prompt_id = tu.get(reward_input, prompt_key)[0]
                data_source = tu.get(reward_input, "data_source")[0]
                if n_trajectory > 1:
                    current_request_keys = [f"{request_id}_{j}" for j in range(n_trajectory)]
                else:
                    current_request_keys = [str(request_id)]
                request_keys.extend(current_request_keys)

                self.request_id_to_n_trajectory[request_id] = n_trajectory
                self.request_id_to_request_keys[request_id] = current_request_keys
                if self.reward_normalization == "batch":
                    self.request_id_to_group[request_id] = data_source
                elif self.reward_normalization == "group":
                    self.request_id_to_group[request_id] = prompt_id
                self.request_id_to_data_source[request_id] = data_source

                attempt_id = self._next_attempt_id()
                for request_key in current_request_keys:
                    if status_ok:
                        existing_fut = self.request_key_to_future.get(request_key)
                        if existing_fut is None or existing_fut.done():
                            fut = asyncio.get_running_loop().create_future()
                            self.request_key_to_future[request_key] = fut
                        self.request_key_to_attempt[request_key] = attempt_id
                    else:
                        psrl_logger.warning(
                            "compute_score:update_request_status(REWARD_RUNNING) failed for uid=%s "
                            "(aborted?), returning None.",
                            request_id,
                        )
                        existing_fut = self.request_key_to_future.pop(request_key, None)
                        if existing_fut is not None and not existing_fut.done():
                            existing_fut.set_result(None)
                        self.request_key_to_reward[request_key] = None

                if status_ok:
                    tu.assign_non_tensor_stack(reward_input, "request_keys", [current_request_keys])
                    tu.assign_non_tensor_stack(reward_input, "attempt_id", [attempt_id])
                    dispatch_items.append(reward_input)

            if dispatch_items:
                if self.config.reward.launch_reward_fn_async:
                    for item in dispatch_items:
                        self.dispatch_queue.put_nowait(item)
                else:
                    await self._dispatch_reward_requests(dispatch_items)

            if self.config.reward.launch_reward_fn_async:
                return None

            # NOTE(linsh): Synchronous reward computation supports only one request.
            sync_keys = tu.get(dispatch_items[0], "request_keys")[0] if dispatch_items else request_keys
            sync_results = []
            for request_key in sync_keys:
                if request_key in self.request_key_to_reward:
                    sync_results.append(self.request_key_to_reward.pop(request_key))
                else:
                    sync_results.append(await self.request_key_to_future[request_key])
                    self.request_key_to_future.pop(request_key, None)

        return sync_results[0] if sync_results else None

    async def wait_for_reward_of_requests(self, requests: KVBatchMeta) -> KVBatchMeta:
        """Wait for all requested rewards and write them to the transfer queue."""
        request_keys = requests.keys
        is_validate = requests.partition_id == "val"
        if is_validate != self.is_validate:
            raise ValueError("Reward batch does not belong to this manager's worker pool.")
        partition_id = "val" if is_validate else "train"
        # Padding from `_balance_batch` in `ray_trainer.py`
        padding_keys: set[str] = {
            key for key, tag in zip(requests.keys, requests.tags) if tag.get("is_padding", False)
        }
        # Reward may be ready in group post processing
        ready_keys = {
            key for key, tag in zip(requests.keys, requests.tags) if not is_validate and tag.get("reward_ready", False)
        }
        request_keys_to_put = [key for key in request_keys if key not in ready_keys]
        request_key_to_reward: dict[str, dict] = {}
        futures_to_wait = {}

        for request_key in request_keys_to_put:
            if request_key in padding_keys:
                request_key_to_reward[request_key] = {
                    "reward_score": 0.0,
                    "reward_extra_info": {},
                    "reward_metrics": {},
                    "response_len": 1,
                }
            elif request_key in self.request_key_to_reward:
                request_key_to_reward[request_key] = self.request_key_to_reward.pop(request_key)
            elif request_key in self.request_key_to_future:
                futures_to_wait[request_key] = self.request_key_to_future[request_key]
            else:
                fut = asyncio.get_running_loop().create_future()
                self.request_key_to_future[request_key] = fut
                futures_to_wait[request_key] = fut

        if futures_to_wait:
            results = await asyncio.gather(*futures_to_wait.values())
            for request_key, reward in zip(futures_to_wait.keys(), results):
                request_key_to_reward[request_key] = reward

        fields = []
        for request_key in request_keys_to_put:
            reward = request_key_to_reward[request_key]
            if reward is None:
                reward = self._default_reward_result()
            fields.append(reward)
            self.request_key_to_future.pop(request_key, None)

        for i, f in enumerate(fields):
            raw_response_len = f.pop("response_len", 1)
            if raw_response_len <= 0:
                psrl_logger.error(
                    "wait_for_reward_of_requests: response_len=%d for key=%s, reward_score=%s, reward_keys=%s",
                    raw_response_len,
                    request_keys_to_put[i] if i < len(request_keys_to_put) else "?",
                    f.get("reward_score"),
                    list(f.keys()),
                )
            response_len = max(raw_response_len, 1)
            rm_score_tensor = torch.zeros(response_len, dtype=torch.float32)
            rm_score_tensor[-1] = float(f.get("reward_score", 0.0))
            f["rm_scores"] = rm_score_tensor

        if request_keys_to_put:
            await tq.async_kv_batch_put(
                keys=request_keys_to_put,
                partition_id=partition_id,
                fields=list_of_dict_to_tensordict(fields),
            )

        if not is_validate and self.reward_normalization:
            requests = await self.normalize_reward(requests)

        return requests

    async def wait_for_reward_ready(self, request_keys: list[str]) -> None:
        """
        Write available rewards for request keys to the transfer queue.

        Agent post-processing requires reward scores before the trainer's regular queue write.
        """
        futures_to_wait: dict[str, asyncio.Future] = {}
        resolved: dict[str, dict] = {}
        for request_key in request_keys:
            if request_key in self.request_key_to_reward:
                resolved[request_key] = self.request_key_to_reward[request_key]
            elif request_key in self.request_key_to_future:
                fut = self.request_key_to_future[request_key]
                if fut.done():
                    resolved[request_key] = fut.result()
                else:
                    futures_to_wait[request_key] = fut
            else:
                fut = asyncio.get_running_loop().create_future()
                self.request_key_to_future[request_key] = fut
                futures_to_wait[request_key] = fut

        if futures_to_wait:
            results = await asyncio.gather(*futures_to_wait.values())
            for request_key, result in zip(futures_to_wait.keys(), results):
                resolved[request_key] = result

        fields = []
        for request_key in request_keys:
            reward = resolved.get(request_key)
            if reward is None:
                reward = self._default_reward_result()
            f = reward.copy()
            response_len = max(f.pop("response_len", 1), 1)
            rm_score_tensor = torch.zeros(response_len, dtype=torch.float32)
            rm_score_tensor[-1] = float(f.get("reward_score", 0.0))
            f["rm_scores"] = rm_score_tensor
            fields.append(f)

        if request_keys:
            await tq.async_kv_batch_put(
                keys=request_keys,
                partition_id="train",
                fields=list_of_dict_to_tensordict(fields),
            )
            for request_key in request_keys:
                self.request_key_to_reward.pop(request_key, None)
                self.request_key_to_future.pop(request_key, None)
