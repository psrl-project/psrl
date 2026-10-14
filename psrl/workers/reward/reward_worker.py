import asyncio
import logging
import os
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import ray
import transfer_queue as tq
from omegaconf import DictConfig, ListConfig
from tensordict import TensorDict
from verl.utils import hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.fs import copy_to_local

from psrl.utils.dataset.utils import normalize_reward_model_dicts
from psrl.utils.logger import DualOutputHandler
from psrl.workers.reward.client import RewardManagerClient
from psrl.workers.reward.reward_loop import load_reward_loop
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.utils import ensure_reward_attention_mask

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class RewardSpec:
    reward_manager_type: str
    reward_fn_name: str
    reward_model_name: str | None

    def __hash__(self):
        return hash((self.reward_manager_type, self.reward_fn_name, self.reward_model_name))

    def __eq__(self, other):
        if not isinstance(other, RewardSpec):
            return NotImplemented
        return (
            self.reward_manager_type == other.reward_manager_type
            and self.reward_fn_name == other.reward_fn_name
            and self.reward_model_name == other.reward_model_name
        )

    def key(self) -> str:
        return f"{self.reward_manager_type}/{self.reward_fn_name}/{self.reward_model_name}"


@ray.remote
class RewardLoopWorker:
    """Execute reward computation at request granularity."""

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        reward_model_configs: list[DictConfig],
        reward_model_runtime_infos: dict[str, Any] | None = None,
        worker_id: int = 0,
        worker_num: int = 1,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_model_configs = reward_model_configs
        self.reward_model_to_manager = reward_model_runtime_infos or {}
        self.worker_id = worker_id
        self.worker_num = worker_num

        tq.init()

        worker_cfg = self.config.reward.reward_loop_worker
        self.max_concurrency = worker_cfg.max_concurrency_per_worker
        self.queue_size = worker_cfg.queue_size_per_worker

        self.reward_manager_client: RewardManagerClient | None = None
        self.pending_reward_queue = deque()
        self.reward_tasks: set[asyncio.Task] = set()
        self.request_key_to_task: dict[str, asyncio.Task] = {}
        self.aborted_keys: set[str] = set()
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self.max_concurrency)

        self.running_loop: asyncio.AbstractEventLoop | None = None
        self.busy_loop_task: asyncio.Task | None = None
        self.monitor_task: asyncio.Task | None = None
        self.stop_busy_loop_task = False
        self.completed = 0
        self.failed = 0

        self.reward_spec_to_manager: dict[RewardSpec, RewardLoopBase] = {}
        self._init_reward_fn()

        self.log_prefix = f"RewardLoopWorker_I{worker_id}"
        handler = DualOutputHandler(self.config.psrl.logging_path, self.log_prefix)
        logging.getLogger("psrl").addHandler(handler)
        psrl_logger.addHandler(handler)

    def set_reward_manager(self, reward_manager_handle: ray.actor.ActorHandle):
        self.reward_manager_client = RewardManagerClient(reward_manager_handle, self.max_concurrency)

    async def add_reward_requests(self, requests: list[TensorDict]):
        while len(self.pending_reward_queue) + len(requests) > self.queue_size and not self.stop_busy_loop_task:
            await asyncio.sleep(0.001)
        for request in requests:
            self.pending_reward_queue.append(request)

    def start_busy_loop(self):
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return
        self.stop_busy_loop_task = False
        self.running_loop = asyncio.get_running_loop()
        self.busy_loop_task = self.running_loop.create_task(self._launch_reward_loop())
        self.busy_loop_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        if not self.busy_loop_task or self.busy_loop_task.done():
            return
        self.stop_busy_loop_task = True
        await asyncio.gather(self.busy_loop_task)
        if self.monitor_task is not None:
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)

    async def abort_requests(self, request_keys: list[str]):
        for key in request_keys:
            self.aborted_keys.add(str(key))
            task = self.request_key_to_task.get(str(key))
            if task is not None and not task.done():
                task.cancel()

    def get_stats(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "queue_depth": len(self.pending_reward_queue),
            "active_tasks": len(self.reward_tasks),
            "completed": self.completed,
            "failed": self.failed,
        }

    async def _launch_reward_loop(self):
        while not self.stop_busy_loop_task:
            if not self.pending_reward_queue:
                await asyncio.sleep(0)
                continue

            request: TensorDict = self.pending_reward_queue.popleft()
            request_keys = tu.get(request, "request_keys")[0]
            if any(str(key) in self.aborted_keys for key in request_keys):
                continue

            await self._semaphore.acquire()
            task = asyncio.create_task(self._run_reward_loop_with_release(request))
            task.add_done_callback(self._create_task_done_callback(task, request_keys))
            self.reward_tasks.add(task)
            for key in request_keys:
                self.request_key_to_task[str(key)] = task

    async def _run_reward_loop_with_release(self, request: TensorDict):
        try:
            await self._run_reward_loop(request)
        finally:
            self._semaphore.release()

    def _create_task_done_callback(self, task: asyncio.Task, request_keys: list[str]):
        def task_done_callback(future):
            try:
                future.result()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                psrl_logger.error(f"Reward task failed. Task={task!r}, error={e!r}.\nTraceback:\n{tb_str}")
            finally:
                self.reward_tasks.discard(task)
                for key in request_keys:
                    self.request_key_to_task.pop(str(key), None)

        return task_done_callback

    async def _run_reward_loop(self, request: TensorDict):
        result = None
        error = None
        response_len = 1

        uid = tu.get(request, "uid")[0]
        request_keys = tu.get(request, "request_keys")[0]
        is_validate = tu.get(request, "validate", False)
        attempt_id = tu.get(request, "attempt_id")[0]

        try:
            reward_data = await asyncio.get_event_loop().run_in_executor(None, self.pre_process, request)
            response_len = reward_data["responses"].shape[-1]
            if response_len == 0:
                psrl_logger.error(
                    "reward_worker: response_len=0 for uid=%s keys=%s, prompts_shape=%s, responses_shape=%s",
                    uid,
                    request_keys,
                    reward_data["prompts"].shape,
                    reward_data["responses"].shape,
                )

            result = await self._compute_score(reward_data)
            self.completed += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failed += 1
            error = traceback.format_exc()
            psrl_logger.error("Reward computation failed for uid=%s:\n%s", uid, error)

        reward_result = {
            "uid": uid,
            "request_keys": request_keys,
            "is_validate": is_validate,
            "attempt_id": attempt_id,
            "result": result,
            "error": error,
            "response_len": response_len,
            "worker_id": self.worker_id,
        }

        if self.reward_manager_client is not None:
            await self.reward_manager_client.put_reward_result(reward_result)

    def _init_reward_fn(self):
        input_tokenizer_local_path = copy_to_local(self.config.train_actor_rollout_ref.model.path)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path, trust_remote_code=True)

        for reward_model_config in self.reward_model_configs:
            reward_manager_cfg = reward_model_config.reward_manager
            reward_manager_type = reward_model_config.reward_loop_type
            reward_fn_configs = reward_model_config.reward_fn
            reward_model_name = reward_model_config.get("reward_model_name", "null")
            reward_kwargs = reward_model_config.get("reward_kwargs", {})

            for reward_fn_config in reward_fn_configs:
                reward_fn_name = reward_fn_config.get("name", "null")
                reward_spec = RewardSpec(
                    reward_manager_type=reward_manager_type,
                    reward_fn_name=reward_fn_name,
                    reward_model_name=reward_model_name,
                )
                if reward_spec in self.reward_spec_to_manager:
                    continue

                if reward_manager_type in ("gen", "opd") and (
                    reward_model_name is None or reward_model_name not in self.reward_model_to_manager
                ):
                    raise ValueError(f"Reward model runtime info for {reward_model_name} not found")

                reward_model_manager = self.reward_model_to_manager.get(reward_model_name, None)
                reward_manager = load_reward_loop(
                    self.config,
                    self.input_tokenizer,
                    reward_loop_cfg=reward_manager_cfg,
                    reward_fn_config=reward_fn_config,
                    reward_model_manager=reward_model_manager,
                    **reward_kwargs,
                )
                self.reward_spec_to_manager[reward_spec] = reward_manager

    def _resolve_reward_manager(
        self, reward_model_dicts: list[dict]
    ) -> tuple[list[str], list[RewardLoopBase], list[float]]:
        reward_spec_keys, reward_managers, coefs = [], [], []
        for reward_model_dict in normalize_reward_model_dicts(reward_model_dicts):
            raw_reward_fn = reward_model_dict.get("reward_fn", "default")
            if isinstance(raw_reward_fn, (list, ListConfig)) and len(raw_reward_fn) > 0:
                reward_fn_name = (
                    raw_reward_fn[0].get("name", "default")
                    if isinstance(raw_reward_fn[0], (dict, DictConfig))
                    else str(raw_reward_fn[0])
                )
            elif isinstance(raw_reward_fn, (dict, DictConfig)):
                reward_fn_name = raw_reward_fn.get("name", "default")
            else:
                reward_fn_name = str(raw_reward_fn)

            raw_model_name = reward_model_dict.get("reward_model_name", None)
            if raw_model_name == "null":
                raw_model_name = None

            reward_spec = RewardSpec(
                reward_manager_type=reward_model_dict.get("reward_loop_type", "naive"),
                reward_fn_name=reward_fn_name,
                reward_model_name=raw_model_name,
            )
            manager = self.reward_spec_to_manager.get(reward_spec, None)
            if manager is None:
                candidates = [
                    (spec, mgr)
                    for spec, mgr in self.reward_spec_to_manager.items()
                    if spec.reward_manager_type == reward_spec.reward_manager_type
                ]
                if len(candidates) == 0:
                    all_managers = list(self.reward_spec_to_manager.items())
                    if len(all_managers) == 1:
                        candidates = all_managers
                    elif len(all_managers) > 1:
                        psrl_logger.warning(
                            "No reward spec match for %s among %s. Using first registered manager as fallback.",
                            reward_spec,
                            [s for s, _ in all_managers],
                        )
                        candidates = [all_managers[0]]
                if candidates:
                    reward_spec, manager = candidates[0]

            if manager is None:
                raise ValueError(f"No reward manager registered for spec {reward_spec.key()!r}.")

            reward_spec_keys.append(reward_spec.key())
            reward_managers.append(manager)
            coefs.append(reward_model_dict.get("reward_coef", 1.0))
        return reward_spec_keys, reward_managers, coefs

    def pre_process(self, inputs: TensorDict) -> TensorDict:
        return ensure_reward_attention_mask(inputs)

    async def _compute_score(self, reward_data: TensorDict) -> dict[str, Any]:
        """Run reward loops on pre-processed TensorDict."""
        assert len(reward_data) == 1, "Reward input for _compute_score should contain exactly one sample"

        reward_model_dicts = tu.get(reward_data, "reward_model_dicts")[0]
        reward_spec_keys, reward_managers, reward_coefs = self._resolve_reward_manager(reward_model_dicts)
        futures = [asyncio.create_task(reward_manager.run_single(reward_data)) for reward_manager in reward_managers]
        results = await asyncio.gather(*futures)

        reward_score = np.sum([r["reward_score"] * c for r, c in zip(results, reward_coefs)])
        reward_extra_info_dict = {key: r["reward_extra_info"] for key, r in zip(reward_spec_keys, results)}
        reward_metrics_dict = {key: r.get("reward_metrics", {}) for key, r in zip(reward_spec_keys, results)}

        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info_dict,
            "reward_metrics": reward_metrics_dict,
        }
