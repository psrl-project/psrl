import asyncio
import logging
import os
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any

import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import TensorDict
from verl.utils import hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.fs import copy_to_local

from psrl.utils.dataset.utils import normalize_reward_model_dicts
from psrl.utils.logger import DualOutputHandler
from psrl.workers.elastic.sleep_wake_orchestrator import inference_lease
from psrl.workers.reward.client import RewardManagerClient
from psrl.workers.reward.reward_loop import load_reward_loop
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.reward_protocol import resolve_reward_function_name
from psrl.workers.reward.utils import ensure_reward_attention_mask

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class RewardSpec:
    reward_loop_type: str
    reward_fn_name: str
    reward_model_name: str | None

    def __hash__(self):
        return hash((self.reward_loop_type, self.reward_fn_name, self.reward_model_name))

    def __eq__(self, other):
        if not isinstance(other, RewardSpec):
            return NotImplemented
        return (
            self.reward_loop_type == other.reward_loop_type
            and self.reward_fn_name == other.reward_fn_name
            and self.reward_model_name == other.reward_model_name
        )

    def key(self) -> str:
        return f"{self.reward_loop_type}/{self.reward_fn_name}/{self.reward_model_name}"


@ray.remote
class RewardLoopWorker:
    """Execute reward computation at request granularity."""

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        reward_loop_configs: list[DictConfig],
        reward_model_runtime_infos: dict[str, Any] | None = None,
        worker_id: int = 0,
        worker_num: int = 1,
        *,
        is_validate: bool,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_loop_configs = reward_loop_configs
        self.reward_model_runtime_infos = reward_model_runtime_infos or {}
        self.worker_id = worker_id
        self.worker_num = worker_num
        self.is_validate = is_validate
        self.inference_orchestrator = None

        tq.init()

        worker_cfg = (
            self.config.reward.validation_reward_loop_worker if is_validate else self.config.reward.reward_loop_worker
        )
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

        self.reward_spec_to_loop: dict[RewardSpec, RewardLoopBase] = {}
        self._init_reward_fn()

        pool_name = "ValidationRewardLoopWorker" if is_validate else "RewardLoopWorker"
        self.log_prefix = f"{pool_name}_I{worker_id}"
        handler = DualOutputHandler(self.config.psrl.logging_path, self.log_prefix)
        logging.getLogger("psrl").addHandler(handler)
        psrl_logger.addHandler(handler)

    def set_reward_manager(self, reward_manager_handle: ray.actor.ActorHandle):
        self.reward_manager_client = RewardManagerClient(reward_manager_handle, self.max_concurrency)

    def set_inference_orchestrator(self, orchestrator):
        """Install model scoped GPU admission for generative reward calls."""
        self.inference_orchestrator = orchestrator

    async def add_reward_requests(self, requests: list[TensorDict]):
        if any(bool(tu.get(request, "validate", False)) != self.is_validate for request in requests):
            raise ValueError("Reward request does not belong to this worker's pool.")
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

        for reward_loop_config in self.reward_loop_configs:
            reward_loop_cfg = reward_loop_config.reward_loop
            reward_loop_type = reward_loop_config.reward_loop_type
            reward_fn_configs = reward_loop_config.reward_fn
            reward_model_name = reward_loop_config.get("reward_model_name", None)
            reward_kwargs = reward_loop_config.get("reward_kwargs", {})

            for reward_fn_config in reward_fn_configs:
                reward_fn_name = reward_fn_config.get("name", "null")
                reward_spec = RewardSpec(
                    reward_loop_type=reward_loop_type,
                    reward_fn_name=reward_fn_name,
                    reward_model_name=reward_model_name,
                )
                if reward_spec in self.reward_spec_to_loop:
                    raise ValueError(f"Duplicate reward loop spec {reward_spec.key()!r}.")

                if reward_loop_type in ("gen", "opd") and (
                    reward_model_name is None or reward_model_name not in self.reward_model_runtime_infos
                ):
                    raise ValueError(f"Reward model runtime info for {reward_model_name} not found")

                runtime_info = self.reward_model_runtime_infos.get(reward_model_name, None)
                reward_loop = load_reward_loop(
                    self.config,
                    self.input_tokenizer,
                    reward_loop_cfg=reward_loop_cfg,
                    reward_fn_config=reward_fn_config,
                    reward_model_manager=runtime_info,
                    **reward_kwargs,
                )
                self.reward_spec_to_loop[reward_spec] = reward_loop

    def _resolve_reward_loops(
        self, reward_model_dicts: list[dict]
    ) -> tuple[list[str], list[RewardLoopBase], list[float]]:
        reward_spec_keys, reward_loops, coefs = [], [], []
        for reward_model_dict in normalize_reward_model_dicts(reward_model_dicts):
            reward_fn_name = resolve_reward_function_name(reward_model_dict.get("reward_fn", "default"))

            if reward_fn_name == "default" and reward_model_dict.get("reward_loop_type", "naive") in (
                "naive",
                "dapo",
                "prime",
                "gdpo",
            ):
                reward_fn_name = "compute_score"

            raw_model_name = reward_model_dict.get("reward_model_name", None)
            if raw_model_name == "null":
                raw_model_name = None

            reward_spec = RewardSpec(
                reward_loop_type=reward_model_dict.get("reward_loop_type", "naive"),
                reward_fn_name=reward_fn_name,
                reward_model_name=raw_model_name,
            )
            reward_loop = self.reward_spec_to_loop.get(reward_spec, None)

            if reward_loop is None:
                raise ValueError(f"No reward loop registered for spec {reward_spec.key()!r}.")

            reward_spec_keys.append(reward_spec.key())
            reward_loops.append(reward_loop)
            coefs.append(reward_model_dict.get("reward_coef", 1.0))
        return reward_spec_keys, reward_loops, coefs

    def pre_process(self, inputs: TensorDict) -> TensorDict:
        return ensure_reward_attention_mask(inputs)

    async def _compute_score(self, reward_data: TensorDict) -> dict[str, Any]:
        """Aggregate scalar rewards and score each trajectory with its selected teacher."""
        if len(reward_data) != 1:
            raise ValueError("Reward scoring requires exactly one request.")
        specs = normalize_reward_model_dicts(tu.get(reward_data, "reward_model_dicts")[0])
        keys, reward_loops, coefs = self._resolve_reward_loops(specs)
        teachers = [i for i, spec in enumerate(specs) if spec.get("reward_loop_type") == "opd"]
        enabled = getattr(getattr(self.config, "distillation", None), "enabled", False)
        required = enabled and not self.is_validate
        if len(teachers) > 1 or (required and len(teachers) != 1):
            raise ValueError("Each OPD training request must select exactly one teacher reward model.")
        scalar_indices = [i for i in range(len(reward_loops)) if i not in teachers]
        results = await asyncio.gather(
            *(self._run_scoring_loop(reward_loops[i], specs[i], reward_data) for i in scalar_indices)
        )
        reward = {
            "reward_score": float(sum(r["reward_score"] * coefs[i] for i, r in zip(scalar_indices, results))),
            "reward_extra_info": {keys[i]: r["reward_extra_info"] for i, r in zip(scalar_indices, results)},
            "reward_metrics": {keys[i]: r.get("reward_metrics", {}) for i, r in zip(scalar_indices, results)},
        }
        if teachers:
            index = teachers[0]
            request_keys = tu.get(reward_data, "request_keys")[0]
            trajectories = tu.get(reward_data, "reward_trajectories", [None])[0]
            if trajectories is None:
                if len(request_keys) != 1:
                    raise ValueError("OPD requires the tokens of every training trajectory.")
                trajectory_data = [reward_data]
            else:
                if len(trajectories) != len(request_keys):
                    raise ValueError("OPD trajectory count does not match request_keys.")
                trajectory_data = []
                for trajectory in trajectories:
                    row = tu.get_tensordict(
                        tensor_dict={
                            "prompts": torch.tensor([trajectory["prompts"]], dtype=torch.int64),
                            "responses": torch.tensor([trajectory["responses"]], dtype=torch.int64),
                        },
                    )
                    if trajectory["multi_modal_data"]:
                        tu.assign_non_tensor_stack(row, "multi_modal_data", [trajectory["multi_modal_data"]])
                    trajectory_data.append(self.pre_process(row))
            outputs = await asyncio.gather(
                *(self._run_scoring_loop(reward_loops[index], specs[index], row) for row in trajectory_data)
            )
            reward["teacher_outputs"] = {
                key: {
                    **{field: output[field] for field in ("teacher_logprobs", "teacher_ids") if field in output},
                    "teacher_weight": float(specs[index]["teacher_weight"]),
                    "reward_extra_info": {keys[index]: output.get("reward_extra_info", {})},
                    "reward_metrics": {keys[index]: output.get("reward_metrics", {})},
                }
                for key, output in zip(request_keys, outputs)
            }
        return reward

    async def _run_scoring_loop(self, reward_loop: RewardLoopBase, specification: dict, data: TensorDict):
        """Serialize shared model storage while allowing independent CPU scoring."""
        orchestrator = self.inference_orchestrator if specification.get("reward_loop_type") in ("gen", "opd") else None
        async with inference_lease(orchestrator, "reward", specification.get("reward_model_name", "")):
            return await reward_loop.run_single(data)
