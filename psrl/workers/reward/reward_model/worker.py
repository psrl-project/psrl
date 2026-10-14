# import asyncio
# import logging
# import os
# import queue
# import time
# import warnings
# from collections import defaultdict
# from typing import Any

# import numpy as np
# import ray
# import torch
# import torch.distributed as dist
# from omegaconf import DictConfig, OmegaConf
# from transformers import AutoConfig
# from verl import DataProto
# from verl.single_controller.base import Worker
# from verl.single_controller.base.decorator import Dispatch, register
# from verl.utils import hf_tokenizer, omega_conf_to_dataclass
# from verl.utils.device import get_torch_device
# from verl.utils.fs import copy_to_local
# from verl.utils.model import get_generation_config, update_model_config

# from psrl.utils.logger import (
#     DualOutputHandler,
#     EventType,
#     deprecated,
#     get_worker_info,
#     log_begin_event,
#     log_dual_events,
#     log_end_event,
#     log_single_event,
# )
# from psrl.utils.rollout.rollout_trace import rollout_trace_op
# from psrl.workers.config import HFModelConfig, RolloutConfig
# from psrl.workers.gen import GenInterface, PSRL_vLLMRollout

# psrl_logger = logging.getLogger(__file__)
# psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# @deprecated(reason="Will be merge into PSRL_GenWorker")
# class PSRL_RewardModelWorker(Worker):
#     """
#     A lightweight rollout worker dedicated to reward-model inference.

#     The worker receives preprocessed prompts, forwards them to vLLM, and returns
#     generated completions. No parameter-server specific logic is included.
#     """

#     @staticmethod
#     def configure_worker(
#         config,
#         psrl_config,
#         num_gpus: int | float,
#         dp_idx: int,
#         bundle_indices: list[int] | None,
#     ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
#         """
#         Provides complete worker configuration (resource assignment, init args and environment variables)
#         for vLLM tensor and pipeline parallelism.

#         Args:
#             config (DictConfig): The configuration for the worker.
#             psrl_config (DictConfig): The PSRL configuration.
#             num_gpus (int | float): The number of GPUs available for the worker.
#             dp_idx (int): The data parallel index of the worker.
#             bundle_indices (list[int]): Indices of the bundles to which this worker belongs.

#         Returns:
#             tuple: A tuple containing:
#                 - resources (dict[str, Any]): Resources assigned to the worker.
#                 - env_vars (dict[str, str]): Environment variables for the worker.
#                 - init_kwargs (dict[str, Any]): Initialization arguments for the worker.
#         """
#         resources: dict[str, Any] = {}
#         init_kwargs: dict[str, Any] = {}
#         env_vars: dict[str, str] = {}

#         # Nothing to do for SPMD-style synchronous rollout engines
#         if config is not None and hasattr(config, "rollout") and config.rollout.mode == "sync":
#             return resources, env_vars, init_kwargs

#         resources["num_gpus"] = num_gpus
#         psrl_logger.info(f"Configuring PSRL RewardModelWorker({num_gpus=}, {dp_idx=}, {bundle_indices=})...")

#         # Initialize configuration
#         if bundle_indices is not None:
#             bundle_id = bundle_indices[0] // len(bundle_indices)
#             # NOTE: bundle_id is 0 if we prepare pg for each dp manually
#             seed = dp_idx + 1000 + bundle_id

#             init_kwargs["seed"] = seed
#             # Need to give each DP group its own vllm cache to address:
#             # https://github.com/vllm-project/vllm/issues/18851
#             env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(f"~/.cache/vllm/vllm_{seed}")

#         # Check if this worker is part of a parallel group (TP or TP+PP).
#         is_part_of_parallel_workers = (
#             bundle_indices is not None and len(bundle_indices) > 1
#         ) or bundle_indices is None

#         # Leave the GPU assignment management of inner parallel workers to vLLM + Ray
#         if is_part_of_parallel_workers:
#             resources["num_gpus"] = 0
#             resources["num_cpus"] = 0
#             env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
#             env_vars["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
#             if bundle_indices is not None:
#                 # TODO(zyf): fix this problem
#                 local_ids = [x % 8 for x in bundle_indices]
#                 env_vars["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, local_ids))
#                 env_vars["WORLD_SIZE"] = str(len(bundle_indices))
#         env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
#         env_vars["VLLM_SKIP_P2P_CHECK"] = "1"
#         if config.rollout.disable_attn:
#             warnings.warn(
#                 "CAUTION: you are disabling the attention, "
#                 "this should only be used for analysis purposes, not for training!",
#                 stacklevel=2,
#             )
#             env_vars["VLLM_DISABLE_ATTN"] = "1"
#         return resources, env_vars, init_kwargs

#     def __init__(
#         self,
#         config: DictConfig,
#         psrl_config: DictConfig,
#         gen_interface: GenInterface,
#         reward_model_name: str | None = None,
#         **kwargs,
#     ) -> None:
#         """
#         Initialize the PSRL RewardModelWorker.

#         Args:
#             config (DictConfig): The configuration for the worker.
#             psrl_config (DictConfig): The PSRL configuration.
#             gen_interface (GenInterface): Same shape as rollout (rollout_instance_id, status_queue); ps_manager_handle omitted.
#             **kwargs: Additional keyword arguments, including 'seed'.
#         """
#         super().__init__()
#         self.config = config
#         self.psrl_config = psrl_config
#         self.gen_interface = gen_interface
#         self.status_queue = gen_interface.status_queue
#         self.reward_model_name = reward_model_name
#         self.seed = kwargs.get("seed", 0)
#         self.instance_id = kwargs.get("instance_id", gen_interface.rollout_instance_id)
#         self.rollout: PSRL_vLLMRollout | None = None
#         self.rollout_config: RolloutConfig | None = None
#         self.model_config: HFModelConfig | None = None

#         # Rollout loop management
#         self._generate_loop = asyncio.get_running_loop()  # Background async loop for generation
#         self.gen_task = None  # Generation task in stream mode
#         self._generate_thread = None  # Generation thread in batch mode
#         self._rollout_running = False
#         self.active_tasks = set()  # Active tasks for the current generation loop

#         # Async event management
#         self._is_init_model = asyncio.Event()
#         self._async_interrupt_event = asyncio.Event()
#         self._async_resume_event = asyncio.Event()

#         self.request_num_queue = queue.Queue()
#         self.request_id_to_active_tasks: dict[int, set[asyncio.Task]] = defaultdict(lambda: set())

#         # Track start time for active tasks logging
#         self.active_tasks_start_time = None

#         if self.is_instance_representative_rank:
#             logging_path = getattr(self.psrl_config, "logging_path", None)
#             if logging_path:
#                 self.log_prefix = f"RewardModelWorker_I{self.instance_id}_R{getattr(self, 'rank', 0)}"
#                 psrl_logger.addHandler(DualOutputHandler(logging_path, self.log_prefix))
#         psrl_logger.info(
#             "Initialized RewardModelWorker(instance=%s, rank=%s, seed=%s) on %s",
#             self.instance_id,
#             getattr(self, "rank", 0),
#             self.seed,
#             get_worker_info(),
#         )

#     def _build_distributed(self):
#         """Build the distributed process group for the rollout instance."""
#         # Initialize the distributed process group
#         if not dist.is_initialized():
#             is_cuda_available = torch.cuda.is_available()
#             rank = int(os.environ.get("RANK", 0))
#             world_size = int(os.environ.get("WORLD_SIZE", 1))
#             dist.init_process_group(
#                 backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl",
#                 rank=rank,
#                 world_size=world_size,
#             )

#     @property
#     def is_instance_representative_rank(self) -> bool:
#         return getattr(self, "rank", 0) == 0

#     def get_instance_id(self) -> int:
#         return self.gen_interface.rollout_instance_id

#     def get_node_id(self) -> str:
#         return ray.get_runtime_context().get_node_id()

#     def get_runtime_gpu_ids(self) -> list[int]:
#         accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
#         raw_gpu_ids = accelerator_ids.get("GPU", accelerator_ids.get("NPU", []))
#         return [int(gpu_id) for gpu_id in raw_gpu_ids]

#     # @ray.method(concurrency_group="control")
#     async def sleep(self):
#         self._ensure_model_ready()

#         psrl_logger.info(f"Interrupting generation on instance {self.get_instance_id()} (Double check)")
#         interrupted_request_num = await self.interrupt_generation()
#         psrl_logger.info(f"Interrupted {interrupted_request_num} requests on instance {self.get_instance_id()}")

#         await self.rollout.inference_engine.sleep(level=1)
#         psrl_logger.info(f"Reward model {self.reward_model_name} instance {self.instance_id} sleeping.")

#     # @ray.method(concurrency_group="control")
#     async def wake_up(self):
#         self._ensure_model_ready()
#         await self.rollout.inference_engine.wake_up()
#         psrl_logger.info(f"Reward model {self.reward_model_name} instance {self.instance_id} waking up.")

#         self.resume_generation()
#         psrl_logger.info(f"Resumed generation on instance {self.get_instance_id()}")

#     def _build_rollout(self, trust_remote_code: bool = False) -> PSRL_vLLMRollout:
#         """
#         Build the rollout engine and sharding manager for the PSRL RewardModelWorker.
#         NOTE: This method only supports building for one rollout instance at a time.
#         """
#         rollout_name = self.config.rollout.name
#         assert rollout_name == "vllm", "Only support vLLM rollout for now"
#         try:
#             rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
#             model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
#         except Exception as e:
#             psrl_logger.error(f"Failed to parse rollout config or model config: {e}")
#             raise
#         self.model_config = model_config
#         tp = self.config.rollout.get("tensor_model_parallel_size", 1)
#         pp = self.config.rollout.get("pipeline_model_parallel_size", 1)
#         assert self.world_size == tp * pp, "Only support dp=1 for now"

#         if self.config.rollout.mode == "sync":
#             self._build_distributed()
#         elif self.config.rollout.mode == "psrl_async":
#             # NOTE(lhy): No need to build distributed for psrl_async mode
#             # vllm will handle the distributed communication internally
#             pass
#         else:
#             raise ValueError(f"Invalid rollout mode: {self.config.rollout.mode}")

#         # Build the rollout engine
#         local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
#         # Get the tokenizer
#         self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
#         self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

#         # Get the model config
#         self.model_hf_config = AutoConfig.from_pretrained(
#             local_path, trust_remote_code=trust_remote_code, attn_implementation="flash_attention_2"
#         )
#         # patch for kimi-vl
#         if getattr(self.model_hf_config, "model_type", None) == "kimi_vl":
#             self.model_hf_config.text_config.topk_method = "greedy"
#         override_config_kwargs = {
#             "bos_token_id": self.tokenizer.bos_token_id,
#             "eos_token_id": self.tokenizer.eos_token_id,
#             "pad_token_id": self.tokenizer.pad_token_id,
#         }
#         override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
#         if isinstance(override_model_config, dict) and "model_config" in override_model_config:
#             override_config_kwargs.update(override_model_config["model_config"])  # Megatron model style
#         else:
#             override_config_kwargs.update(override_model_config)  # FSDP model style
#         update_model_config(self.model_hf_config, override_config_kwargs=override_config_kwargs)
#         if getattr(self, "rank", 0) == 0:
#             psrl_logger.info(f"Model config after override: {self.model_hf_config}")

#         get_torch_device().manual_seed(self.seed)

#         psrl_logger.info(f"Building {rollout_name} rollout with seed {self.seed}.")
#         rollout = PSRL_vLLMRollout(
#             psrl_config=self.psrl_config,
#             config=rollout_config,
#             model_config=model_config,
#             seed=self.seed,
#             instance_id=self.get_instance_id(),
#             status_queue=self.status_queue,
#             reward_model_name=self.reward_model_name,
#             is_reward_model=True,
#             init_mode="full",
#         )
#         return rollout

#     def _ensure_model_ready(self) -> None:
#         if not self._is_init_model.is_set() or self.rollout is None:
#             raise RuntimeError("RewardModelWorker is not initialized. Call init_model() first.")

#     def _extract_sampling_params_dict(self, request: DataProto) -> dict[str, Any]:
#         """
#         Extract sampling params for vLLM rollout.

#         Preferred order:
#         1) `request.non_tensor_batch["sampling_params"]` (per-request override)
#         2) `request.meta_info["sampling_params"]`
#         3) fallback to `self.rollout.sampling_params` (rollout default)
#         """
#         # Try per-request override from non-tensor batch.
#         if "sampling_params" in request.non_tensor_batch:
#             sp = request.non_tensor_batch["sampling_params"]
#             # Usually non-tensor batch stores python objects inside numpy array (dtype=object).
#             if isinstance(sp, np.ndarray):
#                 if sp.size == 1:
#                     sp = sp.item()
#                 else:
#                     sp = sp[0]
#             if isinstance(sp, dict):
#                 return sp

#         # Try meta info override.
#         if "sampling_params" in request.meta_info and isinstance(request.meta_info["sampling_params"], dict):
#             return request.meta_info["sampling_params"]

#         # Fallback: use rollout default sampling params.
#         rollout_sp = getattr(self.rollout, "sampling_params", None)
#         if rollout_sp is None:
#             return {}
#         if isinstance(rollout_sp, dict):
#             return rollout_sp
#         # Common cases: vLLM SamplingParams has __dict__ with primitive fields.
#         if hasattr(rollout_sp, "to_dict"):
#             return rollout_sp.to_dict()
#         if hasattr(rollout_sp, "model_dump"):
#             return rollout_sp.model_dump()
#         return dict(vars(rollout_sp))

#     def get_active_task_num(self) -> int:
#         """
#         Get the number of active tasks.
#         """
#         return len(self.active_tasks)

#     def log_active_tasks(self, task_added: bool = False, task_done: bool = False):
#         """
#         Log the active tasks.
#         """
#         assert task_added ^ task_done, "Exactly one of task_added or task_done must be True"
#         active_task_num = self.get_active_task_num()
#         psrl_logger.debug(f"Active tasks: {active_task_num}")
#         if task_added and active_task_num == 1:
#             self.active_tasks_start_time = time.time()
#             log_begin_event(
#                 f"Streaming generate",
#                 psrl_logger,
#                 event_type=EventType.GEN,
#             )
#         if task_done and active_task_num == 0:
#             if self.active_tasks_start_time is not None:
#                 duration = time.time() - self.active_tasks_start_time
#                 log_end_event(
#                     f"Streaming generate",
#                     psrl_logger,
#                     event_type=EventType.GEN,
#                     duration=duration,
#                 )
#                 self.active_tasks_start_time = None
#         # if active_task_num % self.log_active_tasks_interval == 0:
#         #     log_single_event(
#         #         f"Active tasks: {active_task_num} ({active_task_num / self.avg_max_active_tasks_len * 100:.2f}%)",
#         #         psrl_logger,
#         #         event_type=EventType.OTHER,
#         #     )

#     async def _generate_async_task(self, request: DataProto):
#         """
#         An async task to generate sequences for a single request.

#         Args:
#             request (DataProto): The generation request.

#         Returns:
#             The processed generation result.
#         """
#         assert len(request) == 1, f"Expected request length to be 1, got {len(request)}"

#         rollout_instance_id = self.get_instance_id()

#         # Prepare the request for generation
#         meta_info = {
#             "eos_token_id": self.generation_config.eos_token_id
#             if self.generation_config is not None
#             else self.tokenizer.eos_token_id,
#             "pad_token_id": self.generation_config.pad_token_id
#             if self.generation_config is not None
#             else self.tokenizer.pad_token_id,
#         }
#         request.meta_info.update(meta_info)
#         request.non_tensor_batch["rollout_instance_id"] = np.array([rollout_instance_id] * len(request.batch))

#         # Start the generation
#         with log_dual_events("Reward model generate", psrl_logger, event_type=EventType.GEN):
#             sampling_params = self._extract_sampling_params_dict(request)
#             result = await self.rollout.generate_sequences_async(request, sampling_params)

#         assert len(result) == 1, (
#             f"Expected 1 output for single request, got {len(result)} outputs."
#         )

#         interrupted = result.non_tensor_batch["interrupted"][0]

#         if interrupted:
#             # Keep partial generation payload (raw_response_ids/response_unpadded_len)
#             # so router can requeue this request as continuation instead of restarting.
#             psrl_logger.info(
#                 f"Request {request.non_tensor_batch['uid'][0]} is interrupted (instance sleep), "
#                 "returning partial output for continuation"
#             )
#             return result
#         else:
#             psrl_logger.info(f"Request {request.non_tensor_batch['uid'][0]} is completed (finished generation)")

#         return result

#     def _create_task_done_callback(self, request_id: int):
#         # Remove from the active tasks tracker when the task is done
#         def task_done_callback(task):
#             self.request_id_to_active_tasks[request_id].discard(task)
#             self.active_tasks.discard(task)
#             self.log_active_tasks(task_done=True)

#         return task_done_callback

#     async def _async_interrupt_requests(self, request_ids=None) -> int:
#         """Interrupt queued/running requests in reward model engine.

#         If `request_ids` is None, interrupt all requests.
#         Otherwise only interrupt requests whose uid is in `request_ids`.
#         """
#         self._ensure_model_ready()
#         if not request_ids:
#             interrupted_request_num = await self.rollout.interrupt_all_requests_async()
#             psrl_logger.debug(f"Interrupted all {interrupted_request_num} reward-model requests")
#             return interrupted_request_num

#         normalized_request_ids: set[int] = set()
#         for request_id in request_ids:
#             try:
#                 normalized_request_ids.add(int(request_id))
#             except (TypeError, ValueError):
#                 psrl_logger.warning(f"Skip non-integer request ID during interrupt: {request_id!r}")

#         psrl_logger.info(f"Interrupting reward-model requests with IDs: {normalized_request_ids}")

#         if not normalized_request_ids:
#             return 0

#         request_tasks = set()
#         for request_id in normalized_request_ids:
#             if request_id in self.request_id_to_active_tasks:
#                 request_tasks.update(self.request_id_to_active_tasks[request_id])
#             else:
#                 psrl_logger.warning(f"Request ID {request_id} not found in active tasks.")
#         if request_tasks:
#             await self.rollout.interrupt_requests_async(normalized_request_ids)
#             psrl_logger.info(f"Interrupted reward-model requests with IDs: {normalized_request_ids}")
#         return len(request_tasks)

#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     async def interrupt_requests(self, request_ids):
#         """Interrupt specific reward-model requests."""
#         return await self._async_interrupt_requests(request_ids)

#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     async def interrupt_all_requests(self):
#         """Interrupt all reward-model requests."""
#         return await self._async_interrupt_requests()

#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     async def interrupt_generation(self):
#         """Interrupt reward-model generation: block new work, drain engine queue, join active tasks."""
#         self._async_interrupt_event.set()
#         self._async_resume_event.clear()

#         interrupted_request_num = await self.interrupt_all_requests()

#         await asyncio.gather(*self.active_tasks, return_exceptions=True)
#         self.active_tasks.clear()

#         return interrupted_request_num

#     @register(dispatch_mode=Dispatch.ONE_TO_ALL)
#     def resume_generation(self):
#         """Allow reward-model generation to proceed after interrupt_generation."""
#         self._async_resume_event.set()
#         self._async_interrupt_event.clear()

#     @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
#     # @ray.method(concurrency_group="control")
#     def init_model(self):
#         with log_dual_events("Initialize reward model", psrl_logger, event_type=EventType.INIT):
#             self.rollout = self._build_rollout()
#         self._is_init_model.set()

#     # @rollout_trace_op
#     async def generate_async(self, request: DataProto, consolidate: bool = True):
#         """
#         Async generation entry point. This covers the streaming RM inference path.
#         """
#         self._ensure_model_ready()
#         assert len(request) == 1, f"Expected request length to be 1, got {len(request)}"

#         if self._async_interrupt_event and self._async_interrupt_event.is_set():
#             psrl_logger.debug("Reward-model generation interrupted, waiting for resume...")
#             await self._async_resume_event.wait()
#             psrl_logger.debug("Reward-model generation resumed")

#         request_id = int(request.non_tensor_batch["uid"][0])

#         task = self._generate_loop.create_task(self._generate_async_task(request))
#         task.add_done_callback(
#             self._create_task_done_callback(request_id)
#         )
#         self.request_id_to_active_tasks[request_id].add(task)
#         self.active_tasks.add(task)
#         self.log_active_tasks(task_added=True)
#         # Wait for the task to finish
#         result = await task
#         return result
