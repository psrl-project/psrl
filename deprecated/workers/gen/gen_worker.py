import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import os
import time
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import ray
import requests
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from ray.util.queue import Queue as RayQueue
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoConfig
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer, omega_conf_to_dataclass
from verl.utils.device import get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import get_generation_config, update_model_config

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.kv_cache import KVCacheManager
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    get_worker_info,
    log_begin_event,
    log_dual_events,
    log_end_event,
    log_single_event,
)
from psrl.utils.nixl import (
    NIXLInterface,
    resolve_weight_fingerprint_options,
    weight_fingerprint_flow_enabled,
)
from psrl.utils.ray import shared_pull_model_context_async
from psrl.utils.rollout.request_id import (
    normalize_request_ids_for_vllm_abort,
    parse_psrl_uid_from_request_id,
)
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.config import HFModelConfig, RolloutConfig, RolloutInstanceId
from psrl.workers.gen import PSRL_TransformersRollout, PSRL_vLLMRollout
from psrl.workers.gen.engine_http_server import EngineHttpServer
from psrl.workers.gen.utils import TokenInput, TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))
vllm_rollout_logger = logging.getLogger(PSRL_vLLMRollout.__module__)
vllm_rollout_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class GenInterface:
    """Info for the PSRL GenWorker and reward-model workers aligned with rollout."""

    rollout_instance_id: RolloutInstanceId
    status_queue: RayQueue
    ps_manager_handle: ray.actor.ActorHandle | None = None


class PSRL_GenWorker(Worker):
    _PARTIAL_ROLLOUT_TRACE_CHUNK_KEY = "_psrl_partial_rollout_trace_chunk"

    def _partial_rollout_trace_enabled(self) -> bool:
        return self.role != "reward" and bool(
            OmegaConf.select(self.psrl_config, "partial_rollout.trace.enable", default=False)
        )

    @staticmethod
    def _partial_rollout_trace_sequence(value: Any) -> tuple[int, str]:
        """Return a stable, non-reversible summary for a per-request token sequence."""
        if value is None:
            return 0, "empty"
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == 1:
            value = value[0]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
            value[0], (list, tuple, np.ndarray)
        ):
            value = value[0]
        try:
            tokens = np.asarray(value, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError):
            return 0, "unavailable"
        if tokens.size == 0:
            return 0, "empty"
        return int(tokens.size), hashlib.sha256(tokens.tobytes()).hexdigest()[:16]

    @staticmethod
    def _partial_rollout_trace_length(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == 1:
            value = value[0]
        try:
            return int(np.asarray(value).reshape(-1).size)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _partial_rollout_trace_scalar(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(np.asarray(value).reshape(-1)[0])
        except (IndexError, TypeError, ValueError):
            return default

    async def _partial_rollout_trace_start(
        self,
        request: TokenInput,
        *,
        request_version: int,
        previous_instance: RolloutInstanceId | None,
    ) -> dict[str, Any] | None:
        if not self._partial_rollout_trace_enabled():
            return None

        prefix_tokens, prefix_sha256 = self._partial_rollout_trace_sequence(
            request.response_ids
        )
        chunk = self._partial_rollout_trace_scalar(
            request.chunk
        )
        try:
            current_ps_version = int(await self.gen_interface.ps_manager_handle.get_ps_model_version.remote())
        except Exception:
            # Diagnostics must not make a rollout request fail if the optional
            # PS-version snapshot races with shutdown/recovery.
            current_ps_version = -1
            psrl_logger.exception(
                "[PARTIAL_ROLLOUT_TRACE] failed to read current PS version uid=%s",
                request.request_id,
            )
        loaded_version = int(self.curr_rollout_instance_model_version)
        context = {
            "uid": request.request_id,
            "chunk": chunk,
            "partial": request.response_ids is not None,
            "request_version": int(request_version),
            "loaded_version": loaded_version,
            "current_ps_version": current_ps_version,
            "request_staleness": current_ps_version - int(request_version),
            "loaded_staleness": current_ps_version - loaded_version,
            "instance": int(self.get_instance_id()),
            "previous_instance": previous_instance,
            "prefix_tokens": prefix_tokens,
            "prefix_sha256": prefix_sha256,
            "prior_logprob_tokens": self._partial_rollout_trace_length(
                request.rollout_log_probs
            ),
        }
        psrl_logger.warning(
            "[PARTIAL_ROLLOUT_TRACE] stage=generate_start uid=%s chunk=%s partial=%s "
            "request_version=%s loaded_version=%s current_ps_version=%s "
            "request_staleness=%s loaded_staleness=%s instance=%s previous_instance=%s "
            "prefix_tokens=%s prefix_sha256=%s prior_logprob_tokens=%s",
            context["uid"],
            context["chunk"],
            context["partial"],
            context["request_version"],
            context["loaded_version"],
            context["current_ps_version"],
            context["request_staleness"],
            context["loaded_staleness"],
            context["instance"],
            context["previous_instance"] if context["previous_instance"] is not None else "none",
            context["prefix_tokens"],
            context["prefix_sha256"],
            context["prior_logprob_tokens"],
        )
        return context

    def _partial_rollout_trace_result(
        self,
        result: TokenOutput,
        context: dict[str, Any] | None,
    ) -> None:
        if context is None:
            return
        response_tokens, response_sha256 = self._partial_rollout_trace_sequence(
            result.response_ids
        )
        interrupted = result.interrupted
        interrupted_by_scheduler = result.interrupted_by_scheduler
        generated_tokens = response_tokens - context["prefix_tokens"]
        logprob_tokens = self._partial_rollout_trace_length(result.response_log_probs)
        psrl_logger.warning(
            "[PARTIAL_ROLLOUT_TRACE] stage=generate_result uid=%s chunk=%s partial=%s "
            "request_version=%s loaded_version=%s instance=%s prefix_tokens=%s "
            "response_tokens=%s generated_tokens=%s response_sha256=%s "
            "rollout_logprob_tokens=%s interrupted=%s interrupted_by_scheduler=%s",
            context["uid"],
            context["chunk"],
            context["partial"],
            context["request_version"],
            context["loaded_version"],
            context["instance"],
            context["prefix_tokens"],
            response_tokens,
            generated_tokens,
            response_sha256,
            logprob_tokens,
            interrupted,
            interrupted_by_scheduler,
        )
        if interrupted or interrupted_by_scheduler:
            result.extra_fields[self._PARTIAL_ROLLOUT_TRACE_CHUNK_KEY] = int(context["chunk"]) + 1
        else:
            result.extra_fields.pop(self._PARTIAL_ROLLOUT_TRACE_CHUNK_KEY, None)

    def _log_sleep_wake_timing(self, operation: str, stage: str, started_at: float, **details: Any) -> None:
        detail_text = " ".join(f"{key}={value}" for key, value in details.items())
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=gen_worker operation=%s stage=%s "
            "instance=%s role=%s elapsed_s=%.6f%s%s",
            operation,
            stage,
            self.get_instance_id(),
            self.role,
            time.perf_counter() - started_at,
            " " if detail_text else "",
            detail_text,
        )

    def _log_rollout_weight_fingerprints(self, pull_results: Any, *, model_version: int) -> None:
        """Persist EngineCore fingerprint records through the GenWorker logger."""
        if not isinstance(pull_results, (list, tuple)) or not pull_results:
            raise RuntimeError(
                "Rollout weight verification expected one result per vLLM worker, "
                f"but got {type(pull_results).__name__}: {pull_results!r}"
            )

        expected_records = (
            ("raw_fingerprint", "rollout_after_raw_pull"),
            ("final_fingerprint", "rollout_after_param_sync"),
        )
        for worker_index, worker_result in enumerate(pull_results):
            if not isinstance(worker_result, dict):
                raise RuntimeError(
                    "Rollout weight verification received an invalid vLLM worker result: "
                    f"worker_index={worker_index}, result={worker_result!r}"
                )
            if int(worker_result.get("model_version", -1)) != model_version:
                raise RuntimeError(
                    "Rollout weight verification version mismatch in vLLM worker result: "
                    f"worker_index={worker_index}, expected={model_version}, "
                    f"actual={worker_result.get('model_version')!r}"
                )
            for result_key, expected_stage in expected_records:
                record = worker_result.get(result_key)
                if not isinstance(record, dict):
                    raise RuntimeError(
                        "Rollout weight verification did not return a fingerprint record: "
                        f"worker_index={worker_index}, result_key={result_key}, record={record!r}"
                    )
                if record.get("stage") != expected_stage or int(record.get("model_version", -1)) != model_version:
                    raise RuntimeError(
                        "Rollout weight verification returned inconsistent fingerprint metadata: "
                        f"worker_index={worker_index}, expected_stage={expected_stage}, "
                        f"expected_version={model_version}, record={record!r}"
                    )
                psrl_logger.warning("[WEIGHT_FINGERPRINT] %s", json.dumps(record, sort_keys=True))

    async def _log_tp_worker_tms_timing(self, operation: str) -> None:
        """Pull TP-local TMS stages into the GenWorker log."""
        if not os.environ.get("PSRL_VLLM_PATCHES", "").startswith("TMS"):
            return
        started_at = time.perf_counter()
        try:
            snapshots = await self._collective_rpc("get_last_tms_sleep_wake_timing", args=(operation,))
            for snapshot in snapshots:
                timing = snapshot.get("timing")
                if timing is None:
                    psrl_logger.warning(
                        "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker_summary operation=%s "
                        "stage=missing instance=%s role=%s host=%s rank=%s tp_rank=%s pid=%s",
                        operation,
                        self.get_instance_id(),
                        self.role,
                        snapshot["hostname"],
                        snapshot["rank"],
                        snapshot["tp_rank"],
                        snapshot["pid"],
                    )
                    continue
                for stage in timing["stages"]:
                    psrl_logger.warning(
                        "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker_summary operation=%s "
                        "stage=%s tag=%s instance=%s role=%s host=%s rank=%s tp_rank=%s "
                        "pid=%s elapsed_s=%.6f",
                        operation,
                        stage["stage"],
                        stage["tag"],
                        self.get_instance_id(),
                        self.role,
                        snapshot["hostname"],
                        snapshot["rank"],
                        snapshot["tp_rank"],
                        snapshot["pid"],
                        stage["elapsed_s"],
                    )
        except Exception:
            psrl_logger.exception(
                "[VLLM_SLEEP_WAKE_TIMING] failed to collect TP timing operation=%s instance=%s role=%s",
                operation,
                self.get_instance_id(),
                self.role,
            )
        finally:
            self._log_sleep_wake_timing(operation, "tp_timing_collection", started_at)

    async def _log_tp_worker_stage_timing(self, operation: str, stage: str) -> None:
        """Pull worker-extension timing details into the GenWorker log."""
        started_at = time.perf_counter()
        try:
            snapshots = await self._collective_rpc("get_last_worker_stage_timing", args=(stage,))
            for snapshot in snapshots:
                timing = snapshot.get("timing")
                if timing is None:
                    psrl_logger.warning(
                        "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker_summary operation=%s "
                        "stage=%s status=missing instance=%s role=%s host=%s rank=%s tp_rank=%s pid=%s",
                        operation,
                        stage,
                        self.get_instance_id(),
                        self.role,
                        snapshot["hostname"],
                        snapshot["rank"],
                        snapshot["tp_rank"],
                        snapshot["pid"],
                    )
                    continue
                detail_text = " ".join(
                    f"{key}={value:.6f}" if isinstance(value, (int, float)) else f"{key}={value}"
                    for key, value in timing.items()
                )
                psrl_logger.warning(
                    "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker_summary operation=%s "
                    "stage=%s instance=%s role=%s host=%s rank=%s tp_rank=%s pid=%s %s",
                    operation,
                    stage,
                    self.get_instance_id(),
                    self.role,
                    snapshot["hostname"],
                    snapshot["rank"],
                    snapshot["tp_rank"],
                    snapshot["pid"],
                    detail_text,
                )
        except Exception:
            psrl_logger.exception(
                "[VLLM_SLEEP_WAKE_TIMING] failed to collect TP stage=%s instance=%s role=%s",
                stage,
                self.get_instance_id(),
                self.role,
            )
        finally:
            self._log_sleep_wake_timing(operation, "tp_stage_timing_collection", started_at, worker_stage=stage)

    @staticmethod
    def configure_worker(
        config,
        psrl_config,
        num_gpus: int | float,
        dp_idx: int,
        bundle_indices: list[int],
        role: str
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """
        Provides complete worker configuration (resource assignment, init args and environment variables)
        for vLLM tensor and pipeline parallelism.

        Args:
            config (DictConfig): The configuration for the worker.
            psrl_config (DictConfig): The PSRL configuration.
            num_gpus (int | float): The number of GPUs available for the worker.
            dp_idx (int): The data parallel index of the worker.
            bundle_indices (list[int]): Indices of the bundles to which this worker belongs.

        Returns:
            tuple: A tuple containing:
                - resources (dict[str, Any]): Resources assigned to the worker.
                - env_vars (dict[str, str]): Environment variables for the worker.
                - init_kwargs (dict[str, Any]): Initialization arguments for the worker.
        """
        resources: dict[str, Any] = {}
        init_kwargs: dict[str, Any] = {}
        env_vars: dict[str, str] = {}

        resources["num_gpus"] = num_gpus
        psrl_logger.info(f"Configuring PSRL GenWorker({num_gpus=}, {dp_idx=}, {bundle_indices=})...")

        # Initialize configuration
        if bundle_indices is not None:
            bundle_id = bundle_indices[0] // len(bundle_indices)
            # NOTE: bundle_id is 0 if we prepare pg for each dp manually
            seed = dp_idx + 1000 + bundle_id

            init_kwargs["seed"] = seed
            # Need to give each DP group its own vllm cache to address:
            # https://github.com/vllm-project/vllm/issues/18851
            env_vars["VLLM_CACHE_ROOT"] = os.path.expanduser(f"~/.cache/vllm/vllm_{seed}")

        # Check if this worker is part of a parallel group (TP or TP+PP).
        is_part_of_parallel_workers = (
            bundle_indices is not None and len(bundle_indices) > 1
        ) or bundle_indices is None

        # Leave the GPU assignment management of inner parallel workers to vLLM + Ray
        if is_part_of_parallel_workers:
            resources["num_gpus"] = 0
            resources["num_cpus"] = 0
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            env_vars["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            # TODO(zyf): fix this problem
            local_ids = [x % 8 for x in bundle_indices]
            env_vars["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, local_ids))
            env_vars["WORLD_SIZE"] = str(len(bundle_indices))
        env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env_vars["VLLM_SKIP_P2P_CHECK"] = "1"
        # NOTE(linsh): Expandable segments are not compatible with
        # memory pool of sleep mode in vLLM.
        # Please track https://github.com/pytorch/pytorch/issues/147851 for more infos.
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

        if role in ["rollout", "validate", "reward"]:
            # use tms for memory management of model weights and kv cache
            if psrl_config.tms.range == "all" or psrl_config.tms.enable_nixl:
                import torch_memory_saver  # noqa: F401

                dynlib_path = os.path.join(
                    os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                    "torch_memory_saver_hook_mode_preload.abi3.so",
                )
                assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."
                env_vars["LD_PRELOAD"] = dynlib_path
                env_vars["TMS_INIT_ENABLE"] = "0"
                env_vars["TMS_INIT_ENABLE_CPU_BACKUP"] = "0"

            if role != "reward" and psrl_config.tms.enable_cuda_graph:
                env_vars["PSRL_VLLM_PATCHES"] = "TMS:GRAPH"
            elif psrl_config.tms.range == "all":
                env_vars["PSRL_VLLM_PATCHES"] = "TMS"

        arena_config = psrl_config.nixl.get("weight_arena", None)
        rollout_arena_enabled = arena_config is not None and arena_config.get("rollout_enabled", False)
        reward_arena_enabled = arena_config is not None and arena_config.get("reward_enabled", False)
        role_arena_enabled = (role in ["rollout", "validate"] and rollout_arena_enabled) or (
            role == "reward" and reward_arena_enabled
        )
        if role_arena_enabled:
            if role != "reward" and psrl_config.ps_mode not in ("nixl_cpu", "nixl_gpu"):
                raise RuntimeError(
                    f"psrl.nixl.weight_arena for role={role} requires psrl.ps_mode to be nixl_cpu or nixl_gpu."
                )
            if psrl_config.tms.range != "all":
                raise RuntimeError(f"psrl.nixl.weight_arena for role={role} requires psrl.tms.range=all.")
            env_vars["PSRL_VLLM_WEIGHT_ARENA"] = "1"
            # vLLM's Ray executor forwards VLLM_* variables to its inner GPU
            # workers. Keep the PSRL name for the outer worker and mirror it
            # so the patch is applied in the process that owns GPUModelRunner.
            env_vars["VLLM_PSRL_WEIGHT_ARENA"] = "1"

        if config.rollout.disable_attn:
            warnings.warn(
                "CAUTION: you are disabling the attention, "
                "this should only be used for analysis purposes, not for training!",
                stacklevel=2,
            )
            env_vars["VLLM_DISABLE_ATTN"] = "1"
        return resources, env_vars, init_kwargs

    def __init__(
        self,
        config: DictConfig,
        role: str,
        psrl_config: DictConfig,
        gen_interface: GenInterface,
        nixl_interface: NIXLInterface | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the PSRL GenWorker.

        Args:
            config (DictConfig): The configuration for the worker.
            role (str): The role of the worker (e.g., "rollout").
            psrl_config (DictConfig): The PSRL configuration.
            gen_interface (GenInterface): The interface for generation.
            nixl_interface (NIXLInterface | None): The interface for NIXL storage.
            **kwargs: Additional keyword arguments, including 'seed'.
        """
        super().__init__()
        self.config = config
        self.role = role
        self.dtype = self.config.rollout.dtype
        self.psrl_config = psrl_config
        self.gen_interface = gen_interface
        self.nixl_interface = nixl_interface
        self.reward_model_name = kwargs.get("reward_model_name", None)
        self.rm_config = kwargs.get("rm_config", None)
        self.is_teacher_model = kwargs.get("is_teacher_model", False)
        self.instance_id = kwargs.get("instance_id", self.gen_interface.rollout_instance_id)
        self.instance_dist_group = None

        if self.psrl_config.rollout_coordination.redundant_rollout.enable:
            self.avg_max_active_tasks_len = (
                self.psrl_config.rollout_coordination.redundant_rollout.redundant_global_batch_size
                * self.psrl_config.rollout_coordination.redundant_rollout.redundant_rollout_n
                // self.psrl_config.deployment.n_rollout_instances
            )
        else:
            self.avg_max_active_tasks_len = (
                self.psrl_config.staleness_buffer_entries
                * self.psrl_config.rollout_n
                // self.psrl_config.deployment.n_rollout_instances
            )
        self.log_active_tasks_interval = max(self.avg_max_active_tasks_len // 8, 1)

        self._lora_rank = self.config.model.get("lora_rank", 0)
        self._is_lora = self._lora_rank > 0

        self.seed = kwargs.get("seed", 0)

        self.curr_rollout_instance_model_version = 0  # Current model version for the rollout instance

        # Rollout loop management
        self._generate_loop = asyncio.get_running_loop()  # Background async loop for generation
        self.gen_task = None  # Generation task
        self._rollout_running = False
        self.active_tasks = set()  # Active tasks for the current generation loop

        # Async event management
        self._is_init_model = asyncio.Event()
        self._is_init_nixl_client = asyncio.Event()
        self._async_interrupt_event = asyncio.Event()
        self._async_resume_event = asyncio.Event()

        # Task for version update
        self.version_update_task = None

        # Rollout request management
        self.request_queue = deque()
        self.version_to_task_num: dict[int, int] = {}
        self.request_id_to_active_tasks: dict[int, set[asyncio.Task]] = defaultdict(lambda: set())
        self.pending_version_requests: dict[int, list[TokenInput]] = defaultdict(lambda: [])

        # Version ordering management
        self.version_ready_events: dict[int, asyncio.Event] = {}  # Events for when a version is ready to execute

        # NIXL
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None

        # NIXL cache
        self._cached_ps_nixl_agent_names = None
        self._cached_ps_nixl_gen_storage_client_names = None

        # For async model pulling
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        # KV cache manager for trajectory-level operations.
        # Engine is attached after rollout initialization in `_build_rollout`.
        # NOTE(lhy): In DP mode (data_parallel_size > 1) you will see a
        # recurring LMCache ERROR log:
        #   "PrometheusLogger instance already created with different metadata"
        # This is a known third-party bug in LMCache's PrometheusLogger
        # singleton. Root cause: vLLM's DPEngineCoreProc.__init__ mutates
        # kv_transfer_config.engine_id in-place AFTER the scheduler connector
        # has already called PrometheusLogger.GetOrCreate(), so the second call
        # (from LMCacheStatsLogger) sees a different engine_id and triggers the
        # mismatch check.  role also differs between SCHEDULER and WORKER
        # connector inits.  Neither field is a Prometheus label; the singleton
        # is reused correctly and there is no functional impact.  The fix
        # belongs in third_party/LMCache/lmcache/observability.py (ignore
        # engine_id/role in the equality check), but touching third-party code
        # is avoided to keep diffs minimal.
        # NOTE(lhy): KVCacheManager is created inside PSRL_vLLMRollout (which
        # handles apply_env_vars + get_engine_kwargs before engine init).
        # GenWorker accesses it via self.kv_cache_manager after _build_rollout().
        self.kv_cache_manager = None

        # Build logger
        # Only the representative rank will build the logger
        if self.is_instance_representative_rank:
            if self.role == "reward":
                self.log_prefix = f"RewardModelWorker_I{self.get_instance_id()}_R{self.get_instance_local_rank()}"
            else:
                self.log_prefix = f"GenWorker_I{self.get_instance_id()}_R{self.get_instance_local_rank()}"
            worker_log_handler = DualOutputHandler(self.psrl_config.logging_path, self.log_prefix)
            psrl_logger.addHandler(worker_log_handler)
            if self.role == "reward" and not any(
                isinstance(handler, DualOutputHandler)
                and getattr(handler, "log_prefix", None) == self.log_prefix
                for handler in vllm_rollout_logger.handlers
            ):
                vllm_rollout_logger.addHandler(worker_log_handler)
            psrl_logger.info(f"Initialized on {get_worker_info()}.")

        # [Optional] expose this rollout engine via HTTP (OpenAI-compatible) and
        # register it to RolloutGateway for catch-all proxying.
        self._engine_http_server: EngineHttpServer | None = None
        self._engine_http_bind: dict[str, Any] | None = None

        # Populated by trainer (or other coordinator) after gateway starts.
        self._gateway_base_url: str | None = None

    async def _collective_rpc(self, method_name: str, args: tuple = ()):
        """Call a method via collective RPC."""
        assert self.rollout, "Rollout must be initialized before calling _collective_rpc."
        return await self.rollout.inference_engine.collective_rpc(
            method_name,
            args=args,
        )

    async def _preload_reward_weights_to_cpu_cache(self) -> None:
        """Prepare the reward checkpoint's CPU weight source on vLLM workers."""
        assert self.rollout, "Rollout must be initialized before preloading reward weights."
        assert self.rollout.inference_engine is not None, "Reward vLLM engine must be initialized."
        await self.rollout.inference_engine.collective_rpc(
            "preload_weights_to_cpu_cache",
            args=(self.config.model.path, self.config.rollout.load_format),
        )
        psrl_logger.info("Reward model CPU weight source prepared on instance %s.", self.get_instance_id())

    async def _load_reward_weights_from_cpu_cache(self) -> None:
        """Load reward model weights from the vLLM worker CPU cache to GPU."""
        assert self.rollout, "Rollout must be initialized before loading reward weights."
        assert self.rollout.inference_engine is not None, "Reward vLLM engine must be initialized."
        loaded_params = await self.rollout.inference_engine.collective_rpc(
            "load_weights_from_cpu_cache",
            args=(),
        )
        if loaded_params is None:
            raise RuntimeError("Reward model failed to load weights from CPU cache.")

    def _is_transformers_rollout_backend(self) -> bool:
        rollout_name = str(self.config.rollout.name).lower()
        return rollout_name in ("transformers", "hf")

    def _build_distributed(self):
        """Build the distributed process group for the rollout instance."""
        # Initialize the distributed process group
        if not dist.is_initialized():
            is_cuda_available = torch.cuda.is_available()
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            dist.init_process_group(
                backend=("cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl"),
                rank=rank,
                world_size=world_size,
            )

    async def estimate_max_model_len(self):
        """
        Estimate the max model length for the rollout instance.
        """
        await self._is_init_model.wait()
        assert self.rollout, "Rollout must be initialized before calling estimate_max_model_len."
        max_model_len = await self._collective_rpc("estimate_max_model_len", args=())
        return max_model_len

    async def get_weight_arena_info(self) -> list[dict[str, Any]]:
        """Return arena state from every vLLM GPU worker."""
        await self._is_init_model.wait()
        return await self._collective_rpc("get_weight_arena_info", args=())

    async def get_worker_stage_timing(self, stage: str) -> list[dict[str, Any]]:
        """Return a worker-extension timing snapshot from every TP rank."""
        await self._is_init_model.wait()
        return await self._collective_rpc("get_worker_stage_timing", args=(stage,))

    async def init_nixl_client(self):
        """
        Initialize the NIXL client.
        This is implemented via rpc call in the vLLM extension.
        """
        await self._is_init_model.wait()
        assert self.rollout, "Rollout must be initialized before calling init_nixl_client."
        psrl_logger.info("NIXL client initialization begin via rpc call.")
        await self._collective_rpc(
            "init_nixl_client",
            args=(
                self.psrl_config.nixl,
                self.nixl_interface,
                self.get_instance_id(),
                self.psrl_config.logging_path,
            ),
        )
        self._is_init_nixl_client.set()
        psrl_logger.info("NIXL client initialized via rpc call.")

    async def nixl_convert_params(self):
        """
        Convert the model parameters to unified state dict and sharding dict via NIXL.
        This is implemented via rpc call in the vLLM extension.
        """
        await self._is_init_model.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_convert_params."

        psrl_logger.info("NIXL convert params begin via rpc call.")
        await self._collective_rpc("nixl_convert_params", args=(self.config,))
        psrl_logger.info("NIXL convert params done via rpc call.")

    async def nixl_protocol(self, mode: str = "full"):
        """
        Register the state dict and sharding dict to the NIXL client.
        This is implemented via rpc call in the vLLM extension.
        """
        await self._is_init_model.wait()
        await self._is_init_nixl_client.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_protocol."
        psrl_logger.info("NIXL protocol begin via rpc call.")
        await self._collective_rpc("nixl_protocol", args=(self.config, mode))
        psrl_logger.info("NIXL protocol done via rpc call.")

    async def nixl_wake_up(self):
        """Wake up model weights and register for NIXL."""
        await self._is_init_nixl_client.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_wake_up."
        total_start = time.perf_counter()

        # init empty model
        stage_start = time.perf_counter()
        await self.wake_up()
        self._log_sleep_wake_timing("wake", "nixl_engine_wake", stage_start)
        # register local tensors
        stage_start = time.perf_counter()
        await self._collective_rpc("nixl_register_after_wake_up", args=())
        self._log_sleep_wake_timing("wake", "nixl_register_after_wake_up", stage_start)
        await self._log_tp_worker_stage_timing("wake", "nixl_register_after_wake_up")
        self._log_sleep_wake_timing("wake", "nixl_wake_total", total_start)

    async def nixl_sleep(self):
        """Deregister local tensors and put model weights to sleep state (free up GPU memory)."""
        await self._is_init_nixl_client.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_sleep."
        total_start = time.perf_counter()

        # Deregister while the current physical pages are still mapped.
        stage_start = time.perf_counter()
        await self._collective_rpc("nixl_deregister", args=())
        self._log_sleep_wake_timing("sleep", "nixl_deregister", stage_start)
        await self._log_tp_worker_stage_timing("sleep", "nixl_deregister")
        # Put model weights to sleep only after NIXL has released its registrations.
        stage_start = time.perf_counter()
        await self.sleep(log_gpu_memory=False)
        self._log_sleep_wake_timing("sleep", "nixl_engine_sleep", stage_start)
        stage_start = time.perf_counter()
        await self._log_sleep_gpu_memory("after_nixl_deregister")
        self._log_sleep_wake_timing("sleep", "memory_snapshot", stage_start, snapshot_stage="after_nixl_deregister")
        self._log_sleep_wake_timing("sleep", "nixl_sleep_total", total_start)

    async def _nixl_log_shard_info(self, stage: str, max_elements: int = 8):
        """Log NIXL shard info via vLLM extension for sleep/wake_up debugging."""
        label = f"I{self.get_instance_id()}_{stage}"
        await self._collective_rpc("nixl_log_shard_info", args=(label, max_elements))

    async def _log_sleep_gpu_memory(self, stage: str) -> None:
        """Log per-vLLM-worker CUDA memory without making sleep depend on diagnostics."""
        assert self.rollout, "Rollout must be initialized before logging GPU memory."
        try:
            snapshots = await self.rollout.inference_engine.collective_rpc(
                "get_gpu_memory_snapshot",
                timeout=30,
                args=(),
            )
            gib = 1024**3
            for snapshot in snapshots:
                psrl_logger.info(
                    "[VLLM_SLEEP_MEMORY] stage=%s instance=%s host=%s node_id=%s "
                    "rank=%s tp_rank=%s pid=%s device=cuda:%s gpu_uuid=%s gpu_name=%s "
                    "torch_allocated=%.2f GB torch_reserved=%.2f GB "
                    "device_used=%.2f GB device_free=%.2f GB device_total=%.2f GB",
                    stage,
                    self.get_instance_id(),
                    snapshot["hostname"],
                    snapshot["node_id"],
                    snapshot["rank"],
                    snapshot["tp_rank"],
                    snapshot["pid"],
                    snapshot["device"],
                    snapshot["device_uuid"],
                    snapshot["device_name"],
                    snapshot["torch_allocated_bytes"] / gib,
                    snapshot["torch_reserved_bytes"] / gib,
                    snapshot["device_used_bytes"] / gib,
                    snapshot["device_free_bytes"] / gib,
                    snapshot["device_total_bytes"] / gib,
                )
        except Exception:
            psrl_logger.exception(
                "[VLLM_SLEEP_MEMORY] failed to collect memory after stage=%s instance=%s",
                stage,
                self.get_instance_id(),
            )

    async def sleep(self, log_gpu_memory: bool = True):
        """Put model weights to sleep state (free up GPU memory)."""
        total_start = time.perf_counter()
        if self._is_transformers_rollout_backend():
            psrl_logger.info("Transformers rollout sleep is a no-op on instance %s.", self.get_instance_id())
            return
        psrl_logger.info(f"Interrupting generation on instance {self.get_instance_id()} (Double check)")
        stage_start = time.perf_counter()
        interrupted_request_num = await self.interrupt_generation()
        self._log_sleep_wake_timing(
            "sleep", "interrupt_generation", stage_start, interrupted_requests=interrupted_request_num
        )
        psrl_logger.info(f"Interrupted {interrupted_request_num} requests on instance {self.get_instance_id()}")

        stage_start = time.perf_counter()
        await self.rollout.inference_engine.sleep(level=2)
        self._log_sleep_wake_timing("sleep", "engine_sleep", stage_start)
        await self._log_tp_worker_tms_timing("sleep")
        if self.psrl_config.tms.range in ["rollout", "all"]:
            # NOTE(linsh): empty_cache is done in vLLM cumem, but not for TMS.
            # Here we do an aggressive empty cache for TMS.
            stage_start = time.perf_counter()
            aggressive_empty_cache(force_sync=True)
            self._log_sleep_wake_timing("sleep", "aggressive_empty_cache", stage_start)
        if log_gpu_memory:
            stage_start = time.perf_counter()
            await self._log_sleep_gpu_memory("after_sleep")
            self._log_sleep_wake_timing("sleep", "memory_snapshot", stage_start, snapshot_stage="after_sleep")
        self._log_sleep_wake_timing("sleep", "total", total_start)

    async def wake_up(self):
        """Wake up model weights."""
        total_start = time.perf_counter()
        if self._is_transformers_rollout_backend():
            self.resume_generation()
            psrl_logger.info("Transformers rollout wake_up is a no-op on instance %s.", self.get_instance_id())
            return
        if self.role == "reward":
            stage_start = time.perf_counter()
            await self.rollout.inference_engine.wake_up(tags=["weights"])
            self._log_sleep_wake_timing("wake", "engine_wake", stage_start, tags="weights")
            await self._log_tp_worker_tms_timing("wake")
            stage_start = time.perf_counter()
            await self._load_reward_weights_from_cpu_cache()
            self._log_sleep_wake_timing("wake", "load_reward_weights_from_cpu_cache", stage_start)
            await self._log_tp_worker_stage_timing("wake", "load_weights_from_cpu_cache")
            stage_start = time.perf_counter()
            await self.rollout.inference_engine.wake_up(tags=["kv_cache"])
            self._log_sleep_wake_timing("wake", "engine_wake", stage_start, tags="kv_cache")
            await self._log_tp_worker_tms_timing("wake")
            self.resume_generation()
            psrl_logger.info(f"Generation resumed on instance {self.get_instance_id()}")
            self._log_sleep_wake_timing("wake", "total", total_start)
            return
        wake_up_tags = ["weights", "kv_cache"]
        if self.psrl_config.tms.enable_cuda_graph:
            wake_up_tags.append("graph")
        stage_start = time.perf_counter()
        await self.rollout.inference_engine.wake_up(tags=wake_up_tags)
        self._log_sleep_wake_timing("wake", "engine_wake", stage_start, tags=",".join(wake_up_tags))
        await self._log_tp_worker_tms_timing("wake")
        self._log_sleep_wake_timing("wake", "total", total_start)

    async def shutdown_rollout_engine(self, timeout_s: float = 120) -> None:
        """Gracefully stop vLLM and its child workers before terminating this actor."""
        rollout = getattr(self, "rollout", None)
        inference_engine = getattr(rollout, "inference_engine", None)
        if inference_engine is not None:
            try:
                await inference_engine.collective_rpc("close_node_shared_weight_cache", args=())
            finally:
                inference_engine.shutdown(timeout=timeout_s)
                self.rollout = None
        else:
            self.rollout = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def is_rollout_engine_sleeping(self) -> bool:
        """True if vLLM engine is sleeping (weights released); coordinator should not SYNC these instances."""
        await self._is_init_model.wait()
        assert self.rollout is not None
        if self._is_transformers_rollout_backend():
            return False
        return await self.rollout.inference_engine.is_sleeping()

    async def nixl_update_local_info_to_ps(self, ps_worker_node_id_to_idxs: dict):
        """
        Update local NIXL info to the PS workers on the same node with this train worker.
        """
        await self._is_init_nixl_client.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_update_local_info_to_ps."
        await self._collective_rpc("nixl_update_local_info_to_ps", args=(ps_worker_node_id_to_idxs,))

    async def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        """
        Send local NIXL info to the destination NIXL agents.

        Args:
            dst_agent_names (str | list[str]): Destination NIXL agent names
        """
        await self._is_init_nixl_client.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_send_local_info_to."
        await self._collective_rpc("nixl_send_local_info_to", args=(dst_agent_names,))

    async def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for update infos from the storage client.

        Args:
            info_num (int): Number of infos to wait for
        """
        await self._is_init_nixl_client.wait()
        assert self.rollout, "Rollout must be initialized before calling nixl_wait_for_update_infos."
        await self._collective_rpc("nixl_wait_for_update_infos", args=(info_num,))

    # --- KV cache management methods (called by RolloutRouter) ---

    async def kv_get_cache_info(self, tokens: list[int]) -> dict | None:
        """
        Query GPU prefix-cache and LMCache backend usage for the given tokens.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            dict | None: Serialisable dict of `TrajectoryCacheInfo` fields, or
                None if the engine is not yet attached.
        """
        assert self.kv_cache_manager.is_attached, (
            "KVCacheManager engine is not attached. "
            "Call init_model() before calling kv_get_cache_info."
        )
        assert tokens, "tokens must be a non-empty list."
        info = await self.kv_cache_manager.get_cache_info(tokens)
        return asdict(info)

    async def kv_pin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Pin the cached prefix blocks/chunks for `tokens`.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all requested pins succeeded.
        """
        assert self.kv_cache_manager.is_attached, (
            "KVCacheManager engine is not attached. "
            "Call init_model() before calling kv_pin."
        )
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        return await self.kv_cache_manager.pin(tokens, targets)

    async def kv_unpin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Unpin the cached prefix blocks/chunks for `tokens`.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all requested unpins succeeded.
        """
        assert self.kv_cache_manager.is_attached, (
            "KVCacheManager engine is not attached. "
            "Call init_model() before calling kv_unpin."
        )
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        return await self.kv_cache_manager.unpin(tokens, targets)

    async def kv_clear_from_backend(self, tokens: list[int]) -> int:
        """
        Remove the cached prefix chunks for `tokens` from the LMCache backend.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of chunks removed.
        """
        assert self.kv_cache_manager.is_attached, (
            "KVCacheManager engine is not attached. "
            "Call init_model() before calling kv_clear_from_backend."
        )
        assert tokens, "tokens must be a non-empty list."
        return await self.kv_cache_manager.clear_from_backend(tokens)

    async def kv_transfer(
        self,
        tokens: list[int],
        src: tuple[str, str],
        dst: tuple[str, str],
        copy: bool = False,
    ) -> bool:
        """
        Transfer the cached prefix of a trajectory to another (instance, backend).

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            src (tuple[str, str]): Source `(lmcache_instance_id, backend_location)`.
            dst (tuple[str, str]): Destination `(lmcache_instance_id, backend_location)`.
            copy (bool): If True, keep data at source as well.

        Returns:
            bool: True if the Controller acknowledged the move request.
        """
        assert self.kv_cache_manager.is_attached, (
            "KVCacheManager engine is not attached. "
            "Call init_model() before calling kv_transfer."
        )
        assert tokens, "tokens must be a non-empty list."
        assert len(src) == 2, f"src must be a 2-tuple, got length {len(src)}."
        assert len(dst) == 2, f"dst must be a 2-tuple, got length {len(dst)}."
        result = await self.kv_cache_manager.transfer(tokens, src, dst, copy)
        if not result:
            err = getattr(self.kv_cache_manager, '_last_transfer_error', 'unknown')
            psrl_logger.warning(
                f"[KVTransfer] transfer() returned False: src={src}, dst={dst}, "
                f"tokens_len={len(tokens)}, enable_p2p={self.kv_cache_manager._config.enable_p2p}, "
                f"controller_url={getattr(self.kv_cache_manager, '_controller_url', 'NOT_SET')}, "
                f"error={err}"
            )
        return result

    async def kv_transfer_direct(
        self,
        tokens: list[int],
        src: tuple[str, str],
        dst: tuple[str, str],
        copy: bool = False,
    ) -> bool:
        """
        Transfer KV cache directly via ZMQ to local LMCacheWorker, bypassing Controller.

        This is the scalable alternative to `kv_transfer()`. Instead of routing
        through the centralized Controller HTTP endpoint, it sends MoveWorkerMsg
        directly to the local LMCacheWorker's ZMQ socket.

        Under burst scenarios with many concurrent transfers across different
        instances, this eliminates the Controller as a single-point bottleneck.
        Falls back to Controller path on failure.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            src (tuple[str, str]): Source `(lmcache_instance_id, backend_location)`.
            dst (tuple[str, str]): Destination `(lmcache_instance_id, backend_location)`.
            copy (bool): If True, keep data at source as well.

        Returns:
            bool: True if the transfer succeeded.
        """
        assert self.kv_cache_manager.is_attached, (
            "KVCacheManager engine is not attached. "
            "Call init_model() before calling kv_transfer_direct."
        )
        assert tokens, "tokens must be a non-empty list."
        assert len(src) == 2, f"src must be a 2-tuple, got length {len(src)}."
        assert len(dst) == 2, f"dst must be a 2-tuple, got length {len(dst)}."
        return await self.kv_cache_manager.transfer_direct(tokens, src, dst, copy)

    def kv_set_peer_registry(
        self, registry: dict[str, list[str]], worker_zmq_url: str | None = None
    ) -> None:
        """
        Set the peer registry for direct KV transfer bypass.

        Called by `RolloutCoordinator._broadcast_peer_registry()` after P2P init.
        Enables `kv_transfer_direct()` to bypass the centralized Controller. Forwards
        this worker's own global rank so the manager can target the destination's
        same-rank KV shard endpoint.

        Args:
            registry (dict[str, list[str]]): Maps lmcache_instance_id → rank-sorted
                list of peer_init_url (index = global rank).
            worker_zmq_url (str | None): ZMQ URL of local LMCacheWorker REP socket.
        """
        assert self.kv_cache_manager is not None, (
            "KVCacheManager not initialized. Call init_model() first."
        )
        self.kv_cache_manager.set_peer_registry(
            registry, worker_zmq_url, self.get_instance_local_rank()
        )

    def set_lmcache_controller_url(self, controller_url: str) -> None:
        """
        Receive the shared LMCache Controller URL broadcast from `RolloutCoordinator`.

        Called by `RolloutCoordinator.init_lmcache_p2p()` after the single shared
        Controller subprocess has started.  Forwards the URL to `KVCacheManager`
        so that subsequent `kv_transfer` calls can reach it.

        Also initializes the process-local HTTP client (httpx) so that
        `KVCacheManager.transfer()` can call `post()` to the Controller.

        Args:
            controller_url (str): Base URL of the shared Controller, e.g.
                `"http://10.0.0.1:9042"`.
        """
        assert self.kv_cache_manager is not None, (
            "KVCacheManager is not initialized. Call init_model() before set_lmcache_controller_url()."
        )
        from psrl.utils.common.http_utils import init_http_client

        # Initialize HTTP client in this GenWorker process if not already done.
        # server_concurrency=4 and rollout_engine_num=1 are sufficient for
        # Controller /move requests (low concurrency, one controller per job).
        init_http_client(server_concurrency=4, rollout_engine_num=1)
        self.kv_cache_manager.set_controller_url(controller_url)

    async def _consume_kv_cache_hash_queue(self, kv_cache_hash_queue) -> None:
        """
        Background task: consume GPU prefix cache hash snapshots from the EngineCore.

        Runs in the GenWorker event loop. The `StatCollector` (in the same process)
        pushes a hash snapshot to the `kv_cache_hash_queue` every ~100ms. We drain
        the queue and store the latest snapshot in `KVCacheManager`, overwriting any
        previous snapshot. This allows `kv_get_gpu_cache_info_local` to compute
        prefix-cache hits instantly without blocking on EngineCore.

        Args:
            kv_cache_hash_queue (queue.SimpleQueue): In-process queue from the
                `PSRL_vLLMRollout` instance.
        """
        import queue as _queue

        _lmcache_log_count = 0
        _lmcache_hash_verified = False
        while True:
            try:
                # Drain the queue and keep only the latest snapshot.
                latest_snapshot = None
                while True:
                    try:
                        latest_snapshot = kv_cache_hash_queue.get_nowait()
                    except _queue.Empty:
                        break
                if latest_snapshot is not None:
                    self.kv_cache_manager.update_gpu_cache_snapshot(latest_snapshot)
                    # LMCache backend snapshot piggybacked on same payload.
                    lmcache_snapshot = latest_snapshot.get("lmcache")
                    if lmcache_snapshot is not None:
                        self.kv_cache_manager.update_lmcache_backend_snapshot(lmcache_snapshot)
                        _lmcache_log_count += 1
                        # One-time hash consistency verification.
                        if not _lmcache_hash_verified and "hash_probe" in lmcache_snapshot:
                            _lmcache_hash_verified = True
                            expected = lmcache_snapshot["hash_probe"]
                            none_hash = lmcache_snapshot.get("none_hash", 0)
                            local_fn = self.kv_cache_manager._get_lmcache_hash_fn()
                            local_probe = local_fn((none_hash, (0, 1, 2, 3), ()))
                            if local_probe != expected:
                                psrl_logger.error(
                                    "[KVCacheConsumer] HASH MISMATCH! "
                                    f"EngineCore hash_probe={expected}, "
                                    f"GenWorker hash_probe={local_probe}. "
                                    "Python hash() is non-deterministic across processes. "
                                    "Set PYTHONHASHSEED=0 in env BEFORE launching Ray/Python. "
                                    "LMCache local cache-info will ALWAYS return 0 until fixed."
                                )
                            else:
                                psrl_logger.info(
                                    "[KVCacheConsumer] Hash consistency verified OK "
                                    f"(algo={lmcache_snapshot.get('hash_algo', 'N/A')})."
                                )
                        if _lmcache_log_count <= 3 or _lmcache_log_count % 100 == 0:
                            psrl_logger.info(
                                f"[KVCacheConsumer] LMCache snapshot received "
                                f"(#{_lmcache_log_count}): "
                                f"chunk_hash_set size={len(lmcache_snapshot.get('chunk_hash_set', []))}, "
                                f"hash_algo={lmcache_snapshot.get('hash_algo', 'N/A')}"
                            )
                    elif _lmcache_log_count == 0:
                        # Log once if lmcache key is missing from payload.
                        _lmcache_log_count = -1
                        psrl_logger.warning(
                            "[KVCacheConsumer] Snapshot received but no 'lmcache' key. "
                            f"Keys: {list(latest_snapshot.keys())}"
                        )
                # Yield control to avoid busy-spinning.
                await asyncio.sleep(0.05)
            except Exception as e:
                psrl_logger.warning(f"Error consuming kv_cache_hash_queue: {e!r}.")
                await asyncio.sleep(0.5)

    def get_node_id(self) -> str:
        """
        Get the node id of the rollout instance.
        """
        return ray.get_runtime_context().get_node_id()

    def get_runtime_gpu_ids(self) -> list[int]:
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        raw_gpu_ids = accelerator_ids.get("GPU", accelerator_ids.get("NPU", []))
        return [int(gpu_id) for gpu_id in raw_gpu_ids]

    def get_instance_representative_rank(self) -> int:
        """
        The representative rank is the rank 0 of the rollout instance in current implementation (i.e., DP=1).
        """
        return 0

    def get_instance_ranks(self) -> list[int]:
        """
        Get the ranks of the rollout instance.
        The rollout instance is all the ranks of the dist group in current implementation (i.e., DP=1).
        """
        return list(range(self.world_size))

    def get_instance_local_rank(self) -> int:
        """
        Get the local rank of the rollout instance.
        It is just the global rank in the current implementation (i.e., DP=1).
        """
        return self.rank

    def get_instance_local_tp_rank(self) -> int:
        """
        Get the local tp rank of the rollout instance.
        """
        tp_rank = self.rank % self.config.rollout.get("tensor_model_parallel_size", 1)
        return tp_rank

    def get_instance_id(self) -> int:
        """
        Get the ID of the rollout instance.
        It is given by the gen_interface and is an unique ID for the dist group in current implementation (i.e., DP=1).
        """
        return self.gen_interface.rollout_instance_id

    @property
    def is_instance_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the rollout instance in current implementation (i.e., DP=1).
        """
        return self.rank == self.get_instance_representative_rank()

    async def register_rollout_instance(self):
        """Register the rollout instance in the PS worker."""
        assert self.rollout, "Rollout must be initialized before calling register_rollout_instance."
        if hasattr(self, "_is_rollout_instance_registered"):
            return
        if self.is_instance_representative_rank:
            # Only the representative rank needs to register the rollout instance
            await self.gen_interface.ps_manager_handle.register_rollout_instance.remote(self.get_instance_id())
        self._is_rollout_instance_registered = True

    def _broadcast_val_from_representative_rank(self, val: Any | None = None) -> Any:
        # Use torch.distributed.broadcast_object_list for generic object broadcasting
        if self.instance_dist_group is None:
            # Create the instance distribution group if it doesn't exist
            self.instance_dist_group = dist.new_group(ranks=self.get_instance_ranks())
        # Create object list for broadcasting
        obj_list = [val]
        if self.is_instance_representative_rank:
            # Current rank is the representative rank, broadcast object to all instance ranks
            dist.broadcast_object_list(
                obj_list,
                src=self.get_instance_representative_rank(),
                group=self.instance_dist_group,
            )
            return val
        else:
            # Current rank is not the representative rank, receive object from representative rank
            dist.broadcast_object_list(
                obj_list,
                src=self.get_instance_representative_rank(),
                group=self.instance_dist_group,
            )
            return obj_list[0]

    async def _build_rollout(self, init_mode: str = "full", trust_remote_code=False):
        """
        Build the rollout engine and sharding manager for the PSRL GenWorker.

        Args:
            init_mode (str): The initialization mode for the model, either 'full' or 'empty'.
            trust_remote_code (bool): Whether to trust remote code when loading the model.

        NOTE: This method only supports building for one rollout instance at a time.
        """
        rollout_name = str(self.config.rollout.name).lower()
        supported_rollout_names = ("vllm", "transformers", "hf")
        assert rollout_name in supported_rollout_names, (
            f"Unsupported rollout backend {rollout_name!r}, expected one of {supported_rollout_names}."
        )
        assert init_mode in ["full", "empty"], "init_mode must be either 'full' or 'empty'"

        try:
            # NOTE(linsh): For validation (fused), we will use config in `train_actor_rollout_ref.rollout`.
            # For rollout, we will use config in `gen_actor_rollout_ref.rollout`.
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
            model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        except Exception as e:
            psrl_logger.error(f"Failed to parse rollout config or model config: {e}")
            raise
        self.model_config = model_config
        tp = self.config.rollout.get("tensor_model_parallel_size", 1)
        pp = self.config.rollout.get("pipeline_model_parallel_size", 1)
        assert self.world_size == tp * pp, "Only support dp=1 for now"

        # Build the rollout engine
        local_path = copy_to_local(self.config.model.path, use_shm=self.config.model.get("use_shm", False))
        # Get the tokenizer
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        # Get the model config
        self.model_hf_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
        )
        # patch for kimi-vl
        if getattr(self.model_hf_config, "model_type", None) == "kimi_vl":
            self.model_hf_config.text_config.topk_method = "greedy"
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        if isinstance(override_model_config, dict) and "model_config" in override_model_config:
            override_config_kwargs.update(override_model_config["model_config"])  # Megatron model style
        else:
            override_config_kwargs.update(override_model_config)  # FSDP model style
        update_model_config(self.model_hf_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            psrl_logger.info(f"Model config after override: {self.model_hf_config}")

        get_torch_device().manual_seed(self.seed)

        psrl_logger.info(f"Building {rollout_name} rollout with seed {self.seed}.")
        rollout_kwargs = dict(
            psrl_config=self.psrl_config,
            config=rollout_config,
            model_config=model_config,
            seed=self.seed,
            status_queue=self.gen_interface.status_queue,
            instance_id=self.get_instance_id(),
            nixl_interface=self.nixl_interface,
            is_validate=self.role == "validate",
            is_reward_model=self.role == "reward",
            is_teacher_model=self.is_teacher_model,
            reward_model_name=self.reward_model_name,
            rm_config=self.rm_config,
            init_mode=init_mode,
        )
        if rollout_name == "vllm":
            rollout = PSRL_vLLMRollout(**rollout_kwargs)
        else:
            if self.role != "reward" or not self.is_teacher_model:
                raise NotImplementedError("Transformers rollout only supports OPD teacher reward workers.")
            if pp != 1:
                raise NotImplementedError("Transformers rollout does not support pipeline parallelism.")
            ep = int(self.config.rollout.get("expert_parallel_size", 1))
            if ep > 1 and ep != tp:
                raise NotImplementedError(
                    "Transformers rollout uses a 1D EP mesh and requires "
                    f"expert_parallel_size == tensor_model_parallel_size, got {ep=} {tp=}."
                )
            self._build_distributed()
            rollout = PSRL_TransformersRollout(**rollout_kwargs)

        # Non-owner model-parallel ranks do not host the vLLM engine.
        if rollout.inference_engine is not None:
            # Don't keep the dummy data in memory.
            await rollout.inference_engine.reset_mm_cache()

        # Use the KVCacheManager created inside PSRL_vLLMRollout (which already
        # applied env vars and engine kwargs before engine init).
        self.kv_cache_manager = rollout.kv_cache_manager
        self.kv_cache_manager.attach_engine(rollout.inference_engine)
        psrl_logger.info("[KVCacheManager]: Engine attached after rollout initialization.")
        # Set per-instance ID so LMCache P2P uses psrl_instance_{id} as the identifier.
        self.kv_cache_manager.set_instance_id(self.get_instance_id())
        # Start background consumer for GPU prefix cache hash snapshots.
        if hasattr(rollout, "kv_cache_hash_queue"):
            self._kv_cache_hash_consumer_task = asyncio.ensure_future(
                self._consume_kv_cache_hash_queue(rollout.kv_cache_hash_queue)
            )

        return rollout

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def init_model(self, init_mode: str = "full"):
        """Initialize the model for the rollout engine.

        If init_mode is 'full', load the full model weights.
        If init_mode is 'empty', load dummy weights for faster initialization.

        Args:
            init_mode (str): The initialization mode, either 'full' or 'empty'.
        """
        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            self.rollout = await self._build_rollout(
                init_mode, trust_remote_code=self.config.model.get("trust_remote_code", False)
            )
            if self.role == "reward" and self.rollout.inference_engine is not None:
                await self._preload_reward_weights_to_cpu_cache()
                arena_config = self.psrl_config.nixl.get("weight_arena", {})
                reward_arena_enabled = arena_config.get("reward_enabled", False)
                if reward_arena_enabled or not self.psrl_config.deployment.elastic.enable:
                    # Arena-backed reward workers build their final CPU mirror
                    # during initialization so even the first elastic wake uses
                    # a few contiguous H2D copies rather than checkpoint loaders.
                    await self._load_reward_weights_from_cpu_cache()
        self._is_init_model.set()

        # Representative rank can start HTTP server after model built.
        # For other ranks, the server is not needed.
        if (
            self.psrl_config.server_rollout.enable
            and self.is_instance_representative_rank
            and not self._is_transformers_rollout_backend()
        ):
            await self._maybe_start_engine_http_server()

    async def init_and_register_model(self, init_mode: str = "full"):
        """Initialize and register the model for the rollout engine.

        If init_mode is 'full', load the full model weights.
        If init_mode is 'empty', load dummy weights for faster initialization.

        Args:
            init_mode (str): The initialization mode, either 'full' or 'empty'.
        """
        await self.init_model(init_mode)
        await self.register_rollout_instance()

    async def _maybe_start_engine_http_server(self) -> None:
        """Start in-process HTTP server and register it to gateway."""

        if self._engine_http_server is not None:
            return

        # If gateway isn't configured, do nothing.
        if not self._gateway_base_url:
            psrl_logger.warning(
                "Rollout gateway base URL not set; skipping engine HTTP server startup and registration."
            )
            return

        await self._is_init_model.wait()
        # Only support vLLM async rollout engine currently.
        engine = self.rollout.inference_engine

        # Reuse the rollout-cached OpenAI server args/config.
        args = self.rollout.server_args
        if args is None:
            raise RuntimeError("OpenAI server args not initialized on rollout")

        host = ray.util.get_node_ip_address().strip("[]")
        port = int(find_available_port(20000 + 17 * int(self.get_instance_id())))

        self._engine_http_server = EngineHttpServer(host, port, args, engine)
        bind = await self._engine_http_server.start()
        self._engine_http_bind = {"host": bind.host, "port": bind.port, "base_url": bind.base_url}

        # Register to gateway.
        response = requests.post(
            f"{self._gateway_base_url}/add_worker",
            json={
                "instance_id": int(self.get_instance_id()),
                "worker_url": bind.base_url,
            },
        )
        response.raise_for_status()

        psrl_logger.info(
            "Registered rollout engine HTTP endpoint instance_id=%s worker_url=%s to gateway=%s",
            int(self.get_instance_id()),
            bind.base_url,
            self._gateway_base_url,
        )

    def set_rollout_gateway_base_url(self, base_url: str | None):
        """Called by trainer to enable worker self-registration to the gateway."""

        self._gateway_base_url = base_url.rstrip("/") if base_url else None

    def _extract_sampling_params_dict(self, request: TokenInput) -> dict[str, Any]:
        """Extract sampling params for reward-model generation.

        Priority order:
        1. per-request ``sampling_params`` override
        2. reward model ``sampling_config``
        3. rollout default ``SamplingParams``
        """
        if request.sampling_params is not None:
            sp = request.sampling_params
            return sp

        return self.rollout.sampling_params.to_dict()

    def get_active_task_num(self) -> int:
        """
        Get the number of active tasks.
        """
        return len(self.active_tasks)

    def log_active_tasks(self, task_added: bool = False, task_done: bool = False):
        """
        Log the active tasks.
        """
        assert task_added ^ task_done, "Exactly one of task_added or task_done must be True"
        active_task_num = self.get_active_task_num()
        psrl_logger.debug(f"Active tasks: {active_task_num}")
        if task_added and active_task_num == 1:
            self.active_tasks_start_time = time.time()
            log_begin_event(
                f"Generate with model version {self.curr_rollout_instance_model_version}",
                psrl_logger,
                event_type=EventType.GEN,
            )
        if task_done and active_task_num == 0:
            duration = time.time() - self.active_tasks_start_time
            log_end_event(
                f"Generate with model version {self.curr_rollout_instance_model_version}",
                psrl_logger,
                event_type=EventType.GEN,
                duration=duration,
            )
        if active_task_num % self.log_active_tasks_interval == 0:
            log_single_event(
                f"Active tasks: {active_task_num} ({active_task_num / self.avg_max_active_tasks_len * 100:.2f}%)",
                psrl_logger,
                event_type=EventType.OTHER,
            )

    async def ray_pull_model_async(self) -> None:
        """
        Pull the model state dict from PS via CPU and update the rollout model weights.
        In 'cpu' mode, pull the full state dict (potential bottleneck for large models).
        In 'cpu_ref' mode, get the ray object_ref and await it (parallel, non-blocking for PS worker).
        """
        ps_manager_handle = self.gen_interface.ps_manager_handle

        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, pull the full state dict (PS worker will block on transfer)
                model_state_dict_cpu = await ps_manager_handle.pull_model_state_dict_cpu.remote(self.get_instance_id())
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, get the object_ref and await it (PS worker is non-blocking)
                object_ref = await ps_manager_handle.pull_model_state_dict_cpu_ref.remote(self.get_instance_id())
                model_state_dict_cpu = (
                    await object_ref
                )  # This blocks until the state dict is available in the object store
            # Load the model state dict to the vllm model
            # sharding will be handled automatically inside vllm
            # NOTE(linsh): transfer from CPU to GPU is handled inside vLLM extension function `load_weights`.
            params_to_load = [
                (
                    name,
                    (reduce_tensor(param.full_tensor()) if isinstance(param, DTensor) else reduce_tensor(param)),
                )
                for name, param in model_state_dict_cpu.items()
            ]
            if not self.psrl_config.profile.fix_weight:
                loaded_params = await self.rollout.inference_engine.collective_rpc(
                    "load_weights",
                    args=(params_to_load,),
                )
                if loaded_params is None:
                    raise RuntimeError(f"Worker failed to update weights. Result: {loaded_params}")
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")

    async def nixl_pull_model_async(self) -> None:
        """
        Pull the model state dict from PS via NIXL and update the rollout model weights.
        This is implemented via rpc call in the vLLM extension.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", (
            "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        )
        total_start = time.perf_counter()
        ps_manager_handle = self.gen_interface.ps_manager_handle
        metadata_start = time.perf_counter()
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = await ps_manager_handle.get_ps_nixl_agent_names.remote()
        if self._cached_ps_nixl_gen_storage_client_names is None:
            self._cached_ps_nixl_gen_storage_client_names = (
                await ps_manager_handle.get_ps_nixl_gen_storage_client_names.remote()
            )
        self._log_sleep_wake_timing("pull", "fetch_ps_metadata", metadata_start)
        model_version = -1
        fingerprint_options = None
        if self.role == "rollout" and weight_fingerprint_flow_enabled(
            self.psrl_config,
            flow="transfer_chain",
        ):
            model_version = int(await ps_manager_handle.get_ps_model_version.remote("rollout_weight_fingerprint"))
            fingerprint_options = resolve_weight_fingerprint_options(
                self.psrl_config,
                flow="transfer_chain",
                model_version=model_version,
            )
        if not self.psrl_config.profile.fix_weight:
            transfer_start = time.perf_counter()
            pull_results = await self.rollout.inference_engine.collective_rpc(
                "nixl_pull_model_core",
                args=(
                    self._cached_ps_nixl_agent_names,
                    self._cached_ps_nixl_gen_storage_client_names,
                    model_version,
                    fingerprint_options,
                ),
            )
            if fingerprint_options is not None:
                self._log_rollout_weight_fingerprints(pull_results, model_version=model_version)
            self._log_sleep_wake_timing("pull", "nixl_pull_model_core", transfer_start)
            await self._log_tp_worker_stage_timing("pull", "nixl_pull_model_core")
        version_start = time.perf_counter()
        await ps_manager_handle.pull_model_state_dict_nixl.remote(
            self.get_instance_id()
        )  # This only updates the model version
        self._log_sleep_wake_timing("pull", "update_ps_version", version_start)
        self._log_sleep_wake_timing("pull", "nixl_pull_total", total_start)
        psrl_logger.info("NIXL pull model done.")

    async def pull_model_async(self) -> None:
        assert len(self.active_tasks) == 0, f"Cannot pull model while there are {len(self.active_tasks)} active tasks"

        total_start = time.perf_counter()
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            await self.ray_pull_model_async()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            await self.nixl_pull_model_async()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        # Important: the prefix cache needs to be cleared after pulling the model
        reset_start = time.perf_counter()
        await self.rollout.inference_engine.reset_prefix_cache()
        self._log_sleep_wake_timing("pull", "reset_prefix_cache", reset_start)
        self._log_sleep_wake_timing("pull", "pull_model_total", total_start)
        # Clear the LMCache CPU backend so that stale KV entries from the previous
        # model version cannot be reused by subsequent requests.
        if self.kv_cache_manager.should_clear_on_weight_update:
            await self._collective_rpc("lmcache_clear_all_from_backend", args=())
            psrl_logger.debug("Cleared LMCache CPU backend after model weight update.")

    async def _async_interrupt_requests(self, request_ids=None):
        """Interrupt requests in the engine queue (waiting and running).

        If `request_ids` is None, it will interrupt all requests.
        If `request_ids` is provided, it will only interrupt the specified requests.

        Returns:
            int: The number of requests interrupted.
        """
        if not request_ids:
            # Interrupt all requests
            interrupt_request_num = await self.rollout.interrupt_all_requests_async()
            psrl_logger.debug(f"Interrupted all {interrupt_request_num} requests")
            return interrupt_request_num

        engine_request_ids = normalize_request_ids_for_vllm_abort(request_ids)
        request_tasks = set()
        missing_request_ids = []
        for request_id in engine_request_ids:
            lookup_key = parse_psrl_uid_from_request_id(request_id)
            if lookup_key in self.request_id_to_active_tasks:
                request_tasks.update(self.request_id_to_active_tasks[lookup_key])
            else:
                # Waiting-queue ids from scheduler stats are vLLM-internal strings;
                # active tasks are keyed by PSRL integer uid. Unmatched ids may still
                # exist only in the engine queue (not yet tracked locally).
                missing_request_ids.append(request_id)
        psrl_logger.debug(
            "Found %d active tasks for engine request IDs (sample=%s)",
            len(request_tasks),
            engine_request_ids[:10],
        )
        if missing_request_ids:
            psrl_logger.debug(
                "Interrupt engine ids not found in active tasks (count=%d, lookup_keys=%s); ids=%s",
                len(missing_request_ids),
                [parse_psrl_uid_from_request_id(rid) for rid in missing_request_ids[:10]],
                missing_request_ids[:10],
            )

        # Abort using vLLM internal request id strings (not PSRL uid integers).
        # The rollout returns the count actually removed from OutputProcessor;
        # this includes engine-queued requests that have no active Python task.
        interrupt_request_num = await self.rollout.interrupt_requests_async(engine_request_ids)
        psrl_logger.debug(
            "Interrupted %d/%d requests with IDs (active_tasks=%d, ids=%s)",
            interrupt_request_num,
            len(engine_request_ids),
            len(request_tasks),
            request_ids,
        )
        return interrupt_request_num

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def interrupt_requests(self, request_ids):
        """Interrupt specific requests in the engine queue (waiting and running)."""
        return await self._async_interrupt_requests(request_ids)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def interrupt_all_requests(self):
        """Interrupt all requests in the engine queue (waiting and running)."""
        return await self._async_interrupt_requests()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def interrupt_generation(self):
        """Interrupt the generation process."""
        # Interrupt creating generation tasks
        self._async_interrupt_event.set()
        self._async_resume_event.clear()

        # Interrupt all requests in the engine queue (waiting and running)
        interrupted_request_num = await self.interrupt_all_requests()

        # Wait and clean all tasks in self.active_tasks
        await asyncio.gather(*self.active_tasks, return_exceptions=True)
        self.active_tasks.clear()

        return interrupted_request_num

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def resume_generation(self):
        """Resume the generation process."""
        self._async_resume_event.set()
        self._async_interrupt_event.clear()

    # @ray.method(concurrency_group="control")
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def sync_with_ps(self, ps_version: int, interrupt_generation: bool = False, sync_after_wake_up: bool = False) -> int:
        """
        Synchronize the rollout instance with the parameter server.

        This method combines three operations into one:
        1. (Optional) Interrupt the current generation process
        2. Pull the latest model weights from the parameter server
        3. Resume the generation process

        Returns:
            int: The number of requests that were interrupted during the sync process.
        """
        if self.curr_rollout_instance_model_version >= ps_version and not sync_after_wake_up:
            psrl_logger.warning(
                f"No need to sync with PS for instance {self.get_instance_id()}, "
                f"current model version {self.curr_rollout_instance_model_version} "
                f"is greater than or equal to the required PS version {ps_version}"
            )
            return 0

        # Step 1: Interrupt generation
        if interrupt_generation:
            psrl_logger.info(f"Interrupting generation on instance {self.get_instance_id()}")
            interrupted_request_num = await self.interrupt_generation()
            psrl_logger.info(f"Interrupted {interrupted_request_num} requests on instance {self.get_instance_id()}")
        else:
            assert len(self.active_tasks) == 0, (
                "Should not have any active tasks when syncing with PS, "
                "please call `self.interrupt_generation()` in advance or "
                "set `interrupt_generation` to False"
            )

        # Step 2: Pull model
        async with shared_pull_model_context_async(self.gen_interface.ps_manager_handle):
            with log_dual_events("Pull model (partial rollout)", psrl_logger, event_type=EventType.PULL):
                await self.pull_model_async()

        # NOTE(lhy): The version obtained from the PS manager is the actual model version after the pull
        # It may be higher than the required version due to the pushing happens between waiting and pulling
        self.curr_rollout_instance_model_version = (
            await self.gen_interface.ps_manager_handle.get_rollout_instance_model_version.remote(
                self.get_instance_id()
            )
        )
        assert self.curr_rollout_instance_model_version >= ps_version, (
            f"Current rollout instance model version should not be less than the required PS version, "
            f"but got {self.curr_rollout_instance_model_version} vs. {ps_version}"
        )
        if self.curr_rollout_instance_model_version > ps_version:
            psrl_logger.warning(
                f"Actual model version after pull (partial rollout) is {self.curr_rollout_instance_model_version}, "
                f"which is higher than the required PS version {ps_version}"
            )
        if self.rollout.stat_collector is not None:
            self.rollout.stat_collector.record_model_version_update(self.curr_rollout_instance_model_version)

        # Step 3: Resume generation
        psrl_logger.info(f"Resuming generation on instance {self.get_instance_id()}")
        self.resume_generation()
        psrl_logger.info(f"Generation resumed on instance {self.get_instance_id()}")

    def _create_task_done_callback(self, request_id: int, require_version: int):
        # Remove from the active tasks tracker when the task is done
        def task_done_callback(task):
            self.request_id_to_active_tasks[request_id].discard(task)
            self.active_tasks.discard(task)
            self.log_active_tasks(task_done=True)

        return task_done_callback

    async def _generate_async_task(
        self,
        request: TokenInput,
        sampling_params: dict[str, Any] | None = None,
        needed_model_version: int | None = None,
    ):
        """
        An async task to generate sequences for a single request.
        This method handles the generation for a single request, managing model versioning
        and ensuring that the request is processed correctly.

        Args:
            request (DataProto): The generation request.
            sampling_params (dict): The sampling parameters for generation.
            needed_model_version (int): The model version required for this request.

        Returns:
            tuple: A tuple containing the generated sequences and the update status.
        """
        if self.role == "reward":
            request.rollout_instance_id = self.get_instance_id()

            with log_dual_events("Reward model generate", psrl_logger, event_type=EventType.GEN):
                reward_sampling_params = sampling_params or self._extract_sampling_params_dict(request)
                result = await self.rollout.generate_sequences_async(request, reward_sampling_params)

            if result.interrupted:
                psrl_logger.info(
                    "Request %s is interrupted (instance sleep), returning partial output for continuation",
                    request.request_id,
                )
            return result

        assert needed_model_version is not None, "needed_model_version is required for rollout/validate workers"
        assert sampling_params is not None, "sampling_params is required for rollout/validate workers"
        assert self.curr_rollout_instance_model_version >= needed_model_version, (
            f"Rollout model version should not be less than needed version, "
            f"but got {self.curr_rollout_instance_model_version} for needed {needed_model_version}"
        )

        # Update the request status to ROLLOUT_RUNNING
        is_validate = request.is_validate
        previous_instance = request.rollout_instance_id

        # Only update the model version if the request is prompt-only
        if request.raw_response_ids is not None:
            # Indicate it is a partial rollout request, use the original version tag in the request
            model_version = request.version_tag
        else:
            model_version = self.curr_rollout_instance_model_version
            if needed_model_version != model_version:
                psrl_logger.warning(
                    f"Update version_tag of request {request.request_id} "
                    f"from {needed_model_version} to {model_version} due to inconsistent model pull"
                )
                # Update version tag in staleness inventory
                await self.gen_interface.ps_manager_handle.update_request_version_tag.remote(
                    request.request_id, model_version, is_validate
                )
            request.version_tag = model_version

        # Update the request status to ROLLOUT_RUNNING
        update_status_success = await self.gen_interface.ps_manager_handle.update_request_status.remote(
            request.request_id,
            PSRL_RequestStatus.ROLLOUT_RUNNING,
            rollout_instance_id=request.rollout_instance_id,
            model_version=request.version_tag,
            is_validate=is_validate,
        )
        if update_status_success:
            # Prepare the request for generation
            trace_context = await self._partial_rollout_trace_start(
                request,
                request_version=model_version,
                previous_instance=previous_instance,
            )

            # Start the generation
            with log_dual_events(
                f"Core generation with model version {self.curr_rollout_instance_model_version}",
                psrl_logger,
                level=logging.DEBUG,
                event_type=EventType.GEN,
            ):
                vllm_outputs = await self.rollout.raw_generate_sequences_async(request, sampling_params)

            # Unpack the 4-tuple: (idx, RequestOutput, events, start_ts)
            raw_output = vllm_outputs[0] if isinstance(vllm_outputs, list) else vllm_outputs
            _, vllm_output, accumulated_events, gen_start_ts = raw_output
            assert len(vllm_output.outputs) == 1, (
                f"Expected no repeat in generation, got {len(vllm_output.outputs)} outputs."
            )

            result = self.rollout.post_process_outputs(
                request,
                vllm_output,
                accumulated_events=accumulated_events,
                generation_start_wall_ts=gen_start_ts,
            )

            interrupted = result.interrupted
            interrupted_by_scheduler = result.interrupted_by_scheduler

            # Update the request status to ROLLOUT_INTERRUPTED_BY_SCHEDULER or ROLLOUT_INTERRUPTED or RUNNING,
            if interrupted_by_scheduler:
                update_status = PSRL_RequestStatus.ROLLOUT_INTERRUPTED_BY_SCHEDULER
                psrl_logger.debug(f"Request {request.request_id} is interrupted by scheduler (preemption)")
            elif interrupted:
                update_status = PSRL_RequestStatus.ROLLOUT_INTERRUPTED
                psrl_logger.debug(f"Request {request.request_id} is interrupted (partial rollout / instance sleep)")
            else:
                update_status = PSRL_RequestStatus.ROLLOUT_COMPLETED
                psrl_logger.debug(f"Request {request.request_id} is completed (finished generation)")
            update_status_success = await self.gen_interface.ps_manager_handle.update_request_status.remote(
                request.request_id,
                update_status,
                is_validate=is_validate,
            )
            if update_status_success:
                return result, update_status
        # Means the request is aborted
        return None, None

    @rollout_trace_op
    async def generate_async(
        self,
        request: TokenInput,
        sampling_params: dict[str, Any] | None = None,
        consolidate: bool = True,
    ):
        """
        Generate sequences asynchronously.
        This method handles a single async generation request, managing model versioning
        and ensuring that the request is processed in a timely manner.

        Args:
            request (DataProto): The async generation request.
            sampling_params (dict): The sampling parameters for generation.
            consolidate (bool): Whether to consolidate the results after generation.
        """
        if self.role == "reward":
            if self._async_interrupt_event and self._async_interrupt_event.is_set():
                psrl_logger.debug("Generation interrupted, waiting for resume...")
                await self._async_resume_event.wait()
                psrl_logger.debug("Generation resumed")

            request_id = request.request_id
            task = self._generate_loop.create_task(
                self._generate_async_task(request, sampling_params=sampling_params)
            )
            task.add_done_callback(self._create_task_done_callback(request_id, require_version=-1))
            self.request_id_to_active_tasks[request_id].add(task)
            self.active_tasks.add(task)
            self.log_active_tasks(task_added=True)
            result = await task
            return result

        assert consolidate, (
            "Consolidate must be True for async generation for now. "
            "Because the postprocess is need to be done inside the vllm rollout "
            "to mark the requests that are interrupted by the scheduler."
        )

        psrl_logger.debug(
            f"Generating request {request.request_id} "
            f"with needed model version {request.version_tag}"
        )
        # Wait for resuming if the generation is interrupted
        if self._async_interrupt_event and self._async_interrupt_event.is_set():
            psrl_logger.debug("Generation interrupted, waiting for resume...")
            await self._async_resume_event.wait()
            psrl_logger.debug("Generation resumed")

        request_id = request.request_id
        needed_model_version = request.version_tag

        # The router should guarantee the request is assigned to a rollout instance
        # that can directly generate with the needed model version.
        assert needed_model_version <= self.curr_rollout_instance_model_version, (
            f"Needed model version {needed_model_version} should not be greater than "
            f"current rollout instance model version {self.curr_rollout_instance_model_version}."
        )

        # All the partial rollout requests (with version tag less than the current rollout
        # instance model version) should be updated to the current rollout instance model version
        if needed_model_version < self.curr_rollout_instance_model_version:
            psrl_logger.debug(
                f"Request {request_id} needed model version {needed_model_version} is less than "
                f"current rollout instance model version {self.curr_rollout_instance_model_version}, "
                f"we'll update needed model version to {self.curr_rollout_instance_model_version}."
            )
            needed_model_version = self.curr_rollout_instance_model_version

        task = self._generate_loop.create_task(
            self._generate_async_task(request, sampling_params, needed_model_version)
        )
        task.add_done_callback(
            self._create_task_done_callback(
                request.request_id,
                needed_model_version,
            )
        )
        self.request_id_to_active_tasks[request_id].add(task)
        self.active_tasks.add(task)
        self.log_active_tasks(task_added=True)
        # Wait for the task to finish
        result = await task
        return result
