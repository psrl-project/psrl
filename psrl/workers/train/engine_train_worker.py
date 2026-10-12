# Copyright (c) 2025, PSRL Authors.

import logging
import os
from contextlib import nullcontext

import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from transformers import AutoConfig
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.config import DistillationConfig
from verl.workers.engine_workers import ActorRolloutRefWorker

from psrl.utils.common.patch_utils import apply_tms_patch
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.param_sync import ParamSyncPlan

# Set an architecture before guarded Megatron imports on CPU workers.
if not is_cuda_available and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

try:
    from psrl.utils.converter.megatron_converter import convert_megatron_inplace  # noqa: E402
except ImportError:
    convert_megatron_inplace = None
from megatron.bridge.models.conversion.utils import unwrap_model  # noqa: PLC0415

if not is_cuda_available and os.environ.get("TORCH_CUDA_ARCH_LIST") == "8.0":
    del os.environ["TORCH_CUDA_ARCH_LIST"]

from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME  # noqa: E402
from psrl.utils.common.worker_naming import train_client_name  # noqa: E402
from psrl.utils.logger import (  # noqa: E402
    DualOutputHandler,
    EventType,
    MemoryLogger,
    get_worker_info,
    gpu_memory_logger_decorator,
    log_dual_events,
    log_tensor,
)
from psrl.utils.nixl import (  # noqa: E402
    NIXLClientType,
    NIXLStorageClient,
)
from psrl.utils.ray import exclusive_push_model_context  # noqa: E402
from psrl.workers.train.base_train_worker import PSRL_BaseTrainWorker, TrainInterface  # noqa: E402

try:
    from torch_memory_saver import torch_memory_saver
except ImportError:
    torch_memory_saver = None  # type: ignore

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_EngineTrainWorker(ActorRolloutRefWorker, PSRL_BaseTrainWorker):
    """Unified PSRL training worker backed by veRL's engine_workers API.

    Supports both FSDP and Megatron backends via the pluggable BaseEngine.
    Adds PSRL-specific features: NIXL weight streaming, parameter server push/pull,
    and async training via torch_memory_saver sleep/wake lifecycle.

    Usage:
        Instantiated by TaskRunner in main_ppo.py. Pass strategy via
        config.actor.strategy = "fsdp" | "fsdp2" | "megatron".
    """

    def __init__(
        self,
        config: DictConfig,
        role: str,
        psrl_config: DictConfig,
        train_interface: TrainInterface,
        distillation_config: DistillationConfig | None = None,
    ) -> None:
        ActorRolloutRefWorker.__init__(self, config, role, distillation_config)
        PSRL_BaseTrainWorker.__init__(
            self,
            self.rank,
            self.world_size,
            psrl_config,
            train_interface,
        )

        # TODO(linsh): Remove this strategy-specific configuration patch.
        if self.config.actor.strategy == "megatron":
            self.config.actor.megatron.use_per_rank_checkpoint = not self.psrl_config.checkpoint.use_dcp_save

        if self.psrl_config.tms.range in ["train", "all"]:
            if torch_memory_saver is None:
                raise ImportError("torch_memory_saver is required when tms.range is 'train' or 'all'")
            apply_tms_patch()

        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

        if torch.cuda.is_available() and self.psrl_config.memory_logger.enable:
            self.memory_logger = MemoryLogger(
                self.psrl_config.logging_path,
                f"{self.log_prefix}_Mem",
                interval_seconds=self.psrl_config.memory_logger.interval_seconds,
            )
            self.memory_logger.start()
        else:
            self.memory_logger = None

    @property
    def is_train_representative_rank(self) -> bool:
        """Return True for rank 0 (the rank that pushes to PS)."""
        return self.rank == 0

    def get_replica_id(self) -> int:
        """Return the data-parallel replica index via the engine's standard API.

        Must be called after init_model() when the engine is initialized.
        """
        assert self._is_actor and self.actor is not None, (
            "get_replica_id() must be called after init_model() has built the actor."
        )
        return self.actor.engine.get_data_parallel_rank()

    def init_nixl_client(self):
        """Initialize the NIXL GPU-to-GPU weight-streaming client."""
        # DCP `all_gather_object` can corrupt UCX endpoint addresses read by the
        # background progress thread, so disable that thread during DCP saves.
        enable_prog_thread = not self.psrl_config.checkpoint.use_dcp_save
        self.nixl_storage_client = NIXLStorageClient(
            client_name=train_client_name(self.rank),
            server_name=NIXL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PUSH_SIDE,
            nixl_config=self.psrl_config.nixl,
            replica_idx=0,  # replica idx is not used on the train side
            worker_index=self.rank,
            logging_path=self.psrl_config.logging_path,
            enable_prog_thread=enable_prog_thread,
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_convert_params(self):
        """Convert model parameters to NIXL's unified sharding format.

        Branches on backend strategy to call the appropriate converter.
        """
        strategy = self.config.actor.strategy
        model_config = AutoConfig.from_pretrained(
            copy_to_local(self.config.model.path),
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )
        if strategy in {"fsdp", "fsdp2"}:
            from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace

            parameter_mapping = create_parameter_mapping("FSDP", model_config)
            self.unified_state_dict, self.local_sharding_dict = convert_fsdp_inplace(
                parameter_mapping,
                self.actor.engine.module,
                fsdp_strategy=self.config.actor.strategy,
            )
            self.param_sync_plan = ParamSyncPlan()
        elif strategy == "megatron":
            assert convert_megatron_inplace is not None, (
                "psrl.utils.converter.megatron_converter could not be imported. Make sure Megatron-LM is installed."
            )
            parameter_mapping = create_parameter_mapping("Megatron", model_config)
            conversion_result = convert_megatron_inplace(
                parameter_mapping,
                self.actor.engine.module,
            )
            self.unified_state_dict = conversion_result.state_dict
            self.local_sharding_dict = conversion_result.sharding_dict
            self.param_sync_plan = conversion_result.sync_plan
        else:
            raise NotImplementedError(f"nixl_convert_params does not support strategy '{strategy}'.")

    def nixl_protocol(self, mode: str = "full"):
        """Run the 8-step NIXL server handshake.

        Args:
            mode: `full` registers real tensors. `meta` registers only metadata before
                parameter server weights arrive.
        """
        meta_only = mode == "meta"
        if self.unified_state_dict is None or self.local_sharding_dict is None:
            self.nixl_convert_params()
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(self.local_sharding_dict)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(
            self.unified_state_dict, unified_sharding_dict, meta_only=meta_only
        )
        psrl_logger.info("nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info("nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info("nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info("nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info("nixl client protocol done.")
        self.unified_sharding_dict = unified_sharding_dict

    def nixl_sleep(self, mode: str = "full"):
        """Deregister NIXL tensors and release GPU memory."""
        self.sleep_model()
        if mode == "meta":
            return
        self.nixl_storage_client.deregister_local_tensors()

    def nixl_wake_up(self):
        """Restore GPU memory and re-register NIXL tensors after sleep."""
        self.wake_up_model()
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)

    def sleep_model(self):
        """Release GPU memory for model weights (metadata preserved for wake_up).

        Uses torch_memory_saver.pause() when TMS range covers training,
        otherwise falls back to manual storage-resize helpers.
        """
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} sleep")

        # NOTE(lhy): Empty the PyTorch cache before pausing TMS to prevent duplicate
        # `cuMemRelease` calls from the mempool and pause paths.
        aggressive_empty_cache(force_sync=True)
        if self.psrl_config.tms.range in ["train", "all"]:
            torch_memory_saver.pause()
        else:
            strategy = self.config.actor.strategy
            if strategy in {"fsdp", "fsdp2"}:
                self._sleep_fsdp_model(self.actor.engine.module)
            elif strategy == "megatron":
                self._sleep_megatron_model(self.actor.engine.module)
            else:
                raise NotImplementedError(f"sleep_model does not support strategy '{strategy}'.")
            torch.cuda.empty_cache()

        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"After TrainWorker_R{self.rank} sleep")

    def wake_up_model(self):
        """Restore GPU memory allocation without restoring data (NIXL fills it)."""
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} wake_up")

        if self.psrl_config.tms.range in ["train", "all"]:
            torch_memory_saver.resume()
        else:
            strategy = self.config.actor.strategy
            if strategy in {"fsdp", "fsdp2"}:
                self._wake_up_fsdp_model(self.actor.engine.module)
            elif strategy == "megatron":
                self._wake_up_megatron_model(self.actor.engine.module)
            else:
                raise NotImplementedError(f"wake_up_model does not support strategy '{strategy}'.")

        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"After TrainWorker_R{self.rank} wake_up")

    def _restore_non_persistent_buffers_from_ps(self) -> None:
        """
        Restore non-persistent named buffers (e.g. inv_freq) from PS after pull.
        """
        strategy = self.config.actor.strategy
        if strategy in {"fsdp", "fsdp2"}:
            ps_buffers = self._get_non_persistent_buffers_from_ps()
            if not ps_buffers:
                return
            device = get_device_id()
            model = self.actor.engine.module
            for full_name, cpu_tensor in ps_buffers.items():
                *module_path_parts, buf_attr = full_name.split(".")
                module = model
                for part in module_path_parts:
                    module = getattr(module, part, None)
                    if module is None:
                        break
                if module is not None and hasattr(module, buf_attr):
                    existing = getattr(module, buf_attr)
                    if isinstance(existing, torch.Tensor):
                        existing.data.copy_(cpu_tensor.to(device=device, dtype=existing.dtype))
            psrl_logger.info(
                "[_restore_non_persistent_buffers_from_ps] Restored non-persistent FSDP buffers. "
                f"Count={len(ps_buffers)}."
            )
        elif strategy == "megatron":
            from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

            ps_buffers = self._get_non_persistent_buffers_from_ps()

            inv_freq_tensors = {
                name: tensor for name, tensor in ps_buffers.items() if name.endswith(".inv_freq") or name == "inv_freq"
            }
            if not inv_freq_tensors:
                psrl_logger.warning("[_restore_non_persistent_buffers_from_ps] No inv_freq found in PS buffers.")
                return

            # Standard RoPE layers share one `inv_freq` value.
            reference_inv_freq = next(iter(inv_freq_tensors.values()))
            device = torch.cuda.current_device()
            restored = 0
            for model_chunk in self.actor.engine.module:
                for module in model_chunk.modules():
                    if isinstance(module, RotaryEmbedding) and hasattr(module, "inv_freq"):
                        # Cached trigonometric tensors reference released GPU memory.
                        if hasattr(module.forward, "cache_clear"):
                            module.forward.cache_clear()
                        module.inv_freq = reference_inv_freq.to(device=device, dtype=module.inv_freq.dtype)
                        restored += 1
            psrl_logger.info(
                "[_restore_non_persistent_buffers_from_ps] Restored inv_freq and cleared lru_cache. "
                f"Modules={restored}."
            )
        else:
            raise NotImplementedError(
                f"_restore_non_persistent_buffers_from_ps does not support strategy '{strategy}'."
            )

    def _sleep_fsdp_model(self, model):
        """Release GPU memory for FSDP model by resizing local shard storage to 0."""
        from torch.distributed.tensor import DTensor as _DTensor

        for _, param in model.named_parameters():
            local_tensor = param._local_tensor if isinstance(param, _DTensor) else param
            if local_tensor.untyped_storage().size() > 0:
                param._sleep_storage_size = local_tensor.untyped_storage().size()
                local_tensor.untyped_storage().resize_(0)
        for _, buffer in model.named_buffers():
            if buffer is not None and buffer.untyped_storage().size() > 0:
                buffer._sleep_storage_size = buffer.untyped_storage().size()
                buffer.untyped_storage().resize_(0)

    def _wake_up_fsdp_model(self, model):
        """Restore GPU memory allocation for FSDP model (no data copy)."""
        from torch.distributed.tensor import DTensor as _DTensor

        for _, param in model.named_parameters():
            local_tensor = param._local_tensor if isinstance(param, _DTensor) else param
            if hasattr(param, "_sleep_storage_size") and local_tensor.untyped_storage().size() == 0:
                local_tensor.untyped_storage().resize_(param._sleep_storage_size)
        for _, buffer in model.named_buffers():
            if buffer is not None and hasattr(buffer, "_sleep_storage_size") and buffer.untyped_storage().size() == 0:
                buffer.untyped_storage().resize_(buffer._sleep_storage_size)

    def _sleep_megatron_model(self, models):
        """
        Release Megatron DDP buffer storage.

        Parameters and gradients are views into contiguous DDP buffers, so resizing
        those buffers releases the underlying allocation.
        """
        try:
            from megatron.core import DistributedDataParallel as DDP  # noqa: PLC0415
        except ImportError:
            DDP = None

        for model_chunk in models:
            if DDP is not None and isinstance(model_chunk, DDP):
                for buffers in [model_chunk.buffers, model_chunk.expert_parallel_buffers]:
                    for buf in buffers:
                        if buf.param_data is not None and buf.param_data.untyped_storage().size() > 0:
                            buf._sleep_param_data_size = buf.param_data.untyped_storage().size()
                            buf.param_data.untyped_storage().resize_(0)
                        if buf.grad_data is not None and buf.grad_data.untyped_storage().size() > 0:
                            buf._sleep_grad_data_size = buf.grad_data.untyped_storage().size()
                            buf.grad_data.untyped_storage().resize_(0)
            else:
                unwrapped = unwrap_model(model_chunk)
                for _, param in unwrapped.named_parameters():
                    if param.data.untyped_storage().size() > 0:
                        param._sleep_storage_size = param.data.untyped_storage().size()
                        param.data.untyped_storage().resize_(0)
                    if param.grad is not None and param.grad.untyped_storage().size() > 0:
                        param._sleep_grad_storage_size = param.grad.untyped_storage().size()
                        param.grad.untyped_storage().resize_(0)

    def _wake_up_megatron_model(self, models):
        """
        Restore Megatron DDP buffer storage without copying data.

        Existing parameter views remain valid. The subsequent NIXL pull restores weights.
        """
        try:
            from megatron.core import DistributedDataParallel as DDP  # noqa: PLC0415
        except ImportError:
            DDP = None

        for model_chunk in models:
            if DDP is not None and isinstance(model_chunk, DDP):
                for buffers in [model_chunk.buffers, model_chunk.expert_parallel_buffers]:
                    for buf in buffers:
                        if hasattr(buf, "_sleep_param_data_size") and buf.param_data.untyped_storage().size() == 0:
                            buf.param_data.untyped_storage().resize_(buf._sleep_param_data_size)
                        if hasattr(buf, "_sleep_grad_data_size") and buf.grad_data.untyped_storage().size() == 0:
                            buf.grad_data.untyped_storage().resize_(buf._sleep_grad_data_size)
                            buf.grad_data.zero_()
            else:
                unwrapped = unwrap_model(model_chunk)
                for _, param in unwrapped.named_parameters():
                    if hasattr(param, "_sleep_storage_size") and param.data.untyped_storage().size() == 0:
                        param.data.untyped_storage().resize_(param._sleep_storage_size)
                    if (
                        param.grad is not None
                        and hasattr(param, "_sleep_grad_storage_size")
                        and param.grad.untyped_storage().size() == 0
                    ):
                        param.grad.untyped_storage().resize_(param._sleep_grad_storage_size)
                        param.grad.zero_()

    def ray_push_model(self) -> None:
        """Push model weights to the parameter server via CPU Ray object store.

        Modes:
            `cpu`: The parameter server blocks on the state dictionary transfer.
            `cpu_ref`: The train side blocks on `ray.put()`. The parameter server remains non-blocking.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_version = ray.get(ps_manager_handle.get_ps_model_version.remote(debug_info="engine_train_worker"))
        next_version = curr_version + 1

        psrl_logger.info("Gathering per-tensor params for CPU push.")
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        full_state_dict: dict = {}
        for name, param in per_tensor_param:
            if self.is_train_representative_rank:
                full_state_dict[name] = param.to("cpu", non_blocking=True)

        if self.is_train_representative_rank:
            assert len(full_state_dict) > 0, "State dict should not be empty on the representative rank."
            psrl_logger.info("Pushing model to PS (async, representative rank).")
            if self.psrl_config.ps_mode == "cpu":
                ps_manager_handle.push_model_state_dict_cpu.remote(next_version, full_state_dict)
            elif self.psrl_config.ps_mode == "cpu_ref":
                object_ref = ray.put(full_state_dict)
                ps_manager_handle.push_model_state_dict_cpu_ref_list.remote(next_version, [object_ref])
            else:
                raise NotImplementedError(f"ray_push_model does not support ps_mode='{self.psrl_config.ps_mode}'.")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def init_model(self, init_mode: str = "full"):
        """Initialize the actor (and optional ref) model.

        Args:
            init_mode: 'full' loads pre-trained weights from disk.
                       'empty' skips weight loading (used by async NIXL boot path,
                       where NIXL streams the real weights from the parameter server).
        """
        # NOTE(lhy): Initialize the default process group early because DCP resharding
        # can exceed gloo's 30-minute timeout on cold filesystems.
        from verl.utils.distributed import initialize_global_process_group_ray

        initialize_global_process_group_ray(timeout_second=36000)

        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            skip_load_weight = init_mode == "empty"
            strategy = self.config.actor.strategy

            if skip_load_weight:
                OmegaConf.set_struct(self.config, True)
                with open_dict(self.config):
                    if strategy in {"fsdp", "fsdp2"}:
                        self.config.actor.fsdp_config.load_weight = False
                    elif strategy == "megatron":
                        self.config.actor.megatron.load_weight = False

            try:
                ActorRolloutRefWorker.init_model(self)
            finally:
                if skip_load_weight:
                    OmegaConf.set_struct(self.config, True)
                    with open_dict(self.config):
                        if strategy in {"fsdp", "fsdp2"}:
                            self.config.actor.fsdp_config.load_weight = True
                        elif strategy == "megatron":
                            self.config.actor.megatron.load_weight = True

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def update_actor(self, data: TensorDict) -> TensorDict:
        """Train the actor for one step, then push updated weights to the PS."""
        with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
            output = ActorRolloutRefWorker.update_actor(self, data)
        push_model = tu.pop(data, key="push_model", default=True)
        if not push_model:
            return output

        torch.cuda.synchronize()
        context_manager = (
            exclusive_push_model_context(self.train_interface.ps_manager_handle)
            if self.is_train_representative_rank
            else nullcontext()
        )
        with context_manager:
            with log_dual_events("Push model", psrl_logger, event_type=EventType.PUSH):
                PSRL_BaseTrainWorker.push_model(self)
        return output

    def reload_optimizer_after_pull(self):
        """Reload optimizer's fp32/bf16 master params from the model's current bf16 params after NIXL pull."""
        if not (self._is_actor and self.actor is not None):
            return
        engine = self.actor.engine
        if engine.optimizer is None:
            return
        optimizer = engine.optimizer
        if not hasattr(optimizer, "reload_model_params"):
            return
        optimizer.reload_model_params()

    def _debug_log_train_model_info(self, label: str, max_elements: int = 10):
        """Log per-tensor statistics for debugging weight correctness."""
        if not (self._is_actor and self.actor is not None):
            return
        strategy = self.config.actor.strategy
        if strategy in {"fsdp", "fsdp2"}:
            for tensor_type, named_tensors in (
                ("param", self.actor.engine.module.named_parameters()),
                ("buffer", self.actor.engine.module.named_buffers()),
            ):
                for name, tensor in named_tensors:
                    log_tensor(
                        tensor,
                        psrl_logger=psrl_logger,
                        log_prefix=f"{label} {tensor_type} {name}",
                        max_elements=max_elements,
                    )
        elif strategy == "megatron":

            def _iter_model_chunks():
                model = self.actor.engine.module
                if isinstance(model, (list, tuple)):
                    yield from enumerate(model)
                else:
                    yield 0, model

            for chunk_idx, model_chunk in _iter_model_chunks():
                chunk_module = unwrap_model(model_chunk) if unwrap_model is not None else model_chunk
                for tensor_type, named_tensors in (
                    ("param", chunk_module.named_parameters()),
                    ("buffer", chunk_module.named_buffers()),
                ):
                    for name, tensor in named_tensors:
                        log_tensor(
                            tensor,
                            psrl_logger=psrl_logger,
                            log_prefix=f"{label} chunk{chunk_idx} {tensor_type} {name}",
                            max_elements=max_elements,
                        )
        else:
            raise NotImplementedError(f"_debug_log_train_model_info does not support strategy '{strategy}'.")
