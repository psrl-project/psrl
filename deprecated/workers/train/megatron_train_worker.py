import logging
import os
from contextlib import nullcontext
from typing import TYPE_CHECKING

import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tensordict import TensorDict
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id
from verl.utils.fs import copy_to_local
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.megatron_workers import ActorRolloutRefWorker

from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.patch_utils import apply_tms_patch
from psrl.utils.common.worker_naming import train_client_name
from psrl.utils.common.dynamic_import import lazy_import_many_to_globals, lazy_import_to_globals
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    MemoryLogger,
    deprecated,
    get_worker_info,
    gpu_memory_logger_decorator,
    log_dual_events,
    log_tensor,
)
from psrl.utils.nixl import (
    NIXLClientType,
    NIXLInterface,
    NIXLStorageClient,
)
from psrl.utils.optimizer_state import assert_optimizer_state_offloaded
from psrl.utils.ray import exclusive_push_model_context
from psrl.workers.train import PSRL_BaseTrainWorker, TrainInterface

# Make type checking happy
if TYPE_CHECKING:
    from mbridge.core.util import unwrap_model
    from megatron.core import DistributedDataParallel as DDP
    from verl.models.mcore import get_mcore_weight_converter
    from verl.utils.megatron_utils import (
        load_megatron_model_to_gpu,
        offload_megatron_model_to_cpu,
        per_tensor_generator,
    )

    from psrl.utils.converter.megatron_converter import convert_megatron_inplace
    from verl.utils.megatron.router_replay_patch import (
        RouterReplay,
        RouterReplayAction,
    )

    try:
        from torch_memory_saver import torch_memory_saver
    except ImportError:
        pass
else:
    get_mcore_weight_converter = None
    load_megatron_model_to_gpu = None
    offload_megatron_model_to_cpu = None
    per_tensor_generator = None
    DDP = None
    unwrap_model = None
    convert_megatron_inplace = None
    RouterReplay = None
    RouterReplayAction = None
    torch_memory_saver = None

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_MegatronTrainWorker(ActorRolloutRefWorker, PSRL_BaseTrainWorker):
    def __init__(
        self,
        config: DictConfig,
        role: str,
        psrl_config: DictConfig,
        train_interface: TrainInterface,
        nixl_interface: NIXLInterface,
        distillation_config: DictConfig | None = None,
    ) -> None:
        ActorRolloutRefWorker.__init__(self, config, role)
        self.distillation_config = distillation_config
        PSRL_BaseTrainWorker.__init__(
            self,
            self.rank,
            self.world_size,
            psrl_config,
            train_interface,
            nixl_interface,
        )

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None

        if self.psrl_config.tms.range in ["train", "all"]:
            lazy_import_to_globals("torch_memory_saver", "torch_memory_saver")
            apply_tms_patch()

        # Build logger
        self.log_prefix = f"TrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {get_worker_info()}.")

        # Memory logger (periodic + on-demand GPU memory log, same path/prefix pattern with Mem suffix)
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
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        return self.rank == 0

    def get_replica_id(self) -> int:
        """
        Get the replica id (dp id) of the train worker.
        """
        from megatron.core import parallel_state as mpu

        assert mpu.is_initialized(), "Megatron is not initialized."
        return mpu.get_data_parallel_rank()

    def init_nixl_client(self):
        """Initialize the NIXL client."""
        # NOTE(lhy): the init_nixl_client is called before the initialization of the actor module now
        # Because in UCX 1.18.0, this may enhance the communication performance
        # assert self.actor_module, "The actor module must be initialized before calling init_nixl_client."
        # When DCP save is enabled, disable NIXL background UCX progress thread to prevent
        # heap corruption from DCP's all_gather_object (large temporary allocations corrupt
        # UCX endpoint address structures read by the background thread).
        enable_prog_thread = not self.psrl_config.checkpoint.use_dcp_save
        self.nixl_storage_client = NIXLStorageClient(
            client_name=train_client_name(self.rank),
            server_name=NIXL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PUSH_SIDE,
            nixl_config=self.psrl_config.nixl,
            nixl_interface=self.nixl_interface,
            # client_group_id=self.get_replica_id()
            logging_path=self.psrl_config.logging_path,
            enable_prog_thread=enable_prog_thread,
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_convert_params(self):
        """Convert the Megatron model parameters for NIXL storage client."""
        from transformers import AutoConfig

        lazy_import_to_globals("psrl.utils.converter.megatron_converter", "convert_megatron_inplace")
        model_config = AutoConfig.from_pretrained(
            copy_to_local(self.config.model.path),
            trust_remote_code=self.config.model.get("trust_remote_code", False),
        )
        parameter_mapping = create_parameter_mapping("Megatron", model_config)
        self.unified_state_dict, self.local_sharding_dict, self.param_sync_plan = convert_megatron_inplace(
            parameter_mapping,
            self.actor_module,
            return_sync_plan=True,
        )

    def nixl_protocol(self, mode: str = "full"):
        """Run the NIXL server protocol.

        Args:
            mode (str): Mode of conversion, either 'meta' or 'full'.
                'meta' mode converts to meta tensors and skip registering them.
                'full' mode converts to full tensors.

            NOTE: ps storage may init with meta tensors, the register step would be different.
        """
        lazy_import_to_globals("psrl.utils.converter.megatron_converter", "convert_megatron_inplace")
        # Register the state dict and sharding dict to the NIXL client
        meta_only = mode == "meta"
        if self.unified_state_dict is None or self.local_sharding_dict is None:
            self.nixl_convert_params()

        # Register the state dict and sharding dict to the NIXL client
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
        """Deregister local tensors and put model weights to sleep state."""
        full_sleep = mode != "meta"
        if full_sleep:
            if not self._is_offload_optimizer:
                raise RuntimeError(
                    "Full trainer sleep requires train_actor_rollout_ref.actor.megatron."
                    "optimizer_offload=True so optimizer state survives TMS pause/resume."
                )
            torch.cuda.synchronize()
            optimizer_summary = assert_optimizer_state_offloaded(
                self.actor_optimizer,
                stage="before_sleep",
                rank=self.rank,
            )
            psrl_logger.info(
                "[TRAINER_OPTIMIZER_OFFLOAD_CHECK] stage=before_sleep rank=%d summary=%s",
                self.rank,
                optimizer_summary,
            )
        self._psrl_full_sleep_active = full_sleep
        self.sleep_megatron_model()
        if mode == "meta":
            return
        if self.nixl_storage_client.local_client_info is None:
            psrl_logger.warning("Skip NIXL deregistration because local tensors are not registered.")
            return
        self.nixl_storage_client.deregister_local_tensors()

    def sleep_megatron_model(self):
        """
        Release GPU memory for model weights without CPU offloading.
        The model metadata (shape, dtype, etc.) is preserved for later wake_up.
        """
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} sleep")

        # NOTE(lhy): aggressive_empty_cache is used to ensure no torch reserved memory exists
        # so torch won't trigger cudaFree from the mempool side
        # otherwise it will cause double cuMemRelease (first pause, then free) in tms
        aggressive_empty_cache(force_sync=True)
        # Release GPU memory for actor_module parameters
        if self.psrl_config.tms.range in ["train", "all"]:
            torch_memory_saver.pause()
        else:
            self._sleep_megatron_model(self.actor_module)
        # aggressive_empty_cache(force_sync=True)

        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"After TrainWorker_R{self.rank} sleep")

    @deprecated(
        "This method is deprecated, it's reserved as an example "
        "for Megatron manual memory management. Use torch_memory_saver instead."
    )
    def _sleep_megatron_model(self, models):
        """
        Release GPU memory for Megatron model without CPU offloading.
        For DDP-wrapped models, resize the storage of buffers to 0.
        For non-DDP models, resize the storage of parameters to 0.

        Args:
            models: List of model chunks (may be DDP-wrapped)
        """
        for model_chunk in models:
            if isinstance(model_chunk, DDP):
                # Handle DDP-wrapped model
                model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
                for buffers in model_chunk_all_buffers:
                    for buffer in buffers:
                        # Release parameter data storage
                        if buffer.param_data.untyped_storage().size() > 0:
                            buffer.param_data_size = buffer.param_data.untyped_storage().size()
                            buffer.param_data.untyped_storage().resize_(0)

                        # Release gradient data storage
                        if buffer.grad_data.untyped_storage().size() > 0:
                            buffer.grad_data_size = buffer.grad_data.untyped_storage().size()
                            buffer.grad_data.untyped_storage().resize_(0)
            else:
                # Handle non-DDP model (e.g., reference model without DDP wrapper)
                unwrapped = unwrap_model(model_chunk)
                for _, param in unwrapped.named_parameters():
                    if param.data.untyped_storage().size() > 0:
                        # Store the original storage size as an attribute
                        param._sleep_storage_size = param.data.untyped_storage().size()
                        param.data.untyped_storage().resize_(0)
                    if param.grad is not None and param.grad.untyped_storage().size() > 0:
                        param._sleep_grad_storage_size = param.grad.untyped_storage().size()
                        param.grad.untyped_storage().resize_(0)

    def nixl_wake_up(self):
        """Wake up and re-register NIXL after sleep.

        This method restores GPU memory allocation and performs NIXL re-registration
        to handle memory changes after sleep/wake_up cycle.
        """
        self.wake_up_megatron_model()
        if getattr(self, "_psrl_full_sleep_active", False):
            optimizer_summary = assert_optimizer_state_offloaded(
                self.actor_optimizer,
                stage="after_wake",
                rank=self.rank,
            )
            psrl_logger.info(
                "[TRAINER_OPTIMIZER_OFFLOAD_CHECK] stage=after_wake rank=%d summary=%s",
                self.rank,
                optimizer_summary,
            )
        # Re-register the state dict and sharding dict to the NIXL client
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)
        self._psrl_full_sleep_active = False

    def wake_up_megatron_model(self):
        """
        Restore GPU memory allocation for model weights without restoring data.
        The actual data will be transferred via NIXL.
        """
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} wake_up")

        # aggressive_empty_cache(force_sync=True)
        # Restore GPU memory allocation for actor_module
        if self.psrl_config.tms.range in ["train", "all"]:
            torch_memory_saver.resume()
        else:
            self._wake_up_megatron_model(self.actor_module)
        # aggressive_empty_cache(force_sync=True)

        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"After TrainWorker_R{self.rank} wake_up")

    @deprecated(
        "This method is deprecated, it's reserved as an example "
        "for Megatron manual memory management. Use torch_memory_saver instead."
    )
    def _wake_up_megatron_model(self, models):
        """
        Restore GPU memory allocation for Megatron model without restoring data.
        For DDP-wrapped models, resize the storage of buffers back to original size.
        For non-DDP models, resize the storage of parameters back to original size.

        Args:
            models: List of model chunks (may be DDP-wrapped)
        """
        for model_chunk in models:
            if isinstance(model_chunk, DDP):
                # Handle DDP-wrapped model
                model_chunk_all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
                for buffers in model_chunk_all_buffers:
                    for buffer in buffers:
                        # Restore parameter data storage (allocate memory but don't copy data)
                        if hasattr(buffer, "param_data_size") and buffer.param_data.untyped_storage().size() == 0:
                            buffer.param_data.untyped_storage().resize_(buffer.param_data_size)

                        # Restore gradient data storage
                        if hasattr(buffer, "grad_data_size") and buffer.grad_data.untyped_storage().size() == 0:
                            buffer.grad_data.untyped_storage().resize_(buffer.grad_data_size)
                            buffer.grad_data.zero_()  # Initialize gradients to zero
            else:
                # Handle non-DDP model (e.g., reference model without DDP wrapper)
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

    def _restore_non_persistent_buffers_from_ps(self) -> None:
        """
        Restore inv_freq for all RotaryEmbedding modules from PS after pull
        and clear lru_cache to discard stale cos/sin tensors.

        NOTE(lhy): Megatron's RotaryEmbedding.inv_freq is a plain tensor attribute
        (not register_buffer), so it is not covered by named_buffers(). We extract
        inv_freq from the HF-named buffers returned by PS and apply directly.
        """
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        ps_buffers = self._get_non_persistent_buffers_from_ps()

        # Extract inv_freq tensors from PS buffers (HF naming).
        inv_freq_tensors = {
            name: tensor for name, tensor in ps_buffers.items() if name.endswith(".inv_freq") or name == "inv_freq"
        }
        if not inv_freq_tensors:
            psrl_logger.warning("[_restore_non_persistent_buffers_from_ps] No inv_freq found in PS buffers.")
            return

        # Use the first available inv_freq (all layers share the same value for standard RoPE).
        reference_inv_freq = next(iter(inv_freq_tensors.values()))
        device = torch.cuda.current_device()
        restored = 0
        for model_chunk in self.actor_module:
            for module in model_chunk.modules():
                if isinstance(module, RotaryEmbedding) and hasattr(module, "inv_freq"):
                    # Clear lru_cache: cached cos/sin point to freed/garbage GPU memory.
                    if hasattr(module.forward, "cache_clear"):
                        module.forward.cache_clear()
                    module.inv_freq = reference_inv_freq.to(device=device, dtype=module.inv_freq.dtype)
                    restored += 1
        psrl_logger.info(
            f"[_restore_non_persistent_buffers_from_ps] Restored inv_freq and cleared "
            f"lru_cache for {restored} RotaryEmbedding module(s)."
        )

    def ray_push_model(self) -> None:
        """
        Push the model weights to the PS.
        In 'cpu' mode, push the full state dict.
        The PS worker will block on large model transfer (potential bottleneck).
        In 'cpu_ref' mode, push a ray object_ref.
        Only the train worker blocks on ray.put, PS worker is non-blocking.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        curr_ps_model_version = ray.get(
            ps_manager_handle.get_ps_model_version.remote(debug_info="megatron_train_worker")
        )
        next_ps_model_version = curr_ps_model_version + 1
        # Gather the model state dict on rank 0
        psrl_logger.info("Gathering the full state dict on the CPU of the representative rank.")
        if self.weight_converter is None:
            self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
        per_tensor_param = per_tensor_generator(
            self.actor_module,
            self.actor_model_config,
            self.weight_converter,
            self.tf_config,
            self.layer_name_mapping,
        )
        full_state_dict = {}
        for name, param in per_tensor_param:
            if self.is_train_representative_rank:
                full_state_dict[name] = param.to("cpu", non_blocking=True)
        if self.is_train_representative_rank:
            assert len(full_state_dict) > 0, "The model state dict shouldn't be empty on the representative worker."
            psrl_logger.info("Push the model via CPU on the representative rank (async).")
            if self.psrl_config.ps_mode == "cpu":
                # In 'cpu' mode, push the full state dict (PS worker will block on transfer)
                # But the training side does not need to wait for the push to complete,
                # as it can be overlapped with the next-iteration training
                ps_manager_handle.push_model_state_dict_cpu.remote(next_ps_model_version, full_state_dict)
            elif self.psrl_config.ps_mode == "cpu_ref":
                # In 'cpu_ref' mode, push a ray object_ref (PS worker is non-blocking)
                # But the training side needs to wait for the push to complete, as `ray.put` is blocking
                object_ref = ray.put(full_state_dict)  # This blocks until the state dict is in the object store
                ps_manager_handle.push_model_state_dict_cpu_ref_list.remote(
                    next_ps_model_version, [object_ref]
                )  # Tricky part: manually wrap the object_ref in a list to avoid ray dereferencing the full state dict
            else:
                raise NotImplementedError(
                    f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet."
                )
        else:
            assert len(full_state_dict) == 0, "The model state dict should be empty on non-representative workers."

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def init_model(self, init_mode: str = "full"):
        """
        Initialize the Megatron model.
        If init_mode is 'empty', only initialize the model without loading weights.

        Args:
            init_mode (str): 'full' to load weights, 'empty' to only initialize the model.
        """
        lazy_import_to_globals("verl.models.mcore", "get_mcore_weight_converter")
        lazy_import_many_to_globals(
            "verl.utils.megatron_utils",
            ["load_megatron_model_to_gpu", "offload_megatron_model_to_cpu", "per_tensor_generator"],
        )
        lazy_import_to_globals("mbridge.core.util", "unwrap_model")
        lazy_import_to_globals("megatron.core", "DistributedDataParallel", "DDP")
        lazy_import_many_to_globals(
            "verl.utils.megatron.router_replay_patch",
            ["RouterReplay", "RouterReplayAction"],
        )

        skip_load_weight = init_mode == "empty"

        with log_dual_events("Initialize model", psrl_logger, event_type=EventType.INIT):
            # Set load_weight to False temporarily
            if skip_load_weight:
                OmegaConf.set_struct(self.config, True)
                with open_dict(self.config):
                    if OmegaConf.select(self.config, "load_weight"):
                        self.config.load_weight = False

            ActorRolloutRefWorker.init_model(self)
            # Override checkpoint strategy from psrl config (bypasses McoreEngineConfig dataclass).
            self.checkpoint_mananager.use_per_rank_checkpoint = not self.psrl_config.checkpoint.use_dcp_save
            if skip_load_weight:
                OmegaConf.set_struct(self.config, True)
                with open_dict(self.config):
                    if OmegaConf.select(self.config, "load_weight"):
                        self.config.load_weight = True

    def _build_rollout(self, trust_remote_code: bool = False):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def build_rollout(self, trust_remote_code: bool = False):
        ActorRolloutRefWorker._build_rollout(self, trust_remote_code=trust_remote_code)

    # The log_prob in training side may need to be recomputed
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        with log_dual_events("Recompute log_prob", psrl_logger, event_type=EventType.OTHER):
            assert self._is_actor
            if self._is_offload_param:
                load_megatron_model_to_gpu(self.actor_module, load_grad=False)

            tu.assign_non_tensor(
                data,
                micro_batch_size=self.config.rollout.log_prob_micro_batch_size_per_gpu,
                max_token_len=self.config.rollout.log_prob_max_token_len_per_gpu,
                use_dynamic_bsz=self.config.rollout.log_prob_use_dynamic_bsz,
            )

            for key, value in list(data.items()):
                if key != "routed_experts" and isinstance(value, torch.Tensor):
                    data[key] = value.to(get_device_id())

            if self.enable_routing_replay and self.config.actor.router_replay.mode == "R2":
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

            if self.enable_routing_replay and self.config.actor.router_replay.mode == "R3":
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

            output, entropys, layers_topk_idx = self.actor.compute_log_prob(data=data, calculate_entropy=True)
            output = tu.get_tensordict({"recomputed_log_probs": output, "entropys": entropys})

            if self.config.actor.router_replay.mode == "R2":
                output["routed_experts"] = layers_topk_idx
            if self.config.actor.router_replay.mode in ["R2", "R3"]:
                RouterReplay.clear_global_indices()
                RouterReplay.clear_global_router_replay_action()

            output = output.cpu()
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            aggressive_empty_cache(force_sync=True)
            return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def update_actor(self, data: TensorDict) -> TensorDict:
        with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
            output = ActorRolloutRefWorker.update_actor(self, data)
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

    def _debug_log_train_model_info(self, label: str, max_elements: int = 10):
        """Debug log Megatron train model tensor statistics."""

        def _iter_model_chunks():
            model = self.actor_module
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
                        tensor=tensor,
                        psrl_logger=psrl_logger,
                        log_prefix=label,
                        name=f"chunk={chunk_idx}, {tensor_type}={name}",
                        max_elements=max_elements,
                    )
