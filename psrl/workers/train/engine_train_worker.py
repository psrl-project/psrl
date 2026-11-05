# Copyright (c) 2025, PSRL Authors.

import functools
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
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.fs import copy_to_local
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.config import DistillationConfig, TrainingWorkerConfig
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from psrl.utils.common.patch_utils import apply_tms_patch
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.param_sync import ParamSyncPlan
from psrl.utils.quantization import QuantizationPlan

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
from psrl.utils.ps_mode import has_ps_data_plane  # noqa: E402
from psrl.utils.ray import exclusive_push_model_context  # noqa: E402
from psrl.workers.train.base_train_worker import PSRL_BaseTrainWorker, TrainInterface  # noqa: E402

try:
    from torch_memory_saver import torch_memory_saver
except ImportError:
    torch_memory_saver = None  # type: ignore

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_CriticTrainWorker(TrainingWorker):
    """
    Critic worker that initializes the global process group before constructing its engine.

    veRL's `TrainingWorker` builds the engine in `__init__`, so the process group must exist
    before the base constructor runs.
    """

    def __init__(self, config: TrainingWorkerConfig) -> None:
        initialize_global_process_group_ray(timeout_second=36000)
        super().__init__(config)


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

        # Per-rank checkpointing was a patch carried inside the vendored verl checkout:
        # verl's Megatron checkpoint manager called into `psrl.utils.checkpoint` when
        # `use_per_rank_checkpoint` was set. Upstream verl has no such field and no such
        # hook, so the assignment that used to configure it now fails -- the config is in
        # struct mode, and writing an undeclared key raises.
        #
        # Refusing rather than skipping: the flag selects the on-disk checkpoint *format*,
        # so quietly ignoring a request for per-rank saves would write a shape that a
        # later resume cannot read, and the mismatch would surface as a load failure long
        # after the run that produced it.
        # The `use_dcp_save` / per-rank-saver / progress-thread interaction is validated
        # centrally in `psrl.utils.config`, which can see `trainer.save_freq`; this worker
        # only receives its own actor config.

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

        # Colocate stage-boundary offload (design §3.7.3). Lives for the process so its
        # pinned host buffers are allocated once, not per switch.
        from psrl.utils.nixl.stage_offload import TrainerStage  # noqa: PLC0415

        self._stage = TrainerStage()

        # Host-side ref-policy snapshot (design §3.10.2). Built lazily by
        # capture_ref_backup() once the actor is initialised; None until then.
        self._ref_backup = None
        #: True between the first optimizer write of an update and its push.
        self._write_gate_open = False

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
        #
        # Direct supply overrides that: the train rank is a *source* that rollouts read
        # at arbitrary times, including while it is computing, and a transfer with no
        # thread advancing the UCX worker commits successfully and then sits in `PROC`
        # forever. It surfaces far from here, as the reader's
        # The NIXL progress thread must stay on whenever train ranks serve direct reads
        # (direct supply) or whenever DCP save is enabled — nvrx async DCP uses a
        # spawn-based persistent worker that does not inherit the parent's UCX / NIXL
        # context, so saving with the thread running is now safe.  The progress thread
        # is therefore always enabled on direct-supply runs, and always enabled when DCP
        # saving is on; it is disabled only for non-direct-supply runs that also do not
        # save checkpoints.
        serves_direct_reads = not has_ps_data_plane(self.psrl_config.ps_mode)
        enable_prog_thread = serves_direct_reads or self.psrl_config.checkpoint.use_dcp_save
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
            pairwise_descriptors=has_ps_data_plane(self.psrl_config.ps_mode),
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_send_update_infos(self, plan: dict) -> dict:
        """Send this rank's share of an update-info fan-out.

        The meta server serializes each updated source once and describes, per
        destination agent, the sources it reads plus its connection metadata. Every
        train rank sends an interleaved slice of those destinations from its own agent,
        turning an O(readers) serial loop into O(readers / n_train_ranks). The reader
        still receives exactly one notification, so its acknowledgement is unchanged.
        """
        from psrl.utils.nixl.handshake_notif import NotifKind, encode  # noqa: PLC0415

        client = getattr(self, "nixl_storage_client", None)
        if client is None or not plan:
            return {"rank": self.rank, "agents": 0, "bytes": 0}
        destinations = plan["destinations"]
        sources = plan["sources"]
        ordered = sorted(destinations)
        mine = ordered[self.rank :: self.world_size]
        sent_bytes = 0
        for agent_name in mine:
            entry = destinations[agent_name]
            conn_info = entry.get("conn_info")
            if conn_info:
                # Refresh the destination each fan-out. The destination re-registers
                # between switches; an entry cached across that boundary can leave the
                # notification queued nowhere, and the failure is a silent stall for
                # the whole acknowledgement budget.
                try:
                    client.agent.remove_remote_agent(agent_name)
                except Exception:
                    pass  # never added, or already gone
                try:
                    client.agent.add_remote_agent(conn_info)
                except Exception as exc:
                    psrl_logger.warning(
                        f"{self.log_prefix}: could not load update destination "
                        f"{agent_name}: {exc!r}"
                    )
            payload = encode(
                NotifKind.UPDATE_INFO,
                {name: sources[name] for name in entry["sources"]},
            )
            client.agent.send_notif(agent_name, payload)
            sent_bytes += len(payload)
        return {"rank": self.rank, "agents": len(mine), "bytes": sent_bytes}

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
                # Under direct supply the rollouts read these tensors in place, at times
                # this worker does not control, so anything exposed in a corrected form
                # must stay corrected rather than being reverted after a push.
                resident_exposure=not has_ps_data_plane(self.psrl_config.ps_mode),
            )
            self.unified_state_dict = conversion_result.state_dict
            self.local_sharding_dict = conversion_result.sharding_dict
            self.param_sync_plan = conversion_result.sync_plan
        else:
            raise NotImplementedError(f"nixl_convert_params does not support strategy '{strategy}'.")

    def nixl_protocol(self, mode: str = "full", bootstrap_version: int | None = None):
        """Run the 8-step NIXL server handshake.

        Args:
            mode: `full` registers real tensors. `meta` registers only metadata before
                parameter server weights arrive.
            bootstrap_version: Version to stamp the already-loaded weights with, for
                direct supply. The caller supplies it because this runs inside the
                handshake, where asking the PSManager would deadlock against its own
                `wait_for_client_infos`.
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
        nixl_config = self.psrl_config.nixl
        if nixl_config.get("publish_rollout_dtype") is not None:
            self.nixl_storage_client.set_publish_cast(
                PrecisionType.to_dtype(nixl_config.publish_train_dtype),
                PrecisionType.to_dtype(nixl_config.publish_rollout_dtype),
            )
        quantization = QuantizationPlan.from_config(nixl_config.get("publish_quantization"))
        if quantization.is_quantized:
            self.nixl_storage_client.set_publish_quantization(
                quantization,
                reader_quantization=bool(nixl_config.get("reader_quantization", False)),
            )
        self.nixl_storage_client.register_local_tensors(
            self.unified_state_dict, unified_sharding_dict, meta_only=meta_only
        )
        # Stamp between registration and send: `register_local_tensors` is what builds
        # `local_client_info`, and `send_local_info` is what ships it to peers who then
        # cache it. Stamping outside this window does nothing useful -- before, there is
        # no metadata to write into; after, the copy peers hold already says "never
        # published", and the first pull is rejected as stale against a correct bootstrap.
        #
        # The version is passed in rather than read from the PSManager here: this runs
        # *inside* the handshake, and the manager is simultaneously blocked in
        # `wait_for_client_infos` waiting for this very rank. A `ray.get` to it deadlocks
        # until the handshake times out -- measured, not hypothesised.
        if bootstrap_version is not None and not meta_only:
            self.nixl_storage_client.publish_version(bootstrap_version)

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
        """Deregister NIXL tensors and move the trainer off the GPU (design §3.7.3).

        Deregistration comes first: the addresses are about to stop being valid, and the
        prepped handles built from them must go before the memory does.
        """
        if mode != "meta":
            self.nixl_storage_client.deregister_local_tensors()
        self.sleep_model(mode)

    def nixl_wake_up(self):
        """Bring the trainer back and re-register at whatever addresses it now has."""
        self.wake_up_model()
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)

    def _stage_policy(self, mode: str):
        from psrl.utils.nixl.stage_offload import stage_policy  # noqa: PLC0415

        offload_cfg = self.psrl_config.get("colocate_offload", {}) or {}
        # `meta` is the boot path: the actor was built empty and a parameter server will
        # fill it, so there is nothing to keep. Under direct supply there is no `meta`
        # sleep at all (the actor holds real weights from boot); see `ray_trainer`.
        has_ps = has_ps_data_plane(self.psrl_config.ps_mode) or mode == "meta"
        return stage_policy(
            has_ps_data_plane=has_ps,
            offload_optimizer=bool(offload_cfg.get("optimizer", True)),
            tms_pause=self.psrl_config.tms.range in ["train", "all"],
        )

    def _stage_tensors(self):
        """(weights, grads, optimizer) tensors whose storages a switch moves."""
        from psrl.utils.nixl.stage_offload import (  # noqa: PLC0415
            fsdp_weight_tensors,
            megatron_grad_tensors,
            megatron_weight_tensors,
            optimizer_tensors,
        )

        module = self.actor.engine.module
        strategy = self.config.actor.strategy
        if strategy in {"fsdp", "fsdp2"}:
            weights = fsdp_weight_tensors(module)
            grads = [p.grad for p in module.parameters() if p.grad is not None]
        elif strategy == "megatron":
            weights = megatron_weight_tensors(module)
            grads = megatron_grad_tensors(module)
        else:
            raise NotImplementedError(f"stage offload does not support strategy '{strategy}'.")
        return weights, grads, optimizer_tensors(self.actor.engine.optimizer)

    def sleep_model(self, mode: str = "full"):
        """Free the trainer's GPU memory for the rollout, keeping what has no other home.

        Weights are copied to host under direct supply (the trainer is the only copy) and
        released under a parameter server (the switch-back pulls them). Optimizer state
        is copied to host in every mode: fp32 masters and Adam moments exist nowhere else.
        Both used to be dropped -- the weights by the storage resize, the optimizer by a
        TMS pause configured without CPU backup.
        """
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} sleep")

        policy = self._stage_policy(mode)
        weights, grads, opt_tensors = self._stage_tensors()

        # NOTE(lhy): Empty the PyTorch cache before pausing TMS to prevent duplicate
        # `cuMemRelease` calls from the mempool and pause paths.
        aggressive_empty_cache(force_sync=True)
        kept, dropped = self._stage.offload(weights=weights, grads=grads, optimizer=opt_tensors, policy=policy)
        if policy.tms_pause:
            torch_memory_saver.pause()
        torch.cuda.empty_cache()
        psrl_logger.info(
            f"Stage offload ({policy.describe()}): kept {kept.bytes / 2**30:.2f} GiB on host "
            f"in {kept.seconds:.3f}s ({kept.gbps:.1f} GB/s), released "
            f"{dropped.bytes / 2**30:.2f} GiB; device free +{(kept.freed + dropped.freed) / 2**30:.2f} GiB"
        )

        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"After TrainWorker_R{self.rank} sleep")

    def wake_up_model(self):
        """Reallocate, then copy back whatever `sleep_model` kept.

        Kept storages are restored before dropped ones so the weights a parameter-server
        pull is about to overwrite do not compete with the bytes nobody can recreate.
        """
        if self.memory_logger is not None:
            self.memory_logger.log_now(prefix=f"Before TrainWorker_R{self.rank} wake_up")

        if self.psrl_config.tms.range in ["train", "all"]:
            torch_memory_saver.resume()
        restored = self._stage.restore()
        if restored is not None:
            psrl_logger.info(
                f"Stage restore: {restored.bytes / 2**30:.2f} GiB from host in "
                f"{restored.seconds:.3f}s ({restored.gbps:.1f} GB/s)"
            )

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
            # C3: record the publication point on this path too, so the boundary trace
            # does not depend on which ps_mode is configured.
            self.step_boundaries.on_train_step_end(version=next_version)

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
                        # Nothing to set: `load_weight` was a field patched into the
                        # vendored verl checkout, and upstream's Megatron engine has no
                        # equivalent -- it always loads, from a distributed checkpoint or
                        # from HF.
                        #
                        # Letting it load is safe rather than merely tolerable. The point
                        # of "empty" is to skip a load that a later checkpoint restore
                        # overwrites, so the cost is startup time, not correctness -- and
                        # under direct supply the train ranks *are* the weight source, so
                        # initialising them with real weights is the conservative outcome.
                        # The failure this avoids is the opposite one: silently skipping
                        # the load would serve uninitialised weights over RDMA.
                        pass

            try:
                ActorRolloutRefWorker.init_model(self)
            finally:
                if skip_load_weight:
                    OmegaConf.set_struct(self.config, True)
                    with open_dict(self.config):
                        if strategy in {"fsdp", "fsdp2"}:
                            self.config.actor.fsdp_config.load_weight = True
                        # No megatron branch: upstream verl has no `load_weight` field to
                        # restore, and writing one into a struct-mode config raises. See
                        # the matching comment where it would have been set to False.

            self._maybe_use_per_rank_checkpointing()

    def _maybe_use_per_rank_checkpointing(self) -> None:
        """No-op: kept for backward compatibility, replaced by nvrx async DCP.

        Previously swapped in ``PSRL_PerRankMegatronCheckpointManager`` when
        ``use_dcp_save=False``, because the mcore fork-based async writer
        corrupted UCX endpoint addresses read by the NIXL progress thread.
        That constraint is gone: ``save_dist_checkpointing`` now uses the nvrx
        strategy whose persistent worker is spawn-based and never inherits the
        parent's UCX / NIXL context.  The progress thread stays on regardless
        of ``use_dcp_save``.  This method is intentionally left as a no-op
        rather than deleted so callers do not need to be touched.
        """

    def _write_gate_enabled_here(self) -> bool:
        return (
            self._is_actor
            and self.actor is not None
            and self.psrl_config.nixl.get("await_write_gate") is True
        )

    def _install_optimizer_write_gate(self) -> None:
        """Close the pulled version before the optimizer writes its parameters (v2).

        `engine.optimizer_step` is the point where Megatron's master→bf16 copy-back and
        FSDP2's in-place update both happen, and it runs once per mini-batch. The gate is
        opened on the first call of an update and stays open until publish, because every
        mini-batch overwrites the same buffer while only one version is announced.
        """
        if not self._write_gate_enabled_here():
            return
        engine = self.actor.engine
        if getattr(engine, "_psrl_write_gate_installed", False):
            return
        original = engine.optimizer_step

        @functools.wraps(original)
        def gated_optimizer_step(*args, **kwargs):
            self._open_write_gate_once()
            return original(*args, **kwargs)

        engine.optimizer_step = gated_optimizer_step
        engine._psrl_write_gate_installed = True
        psrl_logger.info(f"{self.log_prefix}: installed the optimizer write gate.")

    def _open_write_gate_once(self) -> None:
        """Announce the upcoming write once per update, then synchronise the actor ranks.

        Only rank 0 talks to the manager; every rank waits at a barrier so no rank can
        start writing its shard before the epoch is recorded. The barrier is the same kind
        of collective the optimizer already participates in, and the manager's wait is
        bounded by `preempt_ack_timeout_ms`.
        """
        if self._write_gate_open or not self._write_gate_enabled_here():
            return
        if self.is_train_representative_rank:
            ray.get(self.train_interface.ps_manager_handle.begin_write.remote())
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        self._write_gate_open = True

    def _close_write_gate(self, published: bool) -> None:
        """Finish the update. A publish reopens the version; an aborted update stays closed."""
        if not self._write_gate_open:
            return
        self._write_gate_open = False
        if published or not self.is_train_representative_rank:
            return
        # The parameters were written but no version was published. Opening the buffer
        # would let readers commit a mixture, so keep it closed for the next write.
        psrl_logger.warning(
            f"{self.log_prefix}: update finished without publishing; keeping the version "
            "closed so readers cannot serve the partially updated parameters."
        )
        ray.get(self.train_interface.ps_manager_handle.abort_write.remote())

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def update_actor(self, data: TensorDict) -> TensorDict:
        """Train the actor for one step, then push updated weights to the PS."""
        self._write_gate_open = False
        self._install_optimizer_write_gate()
        published = False
        try:
            with log_dual_events("Train actor", psrl_logger, event_type=EventType.TRAIN):
                output = ActorRolloutRefWorker.update_actor(self, data)
            # The actor's parameters changed; any cached ref snapshot is now stale.
            if self._ref_backup is not None:
                self._ref_backup.mark_modified()
            push_model = tu.pop(data, key="push_model", default=True)
            if not push_model:
                return output

            torch.cuda.synchronize()
            # v2: with the write gate enabled the epoch protocol already excludes readers
            # from the live buffer, and publish is a pure version flip (no bytes move at
            # matched precision). Taking the shared/exclusive lock here would make publish
            # wait for readers that are repairing from peers, which is the delay the gate
            # exists to avoid -- and can deadlock a reader waiting for OPEN. Cast/PS paths
            # keep the lock because their publish refreshes a shared copy.
            gate_enabled = self.psrl_config.nixl.get("await_write_gate") is True
            context_manager = (
                exclusive_push_model_context(self.train_interface.ps_manager_handle)
                if self.is_train_representative_rank and not gate_enabled
                else nullcontext()
            )
            with context_manager:
                with log_dual_events("Push model", psrl_logger, event_type=EventType.PUSH):
                    PSRL_BaseTrainWorker.push_model(self)
            published = True
            return output
        finally:
            self._close_write_gate(published)

    def capture_ref_backup(self) -> None:
        """Snapshot the actor's current weights as the frozen reference policy.

        Called once after init_model() when the actor holds the initial checkpoint.
        After this, compute_ref_log_prob() can restore the initial weights in place for
        a forward pass without a separate model copy (design §3.10.2).

        WeightBackup.capture() is idempotent: if called again it re-captures only when
        the weights are marked dirty (i.e., after an optimizer step). So calling this
        after a warm-start load is safe and simply refreshes the snapshot.
        """
        if not (self._is_actor and self.actor is not None):
            return
        from psrl.utils.nixl.weight_backup import WeightBackup  # noqa: PLC0415

        if self._ref_backup is None:
            params = self.actor.engine.module.parameters()
            self._ref_backup = WeightBackup(params)
            psrl_logger.info(
                f"{self.log_prefix}: WeightBackup built, "
                f"nbytes={self._ref_backup.nbytes / 1e9:.2f} GB"
            )
        nbytes = self._ref_backup.capture("ref")
        psrl_logger.info(
            f"{self.log_prefix}: ref snapshot captured, {nbytes / 1e9:.2f} GB"
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @gpu_memory_logger_decorator(log_only_rank_0=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        """Compute reference-policy log-probs using the frozen host snapshot.

        Swaps the initial-checkpoint weights into the actor's storages in place,
        runs the forward pass, then swaps the current training weights back.
        Cost: one D2H + two H2D at ~46 GB/s per rank, replacing a second GPU copy
        of the model (design §3.10.2).
        """
        if self._ref_backup is None:
            raise RuntimeError(
                "capture_ref_backup() must be called before compute_ref_log_prob(). "
                "Call it once after init_model() completes."
            )
        with self._ref_backup.swapped("ref") as stats:
            output = ActorRolloutRefWorker.compute_log_prob(self, data)
        psrl_logger.info(
            f"{self.log_prefix}: ref forward done — swap {stats.seconds * 1e3:.0f}ms, "
            f"load {stats.loaded_bytes / 1e9:.2f} GB"
        )
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
