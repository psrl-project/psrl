import logging
import os

import torch
from omegaconf import DictConfig
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm_patches.interfaces import supports_weight_layout

from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import gen_client_instance, gen_client_name
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.nixl import (
    NIXLClientType,
    NIXLStorageClient,
)
from psrl.utils.nixl.pull_engine import ChunkUnavailable, PullPreempted, PullVerdict, TornPull
from psrl.utils.quantization import QuantizationPlan

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class vLLMWorkerExtension(vLLMColocateWorkerExtension):
    def load_weights(self, weights, blocking: bool = True):
        """
        Load weights into the vLLM model runner.

        This method rebuilds the weights using the provided function and arguments stemming from `reduce_tensor` calls,
        transfers them to the current CUDA device, and loads them into the vLLM model runner.
        If the weight is a DTensor, it converts it to a full tensor before loading.
        If `blocking` is True, it ensures that all operations are completed before returning.
        If an error occurs during the process, it logs the error and returns None.

        Args:
            weights (List[tuple]): A list of tuples where each tuple contains:
                - name (str): The name of the weight.
                - handle (tuple): A tuple containing the function and its arguments to rebuild the weight.
            blocking (bool): If True, will block until all operations are completed.

        Returns:
            loaded_params: The loaded parameters from the model runner.

        Raises:
            Exception: If there is an error during the loading process.
        """

        def rebuild_weights_generator():
            current_device = torch.cuda.current_device()
            for name, handle in weights:
                func, args = handle
                list_args = list(args)
                # CPU bundle: (type(tensor), storage, metadata)
                if len(list_args) == 3:
                    tensor = func(*list_args)
                    tensor = tensor.to(current_device, non_blocking=True)
                    if isinstance(tensor, DTensor):
                        tensor = tensor.full_tensor()
                else:
                    list_args[6] = get_device_id()
                    tensor = func(*list_args)
                    if isinstance(tensor, DTensor):
                        tensor = tensor.full_tensor()
                yield (name, tensor)

        rebuild_weights = rebuild_weights_generator()
        torch.cuda.synchronize()
        loaded_params = self.model_runner.model.load_weights(weights=rebuild_weights)
        if blocking:
            # Ensure all operations are completed before returning
            torch.cuda.synchronize()
        return loaded_params

    # ----------------------------- NIXL Related -----------------------------
    # Because the model is on another process since vllm V1, we must call the nixl methods via rpc
    def get_instance_local_rank(self):
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group().rank

    def get_instance_local_tp_rank(self):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()

    def get_instance_local_ep_rank(self):
        from vllm.distributed.parallel_state import get_ep_group

        try:
            return get_ep_group().rank_in_group
        except AssertionError:
            # Non-MoE models do not initialize an EP group.
            return 0

    def init_nixl_client(
        self,
        nixl_config: DictConfig,
        replica_idx: int,
        logging_path: str | None = None,
        pairwise_descriptors: bool = True,
        segment_registration: bool | None = None,
    ):
        # NIXL attributes
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        # Initialize the NIXL client
        self.nixl_storage_client = NIXLStorageClient(
            client_name=gen_client_name(replica_idx, self.get_instance_local_rank()),
            server_name=NIXL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=nixl_config,
            replica_idx=replica_idx,
            worker_index=self.get_instance_local_rank(),
            # client_group_id=instance_id,
            logging_path=logging_path,
            pairwise_descriptors=pairwise_descriptors,
            segment_registration=segment_registration,
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")
        if bool(nixl_config.get("reader_quantization", False)):
            plan = QuantizationPlan.from_config(nixl_config.get("publish_quantization"))
            self.nixl_storage_client.set_reader_quantization(plan)
            psrl_logger.info("NIXL reader-side FP8 quantization enabled.")

    def nixl_convert_params(self, model_config: DictConfig, quantization: dict | None = None):
        """Convert the model parameters to unified format.

        Args:
            model_config (DictConfig): Configuration object containing model settings.
            quantization (dict | None): Resolved wire-format plan from the driver.
        """
        vllm_model = self.model_runner.model
        if isinstance(vllm_model, CUDAGraphWrapper):
            vllm_model = vllm_model.unwrap()
        param_mapping = (
            None if supports_weight_layout(vllm_model) else create_parameter_mapping(type(vllm_model), model_config)
        )
        self.unified_state_dict, self.local_sharding_dict = convert_vllm_inplace(
            vllm_model,
            tp_rank=self.get_instance_local_tp_rank(),
            ep_rank=self.get_instance_local_ep_rank(),
            parameter_mapping=param_mapping,
            quantization=QuantizationPlan.from_config(quantization),
        )
        unit_layers = sorted(
            {
                sharding.quant_unit
                for sharding in self.local_sharding_dict.values()
                if getattr(sharding, "quant_unit", None) is not None
            }
        )
        if unit_layers and not self.nixl_storage_client._reader_quantization:
            raise ValueError(
                f"This model fuses {len(unit_layers)} parameter(s) whose checkpoint views are smaller "
                f"than one quantization tile, first {unit_layers[0]!r}. The rollout must quantize them "
                "as one unit, which requires psrl.nixl.reader_quantization=true."
            )

    def nixl_protocol(self, model_config: DictConfig, mode: str = "full", quantization: dict | None = None):
        """Run the NIXL server protocol.

        Args:
            model_config (DictConfig): Configuration object containing training settings.
            mode (str): Mode of registration, either 'meta' or 'full'.
                'meta' mode converts to meta tensors and skip registering their memory.
                'full' mode converts to full tensors.

            NOTE: ps storage may init with meta tensors, the register step would be different.
        """
        # Register the state dict and sharding dict to the NIXL client
        meta_only = mode == "meta"
        if self.unified_state_dict is None or self.local_sharding_dict is None:
            self.nixl_convert_params(model_config, quantization)
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(self.local_sharding_dict)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        # psrl_logger.info(f"unified_sharding_dict: {unified_sharding_dict}")
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

    def nixl_register_after_wake_up(self):
        """Register the model parameters to NIXL after wake up from sleep.

        After sleep/wake_up, the physical memory backing the model weights
        has changed while virtual addresses remain the same. This method performs
        local re-registration:
        1. Reset nixl agent (clears UCX rcache)
        2. Re-registers memory with new physical pages (generates new rkeys)
        """
        torch.cuda.synchronize()
        # Reset nixl agent and reregister to handle physical memory changes
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)

    def nixl_deregister(self):
        """Deregister the model parameters from NIXL."""
        self.nixl_storage_client.deregister_local_tensors()

    def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        """
        Send local NIXL info to the specified destination agents.
        """
        if isinstance(dst_agent_names, str):
            dst_agent_names = [dst_agent_names]
        self.nixl_storage_client.send_local_info_to(dst_agent_names)

    def nixl_wait_for_update_infos(self, info_num: int, timeout: float | None = None):
        """Wait for infos of updated clients for global synchronization.

        Args:
            info_num (int): Number of infos to wait for.
            timeout (float | None): Bound on the wait; None uses the handshake budget.
        """
        self.nixl_storage_client.wait_for_update_infos(info_num)

    def pulled_version_set(self) -> set[int]:
        """Versions the last pull read, for the caller to compare against the announced one."""
        return set(getattr(self, "pulled_version", set()))

    def nixl_mark_sources_unavailable(self, sources: list[str]) -> None:
        """Stop reading from train ranks the coordinator has declared dead (design 3.5.7, D7)."""
        self.nixl_storage_client.pull_engine().mark_unavailable(sources)

    def nixl_pull_model_core(
        self,
        epoch: int,
        participant_instances: list[int] | None = None,
        version_holders: frozenset[str] | None = None,
        target_version: int = -1,
        pull_epoch: int | None = None,
    ) -> PullVerdict:
        """
        Pull the latest weights through the NIXL pull engine.

        Returns a `PullVerdict` rather than a bool, because a pull that did not land splits
        into two cases the caller must answer differently. One wrote nothing, so the
        instance is still consistent at its current version (D6). The other mixed versions,
        so only another whole pull can repair it. The verdict carries the observed versions
        so the caller can also test them against the publisher's invalid set.

        Args:
            epoch (int): Identifier of this sync, shared by every syncing replica and never reused.
            participant_instances (list[int] | None): Replica indices syncing together. None means all.
            version_holders (frozenset[str] | None): Trainer client names that have published
                ``target_version``.  Forwarded to ``pull_engine.pull`` for D1 version-aware source
                filtering.  None means no filtering (backward-compatible default).
            target_version (int): Model version being pulled, for logging.  -1 means unknown.
            pull_epoch (int | None): Writer epoch recorded by the coordinator when this pull
                was admitted. A newer epoch preempts the pull; the engine acknowledges over
                the notification channel and repairs unfinished keys from certified holders.
        """
        self.pull_times = getattr(self, "pull_times", 0) + 1
        allowed = None if participant_instances is None else set(participant_instances)
        participants = None
        if allowed is not None:
            plan = self.nixl_storage_client.comm_plan
            peers = {
                w
                for per_key in plan.peer_groups.values()
                for per_shard in per_key.values()
                for wanters in per_shard.values()
                for w in wanters
            }
            participants = {name for name in peers if gen_client_instance(name) in allowed}
            participants.add(self.nixl_storage_client.client_name)
        try:
            report = self.nixl_storage_client.pull_engine().pull(
                epoch,
                participants,
                version_holders=version_holders,
                target_version=target_version,
                pull_epoch=pull_epoch,
                on_preempt_ack=self.nixl_storage_client.send_pull_preempt_ack,
            )
        except ChunkUnavailable as unavailable:
            # Nothing was written, so the instance is still consistent at its current version.
            psrl_logger.warning(
                f"{self.nixl_storage_client.client_name}: pull {self.pull_times} skipped: {unavailable}"
            )
            # D7: sources that failed during rerouting are now confirmed dead; exclude them
            # from every future pull so they are not retried round after round.
            if unavailable.dead:
                self.nixl_storage_client.pull_engine().mark_unavailable(unavailable.dead)
            self._sync_pulled_versions(set())
            return PullVerdict(landed=False, versions=frozenset(), reason="unavailable")
        except TornPull as torn:
            # Earlier waves landed, so this instance now mixes versions. Reporting it lets
            # the caller re-pull, which overwrites every key.
            psrl_logger.warning(f"{self.nixl_storage_client.client_name}: pull {self.pull_times} torn: {torn}")
            self._sync_pulled_versions(set())
            return PullVerdict(landed=False, versions=frozenset(), reason="torn")
        except PullPreempted as preempted:
            # The writer moved while this pull was reading and no certified holder could
            # supply every unfinished key. The local bytes may mix versions; the caller
            # waits for the next version and re-pulls it whole.
            psrl_logger.warning(
                f"{self.nixl_storage_client.client_name}: pull {self.pull_times} could not be "
                f"repaired after preemption: {preempted}"
            )
            self._sync_pulled_versions(set())
            return PullVerdict(landed=False, versions=frozenset(), reason="preempted")
        # D7: sources that failed mid-pull and were rerouted to a DP replica are confirmed
        # dead; promote them so the next pull skips them rather than re-discovering the
        # failure through another timed-out read.
        if report.failed_sources:
            self.nixl_storage_client.pull_engine().mark_unavailable(report.failed_sources)
        self.cuda_synchronize()
        self._refresh_derived_weights()
        self._sync_pulled_versions(report.versions)
        # Every shard of one pull must come from one version. More than one means some
        # sources had republished mid-pull. Not asserted here: the caller retries.
        if len(report.versions) > 1:
            psrl_logger.warning(
                f"{self.nixl_storage_client.client_name}: pull {self.pull_times} assembled shards from "
                f"versions {sorted(report.versions)}."
            )
            return PullVerdict(landed=False, versions=frozenset(report.versions), reason="torn")
        psrl_logger.info(
            f"{self.nixl_storage_client.client_name}: pull {self.pull_times} (epoch {epoch}) took "
            f"{report.seconds:.3f}s, upstream {report.upstream_bytes / 1e9:.2f} GB in {report.upstream_seconds:.3f}s, "
            f"peer {report.peer_bytes / 1e9:.2f} GB, {report.descriptors} descriptors, "
            f"ipc {report.ipc_bytes / 1e9:.2f} GB ({report.ipc_peer_bytes / 1e9:.2f} GB cross-GPU), "
            f"{report.followers} follower(s), {report.fallback_keys} fallback key(s), "
            f"{report.takeover_keys} key(s) taken over from a replica"
            + (", repaired from holders after preemption." if report.repaired else ".")
        )
        return PullVerdict(landed=True, versions=frozenset(report.versions), repaired=report.repaired)

    def _refresh_derived_weights(self) -> None:
        """Re-run vLLM's post-load hooks after the pull has rewritten the weights.

        vLLM derives runtime state from the parameters it loads: MLA absorbs
        `kv_b_proj` into `W_UK_T`/`W_UV`, fused MoE builds its kernel from the expert
        tensors, and quant methods keep stamps that are not parameters. A pull replaces
        the bytes underneath all of those, so the state has to be rebuilt.

        vLLM 0.30 provides the reload path for exactly this: `weights_already_processed()`
        tells each quant method that the tensors are already in runtime format, so it
        skips the transforms that would otherwise re-quantise or re-lay-out them, and
        only restores the non-exported state and rebuilds kernels. Methods that do not
        declare support raise, which is the signal that the method cannot be reloaded in
        place; the attribute-based MLA refresh below remains as a fallback so the pull
        still lands (with a warning naming the method).
        """
        model = self.model_runner.model
        model_config = getattr(self.model_runner, "model_config", None)
        if model_config is not None:
            try:
                from vllm.model_executor.model_loader.utils import process_weights_after_loading
                from vllm.model_executor.utils import weights_already_processed

                with weights_already_processed():
                    process_weights_after_loading(model, model_config, self.device)
                return
            except Exception as exc:
                psrl_logger.warning(
                    f"{self.nixl_storage_client.client_name}: vLLM's post-load reload hook failed "
                    f"({exc!r}); falling back to derived-weight refresh only."
                )
        dtype = getattr(model_config, "dtype", None)
        for module in model.modules():
            if not (hasattr(module, "W_UK_T") and hasattr(module, "kv_b_proj")):
                continue
            if dtype is None:
                dtype = next(module.parameters()).dtype
            module.process_weights_after_loading(dtype)

    def _sync_pulled_versions(self, versions) -> None:
        """Record what the last pull read, for `pulled_version_set`.

        An empty set is stored for a torn or unavailable pull: nothing about that pull
        should be compared against a version, and the caller already has the verdict.
        """
        self.pulled_version = set(versions)

    def get_total_kv_cache_tokens(self) -> int:
        """Total KV-cache token capacity of this instance, post-allocation.

        Uses vLLM's own `max_concurrency * max_model_len` formula (the same one
        behind the "GPU KV cache size: N tokens" startup log line), which is
        exact for hybrid/sliding-window layouts and already accounts for
        DCP/PCP sharding, unlike a naive `num_gpu_blocks * block_size`.
        """
        assert hasattr(self, "vllm_config"), "vllm_config must be set"
        kv_cache_config = getattr(self.model_runner, "kv_cache_config", None)
        if kv_cache_config is None or not kv_cache_config.kv_cache_groups:
            return 0
        max_concurrency = get_max_concurrency_for_kv_cache_config(self.vllm_config, kv_cache_config)
        return int(max_concurrency * self.vllm_config.model_config.max_model_len)

    def cuda_synchronize(self):
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.cuda_synchronize: {e}") from e
        return None
