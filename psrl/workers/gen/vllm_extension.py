import logging
import os
import time

import torch
from omegaconf import DictConfig
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm_patches.interfaces import supports_weight_layout

from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import gen_client_name
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.nixl import (
    NIXLClientType,
    NIXLStorageClient,
)

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

    def finalize_weight_loading(self) -> None:
        """
        Convert freshly loaded weights to runtime format and wait for completion.
        """
        model = self.model_runner.model
        if isinstance(model, CUDAGraphWrapper):
            model = model.unwrap()
        process_weights_after_loading(
            model,
            self.model_runner.vllm_config.model_config,
            self.model_runner.device,
        )
        torch.cuda.synchronize()

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
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_convert_params(self, model_config: DictConfig):
        """Convert the model parameters to unified format.

        Args:
            config (DictConfig): Configuration object containing training settings.
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
        )

    def nixl_protocol(self, model_config: DictConfig, mode: str = "full"):
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
            self.nixl_convert_params(model_config)
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

    def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for infos of updated clients for global synchronization.

        Args:
            info_num (int): Number of infos to wait for.
        """
        self.nixl_storage_client.wait_for_update_infos(info_num)

    def nixl_pull_model_core(self, ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
        """Pull the model parameters from PS workers via NIXL.

        Args:
            ps_nixl_agent_names (list[str]): List of PS NIXL agent names
            ps_nixl_train_storage_client_names (list[str]): List of PS NIXL train storage client names
        """
        if not hasattr(self, "pull_times"):
            self.pull_times = 0
        self.pull_times += 1
        wait_operations = []
        time_start = time.time()
        for key in self.unified_state_dict:
            for target_agent_name, target_client_name in zip(ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
                shards_to_transfer = self.nixl_storage_client.client_read(
                    target_agent_name,
                    target_client_name,
                    key,
                    f"gen_pull_{self.pull_times}",
                )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, target_client_name, shards_to_transfer))
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        for key, target_client_name, shards_to_transfer in wait_operations:
            self.nixl_storage_client.wait(
                key,
                f"gen_pull_{self.pull_times}",
                "READ",
                target_client=target_client_name,
            )
        self.nixl_storage_client.merge_and_finish_cached_xfer()
        self.cuda_synchronize()
        self.nixl_storage_client.clear_intermediate_cached_data()
        time_end = time.time()
        psrl_logger.info(
            f"{self.nixl_storage_client}: NIXL pull model core done ({self.pull_times} times). "
            f"time: {time_end - time_start}s"
        )

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

    # --- LMCache KV cache management methods ---
    # Invoked via `collective_rpc` from `KVCacheManager`.

    def _get_lmcache_engine(self):
        """
        Return the `LMCacheEngine` from the active KV transfer group.

        vLLM initializes the transfer group when `kv_transfer_config` is set.

        Returns:
            LMCacheEngine: The `lmcache_engine` from the active KV transfer group.
        """
        connector = get_kv_transfer_group()
        assert connector is not None, (
            "KV transfer group is None. Ensure kv_transfer_config is set during vLLM engine initialization."
        )
        assert hasattr(connector, "_lmcache_engine"), (
            f"Connector lacks _lmcache_engine: type={type(connector).__name__!r}. Expected LMCacheConnectorV1."
        )
        engine = connector._lmcache_engine.lmcache_engine
        assert engine is not None, "LMCacheEngine lmcache_engine is None."
        return engine

    def _get_lmcache_chunk_keys(self, tokens: list[int]) -> list:
        """
        Convert a token sequence to a list of `CacheEngineKey` objects.

        Uses the token database attached to the `LMCacheEngine`.

        Args:
            tokens (list[int]): Full token sequence.

        Returns:
            list: `CacheEngineKey` objects for each chunk.
        """
        engine = self._get_lmcache_engine()
        triples = engine.token_database.process_tokens(tokens)
        # `process_tokens` returns `(start, end, key)` triples. Extract the keys.
        return [key for _, _, key in triples]

    def _get_lmcache_total_bytes(self) -> int:
        """
        Return the total byte capacity of the LMCache local CPU backend allocator.

        Returns:
            int: Total bytes, or 0 if the allocator does not expose `total_size`.
        """
        engine = self._get_lmcache_engine()
        backend = engine.storage_manager.local_cpu_backend
        allocator = backend.memory_allocator
        # `MixedMemoryAllocator` / `PagedCpuGpuMemoryAllocator` expose `total_size`.
        return int(getattr(allocator, "total_size", 0))

    def lmcache_pin_backend(self, tokens: list[int]) -> int:
        """
        Pin the cached backend chunks for `tokens` to prevent LRU eviction.

        A held chunk has `ref_count == 2` and cannot be evicted. Exhausting all
        evictable chunks can block retrievals and reject new stores.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of chunks newly pinned (`ref_count` incremented).
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_memory_objs"):
            # Map each PSRL-pinned cache key to its memory object.
            self._psrl_pinned_memory_objs: dict = {}

        engine = self._get_lmcache_engine()
        keys = self._get_lmcache_chunk_keys(tokens)
        backend = engine.storage_manager.local_cpu_backend

        pinned = 0
        for key in keys:
            if key in self._psrl_pinned_memory_objs:
                # Repeated pins must not increment `ref_count`.
                continue
            # `get_blocking` raises `ref_count` to make the object ineligible for eviction.
            memory_obj = backend.get_blocking(key)
            if memory_obj is None:
                # Chunk not present in backend (prefix may be shorter than key list).
                continue
            self._psrl_pinned_memory_objs[key] = memory_obj
            pinned += 1

        # High pin pressure can reject stores and block retrievals.
        hot_cache_size = len(backend.hot_cache)
        if hot_cache_size > 0:
            psrl_pinned_total = len(self._psrl_pinned_memory_objs)
            usage_pct = psrl_pinned_total / hot_cache_size
            if usage_pct > 0.8:
                psrl_logger.warning(
                    f"[LMCache] Backend pin pressure: PSRL holds {psrl_pinned_total}/"
                    f"{hot_cache_size} hot_cache chunks pinned ({usage_pct:.0%}). "
                    "New store() calls may fail and retrieve() calls may spin "
                    "until a PSRL pin is released via lmcache_unpin_backend."
                )
        return pinned

    def lmcache_unpin_backend(self, tokens: list[int]) -> int:
        """
        Unpin the cached backend chunks for `tokens`, allowing LRU eviction.

        Only chunks tracked in `_psrl_pinned_memory_objs` are released, preventing
        interference with active request references.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of chunks whose `ref_count` was decremented.
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_memory_objs"):
            self._psrl_pinned_memory_objs: dict = {}

        keys = self._get_lmcache_chunk_keys(tokens)
        freed = 0
        for key in keys:
            memory_obj = self._psrl_pinned_memory_objs.pop(key, None)
            if memory_obj is None:
                continue
            # The hot cache and PSRL each hold one reference.
            assert memory_obj.get_ref_count() > 1, (
                f"Invalid backend reference count before unpin: ref_count={memory_obj.get_ref_count()}. "
                "Expected more than one reference. "
                "Possible double-unpin or external ref_count corruption."
            )
            memory_obj.ref_count_down()
            freed += 1
        return freed

    def lmcache_clear_all_from_backend(self) -> None:
        """
        Remove all cached KV chunks from the LMCache CPU backend.

        Weight updates must clear stale KV entries before subsequent requests.
        """
        engine = self._get_lmcache_engine()
        engine.clear()
