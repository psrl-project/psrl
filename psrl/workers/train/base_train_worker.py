import logging
import os
import threading
import time
from dataclasses import dataclass

import ray
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from psrl.utils.converter.param_sync import ParamSyncPlan
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@dataclass
class TrainInterface:
    """Info for the PSRL TrainWorker."""

    ps_manager_handle: ray.actor.ActorHandle


class PSRL_BaseTrainWorker:
    """Provide shared parameter synchronization operations for train workers."""

    def __init__(
        self,
        worker_rank: int,
        worker_world_size: int,
        psrl_config: DictConfig,
        train_interface: TrainInterface,
    ):
        self.worker_rank = worker_rank
        self.worker_world_size = worker_world_size
        self.psrl_config = psrl_config
        self.train_interface = train_interface
        self.node_id = None
        self.nixl_storage_client = None
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        self.param_sync_plan = ParamSyncPlan()
        self._cached_ps_nixl_agent_names = None
        self._cached_ps_nixl_train_storage_client_names = None
        self._cached_ps_worker_handles: dict[str, ray.actor.ActorHandle] = {}
        # Cache for non-persistent named buffers fetched from PS (populated lazily).
        self._cached_non_persistent_buffers: dict[str, torch.Tensor] | None = None
        self.nixl_wait_thread = None  # Single thread for all wait operations
        self.nixl_wait_thread_lock = threading.Lock()
        self.nixl_wait_completed = threading.Event()

        self.log_prefix = f"BaseTrainWorker_R{self.rank}"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized on {ray.get_runtime_context().get_node_id()}.")

    def get_node_id(self) -> str:
        """
        Get the node id of the train worker.
        """
        if self.node_id is not None:
            return self.node_id
        self.node_id = ray.get_runtime_context().get_node_id()
        return self.node_id

    @property
    def is_train_representative_rank(self) -> bool:
        """
        Check if the current rank is the representative rank.
        The representative rank is the rank 0 of the PS.
        """
        pass

    def get_replica_id(self) -> int:
        """
        Get the replica id (dp id) of the train worker.
        """
        pass

    def init_nixl_client(self):
        pass

    def nixl_protocol(self, mode: str = "full"):
        pass

    def nixl_sleep(self, mode: str = "full"):
        pass

    def sleep(self):
        pass

    def ray_push_model(self) -> None:
        pass

    def nixl_push_model(self) -> None:
        """
        Push the model weights to the PS via NIXL.

        Usage example:
            # Start the push operation (this will start a background wait thread)
            worker.nixl_push_model()

            # Do other work while push is happening in background...

            # Wait for all push operations to complete
            success = worker.wait_for_nixl_push_completion(timeout=60.0)
            if success:
                print("All NIsXL push operations completed successfully")
            else:
                print("Some NIXL push operations timed out")

            # Or check thread status
            status = worker.get_nixl_wait_thread_status()
            print(f"Thread alive: {status.get('alive', False)}")
        """
        assert self.nixl_storage_client is not None, "nixl_storage_client is not initialized."
        assert self.psrl_config.ps_mode in ("nixl_cpu", "nixl_gpu"), (
            "push_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode, "
            f"got: {self.psrl_config.ps_mode!r}."
        )
        ps_manager_handle = self.train_interface.ps_manager_handle
        psrl_logger.debug("Getting the current PS model version...")
        curr_ps_model_version = ray.get(ps_manager_handle.get_ps_model_version.remote(debug_info="base_train_worker"))
        next_ps_model_version = curr_ps_model_version + 1
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = ray.get(ps_manager_handle.get_ps_nixl_agent_names.remote())
        if self._cached_ps_nixl_train_storage_client_names is None:
            self._cached_ps_nixl_train_storage_client_names = ray.get(
                ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
            )
        psrl_logger.debug(
            f"Pushing model to the PS via NIXL. Version={next_ps_model_version}, "
            f"clients={len(self._cached_ps_nixl_train_storage_client_names)}."
        )

        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is not None and self.nixl_wait_thread.is_alive():
                raise RuntimeError(
                    "Previous NIXL wait thread is still running, "
                    "you should wait for it to complete before calling nixl_push_model again."
                )
            self.nixl_wait_thread = None
            self.nixl_wait_completed.clear()

        def wait_all_operations():
            try:
                # NOTE(lhy): Merge key and shard transfers on the PS side to avoid
                # flooding the Ray actor with one remote call per shard.
                ps_handle_to_precision_transfer_key_and_shards_list: dict[
                    str, list[tuple[str, list[tuple[int, ...]]]]
                ] = {}
                psrl_logger.debug(f"Starting to push model to the PS via NIXL for version {next_ps_model_version}...")
                for key in self.unified_state_dict:
                    wait_operations = []
                    for target_agent_name, target_client_name in zip(
                        self._cached_ps_nixl_agent_names,
                        self._cached_ps_nixl_train_storage_client_names,
                    ):
                        if target_client_name not in self._cached_ps_worker_handles:
                            self._cached_ps_worker_handles[target_client_name] = ray.get(
                                ps_manager_handle.get_ps_worker_handle.remote(target_client_name)
                            )
                        psrl_logger.debug(
                            f"Pushing key={key!r} to client={target_client_name!r} "
                            f"for version={next_ps_model_version}."
                        )
                        try:
                            shards_to_transfer = self.nixl_storage_client.client_write(
                                target_agent_name,
                                target_client_name,
                                key,
                                f"train_push_{next_ps_model_version}",
                            )
                        except Exception as e:
                            psrl_logger.error(
                                f"Error pushing key={key!r} to client={target_client_name!r} "
                                f"for version={next_ps_model_version}: {e!r}."
                            )
                            raise e
                        if len(shards_to_transfer) > 0:
                            wait_operations.append((key, target_client_name, shards_to_transfer))
                    psrl_logger.debug(
                        f"Waiting for NIXL operations. Count={len(wait_operations)}, version={next_ps_model_version}."
                    )
                    for wait_key, wait_target_client_name, wait_shards_to_transfer in wait_operations:
                        try:
                            self.nixl_storage_client.wait(
                                wait_key,
                                f"train_push_{next_ps_model_version}",
                                "WRITE",
                                target_client=wait_target_client_name,
                            )
                        except Exception as e:
                            psrl_logger.error(
                                f"Error waiting for key={wait_key!r} on client={wait_target_client_name!r} "
                                f"for version={next_ps_model_version}: {e!r}."
                            )
                            raise e
                        psrl_logger.debug(
                            f"NIXL wait completed for key={wait_key!r} on client={wait_target_client_name!r}."
                        )
                        if wait_target_client_name not in ps_handle_to_precision_transfer_key_and_shards_list:
                            ps_handle_to_precision_transfer_key_and_shards_list[wait_target_client_name] = []
                        ps_handle_to_precision_transfer_key_and_shards_list[wait_target_client_name].append(
                            (wait_key, wait_shards_to_transfer)
                        )
                precision_transfer_futures = []
                for (
                    target_client_name,
                    precision_transfer_key_and_shards_list,
                ) in ps_handle_to_precision_transfer_key_and_shards_list.items():
                    precision_transfer_futures.append(
                        self._cached_ps_worker_handles[target_client_name].transfer_train_to_gen_merged.remote(
                            precision_transfer_key_and_shards_list
                        )
                    )
                ray.get(precision_transfer_futures)
                psrl_logger.debug("Starting to push model tag to the PS...")
                assert dist.is_initialized(), "Pytorch distributed is not initialized."
                dist.barrier()
                psrl_logger.debug("Barrier done, now pushing model tag to the PS on the representative rank...")
                if self.worker_rank == 0:
                    ray.get(ps_manager_handle.push_model_state_dict_nixl.remote(next_ps_model_version))
                self.nixl_storage_client.clear_intermediate_cached_data()
                self.nixl_wait_completed.set()
                psrl_logger.debug(
                    f"All NIXL push operations completed. Model version={next_ps_model_version} was pushed to the PS."
                )
            except Exception as e:
                raise RuntimeError(f"Error in NIXL wait thread: {e}") from e

        wait_thread = threading.Thread(target=wait_all_operations, daemon=True)
        wait_thread.start()
        with self.nixl_wait_thread_lock:
            self.nixl_wait_thread = wait_thread

    def wait_for_nixl_push_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for the NIXL push wait thread to complete.

        Args:
            timeout (float, optional): Maximum time to wait in seconds. If None, wait indefinitely.

        Returns:
            bool: True if the thread completed successfully, False if timeout occurred or thread failed.
        """
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is None:
                psrl_logger.debug("No NIXL wait thread to wait for.")
                return True

            psrl_logger.debug("Waiting for NIXL wait thread to complete...")
            if timeout is not None:
                if self.nixl_wait_completed.wait(timeout=timeout):
                    self.nixl_wait_thread.join(timeout=1.0)  # Brief join to catch any exceptions
                    if self.nixl_wait_thread.is_alive():
                        psrl_logger.warning("NIXL wait thread is still alive after event was set.")
                        return False
                    psrl_logger.debug("NIXL wait thread completed successfully.")
                    return True
                else:
                    psrl_logger.warning("Timeout waiting for NIXL wait thread to complete.")
                    return False
            else:
                self.nixl_wait_thread.join()
                if self.nixl_wait_thread.is_alive():
                    psrl_logger.warning("NIXL wait thread is still alive after join.")
                    return False
                psrl_logger.debug("NIXL wait thread completed successfully.")
                return True

    def get_nixl_wait_thread_status(self) -> dict:
        """
        Get the status of the NIXL wait thread.

        Returns:
            dict: Dictionary containing thread status information.
        """
        with self.nixl_wait_thread_lock:
            if self.nixl_wait_thread is None:
                return {"has_thread": False, "alive": False, "completed": True}
            return {
                "has_thread": True,
                "alive": self.nixl_wait_thread.is_alive(),
                "completed": self.nixl_wait_completed.is_set(),
            }

    def push_model(self):
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            self.ray_push_model()
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            self.param_sync_plan.before_push(self.unified_state_dict)
            self.nixl_push_model()
            self.wait_for_nixl_push_completion()
            self.param_sync_plan.after_push(self.unified_state_dict)
        else:
            raise NotImplementedError(f"PSRL TrainWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")

    def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        """
        Send local NIXL info to the specified destination agent names.

        Args:
            dst_agent_names (str | list[str]): Destination agent name(s) to send local info to.
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

    def nixl_pull_model(self):
        """Pull the model from the NIXL storage client."""
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", (
            "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        )
        ps_manager_handle = self.train_interface.ps_manager_handle
        if self._cached_ps_nixl_agent_names is None:
            self._cached_ps_nixl_agent_names = ray.get(ps_manager_handle.get_ps_nixl_agent_names.remote())
        if self._cached_ps_nixl_train_storage_client_names is None:
            self._cached_ps_nixl_train_storage_client_names = ray.get(
                ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
            )
        self.nixl_pull_model_core(self._cached_ps_nixl_agent_names, self._cached_ps_nixl_train_storage_client_names)
        self.param_sync_plan.after_pull(self.unified_state_dict)

    def nixl_pull_model_core(self, ps_nixl_agent_names: list[str], ps_nixl_train_storage_client_names: list[str]):
        """
        Core logic for pulling the model from NIXL storage clients.

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
            for target_agent_name, target_client_name in zip(ps_nixl_agent_names, ps_nixl_train_storage_client_names):
                shards_to_transfer = self.nixl_storage_client.client_read(
                    target_agent_name, target_client_name, key, f"train_pull_{self.pull_times}"
                )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, target_client_name, shards_to_transfer))
        # Generation must wait for every NIXL pull operation.
        for key, target_client_name, shards_to_transfer in wait_operations:
            self.nixl_storage_client.wait(
                key, f"train_pull_{self.pull_times}", "READ", target_client=target_client_name
            )
        self.nixl_storage_client.merge_and_finish_cached_xfer()
        torch.cuda.synchronize()
        self.nixl_storage_client.clear_intermediate_cached_data()
        psrl_logger.info(
            f"{self.nixl_storage_client}: NIXL pull model core done "
            f"({self.pull_times} times). time: {time.time() - time_start}s"
        )

    def pull_model(self):
        """Pull the model from the PS via the specified mode.

        Currently we do not support `cpu` and `cpu_ref` modes for pulling the model in trainer.
        """
        if self.psrl_config.ps_mode == "cpu" or self.psrl_config.ps_mode == "cpu_ref":
            raise RuntimeError("ray_pull_model is not supported for TrainWorker in 'cpu' or 'cpu_ref' mode.")
        elif self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu":
            self.nixl_pull_model()
            # NOTE(linsh): Reload after the first pull because empty initialization leaves
            # optimizer master parameters stale, while later optimizer state remains local.
            if self.pull_times == 1:
                self.reload_optimizer_after_pull()
        else:
            raise NotImplementedError(f"PSRL GenWorker does not support PS mode '{self.psrl_config.ps_mode}' yet.")
        self._restore_non_persistent_buffers_from_ps()

    def _restore_non_persistent_buffers_from_ps(self) -> None:
        """Restore non-persistent buffers from the parameter server after pull."""
        raise NotImplementedError

    def _get_any_ps_worker_handle(self) -> ray.actor.ActorHandle:
        """
        Return a handle to the PS storage worker on the same node, falling back
        to the first available worker if none is co-located.

        Result is cached in _cached_ps_worker_handles after first resolution.

        Returns:
            ray.actor.ActorHandle: A handle to a PS storage worker.
        """
        ps_manager_handle = self.train_interface.ps_manager_handle
        my_node_id = self.get_node_id()
        # Prefer PS worker on the same node to avoid cross-node data transfer.
        client_name = ray.get(ps_manager_handle.get_ps_nixl_train_storage_client_name_for_node.remote(my_node_id))
        if client_name is None:
            if self._cached_ps_nixl_train_storage_client_names is None:
                self._cached_ps_nixl_train_storage_client_names = ray.get(
                    ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
                )
            client_name = self._cached_ps_nixl_train_storage_client_names[0]
            psrl_logger.warning(
                f"[_get_any_ps_worker_handle] No PS worker found on node={my_node_id!r}. "
                f"Falling back to client={client_name!r}."
            )
        if client_name not in self._cached_ps_worker_handles:
            self._cached_ps_worker_handles[client_name] = ray.get(
                ps_manager_handle.get_ps_worker_handle.remote(client_name)
            )
        return self._cached_ps_worker_handles[client_name]

    def _get_non_persistent_buffers_from_ps(self) -> dict[str, torch.Tensor]:
        """
        Fetch non-persistent named buffers from the co-located PS storage worker.
        Result is cached after the first call.

        Returns:
            dict[str, torch.Tensor]: Mapping of dotted buffer name to CPU tensor.
        """
        if self._cached_non_persistent_buffers is not None:
            return self._cached_non_persistent_buffers
        ps_handle = self._get_any_ps_worker_handle()
        self._cached_non_persistent_buffers = ray.get(ps_handle.get_non_persistent_named_buffers.remote())
        psrl_logger.debug(
            "[_get_non_persistent_buffers_from_ps] Fetched non-persistent buffers from PS. "
            f"Count={len(self._cached_non_persistent_buffers)}."
        )
        return self._cached_non_persistent_buffers

    def _debug_log_train_info(self, label: str):
        """Debug log the train info."""
        if self.nixl_storage_client is not None:
            self.nixl_storage_client.log_shard_info(label=label)

    def reload_optimizer_after_pull(self):
        """Reload optimizer master parameters from the current model weights."""
        pass

    def _debug_log_train_model_info(self, label: str):
        """Debug log the train model info."""
        pass

    def _debug_log_ps_info(self, label: str):
        """Call debug_log_info on every PSStorageWorker via Ray RPC (rank-0 only to reduce noise)."""
        if self.worker_rank != 0:
            return
        try:
            ps_manager_handle = self.train_interface.ps_manager_handle
            if self._cached_ps_nixl_train_storage_client_names is None:
                self._cached_ps_nixl_train_storage_client_names = ray.get(
                    ps_manager_handle.get_ps_nixl_train_storage_client_names.remote()
                )
            futures = []
            for target_client_name in self._cached_ps_nixl_train_storage_client_names:
                if target_client_name not in self._cached_ps_worker_handles:
                    self._cached_ps_worker_handles[target_client_name] = ray.get(
                        ps_manager_handle.get_ps_worker_handle.remote(target_client_name)
                    )
                ps_worker_handle = self._cached_ps_worker_handles[target_client_name]
                futures.append(ps_worker_handle.debug_log_info.remote(label=label))
            ray.get(futures)
        except Exception as e:
            psrl_logger.warning(f"[{label}] Failed to log PS shard info: {e}")
