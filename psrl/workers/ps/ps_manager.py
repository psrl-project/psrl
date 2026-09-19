import logging
from collections.abc import Mapping
from dataclasses import dataclass

import ray
import transfer_queue as tq
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.tensor import DTensor

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.logger import (
    EventType,
    get_ps_logger,
    get_worker_info,
    log_dual_events,
    log_single_event,
)
from psrl.utils.nixl import NIXLMetaServer
from psrl.utils.ray import add_busy_polling_lock
from psrl.workers.gen.utils import INVALID_ROLLOUT_INSTANCE_ID, RolloutInstanceId
from psrl.workers.ps.broadcast import build_broadcast_plan
from psrl.workers.ps.ps_worker_group import PSWorkerGroup
from psrl.workers.ps.request_status_tracker import RequestStatusTracker, _state_locked
from psrl.workers.ps.staleness_controller import (
    BufferStatus,
    EntryInfo,
    StalenessInventory,
)

psrl_logger = get_ps_logger()


@dataclass
class RolloutInstanceStatus:
    version_tag: int


@dataclass
class ModelStore:
    """Model Storage in the Parameter Server.

    This class holds the model state dictionary (ref) and its version tag.
    """

    version_tag: int
    model_state_dict: Mapping[str, Tensor | DTensor] | None = None
    model_state_dict_ref: ray.ObjectRef | None = None


# TODO(lhy): Ensure `PSManager` is a singleton.
@add_busy_polling_lock
class PSManager(RequestStatusTracker):
    def __init__(
        self,
        psrl_config: DictConfig,
    ) -> None:
        """Initialize the Parameter Server (PS) Manager.

        The PS Manager is responsible for managing model versions, staleness buffers,
        and rollout requests. It coordinates between rollout workers and training workers
        through a staleness-controlled buffer system.

        Args:
            psrl_config (DictConfig): Configuration object containing parameters such as
                rollout_n, staleness buffer entries, staleness limit, etc.
        """
        RequestStatusTracker.__init__(self, psrl_config)

        tq.init()

        if self.psrl_config.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.psrl_config.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.psrl_config.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.psrl_config.rollout_n
            self.alg_rollout_n = self.rollout_n
        self.val_rollout_n = self.psrl_config.val_rollout_n

        self.rollout_instance_tracker: dict[RolloutInstanceId, RolloutInstanceStatus] = {}
        # NOTE(lhy): Version zero represents the loaded checkpoint and avoids a `None` check
        # during the initial pull before the first training push.
        self.model_store: ModelStore = ModelStore(version_tag=0)

        self.staleness_inventory: StalenessInventory | None = None

        self.val_staleness_inventory: StalenessInventory | None = None

        self.check_abort_versions = set()
        self.max_aborted_version = -1

        if self.psrl_config.rollout_coordination.redundant_rollout.enable:
            entries_per_buffer = self.psrl_config.rollout_coordination.redundant_rollout.redundant_global_batch_size
            ready_entries_per_buffer = self.psrl_config.rollout_coordination.redundant_rollout.alg_global_batch_size
        else:
            entries_per_buffer = self.psrl_config.staleness_buffer_entries
            ready_entries_per_buffer = entries_per_buffer

        self.staleness_inventory = StalenessInventory(
            num_entries=entries_per_buffer,
            ready_num_entries=ready_entries_per_buffer,
            staleness=self.psrl_config.staleness,
            rollout_n=self.rollout_n,
        )

        self.val_staleness_inventory = StalenessInventory(
            num_entries=ready_entries_per_buffer,
            ready_num_entries=ready_entries_per_buffer,
            staleness=None,
            rollout_n=self.val_rollout_n,
            is_validate=True,
        )

        self.expected_agents = 0
        self.nixl_meta_server: NIXLMetaServer | None = None
        self.ps_worker_group: PSWorkerGroup | None = None
        self.ps_nixl_agent_names: list[str] | None = None
        self.ps_nixl_train_storage_client_names: list[str] | None = None
        self.ps_nixl_gen_storage_client_names: list[str] | None = None
        # NOTE(claude): Worker handles stay rank ordered so broadcast coordination can index them by rank.
        self.ps_worker_handles_by_rank: list = []

        self._exclusive_push_locked = False
        self._shared_pull_count = 0

        self.grpc_state_server = None
        self.grpc_state_port: int | None = None

        psrl_logger.info(f"PSManager initialized on {get_worker_info()}.")

    # ------- GRPC STATE SERVER -------

    def start_grpc_server(self) -> int:
        """Start PS Manager gRPC server and return the bound port.

        Returns:
            int: gRPC server port number
        """
        if self.grpc_state_server is None:
            from psrl.grpc.ps_manager_service import PSManagerGrpcServer

            self.grpc_state_server = PSManagerGrpcServer(self)
        port = find_available_port(base_port=8100)
        self.grpc_state_port = self.grpc_state_server.start(port)
        return self.grpc_state_port

    def stop_grpc_server(self, grace: float = 1.0):
        """Stop PS Manager gRPC server if it is running."""
        if self.grpc_state_server is None:
            return
        self.grpc_state_server.stop(grace=grace)
        self.grpc_state_port = None

    def get_grpc_server_port(self) -> int | None:
        """Get current PS Manager gRPC server port."""
        return self.grpc_state_port

    @_state_locked
    def register_rollout_instance(self, rollout_instance_ids: RolloutInstanceId | list[RolloutInstanceId]):
        """Register a new rollout instance with the PS Manager.

        Args:
            rollout_instance_ids (RolloutInstanceId | list[RolloutInstanceId]):
                Unique identifier for the rollout instance(s)
        """
        if not isinstance(rollout_instance_ids, list):
            rollout_instance_ids = [rollout_instance_ids]

        for rollout_instance_id in rollout_instance_ids:
            self.rollout_instance_tracker[rollout_instance_id] = RolloutInstanceStatus(version_tag=0)

    # ------- STALENESS INVENTORY MANAGEMENT -------

    @_state_locked
    def set_val_staleness_inventory_capacity(self, ready_num_entries: int, num_entries: int = None):
        """Set the capacity of the validation staleness inventory.

        Args:
            ready_num_entries (int): Number of entries that can be marked as READY in each buffer
            num_entries (int): Total number of entries in each staleness buffer
        """
        if num_entries is None:
            num_entries = ready_num_entries
        self.val_staleness_inventory.create_buffer_with_capacity(ready_num_entries, num_entries)

    @_state_locked
    def ensure_train_buffer_exists(self, buffer_id: int):
        """Ensure a training buffer exists in the staleness inventory."""
        self.staleness_inventory.ensure_buffer_exists(buffer_id)

    @_state_locked
    def ensure_validate_buffer_exists(self):
        """Ensure a validation buffer exists in the staleness inventory."""
        self.val_staleness_inventory.ensure_buffer_exists()

    @_state_locked
    def get_max_reserve_num(self, model_version, is_validate: bool = False) -> int:
        """Get the maximum number of entries that can be reserved for a specific model version.

        Args:
            model_version (int): The model version to reserve entries for
            is_validate (bool): Whether to use the validation staleness inventory
        Returns:
            int: The maximum number of entries that can be reserved for the given model version
        """
        max_staleness_buffer_id = model_version + self.psrl_config.staleness if not is_validate else None
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        return staleness_inventory.get_empty_entries_total_num(max_staleness_buffer_id)

    # Older requests may run on rollout instances that already advanced to a newer model.
    @_state_locked
    def update_request_version_tag(
        self,
        request_id: int,
        new_version_tag: int,
        is_validate: bool = False,
    ):
        """Update the version tag of a specific request in the staleness inventory.

        Args:
            request_id (int): The unique identifier of the request
            new_version_tag (int): The new version tag to set for the request
            is_validate (bool): Whether to use the validation staleness inventory
        """
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        staleness_inventory.update_request_version_tag(
            request_id=request_id,
            new_version_tag=new_version_tag,
        )

    @_state_locked
    def update_request_instance_id(
        self,
        request_id: int,
        new_instance_id: RolloutInstanceId,
        is_validate: bool = False,
    ):
        """Update the instance id of a specific request in the staleness inventory.

        Args:
            request_id (int): The unique identifier of the request
            new_instance_id (RolloutInstanceId): The new rollout instance id to set for the request
            is_validate (bool): Whether to use the validation staleness inventory
        """
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        staleness_inventory.update_request_instance_id(
            request_id=request_id,
            new_instance_id=new_instance_id,
        )

    @_state_locked
    def update_request_n_trajectory(
        self,
        request_id: int,
        n_trajectory: int,
        is_validate: bool = False,
    ):
        """Update a request's trajectory count in both PS metadata stores.

        Args:
            request_id (int): The unique identifier of the request
            n_trajectory (int): The number of trajectories to set for the request
            is_validate (bool): Whether to use the validation staleness inventory
        """
        super().update_request_n_trajectory(request_id, n_trajectory)
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        staleness_inventory.update_request_n_trajectory(
            request_id=request_id,
            new_n_trajectory=n_trajectory,
        )

    @_state_locked
    def move_occupied_entries(
        self,
        prompt_ids: int | list[int],
        buffer_id: int,
        is_validate: bool = False,
    ):
        """Move occupied entries to a specific buffer."""
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        staleness_inventory.move_occupied_entries(prompt_ids, buffer_id)

    @_state_locked
    def clear_occupied_entries(
        self,
        prompt_ids: int | list[int],
        is_validate: bool = False,
    ):
        """Clear occupied entries in the staleness inventory.

        Args:
            prompt_ids (Union[int, List[int]]): The prompt ids to clear
            is_validate (bool): Whether to use the validation staleness inventory
        """
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        staleness_inventory.clear_occupied_entries(prompt_ids)

    @_state_locked
    def clear_reserved_entries(
        self,
        prompt_ids: int | list[int],
        move_across_buffer: bool = False,
        is_validate: bool = False,
    ):
        """Clear reserved entries in the staleness inventory.

        Args:
            prompt_ids (Union[int, List[int]]): The prompt ids to clear
            move_across_buffer (bool): Whether to move entries across buffers when clearing
            is_validate (bool): Whether to use the validation staleness inventory
        """
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        staleness_inventory.clear_reserved_entries(prompt_ids, move_across_buffer)

    @_state_locked
    def get_min_pending_buffer(self, is_validate: bool = False) -> int:
        """Get the minimum pending buffer id in the staleness inventory."""
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        pending_buffers = staleness_inventory.get_buffers_with_capacity()
        if not pending_buffers:
            return staleness_inventory.buffer_id
        else:
            return min(pending_buffers)

    def maybe_delete_buffer(self, buffer_id: int, is_validate: bool = False):
        """Maybe delete a buffer from the staleness inventory.

        When RESERVE entries are cleared from a buffer, we can not delete it immediately
        because we rely on the READY buffer status to awake training workers.

        This method checks if the buffer can be deleted based on the current PS model version
        because the PS model version indicates which buffers have been consumed by training workers.
        """
        if is_validate:
            self.val_staleness_inventory.delete_buffer(buffer_id)
            return

        ready_for_delete_buffer_ids = self.staleness_inventory.get_ready_for_delete_buffer_ids()
        if buffer_id in ready_for_delete_buffer_ids:
            for bid in sorted(list(ready_for_delete_buffer_ids)):
                if bid <= buffer_id:
                    psrl_logger.info(
                        f"Clearing buffer marked for deletion: {bid}. Latest pushed model version: {buffer_id}."
                    )
                    self.staleness_inventory.delete_buffer(bid)
                else:
                    break

    @_state_locked
    def can_reserve_request(
        self,
        request_idx: int | list[int],
        model_versions: list[int],
        without_new_reserve_entry: bool = False,
        is_validate: bool | list[bool] = False,
    ) -> list[bool] | list[list[bool]]:
        """
        Check whether a request can be reserved for a given group of model versions.

        Args:
            request_idx (int): The request index
            model_versions (list[int]): The model versions that need to be checked
            without_new_reserve_entry (bool):
                Whether to check if the request can be reserved
                without a new reserve entry
            is_validate (bool | list[bool]): Whether to use the validation staleness inventory
        Returns:
            list[bool] | list[list[bool]]: Whether the request(s) can be reserved for each model version
        """
        assert isinstance(is_validate, bool) or len(is_validate) == len(request_idx), (
            "is_validate should be a bool or a list of bools with the same length as request_idx."
        )

        if not isinstance(request_idx, list):
            request_idx = [request_idx]
            is_single_request = True
        else:
            is_single_request = False

        if isinstance(is_validate, bool):
            validate_per_request = [is_validate] * len(request_idx)
        else:
            validate_per_request = is_validate

        multi_results = []
        for request_id, is_validate in zip(request_idx, validate_per_request):
            results = []
            staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
            rollout_n = self.val_rollout_n if is_validate else self.rollout_n
            for model_version in model_versions:
                assert model_version != -1, "Model version should not be -1 when checking if a request can be reserved"
                assert request_id not in self._abort_request_ids, (
                    f"Aborted request ID: {request_id}. It cannot be checked."
                )
                if model_version <= self.max_aborted_version:
                    results.append(False)
                    continue
                entry_info = EntryInfo(
                    rollout_instance_id=INVALID_ROLLOUT_INSTANCE_ID,  # Not important for this check
                    prompt_id=request_id // rollout_n,
                    request_idx=request_id % rollout_n,
                    model_version=model_version,
                    n_trajectory=1,
                    is_validate=is_validate,
                )
                if without_new_reserve_entry:
                    results.append(
                        staleness_inventory.can_reserve_data_without_new_reserve_entry(entry_info, model_version)
                    )
                else:
                    results.append(staleness_inventory.can_reserve_data(entry_info, model_version))
            multi_results.append(results)

        if is_single_request:
            return multi_results[0]
        else:
            return multi_results

    @_state_locked
    def get_reserve_indicator(
        self,
        request_id: int,
        model_versions: list[int],
        is_validate: bool = False,
    ) -> list[float]:
        """
        Get the indicator of reserving a request for a given model version.
        indicator = inf: cannot reserve
        indicator = -inf: can reserve without new reserve entry
        indicator = -x: can reserve with new reserve entry in x-th pending buffer id

        Args:
            request_id (int): The request id
            model_versions (List[int]): The model versions that need to be checked
            is_validate (bool): Whether to use the validation staleness inventory
        Returns:
            List[int]: The indicator of reserving a request for each model version
        """
        indicators = []
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        for model_version in model_versions:
            assert model_version != -1, (
                "Model version should not be -1 when getting the indicator of reserving a request"
            )
            assert request_id not in self._abort_request_ids, (
                f"Aborted request ID: {request_id}. It cannot be checked."
            )
            if model_version <= self.max_aborted_version:
                indicators.append(float("inf"))
                continue
            entry_info = EntryInfo(
                rollout_instance_id=INVALID_ROLLOUT_INSTANCE_ID,  # Not important for this check
                prompt_id=request_id // rollout_n,
                request_idx=request_id % rollout_n,
                model_version=model_version,
                n_trajectory=1,
                is_validate=is_validate,
            )
            if staleness_inventory.can_reserve_data_without_new_reserve_entry(entry_info, model_version):
                indicators.append(float("-inf"))
            elif staleness_inventory.can_reserve_data(entry_info, model_version):
                max_pending_buffer_id = staleness_inventory.get_max_pending_buffer_id(
                    model_version + self.psrl_config.staleness
                )
                indicators.append(-max_pending_buffer_id)
            else:
                indicators.append(float("inf"))
        return indicators

    @_state_locked
    def reserve_rollout_instance_requests(
        self,
        rollout_instance_ids: RolloutInstanceId | list[RolloutInstanceId],
        request_ids: int | list[int],
        model_versions: int | list[int],
        guarantee_not_aborted: bool = True,
        is_validate: bool = False,
    ) -> tuple[list[int | None], list[int | None]]:
        """
        Reserve requests for specific rollout instances and model versions.

        This method will reserve buffer entries for requests in the specified rollout instance,
        without storing actual rollout data in the staleness inventory.
        For Group Sampling, it supports reserving multiple buffer entries if `by_parent` is True.
        Note that the request will be ignored if the group entries have been reserved.

        Args:
            rollout_instance_ids (Union[RolloutInstanceId, List[RolloutInstanceId]]): The rollout instance ids
            request_ids (Union[int, List[int]]): The request ids
            model_versions (Union[int, List[int]]): The model versions
            guarantee_not_aborted (bool): Whether to guarantee that the requests are not aborted
            is_validate (bool): Whether to use the validation staleness inventory
        Returns:
            Tuple[Optional[List[int]], Optional[List[int]]]: A tuple containing two lists:
                - List of reserved buffer ids
                - List of reserved entry ids
            If reservation fails, returns (None, None).
        """
        if not isinstance(rollout_instance_ids, list):
            rollout_instance_ids = [rollout_instance_ids]
        if not isinstance(request_ids, list):
            request_ids = [request_ids]
        if not isinstance(model_versions, list):
            model_versions = [model_versions]

        for rollout_instance_id in rollout_instance_ids:
            assert (
                rollout_instance_id == INVALID_ROLLOUT_INSTANCE_ID
                or rollout_instance_id in self.rollout_instance_tracker
            ), f"Unregistered rollout instance: {rollout_instance_id}."

        entry_ids = []
        buffer_ids = []
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        for rollout_instance_id, request_id, model_version in zip(rollout_instance_ids, request_ids, model_versions):
            assert model_version != -1, "Model version should not be -1 when reserving a request"
            if guarantee_not_aborted:
                assert request_id not in self._abort_request_ids, (
                    f"Aborted request ID: {request_id}. It cannot be reserved."
                )
            else:
                if self.check_aborted_requests(request_id, remove=True):
                    entry_ids.append(None)
                    buffer_ids.append(None)
                    continue
            assert is_validate or model_version > self.max_aborted_version, (
                f"Cannot reserve model version: {model_version}. "
                f"It must exceed the maximum aborted version: {self.max_aborted_version}."
            )
            max_staleness_buffer_id = model_version + self.psrl_config.staleness if not is_validate else None
            # The reserved model version may be ahead of the rollout instance version.
            entry_info = EntryInfo(
                rollout_instance_id=rollout_instance_id,
                prompt_id=request_id // rollout_n,
                request_idx=request_id % rollout_n,
                model_version=model_version,
                n_trajectory=1,
                is_validate=is_validate,
            )

            buffer_id, entry_id = staleness_inventory.reserve_data(
                entry_info=entry_info, max_staleness_buffer_id=max_staleness_buffer_id
            )

            if buffer_id is None or entry_id is None:
                raise RuntimeError(
                    f"Failed to reserve entry for request {request_id} in rollout instance {rollout_instance_id} "
                    f"with model version {model_version}. "
                    f"Please check if the staleness inventory is full or the model version is too old."
                )

            entry_ids.append(entry_id)
            buffer_ids.append(buffer_id)

        return buffer_ids, entry_ids

    @_state_locked
    def abort_reserved_requests(self, buffer_id: int) -> tuple[int, list[int]]:
        """Abort the reserved requests for a specific buffer.

        Args:
            buffer_id (int): The ID of the buffer that is waiting for consumption.
        Returns:
            A tuple of (number of aborted entry ids, list of request ids that are aborted).
        """
        reserved_entry_ids = self.staleness_inventory.buffers[buffer_id].get_reserved_entry_ids()
        abort_request_ids = list()
        for entry_id in reserved_entry_ids:
            entry_info = self.staleness_inventory.buffers[buffer_id].entries[entry_id].entry_info
            prompt_id = entry_info.prompt_id
            # NOTE(lhy): Abort every request for the prompt, including unrecorded requests.
            abort_request_ids.extend([prompt_id * self.rollout_n + i for i in range(self.rollout_n)])
            """
            request_idxs = entry_info.request_idx
            if not isinstance(request_idxs, list):
                request_idxs = [request_idxs]
            for request_idx in request_idxs:
                request_id = prompt_id * self.rollout_n + request_idx
                abort_request_ids.append(request_id)
            """
        self.abort_requests(abort_request_ids)
        return len(reserved_entry_ids), abort_request_ids

    @_state_locked
    def abort_requests(
        self,
        request_ids: list[int] | int,
        abort_group: bool = True,
        blocking: bool = False,
    ):
        """Abort specific rollout requests.
        This method will abort the specified requests and update the staleness inventory accordingly.

        Args:
            request_ids (Union[List[int], int]): The unique identifiers of the requests to abort.
            abort_group (bool, optional): Whether to abort the entire group of requests
                if the group will not satisfy sampling requirements after abortion. Defaults to True.
            blocking (bool, optional): Whether to block until the abortion is complete. Defaults to False.
        """
        if not isinstance(request_ids, list):
            request_ids = [request_ids]

        request_ids = set(request_ids)
        abort_request_ids = request_ids
        prompt_id_to_abort_request_idxs = {}
        for request_id in request_ids:
            prompt_id = request_id // self.rollout_n
            if prompt_id not in prompt_id_to_abort_request_idxs:
                prompt_id_to_abort_request_idxs[prompt_id] = []
            prompt_id_to_abort_request_idxs[prompt_id].append(request_id % self.rollout_n)
        clear_entries = []
        for prompt_id, abort_request_idxs in prompt_id_to_abort_request_idxs.items():
            if prompt_id not in self.staleness_inventory.data_tracker:
                # A group can fail before reservation. Keep its IDs aborted so late status updates are rejected.
                psrl_logger.warning(
                    f"Prompt ID: {prompt_id}. Its group did not complete before reservation. "
                    f"Skipping inventory cleanup for UIDs: "
                    f"{[prompt_id * self.rollout_n + idx for idx in abort_request_idxs]}."
                )
                continue
            assert prompt_id in self.staleness_inventory.data_tracker, (
                f"Prompt ID: {prompt_id}. An existing data tracker mapping is required."
            )
            buffer_id, entry_id = self.staleness_inventory.data_tracker[prompt_id]
            entry_info = self.staleness_inventory.buffers[buffer_id].entries[entry_id].entry_info
            all_requests_of_entry = entry_info.get_all_request_relative_ids()
            rest_requests_of_entry = set(all_requests_of_entry) - set(abort_request_idxs)
            if abort_group and len(rest_requests_of_entry) < self.alg_rollout_n:
                abort_request_ids = abort_request_ids.union(set(rest_requests_of_entry))
                clear_entries.append(entry_info.prompt_id)
                psrl_logger.info(f"Abort entire entry: (buffer {buffer_id}, entry {entry_id}) for prompt {prompt_id}")
            else:
                update_idxs = []
                assert isinstance(entry_info.request_idx, list), "entry_info.request_idx should be a list."
                for i, request_idx in enumerate(entry_info.request_idx):
                    if request_idx not in abort_request_idxs:
                        update_idxs.append(i)
                if isinstance(entry_info.rollout_instance_id, list):
                    entry_info.rollout_instance_id = [entry_info.rollout_instance_id[i] for i in update_idxs]
                if isinstance(entry_info.model_version, list):
                    entry_info.model_version = [entry_info.model_version[i] for i in update_idxs]
                psrl_logger.info(
                    f"Aborted requests from buffer {buffer_id}, entry {entry_id}, prompt {prompt_id}. "
                    f"Old request indices: {entry_info.request_idx}. "
                    f"New request indices: {[entry_info.request_idx[i] for i in update_idxs]}."
                )
                entry_info.request_idx = [entry_info.request_idx[i] for i in update_idxs]
                self.staleness_inventory.buffers[buffer_id].entries[entry_id].entry_info = entry_info

        self.staleness_inventory.clear_reserved_entries(clear_entries)
        self._abort_requests(list(abort_request_ids), blocking)
        psrl_logger.debug(f"Abort requests done: {abort_request_ids=}, {clear_entries=}")

    @_state_locked
    def check_aborted_model_versions(self, model_versions: int | list[int]) -> bool | list[bool]:
        """Check if the model versions are aborted.

        Args:
            model_versions: The list of model versions to check.
        Returns:
            A boolean or a list of booleans indicating whether the model versions are aborted.
        """
        if not isinstance(model_versions, list):
            return model_versions <= self.max_aborted_version
        return [model_version <= self.max_aborted_version for model_version in model_versions]

    @_state_locked
    def check_aborted_requests(self, request_ids: int | list[int], remove: bool = False) -> bool | list[bool]:
        """Check if the requests are aborted and remove them from the abort set if needed.

        Args:
            request_ids: The list of request ids to check.
            remove: Whether to remove the requests from the abort set if they are aborted.
        Returns:
            A boolean or a list of booleans indicating whether the requests are aborted.
        """
        if not isinstance(request_ids, list):
            return self._check_aborted_request(request_ids, remove)
        return [self._check_aborted_request(request_id, remove) for request_id in request_ids]

    def _abort_after_buffer_ready(self, buffer_id: int) -> set[int]:
        """
        Check and interrupt rollout instances if necessary based on the ready buffer.

        Args:
            buffer_id (int): The ID of the buffer that is ready.
        Returns:
            A set of request ids that are aborted.
        """
        curr_ps_model_version = self.get_ps_model_version(debug_info="ps_manager")
        ready_buffer_ids = self.staleness_inventory.ready_buffer_ids().copy()
        abort_request_ids = set()
        curr_abort_versions = set()

        if buffer_id >= self.psrl_config.staleness:
            # A ready buffer constrains requests one staleness window behind it.
            self.check_abort_versions.add(buffer_id - self.psrl_config.staleness)

        while len(self.check_abort_versions) > 0:
            version_to_abort = min(self.check_abort_versions)
            # NOTE(linsh): Buffers can become ready out of order. Abort only after every
            # buffer from the current model version through the staleness window is ready.
            buffer_range = set(
                range(
                    max(version_to_abort, curr_ps_model_version),
                    version_to_abort + self.psrl_config.staleness + 1,
                )
            )
            psrl_logger.debug(
                f"Checking abort for version: {version_to_abort}. "
                f"Required buffer range: {buffer_range}. Ready buffers: {ready_buffer_ids}."
            )
            if buffer_range.issubset(ready_buffer_ids):
                curr_abort_versions.add(version_to_abort)
                self.check_abort_versions.discard(version_to_abort)
            else:
                break

        psrl_logger.info(
            f"Aborting versions: {curr_abort_versions}. Trigger buffer: {buffer_id}. The buffer is ready."
        )
        curr_abort_versions = sorted(list(curr_abort_versions))
        for abort_version in curr_abort_versions:
            self.max_aborted_version = max(self.max_aborted_version, abort_version)
            requests_of_abort_version = self.get_requests_of_abort_version(abort_version)
            psrl_logger.info(f"Requests to abort for version {abort_version}: {requests_of_abort_version}.")
            abort_request_ids = abort_request_ids.union(requests_of_abort_version)

        if abort_request_ids:
            with log_dual_events(
                f"Abort {len(abort_request_ids)} requests in staleness check",
                psrl_logger,
                level=logging.INFO,
                event_type=EventType.OTHER,
            ):
                self.abort_requests(list(abort_request_ids))

        for buffer_id in ready_buffer_ids:
            if self.staleness_inventory.buffers[buffer_id].get_reserve_entry_num() == 0:
                if curr_ps_model_version > buffer_id:
                    psrl_logger.info(
                        f"Deleting buffer after aborting requests: {buffer_id}. "
                        f"Current PS model version: {curr_ps_model_version}."
                    )
                    self.staleness_inventory.delete_buffer(buffer_id)
                else:
                    psrl_logger.info(f"Marking buffer for deletion after aborting requests: {buffer_id}.")
                    self.staleness_inventory.mark_buffer_for_deletion(buffer_id)

        psrl_logger.debug(
            f"Staleness abort check complete. Buffer: {buffer_id}. "
            f"Current PS model version: {curr_ps_model_version}. "
            f"Original ready buffers: {ready_buffer_ids}. Aborted versions: {curr_abort_versions}. "
            f"Aborted request count: {len(abort_request_ids)}. "
            f"Current ready buffers: {self.staleness_inventory.ready_buffer_ids()}."
        )

        return abort_request_ids

    @_state_locked
    def handle_ready_buffer(self, buffer_id: int) -> set[int]:
        """
        Handle the ready buffer.

        Args:
            buffer_id (int): The ID of the buffer that is ready.
        Returns:
            A set of request ids that are aborted.
        """
        self.rollout_coordinator.update_ready_buffer.remote(buffer_id)
        abort_request_ids = self._abort_after_buffer_ready(buffer_id)
        return abort_request_ids

    @_state_locked
    def occupy_rollout_instance_request(
        self,
        prompt_id: int,
        request_ids: int | list[int] | None = None,
        is_validate: bool = False,
    ) -> tuple[int | None, BufferStatus | None, EntryInfo]:
        """
        Store a finished request in the staleness inventory, maybe occupy the buffer
        if one of the following requirements is met:
        (1). No parent requests (`parent_id` is None). Note that in this case the data will
        directly occupy the buffer, bypassing storing in the data pool.
        (2). Collect required `rollout_n` requests for group sampling.

        Args:
            prompt_id (int): The prompt id of the request
            request_ids (Optional[Union[int, List[int]]]): The request ids to occupy
            is_validate (bool): Whether to use the validation staleness inventory

        Returns:
            Tuple[Optional[int], Optional[BufferStatus], EntryInfo]: A tuple containing:
                - buffer_id (Optional[int]): The buffer id where the data is stored, or None if not occupied
                - occupy_num (Optional[BufferStatus]): The buffer status after occupation, or None if not occupied
                - entry_info (EntryInfo): The entry information of the occupied data
        """
        if request_ids is None:
            request_ids = [prompt_id]

        is_aborted = self.check_aborted_requests(request_ids, remove=True)
        filtered_request_ids = [request_id for i, request_id in enumerate(request_ids) if not is_aborted[i]]
        staleness_inventory = self.val_staleness_inventory if is_validate else self.staleness_inventory
        self.remove_train_ready_request(filtered_request_ids)

        if len(filtered_request_ids) == 0:
            assert prompt_id not in staleness_inventory.data_tracker, (
                f"Occupy failed due to staleness limit, but aborted data exists in the inventory: {prompt_id}."
            )
            return None, None, None

        buffer_id, entry_id, occupy_num = staleness_inventory.occupy_data_with_reserve(prompt_id)
        if buffer_id is None:
            raise RuntimeError("Unexpected error: buffer id is None")

        entry_info = staleness_inventory.buffers[buffer_id].entries[entry_id].entry_info
        return buffer_id, occupy_num, entry_info

    # ------- MODEL VERSION MANAGEMENT -------

    def get_all_rollout_instance_model_versions(self) -> dict[RolloutInstanceId, int]:
        """Get all rollout instance model versions.

        Returns:
            Dict[RolloutInstanceId, int]: A dictionary mapping rollout instance IDs to their model versions
        """
        return {
            instance_id: instance_status.version_tag
            for instance_id, instance_status in self.rollout_instance_tracker.items()
        }

    @_state_locked
    def get_rollout_instance_model_version(self, rollout_instance_id: RolloutInstanceId) -> int:
        """Get the model version for a specific rollout instance.

        Args:
            rollout_instance_id (RolloutInstanceId): The rollout instance id

        Returns:
            int: The model version for the specified rollout instance
        """
        assert rollout_instance_id in self.rollout_instance_tracker, (
            f"Unregistered rollout instance: {rollout_instance_id}."
        )

        return self.rollout_instance_tracker[rollout_instance_id].version_tag

    def get_ps_model_version(self, debug_info: str = None) -> int:
        """Get the current model version."""
        psrl_logger.debug(f"Getting the PS model version. Model store: {self.model_store}. Debug info: {debug_info}.")
        if self.model_store is None:
            return 0
        return self.model_store.version_tag

    def _update_rollout_instance_model_version_tag_to_latest(
        self,
        rollout_instance_ids: RolloutInstanceId | list[RolloutInstanceId],
    ):
        """Update the rollout instance model version to the latest model version."""
        if not isinstance(rollout_instance_ids, list):
            rollout_instance_ids = [rollout_instance_ids]

        for rollout_instance_id in rollout_instance_ids:
            assert rollout_instance_id in self.rollout_instance_tracker, (
                f"Unregistered rollout instance: {rollout_instance_id}."
            )

            if self.rollout_instance_tracker[rollout_instance_id].version_tag != self.model_store.version_tag:
                # NOTE(lhy): The coordinator is unnecessary when initialization keeps the version at zero.
                assert self.rollout_coordinator is not None, (
                    "Rollout coordinator is not set. Please set it before updating rollout instance model version."
                )
                self.rollout_instance_tracker[rollout_instance_id].version_tag = self.model_store.version_tag
                self.rollout_coordinator.set_rollout_instance_model_version.remote(
                    rollout_instance_id=rollout_instance_id,
                    version_tag=self.model_store.version_tag,
                )
                psrl_logger.info(
                    f"Updated rollout instance: {rollout_instance_id}. Model version: {self.model_store.version_tag}."
                )

    # ------- PS NIXL CONTROL PLANE -------

    def init_nixl_server(self, expected_agents: int):
        """Initialize the NIXL server for distributed communication.

        Args:
            expected_agents (int): Number of expected NIXL clients to connect
        """
        self.expected_agents = expected_agents
        broadcast_init_enabled = self.psrl_config.broadcast_init.enabled
        self.nixl_meta_server = NIXLMetaServer(
            "NIXLMetaServer",
            self.psrl_config.nixl,
            broadcast_init_enabled=broadcast_init_enabled,
        )

    def nixl_protocol(self):
        """Coordinate NIXL metadata exchange across all clients."""
        psrl_logger.info(
            f"NIXL protocol step 1. Waiting for client shardings. Expected clients: {self.expected_agents}."
        )
        self.nixl_meta_server.wait_for_client_shardings(self.expected_agents)
        psrl_logger.info("NIXL protocol step 2. Building unified sharding.")
        self.nixl_meta_server.make_unified_sharding()
        psrl_logger.info("NIXL protocol step 3. Publishing client shardings.")
        self.nixl_meta_server.notify_all_client_shardings()
        psrl_logger.info(f"NIXL protocol step 4. Waiting for client info. Expected clients: {self.expected_agents}.")
        self.nixl_meta_server.wait_for_client_infos(self.expected_agents)
        psrl_logger.info("NIXL protocol step 5. Building the communication plan.")
        self.nixl_meta_server.make_comm_plan()
        psrl_logger.info("NIXL protocol step 6. Publishing client info and the communication plan.")
        self.nixl_meta_server.notify_all_client_infos_and_comm_plan()
        psrl_logger.info(
            f"NIXL protocol step 7. Waiting for temporary mappings. Expected clients: {self.expected_agents}."
        )
        self.nixl_meta_server.wait_for_client_temp_mappings(self.expected_agents)
        psrl_logger.info("NIXL protocol step 8. Publishing temporary mappings.")
        self.nixl_meta_server.notify_all_client_temp_mappings()
        psrl_logger.info("NIXL protocol complete.")

    def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for NIXL clients to send updated infos to the meta server.

        Args:
            info_num (int): Number of client infos to wait for
        """
        self.nixl_meta_server.wait_for_update_infos(info_num)

    def shutdown_nixl_server(self):
        """Shutdown NIXL meta server cleanly.

        This avoids leaving UCX endpoints active while Ray tears down actors.
        """
        if self.nixl_meta_server is None:
            return
        self.nixl_meta_server.shutdown()
        self.nixl_meta_server = None

    def nixl_broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        """Broadcast updated client infos to specified NIXL clients.

        Args:
            dst_agent_names (List[str]): List of destination agent names to receive the updates
            update_client_names (List[str]): List of client names whose infos are updated
        """
        self.nixl_meta_server.broadcast_update_client_infos(dst_agent_names, update_client_names)

    def bind_ps_worker_group(self, ps_worker_group: PSWorkerGroup):
        """Bind the PS worker group to the PSManager.

        This method establishes the connection between the PSManager and the
        PS worker group, enabling distributed model storage and retrieval.

        Args:
            ps_worker_group (PSWorkerGroup): The PS worker group to bind
        """
        self.ps_worker_group = ps_worker_group
        ps_nixl_agent_name_futures = self.ps_worker_group.execute_all_async("get_nixl_agent_name")
        ps_nixl_train_storage_client_name_futures = self.ps_worker_group.execute_all_async(
            "get_nixl_train_storage_client_name"
        )
        ps_nixl_gen_storage_client_name_futures = self.ps_worker_group.execute_all_async(
            "get_nixl_gen_storage_client_name"
        )
        self.ps_nixl_agent_names = ray.get(ps_nixl_agent_name_futures)
        self.ps_nixl_train_storage_client_names = ray.get(ps_nixl_train_storage_client_name_futures)
        self.ps_nixl_gen_storage_client_names = ray.get(ps_nixl_gen_storage_client_name_futures)
        # NOTE(claude): Workers are rank ordered, so broadcast coordination can index them by rank.
        self.ps_worker_handles_by_rank = list(self.ps_worker_group._workers)
        psrl_logger.info(
            f"PS worker group initialized with NIXL agent names: {self.ps_nixl_agent_names}, "
            f"train storage client names: {self.ps_nixl_train_storage_client_names}, "
            f"gen storage client names: {self.ps_nixl_gen_storage_client_names}"
        )
        if self.psrl_config.broadcast_init.enabled:
            self.enable_broadcast_init_on_server()

    def enable_broadcast_init_on_server(self) -> None:
        """
        Report that broadcast initialization was enabled during meta server construction.
        """
        psrl_logger.info(
            "[enable_broadcast_init_on_server] Broadcast initialization is active. "
            "PS-to-PS ClientInfos were distributed during nixl_protocol phase 2b."
        )

    def _coordinate_broadcast_init(self) -> None:
        """
        Broadcast checkpoint weights through the PS worker tree.

        Every round is a barrier because a worker can send only after receiving all weights.
        """
        world_size = len(self.ps_worker_handles_by_rank)
        plan = build_broadcast_plan(
            world_size=world_size,
            algorithm=self.psrl_config.broadcast_init.algorithm,
        )
        psrl_logger.info(
            f"[_coordinate_broadcast_init] world_size={world_size}, "
            f"algorithm={self.psrl_config.broadcast_init.algorithm!r}, "
            f"num_rounds={plan.num_rounds()}."
        )

        for round_idx in range(plan.num_rounds()):
            senders = plan.senders_in_round(round_idx)
            if not senders:
                continue
            psrl_logger.info(f"[_coordinate_broadcast_init] round {round_idx}: senders={senders}.")
            futures = [
                self.ps_worker_handles_by_rank[rank].broadcast_send_to_children.remote(round_idx, plan)
                for rank in senders
            ]
            ray.get(futures)  # Each sender must finish before the next round.

        psrl_logger.info(
            "[_coordinate_broadcast_init] All rounds are complete. Starting train-to-generation transfer."
        )
        ray.get([w.do_transfer_train_to_gen_after_broadcast.remote() for w in self.ps_worker_handles_by_rank])
        psrl_logger.info("[_coordinate_broadcast_init] Broadcast initialization complete.")

    def get_ps_worker_handle(self, client_name: str) -> ray.actor.ActorHandle:
        """Get the PS worker handle by the client name."""
        assert self.ps_worker_group is not None, (
            "The PS worker group must be initialized before calling get_ps_worker_handle."
        )
        worker = self.ps_worker_group.distinguish_worker_by_method(
            lambda worker: (
                client_name == ray.get(worker.get_nixl_train_storage_client_name.remote())
                or client_name == ray.get(worker.get_nixl_gen_storage_client_name.remote())
            )
        )
        return worker

    def get_ps_nixl_agent_names(self) -> list[str]:
        """Get the NIXL agent name of the PS worker group."""
        assert self.ps_nixl_agent_names is not None, (
            "The PS worker group must be initialized before calling get_ps_nixl_agent_names."
        )
        return self.ps_nixl_agent_names

    def get_ps_nixl_train_storage_client_names(self) -> list[str]:
        """Get the NIXL train storage client name of the PS worker group."""
        assert self.ps_nixl_train_storage_client_names is not None, (
            "The PS worker group must be initialized before calling get_ps_nixl_train_storage_client_name."
        )
        return self.ps_nixl_train_storage_client_names

    def get_ps_nixl_gen_storage_client_names(self) -> list[str]:
        """Get the NIXL gen storage client name of the PS worker group."""
        assert self.ps_nixl_gen_storage_client_names is not None, (
            "The PS worker group must be initialized before calling get_ps_nixl_gen_storage_client_name."
        )
        return self.ps_nixl_gen_storage_client_names

    def get_ps_nixl_train_storage_client_name_for_node(self, node_id: str) -> str | None:
        """
        Return the NIXL train storage client name for the PS worker on the given node.

        Builds a node_id → client_name mapping lazily on first call and caches it.
        Returns None if no PS worker is found on the given node (caller should fall back).

        Args:
            node_id (str): Ray node ID of the requesting worker.

        Returns:
            str | None: The NIXL train storage client name, or None if not found.
        """
        if not hasattr(self, "_ps_node_id_to_train_client_name"):
            self._ps_node_id_to_train_client_name: dict[str, str] = {}
            for worker in self.ps_worker_group._workers:
                worker_node_id = ray.get(worker.get_node_id.remote())
                client_name = ray.get(worker.get_nixl_train_storage_client_name.remote())
                self._ps_node_id_to_train_client_name[worker_node_id] = client_name
        return self._ps_node_id_to_train_client_name.get(node_id, None)

    # ------- MODEL PUSH/PULL -------

    def _try_acquire_exclusive_push_lock(self) -> bool:
        """
        Try to acquire the exclusive push lock.
        Returns True if acquired, False if already locked.

        This method ensures that:
        - No push is in progress (_exclusive_push_locked == False)
        - No pull is in progress (_shared_pull_count == 0)
        """
        if self._exclusive_push_locked or self._shared_pull_count > 0:
            return False
        self._exclusive_push_locked = True
        return True

    def _release_exclusive_push_lock(self):
        """Release the exclusive push lock."""
        assert self._exclusive_push_locked, "Exclusive push lock is not locked"
        self._exclusive_push_locked = False

    def _try_acquire_shared_pull_lock(self) -> bool:
        """
        Try to acquire the shared pull lock.
        Returns True if acquired, False if a push is in progress.

        This method ensures that:
        - No push is in progress (_exclusive_push_locked == False)
        Multiple pull operations can run concurrently.
        """
        if self._exclusive_push_locked:
            return False
        self._shared_pull_count += 1
        return True

    def _release_shared_pull_lock(self):
        """Release the shared pull lock."""
        assert self._shared_pull_count > 0, "Shared pull lock count is already 0"
        self._shared_pull_count -= 1

    @_state_locked
    def init_model_version_for_resume(self, version: int):
        """
        Initialize the PS model version for resume without triggering buffer deletion
        or coordinator assertions.

        This sets the model_store to the given version so that the next push_model call
        from train workers will correctly compute next_version = version + 1.

        Args:
            version (int): The version to initialize to (typically `checkpoint_step-1`).
        """
        self.model_store = ModelStore(version_tag=version)
        if self.rollout_coordinator is not None:
            self.rollout_coordinator.init_ps_model_version.remote(version)
        psrl_logger.info(f"Initialized model version for resume: {version}")

    @_state_locked
    def push_model_state_dict_cpu(self, version_tag: int, model_state_dict: Mapping[str, Tensor | DTensor] | None):
        """
        Push a model to the PS. In 'cpu' mode, store the real state dict. In 'cpu_ref' mode, this should not be called.
        This method will block until the state dict is received by the PS worker
        (potential bottleneck for large models).

        Args:
            version_tag (int): The version tag of the model
            model_state_dict (Optional[Mapping[str, Union[Tensor, DTensor]]]): The model state dict to push
        """
        assert self.psrl_config.ps_mode == "cpu", "push_model_state_dict_cpu should only be used in 'cpu' mode."

        self.model_store = ModelStore(version_tag=version_tag, model_state_dict=model_state_dict)
        if version_tag > 0:
            self.maybe_delete_buffer(version_tag - 1)
        self.rollout_coordinator.set_ps_model_version.remote(version_tag)
        log_single_event(
            f"Model with version tag {version_tag} pushed successfully",
            psrl_logger,
            event_type=EventType.PUSH,
        )

    # NOTE(lhy): Ray dereferences only top-level task or actor arguments, not object
    # references nested in containers.
    @_state_locked
    def push_model_state_dict_cpu_ref_list(self, version_tag: int, model_state_dict_ref_list: list[ray.ObjectRef]):
        """
        Push a model to the PS by storing a ray object_ref. Only used in 'cpu_ref' mode.
        This method is non-blocking for the PS worker and only updates metadata (no large data transfer here).

        Args:
            version_tag (int): The version tag of the model
            model_state_dict_ref_list (List[ray.ObjectRef]): The list of ray object_refs to push
        """
        assert self.psrl_config.ps_mode == "cpu_ref", (
            "push_model_state_dict_ref should only be used in 'cpu_ref' mode."
        )
        assert len(model_state_dict_ref_list) == 1, "Only one model state dict ref is supported in 'cpu_ref' mode."

        self.model_store = ModelStore(version_tag=version_tag, model_state_dict_ref=model_state_dict_ref_list[0])
        if version_tag > 0:
            self.maybe_delete_buffer(version_tag - 1)
        self.rollout_coordinator.set_ps_model_version.remote(version_tag)
        log_single_event(
            f"Model with version tag {version_tag} (ref) pushed successfully",
            psrl_logger,
            event_type=EventType.PUSH,
        )

    @_state_locked
    def push_model_state_dict_nixl(self, version_tag: int):
        """
        Record the version tag of the model state dict pushed to the PS via NIXL.
        The actual model state dict is stored in the PS worker group.
        """
        self.model_store = ModelStore(
            version_tag=version_tag,
        )
        if version_tag > 0:
            self.maybe_delete_buffer(version_tag - 1)
        self.rollout_coordinator.set_ps_model_version.remote(version_tag)
        log_single_event(
            f"Model with version tag {version_tag} (nixl) pushed successfully",
            psrl_logger,
            event_type=EventType.PUSH,
        )

    @_state_locked
    def pull_model_state_dict_cpu(
        self, rollout_instance_id: RolloutInstanceId
    ) -> Mapping[str, Tensor | DTensor] | None:
        """
        Pull the latest model state dict from PS via CPU. Only used in 'cpu' mode.
        This will block until the state dict is transferred (potential bottleneck for large models).

        Args:
            rollout_instance_id (RolloutInstanceId): The rollout instance id

        Returns:
            Optional[Mapping[str, Union[Tensor, DTensor]]]: The model state dict
        """
        assert self.psrl_config.ps_mode == "cpu", "pull_model_state_dict_cpu should only be used in 'cpu' mode."
        assert self.model_store is not None, "Model instance is not initialized."

        instance_ids = rollout_instance_id if isinstance(rollout_instance_id, list) else [rollout_instance_id]

        log_single_event(
            f"Rollout instance {rollout_instance_id} pulling latest model "
            f"with version tag {self.model_store.version_tag}",
            psrl_logger,
            event_type=EventType.PULL,
        )
        self._update_rollout_instance_model_version_tag_to_latest(instance_ids)
        return self.model_store.model_state_dict

    @_state_locked
    def pull_model_state_dict_cpu_ref(
        self, rollout_instance_ids: RolloutInstanceId | list[RolloutInstanceId]
    ) -> ray.ObjectRef:
        """
        Return the ray object_ref for the latest model state dict. Only used in 'cpu_ref' mode.
        This is a fast operation (no large data transfer here).

        Args:
            rollout_instance_ids (RolloutInstanceId | list[RolloutInstanceId]): The rollout instance id

        Returns:
            ray.ObjectRef: The ray object_ref for the latest model state dict
        """
        assert self.psrl_config.ps_mode == "cpu_ref", "get_model_state_dict_ref should only be used in 'cpu_ref' mode."
        assert self.model_store is not None, "Model instance is not initialized."

        if not isinstance(rollout_instance_ids, list):
            rollout_instance_ids = [rollout_instance_ids]

        log_single_event(
            f"Rollout instance {rollout_instance_ids} pulling latest model "
            f"with version tag {self.model_store.version_tag} (ref)",
            psrl_logger,
            event_type=EventType.PULL,
        )
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_ids)
        return self.model_store.model_state_dict_ref

    @_state_locked
    def pull_model_state_dict_nixl(self, rollout_instance_ids: RolloutInstanceId | list[RolloutInstanceId]):
        """
        Pull the latest model state dict from PS via NIXL. Only used in 'nixl_cpu' or 'nixl_gpu' mode.
        This only updates the version tag of the model state dict pulled from the PS.
        The actual model state dict is stored in the PS worker group.
        """
        assert self.psrl_config.ps_mode == "nixl_cpu" or self.psrl_config.ps_mode == "nixl_gpu", (
            "pull_model_state_dict_nixl should only be used in 'nixl_cpu' or 'nixl_gpu' mode."
        )
        assert self.model_store is not None, "Model instance is not initialized."

        if not isinstance(rollout_instance_ids, list):
            rollout_instance_ids = [rollout_instance_ids]

        log_single_event(
            f"Rollout instance {rollout_instance_ids} pulling latest model "
            f"with version tag {self.model_store.version_tag} (nixl)",
            psrl_logger,
            event_type=EventType.PULL,
        )
        self._update_rollout_instance_model_version_tag_to_latest(rollout_instance_ids)
