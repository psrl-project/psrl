import enum
import threading
from enum import Enum
from functools import wraps

import ray
from omegaconf import DictConfig

from psrl.utils.logger import DualOutputHandler, get_ps_logger
from psrl.utils.server.command import Command, CommandType
from psrl.utils.transferqueue_utils import PayloadState, clear_payload, request_payload_keys
from psrl.workers.gen.utils import INVALID_ROLLOUT_INSTANCE_ID, RolloutInstanceId
from psrl.workers.ps.staleness_controller import EntryInfo

psrl_logger = get_ps_logger()


def _state_locked(func):
    """Protect request metadata maps shared across Ray actor and gRPC threads."""

    @wraps(func)
    def _wrapped(self, *args, **kwargs):
        lock = self._state_lock
        with lock:
            return func(self, *args, **kwargs)

    return _wrapped


class PSRL_RequestStatus(Enum):
    """Represents the status of a request in the system.

    PENDING: Request is queued in data queue, waiting for dispatch
    RUNNING: Request is running (generic running state)
    ROLLOUT_ROUTING: Request is in the router and is waiting for a specific instance to be dispatched
    ROLLOUT_DISPATCHED: Request is dispatched to the instance engine
    ROLLOUT_RUNNING: Request is in the instance engine and is being rolled out
    ROLLOUT_INTERRUPTED: Rollout was interrupted (by coordinator for Partial Rollout, or by
        scheduler via KV cache preemption), put into replay buffer
    REWARD_RUNNING: Request is running in the reward manager
    REWARD_COMPLETED: Request is completed in the reward manager
    COMPLETED: Request is completed (generic completed state)
    """

    PENDING = enum.auto()
    RUNNING = enum.auto()
    ROLLOUT_ROUTING = enum.auto()
    ROLLOUT_DISPATCHED = enum.auto()
    ROLLOUT_RUNNING = enum.auto()
    ROLLOUT_INTERRUPTED = enum.auto()
    ROLLOUT_COMPLETED = enum.auto()
    REWARD_RUNNING = enum.auto()
    REWARD_COMPLETED = enum.auto()
    COMPLETED = enum.auto()


# NOTE(lhy): Status changes and payload commits occur in separate actors. Cleanup filters
# missing keys because it can race a commit.
TQ_PAYLOAD_CLEANUP_STATUSES: frozenset = frozenset(
    {
        PSRL_RequestStatus.ROLLOUT_COMPLETED,
        PSRL_RequestStatus.REWARD_RUNNING,
        PSRL_RequestStatus.REWARD_COMPLETED,
    }
)


class RequestStatusTracker:
    """
    Manage request status and metadata across processing stages.
    """

    def __init__(self, psrl_config: DictConfig):
        self.psrl_config = psrl_config
        self._state_lock = threading.RLock()
        self._request_id_to_status: dict[int, PSRL_RequestStatus] = {}
        self._request_infos = {}
        self._status_to_request_ids = {status: set() for status in PSRL_RequestStatus}
        self._abort_request_ids = set()
        self._running_min_version = 0

        if self.psrl_config.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.psrl_config.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.psrl_config.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.psrl_config.rollout_n
            self.alg_rollout_n = self.rollout_n
        self.val_rollout_n = self.psrl_config.val_rollout_n

        self.rollout_coordinator: ray.actor.ActorHandle | None = None

        self.reward_manager: ray.actor.ActorHandle | None = None

        self.log_prefix = "RequestStatusTracker"
        psrl_logger.addHandler(DualOutputHandler(self.psrl_config.logging_path, self.log_prefix))
        psrl_logger.info("Initialized RequestStatusTracker.")

    def set_rollout_coordinator(self, rollout_coordinator: ray.actor.ActorHandle):
        """Set the reference to the rollout coordinator."""
        self.rollout_coordinator = rollout_coordinator

    def set_reward_manager(self, reward_manager: ray.actor.ActorHandle):
        """Set the reference to the reward manager."""
        self.reward_manager = reward_manager

    @_state_locked
    def update_request_status(
        self,
        request_id: list[int] | int,
        status: list[PSRL_RequestStatus] | PSRL_RequestStatus,
        model_version: list[int] | int = -1,
        rollout_instance_id: list[RolloutInstanceId] | RolloutInstanceId = INVALID_ROLLOUT_INSTANCE_ID,
        is_validate: bool = False,
    ) -> list[bool] | bool:
        """Update the status of requests.

        This method first checks if requests are marked for abortion. If so,
        it returns False and removes them from the status map. It also validates
        that requests are not stale before updating their status.

        Args:
            request_id (Union[List[int], int]): The unique identifier(s) of the request(s)
            status (Union[List[PSRL_RequestStatus], PSRL_RequestStatus]): The new status(es) to set
            model_version (int, optional): The model version of the request. Defaults to -1
            rollout_instance_id (RolloutInstanceId, optional):
                The instance ID of the rollout worker. Defaults to (-1, -1)
            is_validate (bool, optional): Whether the request is for validation. Defaults to False

        Returns:
            Union[List[bool], bool]: True if status was updated successfully, False if request was aborted
        """
        if not isinstance(request_id, list):
            request_id = [request_id]
        if not isinstance(model_version, list):
            model_version = [model_version] * len(request_id)
        if not isinstance(rollout_instance_id, list):
            rollout_instance_id = [rollout_instance_id] * len(request_id)

        assert len(request_id) == len(model_version) == len(rollout_instance_id), (
            "request_id, model_version, and rollout_instance_id must have the same length."
        )

        if not isinstance(status, list):
            status = [status] * len(request_id)
        for s in status:
            if s not in PSRL_RequestStatus:
                raise ValueError(f"Invalid status: {s}")

        request_update_success = [True for _ in range(len(request_id))]

        abort_payload_keys = []
        for i, req_id in enumerate(request_id):
            if self._check_aborted_request(req_id, remove=True):
                request_update_success[i] = False
                # NOTE(linsh): Payload cleanup for aborted requests belongs to `_abort_requests`.
                continue

            if req_id in self._request_infos:
                if rollout_instance_id[i] != INVALID_ROLLOUT_INSTANCE_ID:
                    self._request_infos[req_id].rollout_instance_id = rollout_instance_id[i]
                if model_version[i] != -1:
                    self._request_infos[req_id].model_version = model_version[i]
                request_version = self._request_infos[req_id].model_version
                if not is_validate and request_version != -1 and request_version < self._running_min_version:
                    psrl_logger.warning(
                        f"Cannot update stale request. Request ID: {req_id}. "
                        f"Request version: {request_version}. Minimum running version: {self._running_min_version}."
                    )
                    request_update_success[i] = False
                    current_status = self._request_id_to_status.get(req_id)
                    if current_status in TQ_PAYLOAD_CLEANUP_STATUSES:
                        abort_payload_keys.extend(self._request_tq_keys(req_id))
                    continue
            else:
                raise KeyError(f"Request ID {req_id} not found in request info map.")

            if req_id in self._request_id_to_status:
                new_status = status[i]
                old_status = self._request_id_to_status[req_id]
                if old_status != new_status:
                    self._status_to_request_ids[old_status].discard(req_id)
                self._status_to_request_ids[new_status].add(req_id)
                self._request_id_to_status[req_id] = new_status
            else:
                raise KeyError(f"Request ID {req_id} not found in status map.")

        if abort_payload_keys:
            clear_payload(
                keys=abort_payload_keys,
                partition_id="val" if is_validate else "train",
                state=PayloadState.DROPPED,
            )

        return request_update_success[0] if len(request_update_success) == 1 else request_update_success

    def get_request_status(self, request_id: list[int] | int):
        """Get the current status of requests.

        Args:
            request_id (Union[List[int], int]): The identifier(s) of the request(s)

        Returns:
            Union[List[PSRL_RequestStatus], PSRL_RequestStatus]: The current status(es) of the request(s)

        Raises:
            KeyError: If one or more request IDs are not found
        """
        if not isinstance(request_id, list):
            request_id = [request_id]

        status_list = [self._request_id_to_status.get(req_id, None) for req_id in request_id]
        if any(status is None for status in status_list):
            raise KeyError(f"One or more request IDs not found: {request_id}")

        return status_list

    def remove_train_ready_request(self, request_id: list[int] | int):
        """Remove requests that are ready for training.

        This method removes completed requests from the tracking system
        after they have been processed by the reward manager.
        It is a wrapper function for the `remove_request` method.

        Args:
            request_id (Union[List[int], int]): The unique identifier(s) of the request(s) to remove

        Raises:
            AssertionError: If request is not found or not in correct status
        """
        psrl_logger.debug(f"Removing train-ready requests after reward completion. Request IDs: {request_id}.")
        if not isinstance(request_id, list):
            request_id = [request_id]

        for req_id in request_id:
            assert req_id in self._request_id_to_status, f"Request not found in status map. ID: {req_id}."
            assert req_id in self._request_infos, f"Request not found in request info. ID: {req_id}."
            assert self._request_id_to_status[req_id] == PSRL_RequestStatus.COMPLETED, (
                f"Request is not complete. ID: {req_id}."
            )
            assert req_id not in self._abort_request_ids, (
                f"Aborted request cannot be removed as train ready. ID: {req_id}."
            )

        self.remove_request(request_id)

    def get_all_request_statuses(self) -> dict:
        """Get the statuses of all requests currently being tracked.

        Returns:
            dict: A dictionary mapping request IDs to their current statuses
        """
        return self._request_id_to_status.copy()

    def get_requests_by_status(self, status: PSRL_RequestStatus) -> set:
        """
        Get all requests that have a specific status.

        Args:
            status (PSRL_RequestStatus): The status to filter requests by.

        Returns:
            set: A set of request ids that match the specified status.
        """
        return self._status_to_request_ids.get(status, set())

    @_state_locked
    def add_request(
        self,
        request_id: list[int] | int,
        rollout_instance_id: list[RolloutInstanceId] | RolloutInstanceId = INVALID_ROLLOUT_INSTANCE_ID,
        model_version: list[int] | int = -1,
        status: (list[PSRL_RequestStatus] | PSRL_RequestStatus) = PSRL_RequestStatus.PENDING,
        is_validate: list[bool] | bool = False,
    ):
        """
        Add new requests to the status manager.

        Args:
            request_id (Union[List[int], int]): The unique identifier(s) of the request(s).
            rollout_instance_id (Union[List[RolloutInstanceId], RolloutInstanceId], optional):
                The instance ID(s) of the rollout worker(s). Defaults to (-1, -1).
            model_version (Union[List[int], int], optional):
                The model version(s) of the request(s). Defaults to -1.
            status (Union[List[PSRL_RequestStatus], PSRL_RequestStatus], optional):
                The initial status(es) of the request(s). Defaults to PSRL_RequestStatus.PENDING.
            is_validate (Union[List[bool], bool], optional):
                Whether the request(s) are for validation. Defaults to False.
        """
        if not isinstance(request_id, list):
            request_id = [request_id]
        if not isinstance(rollout_instance_id, list):
            rollout_instance_id = [rollout_instance_id] * len(request_id)
        if not isinstance(model_version, list):
            model_version = [model_version] * len(request_id)
        if not isinstance(status, list):
            status = [status] * len(request_id)
        if not isinstance(is_validate, list):
            is_validate = [is_validate] * len(request_id)

        assert len(request_id) == len(rollout_instance_id) == len(model_version) == len(status) == len(is_validate), (
            "All parameter lists must have the same length."
        )

        for i, req_id in enumerate(request_id):
            rollout_n = self.val_rollout_n if is_validate[i] else self.rollout_n
            self._request_infos[req_id] = EntryInfo(
                prompt_id=req_id // rollout_n,
                request_idx=req_id % rollout_n,
                rollout_instance_id=rollout_instance_id[i],
                model_version=model_version[i],
                n_trajectory=1,
                is_validate=is_validate[i],
            )
            self._request_id_to_status[req_id] = status[i]
            self._status_to_request_ids[status[i]].add(req_id)

    def _check_aborted_request(self, request_id: int, remove: bool = True) -> bool:
        """Check if the request is aborted and remove it from the status tracker if needed.

        Args:
            request_id: The id of the request to check.
            remove: Whether to remove the request from the status tracker if it is aborted.
        Returns:
            True if the request is aborted, False otherwise.
        """
        if request_id in self._abort_request_ids:
            if remove:
                self.remove_request(request_id)
            return True
        if request_id not in self._request_id_to_status:
            assert request_id not in self._request_infos, (
                f"Untracked request remains in request info. ID: {request_id}."
            )
            return True
        if request_id not in self._request_infos:
            assert request_id not in self._request_id_to_status, (
                f"Request without info remains in status map. ID: {request_id}."
            )
            return True
        return False

    def _request_tq_keys(self, request_id: int) -> list[str]:
        """Reconstruct the committed payload keys for a tracked request."""
        n_trajectory = self._request_infos[request_id].n_trajectory
        return request_payload_keys(request_id, n_trajectory)

    @_state_locked
    def update_request_n_trajectory(self, request_id: int, n_trajectory: int) -> None:
        """Keep the status tracker's payload-key metadata in sync with TITO output."""
        if n_trajectory < 1:
            raise ValueError(f"n_trajectory must be positive, got {n_trajectory}.")
        if request_id not in self._request_infos:
            raise KeyError(f"Request ID {request_id} not found in request infos.")
        self._request_infos[request_id].n_trajectory = n_trajectory

    def _abort_requests(self, request_ids: list[int] | int, blocking: bool = False):
        """
        Mark requests for abortion.
        If the request is in the COMPLETED status, simply delete it from the status tracker.
        Otherwise, the request will not be aborted immediately,
        but will be aborted next time when the request is updated.

        Args:
            request_ids (Union[List[int], int]): The unique identifiers of the requests to abort.
            blocking (bool, optional): Whether to block until the abortion is complete. Defaults to False.
        """
        if not isinstance(request_ids, list):
            request_ids = [request_ids]

        request_ids = set(request_ids)
        filtered_request_ids = [req_id for req_id in request_ids if req_id in self._request_id_to_status]
        if filtered_request_ids:
            psrl_logger.info(f"Added requests to abort set: {filtered_request_ids}.")
        self._abort_request_ids.update(filtered_request_ids)

        status_to_req_ids = self.classify_requests_in_status(filtered_request_ids)

        abort_requests_for_rollout = set()
        abort_requests_for_reward = set()
        abort_requests_for_completed = set()
        abort_payload_keys: list[str] = []

        for status, req_ids in status_to_req_ids.items():
            psrl_logger.info(f"Classifying aborted requests in status {status}: {req_ids}")
            if status in {PSRL_RequestStatus.ROLLOUT_RUNNING}:
                abort_requests_for_rollout.update(req_ids)
            elif status in {PSRL_RequestStatus.REWARD_RUNNING}:
                abort_requests_for_reward.update(req_ids)
            elif status in {PSRL_RequestStatus.COMPLETED}:
                abort_requests_for_completed.update(req_ids)
            if status in TQ_PAYLOAD_CLEANUP_STATUSES:
                for req_id in req_ids:
                    abort_payload_keys.extend(self._request_tq_keys(req_id))

        futures = []
        if abort_requests_for_rollout:
            psrl_logger.debug(f"Aborting requests in rollout stages: {abort_requests_for_rollout}")
            instance_to_request_ids = self.classify_requests_in_instance(list(abort_requests_for_rollout))
            futures.append(
                self.rollout_coordinator.exec_command.remote(
                    Command(
                        type=CommandType.ABORT,
                        instance_to_uids=instance_to_request_ids,
                    ),
                    blocking=blocking,
                )
            )
            psrl_logger.debug(f"Abort command sent to rollout coordinator for requests: {abort_requests_for_rollout}")

        if abort_requests_for_reward:
            psrl_logger.debug(f"Aborting requests in reward stages: {abort_requests_for_reward}")
            futures.append(
                self.reward_manager.exec_command.remote(
                    Command(
                        type=CommandType.ABORT,
                        uids=list(abort_requests_for_reward),
                    ),
                    blocking=blocking,
                )
            )
            psrl_logger.debug(f"Abort command sent to reward manager for requests: {abort_requests_for_reward}")

        if abort_requests_for_completed:
            self.remove_request(list(abort_requests_for_completed))

        if futures and blocking:
            ray.get(futures)

        # Clear only payloads known to be committed.
        if abort_payload_keys:
            clear_payload(
                keys=abort_payload_keys,
                partition_id="train",
                state=PayloadState.DROPPED,
            )

    def classify_requests_in_status(self, request_ids: list[int] | int) -> dict:
        """
        Classify the requests in `request_ids` into their current statuses.

        Args:
            request_ids (List[int], int): The unique identifiers of the requests to classify.

        Returns:
            dict: A dictionary mapping statuses to sets of request IDs that are in those statuses.
        """
        if not isinstance(request_ids, list):
            request_ids = [request_ids]

        classified_requests = {}
        for req_id in request_ids:
            if req_id in self._request_id_to_status:
                status = self._request_id_to_status[req_id]
                if status not in classified_requests:
                    classified_requests[status] = set()
                classified_requests[status].add(req_id)
            else:
                raise KeyError(f"Request ID {req_id} not found in status map.")

        return classified_requests

    def classify_requests_in_instance(self, request_ids: list[int] | int) -> dict:
        """
        Classify the requests in `request_ids` into their associated rollout instances.

        Args:
            request_ids (List[int], int): The unique identifiers of the requests to classify.

        Returns:
            dict: A dictionary mapping instance IDs to sets of request IDs that belong to those instances.
        """
        if not isinstance(request_ids, list):
            request_ids = [request_ids]

        classified_requests = {}
        for req_id in request_ids:
            if req_id in self._request_infos:
                instance_id = self._request_infos[req_id].rollout_instance_id
                if instance_id not in classified_requests:
                    classified_requests[instance_id] = set()
                classified_requests[instance_id].add(req_id)
            else:
                raise KeyError(f"Request ID {req_id} not found in request infos.")

        return classified_requests

    @_state_locked
    def get_recorded_child_requests(self, parent_id: list[int] | int, is_validate: bool = False) -> set[int]:
        """
        Get the recorded child requests for a given parent request ID.

        Args:
            parent_id (List[int], int): The unique identifier of the parent request.
            is_validate (bool, optional): Whether the request is for validation. Defaults to False.

        Returns:
            set[int]: A list of child request IDs associated with the parent request.
        """
        if not isinstance(parent_id, list):
            parent_id = [parent_id]

        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        child_requests = []
        for pid in parent_id:
            for i in range(rollout_n):
                child_id = pid * rollout_n + i
                if child_id in self._request_infos:
                    child_requests.append(child_id)

        return set(child_requests)

    def get_request_info(self, request_id: list[int] | int) -> list[EntryInfo] | EntryInfo | None:
        """
        Get the EntryInfo for a specific request ID.

        Args:
            request_id (List[int], int): The unique identifier of the request.

        Returns:
            EntryInfo: The EntryInfo objects associated with the request IDs, or None if not found.
        """
        if not isinstance(request_id, list):
            request_id = [request_id]

        request_infos = [self._request_infos.get(req_id, None) for req_id in request_id]
        if any(info is None for info in request_infos):
            raise KeyError(f"One or more request IDs not found: {request_id}")

        if len(request_infos) == 1:
            return request_infos[0]
        else:
            return self._request_infos.get(request_id, None)

    def update_request_info(self, entry_info: list[EntryInfo] | EntryInfo):
        """
        Update the EntryInfo for a specific request ID.

        Args:
            entry_info (List[EntryInfo], EntryInfo): The new EntryInfo object to associate with the request ID.
        """
        if not isinstance(entry_info, list):
            entry_info = [entry_info]

        for info in entry_info:
            request_id = info.request_id
            if request_id not in self._request_infos:
                raise KeyError(f"Request ID {request_id} not found.")

        for info in entry_info:
            request_id = info.request_id
            if request_id in self._request_infos:
                self._request_infos[request_id] = info
            else:
                raise KeyError(f"Request ID {request_id} not found.")

    def remove_request(self, request_id: list[int] | int):
        """
        Remove a request and its associated status.

        Args:
            request_id (List[int], int): The unique identifier of the request to abort.
        """
        psrl_logger.debug(f"Removing requests from status tracker. Request IDs: {request_id}.")
        if not isinstance(request_id, list):
            request_id = [request_id]

        for req_id in request_id:
            if req_id in self._request_id_to_status:
                status = self._request_id_to_status[req_id]
                self._status_to_request_ids[status].discard(req_id)
                self._request_id_to_status.pop(req_id)

            if req_id in self._request_infos:
                self._request_infos.pop(req_id)

            if req_id in self._abort_request_ids:
                self._abort_request_ids.remove(req_id)

    def clear_request_status_manager(self):
        """
        Clear all requests and their statuses.
        """
        self._request_id_to_status.clear()
        self._status_to_request_ids = {status: set() for status in PSRL_RequestStatus}
        self._request_infos.clear()
        self._abort_request_ids.clear()

    def get_requests_ids_of_version(self, version: int) -> set[int]:
        """
        Get all requests associated with a specific version.

        Args:
            version (int): The version to filter requests by.

        Returns:
            set[int]: A set of request IDs that match the specified version.
        """
        assert version >= 0, "Version must be a non-negative integer."
        return {req_id for req_id, info in self._request_infos.items() if info.model_version == version}

    def get_dispatched_requests_of_instance(self, instance_id: RolloutInstanceId) -> set[int]:
        """
        Get all dispatched requests associated with a specific rollout instance.

        Args:
            instance_id (RolloutInstanceId): The ID of the rollout instance to filter requests by.

        Returns:
            set[int]: A set of requests that are dispatched to the specified instance.
        """
        return {info for req_id, info in self._request_infos.items() if info.rollout_instance_id == instance_id}

    def get_requests_of_abort_version(self, version: int) -> set[int]:
        """
        Get all requests associated with a specific abort version.

        Args:
            version (int): The version to abort requests for.

        Returns:
            set[int]: A set of request IDs that match the specified abort version.
        """
        assert version >= 0, "Version must be a non-negative integer."

        abort_request_ids = set()
        for req_id, info in self._request_infos.items():
            if req_id in self._abort_request_ids:
                continue
            if not info.is_validate and info.model_version == version:
                abort_request_ids.add(req_id)
        self._running_min_version = max(self._running_min_version, version + 1)
        return abort_request_ids

    def update_request_version(self, request_id: list[int] | int, new_version: int):
        """
        Update the version of a request.

        Args:
            request_id (List[int], int): The unique identifier of the request.
            new_version (int): The new version to set for the request.
        """
        if not isinstance(request_id, list):
            request_id = [request_id]

        for req_id in request_id:
            if req_id in self._request_infos:
                self._request_infos[req_id].model_version = new_version
            else:
                raise KeyError(f"Request ID {req_id} not found.")

    def update_request_instance(self, request_id: list[int] | int, new_instance: int):
        """
        Update the instance of a request.

        Args:
            request_id (List[int], int): The unique identifier of the request.
            new_instance (id): The new instance to set for the request.
        """
        if not isinstance(request_id, list):
            request_id = [request_id]

        for req_id in request_id:
            if req_id in self._request_infos:
                self._request_infos[req_id].instance = new_instance
            else:
                raise KeyError(f"Request ID {req_id} not found.")
