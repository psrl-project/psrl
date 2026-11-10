import enum
from dataclasses import dataclass
from functools import wraps

import numpy as np

from psrl.utils.logger import get_ps_logger
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = get_ps_logger()


def _state_locked(func):
    """Protect request metadata maps shared across Ray actor and gRPC threads."""

    @wraps(func)
    def _wrapped(self, *args, **kwargs):
        lock = self._state_lock
        with lock:
            return func(self, *args, **kwargs)

    return _wrapped


class EntryCategory(enum.Enum):
    """Enum for the category of an entry in the buffer.

    EMPTY: The entry is empty and available for reservation
    RESERVED: The entry is reserved for a request but not yet occupied
    OCCUPIED: The entry is occupied with data and available for training
    """

    EMPTY = 0
    RESERVED = 1
    OCCUPIED = 2


@dataclass
class EntryInfo:
    """
    Metadata for an entry in the buffer.

    This class is used to track the metadata of an entry in the buffer,
    including the rollout instance ID, request ID, and model version.

    Args:
        rollout_instance_id (RolloutInstanceId | list[RolloutInstanceId]):
            The ID(s) of the rollout instance this entry belongs to.
        prompt_id (int): The global unique prompt ID.
        request_idx (int | list[int]): The relative request ID(s) inside a group.
        model_version (int | list[int]): The model version(s) when generating this entry.
        n_trajectory (int | list[int]): Number of trajectories represented by this entry.
    """

    rollout_instance_id: RolloutInstanceId | list[RolloutInstanceId]
    prompt_id: int
    request_idx: int | list[int]
    # Generated model versions must remain within the occupied buffer's staleness window.
    model_version: int | list[int]
    n_trajectory: int | list[int] = 1
    is_validate: bool = False

    def __hash__(self):
        return hash(self.prompt_id)

    def __eq__(self, other):
        return isinstance(other, EntryInfo) and self.prompt_id == other.prompt_id

    def get_all_request_relative_ids(self) -> list[int]:
        """Get all relative request IDs associated with this entry info."""
        if isinstance(self.request_idx, list):
            return self.request_idx
        else:
            return [self.request_idx]

    def get_all_request_ids(self, rollout_n: int) -> list[int]:
        """Get all request IDs associated with this entry info."""
        if isinstance(self.request_idx, list):
            return [self.prompt_id * rollout_n + idx for idx in self.request_idx]
        else:
            return [self.prompt_id * rollout_n + self.request_idx]

    def get_entry_version(self) -> int:
        """Get the minimum model version associated with this entry info."""
        if isinstance(self.model_version, list):
            return min(self.model_version)
        else:
            return self.model_version


def _is_scalar_request_idx(request_idx) -> bool:
    """Return True when *request_idx* is a single integer index, not a group list."""
    return isinstance(request_idx, (int, np.integer)) and not isinstance(request_idx, bool)


def _assert_scalar_request_idx(request_idx) -> None:
    if not _is_scalar_request_idx(request_idx):
        raise AssertionError(
            f"Request idx must be a list or a scalar integer, got {request_idx!r} (type={type(request_idx).__name__})"
        )


@dataclass
class Entry:
    """
    An entry in the buffer.

    This class is used to track the category, rollout data of an entry in the buffer and
    the metadata of the entry.

    Args:
        category (EntryCategory): The category of the entry (EMPTY, RESERVED, OCCUPIED).
        entry_info (Optional[EntryInfo]): The metadata of the entry.
    """

    category: EntryCategory
    entry_info: EntryInfo | None = None


class BufferStatus(enum.Enum):
    """Enum for the status of a buffer.

    READY: All entries are OCCUPIED and the buffer is ready for training
    STUCK: Mixed OCCUPIED and RESERVED entries with no EMPTY slots
    PENDING: Has at least one EMPTY slot and is still accepting new entries
    READY_WITH_CAPACITY: Required entries are OCCUPIED, but still has capacity for more entries
    """

    READY = 0
    STUCK = 1
    PENDING = 2
    READY_WITH_CAPACITY = 3


class StalenessBuffer:
    """Buffer for managing staleness-controlled entries.

    This class manages a fixed-size buffer of entries, each of which can be EMPTY,
    RESERVED, or OCCUPIED. It provides methods for inserting, deleting, and querying
    entries, as well as determining the buffer's status for training readiness.

    Args:
        num_entries (int): The number of entries in the buffer
        ready_num_entries (int): The number of entries in the buffer that are sufficient for one training batch
        staleness (int): The staleness tolerance for this buffer
    """

    def __init__(self, num_entries: int, ready_num_entries: int, staleness: int):
        self.num_entries = num_entries
        self.ready_num_entries = ready_num_entries
        self.entries: list[Entry] = [Entry(category=EntryCategory.EMPTY) for _ in range(num_entries)]
        self.staleness = staleness

    def get_first_non_occupied(self) -> int:
        """Get the index of the first non-occupied entry.

        Returns:
            int: Index of the first non-occupied entry, or `self.num_entries` if all are OCCUPIED
        """
        for idx in range(self.num_entries):
            if self.entries[idx].category != EntryCategory.OCCUPIED:
                return idx
        return self.num_entries

    def get_last_non_reserved(self) -> int:
        """
        Returns the index of the last entry that is not RESERVED, or -1 if all are RESERVED.

        Returns:
            int: Index of the last non-reserved entry, or -1 if none found.
        """
        for idx in reversed(range(self.num_entries)):
            if self.entries[idx].category != EntryCategory.RESERVED:
                return idx
        return -1

    def get_reserve_entry_num(self) -> int:
        """Count number of RESERVED entries in the buffer"""
        return np.sum([1 for entry in self.entries if entry.category == EntryCategory.RESERVED])

    def get_reserved_entry_ids(self) -> list[int]:
        """Get all RESERVED entry IDs in the buffer"""
        return [idx for idx, entry in enumerate(self.entries) if entry.category == EntryCategory.RESERVED]

    def insert(
        self,
        entry_id: int,
        category: EntryCategory,
        entry_info: EntryInfo | None = None,
    ):
        """
        Insert an entry at the specified position.

        Args:
            entry_id (int): The index to insert the entry at.
            category (EntryCategory): The category of the entry.
            entry_info (EntryInfo, optional): The metadata for the entry.
        Raises:
            AssertionError: If entry_id is out of range.
        """
        assert 0 <= entry_id < self.num_entries, (
            f"Invalid entry ID: {entry_id}. Expected range: [0, {self.num_entries})."
        )

        self.entries[entry_id] = Entry(category=category, entry_info=entry_info)

    def delete(self, entry_id: int):
        """
        Delete (reset) the entry at the specified position, making it EMPTY.

        Args:
            entry_id (int): The index of the entry to delete.
        Raises:
            AssertionError: If entry_id is out of range.
        """
        assert 0 <= entry_id < self.num_entries, (
            f"Invalid entry ID: {entry_id}. Expected range: [0, {self.num_entries})."
        )

        self.entries[entry_id] = Entry(category=EntryCategory.EMPTY, entry_info=None)

    def get_status(self) -> BufferStatus:
        """
        Determine the buffer status based on entry states.

        - READY: All entries required for training are occupied and no EMPTY entries are available
        - READY_WITH_CAPACITY: All entries required for training are occupied and some EMPTY entries are available
        - STUCK: Not ready, and all entries are reserved/occupied
        - PENDING: Not ready, and at least one EMPTY entry in the buffer

        Returns:
            BufferStatus: The current status of the buffer (READY, STUCK, or PENDING).
        Raises:
            AssertionError: If buffer state invariants are violated.
        """
        first_non_occupied = self.get_first_non_occupied()

        if first_non_occupied == self.ready_num_entries:
            if any(entry.category == EntryCategory.EMPTY for entry in self.entries[self.ready_num_entries :]):
                return BufferStatus.READY_WITH_CAPACITY
            else:
                return BufferStatus.READY

        last_non_reserved = self.get_last_non_reserved()
        if first_non_occupied == last_non_reserved + 1:
            assert all(
                entry.category == EntryCategory.OCCUPIED for entry in self.entries[:first_non_occupied]
            ) and all(entry.category == EntryCategory.RESERVED for entry in self.entries[last_non_reserved + 1 :]), (
                "STUCK buffer must have all OCCUPIED before first non-occupied and "
                "all RESERVED after last non-reserved"
            )
            return BufferStatus.STUCK

        assert any(entry.category == EntryCategory.EMPTY for entry in self.entries), (
            "PENDING buffer must have at least one EMPTY entry"
        )
        return BufferStatus.PENDING

    def get_empty_entries_num(self) -> int:
        """Count number of EMPTY entries in the buffer"""
        return sum(1 for entry in self.entries if entry.category == EntryCategory.EMPTY)

    def clear(self):
        """Clear all entries in the buffer, resetting them to EMPTY."""
        self.entries = [Entry(category=EntryCategory.EMPTY) for _ in range(self.num_entries)]


class StalenessInventory:
    """
    Inventory for managing multiple staleness buffers and their data.

    This class manages a collection of staleness buffers, tracks data and entry mappings, and provides
    methods for reserving, occupying, and consuming buffer entries under staleness constraints.

    Args:
        num_entries (int): Number of entries per buffer.
        ready_num_entries (int): The number of entries in the buffer that are sufficient for one training batch
        staleness (int): The staleness tolerance for this buffer
        rollout_n (int): The number of rollout instances
        is_validate (bool): Whether in validation mode.
    """

    def __init__(
        self,
        num_entries: int,
        ready_num_entries: int,
        staleness: int,
        rollout_n: int,
        is_validate: bool = False,
    ):
        self.staleness = staleness
        self.buffer_id = 0
        self.num_entries = num_entries
        self.ready_num_entries = ready_num_entries
        self.rollout_n = rollout_n
        self.is_validate = is_validate

        self.buffers: dict[int, StalenessBuffer] = {}
        self.data_tracker: dict[int, tuple[int, int]] = {}
        # Cache buffer IDs by status to avoid scanning every buffer on each query.
        self._buffer_ids_by_status: dict[BufferStatus, set[int]] = {status: set() for status in BufferStatus}

        self._ready_for_delete_buffer_ids: set[int] = set()

        if self.is_validate:
            if self.staleness:
                self.staleness = None
                psrl_logger.warning("Validation inventory must have staleness=None, setting staleness to None.")
            if self.num_entries != self.ready_num_entries:
                self.num_entries = self.ready_num_entries
                psrl_logger.warning(
                    "Validation inventory must have num_entries equal to ready_num_entries, "
                    "setting num_entries to ready_num_entries."
                )

    def create_buffer_with_capacity(self, ready_num_entries: int, num_entries: int):
        """
        Set the size of buffers in the inventory.

        Args:
            ready_num_entries (int): The number of entries in the buffer that are sufficient for one training batch
            num_entries (int): The number of entries per buffer.
        """
        assert self.is_validate, (
            "Buffer size can only be dynamically set for validation inventory "
            "because only one buffer is used during validation."
        )
        self.num_entries = num_entries
        self.ready_num_entries = ready_num_entries
        self.create_buffer(self.buffer_id)

    def create_buffer(self, buffer_id: int):
        """
        Create a new buffer with the specified ID and fixed-size.

        Args:
            buffer_id (int): The ID of the buffer to create.
        Raises:
            AssertionError: If the buffer already exists.
        """
        assert buffer_id == self.buffer_id, f"Out-of-sequence buffer ID: {buffer_id}. Expected ID: {self.buffer_id}."

        buffer = StalenessBuffer(self.num_entries, self.ready_num_entries, self.staleness)
        self.buffers[buffer_id] = buffer
        psrl_logger.debug(f"[Buffer Create]: Created buffer: {buffer_id}. Current buffer IDs: {self.buffers.keys()}.")
        self._update_buffer_status(buffer_id)
        self.buffer_id += 1

    def delete_buffer(self, buffer_id: int):
        """
        Delete the buffer with the specified ID and remove all associated entries from the data tracker.

        Args:
            buffer_id (int): The ID of the buffer to delete.
        """
        if buffer_id not in self.buffers:
            return

        entries_to_remove = [
            entry.entry_info for entry in self.buffers[buffer_id].entries if entry.category != EntryCategory.EMPTY
        ]
        for entry_info in entries_to_remove:
            assert entry_info.prompt_id in self.data_tracker, f"Entry info missing from data tracker: {entry_info}."
            del self.data_tracker[entry_info.prompt_id]
        for status_set in self._buffer_ids_by_status.values():
            status_set.discard(buffer_id)
        del self.buffers[buffer_id]
        if buffer_id in self._ready_for_delete_buffer_ids:
            self._ready_for_delete_buffer_ids.remove(buffer_id)
        psrl_logger.debug(f"[Buffer Delete]: Deleted buffer: {buffer_id}. Current buffer IDs: {self.buffers.keys()}.")

    def mark_buffer_for_deletion(self, buffer_id: int):
        """
        Mark a buffer for deletion.
        """
        assert (
            self.get_buffer_status(buffer_id) == BufferStatus.READY
            or self.get_buffer_status(buffer_id) == BufferStatus.READY_WITH_CAPACITY
        ), f"Buffer must be ready before deletion. ID: {buffer_id}."
        self._ready_for_delete_buffer_ids.add(buffer_id)

    def get_ready_for_delete_buffer_ids(self) -> set[int]:
        """
        Get all buffer IDs that are ready for deletion.
        """
        return self._ready_for_delete_buffer_ids

    def get_buffer_status(self, buffer_id: int) -> BufferStatus:
        """
        Get the status of a specific buffer.

        Args:
            buffer_id (int): The ID of the buffer.
        Returns:
            BufferStatus: The status of the buffer.
        Raises:
            ValueError: If the buffer does not exist or has no status.
        """
        # Read the status cache to avoid recomputing every buffer status.
        if buffer_id not in self.buffers:
            raise ValueError(f"Buffer {buffer_id} does not exist")
        for status in BufferStatus:
            if buffer_id in self._buffer_ids_by_status[status]:
                return status
        raise ValueError(f"Buffer {buffer_id} has no status in inventory")

    def get_buffers_not_ready(self) -> set[int]:
        """
        Get all buffer IDs that are not in READY state.

        Returns:
            Set[int]: A set of buffer IDs that are not in READY state.
        """
        return self._buffer_ids_by_status[BufferStatus.STUCK] | self._buffer_ids_by_status[BufferStatus.PENDING]

    def get_buffers_with_capacity(self) -> set[int]:
        """
        Get all buffer IDs that are either READY_WITH_CAPACITY or PENDING.

        Returns:
            Set[int]: A set of buffer IDs with capacity for new entries.
        """
        return (
            self._buffer_ids_by_status[BufferStatus.PENDING]
            | self._buffer_ids_by_status[BufferStatus.READY_WITH_CAPACITY]
        )

    def _update_buffer_status(self, buffer_id: int) -> BufferStatus:
        """
        Update the internal status tracking for a buffer.

        Args:
            buffer_id (int): The ID of the buffer to update.
        Returns:
            BufferStatus: The new status of the buffer after update.
        """
        if buffer_id not in self.buffers:
            return

        buffer = self.buffers[buffer_id]
        for status in BufferStatus:
            if buffer_id in self._buffer_ids_by_status[status]:
                self._buffer_ids_by_status[status].remove(buffer_id)
                break

        new_status = buffer.get_status()
        self._buffer_ids_by_status[new_status].add(buffer_id)
        return new_status

    def min_ready_buffer_id(self) -> int | None:
        """
        Get the min buffer ID that is in READY state.

        Returns:
            Optional[int]: The min READY buffer ID, or None if none exist.
        """
        ready_buffers = self.ready_buffer_ids()
        return min(ready_buffers) if ready_buffers else None

    def max_ready_buffer_id(self) -> int | None:
        """
        Get the max buffer ID that is in READY state.

        Returns:
            Optional[int]: The max READY buffer ID, or None if none exist.
        """
        ready_buffers = self.ready_buffer_ids()
        return max(ready_buffers) if ready_buffers else None

    def ready_buffer_ids(self) -> set[int]:
        """
        Get all buffer IDs that are in READY or READY_WITH_CAPACITY state.

        Returns:
            Set[int]: A set of READY buffer IDs.
        """
        return (
            self._buffer_ids_by_status[BufferStatus.READY]
            | self._buffer_ids_by_status[BufferStatus.READY_WITH_CAPACITY]
        )

    def min_not_ready_buffer_id(self) -> int | None:
        """
        Get the smallest buffer ID that is not in READY state (i.e., STUCK or PENDING).

        Returns:
            Optional[int]: The smallest non-READY buffer ID, or None if none exist.
        """
        non_ready = self._buffer_ids_by_status[BufferStatus.STUCK] | self._buffer_ids_by_status[BufferStatus.PENDING]
        return min(non_ready) if non_ready else None

    def ensure_buffer_exists(self, max_staleness_buffer_id: int | None = None):
        """
        Ensure all buffers up to max_buffer_id exist. Create missing buffers as needed.

        Args:
            max_staleness_buffer_id (int | None): The maximum buffer ID to ensure exists.
        """
        if self.is_validate:
            return

        assert max_staleness_buffer_id is not None, "max_staleness_buffer_id must be provided for training inventory"

        next_buffer_id = self.buffer_id
        for buffer_id in range(next_buffer_id, max_staleness_buffer_id + 1):
            self.create_buffer(buffer_id)

    def get_empty_entries_total_num(self, max_staleness_buffer_id: int | None) -> int:
        """
        Count the total number of EMPTY entries in all PENDING buffers up to max_staleness_buffer_id.

        Args:
            max_staleness_buffer_id (int | None): The maximum buffer ID to consider.
        Returns:
            int: The total number of EMPTY entries in eligible buffers.
        """
        if not self.is_validate:
            assert max_staleness_buffer_id is not None, (
                "max_staleness_buffer_id must be provided for non-validation inventory"
            )
            self.ensure_buffer_exists(max_staleness_buffer_id)

        pending_buffers = self.get_buffers_with_capacity()
        return sum(
            self.buffers[bid].get_empty_entries_num()
            for bid in pending_buffers
            if (max_staleness_buffer_id is None or bid <= max_staleness_buffer_id)
        )

    def get_max_pending_buffer_id(self, max_staleness_buffer_id: int) -> int:
        """
        Get the max buffer ID that is in PENDING state.

        Args:
            max_staleness_buffer_id (int): The maximum buffer ID to consider.
        Returns:
            Optional[int]:
                The max buffer ID that is in PENDING state.
                If no PENDING buffers are available, return None.
        """
        pending_buffers = self.get_buffers_with_capacity()
        filtered_buffers = [bid for bid in pending_buffers if bid <= max_staleness_buffer_id]
        return max(filtered_buffers) if filtered_buffers else None

    def can_reserve_data_without_new_reserve_entry(self, entry_info: EntryInfo, model_version: int) -> bool:
        """
        Check whether an entry can be reserved for a given model version without a new reserve entry.
        """
        staleness = self.staleness if self.staleness is not None else 0
        if entry_info.prompt_id in self.data_tracker:
            if self.is_validate:
                return True
            buffer_id, _ = self.data_tracker[entry_info.prompt_id]
            if model_version + staleness >= buffer_id:
                return True
        return False

    def can_reserve_data(self, entry_info: EntryInfo, model_version: int) -> bool:
        """
        Check whether an entry can be reserved for a given model version.

        Args:
            entry_info (EntryInfo): The entry metadata to reserve.
            model_version (int): The model version to check.
        Returns:
            bool: Whether the entry can be reserved for the given model version
        """
        # Validation treats absent staleness as zero.
        staleness = self.staleness if self.staleness is not None else 0
        if entry_info.prompt_id in self.data_tracker:
            # Existing prompt groups reuse their reservation while the model version remains admissible.
            buffer_id, _ = self.data_tracker[entry_info.prompt_id]
            if model_version + staleness >= buffer_id:
                return True
            else:
                return False
        self.ensure_buffer_exists(model_version + staleness)
        pending_buffers = self.get_buffers_with_capacity()
        if self.is_validate:
            return any(buffer_id not in self._ready_for_delete_buffer_ids for buffer_id in pending_buffers)

        # Get all PENDING buffers within the staleness limit.
        candidate_ids = [
            bid
            for bid in pending_buffers
            if model_version <= bid <= model_version + staleness and bid not in self._ready_for_delete_buffer_ids
        ]
        if not candidate_ids:
            return False
        return True

    def reserve_data(
        self, entry_info: EntryInfo, max_staleness_buffer_id: int | None
    ) -> tuple[int | None, int | None]:
        """
        Reserve an entry for a rollout instance in an appropriate buffer.

        Args:
            entry_info (EntryInfo): The entry metadata to reserve.
            max_staleness_buffer_id (int | None): The maximum buffer ID allowed by staleness.
        Returns:
            Tuple[Optional[int], Optional[int]]: The buffer ID and entry ID reserved, or (None, None) if not available.
        """
        if self.is_validate and max_staleness_buffer_id is not None:
            max_staleness_buffer_id = None
            psrl_logger.warning("Validation inventory must have max_staleness_buffer_id=None, setting it to None.")

        if entry_info.prompt_id in self.data_tracker:
            buffer_id, entry_id = self.data_tracker[entry_info.prompt_id]
            psrl_logger.debug(
                f"[Reserved Entry Update]: Existing reservation. Entry: {entry_info}. "
                f"Buffer: {buffer_id}. Slot: {entry_id}."
            )
            tracked_entry_info = self.buffers[buffer_id].entries[entry_id].entry_info
            if not isinstance(tracked_entry_info.request_idx, list):
                tracked_entry_info.request_idx = [tracked_entry_info.request_idx]
            entry_request_idx = entry_info.request_idx
            if entry_request_idx in tracked_entry_info.request_idx:
                # Gateway retries return the existing reservation without duplicating request indices.
                psrl_logger.warning(
                    f"Returning existing reservation for a duplicate request. Entry: {entry_info}. "
                    f"Buffer: {buffer_id}. Slot: {entry_id}."
                )
                return buffer_id, entry_id
            tracked_entry_info.request_idx.append(entry_request_idx)

            if (
                not isinstance(tracked_entry_info.model_version, list)
                and tracked_entry_info.model_version != entry_info.model_version
            ):
                tracked_entry_info.model_version = [tracked_entry_info.model_version] * len(
                    tracked_entry_info.request_idx
                )
                tracked_entry_info.model_version.append(entry_info.model_version)
            elif isinstance(tracked_entry_info.model_version, list):
                tracked_entry_info.model_version.append(entry_info.model_version)

            if (
                not isinstance(tracked_entry_info.rollout_instance_id, list)
                and tracked_entry_info.rollout_instance_id != entry_info.rollout_instance_id
            ):
                tracked_entry_info.rollout_instance_id = [tracked_entry_info.rollout_instance_id] * (
                    len(tracked_entry_info.request_idx) - 1
                )
                tracked_entry_info.rollout_instance_id.append(entry_info.rollout_instance_id)
            elif isinstance(tracked_entry_info.rollout_instance_id, list):
                tracked_entry_info.rollout_instance_id.append(entry_info.rollout_instance_id)

            if (
                not isinstance(tracked_entry_info.n_trajectory, list)
                and tracked_entry_info.n_trajectory != entry_info.n_trajectory
            ):
                tracked_entry_info.n_trajectory = [tracked_entry_info.n_trajectory] * (
                    len(tracked_entry_info.request_idx) - 1
                )
                tracked_entry_info.n_trajectory.append(entry_info.n_trajectory)
            elif isinstance(tracked_entry_info.n_trajectory, list):
                tracked_entry_info.n_trajectory.append(entry_info.n_trajectory)

            self.buffers[buffer_id].entries[entry_id].entry_info = tracked_entry_info
            return buffer_id, entry_id

        if not self.is_validate:
            assert max_staleness_buffer_id is not None, (
                "max_staleness_buffer_id must be provided for non-validation inventory"
            )
            self.ensure_buffer_exists(max_staleness_buffer_id)

        pending_buffers = self.get_buffers_with_capacity()
        if not max_staleness_buffer_id:
            assert len(pending_buffers) == 1, (
                f"Only one PENDING buffer should exist when max_staleness_buffer_id is None, "
                f"but got {pending_buffers}, and current buffers are {self.buffers.keys()}."
            )
            candidate_ids = list(pending_buffers)
        else:
            candidate_ids = [
                bid
                for bid in pending_buffers
                if max_staleness_buffer_id - self.staleness <= bid <= max_staleness_buffer_id
                and bid not in self._ready_for_delete_buffer_ids
            ]

        if not candidate_ids:
            return None, None

        target_buffer_id = max(candidate_ids)
        buffer = self.buffers[target_buffer_id]
        entry_id = buffer.get_last_non_reserved()
        assert entry_id != -1 and buffer.entries[entry_id].category == EntryCategory.EMPTY, (
            f"Non-reserved entry must be empty. Actual entry: {buffer.entries[entry_id]}. "
            f"Buffer: {target_buffer_id}. Slot: {entry_id}."
        )

        buffer.insert(entry_id, EntryCategory.RESERVED, entry_info=entry_info)
        self.data_tracker[entry_info.prompt_id] = (target_buffer_id, entry_id)
        self._update_buffer_status(target_buffer_id)

        psrl_logger.debug(
            f"[Entry Reserve]: Reserved entry: {entry_info}. Buffer: {target_buffer_id}. Slot: {entry_id}."
        )

        return target_buffer_id, entry_id

    def update_request_version_tag(
        self,
        request_id: int,
        new_version_tag: int,
    ):
        """
        Update the version tag of a specific request in the data tracker and buffer.

        Args:
            request_id (int): The global unique request ID to update.
            new_version_tag (int): The new model version tag to set.
        Raises:
            AssertionError: If the request ID is not found or the new version tag is out of staleness bounds.
        """

        prompt_id = request_id // self.rollout_n
        request_idx = request_id % self.rollout_n
        entry_info_to_update = None
        if prompt_id not in self.data_tracker:
            raise AssertionError(f"Prompt ID {prompt_id} not found in data tracker")

        buffer_id, entry_id = self.data_tracker[prompt_id]
        entry_info_to_update = self.buffers[buffer_id].entries[entry_id].entry_info

        if isinstance(entry_info_to_update.model_version, list):
            request_idx_in_list = entry_info_to_update.request_idx.index(request_idx)
            entry_info_to_update.model_version[request_idx_in_list] = new_version_tag
        elif entry_info_to_update.model_version != new_version_tag:
            if isinstance(entry_info_to_update.request_idx, list):
                request_num = len(entry_info_to_update.request_idx)
                request_idx_in_list = entry_info_to_update.request_idx.index(request_idx)
                entry_info_to_update.model_version = [entry_info_to_update.model_version] * request_num
                entry_info_to_update.model_version[request_idx_in_list] = new_version_tag
            else:
                _assert_scalar_request_idx(entry_info_to_update.request_idx)
                entry_info_to_update.model_version = new_version_tag

        psrl_logger.debug(
            f"[Entry Update]: Updated model version. Request index: {request_idx}. "
            f"Buffer: {buffer_id}. Slot: {entry_id}. Entry: {entry_info_to_update}. "
            f"New version: {new_version_tag}."
        )

    def update_request_instance_id(
        self,
        request_id: int,
        new_instance_id: RolloutInstanceId,
    ):
        """
        Update the instance id of a specific request in the data tracker and buffer.

        Args:
            request_id (int): The global unique request ID to update.
            new_instance_id (RolloutInstanceId): The new instance id to set.
        Raises:
            AssertionError: If the request ID is not found or the new instance id is out of bounds.
        """
        prompt_id = request_id // self.rollout_n
        request_idx = request_id % self.rollout_n
        if prompt_id not in self.data_tracker:
            raise AssertionError(f"Prompt ID {prompt_id} not found in data tracker")

        buffer_id, entry_id = self.data_tracker[prompt_id]
        entry_info_to_update = self.buffers[buffer_id].entries[entry_id].entry_info

        if isinstance(entry_info_to_update.rollout_instance_id, list):
            request_idx_in_list = entry_info_to_update.request_idx.index(request_idx)
            entry_info_to_update.rollout_instance_id[request_idx_in_list] = new_instance_id
        elif entry_info_to_update.rollout_instance_id != new_instance_id:
            if isinstance(entry_info_to_update.request_idx, list):
                request_num = len(entry_info_to_update.request_idx)
                request_idx_in_list = entry_info_to_update.request_idx.index(request_idx)
                entry_info_to_update.rollout_instance_id = [entry_info_to_update.rollout_instance_id] * request_num
                entry_info_to_update.rollout_instance_id[request_idx_in_list] = new_instance_id
            else:
                _assert_scalar_request_idx(entry_info_to_update.request_idx)
                entry_info_to_update.rollout_instance_id = new_instance_id

        psrl_logger.debug(
            f"[Entry Update]: Updated rollout instance. Request index: {request_idx}. "
            f"Buffer: {buffer_id}. Slot: {entry_id}. Entry: {entry_info_to_update}. "
            f"New instance ID: {new_instance_id}."
        )

    def update_request_n_trajectory(
        self,
        request_id: int,
        new_n_trajectory: int,
    ):
        """
        Update the number of trajectories for a specific request in the data tracker and buffer.

        Args:
            request_id (int): The global unique request ID to update.
            n_trajectory (int): The new number of trajectories to set.
        Raises:
            AssertionError: If the request ID is not found or the new number of trajectories is invalid.
        """
        prompt_id = request_id // self.rollout_n
        request_idx = request_id % self.rollout_n
        if prompt_id not in self.data_tracker:
            raise AssertionError(f"Prompt ID {prompt_id} not found in data tracker")

        buffer_id, entry_id = self.data_tracker[prompt_id]
        entry_info_to_update = self.buffers[buffer_id].entries[entry_id].entry_info

        if isinstance(entry_info_to_update.n_trajectory, list):
            request_idx_in_list = entry_info_to_update.request_idx.index(request_idx)
            entry_info_to_update.n_trajectory[request_idx_in_list] = new_n_trajectory
        elif entry_info_to_update.n_trajectory != new_n_trajectory:
            if isinstance(entry_info_to_update.request_idx, list):
                request_num = len(entry_info_to_update.request_idx)
                request_idx_in_list = entry_info_to_update.request_idx.index(request_idx)
                entry_info_to_update.n_trajectory = [entry_info_to_update.n_trajectory] * request_num
                entry_info_to_update.n_trajectory[request_idx_in_list] = new_n_trajectory
            else:
                _assert_scalar_request_idx(entry_info_to_update.request_idx)
                entry_info_to_update.n_trajectory = new_n_trajectory

        psrl_logger.debug(
            f"[Entry Update]: Updated trajectory count. Request index: {request_idx}. "
            f"Buffer: {buffer_id}. Slot: {entry_id}. Entry: {entry_info_to_update}. "
            f"New count: {new_n_trajectory}."
        )

    def clear_buffer(
        self,
        buffer_id: int,
    ):
        """
        Clear all entries in the buffer, resetting them to EMPTY and updating the data tracker.

        Args:
            buffer_id (int): The ID of the buffer to clear.
        """
        entries = self.buffers[buffer_id].entries
        entry_infos = [entry.entry_info for entry in entries]
        self.buffers[buffer_id].clear()
        for entry_info in entry_infos:
            del self.data_tracker[entry_info.prompt_id]
        self._update_buffer_status(buffer_id)

    def move_occupied_entries(
        self,
        entry_infos: EntryInfo | list[EntryInfo],
        buffer_id: int,
    ):
        """
        Move occupied entries to a specific buffer.

        `buffer_id` is used only to verify the destination chosen by staleness rules.

        Args:
            entry_infos (Union[EntryInfo, List[EntryInfo]]): The entry infos to move.
            buffer_id (int): The ID of the buffer to move the occupied entries to.
        """
        if not isinstance(entry_infos, list):
            entry_infos = [entry_infos]
        prompt_ids = [entry_info.prompt_id for entry_info in entry_infos]
        self.clear_occupied_entries(prompt_ids)
        for entry_info in entry_infos:
            occupied_buffer_id, _, _ = self.occupy_data_without_reserve(entry_info)
            assert occupied_buffer_id == buffer_id, (
                f"Occupied buffer differs from target. Actual ID: {occupied_buffer_id}. Target ID: {buffer_id}."
            )

    def clear_occupied_entries(
        self,
        prompt_ids: int | list[int],
    ):
        """
        Clear OCCUPIED entries from buffers and update data tracker.
        During clearing, move the last OCCUPIED entry to fill the cleared entry to maintain contiguity.

        Args:
            prompt_ids (Union[int, List[int]]): The prompt IDs to clear.
        """
        if not isinstance(prompt_ids, list):
            prompt_ids = [prompt_ids]
        updated_buffer_ids = []
        for prompt_id in prompt_ids:
            assert prompt_id in self.data_tracker, f"Cannot clear an untracked occupied entry. Prompt ID: {prompt_id}."
            buffer_id, entry_id = self.data_tracker[prompt_id]
            buffer = self.buffers[buffer_id]
            last_occupied_entry_id = buffer.get_first_non_occupied() - 1
            assert 0 <= entry_id <= last_occupied_entry_id, (
                f"Entry to clear is outside the occupied prefix. Entry ID: {entry_id}. "
                f"Buffer ID: {buffer_id}. Last occupied entry ID: {last_occupied_entry_id}."
            )
            psrl_logger.debug(
                f"[Occupied Entry Clear]: Cleared entry: {buffer.entries[entry_id].entry_info}. "
                f"Buffer: {buffer_id}. Slot: {entry_id}."
            )
            buffer.delete(entry_id)
            del self.data_tracker[prompt_id]

            if entry_id < last_occupied_entry_id:
                # Preserve the contiguous occupied prefix by moving its last entry into the gap.
                buffer.entries[entry_id] = buffer.entries[last_occupied_entry_id]
                buffer.delete(last_occupied_entry_id)
                moved_entry_info = buffer.entries[entry_id].entry_info
                assert moved_entry_info is not None, "Moved entry must not be None"
                self.data_tracker[moved_entry_info.prompt_id] = (buffer_id, entry_id)
                psrl_logger.debug(
                    f"[Occupied Entry Move]: Moved entry: {moved_entry_info}. "
                    f"Buffer: {buffer_id}. Source slot: {last_occupied_entry_id}. "
                    f"Target slot: {entry_id}."
                )
            updated_buffer_ids.append(buffer_id)

        for buffer_id in set(updated_buffer_ids):
            self._update_buffer_status(buffer_id)

    def clear_reserved_entries(
        self,
        prompt_ids: int | list[int],
        move_across_buffer: bool = False,
    ):
        """
        Clear RESERVED entries from buffers and update data tracker.
        It may involve moving other RESERVED entries with the same model version to fill the blanks across buffers.

        Args:
            prompt_ids (Union[int, List[int]]): The prompt IDs to clear.
            move_across_buffer (bool): Whether to move RESERVED entries across buffers to fill cleared entries.
                Currently only used for OCCUPY operation.
        """
        if self.is_validate and move_across_buffer:
            move_across_buffer = False
            psrl_logger.warning("Validation inventory must have move_across_buffer=False, setting it to False.")

        if not isinstance(prompt_ids, list):
            prompt_ids = [prompt_ids]

        changed_buffer_ids = set()
        for prompt_id in prompt_ids:
            if prompt_id not in self.data_tracker:
                continue
            buffer_id, entry_id = self.data_tracker[prompt_id]
            buffer = self.buffers[buffer_id]
            entry_info = buffer.entries[entry_id].entry_info
            if not move_across_buffer:
                last_reserved_entry_id = buffer.get_last_non_reserved() + 1
                buffer.delete(entry_id)
                del self.data_tracker[prompt_id]
                psrl_logger.debug(
                    f"[Reserved Entry Clear]: Cleared entry: {entry_info}. Buffer: {buffer_id}. Slot: {entry_id}."
                )
                changed_buffer_ids.add(buffer_id)
                # Preserve the reserved suffix by moving its boundary entry into the gap.
                if entry_id > last_reserved_entry_id:
                    buffer.entries[entry_id] = buffer.entries[last_reserved_entry_id]
                    buffer.delete(last_reserved_entry_id)
                    moved_entry_info = buffer.entries[entry_id].entry_info
                    assert moved_entry_info is not None, "Moved entry must not be None"
                    self.data_tracker[moved_entry_info.prompt_id] = (
                        buffer_id,
                        entry_id,
                    )
                    psrl_logger.debug(
                        f"[Reserved Entry Move (1/1)]: Moved entry: {moved_entry_info}. "
                        f"Buffer: {buffer_id}. Source slot: {last_reserved_entry_id}. "
                        f"Target slot: {entry_id}."
                    )
            else:
                # Group sampling uses the minimum version as the entry's staleness constraint.
                model_version = entry_info.get_entry_version()
                first_reserved_entry_id = None
                exchange_buffer_id = None
                # NOTE(lhy): Search only earlier buffers inside the staleness window because later
                # moves cannot create useful capacity in the current buffer.
                for bid in range(model_version, min(model_version + self.staleness, buffer_id) + 1):
                    if bid not in self.buffers or self.buffers[bid].get_status() in [
                        BufferStatus.READY,
                        BufferStatus.READY_WITH_CAPACITY,
                    ]:
                        continue
                    b = self.buffers[bid]
                    for eid, entry in enumerate(b.entries):
                        # Later slots in the same buffer cannot fill an earlier gap.
                        if bid == buffer_id and eid >= entry_id:
                            break
                        if entry.category == EntryCategory.RESERVED:
                            entry_model_version = entry.entry_info.get_entry_version()
                            if (
                                entry_model_version >= buffer_id - self.staleness
                                and entry_model_version <= buffer_id
                                and entry.entry_info.prompt_id not in prompt_ids
                            ):
                                first_reserved_entry_id = eid
                                exchange_buffer_id = bid
                                break
                    if first_reserved_entry_id is not None and exchange_buffer_id is not None:
                        break

                if first_reserved_entry_id is not None and exchange_buffer_id is not None:
                    exchange_buffer = self.buffers[exchange_buffer_id]
                    last_reserved_entry_id = exchange_buffer.get_last_non_reserved() + 1
                    buffer.delete(entry_id)
                    del self.data_tracker[prompt_id]
                    psrl_logger.debug(
                        f"[Reserved Entry Clear]: Cleared entry: {entry_info}. Buffer: {buffer_id}. Slot: {entry_id}."
                    )
                    changed_buffer_ids.add(buffer_id)
                    first_reserved_entry_info = exchange_buffer.entries[first_reserved_entry_id].entry_info
                    assert first_reserved_entry_info is not None, "First reserved entry to move must not be None"
                    buffer.entries[entry_id] = exchange_buffer.entries[first_reserved_entry_id]
                    exchange_buffer.delete(first_reserved_entry_id)
                    self.data_tracker[first_reserved_entry_info.prompt_id] = (
                        buffer_id,
                        entry_id,
                    )
                    psrl_logger.debug(
                        f"[Reserved Entry Move (1/2)]: Moved entry: {first_reserved_entry_info}. "
                        f"Source buffer: {exchange_buffer_id}. Source slot: {first_reserved_entry_id}. "
                        f"Target buffer: {buffer_id}. Target slot: {entry_id}."
                    )

                    # Preserve the source buffer's reserved suffix after the move.
                    if first_reserved_entry_id > last_reserved_entry_id:
                        exchange_buffer.entries[first_reserved_entry_id] = exchange_buffer.entries[
                            last_reserved_entry_id
                        ]
                        exchange_buffer.delete(last_reserved_entry_id)
                        moved_entry_info = exchange_buffer.entries[first_reserved_entry_id].entry_info
                        assert moved_entry_info is not None, "Moved entry must not be None"
                        self.data_tracker[moved_entry_info.prompt_id] = (
                            exchange_buffer_id,
                            first_reserved_entry_id,
                        )
                        psrl_logger.debug(
                            f"[Reserved Entry Move (2/2)]: Moved entry: {moved_entry_info}. "
                            f"Buffer: {exchange_buffer_id}. Source slot: {last_reserved_entry_id}. "
                            f"Target slot: {first_reserved_entry_id}."
                        )

                    if exchange_buffer_id not in changed_buffer_ids:
                        changed_buffer_ids.add(exchange_buffer_id)
                else:
                    buffer.delete(entry_id)
                    del self.data_tracker[prompt_id]
                    psrl_logger.debug(
                        f"[Reserved Entry Clear]: Cleared entry: {entry_info}. Buffer: {buffer_id}. Slot: {entry_id}."
                    )
                    changed_buffer_ids.add(buffer_id)

        for buffer_id in changed_buffer_ids:
            self._update_buffer_status(buffer_id)

    def occupy_data_without_reserve(
        self,
        entry_info: EntryInfo,
    ):
        """
        Append data to an appropriate buffer, occupying it.

        Args:
            entry_info (EntryInfo): The entry metadata to occupy.
        """

        model_version = entry_info.get_entry_version()
        pending_buffers = self.get_buffers_with_capacity()
        candidate_ids = [
            bid
            for bid in pending_buffers
            if (
                model_version <= bid <= model_version + self.staleness
                and self.buffers[bid].get_first_non_occupied() < self.buffers[bid].ready_num_entries
            )
        ]
        assert candidate_ids, "No suitable PENDING buffer found."

        buffer_id = min(candidate_ids)
        buffer = self.buffers[buffer_id]
        entry_id = buffer.get_first_non_occupied()
        assert (
            entry_id < self.buffers[buffer_id].ready_num_entries
            and buffer.entries[entry_id].category == EntryCategory.EMPTY
        ), (
            f"Non-occupied entry must be empty. Actual entry: {buffer.entries[entry_id]}. "
            f"Buffer: {buffer_id}. Slot: {entry_id}."
        )

        buffer.insert(entry_id, EntryCategory.OCCUPIED, entry_info=entry_info)
        self.data_tracker[entry_info.prompt_id] = (buffer_id, entry_id)
        psrl_logger.debug(
            f"[Entry Occupy (without reserve)]: Occupied entry: {entry_info}. Buffer: {buffer_id}. Slot: {entry_id}."
        )
        occupy_num = buffer.get_first_non_occupied()
        self._update_buffer_status(buffer_id)
        return buffer_id, entry_id, occupy_num

    def occupy_data_with_reserve(
        self,
        prompt_id: int,
    ) -> tuple[int | None, int | None, int | None]:
        """
        Move data to the first non-occupied entry in an appropriate buffer, occupying it.

        Args:
            prompt_id (int): The prompt ID to occupy.
        Returns:
            Tuple[Optional[int], Optional[int], Optional[int]]:
                The buffer ID, entry ID, and occupy number after occupying,
                or (None, None, None) if no suitable buffer/entry is available.
        """
        assert prompt_id in self.data_tracker, (
            f"Prompt must have an existing mapping. ID: {prompt_id}. Data tracker: {self.data_tracker}."
        )

        old_buffer_id, old_entry_id = self.data_tracker[prompt_id]
        entry_info = self.buffers[old_buffer_id].entries[old_entry_id].entry_info

        model_version = entry_info.get_entry_version()
        if self.is_validate:
            buffer_id = self.min_not_ready_buffer_id()
            buffer = self.buffers[buffer_id]
            have_possible_occupy_target = buffer.get_first_non_occupied() < buffer.ready_num_entries
            possible_occupy_target_buffer_id = buffer_id
            possible_occupy_target_entry_id = buffer.get_first_non_occupied()
        else:
            min_buffer_id = model_version - self.staleness
            have_possible_occupy_target = False
            possible_occupy_target_buffer_id = None
            possible_occupy_target_entry_id = None
            for buffer_id in range(min_buffer_id, model_version + self.staleness + 1):
                if buffer_id not in self.buffers:
                    continue
                buffer = self.buffers[buffer_id]
                if buffer.get_first_non_occupied() < buffer.ready_num_entries:
                    have_possible_occupy_target = True
                    possible_occupy_target_buffer_id = buffer_id
                    possible_occupy_target_entry_id = buffer.get_first_non_occupied()
                    break

        assert have_possible_occupy_target, f"No occupation target found. Prompt ID: {prompt_id}."

        self.clear_reserved_entries(prompt_id, move_across_buffer=(not self.is_validate))

        if old_entry_id < self.buffers[old_buffer_id].ready_num_entries:
            pending_buffers = self.get_buffers_with_capacity()
            if not self.is_validate:
                candidate_ids = [
                    bid
                    for bid in pending_buffers
                    if (
                        model_version <= bid <= model_version + self.staleness
                        and self.buffers[bid].get_first_non_occupied() < self.buffers[bid].ready_num_entries
                    )
                ]
            else:
                candidate_ids = list(pending_buffers)

            assert candidate_ids, (
                f"No pending buffer is available during occupation. Prompt ID: {prompt_id}. "
                f"Pending buffers: {pending_buffers}. Reserved buffer: {old_buffer_id}. "
                f"Reserved slot: {old_entry_id}. "
                f"Post-clear status: {self.buffers[old_buffer_id].get_status()}. "
                f"First non-occupied slot: {self.buffers[old_buffer_id].get_first_non_occupied()}."
            )

            buffer_id = min(candidate_ids)
            buffer = self.buffers[buffer_id]
            entry_id = buffer.get_first_non_occupied()
            assert (
                entry_id < self.buffers[buffer_id].ready_num_entries
                and buffer.entries[entry_id].category == EntryCategory.EMPTY
            ), (
                f"Non-occupied entry must be empty. Actual entry: {buffer.entries[entry_id]}. "
                f"Buffer: {buffer_id}. Slot: {entry_id}."
            )

            buffer.insert(entry_id, EntryCategory.OCCUPIED, entry_info=entry_info)
            self.data_tracker[entry_info.prompt_id] = (buffer_id, entry_id)
            psrl_logger.debug(
                f"[Entry Occupy (with reserve)]: Occupied entry: {entry_info}. Buffer: {buffer_id}. Slot: {entry_id}."
            )
            occupy_num = buffer.get_first_non_occupied()
            self._update_buffer_status(buffer_id)
            return buffer_id, entry_id, occupy_num

        else:
            # Redundant rollouts may move to a later buffer than their original reservation.
            pending_buffers = self.get_buffers_with_capacity()
            not_ready_buffers = self.get_buffers_not_ready()
            candidate_ids = [
                bid
                for bid in not_ready_buffers
                if (
                    model_version <= bid <= model_version + self.staleness
                    and self.buffers[bid].get_first_non_occupied() < self.buffers[bid].ready_num_entries
                )
            ]

            assert candidate_ids, (
                f"No not-ready buffer is available during occupation. Prompt ID: {prompt_id}. "
                f"Not-ready buffers: {not_ready_buffers}. Reserved buffer: {old_buffer_id}. "
                f"Reserved slot: {old_entry_id}. Expected buffer: {possible_occupy_target_buffer_id}. "
                f"Expected slot: {possible_occupy_target_entry_id}. "
                f"Actual status: {self.buffers[possible_occupy_target_buffer_id].get_status()}. "
                f"First non-occupied slot: "
                f"{self.buffers[possible_occupy_target_buffer_id].get_first_non_occupied()}."
            )

            for buffer_id in candidate_ids:
                buffer = self.buffers[buffer_id]
                entry_id = buffer.get_first_non_occupied()
                assert entry_id < self.buffers[buffer_id].ready_num_entries, (
                    f"Non-occupied entry exceeds ready capacity. Entry ID: {entry_id}."
                )

                if buffer.entries[entry_id].category == EntryCategory.EMPTY:
                    pass
                else:
                    assert buffer.entries[entry_id].category == EntryCategory.RESERVED, (
                        f"Entry must be reserved. Actual entry: {buffer.entries[entry_id]}. "
                        f"Buffer: {buffer_id}. Slot: {entry_id}."
                    )
                    available_buffers = sorted(
                        [
                            bid
                            for bid in pending_buffers
                            if bid < buffer_id and bid not in self._ready_for_delete_buffer_ids
                        ],
                        reverse=True,
                    )
                    if len(available_buffers) == 0:
                        continue
                    exchange_buffer_id = available_buffers[0]
                    exchange_buffer = self.buffers[exchange_buffer_id]
                    last_empty_entry_id = exchange_buffer.get_last_non_reserved()
                    assert (
                        last_empty_entry_id >= 0
                        and exchange_buffer.entries[last_empty_entry_id].category == EntryCategory.EMPTY
                    ), f"Exchange slot must be empty. Buffer: {exchange_buffer_id}. Slot: {last_empty_entry_id}."

                    exchange_buffer.entries[last_empty_entry_id] = buffer.entries[entry_id]
                    original_entry_info = buffer.entries[entry_id].entry_info
                    buffer.delete(entry_id)
                    self.data_tracker[original_entry_info.prompt_id] = (
                        exchange_buffer_id,
                        last_empty_entry_id,
                    )
                    psrl_logger.debug(
                        f"[Reserved Entry Move (Due to Redundant Rollout)]: Moved entry: {original_entry_info}. "
                        f"Source buffer: {buffer_id}. Source slot: {entry_id}. "
                        f"Target buffer: {exchange_buffer_id}. Target slot: {last_empty_entry_id}."
                    )
                    self._update_buffer_status(exchange_buffer_id)

                buffer.insert(entry_id, EntryCategory.OCCUPIED, entry_info=entry_info)
                self.data_tracker[entry_info.prompt_id] = (buffer_id, entry_id)
                psrl_logger.debug(
                    f"[Entry Occupy (Redundant Rollout Becomes Useful)]: Occupied entry: {entry_info}. "
                    f"Buffer: {buffer_id}. Slot: {entry_id}."
                )
                occupy_num = buffer.get_first_non_occupied()
                self._update_buffer_status(buffer_id)
                return buffer_id, entry_id, occupy_num

            # If compaction cannot free a ready slot, retain the redundant result in overflow capacity.
            candidate_ids = [
                bid
                for bid in pending_buffers
                if (
                    model_version <= bid <= model_version + self.staleness
                    and self.buffers[bid].get_first_non_occupied() < self.buffers[bid].ready_num_entries
                )
            ]
            buffer_id = min(candidate_ids)
            buffer = self.buffers[buffer_id]
            entry_id = buffer.get_first_non_occupied()
            assert (
                entry_id >= self.buffers[buffer_id].ready_num_entries
                and buffer.entries[entry_id].category == EntryCategory.EMPTY
            ), (
                f"Overflow entry must be empty and beyond ready capacity. "
                f"Actual entry: {buffer.entries[entry_id]}. Buffer: {buffer_id}. Slot: {entry_id}."
            )
            buffer.insert(entry_id, EntryCategory.OCCUPIED, entry_info=entry_info)
            self.data_tracker[entry_info.prompt_id] = (buffer_id, entry_id)
            psrl_logger.debug(
                f"[Entry Occupy (Redundant Rollout is Wasted)]: Occupied entry: {entry_info}. "
                f"Buffer: {buffer_id}. Slot: {entry_id}."
            )
            occupy_num = buffer.get_first_non_occupied()
            self._update_buffer_status(buffer_id)
            return buffer_id, entry_id, occupy_num
