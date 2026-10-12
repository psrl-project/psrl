# tests/staleness/test_staleness_controller.py
import pytest
from psrl.workers.ps.staleness_controller import (
    EntryCategory,
    EntryInfo,
    StalenessInventory,
)

pytestmark = pytest.mark.cpu_test


# ── EntryInfo hashing (migrated from unit_tests/staleness/test_hash.py) ──────


class TestEntryInfoHashing:
    def test_same_prompt_id_is_equal(self):
        """EntryInfo equality is determined solely by prompt_id."""
        e1 = EntryInfo(rollout_instance_id=("w", 0), prompt_id=1, request_idx=0, model_version=5)
        e2 = EntryInfo(rollout_instance_id=("w", 1), prompt_id=1, request_idx=3, model_version=9)
        assert e1 == e2

    def test_different_prompt_id_is_not_equal(self):
        e1 = EntryInfo(rollout_instance_id=("w", 0), prompt_id=1, request_idx=0, model_version=5)
        e2 = EntryInfo(rollout_instance_id=("w", 0), prompt_id=2, request_idx=0, model_version=5)
        assert e1 != e2

    def test_same_prompt_id_has_same_hash(self):
        e1 = EntryInfo(rollout_instance_id=("w", 0), prompt_id=42, request_idx=0, model_version=1)
        e2 = EntryInfo(rollout_instance_id=("w", 99), prompt_id=42, request_idx=7, model_version=9)
        assert hash(e1) == hash(e2)

    def test_usable_as_dict_key(self, dummy_entry_info):
        d = {dummy_entry_info: "value"}
        assert d[dummy_entry_info] == "value"


# StalenessBuffer low-level insert/delete


class TestStalenessBufferInsert:
    def test_initial_all_entries_empty(self, staleness_buffer):
        for entry in staleness_buffer.entries:
            assert entry.category == EntryCategory.EMPTY

    def test_insert_reserved_marks_entry(self, staleness_buffer, dummy_entry_info):
        staleness_buffer.insert(0, EntryCategory.RESERVED, dummy_entry_info)
        assert staleness_buffer.entries[0].category == EntryCategory.RESERVED

    def test_insert_occupied_marks_entry(self, staleness_buffer, dummy_entry_info):
        staleness_buffer.insert(0, EntryCategory.OCCUPIED, dummy_entry_info)
        assert staleness_buffer.entries[0].category == EntryCategory.OCCUPIED

    def test_delete_resets_to_empty(self, staleness_buffer, dummy_entry_info):
        staleness_buffer.insert(0, EntryCategory.RESERVED, dummy_entry_info)
        staleness_buffer.delete(0)
        assert staleness_buffer.entries[0].category == EntryCategory.EMPTY

    def test_out_of_bounds_insert_raises(self, staleness_buffer, dummy_entry_info):
        with pytest.raises(AssertionError):
            staleness_buffer.insert(999, EntryCategory.RESERVED, dummy_entry_info)

    def test_out_of_bounds_delete_raises(self, staleness_buffer):
        with pytest.raises(AssertionError):
            staleness_buffer.delete(999)


# StalenessInventory capacity and buffer creation


class TestStalenessInventoryCapacity:
    def test_starts_with_no_buffers(self, staleness_inventory):
        assert len(staleness_inventory.buffers) == 0

    def test_create_buffer_adds_to_inventory(self, staleness_inventory):
        staleness_inventory.create_buffer(0)
        assert len(staleness_inventory.buffers) == 1

    def test_buffer_id_increments_per_creation(self, staleness_inventory):
        staleness_inventory.create_buffer(0)
        staleness_inventory.create_buffer(1)
        assert len(staleness_inventory.buffers) == 2
        buffer_ids = list(staleness_inventory.buffers.keys())
        assert buffer_ids[0] < buffer_ids[1]


# StalenessBuffer.get_status()


class TestStalenessBufferStatus:
    def test_all_empty_is_pending(self, staleness_buffer):
        """A fresh buffer with all EMPTY entries has PENDING status."""
        from psrl.workers.ps.staleness_controller import BufferStatus

        status = staleness_buffer.get_status()
        assert status == BufferStatus.PENDING

    def test_enough_occupied_for_ready(self, staleness_buffer, dummy_entry_info):
        """Buffer with ready_num_entries OCCUPIED and remaining EMPTY is READY_WITH_CAPACITY."""
        from psrl.workers.ps.staleness_controller import BufferStatus

        # Three occupied entries leave two slots empty.
        for i in range(staleness_buffer.ready_num_entries):
            staleness_buffer.insert(i, EntryCategory.OCCUPIED, dummy_entry_info)
        status = staleness_buffer.get_status()
        assert status == BufferStatus.READY_WITH_CAPACITY

    def test_occupied_then_reserved_is_ready(self, staleness_buffer, dummy_entry_info):
        """Buffer with ready_num_entries OCCUPIED followed by all remaining RESERVED is READY."""
        from psrl.workers.ps.staleness_controller import BufferStatus

        # Slots 0-2 OCCUPIED (satisfies ready_num_entries=3), slots 3-4 RESERVED (no EMPTY left)
        for i in range(staleness_buffer.ready_num_entries):
            staleness_buffer.insert(i, EntryCategory.OCCUPIED, dummy_entry_info)
        for i in range(staleness_buffer.ready_num_entries, staleness_buffer.num_entries):
            staleness_buffer.insert(i, EntryCategory.RESERVED, dummy_entry_info)
        status = staleness_buffer.get_status()
        assert status == BufferStatus.READY

    def test_all_reserved_no_empty_is_stuck(self, staleness_buffer, dummy_entry_info):
        """Buffer with all RESERVED (no EMPTY, not enough OCCUPIED) is STUCK."""
        from psrl.workers.ps.staleness_controller import BufferStatus

        for i in range(staleness_buffer.num_entries):
            staleness_buffer.insert(i, EntryCategory.RESERVED, dummy_entry_info)
        status = staleness_buffer.get_status()
        assert status == BufferStatus.STUCK


class TestStalenessInventoryStalenessWindow:
    def test_reserve_data_returns_buffer_and_entry_id(self, staleness_inventory, dummy_entry_info):
        """Return buffer and entry IDs while creating missing buffers."""
        # For non-validate inventory, reserve_data auto-creates buffers via ensure_buffer_exists.
        # max_staleness_buffer_id must be an int (not None).
        result_buf, result_entry = staleness_inventory.reserve_data(
            entry_info=dummy_entry_info,
            max_staleness_buffer_id=2,  # staleness=2, so window covers buffer IDs 0..2
        )
        assert result_buf is not None
        assert result_entry is not None

    def test_reserve_data_returns_none_when_no_capacity_in_window(self, dummy_entry_info):
        """reserve_data returns (None, None) when no PENDING buffer exists within the staleness window.

        Setup: num_entries=1 per buffer, staleness=2, max_staleness_buffer_id=3, window=[1,3].
        Fill buffers 1, 2, 3 one slot each (→ STUCK). Then a new prompt has no PENDING candidate.
        Note: max_staleness_buffer_id must be > 0 because 0 is falsy and triggers the validate branch.
        """
        inv = StalenessInventory(num_entries=1, ready_num_entries=1, staleness=2, rollout_n=1)
        # Three distinct prompts fill buffers 3, 2, 1 (reserve_data picks the highest PENDING in window)
        for prompt_id in range(3):
            ei = EntryInfo(rollout_instance_id=("worker", 0), prompt_id=prompt_id, request_idx=0, model_version=0)
            inv.reserve_data(entry_info=ei, max_staleness_buffer_id=3)
        # Buffers 1, 2, 3 are now STUCK (1 slot each, fully RESERVED). Buffer 0 is outside window [1,3].
        result = inv.reserve_data(
            entry_info=EntryInfo(rollout_instance_id=("worker", 0), prompt_id=99, request_idx=0, model_version=0),
            max_staleness_buffer_id=3,
        )
        assert result == (None, None)


class TestUpdateRequestMetadata:
    def test_update_request_instance_id_accepts_python_int_request_idx(self):
        """SMG selection commit uses plain int request_idx from reserve_data()."""
        from psrl.workers.gen.utils import INVALID_ROLLOUT_INSTANCE_ID

        rollout_n = 8
        request_id = 227
        inv = StalenessInventory(num_entries=5, ready_num_entries=3, staleness=2, rollout_n=rollout_n)
        entry_info = EntryInfo(
            rollout_instance_id=INVALID_ROLLOUT_INSTANCE_ID,
            prompt_id=request_id // rollout_n,
            request_idx=request_id % rollout_n,
            model_version=0,
            n_trajectory=1,
        )
        inv.reserve_data(entry_info=entry_info, max_staleness_buffer_id=2)

        new_instance_id = ("019f0ec4-f2a9-7531-8939-5af9197513b8", 0)
        inv.update_request_instance_id(request_id=request_id, new_instance_id=new_instance_id)

        buffer_id, entry_id = inv.data_tracker[entry_info.prompt_id]
        stored = inv.buffers[buffer_id].entries[entry_id].entry_info
        assert stored.rollout_instance_id == new_instance_id

    def test_update_request_instance_id_assert_includes_actual_value(self):
        from psrl.workers.gen.utils import INVALID_ROLLOUT_INSTANCE_ID

        rollout_n = 8
        request_id = 227
        inv = StalenessInventory(num_entries=5, ready_num_entries=3, staleness=2, rollout_n=rollout_n)
        entry_info = EntryInfo(
            rollout_instance_id=INVALID_ROLLOUT_INSTANCE_ID,
            prompt_id=request_id // rollout_n,
            request_idx=request_id % rollout_n,
            model_version=0,
            n_trajectory=1,
        )
        buffer_id, entry_id = inv.reserve_data(entry_info=entry_info, max_staleness_buffer_id=2)
        stored = inv.buffers[buffer_id].entries[entry_id].entry_info
        stored.request_idx = "bad-type"

        with pytest.raises(AssertionError, match=r"got 'bad-type' \(type=str\)"):
            inv.update_request_instance_id(
                request_id=request_id,
                new_instance_id=("worker", 1),
            )
