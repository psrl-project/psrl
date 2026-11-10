import threading
from types import SimpleNamespace

import pytest
from psrl.workers.ps.ps_manager import PSManager
from psrl.workers.ps.staleness_controller import EntryInfo, StalenessInventory

pytestmark = pytest.mark.cpu_test


def _validation_entry(prompt_id: int) -> EntryInfo:
    return EntryInfo(
        rollout_instance_id=32,
        prompt_id=prompt_id,
        request_idx=0,
        model_version=0,
        is_validate=True,
    )


def _validation_inventory(capacity: int = 1) -> StalenessInventory:
    inventory = StalenessInventory(
        num_entries=capacity,
        ready_num_entries=capacity,
        staleness=None,
        rollout_n=1,
        is_validate=True,
    )
    inventory.create_buffer_with_capacity(capacity, capacity)
    return inventory


def test_validation_inventory_reservation_ignores_staleness():
    inventory = _validation_inventory()
    entry = _validation_entry(prompt_id=0)

    assert inventory.can_reserve_data(entry, model_version=0)
    assert not inventory.can_reserve_data_without_new_reserve_entry(entry, model_version=0)

    assert inventory.reserve_data(entry, max_staleness_buffer_id=None) == (0, 0)
    assert inventory.can_reserve_data(entry, model_version=100)
    assert inventory.can_reserve_data_without_new_reserve_entry(entry, model_version=100)
    assert not inventory.can_reserve_data(_validation_entry(prompt_id=1), model_version=100)


def test_validation_reserve_indicator_uses_capacity_without_version_arithmetic():
    manager = PSManager.__new__(PSManager)
    manager._state_lock = threading.RLock()
    manager.val_staleness_inventory = _validation_inventory()
    manager.val_rollout_n = 1
    manager.max_aborted_version = -1
    manager._abort_request_ids = set()
    manager.psrl_config = SimpleNamespace(staleness=2)

    assert manager.get_reserve_indicator(0, [0], is_validate=True) == [0.0]

    manager.val_staleness_inventory.reserve_data(
        _validation_entry(prompt_id=0),
        max_staleness_buffer_id=None,
    )
    assert manager.get_reserve_indicator(0, [100], is_validate=True) == [float("-inf")]
    assert manager.get_reserve_indicator(1, [100], is_validate=True) == [float("inf")]
