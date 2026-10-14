# tests/staleness/conftest.py
import pytest
from psrl.workers.ps.staleness_controller import (
    StalenessBuffer,
    StalenessInventory,
)


@pytest.fixture
def staleness_inventory():
    """Fresh StalenessInventory: 5 entries, 3 needed to be ready, staleness=2, rollout_n=1."""
    return StalenessInventory(num_entries=5, ready_num_entries=3, staleness=2, rollout_n=1)


@pytest.fixture
def staleness_buffer():
    """A direct StalenessBuffer (inner class) for low-level insert/delete tests."""
    return StalenessBuffer(num_entries=5, ready_num_entries=3, staleness=2)
