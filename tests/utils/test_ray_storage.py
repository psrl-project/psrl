import pytest
from psrl.utils.ray_storage import plasma_backing_error

pytestmark = pytest.mark.cpu_test


def test_plasma_backing_requires_object_store_plus_reserve() -> None:
    snapshot = {
        "node_id": "node-1",
        "plasma_directory": "/tmp",
        "object_store_bytes": 100,
        "backing_capacity_bytes": 105,
        "shm_free_bytes": 1000,
    }
    error = plasma_backing_error(snapshot, reserve_bytes=10)
    assert error is not None
    assert "backing_capacity_bytes=105" in error


def test_plasma_backing_accepts_sufficient_capacity() -> None:
    snapshot = {
        "node_id": "node-1",
        "plasma_directory": "/dev/shm/psrl-ray-plasma",
        "object_store_bytes": 100,
        "backing_capacity_bytes": 1000,
        "shm_free_bytes": 1000,
    }
    assert plasma_backing_error(snapshot, reserve_bytes=10) is None


def test_plasma_still_requires_shared_memory_reserve() -> None:
    snapshot = {
        "node_id": "node-1",
        "plasma_directory": "/dev/shm/psrl-ray-plasma",
        "object_store_bytes": 100,
        "backing_capacity_bytes": 1000,
        "shm_free_bytes": 5,
    }
    error = plasma_backing_error(snapshot, reserve_bytes=10)
    assert error is not None
    assert "shm_free_bytes=5" in error


def test_plasma_backing_fails_closed_when_proc_diagnostics_are_unavailable() -> None:
    error = plasma_backing_error({"node_id": "node-1", "error": "not found"}, reserve_bytes=10)
    assert error == "node_id=node-1 inspection_error=not found"
