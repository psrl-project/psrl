# tests/parameter_server/test_ps_manager.py

import pytest

pytestmark = pytest.mark.cpu_test


class TestPSManagerImport:
    def test_ps_manager_importable(self):
        from psrl.workers.ps.ps_manager import PSManager

        assert PSManager is not None

    def test_staleness_inventory_importable(self):
        from psrl.workers.ps.staleness_controller import StalenessInventory

        assert StalenessInventory is not None


class TestPSManagerLockSemantics:
    """Test the exclusive-push / shared-pull read-write lock semantics using plain Python."""

    def test_exclusive_push_lock_context_importable(self):
        from psrl.utils.ray.lock_context import exclusive_push_model_context

        assert exclusive_push_model_context is not None

    def test_shared_pull_lock_context_importable(self):
        from psrl.utils.ray.lock_context import shared_pull_model_context

        assert shared_pull_model_context is not None

    def test_busy_polling_ray_lock_acquire_returns_true_when_free(self):
        """BusyPollingRayLock decorator: acquire returns True when the lock is free."""
        from psrl.utils.ray.lock_context import add_busy_polling_lock

        @add_busy_polling_lock
        class _FakeLockActor:
            pass

        actor = _FakeLockActor()
        result = actor.acquire()
        assert result is True

    def test_busy_polling_ray_lock_acquire_returns_false_when_held(self):
        from psrl.utils.ray.lock_context import add_busy_polling_lock

        @add_busy_polling_lock
        class _FakeLockActor:
            pass

        actor = _FakeLockActor()
        first = actor.acquire()  # hold the lock
        assert first is True
        result = actor.acquire()  # attempt to acquire again
        assert result is False

    def test_busy_polling_ray_lock_release_frees_lock(self):
        from psrl.utils.ray.lock_context import add_busy_polling_lock

        @add_busy_polling_lock
        class _FakeLockActor:
            pass

        actor = _FakeLockActor()
        actor.acquire()
        actor.release()
        result = actor.acquire()
        assert result is True
