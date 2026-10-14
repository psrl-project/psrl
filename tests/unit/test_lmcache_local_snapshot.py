"""Unit tests for the current KV cache manager configuration surface."""

import os

import pytest
from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.manager import KVCacheManager

pytestmark = pytest.mark.cpu_test


def test_disabled_manager_has_no_engine_kwargs_or_environment(monkeypatch):
    manager = KVCacheManager(LMCacheConfig(enable=False))

    assert manager.enabled is False
    assert manager.get_engine_kwargs() == {}
    monkeypatch.delenv("LMCACHE_USE_EXPERIMENTAL", raising=False)
    manager.apply_env_vars()
    assert "LMCACHE_USE_EXPERIMENTAL" not in os.environ


def test_enabled_manager_builds_engine_kwargs():
    manager = KVCacheManager(LMCacheConfig(enable=True, offload_size_gb=2.5))

    assert manager.enabled is True
    assert manager.get_engine_kwargs() == {
        "kv_offloading_backend": "lmcache",
        "kv_offloading_size": 2.5,
        "disable_hybrid_kv_cache_manager": True,
    }


def test_set_instance_id_and_version_update_local_state():
    manager = KVCacheManager(LMCacheConfig(enable=False))

    manager.set_instance_id(3)
    manager.set_current_version(7)

    assert manager.config.lmcache_instance_id == "psrl_instance_3"
    assert manager.current_version == 7


def test_peer_registry_replaces_worker_urls_and_keeps_peer_entries():
    manager = KVCacheManager(LMCacheConfig(enable=False))

    manager.set_peer_registry({"psrl_instance_1": ["peer-0"]}, ["worker-0"])
    manager.set_peer_registry({"psrl_instance_2": ["peer-1"]})

    assert manager.peer_registry == {
        "psrl_instance_1": ["peer-0"],
        "psrl_instance_2": ["peer-1"],
    }
    assert manager._worker_zmq_urls == ["worker-0"]
