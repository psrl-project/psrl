import os
from unittest.mock import patch

import pytest
from psrl.utils.kv_cache.config import LMCacheConfig
from psrl.utils.kv_cache.manager import KVCacheManager
from psrl.utils.kv_cache.types import KVCacheBackend

pytestmark = pytest.mark.cpu_test


class TestKVCacheBackend:
    def test_cpu_backend(self):
        assert KVCacheBackend("cpu") == KVCacheBackend.CPU

    def test_disk_backend(self):
        assert KVCacheBackend("disk") == KVCacheBackend.DISK

    def test_remote_backend(self):
        assert KVCacheBackend("remote") == KVCacheBackend.REMOTE

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            KVCacheBackend("invalid")


class TestLMCacheConfig:
    def test_disabled_by_default(self):
        config = LMCacheConfig()
        assert config.enable is False
        assert config.to_engine_kwargs() == {}
        assert config.to_env_vars() == {}

    # --- to_engine_kwargs ---

    def test_cpu_engine_kwargs(self):
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=20.0)
        kwargs = config.to_engine_kwargs()
        assert kwargs["kv_offloading_backend"] == "lmcache"
        assert kwargs["kv_offloading_size"] == 20.0
        # HMA must always be disabled when LMCache is used.
        assert kwargs["disable_hybrid_kv_cache_manager"] is True

    def test_engine_kwargs_always_disables_hma(self):
        """disable_hybrid_kv_cache_manager must be True for every enabled config."""
        for backend in ("cpu", "disk", "remote"):
            config = LMCacheConfig(enable=True, backend=backend)
            assert config.to_engine_kwargs()["disable_hybrid_kv_cache_manager"] is True

    # --- common to_env_vars ---

    def test_experimental_flag_always_set(self):
        config = LMCacheConfig(enable=True, backend="cpu")
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_USE_EXPERIMENTAL"] == "True"

    def test_chunk_size_env_var(self):
        config = LMCacheConfig(enable=True, backend="cpu", chunk_size=128)
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_CHUNK_SIZE"] == "128"

    def test_config_file_override(self):
        config = LMCacheConfig(enable=True, config_file="/path/to/lmcache.yaml")
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_CONFIG_FILE"] == "/path/to/lmcache.yaml"

    # --- CPU backend to_env_vars ---

    def test_cpu_env_vars_basic(self):
        config = LMCacheConfig(enable=True, backend="cpu")
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_LOCAL_CPU"] == "True"

    def test_cpu_reserve_size_zero_not_emitted(self):
        """reserve_local_cpu_size=0 should not add the env var."""
        config = LMCacheConfig(enable=True, backend="cpu", reserve_local_cpu_size=0.0)
        env_vars = config.to_env_vars()
        assert "LMCACHE_RESERVE_LOCAL_CPU_SIZE" not in env_vars

    def test_cpu_reserve_size_positive(self):
        config = LMCacheConfig(enable=True, backend="cpu", reserve_local_cpu_size=2.5)
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_RESERVE_LOCAL_CPU_SIZE"] == "2.5"

    def test_cpu_no_max_local_cpu_size_env_var(self):
        """Ensure vLLM computes each rank's max_local_cpu_size instead of reading an environment variable."""
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=20.0)
        env_vars = config.to_env_vars()
        assert "LMCACHE_MAX_LOCAL_CPU_SIZE" not in env_vars

    # --- disk backend to_env_vars ---

    def test_disk_env_vars(self):
        config = LMCacheConfig(
            enable=True,
            backend="disk",
            offload_size_gb=10.0,
            disk_path="/tmp/lmcache_disk",
            max_disk_size_gb=50.0,
        )
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_LOCAL_DISK"] == "/tmp/lmcache_disk"
        assert env_vars["LMCACHE_MAX_LOCAL_DISK_SIZE"] == "50.0"

    # --- remote backend to_env_vars ---

    def test_remote_env_vars(self):
        config = LMCacheConfig(enable=True, backend="remote", remote_url="redis://host:6379")
        env_vars = config.to_env_vars()
        assert env_vars["LMCACHE_REMOTE_URL"] == "redis://host:6379"

    # --- cache behavior flags ---

    def test_save_decode_cache_default_not_emitted(self):
        config = LMCacheConfig(enable=True, save_decode_cache=False)
        assert "LMCACHE_SAVE_DECODE_CACHE" not in config.to_env_vars()

    def test_save_decode_cache_enabled(self):
        config = LMCacheConfig(enable=True, save_decode_cache=True)
        assert config.to_env_vars()["LMCACHE_SAVE_DECODE_CACHE"] == "True"

    def test_save_unfull_chunk_default_not_emitted(self):
        config = LMCacheConfig(enable=True, save_unfull_chunk=False)
        assert "LMCACHE_SAVE_UNFULL_CHUNK" not in config.to_env_vars()

    def test_save_unfull_chunk_enabled(self):
        config = LMCacheConfig(enable=True, save_unfull_chunk=True)
        assert config.to_env_vars()["LMCACHE_SAVE_UNFULL_CHUNK"] == "True"

    def test_cache_policy_default_not_emitted(self):
        config = LMCacheConfig(enable=True, cache_policy="LRU")
        assert "LMCACHE_CACHE_POLICY" not in config.to_env_vars()

    def test_cache_policy_fifo(self):
        config = LMCacheConfig(enable=True, cache_policy="FIFO")
        assert config.to_env_vars()["LMCACHE_CACHE_POLICY"] == "FIFO"

    def test_async_loading_default_not_emitted(self):
        config = LMCacheConfig(enable=True, enable_async_loading=False)
        assert "LMCACHE_ENABLE_ASYNC_LOADING" not in config.to_env_vars()

    def test_async_loading_enabled(self):
        config = LMCacheConfig(enable=True, enable_async_loading=True)
        assert config.to_env_vars()["LMCACHE_ENABLE_ASYNC_LOADING"] == "True"

    # --- backend enum ---

    def test_get_backend_enum(self):
        config = LMCacheConfig(enable=True, backend="cpu")
        assert config.get_backend_enum() == KVCacheBackend.CPU

    def test_invalid_backend(self):
        config = LMCacheConfig(enable=True, backend="invalid")
        with pytest.raises(ValueError):
            config.get_backend_enum()


class TestKVCacheManager:
    def test_disabled_manager(self):
        config = LMCacheConfig(enable=False)
        manager = KVCacheManager(config)
        assert manager.enabled is False
        assert manager.config.enable is False
        assert manager.get_engine_kwargs() == {}

    def test_enabled_manager(self):
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=15.0)
        manager = KVCacheManager(config)
        assert manager.enabled is True
        status = manager.config
        assert status.enable is True
        assert status.get_backend_enum() == KVCacheBackend.CPU
        assert status.offload_size_gb == 15.0

    def test_get_engine_kwargs(self):
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=20.0)
        manager = KVCacheManager(config)
        kwargs = manager.get_engine_kwargs()
        assert kwargs["kv_offloading_backend"] == "lmcache"
        assert kwargs["kv_offloading_size"] == 20.0
        assert kwargs["disable_hybrid_kv_cache_manager"] is True

    def test_apply_env_vars(self):
        config = LMCacheConfig(enable=True, backend="cpu", chunk_size=512)
        manager = KVCacheManager(config)
        with patch.dict(os.environ, {}, clear=False):
            manager.apply_env_vars()
            assert os.environ["LMCACHE_CHUNK_SIZE"] == "512"
            assert os.environ["LMCACHE_USE_EXPERIMENTAL"] == "True"
            assert os.environ["LMCACHE_LOCAL_CPU"] == "True"

    def test_apply_env_vars_disabled(self):
        config = LMCacheConfig(enable=False)
        manager = KVCacheManager(config)
        original_env = dict(os.environ)
        manager.apply_env_vars()
        for key in os.environ:
            if key.startswith("LMCACHE_") and key not in original_env:
                pytest.fail(f"Unexpected env var set: {key}")

    def test_apply_env_vars_no_max_local_cpu_size(self):
        """Ensure max_local_cpu_size is never set by apply_env_vars."""
        config = LMCacheConfig(enable=True, backend="cpu", offload_size_gb=20.0)
        manager = KVCacheManager(config)
        with patch.dict(os.environ, {}, clear=False):
            manager.apply_env_vars()
            assert "LMCACHE_MAX_LOCAL_CPU_SIZE" not in os.environ

    def test_clear_on_weight_update_flag(self):
        config = LMCacheConfig(enable=True, clear_on_weight_update=True)
        manager = KVCacheManager(config)
        assert manager.should_clear_on_weight_update is True

        config2 = LMCacheConfig(enable=True, clear_on_weight_update=False)
        manager2 = KVCacheManager(config2)
        assert manager2.should_clear_on_weight_update is False

    def test_reserve_cpu_size_applied(self):
        config = LMCacheConfig(enable=True, backend="cpu", reserve_local_cpu_size=3.0)
        manager = KVCacheManager(config)
        with patch.dict(os.environ, {}, clear=False):
            manager.apply_env_vars()
            assert os.environ["LMCACHE_RESERVE_LOCAL_CPU_SIZE"] == "3.0"

    def test_cache_behaviour_flags_applied(self):
        config = LMCacheConfig(
            enable=True,
            save_decode_cache=True,
            save_unfull_chunk=True,
            cache_policy="FIFO",
            enable_async_loading=True,
        )
        manager = KVCacheManager(config)
        with patch.dict(os.environ, {}, clear=False):
            manager.apply_env_vars()
            assert os.environ["LMCACHE_SAVE_DECODE_CACHE"] == "True"
            assert os.environ["LMCACHE_SAVE_UNFULL_CHUNK"] == "True"
            assert os.environ["LMCACHE_CACHE_POLICY"] == "FIFO"
            assert os.environ["LMCACHE_ENABLE_ASYNC_LOADING"] == "True"
