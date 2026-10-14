from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config


class vLLMWorkerExtension:
    def get_total_kv_cache_tokens(self) -> int:
        """Total KV-cache token capacity of this instance, post-allocation.

        Uses vLLM's own `max_concurrency * max_model_len` formula (the same one
        behind the "GPU KV cache size: N tokens" startup log line), which is
        exact for hybrid/sliding-window layouts and already accounts for
        DCP/PCP sharding, unlike a naive `num_gpu_blocks * block_size`.
        """
        assert hasattr(self, "vllm_config"), "vllm_config must be set"
        kv_cache_config = getattr(self.model_runner, "kv_cache_config", None)
        if kv_cache_config is None or not kv_cache_config.kv_cache_groups:
            return 0
        max_concurrency = get_max_concurrency_for_kv_cache_config(self.vllm_config, kv_cache_config)
        return int(max_concurrency * self.vllm_config.model_config.max_model_len)
