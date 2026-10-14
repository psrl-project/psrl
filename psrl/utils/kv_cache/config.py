from dataclasses import dataclass, field

from psrl.utils.kv_cache.types import KVCacheBackend


@dataclass
class LMCacheConfig:
    """
    Configure LMCache KV cache offloading for PSRL.

    vLLM divides `offload_size_gb` across KV ranks before passing each rank's
    budget to LMCache.
    """

    # --- Core ---

    # Whether to enable LMCache KV cache offloading.
    enable: bool = False

    # Offloading backend: "cpu", "disk", or "remote".
    backend: str = "cpu"

    # Total offload budget in GiB. vLLM divides it across KV ranks.
    offload_size_gb: float = 10.0

    # LMCache token chunk size (in tokens) for hash-based KV indexing.
    # Must be a multiple of the model's block size.
    chunk_size: int = 256

    # Path to a full LMCache YAML config file.
    # When set, this overrides all individual fields below (except chunk_size).
    config_file: str | None = None

    # Whether to clear the LMCache KV cache on model weight updates from PS.
    clear_on_weight_update: bool = True

    # Retain version tagged entries for natural LRU eviction.
    # This requires `clear_on_weight_update=False`.
    multi_version_kv: bool = False

    # --- CPU backend ---
    # GiB of CPU memory to reserve from KV offloading when other processes compete for pinned memory.
    reserve_local_cpu_size: float = 0.0

    # --- Disk backend ---

    # Local filesystem path for disk-backed KV storage (required when backend="disk").
    disk_path: str | None = None

    # Maximum disk usage in GiB when using the disk backend.
    max_disk_size_gb: float = 50.0

    # --- Remote backend (Phase 2) ---

    # Remote LMCache server URL (e.g., "redis://host:6379") when backend="remote".
    remote_url: str | None = None

    # --- P2P transfer (for `kv_transfer_trajectory`) ---

    # Enable LMCache P2P backend + Controller for cross-instance KV transfer.
    enable_p2p: bool = False

    # Unique LMCache identifier assigned by `KVCacheManager.set_instance_id`.
    lmcache_instance_id: str = "psrl_instance_0"

    # Transport channel for P2P transfer: "nixl" (RDMA/IB) or "tcp".
    p2p_transfer_channel: str = "nixl"

    # Base port for the LMCache Controller subprocess.
    # `find_available_port()` picks the actual port at runtime.
    controller_base_port: int = 9000

    # Controller startup timeout in seconds for contended manager nodes.
    controller_health_timeout_s: int = 300

    # Host where the LMCache Controller runs (defaults to ps_manager_ip).
    # Set by PSRL before engine init so LMCache workers know where to connect.
    controller_host: str = ""

    # ZMQ ports for Controller ↔ LMCache worker communication.
    controller_pull_port: int = 8300
    controller_reply_port: int = 8400

    # Worker address advertised for P2P pushes during rollout initialization.
    worker_host: str = ""

    # KV workers per instance, assigned from tensor parallelism during rollout initialization.
    num_kv_workers: int = 1

    # Runtime-allocated ports (set by vllm_rollout.py via PortScanner to avoid conflicts).
    # Each list has num_kv_workers entries, one per TP rank.
    allocated_worker_ports: list = field(default_factory=list)
    allocated_p2p_init_ports: list = field(default_factory=list)
    allocated_p2p_lookup_ports: list = field(default_factory=list)

    # --- GPU pin budget ---

    # Maximum simultaneously pinned GPU KV blocks. Zero disables the limit.
    gpu_pin_block_budget: int = 0

    # --- Cache behaviour ---

    # Cache decode outputs when repeated prefixes justify the extra memory.
    save_decode_cache: bool = False

    # Whether to persist a chunk even when it is not yet fully filled.
    # Useful when prompts are shorter than `chunk_size`.
    save_unfull_chunk: bool = False

    # Eviction policy for the local CPU/disk cache.
    # Supported values: "LRU" (least-recently-used) or "FIFO".
    cache_policy: str = "LRU"

    # Retrieve cache entries asynchronously while prefill runs.
    enable_async_loading: bool = False

    # Publish store events so routing can score the off GPU cache tier.
    enable_kv_events: bool = False

    def get_backend_enum(self) -> KVCacheBackend:
        """
        Convert the string backend to `KVCacheBackend` enum.
        """
        return KVCacheBackend(self.backend)

    def to_engine_kwargs(self) -> dict:
        """
        Translate this config into vLLM engine kwargs for `AsyncEngineArgs`.

        Returns:
            dict: Key-value pairs to merge into `llm_kwargs`.
        """
        if not self.enable:
            return {}

        return {
            "kv_offloading_backend": "lmcache",
            "kv_offloading_size": self.offload_size_gb,
            # LMCacheConnectorV1 does not support vLLM's Hybrid Memory Allocator (HMA).
            # HMA must be disabled whenever LMCache is used as the KV offloading backend.
            "disable_hybrid_kv_cache_manager": True,
        }

    def to_env_vars(self) -> dict[str, str]:
        """
        Translate this config into environment variables for LMCache.

        vLLM supplies the per rank cache size separately through its higher priority
        extra config.

        Returns:
            dict[str, str]: Env var name to value pairs.
        """
        if not self.enable:
            return {}

        env_vars: dict[str, str] = {}

        # Activate LMCache v1 experimental config system.
        env_vars["LMCACHE_USE_EXPERIMENTAL"] = "True"

        # Chunk size (env var is read before vLLM extra config kicks in).
        env_vars["LMCACHE_CHUNK_SIZE"] = str(self.chunk_size)

        # The full config file takes precedence over other environment variables.
        if self.config_file:
            env_vars["LMCACHE_CONFIG_FILE"] = self.config_file

        # CPU backend.
        if self.backend == "cpu":
            env_vars["LMCACHE_LOCAL_CPU"] = "True"
            # reserve_local_cpu_size: keep some CPU memory free for other processes.
            if self.reserve_local_cpu_size > 0.0:
                env_vars["LMCACHE_RESERVE_LOCAL_CPU_SIZE"] = str(self.reserve_local_cpu_size)

        # Disk backend.
        if self.backend == "disk" and self.disk_path:
            env_vars["LMCACHE_LOCAL_DISK"] = self.disk_path
            env_vars["LMCACHE_MAX_LOCAL_DISK_SIZE"] = str(self.max_disk_size_gb)

        # Remote backend (Phase 2).
        if self.backend == "remote" and self.remote_url:
            env_vars["LMCACHE_REMOTE_URL"] = self.remote_url

        # Cache behaviour flags.
        if self.save_decode_cache:
            env_vars["LMCACHE_SAVE_DECODE_CACHE"] = "True"
        if self.save_unfull_chunk:
            env_vars["LMCACHE_SAVE_UNFULL_CHUNK"] = "True"
        if self.cache_policy != "LRU":
            env_vars["LMCACHE_CACHE_POLICY"] = self.cache_policy
        if self.enable_async_loading:
            env_vars["LMCACHE_ENABLE_ASYNC_LOADING"] = "True"
        if self.enable_kv_events:
            env_vars["LMCACHE_ENABLE_KV_EVENTS"] = "True"

        # Register P2P workers with the shared controller for explicit transfers.
        if self.enable_p2p:
            env_vars["LMCACHE_ENABLE_CONTROLLER"] = "True"
            env_vars["LMCACHE_LMCACHE_INSTANCE_ID"] = self.lmcache_instance_id
            # Restrict retrieval to explicit moves because automatic P2P reads race
            # with concurrent moves.
            env_vars["LMCACHE_RETRIEVE_LOCATIONS"] = "LocalCPUBackend"
            # Controller ZMQ endpoints (host:port).
            controller_host = self.controller_host or "127.0.0.1"
            env_vars["LMCACHE_CONTROLLER_PULL_URL"] = f"{controller_host}:{self.controller_pull_port}"
            env_vars["LMCACHE_CONTROLLER_REPLY_URL"] = f"{controller_host}:{self.controller_reply_port}"
            if self.p2p_transfer_channel:
                env_vars["LMCACHE_TRANSFER_CHANNEL"] = self.p2p_transfer_channel

            # Per-worker ports (allocated by PortScanner in vllm_rollout.py).
            assert len(self.allocated_worker_ports) == self.num_kv_workers, (
                f"allocated_worker_ports has {len(self.allocated_worker_ports)} entries "
                f"but num_kv_workers={self.num_kv_workers}. "
                "Call vllm_rollout.py port allocation before apply_env_vars()."
            )
            env_vars["LMCACHE_LMCACHE_WORKER_PORTS"] = ",".join(str(p) for p in self.allocated_worker_ports)

            # P2P backend ports.
            env_vars["LMCACHE_ENABLE_P2P"] = "True"
            env_vars["LMCACHE_P2P_HOST"] = self.worker_host
            assert len(self.allocated_p2p_init_ports) == self.num_kv_workers, (
                f"allocated_p2p_init_ports has {len(self.allocated_p2p_init_ports)} entries "
                f"but num_kv_workers={self.num_kv_workers}."
            )
            env_vars["LMCACHE_P2P_INIT_PORTS"] = ",".join(str(p) for p in self.allocated_p2p_init_ports)
            assert len(self.allocated_p2p_lookup_ports) == self.num_kv_workers, (
                f"allocated_p2p_lookup_ports has {len(self.allocated_p2p_lookup_ports)} entries "
                f"but num_kv_workers={self.num_kv_workers}."
            )
            env_vars["LMCACHE_P2P_LOOKUP_PORTS"] = ",".join(str(p) for p in self.allocated_p2p_lookup_ports)

        return env_vars
