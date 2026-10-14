import asyncio
import logging
import uuid
from collections import deque

import msgspec

from psrl.utils.kv_cache.config import LMCacheConfig

psrl_logger = logging.getLogger(__file__)


class KVCacheManager:
    """
    Manage PSRL KV cache operations across vLLM and LMCache.

    Callers provide trajectory tokens, while this manager owns pin budgets and
    peer metadata.
    """

    def __init__(self, config: LMCacheConfig) -> None:
        """
        Initialize the KV cache manager.

        Args:
            config (LMCacheConfig): The resolved LMCache configuration.
        """
        self.config = config
        self._gpu_pin_budget: int = config.gpu_pin_block_budget
        self._pinned_gpu_blocks: int = 0
        self._gpu_pinned_order: deque[list[int]] = deque()

        self.current_version: int = 0

        self._inference_engine = None
        # The shared Controller is still used for LMCache worker registration and
        # peer discovery. Data movement itself uses direct worker ZMQ messages.
        self._controller_url: str | None = None

        # Rank ordered peer metadata lets each local shard target the matching remote rank.
        self.peer_registry: dict[str, list[str]] = {}
        # Rank ordered worker URLs support direct per shard transfers.
        self._worker_zmq_urls: list[str] = []
        # Each rank owns a lazily initialized direct transfer socket.
        self._direct_zmq_sockets: dict[int, object] = {}
        self._direct_zmq_context = None
        # Per rank locks enforce strict REQ send and receive alternation.
        self._direct_zmq_locks: dict[int, asyncio.Lock] = {}

        self._log_init_status()

    # --- Initialization helpers ---

    def _log_init_status(self) -> None:
        """
        Log LMCache initialization status and parameters at INFO level.
        """
        if not self.config.enable:
            psrl_logger.info("[LMCache] KV cache offloading is DISABLED.")
            return

        psrl_logger.info("[LMCache] KV cache offloading is ENABLED with the following parameters:")
        psrl_logger.info(f"backend = {self.config.backend!r}")
        psrl_logger.info(f"offload_size_gb = {self.config.offload_size_gb}")
        psrl_logger.info(f"chunk_size = {self.config.chunk_size}")
        psrl_logger.info(f"cache_policy = {self.config.cache_policy!r}")
        psrl_logger.info(f"save_decode_cache = {self.config.save_decode_cache}")
        psrl_logger.info(f"save_unfull_chunk = {self.config.save_unfull_chunk}")
        psrl_logger.info(f"enable_async_loading = {self.config.enable_async_loading}")
        psrl_logger.info(f"clear_on_weight_update = {self.config.clear_on_weight_update}")
        psrl_logger.info(f"gpu_pin_block_budget = {self.config.gpu_pin_block_budget}")
        if self.config.enable_p2p:
            psrl_logger.info(f"enable_p2p = {self.config.enable_p2p}")
            psrl_logger.info(f"lmcache_instance_id = {self.config.lmcache_instance_id!r}")
        if self.config.config_file:
            psrl_logger.info(f"config_file = {self.config.config_file!r}")
        self._verify_lmcache_importable()

    def _verify_lmcache_importable(self) -> None:
        """
        Verify that the lmcache package is importable and log its version.
        """
        try:
            import lmcache  # type: ignore[import-untyped]

            version = getattr(lmcache, "__version__", "unknown")
            psrl_logger.info(f"[LMCache] lmcache package is importable, version={version!r}.")
        except ImportError:
            psrl_logger.error(
                "[LMCache] lmcache package is NOT importable! "
                "KV cache offloading will NOT work. "
                "Run `bash scripts/install_lmcache.sh` to install it."
            )

    # --- Engine attachment ---

    def attach_engine(self, inference_engine) -> None:
        """
        Attach the vLLM AsyncLLM engine after it has been initialised.

        Must be called before any async KV cache operation.

        Args:
            inference_engine: The `AsyncLLM` (or compatible) engine object whose
                `collective_rpc` method dispatches to vLLM worker processes.
        """
        self._inference_engine = inference_engine
        psrl_logger.info("[LMCache] KVCacheManager: Engine attached.")

    def set_instance_id(self, instance_id: int) -> None:
        """
        Set the per-instance LMCache identifier.

        Must be called once by the vLLM replica setup code after construction,
        passing the numeric instance id assigned to this worker group. Sets
        `lmcache_instance_id` to `"psrl_instance_{instance_id}"`, which the
        LMCache Controller uses to identify KV transfer sources and destinations.

        Args:
            instance_id (int): Numeric identifier of this rollout instance.
        """
        self.config.lmcache_instance_id = f"psrl_instance_{instance_id}"
        psrl_logger.info(f"[LMCache] Instance ID set to {self.config.lmcache_instance_id!r}.")

    def set_controller_url(self, controller_url: str) -> None:
        """
        Set the shared LMCache Controller URL.

        Called by the vLLM replica setup code after
        `RolloutCoordinator.init_lmcache_p2p()` broadcasts the URL of the
        single shared Controller subprocess to all vLLM instances.

        Args:
            controller_url (str): Base URL of the shared Controller, e.g.
                `"http://10.0.0.1:9042"`.
        """
        self._controller_url = controller_url
        psrl_logger.info(f"[LMCache] Controller URL set to {self._controller_url!r}.")

    def set_peer_registry(
        self,
        registry: dict[str, list[str]],
        worker_zmq_urls: list[str] | None = None,
    ) -> None:
        """
        Set the peer registry and local worker ZMQ URLs for direct transfer bypass.

        Maps each LMCache instance_id to its per-rank list of peer_init_url (NIXL
        endpoints), indexed by global rank. When populated along with
        `worker_zmq_urls`, `transfer_direct()` sends one MoveWorkerMsg per local
        rank directly to that rank's LMCacheWorker via ZMQ, bypassing the Controller
        HTTP round-trip. Because KV is sharded per rank (TP heads, PP layers), each
        local rank targets the destination's same-rank endpoint.

        Called by `PSRL_vLLMHttpServer.kv_set_peer_registry()` after
        `RolloutCoordinator._broadcast_peer_registry()` completes.

        Args:
            registry (dict[str, list[str]]): Maps lmcache_instance_id (e.g.
                "psrl_instance_0") to a rank-sorted list of peer_init_url
                (e.g. ["10.0.0.1:18200", "10.0.0.1:18201"]).
            worker_zmq_urls (list[str] | None): Rank-sorted ZMQ REP URLs of this
                replica's local LMCacheWorkers (e.g. ["10.0.0.1:18100", ...]).
                Supplied only by the authoritative broadcast/init path. When None
                (e.g. the per-request servicer seed), this replica's own URLs and
                live sockets are left untouched.
        """
        # Merge full broadcasts and incremental seeds without clobbering peers.
        self.peer_registry.update(registry)
        if worker_zmq_urls is not None:
            # Reset sockets only when the authoritative worker URLs change.
            self._worker_zmq_urls = worker_zmq_urls
            self._reset_all_zmq_sockets()
        psrl_logger.info(
            f"[LMCache] Peer registry updated ({len(registry)} entries this call, "
            f"{len(self.peer_registry)} total), worker_zmq_urls={self._worker_zmq_urls!r}."
        )

    def set_current_version(self, version: int) -> None:
        """
        Set the current model version used to tag new KV cache entries.

        Called locally by the gen server after this replica completes a weight
        pull and reads back its actual model version. Affects all subsequent KV
        store and P2P transfer operations on this instance.

        Args:
            version (int): The new model version number (monotonically increasing).
        """
        self.current_version = version
        psrl_logger.info(f"[LMCache] current_version set to {version}.")

    @property
    def is_attached(self) -> bool:
        """Whether the inference engine has been attached via `attach_engine`."""
        return self._inference_engine is not None

    # --- Legacy Phase 1 helpers (still used for engine init) ---

    @property
    def enabled(self) -> bool:
        """Whether LMCache offloading is enabled."""
        return self.config.enable

    @property
    def should_clear_on_weight_update(self) -> bool:
        """Whether to clear the LMCache KV cache on model weight updates from PS."""
        return self.config.enable and self.config.clear_on_weight_update

    def apply_env_vars(self) -> None:
        """
        Set LMCache environment variables before vLLM engine initialization.

        Must be called before `AsyncEngineArgs` / `AsyncLLM` creation.
        """
        import os

        env_vars = self.config.to_env_vars()
        if not env_vars:
            psrl_logger.info("[LMCache] No environment variables to set (LMCache disabled).")
            return
        psrl_logger.info("[LMCache] Setting environment variables for LMCache...")
        for key, value in env_vars.items():
            os.environ[key] = value
            psrl_logger.info(f"  {key}={value!r}")

    def get_engine_kwargs(self) -> dict:
        """
        Get vLLM engine kwargs for LMCache integration.

        Returns:
            dict: Key-value pairs to merge into vLLM engine arguments.
        """
        kwargs = self.config.to_engine_kwargs()
        if kwargs:
            psrl_logger.info(f"[LMCache] Injecting engine kwargs into vLLM: {kwargs}.")
        else:
            psrl_logger.info("[LMCache] No engine kwargs to inject (LMCache disabled).")
        return kwargs

    # --- Private helpers ---

    def _assert_engine(self) -> None:
        assert self._inference_engine is not None, (
            "KVCacheManager inference engine is not attached. Call attach_engine() after the rollout is initialised."
        )

    async def _rpc(self, method: str, args: tuple) -> object:
        """
        Call a `lmcache_*` method on all vLLM workers via `collective_rpc`.

        `collective_rpc` returns a list with one result per TP rank. We take
        the first element (rank-0), which is canonical for aggregate results.

        Args:
            method (str): The `vllm_extension.py` method name to call.
            args (tuple): Positional arguments forwarded to the worker method.

        Returns:
            object: The result from rank-0 worker.
        """
        results = await self._inference_engine.collective_rpc(method, args=args)
        # `collective_rpc` returns a list, so use the rank zero value.
        return results[0] if isinstance(results, list) else results

    async def _utility(self, method: str, *args) -> object:
        """
        Call a `psrl_*` method on the vLLM `EngineCore` via `call_utility_async`.

        Unlike `_rpc` (which dispatches to Worker processes via `collective_rpc`),
        this routes directly to the `EngineCore` object in the engine-core process.
        `EngineCore.scheduler` is the `RolloutScheduler` instance, so any method
        defined on `RolloutScheduler` (and therefore on `EngineCore` via attribute
        lookup) is reachable here.

        Use this for GPU block-pool operations (`psrl_pin_gpu`,
        `psrl_pin_gpu`, `psrl_unpin_gpu`) which must run in the same process
        as `block_pool` in the EngineCore process. This is safe for TP>1 because
        the state is never copied across process boundaries.

        Args:
            method (str): The `RolloutScheduler` method name to call on EngineCore.
            *args: Positional arguments forwarded to the method.

        Returns:
            object: The return value of the called method.
        """
        return await self._inference_engine.engine_core.call_utility_async(method, *args)

    # --- Public KV cache operations ---

    async def pin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Pin the cached prefix of a trajectory to prevent LRU eviction.

        Supported targets: `"gpu"` (vLLM block pool) and `"backend"` (LMCache).
        GPU pinning is subject to `gpu_pin_block_budget`. If the budget is
        exceeded, the oldest-pinned entry is unpinned first (PSRL-side LRU).

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all requested pin operations succeeded.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid pin targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        ok = True
        if "gpu" in targets:
            ok = ok and await self._pin_gpu(tokens)
        if "backend" in targets:
            pinned: int = await self._rpc("lmcache_pin_backend", (tokens,))
            ok = ok and pinned >= 0
        return ok

    async def unpin(self, tokens: list[int], targets: list[str]) -> bool:
        """
        Unpin the cached prefix of a trajectory, allowing LRU eviction.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            targets (list[str]): Subset of `["gpu", "backend"]`.

        Returns:
            bool: True if all unpin operations completed without error.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        assert targets, "targets must be a non-empty list."
        assert all(t in ("gpu", "backend") for t in targets), (
            f"Invalid unpin targets: {targets!r}. Must be a subset of ['gpu', 'backend']."
        )
        ok = True
        if "gpu" in targets:
            ok = ok and await self._unpin_gpu(tokens)
        if "backend" in targets:
            await self._rpc("lmcache_unpin_backend", (tokens,))
        return ok

    async def transfer_direct(
        self,
        tokens: list[int],
        src: tuple[str, str],
        dst: tuple[str, str],
        copy: bool = False,
        dst_model_version: int = -1,
    ) -> bool:
        """
        Transfer KV cache by sending MoveWorkerMsg directly to local LMCacheWorker.

        Bypasses the centralized Controller HTTP endpoint by constructing the
        same MoveWorkerMsg that the Controller's executor would send, and
        dispatching it directly to the local LMCacheWorker's ZMQ socket.

        This eliminates:
        - HTTP JSON serialization of large token lists
        - Controller single-process GIL bottleneck under burst
        - Controller event loop contention

        Each replica's manager sends to its OWN per-rank Workers, so N instances
        under burst have zero cross-instance contention. KV is sharded per rank, so
        this fans out one MoveWorkerMsg per local rank, each moving that rank's shard
        to the destination's same-rank endpoint. Falls back to re-prefill on failure.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.
            src (tuple[str, str]): Source `(lmcache_instance_id, backend_location)`.
            dst (tuple[str, str]): Destination `(lmcache_instance_id, backend_location)`.
            copy (bool): If True, keep the data at `src` as well.
            dst_model_version (int): Model version to look up on the source when
                multi_version_kv is enabled. -1 means version-agnostic (no tag).

        Returns:
            bool: True if every local rank moved >0 tokens.
        """
        self._assert_engine()
        assert tokens, "tokens must be a non-empty list."
        if not self.config.enable_p2p:
            psrl_logger.warning("[LMCache] transfer_direct() called but enable_p2p is False.")
            return False

        # Missing prerequisites are errors because no safe fallback exists.
        assert self.peer_registry, (
            "[LMCache] transfer_direct() called but peer_registry is empty. Call set_peer_registry() after P2P init."
        )
        assert self._worker_zmq_urls, (
            "[LMCache] transfer_direct() called but worker_zmq_urls is not set. "
            "Call set_peer_registry(registry, worker_zmq_urls) after P2P init."
        )

        dst_instance_id = dst[0]
        dst_urls = self.peer_registry.get(dst_instance_id)
        # Rank pairing requires homogeneous layouts because LMCache cannot reshard KV data.
        num_ranks = len(self._worker_zmq_urls)
        if not dst_urls or len(dst_urls) != num_ranks:
            psrl_logger.warning(
                f"[LMCache] Destination {dst_instance_id!r} has "
                f"{0 if not dst_urls else len(dst_urls)} ranks but this replica has "
                f"{num_ranks}. The TP/PP layout is likely heterogeneous. Skipping direct "
                "transfer, destination will re-prefill."
            )
            return False

        # Fan out one MoveWorkerMsg per local rank, each to its own LMCacheWorker.
        request_configs: dict | None = (
            {"lmcache.tag.model_version": str(dst_model_version)} if dst_model_version >= 0 else None
        )

        async def _move_rank(rank: int) -> int:
            dst_peer_init_url = dst_urls[rank]
            if not dst_peer_init_url:
                psrl_logger.warning(
                    f"[LMCache] No same-rank peer_init_url for {dst_instance_id!r} "
                    f"rank {rank}. Skipping that rank's shard."
                )
                return 0
            return await self._send_move_worker_msg(
                rank=rank,
                tokens=tokens,
                old_position=src[1],  # backend location string
                new_position=(dst_peer_init_url, dst[1]),
                copy=copy,
                request_configs=request_configs,
            )

        per_rank_tokens = await asyncio.gather(*[_move_rank(r) for r in range(num_ranks)])
        all_moved = all(n > 0 for n in per_rank_tokens)
        if all_moved:
            psrl_logger.debug(
                f"[LMCache] Direct transfer succeeded on all {num_ranks} ranks "
                f"from {src!r} to {dst!r} (per-rank tokens: {per_rank_tokens!r})."
            )
        else:
            psrl_logger.info(
                f"[LMCache] Direct transfer moved 0 tokens on some rank for "
                f"{src!r} → {dst!r} (per-rank: {per_rank_tokens!r}). Source may have "
                "evicted or have a layout mismatch. The destination will re-prefill."
            )
        return all_moved

    async def _send_move_worker_msg(
        self,
        rank: int,
        tokens: list[int],
        old_position: str,
        new_position: tuple[str, str],
        copy: bool,
        request_configs: dict | None = None,
    ) -> int:
        """
        Construct and send MoveWorkerMsg directly to one local LMCacheWorker via ZMQ.

        Replicates what LMCacheClusterExecutor.move() does (executor.py:281-350)
        but without the Controller intermediary. Uses async ZMQ for non-blocking I/O.

        Args:
            rank: Local rank whose LMCacheWorker (and dedicated ZMQ socket) to use.
            tokens: Full token sequence.
            old_position: Source backend location (e.g. "LocalCPUBackend").
            new_position: Tuple of (dst_peer_init_url, dst_backend_location).
            copy: Whether to keep data at source.
            request_configs: Optional LMCache request config tags (e.g. model version
                tag). Passed through to MoveWorkerMsg so the worker's lookup() uses
                the matching version-specific key hash.

        Returns:
            int: Number of tokens transferred.
        """
        from lmcache.v1.cache_controller.message import (
            MoveWorkerMsg,
            Msg,
        )

        worker_event_id = f"DirectMove_{uuid.uuid4().hex[:8]}"
        msg = MoveWorkerMsg(
            worker_event_id=worker_event_id,
            old_position=old_position,
            new_position=new_position,
            tokens=tokens,
            copy=copy,
            request_configs=request_configs,
        )

        serialized_msg = msgspec.msgpack.encode(msg)
        lock = self._direct_zmq_locks.setdefault(rank, asyncio.Lock())
        # The per rank lock enforces strict ZMQ REQ send and receive alternation.
        async with lock:
            socket = self._get_or_create_zmq_socket(rank)
            try:
                await socket.send(serialized_msg)
                serialized_resp = await socket.recv()
            except Exception:
                # A failed REQ exchange leaves the socket unusable, so rebuild it
                # before the next call.
                self._reset_zmq_socket(rank)
                raise
        resp = msgspec.msgpack.decode(serialized_resp, type=Msg)

        if hasattr(resp, "num_tokens"):
            return resp.num_tokens
        else:
            psrl_logger.warning(f"[LMCache] Unexpected response type from Worker: {type(resp).__name__}")
            return 0

    def _get_or_create_zmq_socket(self, rank: int):
        """
        Get or lazily create the async ZMQ REQ socket for a given local rank.

        Args:
            rank (int): Local rank whose LMCacheWorker REP socket to connect to.

        Returns:
            zmq.asyncio.Socket: Connected ZMQ REQ socket for `rank`.
        """
        socket = self._direct_zmq_sockets.get(rank)
        if socket is None:
            import zmq
            import zmq.asyncio

            assert 0 <= rank < len(self._worker_zmq_urls), (
                f"[LMCache] rank {rank} out of range for {len(self._worker_zmq_urls)} worker zmq urls."
            )
            worker_zmq_url = self._worker_zmq_urls[rank]
            if self._direct_zmq_context is None:
                self._direct_zmq_context = zmq.asyncio.Context()
            socket = self._direct_zmq_context.socket(zmq.REQ)
            socket.connect(f"tcp://{worker_zmq_url}")
            # Set send/recv timeout to avoid hanging indefinitely.
            socket.setsockopt(zmq.SNDTIMEO, 10000)  # 10s
            socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30s
            self._direct_zmq_sockets[rank] = socket
            psrl_logger.info(f"[LMCache] Direct ZMQ socket connected to {worker_zmq_url} (rank {rank}).")
        return socket

    def _reset_zmq_socket(self, rank: int) -> None:
        """
        Close and discard one rank's direct ZMQ socket so the next call rebuilds it.

        Called after a send/recv failure on the REQ socket. A REQ socket that
        raised mid send→recv is stuck in a bad EFSM state and cannot be reused.
        dropping it here lets `_get_or_create_zmq_socket` reconnect cleanly.

        Caller must hold that rank's `_direct_zmq_locks[rank]`.

        Args:
            rank (int): Local rank whose socket to reset.
        """
        socket = self._direct_zmq_sockets.pop(rank, None)
        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                pass
            psrl_logger.warning(
                f"[LMCache] Direct ZMQ socket for rank {rank} reset after transfer "
                "failure. It will reconnect on the next transfer."
            )

    def _reset_all_zmq_sockets(self) -> None:
        """
        Close and discard all per-rank direct ZMQ sockets.

        Called from `set_peer_registry` so sockets reconnect with the new worker
        URLs on next use.
        """
        for rank in list(self._direct_zmq_sockets.keys()):
            socket = self._direct_zmq_sockets.pop(rank, None)
            if socket is not None:
                try:
                    socket.close(linger=0)
                except Exception:
                    pass

    # --- GPU pin budget internals ---

    async def _pin_gpu(self, tokens: list[int]) -> bool:
        """
        Pin GPU prefix-cache blocks for `tokens`, enforcing the budget.

        `psrl_pin_gpu` returns the number of blocks actually pinned by PSRL.
        Budget enforcement uses that authoritative count, avoiding any separate
        cache-info query path.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the pin succeeded, False if the budget cannot accommodate it.
        """
        pinned: int = await self._utility("psrl_pin_gpu", tokens)
        if pinned <= 0:
            psrl_logger.debug("[LMCache] GPU pin: no matching prefix-cache blocks pinned.")
            return True

        self._pinned_gpu_blocks += pinned
        self._gpu_pinned_order.append(tokens)

        if self._gpu_pin_budget > 0:
            # Evict oldest-pinned trajectories until the budget is satisfied.
            newest_evicted = False
            while self._pinned_gpu_blocks > self._gpu_pin_budget and self._gpu_pinned_order:
                oldest_tokens = self._gpu_pinned_order.popleft()
                if oldest_tokens is tokens:
                    newest_evicted = True
                freed: int = await self._utility("psrl_unpin_gpu", oldest_tokens)
                self._pinned_gpu_blocks = max(0, self._pinned_gpu_blocks - freed)
                psrl_logger.debug(
                    f"[LMCache] GPU pin budget: evicted oldest trajectory "
                    f"({freed} blocks freed, budget={self._gpu_pin_budget})."
                )

            # If the newest trajectory alone exceeds the budget, the eviction
            # loop eventually unpins it. Report the budget miss to the caller.
            if newest_evicted:
                psrl_logger.warning(
                    f"[LMCache] GPU pin budget exceeded after pinning {pinned} blocks (budget={self._gpu_pin_budget})."
                )
                return False

        psrl_logger.debug(
            f"[LMCache] GPU pin: {pinned} blocks pinned, "
            f"total={self._pinned_gpu_blocks}, budget={self._gpu_pin_budget}."
        )
        return True

    async def _unpin_gpu(self, tokens: list[int]) -> bool:
        """
        Unpin GPU prefix-cache blocks for `tokens`.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            bool: True if the unpin succeeded.
        """
        freed: int = await self._utility("psrl_unpin_gpu", tokens)
        self._pinned_gpu_blocks = max(0, self._pinned_gpu_blocks - freed)
        # Remove from order tracking (`deque` does not support arbitrary removal,
        # so rebuild without the evicted entry).
        self._gpu_pinned_order = deque(t for t in self._gpu_pinned_order if t != tokens)
        psrl_logger.debug(f"[LMCache] GPU unpin: {freed} blocks freed, total={self._pinned_gpu_blocks}.")
        return True
