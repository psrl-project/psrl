import logging
import os
import time

from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash, make_block_hash_with_group_id
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.utils import compute_iteration_details

from psrl.utils.logger import FileOnlyHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RolloutScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The server supplies prefill logging settings through scheduler attributes.
        sc = self.scheduler_config
        self._pcomp_enable: bool = bool(getattr(sc, "psrl_prefill_composition_enable", False))
        self._pcomp_logger: logging.Logger | None = None
        self._pcomp_pending: dict[int, tuple[SchedulerOutput, float]] = {}
        self._pcomp_step: int = 0
        if self._pcomp_enable:
            logging_path = str(getattr(sc, "psrl_logging_path", "~/psrl_logs"))
            replica_idx = int(getattr(sc, "psrl_replica_idx", 0))
            self._pcomp_logger = logging.getLogger(f"psrl.prefill_composition.I{replica_idx}")
            self._pcomp_logger.propagate = False
            self._pcomp_logger.setLevel(logging.INFO)
            self._pcomp_logger.addHandler(FileOnlyHandler(logging_path, f"Prefill_I{replica_idx}"))

    def schedule(self) -> SchedulerOutput:
        sched_out = super().schedule()
        if self._pcomp_enable:
            self._pcomp_pending[id(sched_out)] = (sched_out, time.perf_counter())
        return sched_out

    def update_from_output(self, scheduler_output, model_runner_output):
        result = super().update_from_output(scheduler_output, model_runner_output)
        if self._pcomp_enable:
            self._emit_prefill_composition(scheduler_output)
        return result

    def _emit_prefill_composition(self, scheduler_output: SchedulerOutput) -> None:
        entry = self._pcomp_pending.pop(id(scheduler_output), None)
        if entry is None:
            return
        _, sched_ts = entry
        host_ms = (time.perf_counter() - sched_ts) * 1000.0

        iteration_details = compute_iteration_details(scheduler_output)
        if iteration_details.num_ctx_requests == 0:
            return

        self._pcomp_step += 1
        M = scheduler_output.total_num_scheduled_tokens
        nseq = len(scheduler_output.num_scheduled_tokens)

        lines: list[str] = [
            f"step={self._pcomp_step} M={M} nseq={nseq} host_ms={host_ms:.1f}"
            f" ctx_reqs={iteration_details.num_ctx_requests}"
            f" ctx_tokens={iteration_details.num_ctx_tokens}"
            f" gen_reqs={iteration_details.num_generation_requests}"
        ]

        new_req_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        new_req_data = {r.req_id: r for r in scheduler_output.scheduled_new_reqs}
        cached_reqs = scheduler_output.scheduled_cached_reqs
        cached_num_computed = dict(zip(cached_reqs.req_ids, cached_reqs.num_computed_tokens))

        for idx, (req_id, q) in enumerate(scheduler_output.num_scheduled_tokens.items()):
            rid_short = req_id[-8:] if len(req_id) > 8 else req_id
            if req_id in new_req_ids:
                rd = new_req_data[req_id]
                hit = rd.num_computed_tokens
                # Attempt to read local vs external split non-destructively.
                req_obj = self.requests.get(req_id)
                pfs = getattr(req_obj, "prefill_stats", None) if req_obj else None
                if pfs is not None:
                    local_hit = pfs.num_local_cached_tokens
                    ext_hit = pfs.num_external_cached_tokens
                    prompt = pfs.num_prompt_tokens
                    hit_str = f"hit={hit}(L{local_hit}/E{ext_hit}) q={q} prompt={prompt}"
                else:
                    hit_str = f"hit={hit} q={q}"
                lines.append(f"  [{idx}] rid=..{rid_short} new    {hit_str}")
            elif cached_reqs.is_context_phase(req_id):
                ctx = cached_num_computed.get(req_id, 0)
                lines.append(f"  [{idx}] rid=..{rid_short} chunk  ctx={ctx} q={q}")
            else:
                ctx = cached_num_computed.get(req_id, 0)
                lines.append(f"  [{idx}] rid=..{rid_short} decode ctx={ctx} q={q}")

        self._pcomp_logger.info("\n".join(lines))  # type: ignore[union-attr]

    def psrl_get_runtime_stats(
        self,
    ) -> dict:
        """
        Read live queue state without draining iteration metrics or preemption events.
        """
        waiting = [*self.waiting, *self.skipped_waiting]
        requests = [*self.running, *waiting]
        return {
            "req_id_to_prompt_token_num": {req.request_id: req.num_prompt_tokens for req in requests},
            "req_id_to_response_token_num": {req.request_id: req.num_output_tokens for req in requests},
            "req_id_in_waiting": [req.request_id for req in waiting],
            "num_running_reqs": len(self.running),
            "num_waiting_reqs": len(waiting),
            "kv_cache_usage": self.kv_cache_manager.usage,
        }

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None, "Prefix cache statistics are unavailable."
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = self.kv_metrics_collector.drain_events() if self.kv_metrics_collector is not None else []
        spec_stats = spec_decoding_stats
        connector_stats_payload = kv_connector_stats.data if kv_connector_stats else None
        req_id_to_prompt_token_num = {req_id: req.num_prompt_tokens for req_id, req in self.requests.items()}
        req_id_to_response_token_num = {req_id: req.num_output_tokens for req_id, req in self.requests.items()}
        req_id_in_waiting = [req.request_id for req in self.waiting]
        # NOTE(lhy): we need to patch the original vllm SchedulerStats to add:
        # 1. `need_to_abort_reqs` field. This is a set of request IDs that need to be aborted.
        # 2. `req_id_to_prompt_token_num` field. This is a dictionary of request ID to the number of prompt tokens.
        # 3. `req_id_to_response_token_num` field. This is a dictionary of request ID to the number of response tokens.
        # 4. `req_id_in_waiting` field. This is a list of request IDs in the waiting queue.
        # NOTE(lhy): PSRL adds per-request prompt and response token counts to `SchedulerStats`.
        # Preserve and clear `preemption_req_ids` after each statistics snapshot.
        preemption_req_ids = self.preemption_req_ids
        self.preemption_req_ids = []
        return SchedulerStats(
            req_id_to_prompt_token_num=req_id_to_prompt_token_num,
            req_id_to_response_token_num=req_id_to_response_token_num,
            req_id_in_waiting=req_id_in_waiting,
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            num_skipped_waiting_reqs=len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
            preemption_req_ids=preemption_req_ids,
        )

    # --- PSRL GPU Block Pool ---

    def _psrl_get_caching_hash_fn(self):
        """
        Return the token-hashing function used by the block pool.

        Reads prefix_caching_hash_algo from self.vllm_config (available on the Scheduler).
        """
        from vllm.utils.hashing import get_hash_fn_by_name

        hash_algo = self.vllm_config.cache_config.prefix_caching_hash_algo
        return get_hash_fn_by_name(hash_algo)

    def _psrl_iter_gpu_prefix_blocks(self, tokens: list[int]):
        """
        Yield GPU `KVCacheBlock` objects forming the longest contiguous cached prefix.

        Walks the prefix-hash chain on `block_pool.cached_block_hash_to_block`,
        stopping at the first miss.

        Args:
            tokens (list[int]): Full token sequence.

        Yields:
            KVCacheBlock: Blocks in prefix order.
        """
        block_pool = self.kv_cache_manager.block_pool
        block_size = block_pool.hash_block_size
        hash_fn = self._psrl_get_caching_hash_fn()
        # `NONE_HASH` must be initialized before calling `hash_block_tokens`.
        init_none_hash(hash_fn)

        prev_hash = None
        num_full_blocks = len(tokens) // block_size
        for block_idx in range(num_full_blocks):
            start = block_idx * block_size
            end = start + block_size
            chunk = tokens[start:end]
            block_hash = hash_block_tokens(hash_fn, prev_hash, chunk, None)
            prev_hash = block_hash
            # `kv_cache_group_id=0` for standard (non-MLA) models.
            key = make_block_hash_with_group_id(block_hash, 0)
            block = block_pool.cached_block_hash_to_block.get_one_block(key)
            if block is None:
                return  # prefix break
            yield block

    def psrl_pin_gpu(self, tokens: list[int]) -> int:
        """
        Pin GPU prefix-cache blocks for `tokens` by incrementing `ref_cnt`.

        Only pins blocks with `ref_cnt == 0` in the free queue. Tracks pinned block
        IDs in `_psrl_pinned_block_ids` so `psrl_unpin_gpu` cannot decrement
        `ref_cnt` for blocks held by active vLLM requests.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of blocks newly pinned.
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_block_ids"):
            self._psrl_pinned_block_ids: set[int] = set()

        block_pool = self.kv_cache_manager.block_pool
        pinned = 0
        blocks_to_touch = []
        for block in self._psrl_iter_gpu_prefix_blocks(tokens):
            if block.ref_cnt == 0 and block.block_id not in self._psrl_pinned_block_ids:
                blocks_to_touch.append(block)
                self._psrl_pinned_block_ids.add(block.block_id)
                pinned += 1
        if blocks_to_touch:
            # NOTE(claude): Standard models pass one sequence because they have one KV-cache group.
            block_pool.touch((blocks_to_touch,))
            for block in blocks_to_touch:
                assert block.ref_cnt > 0, (
                    f"Invalid block reference count after touch: block_id={block.block_id}, ref_cnt={block.ref_cnt}."
                )
        psrl_logger.debug(f"[LMCache] Scheduler GPU pin: blocks={pinned}, token_count={len(tokens)}.")
        return pinned

    def psrl_unpin_gpu(self, tokens: list[int]) -> int:
        """
        Unpin GPU prefix-cache blocks for `tokens` by decrementing `ref_cnt`.

        Only decrements `ref_cnt` for blocks that PSRL itself pinned (tracked in
        `_psrl_pinned_block_ids`), preventing interference with active requests.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of blocks unpinned.
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_block_ids"):
            self._psrl_pinned_block_ids: set[int] = set()

        block_pool = self.kv_cache_manager.block_pool
        freed = 0
        for block in self._psrl_iter_gpu_prefix_blocks(tokens):
            if block.block_id in self._psrl_pinned_block_ids:
                assert block.ref_cnt > 0, (
                    f"Invalid block reference count before free_blocks: block_id={block.block_id}, "
                    f"ref_cnt={block.ref_cnt}. "
                    "Cannot unpin a block with ref_cnt <= 0."
                )
                block_pool.free_blocks([block])
                self._psrl_pinned_block_ids.discard(block.block_id)
                freed += 1
        psrl_logger.debug(f"[LMCache] Scheduler GPU unpin: blocks={freed}, token_count={len(tokens)}.")
        return freed

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, "Only running requests can be preempted"
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # NOTE(claude): Record `QUEUED` immediately after `PREEMPTED` so resumed
        # scheduler wait time starts at requeue.
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED, timestamp)

        # NOTE(claude): Save output count to detect the first decode token after preemption.
        request._psrl_cycle_output_token_baseline = request.num_output_tokens

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
        # Notify external gateway if threshold is configured and waiting queue
        # is already congested. Local requeuing would only worsen the load.
        threshold = self.scheduler_config.preemption_notification_threshold
        if self.log_stats and threshold is not None and len(self.waiting) > threshold:
            self.preemption_req_ids.append(request.request_id)
