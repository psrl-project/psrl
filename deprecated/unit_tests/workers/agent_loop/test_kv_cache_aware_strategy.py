import pytest

from psrl.workers.agent_loop.route_strategy import KVCacheAwareRouteStrategy
from psrl.workers.gen.stats_collector import EngineStats


def _make_strategy(n=3, max_seqs=4):
    kwargs = {
        "max_concurrent_seqs_per_instance": max_seqs,
        "logger": __import__("logging").getLogger("test"),
    }
    return KVCacheAwareRouteStrategy(n_instances=n, strategy_kwargs=kwargs)


def _make_request(uid=1):
    from unittest.mock import MagicMock
    req = MagicMock()
    req.non_tensor_batch = {"uid": [uid]}
    return req


class TestKVCacheAwareRouteStrategy:
    def test_route_picks_highest_kv_score(self):
        """Candidate with the highest KV hit score is selected."""
        strategy = _make_strategy(n=3)
        request = _make_request()
        # instance 2 has the highest cached token score
        result = strategy.route(
            request,
            candidates=[0, 1, 2],
            route_kwargs={
                "kv_hit_scores": {0: 5, 1: 10, 2: 20},
                "candidate_indicator_list": [0, 0, 0],
            },
        )
        assert result == 2

    def test_route_tiebreak_by_request_count(self):
        """When KV scores are equal, the least-loaded instance is selected."""
        strategy = _make_strategy(n=3)
        # Manually set load imbalance
        strategy.instance_request_counts[0] = 3
        strategy.instance_request_counts[1] = 1
        strategy.instance_request_counts[2] = 2
        request = _make_request()
        result = strategy.route(
            request,
            candidates=[0, 1, 2],
            route_kwargs={
                "kv_hit_scores": {0: 0, 1: 0, 2: 0},
                "candidate_indicator_list": [0, 0, 0],
            },
        )
        assert result == 1  # least loaded

    def test_route_respects_concurrent_cap(self):
        """Best KV candidate is skipped when at max_concurrent_seqs cap."""
        strategy = _make_strategy(n=2, max_seqs=2)
        strategy.instance_request_counts[0] = 2  # at cap
        strategy.instance_request_counts[1] = 0
        request = _make_request()
        result = strategy.route(
            request,
            candidates=[0, 1],
            route_kwargs={
                "kv_hit_scores": {0: 100, 1: 5},
                "candidate_indicator_list": [0, 0],
            },
        )
        assert result == 1  # falls back to instance 1 even though lower score

    def test_route_all_at_cap_returns_none(self):
        """Returns None when all candidates are at the concurrency cap."""
        strategy = _make_strategy(n=2, max_seqs=2)
        strategy.instance_request_counts[0] = 2
        strategy.instance_request_counts[1] = 2
        request = _make_request()
        result = strategy.route(
            request,
            candidates=[0, 1],
            route_kwargs={
                "kv_hit_scores": {0: 10, 1: 10},
                "candidate_indicator_list": [0, 0],
            },
        )
        assert result is None

    def test_route_missing_kv_scores_asserts(self):
        """AssertionError raised when kv_hit_scores is absent from route_kwargs."""
        strategy = _make_strategy()
        request = _make_request()
        with pytest.raises(AssertionError):
            strategy.route(
                request,
                candidates=[0, 1],
                route_kwargs={"candidate_indicator_list": [0, 0]},
            )

    def test_route_empty_candidates_returns_none(self):
        """Returns None when candidates list is empty."""
        strategy = _make_strategy()
        request = _make_request()
        result = strategy.route(
            request,
            candidates=[],
            route_kwargs={
                "kv_hit_scores": {},
                "candidate_indicator_list": [],
            },
        )
        assert result is None

    def test_route_increments_request_count(self):
        """Request count for the selected instance is incremented by route()."""
        strategy = _make_strategy(n=2)
        assert strategy.instance_request_counts[1] == 0
        request = _make_request()
        result = strategy.route(
            request,
            candidates=[0, 1],
            route_kwargs={
                "kv_hit_scores": {0: 0, 1: 10},
                "candidate_indicator_list": [0, 0],
            },
        )
        assert result == 1
        assert strategy.instance_request_counts[1] == 1

    def test_pop_request_decrements_request_count(self):
        """pop_request() decrements the request count for the instance."""
        strategy = _make_strategy(n=2)
        strategy.instance_request_counts[0] = 3
        request = _make_request()
        strategy.pop_request(request, instance_id=0)
        assert strategy.instance_request_counts[0] == 2

    def test_update_instance_status_syncs_counts(self):
        """update_instance_to_engine_status() syncs instance_request_counts."""
        strategy = _make_strategy(n=2)
        snapshot = EngineStats.get_default_snapshot()
        snapshot["scheduler_stats"] = {
            "num_running_reqs": 3,
            "num_waiting_reqs": 1,
        }
        stats = {
            0: EngineStats(instance_id=0, model_version=0, snapshot=snapshot),
        }
        strategy.update_instance_to_engine_status(stats)
        assert strategy.instance_request_counts[0] == 4  # running + waiting


import asyncio
from unittest.mock import AsyncMock, MagicMock


class TestRouterKVScoreQuery:
    """Integration-style tests for the step-6 KV score injection in router."""

    def _make_router_shell(self, method="kv_cache_aware", timeout_ms=200):
        """
        Return a minimal RolloutRouter-like object with only the attributes
        used by the step-6 code path.
        """
        router = MagicMock()
        router.config.psrl.rollout_coordination.routing_strategy.method = method
        router.config.psrl.rollout_coordination.routing_strategy.kv_query_timeout_ms = timeout_ms
        router.request_to_tokens = {}
        # Two fake worker group handles.
        wg0 = MagicMock()
        wg1 = MagicMock()
        router.rollout_wg_list = [wg0, wg1]
        return router

    def _make_cache_info_dict(self, gpu_blocks=5, lmcache_chunks=3):
        """Return the raw dict that execute_rank_zero_async returns."""
        from psrl.utils.kv_cache.types import TrajectoryCacheInfo
        import dataclasses
        info = TrajectoryCacheInfo(
            total_tokens=256,
            lmcache_cached_chunks=lmcache_chunks,
            lmcache_cached_tokens=lmcache_chunks * 256,
            lmcache_bytes=0,
            lmcache_total_bytes=0,
            lmcache_usage_pct=0.0,
            gpu_cached_blocks=gpu_blocks,
            gpu_cached_tokens=gpu_blocks * 16,
            gpu_total_blocks=100,
            gpu_usage_pct=gpu_blocks / 100,
            gpu_pinned=False,
            backend_pinned=False,
        )
        return dataclasses.asdict(info)

    def test_cold_start_asserts_missing_uid(self):
        """When tokens are not registered for a uid, _inject_kv_hit_scores raises AssertionError."""
        router = self._make_router_shell()
        router.request_to_tokens = {}
        candidates = [0, 1]
        route_kwargs = {}

        async def _run():
            from psrl.workers.agent_loop.router import RolloutRouter

            await RolloutRouter._inject_kv_hit_scores(
                router, request_id=42, candidates=candidates, route_kwargs=route_kwargs
            )

        with pytest.raises(AssertionError, match="uid=42 not found in request_to_tokens"):
            asyncio.run(_run())

        router.rollout_wg_list[0].execute_rank_zero_async.assert_not_called()
        router.rollout_wg_list[1].execute_rank_zero_async.assert_not_called()

    def test_scores_computed_from_rpc(self):
        """KV scores equal max(gpu_cached_tokens, lmcache_cached_tokens) per instance."""
        router = self._make_router_shell()
        router.request_to_tokens = {42: [1, 2, 3]}
        candidates = [0, 1]
        route_kwargs = {}

        router.rollout_wg_list[0].execute_rank_zero_async = AsyncMock(
            return_value=self._make_cache_info_dict(gpu_blocks=5, lmcache_chunks=3)
        )
        router.rollout_wg_list[1].execute_rank_zero_async = AsyncMock(
            return_value=self._make_cache_info_dict(gpu_blocks=2, lmcache_chunks=1)
        )

        async def _run():
            from psrl.workers.agent_loop.router import RolloutRouter

            await RolloutRouter._inject_kv_hit_scores(
                router, request_id=42, candidates=candidates, route_kwargs=route_kwargs
            )

        asyncio.run(_run())

        # gpu_tokens = blocks*16, lmcache_tokens = chunks*256; score = max of the two
        assert route_kwargs["kv_hit_scores"] == {0: 768, 1: 256}

    def test_rpc_timeout_scores_zero(self):
        """Instance that exceeds kv_query_timeout_ms gets score 0."""
        router = self._make_router_shell(timeout_ms=1)
        router.request_to_tokens = {42: [1, 2, 3]}
        candidates = [0, 1]
        route_kwargs = {}

        async def _slow(*_):
            await asyncio.sleep(10)

        router.rollout_wg_list[0].execute_rank_zero_async = AsyncMock(
            return_value=self._make_cache_info_dict(gpu_blocks=5, lmcache_chunks=0)
        )
        router.rollout_wg_list[1].execute_rank_zero_async = _slow

        async def _run():
            from psrl.workers.agent_loop.router import RolloutRouter

            await RolloutRouter._inject_kv_hit_scores(
                router, request_id=42, candidates=candidates, route_kwargs=route_kwargs
            )

        asyncio.run(_run())

        assert route_kwargs["kv_hit_scores"][0] == 80  # max(5*16, 0*256)
        assert route_kwargs["kv_hit_scores"][1] == 0

    def test_rpc_exception_scores_zero(self):
        """Instance that raises an exception gets score 0."""
        router = self._make_router_shell()
        router.request_to_tokens = {42: [1, 2, 3]}
        candidates = [0, 1]
        route_kwargs = {}

        router.rollout_wg_list[0].execute_rank_zero_async = AsyncMock(
            return_value=self._make_cache_info_dict(gpu_blocks=7, lmcache_chunks=2)
        )
        router.rollout_wg_list[1].execute_rank_zero_async = AsyncMock(
            side_effect=RuntimeError("RPC failed")
        )

        async def _run():
            from psrl.workers.agent_loop.router import RolloutRouter

            await RolloutRouter._inject_kv_hit_scores(
                router, request_id=42, candidates=candidates, route_kwargs=route_kwargs
            )

        asyncio.run(_run())

        assert route_kwargs["kv_hit_scores"][0] == 512  # max(7*16=112, 2*256=512)
        assert route_kwargs["kv_hit_scores"][1] == 0

    def test_non_kv_strategy_skips_injection(self):
        """No kv_hit_scores key added when method != kv_cache_aware."""
        router = self._make_router_shell(method="random")
        router.request_to_tokens = {42: [1, 2, 3]}
        candidates = [0, 1]
        route_kwargs = {}

        async def _run():
            from psrl.workers.agent_loop.router import RolloutRouter

            await RolloutRouter._inject_kv_hit_scores(
                router, request_id=42, candidates=candidates, route_kwargs=route_kwargs
            )

        asyncio.run(_run())

        assert "kv_hit_scores" not in route_kwargs
