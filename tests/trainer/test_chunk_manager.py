"""Unit tests for the manager's chunk-yielding path.

We call the manager methods directly (no Ray), simulating what
`occupy_requests` would do when prompt groups complete.

All tests use `asyncio.run()` so they work without `pytest-asyncio`.
"""

import asyncio
from unittest.mock import MagicMock

import pytest
from transfer_queue import KVBatchMeta

pytestmark = pytest.mark.cpu_test


# Helpers


def _make_entry_info(
    prompt_id: int,
    request_idx: int = 0,
    model_version: int = 1,
    n_trajectory: int = 1,
):
    """Build a minimal EntryInfo-like mock."""
    ei = MagicMock()
    ei.prompt_id = prompt_id
    ei.get_entry_version.return_value = model_version
    ei.request_idx = request_idx
    ei.n_trajectory = n_trajectory
    ei.model_version = model_version
    return ei


class FakeManager:
    """Minimal stub of `PSRL_AgentLoopManager` for testing the chunk state machine.

    Borrows `_emit_pending_chunks` and `wait_for_training_chunk` directly from
    the real class so the tests exercise real production code.
    """

    def __init__(self, chunk_size: int | None, ready_total: int, rollout_n: int = 1) -> None:
        from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager

        self.train_chunk_size = chunk_size
        self._train_chunk_consumed: dict[int, int] = {}
        self._train_chunk_emitted_entry_ids: dict[int, set[int]] = {}
        self._train_chunk_waiters: dict[tuple, list] = {}
        self._resolved_train_chunks: dict[tuple, tuple] = {}
        self.ready_entries_per_buffer = ready_total
        self.rollout_n = rollout_n
        self.train_accumulated_buffers: dict[int, dict[int, list]] = {}
        self.train_accumulated_buffer_size: dict[int, int] = {}

        # The refill breaker is latch-free here: these tests cover the healthy
        # chunk path, so the guard must read as "not tripped".
        self._refill_breaker_diagnosis = None

        # Bind real methods from the production class.
        self._emit_pending_chunks = PSRL_AgentLoopManager._emit_pending_chunks.__get__(self)
        self._refill_breaker_error = PSRL_AgentLoopManager._refill_breaker_error.__get__(self)
        self._raise_if_refill_breaker_tripped = PSRL_AgentLoopManager._raise_if_refill_breaker_tripped.__get__(self)
        self.wait_for_training_chunk = PSRL_AgentLoopManager.wait_for_training_chunk.__get__(self)

    def entry_infos_to_kv_batch_meta(self, entries, is_validate: bool = False) -> KVBatchMeta:
        """Return a minimal KVBatchMeta for the given entry list."""
        keys = []
        tags = []
        for entry in entries:
            for trajectory_index in range(entry.n_trajectory):
                key = str(entry.prompt_id) if entry.n_trajectory == 1 else f"{entry.prompt_id}_{trajectory_index}"
                keys.append(key)
                tags.append({"uid": entry.prompt_id})
        return KVBatchMeta(keys=keys, tags=tags, partition_id="train")

    def _add_groups(
        self,
        buffer_id: int,
        n_groups: int,
        start_prompt_id: int = 0,
        model_version: int = 1,
        n_trajectory: int = 1,
    ) -> None:
        """Append n_groups fake EntryInfo objects to `train_accumulated_buffers[buffer_id]`."""
        if buffer_id not in self.train_accumulated_buffers:
            self.train_accumulated_buffers[buffer_id] = {}
        entries = [
            _make_entry_info(
                start_prompt_id + i,
                model_version=model_version,
                n_trajectory=n_trajectory,
            )
            for i in range(n_groups)
        ]
        self.train_accumulated_buffers[buffer_id].setdefault(model_version, []).extend(entries)
        self.train_accumulated_buffer_size[buffer_id] = self.train_accumulated_buffer_size.get(buffer_id, 0) + n_groups


# Tests


class TestChunkEmission:
    """Behavioral tests for `_emit_pending_chunks` and `wait_for_training_chunk`."""

    def test_waiter_registered_before_chunk_arrives(self):
        """Waiter registered first resolves when `_emit_pending_chunks` fires.

        Simulates the common async ordering: trainer calls
        `wait_for_training_chunk` before the buffer has enough groups.
        """

        async def _run():
            mgr = FakeManager(chunk_size=2, ready_total=4)
            buf = 0

            # Register waiter for chunk_index=0 BEFORE any groups accumulate.
            waiter = asyncio.create_task(mgr.wait_for_training_chunk(buf, 0))
            await asyncio.sleep(0)  # Yield so the task registers the future.

            # Accumulate 2 groups (exactly one chunk).
            mgr._add_groups(buf, n_groups=2)
            mgr._emit_pending_chunks(buf)

            chunk_meta, is_last = await waiter
            assert isinstance(chunk_meta, KVBatchMeta)
            assert len(chunk_meta) == 2
            # More chunks pending (total=4, chunk_size=2).
            assert is_last is False, f"Expected is_last=False, got {is_last!r}."

        asyncio.run(_run())

    def test_chunk_arrives_before_waiter(self):
        """Pre-resolved chunk is consumed correctly by a late-arriving waiter.

        Simulates the case where the rollout side is fast: the chunk is already
        in `_resolved_train_chunks` when `wait_for_training_chunk` runs.
        """

        async def _run():
            mgr = FakeManager(chunk_size=2, ready_total=4)
            buf = 1

            # Emit first chunk before any waiter is registered.
            mgr._add_groups(buf, n_groups=2)
            mgr._emit_pending_chunks(buf)

            # Resolved chunk should be stored.
            assert (buf, 0) in mgr._resolved_train_chunks, "Expected pre-resolved chunk in _resolved_train_chunks."

            # Late-arriving waiter should resolve immediately.
            chunk_meta, is_last = await mgr.wait_for_training_chunk(buf, 0)
            assert isinstance(chunk_meta, KVBatchMeta)
            assert len(chunk_meta) == 2
            assert is_last is False

            assert (buf, 0) not in mgr._resolved_train_chunks, (
                "Resolved chunk should have been consumed by wait_for_training_chunk."
            )

        asyncio.run(_run())

    def test_is_last_on_exact_divisible(self):
        """Mark the second of two evenly sized chunks as final."""

        async def _run():
            mgr = FakeManager(chunk_size=2, ready_total=4)
            buf = 2

            # Accumulate all 4 groups at once and emit.
            mgr._add_groups(buf, n_groups=4)
            mgr._emit_pending_chunks(buf)

            _, is_last_0 = await mgr.wait_for_training_chunk(buf, 0)
            assert is_last_0 is False, f"Expected is_last=False for chunk 0, got {is_last_0!r}."

            _, is_last_1 = await mgr.wait_for_training_chunk(buf, 1)
            assert is_last_1 is True, f"Expected is_last=True for chunk 1 (last chunk), got {is_last_1!r}."

        asyncio.run(_run())

    def test_tail_chunk_is_last_on_remainder(self):
        """When total groups is NOT divisible by chunk_size, tail chunk carries is_last=True.

        5 groups, chunk_size=2 → 2 full chunks + 1 tail chunk.
        Tail (index 2) has is_last=True and contains 1 group.
        """

        async def _run():
            mgr = FakeManager(chunk_size=2, ready_total=5)
            buf = 3

            # Accumulate all 5 groups at once (simulating buffer-full trigger).
            mgr._add_groups(buf, n_groups=5)
            mgr._emit_pending_chunks(buf)

            _, is_last_0 = await mgr.wait_for_training_chunk(buf, 0)
            assert is_last_0 is False, f"Expected is_last=False for chunk 0, got {is_last_0!r}."

            _, is_last_1 = await mgr.wait_for_training_chunk(buf, 1)
            assert is_last_1 is False, f"Expected is_last=False for chunk 1, got {is_last_1!r}."

            chunk_2, is_last_2 = await mgr.wait_for_training_chunk(buf, 2)
            assert is_last_2 is True, f"Expected is_last=True for tail chunk 2, got {is_last_2!r}."
            # Tail carries the single remaining group.
            assert len(chunk_2) == 1, f"Tail chunk should have 1 entry, got {len(chunk_2)}."

        asyncio.run(_run())

    def test_chunk_size_none_no_op(self):
        """`_emit_pending_chunks` is a strict no-op when `train_chunk_size is None`."""
        mgr = FakeManager(chunk_size=None, ready_total=4)
        buf = 4

        mgr._add_groups(buf, n_groups=4)
        mgr._emit_pending_chunks(buf)

        assert not mgr._train_chunk_waiters, (
            f"Expected empty _train_chunk_waiters when chunk_size is None, "
            f"got {list(mgr._train_chunk_waiters.keys())!r}."
        )
        assert not mgr._resolved_train_chunks, (
            f"Expected empty _resolved_train_chunks when chunk_size is None, "
            f"got {list(mgr._resolved_train_chunks.keys())!r}."
        )
        assert not mgr._train_chunk_consumed, (
            f"Expected empty _train_chunk_consumed when chunk_size is None, got {mgr._train_chunk_consumed!r}."
        )

    def test_accumulation_order_not_resorted_across_emits(self):
        """Later chunks must not re-slice a prompt_id-resorted full list.

        If high prompt_ids finish first, re-sorting the full buffer on the
        second emit would duplicate those groups and drop the late low ids.
        """

        async def _run():
            mgr = FakeManager(chunk_size=2, ready_total=4)
            buf = 0

            # High prompt_ids complete first.
            mgr._add_groups(buf, n_groups=2, start_prompt_id=2, model_version=2)
            mgr._emit_pending_chunks(buf)
            c0, last0 = await mgr.wait_for_training_chunk(buf, 0)
            assert sorted(map(int, c0.keys)) == [2, 3]
            assert last0 is False

            # Low prompt_ids complete later.
            mgr._add_groups(buf, n_groups=2, start_prompt_id=0, model_version=1)
            mgr._emit_pending_chunks(buf)
            assert (buf, 1) in mgr._resolved_train_chunks, "chunk 1 must be emitted"
            c1, last1 = await mgr.wait_for_training_chunk(buf, 1)
            assert last1 is True
            assert sorted(map(int, c1.keys)) == [0, 1], f"chunk 1 should be the late groups [0,1], got {c1.keys}"
            assert set(c0.keys).isdisjoint(c1.keys)

        asyncio.run(_run())

    def test_multi_trajectory_request_is_not_split(self):
        """A request remains one chunk unit even when it expands to trajectories."""

        async def _run():
            mgr = FakeManager(chunk_size=1, ready_total=2)
            buf = 1

            mgr._add_groups(buf, n_groups=1, n_trajectory=3)
            mgr._emit_pending_chunks(buf)
            chunk, is_last = await mgr.wait_for_training_chunk(buf, 0)

            assert chunk.keys == ["0_0", "0_1", "0_2"]
            assert {tag["uid"] for tag in chunk.tags} == {0}
            assert is_last is False

        asyncio.run(_run())
