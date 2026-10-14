"""Run CPU-only unit tests for `batch_spec.py`."""

import pytest
from psrl.bench.chunked_prefill.batch_spec import (
    BatchRequest,
    blocks_needed,
    format_batch_spec,
    parse_batch_spec,
    total_query_tokens,
)


class TestParseBatchSpec:
    def test_pure_prefill_k_suffix(self):
        reqs = parse_batch_spec("q2k")
        assert reqs == [BatchRequest(q_len=2048, kv_len=2048)]

    def test_decode(self):
        reqs = parse_batch_spec("q1s1k")
        assert reqs == [BatchRequest(q_len=1, kv_len=1024)]

    def test_count_prefix(self):
        reqs = parse_batch_spec("8q1s1k")
        assert len(reqs) == 8
        assert all(r == BatchRequest(q_len=1, kv_len=1024) for r in reqs)

    def test_mixed_spec(self):
        reqs = parse_batch_spec("2q2k_32q1s1k")
        assert reqs[:2] == [BatchRequest(q_len=2048, kv_len=2048)] * 2
        assert reqs[2:] == [BatchRequest(q_len=1, kv_len=1024)] * 32
        assert len(reqs) == 34

    def test_extend(self):
        reqs = parse_batch_spec("q4s1k")
        assert reqs == [BatchRequest(q_len=4, kv_len=1024)]
        assert reqs[0].is_extend

    def test_no_k_suffix(self):
        reqs = parse_batch_spec("q512s2048")
        assert reqs == [BatchRequest(q_len=512, kv_len=2048)]

    def test_single_token_prefill(self):
        reqs = parse_batch_spec("q1")
        assert reqs == [BatchRequest(q_len=1, kv_len=1)]
        assert reqs[0].is_prefill

    def test_invalid_segment_raises(self):
        with pytest.raises(ValueError, match="Invalid batch spec segment"):
            parse_batch_spec("invalid")

    def test_kv_less_than_q_raises(self):
        with pytest.raises(ValueError):
            BatchRequest(q_len=512, kv_len=256)


class TestBatchRequestProperties:
    def test_is_decode(self):
        r = BatchRequest(q_len=1, kv_len=1024)
        assert r.is_decode
        assert not r.is_prefill
        assert not r.is_extend

    def test_is_prefill(self):
        r = BatchRequest(q_len=2048, kv_len=2048)
        assert r.is_prefill
        assert not r.is_decode
        assert not r.is_extend

    def test_is_extend(self):
        r = BatchRequest(q_len=4, kv_len=1024)
        assert r.is_extend
        assert not r.is_decode
        assert not r.is_prefill

    def test_context_len(self):
        r = BatchRequest(q_len=128, kv_len=512)
        assert r.context_len == 384

    def test_as_tuple(self):
        r = BatchRequest(q_len=4, kv_len=1024)
        assert r.as_tuple() == (4, 1024)


class TestHelpers:
    def test_total_query_tokens(self):
        reqs = parse_batch_spec("2q2k_32q1s1k")
        assert total_query_tokens(reqs) == 2 * 2048 + 32 * 1

    def test_blocks_needed(self):
        reqs = [BatchRequest(q_len=2048, kv_len=2048)]
        assert blocks_needed(reqs, block_size=16) == 128

    def test_blocks_needed_partial(self):
        reqs = [BatchRequest(q_len=1, kv_len=1)]
        assert blocks_needed(reqs, block_size=16) == 1  # ceiling

    def test_format_batch_spec_prefill(self):
        reqs = [BatchRequest(q_len=2048, kv_len=2048)]
        result = format_batch_spec(reqs)
        assert "prefill" in result
        assert "2048" in result

    def test_format_batch_spec_mixed(self):
        reqs = parse_batch_spec("2q2k_32q1s1k")
        result = format_batch_spec(reqs)
        assert "prefill" in result
        assert "decode" in result
