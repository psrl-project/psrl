"""
Parse compact batch descriptions for chunked prefill micro-benchmarks.

The grammar is `(<count>?)q<q_len>(k?)(s<seq_len>(k?))?`. The optional `count`
defaults to one, and `k` multiplies a value by 1024.

Examples:
    `q2k` means one request with `q_len=2048` and `kv_len=2048`.
    `q1s1k` means one request with `q_len=1` and `kv_len=1024`.
    `8q1s1k` means eight identical decode requests.
    `2q2k_32q1s1k` combines two prefill and 32 decode requests.

Specifications are compatible with
`third_party/vllm/benchmarks/attention_benchmarks/batch_spec.py`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchRequest:
    """Represents a single request's token geometry in one engine step."""

    q_len: int
    """Query length: number of new tokens computed in this step."""

    kv_len: int
    """Total KV-cache length (context + query) at the end of this step."""

    def __post_init__(self) -> None:
        if self.q_len <= 0:
            raise ValueError(f"q_len must be > 0, got {self.q_len!r}.")
        if self.kv_len < self.q_len:
            raise ValueError(f"kv_len must be >= q_len, got kv_len={self.kv_len!r}, q_len={self.q_len!r}.")

    @property
    def context_len(self) -> int:
        """Number of KV tokens already in cache before this step."""
        return self.kv_len - self.q_len

    @property
    def is_decode(self) -> bool:
        """True when this is a pure decode request (q_len == 1)."""
        return self.q_len == 1

    @property
    def is_prefill(self) -> bool:
        """True when this is a pure prefill (no prior context)."""
        return self.context_len == 0

    @property
    def is_extend(self) -> bool:
        """True when this is a context-extension (q_len > 1, context_len > 0)."""
        return self.q_len > 1 and self.context_len > 0

    def as_tuple(self) -> tuple[int, int]:
        """Return ``(q_len, kv_len)`` as a plain tuple."""
        return (self.q_len, self.kv_len)


# --- Internal Helpers ---

_SEG_RE = re.compile(r"^(?:(\d+))?q(\d+)(k?)(?:s(\d+)(k?))?$")


def _parse_size(digits: str, k: str) -> int:
    """Parse an integer with an optional ``'k'`` multiplier."""
    v = int(digits)
    return v * 1024 if k == "k" else v


# --- Public API ---


def parse_batch_spec(spec: str) -> list[BatchRequest]:
    """
    Parse a batch specification string into a list of `BatchRequest` objects.

    Args:
        spec (str): Batch specification string, e.g. ``"2q2k_32q1s1k"``.

    Returns:
        list[BatchRequest]: Ordered list of request descriptors.

    Raises:
        ValueError: If any segment in ``spec`` cannot be parsed.
    """
    requests: list[BatchRequest] = []
    for seg in spec.split("_"):
        m = _SEG_RE.match(seg)
        if not m:
            raise ValueError(
                f"Invalid batch spec segment {seg!r}. "
                "Expected format: (<count>?)q<q_len>(k?)(s<seq_len>(k?))?  "
                "e.g. 'q2k', '32q1s1k', '2q512s2k'."
            )
        count = int(m.group(1)) if m.group(1) else 1
        q_len = _parse_size(m.group(2), m.group(3))
        kv_len = _parse_size(m.group(4), m.group(5)) if m.group(4) else q_len
        requests.extend([BatchRequest(q_len=q_len, kv_len=kv_len)] * count)
    return requests


def format_batch_spec(requests: list[BatchRequest]) -> str:
    """
    Produce a compact human-readable summary of a request list.

    Args:
        requests (list[BatchRequest]): The request list to summarise.

    Returns:
        str: A description like ``"2×prefill(q=2048) 32×decode(ctx=1024)"``.
    """
    counter: Counter[str] = Counter()
    for r in requests:
        if r.is_prefill:
            label = f"prefill(q={r.q_len})"
        elif r.is_decode:
            label = f"decode(ctx={r.context_len})"
        else:
            label = f"extend(q={r.q_len},ctx={r.context_len})"
        counter[label] += 1
    parts = [f"{v}×{k}" for k, v in counter.items()]
    return "  ".join(parts) if parts else "(empty)"


def total_query_tokens(requests: list[BatchRequest]) -> int:
    """Return the total number of query tokens across all requests."""
    return sum(r.q_len for r in requests)


def blocks_needed(requests: list[BatchRequest], block_size: int) -> int:
    """
    Compute the total number of KV-cache blocks required for the given requests.

    Args:
        requests (list[BatchRequest]): Request list.
        block_size (int): Block size in tokens (must be > 0).

    Returns:
        int: Total blocks needed.
    """
    return sum(math.ceil(r.kv_len / block_size) for r in requests)
