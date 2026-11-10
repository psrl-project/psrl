import pytest
from vllm.distributed.parallel_state import (
    _node_count,
    _shared_memory_domain_id,
    in_the_same_node_as,
)

pytestmark = pytest.mark.cpu_test


class _FakeStatelessProcessGroup:
    def __init__(self, domains, rank: int = 0) -> None:
        self.domains = domains
        self.rank = rank
        self.world_size = len(domains)

    def broadcast_obj(self, obj, src: int):
        return self.domains[src]


def test_shared_memory_domain_id_is_stable_without_allocating_shm() -> None:
    first = _shared_memory_domain_id()
    second = _shared_memory_domain_id()

    assert first == second
    assert len(first) == 3
    assert first[0]


def test_same_node_detection_uses_shared_memory_domain_identity() -> None:
    domains = [
        ("node-a", 1, 10),
        ("node-a", 1, 10),
        ("node-b", 2, 11),
        ("node-b", 2, 11),
    ]
    group = _FakeStatelessProcessGroup(domains, rank=1)

    assert in_the_same_node_as(group, source_rank=0) == [True, True, False, False]
    assert in_the_same_node_as(group, source_rank=2) == [False, False, True, True]
    assert _node_count(group) == 2
