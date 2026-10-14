# tests/test_thunder_agent_scheduler.py
import pytest
from psrl.workers.gen.rollout_coordination.session.base import (
    SESSION_HUNG,
    SESSION_RUNNING,
    STATUS_ENV,
    STATUS_GENERATE,
    InstanceCapacity,
    SessionInfo,
)
from psrl.workers.gen.rollout_coordination.session.thunder_agent import (
    ThunderAgentScheduler,
)

pytestmark = pytest.mark.cpu_test


I0 = ("w0", 0)
I1 = ("w1", 0)


def _sched(env_token_weight=1.0, buffer_per_session=0, global_scope=False):
    # buffer_per_session=0 by default keeps the arithmetic easy to reason about.
    return ThunderAgentScheduler(
        env_token_weight=env_token_weight,
        buffer_per_session=buffer_per_session,
        global_scope=global_scope,
    )


def _session(sid, instance_id, status, tokens, hang_state=SESSION_RUNNING):
    return SessionInfo(
        session_id=sid,
        instance_id=instance_id,
        status=status,
        hang_state=hang_state,
        total_tokens=tokens,
    )


class TestHang:
    def test_no_hang_when_within_capacity(self):
        sched = _sched()
        instances = [InstanceCapacity(I0, total_kv_tokens=1000, used_tokens=300)]
        sessions = [
            _session("a", I0, STATUS_GENERATE, 200),
            _session("b", I0, STATUS_ENV, 100),
        ]
        to_hang, to_continue = sched.decide(instances, sessions)
        assert to_hang == []
        assert to_continue == []

    def test_hang_env_before_generate(self):
        # attributed used = 600 (gen) + 400 (env) = 1000 > 800 capacity.
        # used_tokens large so min() picks attributed. Hang smallest env first.
        sched = _sched()
        instances = [InstanceCapacity(I0, total_kv_tokens=800, used_tokens=100_000)]
        sessions = [
            _session("gen_big", I0, STATUS_GENERATE, 600),
            _session("env_small", I0, STATUS_ENV, 150),
            _session("env_big", I0, STATUS_ENV, 250),
        ]
        to_hang, _ = sched.decide(instances, sessions)
        # Both environment sessions must be evicted before attributed usage fits.
        assert to_hang == ["env_small", "env_big"]

    def test_hang_generate_only_when_no_env_left(self):
        sched = _sched()
        instances = [InstanceCapacity(I0, total_kv_tokens=500, used_tokens=100_000)]
        sessions = [
            _session("g1", I0, STATUS_GENERATE, 400),
            _session("g2", I0, STATUS_GENERATE, 300),
        ]
        to_hang, _ = sched.decide(instances, sessions)
        # used = 700 > 500. Smallest generate first (g2=300) → remaining = 500-400=100 ≥ 0.
        assert to_hang == ["g2"]

    def test_smallest_first_within_group(self):
        sched = _sched()
        instances = [InstanceCapacity(I0, total_kv_tokens=1000, used_tokens=100_000)]
        sessions = [
            _session("env_a", I0, STATUS_ENV, 500),
            _session("env_b", I0, STATUS_ENV, 400),
            _session("env_c", I0, STATUS_ENV, 300),
        ]
        to_hang, _ = sched.decide(instances, sessions)
        # used = 1200 > 1000. Evict smallest (300) → 900 ≤ 1000 done.
        assert to_hang == ["env_c"]

    def test_env_token_weight_reserves_less_for_env(self):
        instances = [InstanceCapacity(I0, total_kv_tokens=1000, used_tokens=0)]
        sessions = [
            _session("e1", I0, STATUS_ENV, 600),
            _session("e2", I0, STATUS_ENV, 600),
        ]
        assert _sched(env_token_weight=1.0).decide(instances, sessions)[0] == ["e1"]
        assert _sched(env_token_weight=0.5).decide(instances, sessions)[0] == []


class TestContinue:
    def test_continue_when_capacity_frees(self):
        sched = _sched()
        # One running gen (200) + one hung (300). Capacity 1000, used small.
        instances = [InstanceCapacity(I0, total_kv_tokens=1000, used_tokens=200)]
        sessions = [
            _session("run", I0, STATUS_GENERATE, 200),
            _session("hung", I0, STATUS_ENV, 300, hang_state=SESSION_HUNG),
        ]
        to_hang, to_continue = sched.decide(instances, sessions)
        assert to_hang == []
        assert to_continue == [("hung", I0)]

    def test_no_continue_when_still_full(self):
        sched = _sched()
        # The running generation fills the instance, so the hung session cannot continue.
        instances = [InstanceCapacity(I0, total_kv_tokens=500, used_tokens=100_000)]
        sessions = [
            _session("run", I0, STATUS_GENERATE, 500),
            _session("hung", I0, STATUS_ENV, 300, hang_state=SESSION_HUNG),
        ]
        _, to_continue = sched.decide(instances, sessions)
        assert to_continue == []

    def test_continue_smallest_first_bfd(self):
        sched = _sched()
        # Room for 350 tokens. Two hung: 200 and 300 → only the 200 fits.
        instances = [InstanceCapacity(I0, total_kv_tokens=350, used_tokens=0)]
        sessions = [
            _session("hung_big", I0, STATUS_ENV, 300, hang_state=SESSION_HUNG),
            _session("hung_small", I0, STATUS_ENV, 200, hang_state=SESSION_HUNG),
        ]
        _, to_continue = sched.decide(instances, sessions)
        assert to_continue == [("hung_small", I0)]

    def test_hung_only_continues_on_its_pinned_instance(self):
        sched = _sched()
        # I0 is full, while I1 has room. A session hung on I0 must not continue via I1
        # in bucketed scope (per-instance readmission, the default).
        instances = [
            InstanceCapacity(I0, total_kv_tokens=100, used_tokens=100_000),
            InstanceCapacity(I1, total_kv_tokens=10_000, used_tokens=0),
        ]
        sessions = [_session("hung", I0, STATUS_ENV, 300, hang_state=SESSION_HUNG)]
        _, to_continue = sched.decide(instances, sessions)
        assert to_continue == []


class TestContinueGlobalBfd:
    """Global scope: continue uses global BFD and may relocate the session."""

    def test_hung_relocates_to_emptiest_instance(self):
        # I0 is full, while I1 has room. In global scope
        # the session is readmitted onto I1 and pinned there.
        sched = _sched(global_scope=True)
        instances = [
            InstanceCapacity(I0, total_kv_tokens=100, used_tokens=100_000),
            InstanceCapacity(I1, total_kv_tokens=10_000, used_tokens=0),
        ]
        sessions = [_session("hung", I0, STATUS_ENV, 300, hang_state=SESSION_HUNG)]
        _, to_continue = sched.decide(instances, sessions)
        assert to_continue == [("hung", I1)]

    def test_bfd_largest_to_emptiest(self):
        # Two hung sessions, two instances with different room. BFD places the
        # largest onto the emptiest instance first.
        sched = _sched(global_scope=True)
        instances = [
            InstanceCapacity(I0, total_kv_tokens=400, used_tokens=0),
            InstanceCapacity(I1, total_kv_tokens=1000, used_tokens=0),
        ]
        sessions = [
            _session("big", I0, STATUS_ENV, 500, hang_state=SESSION_HUNG),
            _session("small", I0, STATUS_ENV, 300, hang_state=SESSION_HUNG),
        ]
        _, to_continue = sched.decide(instances, sessions)
        # Best-fit decreasing places both sessions on I1.
        assert ("big", I1) in to_continue
        assert ("small", I1) in to_continue
        assert len(to_continue) == 2

    def test_selects_smallest_first_when_capacity_limited(self):
        # Idle-turn hanging makes selection depend only on token count, so the smaller session wins.
        sched = _sched(global_scope=True)
        instances = [InstanceCapacity(I0, total_kv_tokens=350, used_tokens=0)]
        sessions = [
            _session("big", I0, STATUS_ENV, 300, hang_state=SESSION_HUNG),
            _session("small", I0, STATUS_ENV, 200, hang_state=SESSION_HUNG),
        ]
        _, to_continue = sched.decide(instances, sessions)
        assert to_continue == [("small", I0)]


class TestPinningAndBuffer:
    def test_unpinned_sessions_ignored(self):
        sched = _sched()
        instances = [InstanceCapacity(I0, total_kv_tokens=100, used_tokens=100_000)]
        sessions = [_session("nopin", None, STATUS_GENERATE, 500)]
        to_hang, to_continue = sched.decide(instances, sessions)
        assert to_hang == []
        assert to_continue == []

    def test_buffer_counts_against_capacity(self):
        # buffer_per_session=100, two running sessions → 200 buffer.
        # attributed 700 + buffer 200 = 900 > 800 → hang one.
        sched = _sched(buffer_per_session=100)
        instances = [InstanceCapacity(I0, total_kv_tokens=800, used_tokens=100_000)]
        sessions = [
            _session("g1", I0, STATUS_GENERATE, 400),
            _session("g2", I0, STATUS_GENERATE, 300),
        ]
        to_hang, _ = sched.decide(instances, sessions)
        # Evicting `g2` leaves 300 tokens free after the session buffer.
        assert to_hang == ["g2"]

    def test_min_with_measured_used_tokens(self):
        # attributed = 1000 but engine measured only 400 resident (prefix sharing).
        # effective_used = min(1000, 400) = 400 ≤ 800 → no hang despite big attribution.
        sched = _sched()
        instances = [InstanceCapacity(I0, total_kv_tokens=800, used_tokens=400)]
        sessions = [
            _session("g1", I0, STATUS_GENERATE, 500),
            _session("g2", I0, STATUS_GENERATE, 500),
        ]
        to_hang, _ = sched.decide(instances, sessions)
        assert to_hang == []
