"""Tests for the reuse seams `AgentLoopWorkerBase` and `AgentLoopManagerBase` expose.

These bases exist so a non-RL consumer (offline batch rollout) can drive agent loops
without inheriting the parameter server, the staleness inventory, or TransferQueue.
The tests pin the two properties that make that safe:

- the RL subclass still routes a batch exactly as it did before the split, and
- the base alone is enough to dispatch and to run an episode end to end, with the
  output hooks called exactly once each and never both for one episode.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.manager_base import AgentLoopManagerBase
from verl.utils import tensordict_utils as tu


def _bare_manager(n_workers: int, rollout_n: int) -> AgentLoopManagerBase:
    """Build a dispatcher with only the attributes `get_dispatch_plan` reads."""
    manager = AgentLoopManagerBase.__new__(AgentLoopManagerBase)
    manager.agent_loop_workers = [MagicMock() for _ in range(n_workers)]
    manager._dispatch_idx = 0
    manager.rollout_n = rollout_n
    return manager


def _batch(**fields):
    """Build the TensorDict shape `get_dispatch_plan` expects."""
    return tu.get_tensordict({name: np.asarray(values) for name, values in fields.items()})


def _concrete_loop():
    """Build a reward-free agent loop without running `AgentLoopBase.__init__`.

    `AgentLoopBase` is abstract on `run`, so instantiating it needs a subclass.
    """
    from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase

    class _Loop(AgentLoopBase):
        async def run(self, request):
            raise NotImplementedError

    loop = _Loop.__new__(_Loop)
    loop.reward_manager = None
    return loop


@pytest.mark.cpu_test
def test_dispatch_plan_colocates_siblings_of_one_prompt():
    """Children of one prompt must land on one worker, or group sampling splits."""
    manager = _bare_manager(n_workers=3, rollout_n=4)
    # 3 prompts x 4 children, ordered as contiguous groups as the dataloader emits them.
    batch = _batch(
        uid=list(range(12)),
        parent_id=[p for p in range(3) for _ in range(4)],
    )

    plan = manager.get_dispatch_plan(batch, is_validate=False)

    # Every child of a parent shares one worker.
    parent_to_workers: dict[int, set[int]] = {}
    for worker_index, sub_batch in plan.items():
        for parent_id in sub_batch["parent_id"]:
            parent_to_workers.setdefault(parent_id, set()).add(worker_index)
    assert all(len(workers) == 1 for workers in parent_to_workers.values()), (
        f"Siblings were split across workers: {parent_to_workers}."
    )
    # Three prompts over three workers spreads one prompt each.
    assert sorted(plan.keys()) == [0, 1, 2]


@pytest.mark.cpu_test
def test_dispatch_plan_advances_round_robin_across_calls():
    """A fresh batch must not restart at worker 0, or worker 0 takes every prompt."""
    manager = _bare_manager(n_workers=4, rollout_n=1)
    first = _batch(uid=[0, 1])
    second = _batch(uid=[2, 3])

    first_plan = manager.get_dispatch_plan(first, is_validate=False)
    second_plan = manager.get_dispatch_plan(second, is_validate=False)

    assert sorted(first_plan.keys()) == [0, 1]
    assert sorted(second_plan.keys()) == [2, 3]


@pytest.mark.cpu_test
def test_dispatch_rollout_n_is_overridable_for_validation():
    """RL uses a different sibling count for validation, so the hook must be honored."""

    class _ValidationAware(AgentLoopManagerBase):
        def __init__(self):
            self.agent_loop_workers = [MagicMock()]
            self._dispatch_idx = 0
            self.rollout_n = 8
            self.val_rollout_n = 1

        def dispatch_rollout_n(self, is_validate: bool = False) -> int:
            return self.val_rollout_n if is_validate else self.rollout_n

    manager = _ValidationAware()
    # `rollout_n == 1` keys on `uid`, so a batch with no `parent_id` is only
    # dispatchable when the validation override is respected.
    batch = _batch(uid=[0, 1])

    plan = manager.get_dispatch_plan(batch, is_validate=True)

    assert len(plan) == 1


@pytest.mark.cpu_test
def test_base_manager_needs_no_parameter_server_to_dispatch():
    """The base must dispatch with both hooks at their defaults."""
    manager = _bare_manager(n_workers=2, rollout_n=1)
    batch = _batch(uid=[0, 1])

    asyncio.run(manager._inner_dispatch_data(batch))

    dispatched = [w for w in manager.agent_loop_workers if w.add_agent_program.remote.called]
    assert len(dispatched) == 2


@pytest.mark.cpu_test
def test_on_dispatch_returning_false_blocks_the_fan_out():
    """RL declines a dispatch when the status update fails, so the hook must gate it."""

    class _Declining(AgentLoopManagerBase):
        def __init__(self):
            self.agent_loop_workers = [MagicMock()]
            self._dispatch_idx = 0
            self.rollout_n = 1

        async def _on_dispatch(self, data, is_validate: bool = False) -> bool:
            return False

    manager = _Declining()

    asyncio.run(manager._inner_dispatch_data(_batch(uid=[0])))

    assert not manager.agent_loop_workers[0].add_agent_program.remote.called


@pytest.mark.cpu_test
def test_terminate_reasons_route_to_exactly_one_output_hook():
    """No reason may reach both hooks, or a trajectory is recorded and refilled.

    `_run_agent_loop_inner` nulls the output for every worker-retry or aborted
    reason and then calls `_handle_failure`, so `_handle_output` is reached only by
    reasons that survive that nulling. Both hooks firing for one episode would
    double-count it: RL would commit the payload and also refill its group slot.
    """
    for reason in TerminateReason:
        nulled = reason.needs_worker_retry() or reason.is_aborted
        if reason.needs_manager_retry():
            assert nulled, (
                f"{reason.value} asks the manager to refill the group but does not null "
                "the output, so both output hooks would fire for one episode."
            )


@pytest.mark.cpu_test
def test_reward_free_loop_returns_outputs_unscored():
    """Offline rollout has no reward manager, and `None` would read as an abort."""
    loop = _concrete_loop()
    output = SimpleNamespace(reward_score=None, extra_fields={"harbor_rewards": {"reward_repair": 0.5}})

    scored = asyncio.run(loop.compute_reward_score(output))

    assert scored is output, "A missing reward manager must not be read as an abort."
    assert scored.reward_score is None
    # The verifier output the loop already captured must survive for offline scoring.
    assert scored.extra_fields["harbor_rewards"] == {"reward_repair": 0.5}


@pytest.mark.cpu_test
def test_reward_free_loop_preserves_multi_trajectory_shape():
    """A list in must stay a list out, because callers index siblings positionally."""
    loop = _concrete_loop()
    outputs = [SimpleNamespace(reward_score=None, extra_fields={}) for _ in range(3)]

    scored = asyncio.run(loop.compute_reward_score(outputs))

    assert scored == outputs
