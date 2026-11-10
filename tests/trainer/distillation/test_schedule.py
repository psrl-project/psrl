"""Ensure synchronous OPD publishes teacher targets before the actor update."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psrl.trainer.ppo.strategies import full_batch

pytestmark = pytest.mark.cpu_test


def test_synchronous_opd_waits_for_queue_commit(monkeypatch):
    batch = SimpleNamespace(keys=["sample"], partition_id="train", extra_info={}, teacher_ready=False)

    def commit(data):
        data.teacher_ready = True
        return data

    def update(data, **kwargs):
        assert data.teacher_ready, "Synchronous OPD must commit targets before actor updates."
        return data

    manager = SimpleNamespace(wait_for_reward_of_requests=SimpleNamespace(remote=Mock(side_effect=commit)))
    trainer = SimpleNamespace(
        elastic_mode=False,
        global_steps=1,
        use_reference_policy=False,
        use_critic=False,
        config=SimpleNamespace(
            trainer=SimpleNamespace(balance_batch=False, critic_warmup=0),
            reward=SimpleNamespace(launch_reward_fn_async=False),
            distillation=SimpleNamespace(enabled=True),
            gen_actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=1.0)),
        ),
        agent_loop_manager=SimpleNamespace(wait_for_training_batch=SimpleNamespace(remote=Mock(return_value=batch))),
        reward_manager=manager,
        replay_buffer=SimpleNamespace(sample=Mock()),
        switch_to_trainer_mode=Mock(),
        _compute_old_log_prob=Mock(return_value=batch),
        _compute_advantage=Mock(return_value=batch),
        _update_actor=Mock(side_effect=update),
    )
    monkeypatch.setattr(full_batch.ray, "get", lambda value: value)
    strategy = full_batch.FullBatchStepStrategy(trainer)
    monkeypatch.setattr(strategy, "_run_ckpt_and_validate", Mock())
    strategy.run_step(0, {}, {})
    manager.wait_for_reward_of_requests.remote.assert_called_once_with(batch)
    trainer._update_actor.assert_called_once()
