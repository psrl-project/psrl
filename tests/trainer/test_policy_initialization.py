"""Exercise startup ownership with real trainer methods and isolated worker RPCs."""

from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psrl.trainer.ppo import ray_trainer

pytestmark = pytest.mark.cpu_test


def _trainer(monkeypatch, ps_mode, validation_first):
    monkeypatch.setattr(ray_trainer.ray, "get", lambda refs: refs)
    events = []

    def remote(name):
        return SimpleNamespace(remote=Mock(side_effect=lambda *args, **kwargs: events.append((name, args)) or []))

    actor = SimpleNamespace(
        init_model=Mock(side_effect=lambda mode: events.append(("actor_init", mode))),
        execute_all_async=Mock(side_effect=lambda *args: events.append(("actor_rpc", args)) or []),
        load_checkpoint=Mock(side_effect=lambda *args, **kwargs: events.append(("checkpoint", args))),
    )
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.elastic_mode = False
    trainer.is_rollout_mode_in_actor = validation_first
    trainer.config = SimpleNamespace(
        psrl=SimpleNamespace(ps_mode=ps_mode, lmcache={}),
        trainer=SimpleNamespace(del_local_ckpt_after_load=False),
    )
    trainer.validate_wg_list = [object()]
    trainer.init_rollout_servers = Mock(side_effect=lambda *args, **kwargs: events.append(("validate_init", None)))
    trainer.rollout_coordinator = SimpleNamespace(
        sleep=remote("validate_sleep"),
        init_nixl_client=remote("coordinator_nixl"),
        initial_pull_from_ps=remote("initial_gen_pull"),
    )
    trainer._resolve_resume_checkpoint_paths = Mock(return_value=None)
    future = Future()
    future.set_result(None)
    return trainer, actor, events, future


@pytest.mark.parametrize("ps_mode", ["cpu", "cpu_ref"])
@pytest.mark.parametrize("validation_first", [False, True])
def test_cpu_startup_loads_real_weights_before_the_first_generation(monkeypatch, ps_mode, validation_first):
    trainer, actor, events, future = _trainer(monkeypatch, ps_mode, validation_first)
    trainer._initialize_policy_workers({"actor": actor}, [], [], future, Mock())
    assert ("actor_init", "full") in events, "CPU transport cannot initialize the actor with empty weights."
    assert events.index(("actor_rpc", ("ray_push_model", 0))) < events.index(
        ("initial_gen_pull", ("all" if validation_first else "rollout",))
    ), "The parameter server must own checkpoint weights before any rollout pull."
    if validation_first:
        assert events.index(("actor_rpc", ("sleep_for_validation",))) < events.index(("validate_init", None)), (
            "Validation may allocate only after the actor releases shared storage."
        )
    else:
        assert events.index(("validate_sleep", ("validate",))) < events.index(("actor_init", "full")), (
            "Training may allocate only after validation releases shared storage."
        )
    assert not any(name == "actor_rpc" and args[0].startswith("nixl") for name, args in events), (
        "CPU transport must not create or use NIXL clients."
    )


def test_cpu_resume_publishes_checkpoint_version_before_the_first_pull(monkeypatch):
    trainer, actor, events, future = _trainer(monkeypatch, "cpu_ref", True)
    trainer._resolve_resume_checkpoint_paths.return_value = ("checkpoint/global_step_4/actor", None)
    trainer._initialize_policy_workers({"actor": actor}, [], [], future, Mock())
    assert events.index(("checkpoint", ("checkpoint/global_step_4/actor",))) < events.index(
        ("actor_rpc", ("ray_push_model", 4))
    ), "Resume weights must reach PS before validation or rollout starts."
    assert trainer._actor_resume_loaded, "Checkpoint loading must not be repeated while the actor sleeps."


@pytest.mark.parametrize("validation_first", [False, True])
def test_nixl_startup_never_pulls_into_sleeping_actor_storage(monkeypatch, validation_first):
    trainer, actor, events, future = _trainer(monkeypatch, "nixl_cpu", validation_first)
    trainer._initialize_policy_workers({"actor": actor}, [], [], future, Mock())
    trainer._pull_initial_policy_weights()
    pull = ("actor_rpc", ("pull_model",))
    assert (pull in events) is not validation_first, "The sleeping actor must pull only after its next wake."
    if validation_first:
        assert events.index(("actor_rpc", ("nixl_sleep", "meta"))) < events.index(("validate_init", None)), (
            "The actor must release memory before initial validation allocates."
        )


def test_colocated_training_commits_async_rewards_before_claiming_gpu_ownership(monkeypatch):
    from psrl.trainer.ppo.strategies.full_batch import FullBatchStepStrategy

    events = []
    batch = SimpleNamespace(keys=["sample"], partition_id="train", extra_info={})
    monkeypatch.setattr(ray_trainer.ray, "get", lambda ref: ref)
    trainer = SimpleNamespace(
        config=SimpleNamespace(
            psrl=SimpleNamespace(deployment=SimpleNamespace(mode="colocated")),
            reward=SimpleNamespace(launch_reward_fn_async=True),
            trainer=SimpleNamespace(balance_batch=False),
            gen_actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(temperature=1.0)),
        ),
        elastic_mode=True,
        release_trainer_for_wait=Mock(),
        agent_loop_manager=SimpleNamespace(wait_for_training_batch=SimpleNamespace(remote=Mock(return_value=batch))),
        replay_buffer=SimpleNamespace(sample=Mock()),
        reward_manager=SimpleNamespace(
            wait_for_reward_of_requests=SimpleNamespace(
                remote=Mock(side_effect=lambda batch: events.append("reward_ready") or batch)
            ),
        ),
        switch_to_trainer_mode=Mock(side_effect=lambda: events.append("training_owns_gpu")),
        global_steps=1,
        _compute_old_log_prob=Mock(side_effect=RuntimeError("Computation boundary reached.")),
    )
    strategy = object.__new__(FullBatchStepStrategy)
    strategy.trainer = trainer
    with pytest.raises(RuntimeError, match="Computation boundary"):
        strategy.run_step(0, {}, {})
    assert events == ["reward_ready", "training_owns_gpu"], (
        "A trainer lease must never block reward work required by that same training batch."
    )
