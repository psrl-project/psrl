"""CPU-only: verify main_ppo no longer imports deprecated worker modules."""

import pytest

pytestmark = pytest.mark.cpu_test


def test_deprecated_fsdp_workers_not_imported_by_main_ppo():
    """main_ppo should not cause fsdp_workers/megatron_workers to be imported."""
    import sys

    # Snapshot modules before fresh import
    modules_before = set(sys.modules.keys())

    # Evict main_ppo so it re-executes on import
    for mod in list(sys.modules.keys()):
        if "psrl.trainer.main_ppo" in mod:
            del sys.modules[mod]

    import psrl.trainer.main_ppo  # noqa: F401

    modules_added = set(sys.modules.keys()) - modules_before

    assert not any("fsdp_workers" in m for m in modules_added), (
        f"main_ppo import pulled in fsdp_workers: {[m for m in modules_added if 'fsdp_workers' in m]}"
    )
    assert not any("megatron_workers" in m for m in modules_added), (
        f"main_ppo import pulled in megatron_workers: {[m for m in modules_added if 'megatron_workers' in m]}"
    )


def test_engine_train_worker_referenced_in_task_runner():
    """TaskRunner.add_actor_rollout_worker must reference PSRL_EngineTrainWorker."""
    import inspect

    from psrl.trainer.main_ppo import TaskRunner

    src = inspect.getsource(TaskRunner.add_actor_rollout_worker)
    assert "PSRL_EngineTrainWorker" in src


def test_critic_worker_uses_training_worker():
    """TaskRunner.add_critic_worker must use verl's unified TrainingWorker (not legacy fsdp/megatron workers)."""
    import inspect

    from psrl.trainer.main_ppo import TaskRunner

    src = inspect.getsource(TaskRunner.add_critic_worker)
    assert "TrainingWorker" in src
    assert "fsdp_workers" not in src
    assert "megatron_workers" not in src
