"""CPU-only structural tests for PSRL_EngineTrainWorker."""

import pytest

pytestmark = pytest.mark.cpu_test


def test_engine_train_worker_inherits_both_bases():
    from psrl.workers.train.base_train_worker import PSRL_BaseTrainWorker
    from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker
    from verl.workers.engine_workers import ActorRolloutRefWorker

    assert issubclass(PSRL_EngineTrainWorker, ActorRolloutRefWorker)
    assert issubclass(PSRL_EngineTrainWorker, PSRL_BaseTrainWorker)


def test_engine_train_worker_importable_from_package():
    from psrl.workers.train import PSRL_EngineTrainWorker

    assert PSRL_EngineTrainWorker is not None
