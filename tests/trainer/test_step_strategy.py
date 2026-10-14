from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from psrl.trainer.ppo.strategies import STAGE_META, build_step_strategy

pytestmark = pytest.mark.cpu_test


class TestStageMeta:
    def test_all_stages_tagged(self):
        for stage in ["old_log_prob", "ref_log_prob", "values", "reward"]:
            assert STAGE_META[stage] == "per_sample"
        assert STAGE_META["advantage"] == "batch_coupled"
        assert STAGE_META["update_actor"] == "optimizer_step"
        assert STAGE_META["update_critic"] == "optimizer_step"


class TestBuildStepStrategy:
    def test_none_granularity_returns_full_batch(self):
        from psrl.trainer.ppo.strategies import FullBatchStepStrategy

        trainer = MagicMock()
        strategy = build_step_strategy(None, trainer)
        assert isinstance(strategy, FullBatchStepStrategy)

    def test_explicit_none_returns_full_batch(self):
        from psrl.trainer.ppo.strategies import FullBatchStepStrategy

        trainer = MagicMock()
        cfg = MagicMock()
        cfg.granularity = "none"
        strategy = build_step_strategy(cfg, trainer)
        assert isinstance(strategy, FullBatchStepStrategy)

    def test_fine_grain_rejects_elastic_deployment(self):
        from psrl.trainer.ppo.strategies.fine_grain_overlap import FineGrainOverlapStrategy

        trainer = SimpleNamespace(elastic_mode=True)
        with pytest.raises(ValueError, match="elastic deployments"):
            FineGrainOverlapStrategy(trainer, SimpleNamespace())
