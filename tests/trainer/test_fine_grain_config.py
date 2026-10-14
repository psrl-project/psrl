import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


def _make_config(
    granularity="none",
    multiplier=1,
    overlap_scope="recompute",
    train_batch_size=64,
    rollout_n=4,
    mini_batch=32,
    micro_batch_per_gpu=2,
    dp_size=4,
):
    """Build a minimal OmegaConf config that `resolve_fine_grain_chunk_size` needs."""
    cfg = OmegaConf.create(
        {
            "psrl": {
                "fine_grain_overlap": {
                    "granularity": granularity,
                    "multiplier": multiplier,
                    "overlap_scope": overlap_scope,
                }
            },
            "data": {"train_batch_size": train_batch_size},
            "gen_actor_rollout_ref": {"rollout": {"n": rollout_n}},
            "train_actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": mini_batch,
                    "ppo_micro_batch_size_per_gpu": micro_batch_per_gpu,
                    "strategy": "fsdp2",
                    "ppo_epochs": 1,
                    "use_dynamic_bsz": False,
                }
            },
        }
    )
    return cfg, dp_size


class TestResolveChunkSize:
    def test_none_granularity_returns_none(self):
        from psrl.utils.config import resolve_fine_grain_chunk_size

        cfg, dp = _make_config(granularity="none")
        gran, chunk_groups = resolve_fine_grain_chunk_size(cfg, dp)
        assert gran == "none"
        assert chunk_groups == 64  # full batch groups

    def test_mini_batch_multiplier_1(self):
        from psrl.utils.config import resolve_fine_grain_chunk_size

        # train_batch_size=64, rollout_n=4, mini_batch=32 => chunk=32*4=128 samples => 128/4=32 groups
        cfg, dp = _make_config(granularity="mini_batch", multiplier=1, train_batch_size=64, rollout_n=4, mini_batch=32)
        gran, chunk_groups = resolve_fine_grain_chunk_size(cfg, dp)
        assert gran == "mini_batch"
        assert chunk_groups == 32

    def test_micro_batch_clamps_to_mini_batch(self):
        from psrl.utils.config import resolve_fine_grain_chunk_size

        # micro*per_gpu*mult > mini => clamp to mini
        # micro_per_gpu=4, dp=4 => micro_samples=16 * mult=10=160 > mini*rollout_n=128 => clamp
        cfg, dp = _make_config(
            granularity="micro_batch", multiplier=10, micro_batch_per_gpu=4, dp_size=4, rollout_n=4, mini_batch=32
        )
        gran, chunk_groups = resolve_fine_grain_chunk_size(cfg, dp)
        assert gran == "mini_batch"

    def test_mini_batch_clamps_to_full_batch(self):
        from psrl.utils.config import resolve_fine_grain_chunk_size

        # mini*mult > full_batch => clamp to full_batch => granularity=none
        cfg, dp = _make_config(
            granularity="mini_batch", multiplier=100, train_batch_size=64, rollout_n=4, mini_batch=32
        )
        gran, chunk_groups = resolve_fine_grain_chunk_size(cfg, dp)
        assert gran == "none"

    def test_multiplier_zero_raises(self):
        from psrl.utils.config import resolve_fine_grain_chunk_size

        cfg, dp = _make_config(granularity="mini_batch", multiplier=0)
        with pytest.raises(ValueError, match="multiplier"):
            resolve_fine_grain_chunk_size(cfg, dp)

    def test_micro_batch_recompute_with_dynamic_bsz_raises(self):
        """micro_batch + recompute + use_dynamic_bsz must raise even without pre_step."""
        from psrl.utils.config import resolve_fine_grain_chunk_size

        cfg = OmegaConf.create(
            {
                "psrl": {
                    "fine_grain_overlap": {
                        "granularity": "micro_batch",
                        "multiplier": 1,
                        "overlap_scope": "recompute",
                    }
                },
                "data": {"train_batch_size": 64},
                "gen_actor_rollout_ref": {"rollout": {"n": 4}},
                "train_actor_rollout_ref": {
                    "actor": {
                        "ppo_mini_batch_size": 32,
                        "ppo_micro_batch_size_per_gpu": None,  # undefined with dynamic_bsz
                        "strategy": "fsdp2",
                        "ppo_epochs": 1,
                        "use_dynamic_bsz": True,
                    }
                },
            }
        )
        with pytest.raises(ValueError, match="dynamic_bsz"):
            resolve_fine_grain_chunk_size(cfg, dp_size=4)


class TestFineGrainElastic:
    def test_mini_batch_with_elastic_raises(self):
        from psrl.utils.config import validate_fine_grain_elastic

        cfg, _ = _make_config(granularity="mini_batch")
        cfg.psrl.deployment = {
            "elastic": {"enable": True},
        }
        with pytest.raises(ValueError, match="elastic.enable"):
            validate_fine_grain_elastic(cfg)

    def test_none_granularity_allows_elastic(self):
        from psrl.utils.config import validate_fine_grain_elastic

        cfg, _ = _make_config(granularity="none")
        cfg.psrl.deployment = {
            "elastic": {"enable": True},
        }
        validate_fine_grain_elastic(cfg)

    def test_mini_batch_without_elastic_is_allowed(self):
        from psrl.utils.config import validate_fine_grain_elastic

        cfg, _ = _make_config(granularity="mini_batch")
        cfg.psrl.deployment = {
            "elastic": {"enable": False},
        }
        validate_fine_grain_elastic(cfg)
