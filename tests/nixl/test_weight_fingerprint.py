import importlib.util
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test

_MODULE_PATH = Path(__file__).parents[2] / "psrl/utils/nixl/fingerprint.py"
_SPEC = importlib.util.spec_from_file_location("psrl_weight_fingerprint_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

assert_positive_standard_gamma_mean = _MODULE.assert_positive_standard_gamma_mean
compare_weight_fingerprints = _MODULE.compare_weight_fingerprints
fingerprint_tensor_mapping = _MODULE.fingerprint_tensor_mapping
precision_sensitive_parameter_stats = _MODULE.precision_sensitive_parameter_stats
resolve_weight_fingerprint_options = _MODULE.resolve_weight_fingerprint_options


def _mapping(tensor: torch.Tensor):
    return {("layer.weight", (0,)): tensor}


def test_sampled_fingerprint_is_deterministic_and_detects_sampled_change() -> None:
    tensor = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    before = fingerprint_tensor_mapping(_mapping(tensor), mode="sampled", sample_count=8)
    repeated = fingerprint_tensor_mapping(_mapping(tensor), mode="sampled", sample_count=8)

    assert before["digest"] == repeated["digest"]
    tensor[-1, -1] += 1
    after = fingerprint_tensor_mapping(_mapping(tensor), mode="sampled", sample_count=8)

    comparison = compare_weight_fingerprints(before, after)
    assert comparison["match"] is False
    assert comparison["differing_tensors"] == ["layer.weight|(0,)"]


def test_full_fingerprint_detects_change_outside_sparse_samples() -> None:
    tensor = torch.arange(64, dtype=torch.float32)
    before = fingerprint_tensor_mapping(_mapping(tensor), mode="full", chunk_bytes=7)
    tensor[1] += 1
    after = fingerprint_tensor_mapping(_mapping(tensor), mode="full", chunk_bytes=7)

    assert before["digest"] != after["digest"]


def test_content_digest_uses_logical_values_while_mapping_digest_tracks_layout() -> None:
    contiguous = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    non_contiguous = contiguous.t().contiguous().t()
    assert not non_contiguous.is_contiguous()

    contiguous_fp = fingerprint_tensor_mapping(_mapping(contiguous), mode="full")
    non_contiguous_fp = fingerprint_tensor_mapping(_mapping(non_contiguous), mode="full")
    contiguous_sampled = fingerprint_tensor_mapping(_mapping(contiguous), mode="sampled", sample_count=5)
    non_contiguous_sampled = fingerprint_tensor_mapping(_mapping(non_contiguous), mode="sampled", sample_count=5)

    assert contiguous_fp["digest"] == non_contiguous_fp["digest"]
    assert contiguous_sampled["digest"] == non_contiguous_sampled["digest"]
    assert contiguous_fp["mapping_digest"] != non_contiguous_fp["mapping_digest"]


def test_version_schedule_prefers_full_fingerprint() -> None:
    config = OmegaConf.create(
        {
            "nixl": {
                "weight_verification": {
                    "enable": True,
                    "transfer_chain": True,
                    "trainer_sleep_wake": False,
                    "sample_every_n_versions": 2,
                    "full_every_n_versions": 4,
                    "sample_count_per_tensor": 8,
                    "full_hash_chunk_bytes": 1024,
                    "include_tensor_digests": True,
                    "fail_on_mismatch": False,
                }
            }
        }
    )

    assert resolve_weight_fingerprint_options(config, flow="transfer_chain", model_version=1) is None
    assert resolve_weight_fingerprint_options(config, flow="transfer_chain", model_version=2)["mode"] == "sampled"
    assert resolve_weight_fingerprint_options(config, flow="transfer_chain", model_version=4)["mode"] == "full"
    assert resolve_weight_fingerprint_options(config, flow="trainer_sleep_wake", model_version=4) is None


def test_precision_sensitive_stats_summarize_matching_tensors() -> None:
    state_dict = {
        "model.language_model.layers.0.linear_attn.norm.weight": torch.tensor([0.5, 1.5]),
        "model.language_model.layers.0.linear_attn.A_log": torch.tensor([-2.0, 2.0], dtype=torch.float32),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4),
    }
    stats = precision_sensitive_parameter_stats(state_dict)
    assert stats["linear_attn.norm.weight"]["tensor_count"] == 1
    assert stats["linear_attn.norm.weight"]["mean"] == pytest.approx(1.0)
    assert stats["A_log"]["dtypes"] == ["torch.float32"]
    assert "q_proj" not in stats


def test_non_positive_gamma_mean_is_rejected() -> None:
    stats = {"linear_attn.norm.weight": {"mean": -0.4}}
    with pytest.raises(RuntimeError, match="must not apply Megatron zero-centered-gamma"):
        assert_positive_standard_gamma_mean(stats, rank=0)
    assert_positive_standard_gamma_mean({"linear_attn.norm.weight": {"mean": 0.8}}, rank=0)
