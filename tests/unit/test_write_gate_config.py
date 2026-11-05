"""What the write gate resolves to, and why each topology needs or skips it (v2).

The gate exists exactly where rollouts read the trainer's live parameter bytes. The
verdict is written into the config so a run's dump shows which path it took.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from psrl.utils.config import _resolve_write_gate

pytestmark = pytest.mark.cpu_test


def _config(
    ps_mode: str = "nixl_direct",
    train_dtype: str = "bf16",
    rollout_dtype: str = "bfloat16",
    quantization_scheme: str = "none",
    reader_quantization: bool = False,
    colocate: bool = False,
    decided="auto",
):
    return OmegaConf.create(
        {
            "psrl": {
                "ps_mode": ps_mode,
                "colocate_validate_and_train": colocate,
                "nixl": {
                    "await_write_gate": decided,
                    "publish_train_dtype": train_dtype,
                    "publish_rollout_dtype": rollout_dtype,
                    "publish_quantization": {"scheme": quantization_scheme},
                    "reader_quantization": reader_quantization,
                },
            }
        }
    )


def test_direct_supply_at_matched_precision_enables_the_gate():
    config = _config()
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is True


def test_a_fp8_rollout_reads_a_published_copy_and_skips_the_gate():
    config = _config(quantization_scheme="fp8")
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is False


def test_a_cast_path_reads_a_published_copy_and_skips_the_gate():
    config = _config(train_dtype="fp32", rollout_dtype="bfloat16")
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is False


def test_a_ps_data_plane_reads_ps_memory_and_skips_the_gate():
    config = _config(ps_mode="nixl_cpu")
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is False


def test_colocated_validation_does_not_disable_the_gate_for_a_disaggregated_rollout():
    """`colocate_validate_and_train` only places validation; rollout still reads live
    parameters from another node, which is the topology the old barrier criterion got
    wrong."""
    config = _config(colocate=True)
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is True


def test_the_write_gate_supersedes_the_publish_barrier():
    """Both mechanisms together would invalidate versions the gate already proved safe."""
    config = _config()
    config.psrl.nixl.await_readers_before_publish = True
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is True
    assert config.psrl.nixl.await_readers_before_publish is False


@pytest.mark.parametrize("decided", [True, False])
def test_an_explicit_boolean_is_never_overridden(decided):
    config = _config(decided=decided)
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is decided


def test_reader_side_fp8_quantization_reads_live_bf16_and_enables_the_gate():
    config = _config(quantization_scheme="fp8", reader_quantization=True)
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is True


def test_reader_quantization_needs_both_fp8_and_direct_supply():
    config = _config(ps_mode="nixl_cpu", quantization_scheme="fp8", reader_quantization=True)
    _resolve_write_gate(config)
    assert config.psrl.nixl.await_write_gate is False
