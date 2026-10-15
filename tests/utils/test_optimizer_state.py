from types import SimpleNamespace

import pytest
import torch
from psrl.utils.optimizer_state import (
    assert_optimizer_state_offloaded,
    summarize_optimizer_state_devices,
)

pytestmark = pytest.mark.cpu_test


def test_optimizer_state_summary_accepts_nested_cpu_tensors():
    optimizer = SimpleNamespace(
        state={
            object(): {
                "step": torch.tensor(1),
                "moments": [torch.ones(2), {"exp_avg_sq": torch.zeros(3)}],
            }
        }
    )

    summary = assert_optimizer_state_offloaded(optimizer, stage="before_sleep", rank=0)

    assert summary["tensor_count"] == 3
    assert summary["device_tensor_counts"] == {"cpu": 3}
    assert summary["non_cpu_tensor_count"] == 0


def test_optimizer_state_summary_rejects_non_cpu_tensor():
    optimizer = SimpleNamespace(state={object(): {"exp_avg": torch.empty(4, device="meta")}})

    with pytest.raises(RuntimeError, match="non_cpu_tensor_count=1"):
        assert_optimizer_state_offloaded(optimizer, stage="before_sleep", rank=3)

    summary = summarize_optimizer_state_devices(optimizer)
    assert summary["device_tensor_counts"] == {"meta": 1}
    assert summary["non_cpu_tensors"][0]["path"] == "state[0].exp_avg"


def test_empty_optimizer_state_is_valid_before_first_update():
    summary = assert_optimizer_state_offloaded(
        SimpleNamespace(state={}),
        stage="before_sleep",
        rank=0,
    )

    assert summary == {
        "tensor_count": 0,
        "total_bytes": 0,
        "device_tensor_counts": {},
        "non_cpu_tensor_count": 0,
        "non_cpu_tensors": [],
    }


def test_megatron_wrapped_optimizer_state_and_master_shards_are_checked():
    wrapped_optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(
            state={
                "parameter": {
                    "exp_avg": torch.ones(2),
                    "exp_avg_sq": torch.ones(2),
                }
            }
        ),
        shard_fp32_from_float16_groups=[[torch.ones(3)]],
    )

    summary = assert_optimizer_state_offloaded(wrapped_optimizer, stage="before_sleep", rank=0)

    assert summary["tensor_count"] == 3
    assert summary["device_tensor_counts"] == {"cpu": 3}


def test_megatron_chained_optimizer_reports_nested_non_cpu_state():
    wrapped_optimizer = SimpleNamespace(
        optimizer=SimpleNamespace(state={"parameter": {"exp_avg": torch.empty(2, device="meta")}}),
        shard_fp32_from_float16_groups=[],
    )
    chained_optimizer = SimpleNamespace(chained_optimizers=[wrapped_optimizer])

    with pytest.raises(RuntimeError, match=r"chained_optimizers\[0\]\.state\[0\]\.exp_avg"):
        assert_optimizer_state_offloaded(chained_optimizer, stage="after_wake", rank=2)
