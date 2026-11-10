"""
Verify Megatron tensor parallel sharding without deprecated bridge lookups.
"""

from types import SimpleNamespace

import pytest
import torch
from psrl.utils.converter.megatron_converter import MegatronConverter

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("partition_dim", [-1, 0, 1])
def test_tensor_parallel_sharding_uses_parameter_metadata(partition_dim):
    converter = object.__new__(MegatronConverter)
    converter.mpu = SimpleNamespace(tp_size=4, tp_rank=2, etp_size=1)
    parameter = torch.nn.Parameter(torch.ones(8, 8))
    parameter.tensor_model_parallel = True
    parameter.partition_dim = partition_dim

    sharding = converter.get_sharding_for_param("decoder.layers.0.mlp.linear_fc1.weight", parameter)

    assert dict(sharding.shard_mesh) == {max(partition_dim, 0): 4}, (
        "Tensor parallel sharding must retain the parallel size and normalized partition dimension."
    )
    assert sharding.shard_indices == [(2,)], "Tensor parallel sharding must retain the local rank."
