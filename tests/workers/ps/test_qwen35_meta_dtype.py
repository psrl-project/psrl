import pytest
import torch
from psrl.utils.converter.model_dtypes import fix_meta_model_dtypes
from torch import nn

pytestmark = pytest.mark.cpu_test


class Qwen3_5TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(4, dtype=torch.bfloat16))
        self.proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)


def test_qwen35_a_log_is_restored_to_fp32_on_ps_meta_model():
    model = Qwen3_5TestModel()

    fixed_count = fix_meta_model_dtypes(model)

    assert fixed_count == 1
    assert model.A_log.dtype == torch.float32
    assert model.proj.weight.dtype == torch.bfloat16
