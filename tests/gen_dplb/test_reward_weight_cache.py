from contextlib import nullcontext
from types import SimpleNamespace

import psrl.workers.gen.vllm_extension as extension_module
import pytest
import torch
from vllm.distributed import parallel_state
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
    convert_to_unquantized_kernel_format,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase

pytestmark = pytest.mark.cpu_test


def test_checkpoint_reload_preserves_flashinfer_expert_output(monkeypatch):
    class ExpertMethod(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs):
            pass

        def apply(self, layer, inputs):
            up, gate = torch.nn.functional.linear(inputs, layer.w13_weight[0]).chunk(2, dim=-1)
            return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, layer.w2_weight[0])

        def process_weights_after_loading(self, layer):
            w13, w2 = convert_to_unquantized_kernel_format(
                UnquantizedMoeBackend.FLASHINFER_CUTLASS,
                layer,
                layer.w13_weight,
                layer.w2_weight,
            )
            with torch.no_grad():
                layer.w13_weight.copy_(w13)
                layer.w2_weight.copy_(w2)

    class ExpertModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w13_weight = torch.nn.Parameter(torch.zeros(1, 4, 2), requires_grad=False)
            self.w2_weight = torch.nn.Parameter(torch.zeros(1, 2, 2), requires_grad=False)
            self.moe_config = SimpleNamespace(is_act_and_mul=True)
            self.quant_method = ExpertMethod()

        def load_weights(self, weights):
            loaded = set()
            with torch.no_grad():
                for name, tensor in weights:
                    getattr(self, name).copy_(tensor)
                    loaded.add(name)
            return loaded

    monkeypatch.setattr(extension_module, "set_current_vllm_config", lambda config: nullcontext())
    config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32, quantization=None),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    model = ExpertModel()
    inputs = torch.tensor([[0.4, -0.7]])
    addresses = [param.data_ptr() for param in model.parameters()]
    for scale in (1.0, 0.5):
        gate = torch.tensor([[1.0, 2.0], [-1.0, 0.5]]) * scale
        up = torch.tensor([[0.5, -1.0], [2.0, 1.0]]) * scale
        down = torch.tensor([[1.0, 0.2], [-0.3, 1.0]])
        checkpoint = [("w13_weight", torch.cat((gate, up)).unsqueeze(0)), ("w2_weight", down.unsqueeze(0))]
        expected = (torch.nn.functional.silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T
        loaded = extension_module._load_model_weights_with_vllm_config(model, iter(checkpoint), config)
        torch.testing.assert_close(model.quant_method.apply(model, inputs), expected)
        assert loaded == {"w13_weight", "w2_weight"}, "All expert checkpoint tensors must be loaded."
        assert [param.data_ptr() for param in model.parameters()] == addresses, "Weight storage must remain stable."


def test_publish_node_shared_reward_arena_uses_ray_runtime_context(monkeypatch):
    group = SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: group)

    runtime_context = SimpleNamespace(
        get_job_id=lambda: SimpleNamespace(hex=lambda: "job-id"),
        get_node_id=lambda: "node-id",
    )
    monkeypatch.setattr(extension_module.ray, "get_runtime_context", lambda: runtime_context)

    cache = SimpleNamespace(
        diagnostics=lambda: {
            "backend": "shm",
            "node_id": "node-id",
            "cache_key": "cache-key",
            "role": "reward",
            "mapped_bytes": 128,
            "unique_cache_bytes": 128,
            "inode": 123,
            "private_bytes": 0,
            "pss_bytes": 128,
        }
    )
    from psrl.utils import node_shared_weight_cache

    publish_kwargs = {}

    def fake_publish(*args, **kwargs):
        publish_kwargs.update(kwargs)
        return cache

    monkeypatch.setattr(node_shared_weight_cache, "publish_or_map_weight_arena", fake_publish)

    extension = object.__new__(extension_module.vLLMWorkerExtension)
    extension.model_runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(enable_expert_parallel=False),
        ),
    )
    extension.model_config = SimpleNamespace(dtype=torch.float16, quantization=None)

    result = extension._publish_node_shared_reward_arena(
        arena_handle=object(),
        arena_config={},
        model_id="reward-model",
        checkpoint_fingerprint="fingerprint",
    )

    assert result is cache
    assert publish_kwargs["job_id"] == "job-id"
    assert publish_kwargs["node_id"] == "node-id"
