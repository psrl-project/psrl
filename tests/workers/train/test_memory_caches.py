"""Verify derived Megatron tensors cannot survive a storage handoff."""

import sys
from functools import lru_cache
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from psrl.workers.train.engine_train_worker import clear_megatron_memory_caches

pytestmark = pytest.mark.cpu_test


def test_sleep_releases_rotary_workspaces_dummy_gradients_and_global_buffers(monkeypatch):
    @lru_cache
    def forward(length):
        return object()

    @lru_cache
    def workspace(device):
        return object()

    forward(64)
    workspace(0)
    global_buffer = SimpleNamespace(buffer={"attention": object()})
    dummy_gradients = {"wgrad": object()}
    modules = {
        "megatron.core": {"parallel_state": SimpleNamespace(get_global_memory_buffer=lambda: global_buffer)},
        "transformer_engine.pytorch.cpp_extensions.gemm": {"get_cublas_workspace": workspace},
        "transformer_engine.pytorch.module": {"base": SimpleNamespace(_dummy_wgrads=dummy_gradients)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    cublas_clear = Mock()
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", cublas_clear)
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(forward=forward)])

    clear_megatron_memory_caches([model])

    assert forward.cache_info().currsize == 0, "RoPE derived tensors must not retain GPU storage."
    assert workspace.cache_info().currsize == 0, "TE workspaces must not retain GPU storage."
    assert not dummy_gradients and not global_buffer.buffer, "Reusable gradient and attention buffers must release."
    cublas_clear.assert_called_once_with()


def test_weight_pull_restores_multimodal_rotary_frequencies(monkeypatch):
    from psrl.workers.train import engine_train_worker as module

    rotary = SimpleNamespace(inv_freq=torch.zeros(4), forward=lambda: None)
    chunk = SimpleNamespace(modules=lambda: [rotary])
    worker = SimpleNamespace(
        config=SimpleNamespace(actor=SimpleNamespace(strategy="megatron")),
        actor=SimpleNamespace(engine=SimpleNamespace(module=[chunk])),
        _get_non_persistent_buffers_from_ps=lambda: {"rotary_emb.inv_freq": torch.ones(4)},
    )
    monkeypatch.setattr(module, "clear_megatron_memory_caches", Mock())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    module.PSRL_EngineTrainWorker._restore_non_persistent_buffers_from_ps(worker)
    assert torch.equal(rotary.inv_freq, torch.ones(4)), (
        "Multimodal rotary modules must restore frequencies after the same storage handoff as standard RoPE."
    )
