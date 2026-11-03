"""A fused QKV bias round trips through the concatenated conversion."""

from __future__ import annotations

import torch
from psrl.utils.converter.model_mappings import slice_qkv_proj_megatron
from psrl.utils.converter.param_sync import ConcatenatedQKVSync


def _sync(source: torch.Tensor, *, key: str, vision_head_size: int | None = None) -> ConcatenatedQKVSync:
    return ConcatenatedQKVSync(
        key=key,
        megatron_name="vision_model.decoder.layers.0.self_attention.linear_qkv.bias",
        source_param=source,
        num_heads=4,
        num_kv_heads=2,
        head_size=3,
        tp_size=1,
        vision_head_size=vision_head_size,
    )


def test_a_one_dimensional_fused_qkv_is_not_reshaped_by_a_hidden_size():
    torch.manual_seed(0)
    source = torch.randn(24)
    q, k, v = slice_qkv_proj_megatron(
        fused_param=source,
        num_heads=4,
        num_kv_heads=2,
        head_size=3,
        tp_size=1,
    )
    assert q.shape == (2, 6, 1)
    assert k.shape == (2, 3, 1)
    assert torch.equal(torch.cat((q, k, v), dim=1).reshape(-1), source)


def test_a_visual_bias_is_exposed_as_the_flattened_head_view():
    torch.manual_seed(0)
    source = torch.nn.Parameter(torch.randn(24))
    sync = _sync(source, key="model.visual.blocks.0.attn.qkv.bias", vision_head_size=3)
    state = {sync.key: torch.zeros(8, 3, 1)}

    with torch.no_grad():
        sync.before_push(state)
    assert state[sync.key].shape == (8, 3, 1)
    assert torch.equal(state[sync.key], source.detach().reshape(8, 3, 1))

    updated = torch.arange(24, dtype=torch.float32).reshape(8, 3, 1)
    state[sync.key] = updated
    with torch.no_grad():
        sync.after_pull(state)
    assert torch.equal(source.detach(), updated.reshape(24))


def test_a_plain_bias_survives_the_push_and_pull_sync():
    torch.manual_seed(0)
    source = torch.nn.Parameter(torch.randn(24))
    sync = _sync(source, key="model.layers.0.self_attn.qkv.bias")
    state = {sync.key: torch.zeros(24)}

    with torch.no_grad():
        sync.before_push(state)
    assert torch.equal(state[sync.key], source.detach())

    updated = torch.arange(24, dtype=torch.float32)
    state[sync.key] = updated
    with torch.no_grad():
        sync.after_pull(state)
    assert torch.equal(source.detach(), updated)


def test_a_two_dimensional_weight_still_round_trips():
    torch.manual_seed(0)
    source = torch.nn.Parameter(torch.randn(24, 3))
    sync = ConcatenatedQKVSync(
        key="model.layers.0.self_attn.qkv.weight",
        megatron_name="decoder.layers.0.self_attention.linear_qkv.weight",
        source_param=source,
        num_heads=4,
        num_kv_heads=2,
        head_size=3,
        tp_size=1,
    )
    state = {sync.key: torch.zeros(24, 3)}

    with torch.no_grad():
        sync.before_push(state)
    assert torch.equal(state[sync.key], source.detach())

    updated = torch.arange(72, dtype=torch.float32).reshape(24, 3)
    state[sync.key] = updated
    with torch.no_grad():
        sync.after_pull(state)
    assert torch.equal(source.detach(), updated)
