"""Distributed GPU smoke test for direct FSDP2 weight arena materialization."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from psrl.utils.weight_arena import (
    get_fsdp_param_groups,
    materialize_fsdp2_model_weights_in_arena,
)
from torch import nn
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh


class TinyTiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.projection = nn.Linear(16, 16)
        self.output = nn.Linear(16, 32, bias=False)
        self.output.weight = self.embedding.weight
        self.register_buffer("scale", torch.linspace(0.5, 1.5, 16))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids).mean(dim=1)
        hidden = torch.tanh(self.projection(hidden)) * self.scale
        return self.output(hidden)


def _fsdp_parameters(model: nn.Module) -> list[object]:
    state = model._get_fsdp_state()
    return [fsdp_param for group in get_fsdp_param_groups(state) for fsdp_param in group.fsdp_params]


def _full_state(model: nn.Module) -> dict[str, torch.Tensor]:
    options = StateDictOptions(full_state_dict=True, cpu_offload=True, broadcast_from_rank0=True)
    return get_model_state_dict(model, options=options)


def _load_state_in_arena(model, state, handle):
    handle.prepare_state_dict_load()
    succeeded = False
    try:
        set_model_state_dict(model, state, options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True))
        succeeded = True
    finally:
        handle.finish_state_dict_load(succeeded=succeeded)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    mesh = init_device_mesh("cuda", (world_size,))

    torch.manual_seed(20260806)
    model = TinyTiedModel()
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    baseline = TinyTiedModel()
    baseline.load_state_dict(initial_state)
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    expected_initial = baseline(token_ids).detach()

    model.to_empty(device="meta")
    fully_shard(model, mesh=mesh, reshard_after_forward=False)
    # FSDP2 may materialize module buffers on the target device while parameter
    # shards remain meta. This mirrors rotary_emb.inv_freq in production models.
    model.scale = initial_state["scale"].to(device)
    fsdp_params = _fsdp_parameters(model)
    parameter_ids = tuple(id(fsdp_param.sharded_param) for fsdp_param in fsdp_params)
    cuda_allocated_before = torch.cuda.memory_allocated(device)
    handle = materialize_fsdp2_model_weights_in_arena(
        model,
        device=device,
        max_chunk_bytes=2048,
        alignment_bytes=256,
    )
    cuda_allocated_after = torch.cuda.memory_allocated(device)
    assert parameter_ids == tuple(id(fsdp_param.sharded_param) for fsdp_param in fsdp_params)
    _load_state_in_arena(model, initial_state if rank == 0 else {}, handle)

    actual_initial = model(token_ids.to(device)).detach().cpu()
    torch.testing.assert_close(actual_initial, expected_initial, rtol=1e-5, atol=1e-6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss = model(token_ids.to(device)).float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected_after_step = model(token_ids.to(device)).detach().cpu()
    checkpoint = _full_state(model)

    with torch.no_grad():
        for arena in handle.arenas:
            arena.zero_()
    _load_state_in_arena(model, checkpoint if rank == 0 else {}, handle)
    actual_after_reload = model(token_ids.to(device)).detach().cpu()
    torch.testing.assert_close(actual_after_reload, expected_after_step, rtol=1e-5, atol=1e-6)
    handle.assert_fsdp_bindings_unchanged()

    rank_result = {
        "rank": rank,
        "arena_count": handle.stats.arena_count,
        "binding_count": handle.stats.tensor_binding_count,
        "arena_bytes": handle.stats.total_arena_bytes,
        "cuda_allocation_delta": cuda_allocated_after - cuda_allocated_before,
        "max_abs_initial_error": float((actual_initial - expected_initial).abs().max()),
        "max_abs_reload_error": float((actual_after_reload - expected_after_step).abs().max()),
    }
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, rank_result)
    if rank == 0:
        print(json.dumps({"world_size": world_size, "ranks": gathered}, sort_keys=True))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
