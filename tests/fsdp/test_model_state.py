"""
Verify checkpoint loading and NIXL parameter conversion for both FSDP backends.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _assert_converted_shards(state, shardings, reference):
    import torch

    assert state, "Converted parameters must be present."
    assert state.keys() == shardings.keys(), "Every converted parameter must have sharding metadata."
    for name, tensor in state.items():
        expected = reference[name]
        sharding = shardings[name]
        shards = sharding.get_local_sharded_tensors(tensor)
        for coordinates, local in zip(sharding.shard_indices, shards, strict=True):
            slices = [slice(None)] * expected.ndim
            for (dimension, count), coordinate in zip(sharding.shard_mesh.items(), coordinates, strict=True):
                assert expected.shape[dimension] % count == 0, f"Invalid global shard shape for {name!r}."
                length = expected.shape[dimension] // count
                slices[dimension] = slice(coordinate * length, (coordinate + 1) * length)
            torch.testing.assert_close(local.detach().cpu(), expected[tuple(slices)])


def _run_model_check(model_path, strategy):
    import torch
    import torch.distributed as distributed
    from psrl.utils.converter import create_parameter_mapping
    from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace
    from psrl.utils.converter.hf_converter import convert_hf_inplace
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
    from transformers import AutoModelForCausalLM
    from verl.utils.fsdp_utils import (
        apply_fsdp2,
        fsdp2_load_full_state_dict,
        get_fsdp_wrap_policy,
        get_init_weight_context_manager,
    )

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init_process_group(backend="nccl")
    try:
        mesh = init_device_mesh("cuda", (distributed.get_world_size(),))
        reference = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, local_files_only=True)
        config = reference.config
        full_state = reference.state_dict() if distributed.get_rank() == 0 else {}
        if strategy == "fsdp":
            model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
            if distributed.get_rank() == 0:
                model.load_state_dict(full_state)
            model = FullyShardedDataParallel(
                model,
                auto_wrap_policy=get_fsdp_wrap_policy(model),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=torch.cuda.current_device(),
                device_mesh=mesh,
                sync_module_states=True,
            )
        else:
            with get_init_weight_context_manager(use_meta_tensor=True, mesh=mesh)():
                model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
            apply_fsdp2(model, {"mesh": mesh}, {})
            fsdp2_load_full_state_dict(model, full_state, device_mesh=mesh)

        state, shardings = convert_fsdp_inplace(create_parameter_mapping("FSDP", config), model, strategy)
        expected, _ = convert_hf_inplace(create_parameter_mapping("HF", config), reference)
        _assert_converted_shards(state, shardings, expected)

        restored = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        if distributed.get_rank() == 0:
            checkpoint = reference.state_dict()
            assert restored.keys() == checkpoint.keys(), "The reconstructed checkpoint must preserve all keys."
            for name, tensor in restored.items():
                torch.testing.assert_close(tensor, checkpoint[name])
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_checkpoint_and_converter(gpu_model_path, gpu_tensor_parallel_size, strategy):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpu_tensor_parallel_size}",
            str(Path(__file__).resolve()),
            "--model",
            gpu_model_path,
            "--strategy",
            strategy,
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"FSDP loading or conversion failed. {result.stdout} {result.stderr}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", choices=("fsdp", "fsdp2"), required=True)
    arguments = parser.parse_args()
    _run_model_check(arguments.model, arguments.strategy)
