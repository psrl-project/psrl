"""
Verify Megatron Bridge initialization and parameter conversion across parallel layouts.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _run_model_check(model_path, tensor_parallel_size, pipeline_parallel_size, virtual_pipeline_size):
    import torch
    import torch.distributed as distributed
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from psrl.utils.converter import create_parameter_mapping
    from psrl.utils.converter.megatron_converter import convert_megatron_inplace
    from transformers import AutoConfig

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    distributed.init_process_group(backend="nccl")
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            virtual_pipeline_model_parallel_size=virtual_pipeline_size,
        )
        model_parallel_cuda_manual_seed(0)
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        bridge = AutoBridge.from_hf_config(config)
        provider = bridge.to_megatron_provider(load_weights=False)
        provider.apply_overrides_and_finalize(
            dtype=torch.bfloat16,
            overrides={
                "tensor_model_parallel_size": tensor_parallel_size,
                "pipeline_model_parallel_size": pipeline_parallel_size,
                "virtual_pipeline_model_parallel_size": virtual_pipeline_size,
                "sequence_parallel": False,
            },
        )
        modules = provider.provide_distributed_model(wrap_with_ddp=False)
        with torch.no_grad():
            for module in modules:
                for parameter in module.parameters():
                    parameter.fill_(1)

        mapping = create_parameter_mapping("Megatron", config)
        conversion = convert_megatron_inplace(mapping, modules)
        assert conversion.state_dict, "Every rank must expose converted parameters."
        assert conversion.state_dict.keys() == conversion.sharding_dict.keys(), (
            "Every converted parameter must have sharding metadata."
        )
        conversion.sync_plan.before_push(conversion.state_dict)
        for name, tensor in conversion.state_dict.items():
            assert conversion.sharding_dict[name].shard_indices, f"Missing local shards for {name!r}."
            torch.testing.assert_close(tensor, torch.ones_like(tensor))
        conversion.sync_plan.after_push(conversion.state_dict)
        distributed.barrier()
    finally:
        parallel_state.destroy_model_parallel()
        distributed.destroy_process_group()


@pytest.mark.parametrize(("pipeline_parallel_size", "virtual_pipeline_size"), [(1, None), (2, 2)])
def test_megatron(gpu_model_path, gpu_tensor_parallel_size, pipeline_parallel_size, virtual_pipeline_size):
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={gpu_tensor_parallel_size * pipeline_parallel_size}",
        str(Path(__file__).resolve()),
        "--model",
        gpu_model_path,
        "--tp",
        str(gpu_tensor_parallel_size),
        "--pp",
        str(pipeline_parallel_size),
    ]
    if virtual_pipeline_size is not None:
        command.extend(["--vp", str(virtual_pipeline_size)])
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"Megatron initialization or conversion failed. {result.stdout} {result.stderr}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--vp", type=int)
    arguments = parser.parse_args()
    _run_model_check(arguments.model, arguments.tp, arguments.pp, arguments.vp)
