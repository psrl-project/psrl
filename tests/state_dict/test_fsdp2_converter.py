"""
Example usage of vLLM to HuggingFace state dict converter.

This example shows how to convert a vLLM model's state dict to HuggingFace format
using the new class-based API.
"""

import os
import time

from psrl.utils.converter.fsdp_converter import convert_fsdp_inplace
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM
from verl.utils.fsdp_utils import (
    apply_fsdp2,
)


def example_with_real_model():
    """Example with a real vLLM model instance, now supports distributed torchrun."""
    import torch.distributed as dist

    # Read distributed info from environment variables
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    psrl_workspace = os.environ.get("PSRL_WORKSPACE", "./psrl_workspace")
    print(f"rank: {rank}, world_size: {world_size}, psrl_workspace: {psrl_workspace}")

    # Initialize torch distributed
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    model = AutoModelForCausalLM.from_pretrained(f"{psrl_workspace}/models/Qwen2.5-0.5B-Instruct")
    fsdp_kwargs = {
        "mesh": init_device_mesh("cuda", mesh_shape=(world_size,)),
    }
    apply_fsdp2(model, fsdp_kwargs, {})
    fsdp_model = model

    # Get the model class
    model_class = type(fsdp_model)
    print(f"[rank{rank}] model_class: {model_class}")
    print(f"[rank{rank}] fsdp_model: {fsdp_model}")

    # Get the state dict
    start_time = time.time()
    fsdp_state_dict = fsdp_model.state_dict()
    end_time = time.time()
    print(f"[rank{rank}] State dict loaded in {end_time - start_time:.2f} seconds")
    for name, param in fsdp_state_dict.items():
        # for name, param in fsdp_model.named_parameters():
        print(f"[rank{rank}] {name}: {param.to_local().shape}, {param.placements}")

    # Convert to HuggingFace format
    from psrl.utils.converter import create_parameter_mapping

    parameter_mapping = create_parameter_mapping("FSDP", model.config)
    hf_state_dict, sharding = convert_fsdp_inplace(parameter_mapping, fsdp_model)

    # Save the converted state dict for each rank
    print("=" * 50)
    for name in hf_state_dict:
        print(f"[rank{rank}] {name}: {hf_state_dict[name].shape}, {hf_state_dict[name].sum()}, {sharding[name]}")
    # torch.save(hf_state_dict, f"converted_model_rank{rank}.pth")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    print("Real model conversion example...")
    example_with_real_model()
