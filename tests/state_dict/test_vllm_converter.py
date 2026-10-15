"""
Verify the vLLM weight layout and Hugging Face conversion on real GPU workers.
"""

import argparse
import os
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _inspect_model(model):
    import torch
    from psrl.utils.converter.vllm_converter import convert_vllm_inplace
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm_patches.interfaces import supports_weight_layout
    from vllm_patches.weight_layout import WeightLayoutPlan

    assert supports_weight_layout(model), "The model must expose its weight layout."
    plan = model.build_weight_layout()
    assert isinstance(plan, WeightLayoutPlan), "The model must return a weight layout plan."
    resolved = plan.flatten()
    for name in model.state_dict():
        if "qkv_proj" in name or "gate_up_proj" in name:
            module = model.get_submodule(name.rsplit(".", 1)[0])
            assert resolved.matches_rules(name, module), f"Missing layout rules for packed parameter {name!r}."

    state, shardings = convert_vllm_inplace(model, tp_rank=get_tensor_model_parallel_rank())
    assert state, "Converted parameters must be present."
    assert state.keys() == shardings.keys(), "Every converted parameter must have sharding metadata."
    for name, tensor in state.items():
        assert isinstance(tensor, torch.Tensor), f"Expected a tensor for {name!r}."
        assert shardings[name].shard_indices, f"Expected local shards for {name!r}."
    for suffix in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "gate_proj.weight", "up_proj.weight"):
        assert any(key.endswith(suffix) for key in state), f"Missing split parameter {suffix!r}."
    assert not any("qkv_proj" in key or "gate_up_proj" in key for key in state), (
        "Packed projections must be decomposed into Hugging Face parameters."
    )
    return len(state)


def _run_gpu_checks(model_path, tensor_parallel_size):
    from vllm import LLM

    engine = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="mp",
        enforce_eager=True,
        max_model_len=128,
        max_num_seqs=1,
        gpu_memory_utilization=0.25,
        seed=0,
    )
    parameter_counts = engine.apply_model(_inspect_model)
    assert len(parameter_counts) == tensor_parallel_size, "Every tensor parallel rank must be inspected."
    assert len(set(parameter_counts)) == 1, "Tensor parallel ranks must expose the same parameter count."


def test_real_vllm_weight_layout(gpu_model_path, gpu_tensor_parallel_size):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.state_dict.test_vllm_converter",
            "--model",
            gpu_model_path,
            "--tp",
            str(gpu_tensor_parallel_size),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
    )
    assert result.returncode == 0, f"vLLM worker conversion failed. {result.stdout} {result.stderr}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=2)
    arguments = parser.parse_args()
    from tests.state_dict.test_vllm_converter import _run_gpu_checks

    _run_gpu_checks(arguments.model, arguments.tp)
