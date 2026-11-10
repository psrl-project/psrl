"""
Verify collectives across native vLLM NCCL suspend and resume cycles.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _run_collective_check():
    import torch
    import torch.distributed as distributed
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.distributed.device_communicators.nccl_suspend import resume_nccl_comms, suspend_nccl_comms

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    config = VllmConfig(
        parallel_config=ParallelConfig(tensor_parallel_size=world_size, disable_custom_all_reduce=True)
    )
    with set_current_vllm_config(config):
        parallel_state.init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank)
        try:
            parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
            group = parallel_state.get_tp_group()
            communicator = group.device_communicator.pynccl_comm
            assert communicator.available and not communicator.disabled, "Native PyNccl must be active."
            for cycle in range(3):
                value = torch.full((1024,), float(rank + cycle + 1), device=f"cuda:{local_rank}")
                actual = communicator.all_reduce(value)
                expected = world_size * (world_size - 1) / 2 + world_size * (cycle + 1)
                torch.testing.assert_close(actual, torch.full_like(value, expected))
                torch.cuda.synchronize()
                if cycle == 2:
                    break
                distributed.barrier(group=group.cpu_group)
                suspended = suspend_nccl_comms()
                assert suspended["n"] == suspended["total"] > 0, f"Suspend failed: {suspended!r}."
                resumed = resume_nccl_comms()
                assert resumed["n"] == resumed["total"] > 0, f"Resume failed: {resumed!r}."
        finally:
            parallel_state.cleanup_dist_env_and_memory()


def test_collectives_survive_nccl_suspend_resume():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"NCCL suspend/resume verification failed. {result.stdout} {result.stderr}"


if __name__ == "__main__":
    _run_collective_check()
