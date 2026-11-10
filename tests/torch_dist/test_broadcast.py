"""
Verify subgroup broadcasts leave nonmember ranks unchanged.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu_test, pytest.mark.integration]


def _run_broadcast():
    import torch
    import torch.distributed as distributed

    distributed.init_process_group(backend="gloo")
    try:
        members = [0, 2, 3]
        group = distributed.new_group(ranks=members)
        rank = distributed.get_rank()
        version = torch.tensor([22 if rank == 0 else 0], dtype=torch.long)
        if rank in members:
            distributed.broadcast(version, src=0, group=group)
            assert version.item() == 22, "Every subgroup member must receive the source version."
        else:
            assert version.item() == 0, "Nonmembers must retain their local version."
        distributed.barrier()
    finally:
        distributed.destroy_process_group()


def test_subgroup_broadcast():
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            str(Path(__file__).resolve()),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"Subgroup broadcast failed. {result.stdout} {result.stderr}"


if __name__ == "__main__":
    _run_broadcast()
