"""
Verify direct weight arenas through native FSDP2 and vLLM execution.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_fsdp2_direct_arena(gpu_tensor_parallel_size):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={gpu_tensor_parallel_size}",
            str(Path(__file__).with_name("fsdp2_weight_arena_gpu_smoke.py")),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"FSDP2 arena verification failed. {result.stdout} {result.stderr}"


@pytest.mark.parametrize("cudagraph", [False, True])
def test_vllm_direct_arena_matches_baseline(gpu_model_path, tmp_path, cudagraph):
    from torch_memory_saver import configure_subprocess

    generations = {}
    for mode in ("baseline", "direct"):
        output = tmp_path / f"{mode}.json"
        command = [
            sys.executable,
            str(Path(__file__).with_name("vllm_direct_arena_gpu_smoke.py")),
            "--model",
            gpu_model_path,
            "--mode",
            mode,
            "--output",
            str(output),
        ]
        if cudagraph:
            command.append("--cudagraph")
        if mode == "direct":
            command.extend(["--tms-cycles", "2"])
        with configure_subprocess():
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,
                env={
                    **os.environ,
                    "LD_LIBRARY_PATH": os.pathsep.join(
                        [str(Path(sys.prefix) / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]
                    ),
                    "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
                    "PSRL_VLLM_WEIGHT_ARENA": "1",
                    "PSRL_VLLM_PATCHES": "TMS:GRAPH" if cudagraph else "TMS",
                },
            )
        assert result.returncode == 0, f"vLLM arena verification failed. {result.stdout} {result.stderr}"
        generations[mode] = json.loads(output.read_text())

    expected = generations["baseline"]["generations"][0]
    for actual in generations["direct"]["generations"]:
        assert actual["token_ids"] == expected["token_ids"], "Direct loading must preserve generated tokens."
        assert actual["selected_logprobs"] == pytest.approx(expected["selected_logprobs"], abs=1e-5), (
            "Direct loading and wake cycles must preserve token probabilities."
        )
    assert all(size > 0 for size in generations["direct"]["snapshot_bytes"]), (
        "The arena snapshot must contain weights."
    )
