import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _run_offload_check(
    init_file: Path,
) -> None:
    import torch
    import torch.distributed as distributed
    from megatron.core import parallel_state
    from nixl._api import nixl_agent, nixl_agent_config
    from psrl.utils.common.patch_utils import apply_tms_patch
    from psrl.utils.memory.optimizer_state import assert_optimizer_state_offloaded
    from torch_memory_saver import torch_memory_saver
    from verl.utils.megatron_utils import load_megatron_optimizer, offload_megatron_optimizer

    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    apply_tms_patch()
    agent = nixl_agent("optimizer-offload-test", nixl_agent_config(False, False))
    distributed.init_process_group(backend="gloo", init_method=init_file.as_uri(), rank=0, world_size=1)
    try:
        parallel_state.initialize_model_parallel()
        with torch_memory_saver.region(enable_cpu_backup=True):
            # Exceed 1 GiB to exercise the large host allocation path.
            parameter = torch.nn.Parameter(torch.full(((1 << 28) + 256,), 0.25, device="cuda"))
            sentinel = torch.arange(1024, device="cuda")
            optimizer = torch.optim.Adam([parameter], lr=0.01, foreach=False)
            wrapped_optimizer = SimpleNamespace(
                optimizer=optimizer,
                shard_fp32_from_float16_groups=[[parameter]],
            )

        for step in (1, 2):
            with torch_memory_saver.region(enable_cpu_backup=True):
                if step > 1:
                    load_megatron_optimizer(wrapped_optimizer)
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                parameter.grad = None
                offload_megatron_optimizer(wrapped_optimizer)

            assert_optimizer_state_offloaded(wrapped_optimizer, stage="before_sleep", rank=0)
            torch.cuda.synchronize()
            torch_memory_saver.pause()
            torch_memory_saver.resume()
            assert_optimizer_state_offloaded(wrapped_optimizer, stage="after_wake", rank=0)

            state = optimizer.state[parameter]
            expected_values = (
                (parameter, 0.25 - 0.01 * step),
                (state["exp_avg"], 1 - 0.9**step),
                (state["exp_avg_sq"], 1 - 0.999**step),
            )
            for tensor, expected in expected_values:
                bounds = torch.stack(torch.aminmax(tensor.detach()))
                torch.testing.assert_close(bounds, torch.full_like(bounds, expected))
            torch.testing.assert_close(sentinel, torch.arange(1024, device="cuda"))
        del agent
    finally:
        parallel_state.destroy_model_parallel()
        distributed.destroy_process_group()


def test_large_optimizer_state_survives_tms_sleep_and_updates(
    tmp_path: Path,
) -> None:
    from omegaconf import OmegaConf
    from psrl.utils.common.tms_env import build_ray_train_worker_tms_env

    env = os.environ.copy()
    worker_env = build_ray_train_worker_tms_env(OmegaConf.create({"tms": {"range": "train"}}))
    assert worker_env is not None, "TMS must be enabled for the optimizer offload test."
    env.update(worker_env)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--init-file", str(tmp_path / "distributed")],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"Optimizer offload or TMS state preservation failed. {result.stdout} {result.stderr}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-file", type=Path, required=True)
    arguments = parser.parse_args()
    _run_offload_check(arguments.init_file)
