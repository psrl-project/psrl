import os
from uuid import uuid4

import pytest
from psrl.workers.gen.utils import RolloutInstanceId
from psrl.workers.ps.staleness_controller import EntryInfo


def pytest_addoption(parser):
    parser.addoption("--run-gpu", action="store_true", help="Run tests that allocate GPU resources.")
    parser.addoption("--gpu-model-path", help="Local model checkpoint for GPU integration tests.")
    parser.addoption("--gpu-tp", type=int, default=2, help="Tensor parallel size for GPU converter tests.")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip = pytest.mark.skip(reason="GPU execution requires --run-gpu.")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def gpu_model_path(pytestconfig):
    from pathlib import Path

    model_path = pytestconfig.getoption("--gpu-model-path")
    if model_path is None or not Path(model_path).is_dir():
        pytest.fail("GPU model tests require --gpu-model-path pointing to a local checkpoint.")
    return str(Path(model_path).resolve())


@pytest.fixture(scope="session")
def gpu_tensor_parallel_size(pytestconfig):
    size = pytestconfig.getoption("--gpu-tp")
    if size < 1:
        pytest.fail("--gpu-tp must be positive.")
    return size


# Ray cluster fixtures (used by integration tests only)


@pytest.fixture(scope="session")
def ray_cluster(tmp_path_factory, pytestconfig):
    """Start a single-node Ray cluster once for the whole test session."""
    import ray

    already_running = ray.is_initialized()
    if not already_running:
        init_kwargs = {
            "ignore_reinit_error": True,
            "include_dashboard": False,
            "log_to_driver": False,
            "namespace": f"psrl-tests-{uuid4().hex}",
        }
        ray_address = os.environ.get("RAY_ADDRESS")
        if ray_address:
            init_kwargs["address"] = ray_address
        else:
            init_kwargs["address"] = "local"
            if pytestconfig.getoption("--run-gpu"):
                import torch

                init_kwargs["num_gpus"] = torch.cuda.device_count()
                init_kwargs["num_cpus"] = 16
            else:
                init_kwargs["num_gpus"] = 0
                init_kwargs["num_cpus"] = 4
            init_kwargs["_temp_dir"] = str(tmp_path_factory.mktemp("ray"))
        ray.init(**init_kwargs)
    yield
    if not already_running:
        ray.shutdown()


# Shared data helpers (cpu_test safe)


@pytest.fixture
def dummy_rollout_instance_id() -> RolloutInstanceId:
    """Return a `tuple[str, int]` because `RolloutInstanceId` is a type alias."""
    return ("worker", 0)


@pytest.fixture
def dummy_entry_info(dummy_rollout_instance_id) -> EntryInfo:
    return EntryInfo(
        rollout_instance_id=dummy_rollout_instance_id,
        prompt_id=42,
        request_idx=0,
        model_version=1,
    )
