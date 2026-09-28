# tests/conftest.py
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# Load Ray-dependent modules directly so CPU tests can collect without Ray or Torch.


def _load_module_direct(dotted_name: str, file_path: str) -> object:
    """Load a .py file directly by path and register it under dotted_name."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fake_module(dotted_name: str, **attrs) -> object:
    """Register a minimal module stub for CPU-only test collection."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    mod = types.ModuleType(dotted_name)
    for name, value in attrs.items():
        setattr(mod, name, value)
    sys.modules[dotted_name] = mod
    return mod


_HERE = os.path.dirname(__file__)
_PSRL = os.path.join(_HERE, "../psrl")

# Stub out psrl.utils.logger so staleness_controller can import it on CPU.
if "psrl.utils.logger" not in sys.modules:
    sys.modules["psrl.utils.logger"] = MagicMock()

# NOTE(lhy): Stub agent-loop types so CPU tests do not import Torch during collection.
_gen_utils = _fake_module(
    "psrl.workers.gen.utils",
    RolloutInstanceId=tuple[str, int],
    INVALID_ROLLOUT_INSTANCE_ID=("", -1),
    TokenInput=MagicMock,
    TokenOutput=MagicMock,
    # Also imported by `vllm_async_server`, which a serving-backend test reaches.
    DEFAULT_TIMEOUT=60.0,
    DEFAULT_MAX_CONNECTIONS=2000,
)
RolloutInstanceId = _gen_utils.RolloutInstanceId

# Load staleness_controller directly (avoids ray via ps/__init__.py)
_staleness_controller = _load_module_direct(
    "psrl.workers.ps.staleness_controller",
    os.path.join(_PSRL, "workers/ps/staleness_controller.py"),
)
EntryInfo = _staleness_controller.EntryInfo

# Ray cluster fixtures (used by integration tests only)


@pytest.fixture(scope="session")
def ray_cluster():
    """Start a single-node Ray cluster once for the whole test session."""
    import ray

    already_running = ray.is_initialized()
    if not already_running:
        ray.init(
            num_cpus=4,
            ignore_reinit_error=True,
            include_dashboard=False,
            log_to_driver=False,
        )
    yield
    if not already_running:
        ray.shutdown()


@pytest.fixture(scope="function")
def ray_cluster_fn(ray_cluster):
    """Reuse the session Ray cluster and kill named actors after each test."""
    import warnings

    import ray

    yield
    for actor_info in ray.util.list_named_actors(all_namespaces=True):
        try:
            ray.kill(ray.get_actor(actor_info["name"], namespace=actor_info.get("namespace")))
        except Exception as e:
            warnings.warn(f"Failed to kill Ray actor {actor_info.get('name', '?')}: {e}", stacklevel=2)


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
