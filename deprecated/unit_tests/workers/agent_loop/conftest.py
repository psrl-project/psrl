"""Pytest configuration: mock heavy dependencies before any test imports.

The ``psrl.workers.agent_loop`` package ``__init__`` and
``psrl.workers.gen.__init__`` both drag in heavy vllm / Ray / verl
dependencies that are unavailable in unit-test environments.  This
conftest loads *only* the two modules under test by path (bypassing
their package ``__init__`` files) and registers them in ``sys.modules``
so the normal ``from psrl.workers.agent_loop.route_strategy import …``
import in the test file resolves without triggering the package init.
"""
import importlib.util as _ilu
import pathlib
import sys
import types


def _fake_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# ---------------------------------------------------------------------------
# Root project path (three levels up from this conftest).
# ---------------------------------------------------------------------------
_ROOT = pathlib.Path(__file__).parent.parent.parent.parent  # …/psrl_agent


# ---------------------------------------------------------------------------
# Stub out vllm (importing it triggers a Ray cluster connection that times
# out in unit-test environments where no Ray GCS is running).
# ---------------------------------------------------------------------------
_fake_module("vllm")
_fake_module("vllm.config", VllmConfig=type("VllmConfig", (), {}))
_fake_module("vllm.v1")
_fake_module("vllm.v1.metrics")
_fake_module(
    "vllm.v1.metrics.loggers",
    StatLoggerBase=type("StatLoggerBase", (), {}),
)
_fake_module(
    "vllm.v1.metrics.stats",
    IterationStats=type("IterationStats", (), {}),
    MultiModalCacheStats=type("MultiModalCacheStats", (), {}),
    SchedulerStats=type("SchedulerStats", (), {}),
)
_fake_module(
    "vllm.sampling_params",
    RequestOutputKind=type("RequestOutputKind", (), {}),
)

# ---------------------------------------------------------------------------
# Stub out verl (only DataProto is used by route_strategy).
# ---------------------------------------------------------------------------
_verl = _fake_module("verl")
_verl.DataProto = type("DataProto", (), {})

# ---------------------------------------------------------------------------
# Stub out ray (no Ray GCS in unit-test environments).
# ---------------------------------------------------------------------------
_ray = _fake_module("ray")


def _ray_remote(*args, **kwargs):
    """Passthrough decorator stub for @ray.remote."""
    def _decorator(cls_or_fn):
        return cls_or_fn
    if args and callable(args[0]):
        return args[0]
    return _decorator


def _ray_method(*args, **kwargs):
    """Passthrough decorator stub for @ray.method."""
    def _decorator(fn):
        return fn
    if args and callable(args[0]):
        return args[0]
    return _decorator


_ray.remote = _ray_remote
_ray.method = _ray_method
_ray.ObjectRef = type("ObjectRef", (), {})

# ---------------------------------------------------------------------------
# Stub out omegaconf.
# ---------------------------------------------------------------------------
_fake_module("omegaconf", DictConfig=type("DictConfig", (), {}))

# ---------------------------------------------------------------------------
# Stub out tensordict.
# ---------------------------------------------------------------------------
_fake_module("tensordict", TensorDict=type("TensorDict", (), {}))

# ---------------------------------------------------------------------------
# Stub psrl.utils.logger so stats_collector can import FileOnlyHandler.
# ---------------------------------------------------------------------------
if "psrl" not in sys.modules:
    sys.modules["psrl"] = types.ModuleType("psrl")
if "psrl.utils" not in sys.modules:
    sys.modules["psrl.utils"] = types.ModuleType("psrl.utils")
_fake_module(
    "psrl.utils.logger",
    FileOnlyHandler=type("FileOnlyHandler", (), {}),
    DualOutputHandler=type("DualOutputHandler", (), {}),
    EventType=type("EventType", (), {}),
    deprecated=lambda msg: (lambda f: f),
    log_dual_events=lambda *a, **kw: None,
)

# ---------------------------------------------------------------------------
# Stub psrl.utils.ray.
# ---------------------------------------------------------------------------
_fake_module("psrl.utils.ray", AsyncBusyPollingRayLock=type("AsyncBusyPollingRayLock", (), {}))

# ---------------------------------------------------------------------------
# Stub psrl.utils.rollout and psrl.utils.rollout.rollout_trace.
# ---------------------------------------------------------------------------
_fake_module("psrl.utils.rollout")
_fake_module("psrl.utils.rollout.rollout_trace", rollout_trace_op=lambda *a, **kw: None)

# ---------------------------------------------------------------------------
# Stub psrl.utils.kv_cache package and load types.py directly by file.
# ---------------------------------------------------------------------------
_fake_module("psrl.utils.kv_cache")
_kv_types_spec = _ilu.spec_from_file_location(
    "psrl.utils.kv_cache.types",
    _ROOT / "psrl" / "utils" / "kv_cache" / "types.py",
)
_kv_types_mod = _ilu.module_from_spec(_kv_types_spec)
sys.modules["psrl.utils.kv_cache.types"] = _kv_types_mod
_kv_types_spec.loader.exec_module(_kv_types_mod)

# ---------------------------------------------------------------------------
# Stub psrl.workers.agent_loop.request_queue.
# ---------------------------------------------------------------------------
_fake_module(
    "psrl.workers.agent_loop.request_queue",
    MultiPriorityRequestQueue=type("MultiPriorityRequestQueue", (), {}),
    PriorityRequestQueue=type("PriorityRequestQueue", (), {}),
    RequestSortIndicator=type("RequestSortIndicator", (), {}),
)

# ---------------------------------------------------------------------------
# Stub psrl.workers.ps.request_status_tracker.
# ---------------------------------------------------------------------------
if "psrl.workers" not in sys.modules:
    sys.modules["psrl.workers"] = types.ModuleType("psrl.workers")
if "psrl.workers.ps" not in sys.modules:
    sys.modules["psrl.workers.ps"] = types.ModuleType("psrl.workers.ps")
_fake_module(
    "psrl.workers.ps.request_status_tracker",
    PSRL_RequestStatus=type("PSRL_RequestStatus", (), {}),
)

# ---------------------------------------------------------------------------
# Load psrl.workers.gen.stats_collector directly by file, bypassing the
# gen package __init__ (which imports vllm_rollout and other heavy modules).
# ---------------------------------------------------------------------------
_stats_spec = _ilu.spec_from_file_location(
    "psrl.workers.gen.stats_collector",
    _ROOT / "psrl" / "workers" / "gen" / "stats_collector.py",
)
_stats_mod = _ilu.module_from_spec(_stats_spec)
sys.modules["psrl.workers.gen.stats_collector"] = _stats_mod
_stats_spec.loader.exec_module(_stats_mod)

# Provide a minimal psrl.workers.gen package stub so submodule lookups work.
if "psrl.workers" not in sys.modules:
    sys.modules["psrl.workers"] = types.ModuleType("psrl.workers")
_gen_pkg = types.ModuleType("psrl.workers.gen")
_gen_pkg.EngineStats = _stats_mod.EngineStats
_gen_pkg.StatCollector = getattr(_stats_mod, "StatCollector", None)
sys.modules["psrl.workers.gen"] = _gen_pkg

# ---------------------------------------------------------------------------
# Load psrl.workers.agent_loop.route_strategy directly by file, bypassing
# the agent_loop package __init__ (which imports manager, router, etc.).
# ---------------------------------------------------------------------------
_rs_spec = _ilu.spec_from_file_location(
    "psrl.workers.agent_loop.route_strategy",
    _ROOT / "psrl" / "workers" / "agent_loop" / "route_strategy.py",
)
_rs_mod = _ilu.module_from_spec(_rs_spec)
sys.modules["psrl.workers.agent_loop.route_strategy"] = _rs_mod

# Provide a minimal psrl.workers.agent_loop package stub.
if "psrl.workers.agent_loop" not in sys.modules:
    _al_pkg = types.ModuleType("psrl.workers.agent_loop")
    sys.modules["psrl.workers.agent_loop"] = _al_pkg

_rs_spec.loader.exec_module(_rs_mod)

# ---------------------------------------------------------------------------
# Load psrl.workers.agent_loop.router directly by file, bypassing the
# agent_loop package __init__ (which imports other heavy modules).
# ---------------------------------------------------------------------------
_router_spec = _ilu.spec_from_file_location(
    "psrl.workers.agent_loop.router",
    _ROOT / "psrl" / "workers" / "agent_loop" / "router.py",
)
_router_mod = _ilu.module_from_spec(_router_spec)
sys.modules["psrl.workers.agent_loop.router"] = _router_mod
_router_spec.loader.exec_module(_router_mod)
