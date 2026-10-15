import importlib.util
import os
import sys
from dataclasses import dataclass, field

from verl.base_config import BaseConfig
from verl.utils.profiler import ProfilerConfig

from .model import HFModelConfig
from .rollout import PoolingConfig, RolloutConfig, SamplingConfig, ServerConfig

__all__ = [
    "SandboxFusionConfig",
    "RewardModelConfig",
    "SingleRewardModelConfig",
    "RewardLoopWorkerConfig",
    "MultiRewardModelConfig",
    "resolve_active_managers",
]


def get_custome_process_fn(file_path, function_name):
    if not file_path:
        return None

    assert function_name is not None, "Custom reward function name is required."
    module_name = f"custom_reward_module_{function_name}"
    module = sys.modules.get(module_name, None)

    if module is None:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Reward process function file '{file_path}' not found.")

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        assert spec is not None, f"Could not load a custom reward module from {file_path!r}."
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            assert spec.loader is not None, f"Custom reward module {module_name!r} has no loader."
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward preprocess function '{function_name}' not found in '{module.__file__}'.")

    print(f"using customized reward function '{function_name}' from '{module.__file__}'")
    raw_fn = getattr(module, function_name)
    return raw_fn


@dataclass
class SandboxFusionConfig(BaseConfig):
    """Configuration for cloud/local sandbox fusion.

    Args:
        url (str | None): Cloud/local function URL for sandbox execution.
        max_concurrent (int): Max concurrent requests allowed to sandbox.
        memory_limit_mb (int): Max memory limit for each sandbox process in MB.
    """

    url: str | None = None
    max_concurrent: int = 64
    memory_limit_mb: int = 1024


@dataclass
class RewardModelConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields

    enable: bool = False
    model_type: str = "discriminative"
    name: str = "sglang"
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    reward_manager: str = "naive"
    launch_reward_fn_async: bool = False
    # Optional cap for concurrently occupied requests per RM instance.
    # None means unlimited (backward-compatible behavior).
    max_concurrent_requests_per_instance: int | None = None

    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.5
    free_cache_engine: bool = True
    tensor_model_parallel_size: int = 2

    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    max_new_tokens: int = 4096

    engine_kwargs: dict = field(default_factory=dict)
    max_num_seqs: int = 1024

    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    input_model_config: HFModelConfig = field(default_factory=HFModelConfig)
    model_config: HFModelConfig = field(default_factory=HFModelConfig)
    server_config: ServerConfig = field(default_factory=ServerConfig)


@dataclass
class SingleRewardModelConfig(BaseConfig):
    """Configuration for a single reward model in the multi-reward model setup.

    Args:
        reward_loop_type (Optional[str]): Type of reward loop (naive, dapo, gen, None).
        reward_fn (Optional[str]): Reward function name (default, None).
        reward_model_name (Optional[str]): Name of the model.
        enable_resource_pool (bool): Whether to enable resource pool.
        n_gpus_per_node (int): Number of GPUs per node.
        num_replicas (int): Number of replicas.
        nnodes (int): Number of nodes.
        rollout_ngpus_per_instance_per_node (int): Number of GPUs per instance per node for rollout.
        rollout_nnodes_per_instance (int): Number of nodes per instance for rollout.
        model (HFModelConfig): Model configuration.
        rollout (RolloutConfig): Rollout configuration.
        sampling_config (SamplingConfig): Sampling configuration.
        pooling_config (PoolingConfig): Pooling configuration.
        sandbox_fusion (SandboxFusionConfig): Sandbox fusion configuration.
    """

    reward_loop_type: str | None = None
    reward_fn: list[str] | None = None
    reward_model_name: str | None = None
    enable_resource_pool: bool = False
    n_gpus_per_node: int = 0
    num_replicas: int = 1
    nnodes: int = 1
    rollout_ngpus_per_instance_per_node: int = 1
    rollout_nnodes_per_instance: int = 1
    model: HFModelConfig = field(default_factory=HFModelConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)
    pooling_config: PoolingConfig = field(default_factory=PoolingConfig)
    sandbox_fusion: SandboxFusionConfig = field(default_factory=SandboxFusionConfig)
    reward_loop_kwargs: dict = field(default_factory=dict)
    # Optional cap for concurrently occupied requests per RM instance.
    # None means unlimited (backward-compatible behavior).
    max_concurrent_requests_per_instance: int | None = None


@dataclass
class RewardLoopWorkerConfig(BaseConfig):
    """Configuration for the reward loop worker pool."""

    num_workers: int = 8
    max_concurrency_per_worker: int = 256
    queue_size_per_worker: int = 4096
    placement: str = "reward_service_ip"
    result_queue_size: int = 65536
    result_drain_batch_size: int = 1024


@dataclass
class MultiRewardModelConfig(BaseConfig):
    """Configuration for multi-reward model setup.

    Args:
        launch_reward_fn_async (bool): Whether to launch reward function asynchronously.
        # reward_models (list[SingleRewardModelConfig]): List of reward model configurations.
        active_managers (list[str]): Ordered list of manager names to activate.
        managers (dict[str, SingleRewardModelConfig]): Named reward manager definitions.
        profiler (ProfilerConfig): Profiler configuration.
    """

    launch_reward_fn_async: bool = False
    # reward_models: list[SingleRewardModelConfig] = field(default_factory=list)
    # profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    active_managers: list[str] = field(default_factory=list)
    managers: dict[str, SingleRewardModelConfig] = field(default_factory=dict)
    reward_loop_worker: RewardLoopWorkerConfig = field(default_factory=RewardLoopWorkerConfig)
    validation_reward_loop_worker: RewardLoopWorkerConfig = field(
        default_factory=lambda: RewardLoopWorkerConfig(num_workers=2)
    )
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)


def resolve_active_managers(reward_config) -> list:
    """Resolve active_managers names to an ordered list of reward model configurations.

    Takes the reward config (either a dataclass instance or an OmegaConf DictConfig)
    and returns the list of manager configs corresponding to the names in active_managers.

    Args:
        reward_config: The reward configuration object containing ``active_managers``
            and ``managers`` fields.

    Returns:
        list: Ordered list of reward model configurations (SingleRewardModelConfig or DictConfig).

    Raises:
        ValueError: If a name in active_managers is not found in managers.
    """
    active = reward_config.active_managers
    managers = reward_config.managers
    resolved = []
    for name in active:
        if name not in managers:
            available = list(managers.keys())
            raise ValueError(
                f"Reward manager '{name}' in active_managers not found in managers. Available: {available}"
            )
        resolved.append(managers[name])
    return resolved
