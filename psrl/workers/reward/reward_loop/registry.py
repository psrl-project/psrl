import asyncio
import inspect
from collections.abc import Callable
from functools import partial
from typing import Any, cast

from omegaconf import DictConfig
from verl.workers.config.reward import RewardManagerConfig

from psrl.utils.reward_score import default_compute_score_async
from psrl.workers.reward.gen_reward_function import get_gen_reward_function_cls
from psrl.workers.reward.reward_loop.base import RawRewardFn, RewardLoopBase
from psrl.workers.reward.reward_protocol import RewardModelRuntimeInfo

__all__ = ["register", "get_reward_loop_cls", "load_reward_loop"]

REWARD_LOOP: dict[str, type[RewardLoopBase]] = {}


def register(name: str) -> Callable[[type[RewardLoopBase]], type[RewardLoopBase]]:
    """Decorator to register a reward loop class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop.
    """

    def decorator(cls: type[RewardLoopBase]) -> type[RewardLoopBase]:
        if name in REWARD_LOOP and REWARD_LOOP[name] != cls:
            raise ValueError(f"Reward loop {name!r} has already been registered: {REWARD_LOOP[name]} vs {cls}.")
        REWARD_LOOP[name] = cls
        return cls

    return decorator


def get_reward_loop_cls(name: str) -> type[RewardLoopBase]:
    """Get the reward loop class with a given name.

    Args:
        name: `(str)`
            The name of the reward loop.

    Returns:
        `(type)`: The reward loop class.
    """
    if name not in REWARD_LOOP:
        raise ValueError(f"Unknown reward loop: {name!r}.")
    return REWARD_LOOP[name]


def get_custom_reward_fn(reward_fn_config: DictConfig) -> RawRewardFn | None:
    """Load and return a custom reward function from external file.

    Dynamically imports a reward function from a specified file path and wraps
    it with additional keyword arguments from the configuration.

    Args:
        reward_fn_config (DictConfig): Configuration dictionary containing custom_reward_function
                                      settings with 'path', 'name', and 'reward_kwargs' fields.

    Returns:
        callable or None: Wrapped reward function with merged kwargs, or None
                         if no custom reward function is configured.

    Raises:
        FileNotFoundError: If the specified reward function file doesn't exist.
        RuntimeError: If there's an error loading the module from file.
        AttributeError: If the specified function name isn't found in the module.
    """

    module_path = reward_fn_config.get("path")
    if not module_path:
        return None

    fn_name = reward_fn_config.get("name")
    assert fn_name is not None, "Reward function name is required."

    from verl.utils.import_utils import load_extern_object

    raw_fn = load_extern_object(module_path=module_path, object_name=fn_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    if not inspect.iscoroutinefunction(raw_fn):
        return partial(_call_with_kwargs, raw_fn, reward_kwargs)
    else:
        return partial(_call_with_kwargs_async, raw_fn, reward_kwargs)


def load_reward_loop(
    config: DictConfig,
    tokenizer: Any,
    reward_loop_cfg: RewardManagerConfig,
    reward_fn_config: DictConfig,
    reward_model_manager: RewardModelRuntimeInfo | None = None,
    **reward_kwargs: Any,
) -> RewardLoopBase:
    """Load the reward loop based on the configuration.

    Args:
        config: `(DictConfig)`
            The configuration for the reward loop.
        tokenizer: `(Any)`
            The tokenizer for the input.
        reward_loop_cfg: `(RewardManagerConfig)`
            The configuration that selects the reward loop class.
        reward_fn_config: `(DictConfig)`
            The configuration for the reward function.
        reward_model_manager: `(RewardModelRuntimeInfo | None)`
            The reward model runtime configuration, required by generative reward loops.
        **reward_kwargs: `(Any)`
            Additional keyword arguments for the reward loop.
    Returns:
        `(RewardLoopBase)`: The reward loop instance.
    """
    reward_loop_cls: type[RewardLoopBase]
    if reward_loop_cfg.source == "register":
        reward_loop_cls = get_reward_loop_cls(reward_loop_cfg.name)
    elif reward_loop_cfg.source == "importlib":
        from verl.trainer.config.config import ModuleConfig
        from verl.utils.import_utils import load_extern_object

        module_cfg: ModuleConfig | None = reward_loop_cfg.module
        assert module_cfg is not None and module_cfg.path is not None, (
            f"Module path is required when {reward_loop_cfg.source=}, but got {module_cfg=}."
        )
        reward_loop_cls_name = reward_loop_cfg.name
        reward_loop_cls = cast(
            "type[RewardLoopBase]",
            load_extern_object(module_path=module_cfg.path, object_name=reward_loop_cls_name),
        )

    if reward_loop_cfg.source == "register" and reward_loop_cfg.name == "gen":
        gen_reward_function_cls = get_gen_reward_function_cls(reward_fn_config.name)
        return reward_loop_cls(
            config=config,
            tokenizer=tokenizer,
            reward_model_manager=reward_model_manager,
            reward_function=gen_reward_function_cls(),
            **reward_kwargs,
        )
    if reward_loop_cfg.source == "register" and reward_loop_cfg.name == "opd":
        return reward_loop_cls(
            config=config,
            tokenizer=tokenizer,
            reward_model_manager=reward_model_manager,
            **reward_kwargs,
        )

    compute_score = get_custom_reward_fn(reward_fn_config)
    final_compute_score = compute_score

    default_compute_score_ = default_compute_score_async

    if compute_score is None:
        sandbox_config = reward_kwargs.get("sandbox_fusion", None)
        sandbox_url = sandbox_config.get("url") if sandbox_config else None
        memory_limit_mb = sandbox_config.get("memory_limit_mb", 1024) if sandbox_config else 1024
        if sandbox_url:
            # The semaphore must be created in the event loop that uses it.
            max_concurrent = reward_kwargs.get("max_concurrent", 64)
            _concurrent_semaphore = asyncio.Semaphore(max_concurrent)
            final_compute_score = partial(
                default_compute_score_,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=_concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
            )
        else:
            final_compute_score = default_compute_score_

    return reward_loop_cls(
        config=config,
        tokenizer=tokenizer,
        compute_score=final_compute_score,
        **reward_kwargs,
    )


def _call_with_kwargs(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return raw_fn(*args, **merged_kwargs)


async def _call_with_kwargs_async(raw_fn, extra_kwargs, *args, **kwargs):
    """Calls `raw_fn` by merging `extra_kwargs` into call-time `kwargs`, with `extra_kwargs` taking precedence.

    This function is used to merge additional keyword arguments with the original function's arguments.
    """
    merged_kwargs = {**kwargs, **extra_kwargs}
    return await raw_fn(*args, **merged_kwargs)
