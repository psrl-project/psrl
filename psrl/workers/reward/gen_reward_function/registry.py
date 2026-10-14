from collections.abc import Callable

from psrl.workers.reward.gen_reward_function.base import GenRewardFunctionBase

GEN_REWARD_FUNCTION_REGISTRY: dict[str, type[GenRewardFunctionBase]] = {}


def gen_reward_func(
    name: str,
) -> Callable[[type[GenRewardFunctionBase]], type[GenRewardFunctionBase]]:
    """Decorator to register a gen reward function class with a given name.

    Args:
        name: `(str)`
            The name of the gen reward function.
    """

    def decorator(cls: type[GenRewardFunctionBase]) -> type[GenRewardFunctionBase]:
        if name in GEN_REWARD_FUNCTION_REGISTRY and GEN_REWARD_FUNCTION_REGISTRY[name] != cls:
            raise ValueError(
                f"gen reward function {name} has already been registered: "
                f"{GEN_REWARD_FUNCTION_REGISTRY[name]} vs {cls}"
            )
        GEN_REWARD_FUNCTION_REGISTRY[name] = cls
        return cls

    return decorator


def get_gen_reward_function_cls(name: str) -> type[GenRewardFunctionBase]:
    """Get the gen reward function class with a given name.

    Args:
        name: `(str)`
            The name of the gen reward function.

    Returns:
        `(type)`: The gen reward function class.
    """

    if name not in GEN_REWARD_FUNCTION_REGISTRY:
        raise ValueError(
            f"Unknown gen reward function: {name}. Available options: {list(GEN_REWARD_FUNCTION_REGISTRY.keys())}"
        )
    cls = GEN_REWARD_FUNCTION_REGISTRY[name]

    return cls
