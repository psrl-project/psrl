from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, ListConfig


@dataclass
class RewardModelRuntimeInfo:
    """Runtime view shared by generative and teacher reward loops."""

    gateway_url: str
    reward_model_tokenizer: Any
    reward_model_config: DictConfig

    def get_gateway_url(self) -> str:
        return self.gateway_url

    def get_reward_model_tokenizer(self):
        return self.reward_model_tokenizer


def resolve_reward_function_name(value: Any) -> str:
    """Resolve one explicit reward function from a dataset specification."""
    if isinstance(value, (list, tuple, ListConfig)):
        if len(value) != 1:
            raise ValueError("Each reward specification must select exactly one reward function.")
        value = value[0]
    if isinstance(value, (dict, DictConfig)):
        value = value.get("name", "default")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Invalid reward function name {value!r}.")
    return value
