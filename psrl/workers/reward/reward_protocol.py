from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass
class RewardModelRuntimeInfo:
    """Runtime view required by GenRewardLoop."""

    gateway_url: str
    reward_model_tokenizer: Any
    reward_model_config: DictConfig

    def get_gateway_url(self) -> str:
        return self.gateway_url

    def get_reward_model_tokenizer(self):
        return self.reward_model_tokenizer
