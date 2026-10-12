from dataclasses import dataclass

import ray
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer
from verl.utils.dataset.rl_dataset import RLHFDataset

from psrl.sandbox.manager import SandboxManager


@dataclass(frozen=True)
class AgentLoopContext:
    """Bundle framework-owned dependencies for one agent-loop instance."""

    config: DictConfig
    rollout_gateway_url: str
    session_router_url: str
    reward_manager: ray.actor.ActorHandle
    ps_manager_handle: ray.actor.ActorHandle
    tokenizer: AutoTokenizer
    processor: AutoProcessor | None
    dataset_cls: type[RLHFDataset]
    data_config: DictConfig
    sandbox_manager: SandboxManager
