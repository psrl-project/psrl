# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
#
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#
# limitations under the License.
"""Define PSRL rollout configuration extensions."""

from dataclasses import dataclass, field

from omegaconf import MISSING
from verl.base_config import BaseConfig
from verl.workers.config.rollout import (
    AgentLoopConfig as _VeRLAgentLoopConfig,
)
from verl.workers.config.rollout import (
    CheckpointEngineConfig,
    CustomAsyncServerConfig,
    PrometheusConfig,
    SamplingConfig,
    ServerConfig,
    TraceConfig,
)
from verl.workers.config.rollout import (
    MultiTurnConfig as _VeRLMultiTurnConfig,
)
from verl.workers.config.rollout import (
    RolloutConfig as _VeRLRolloutConfig,
)

from psrl.sandbox.config import SandboxManagerConfig


@dataclass
class PoolingConfig(BaseConfig):
    """Configuration for vLLM pooling models (e.g., reward/embedding models)."""

    normalize: bool = False
    use_activation: bool = False


@dataclass
class EnvironmentConfig(BaseConfig):
    name: str | None = MISSING
    step_timeout: float | None = None


@dataclass
class AgentDataConfig(BaseConfig):
    name: str | None = MISSING


@dataclass
class MultiTurnConfig(_VeRLMultiTurnConfig):
    _mutable_fields = {"max_turns"}

    enable: bool = False
    max_turns: int | None = None
    tool_config_path: str | None = None
    function_tool_path: str | None = None
    max_parallel_calls: int = 1
    max_tool_response_length: int = 256
    tool_response_truncate_side: str = "middle"
    use_inference_chat_template: bool = False
    tokenization_sanity_check_mode: str = "strict"
    format: str = "hermes"


@dataclass
class AgentLoopConfig(_VeRLAgentLoopConfig):
    """Configure PSRL agent loops."""

    route_strategy: str = "round_robin"
    trajectory_timeout: float | None = None
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: AgentDataConfig = field(default_factory=AgentDataConfig)
    sandbox: SandboxManagerConfig = field(default_factory=SandboxManagerConfig)
    retry_limit: int = 1
    raise_on_error: bool = True
    gamma: float = 0.0
    reward_bonus_coeff: float = 0.0
    traj_reward_mode: str = "traj"
    default_agent_loop: str = "generate_only_agent"
    # Node IPs allowed to host agent loop workers. Empty means every alive node. Naming a
    # subset keeps rollout off a node whose degraded Docker daemon accepts then hangs actors.
    node_ips: list[str] = field(default_factory=list)
    # DAPO Overlong Filtering. Zero the loss mask of budget-truncated trajectories, so their
    # tokens carry no gradient while their reward still moves the GRPO group baseline.
    #
    # Unlike verl's length-proportional Soft Overlong Punishment, this shapes the mask.
    # Off by default because it discards real rollout tokens.
    overlong_filtering: bool = False


@dataclass
class RolloutConfig(_VeRLRolloutConfig):
    """Extend veRL rollout configuration with PSRL settings."""

    disable_attn: bool = False
    runner: str = "generate"
    task: str = "generate"
    reward_kwargs: PoolingConfig = field(default_factory=PoolingConfig)

    enable_weights_cpu_backup: bool = False

    multi_turn: MultiTurnConfig = field(default_factory=MultiTurnConfig)

    agent: AgentLoopConfig = field(default_factory=AgentLoopConfig)

    # The SMG router uses this template to render TITO session prompts.
    chat_template: str | None = None


__all__ = [
    "SamplingConfig",
    "MultiTurnConfig",
    "CustomAsyncServerConfig",
    "AgentLoopConfig",
    "TraceConfig",
    "ServerConfig",
    "PrometheusConfig",
    "RolloutConfig",
    "PoolingConfig",
    "CheckpointEngineConfig",
    "EnvironmentConfig",
    "AgentDataConfig",
    "SandboxManagerConfig",
]
