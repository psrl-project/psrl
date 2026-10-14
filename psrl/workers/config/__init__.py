# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
#
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#
# limitations under the License.
"""Export common veRL configurations and PSRL-specific definitions."""

from verl.utils.qat import QATConfig  # noqa: F401
from verl.workers.config.actor import (  # noqa: F401
    ActorConfig,
    FSDPActorConfig,
    McoreActorConfig,
    PolicyLossConfig,
    RouterReplayConfig,
    TorchTitanActorConfig,
    VeOmniActorConfig,
)
from verl.workers.config.critic import (  # noqa: F401
    CriticConfig,
    FSDPCriticConfig,
    FSDPCriticModelCfg,
    McoreCriticConfig,
)
from verl.workers.config.engine import (  # noqa: F401
    EngineConfig,
    EngineRouterReplayConfig,
    FSDPEngineConfig,
    McoreEngineConfig,
    TrainingWorkerConfig,
)
from verl.workers.config.megatron_peft import get_peft_cls  # noqa: F401
from verl.workers.config.model import HFModelConfig, MtpConfig  # noqa: F401
from verl.workers.config.optimizer import (  # noqa: F401
    FSDPOptimizerConfig,
    McoreOptimizerConfig,
    OptimizerConfig,
    build_optimizer,
)

from .distillation import *  # noqa: F401, F403
from .reward_model import *  # noqa: F401, F403
from .rollout import *  # noqa: F401, F403

MegatronEngineConfig = McoreEngineConfig
