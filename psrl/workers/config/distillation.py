# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License").
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass, field

from verl.base_config import BaseConfig

__all__ = ["DistillationLossConfig", "DistillationConfig"]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., `log q - log p` where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: int | None = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: float | None = 10.0
    log_prob_min_clamp: float | None = -10.0

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Global DP size, token count, and batch size determine loss aggregation.
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: dict | None = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from psrl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)
        if self.loss_settings.use_topk and self.topk is None:
            raise ValueError("distillation_loss.topk must be set when using a top-k distillation loss.")

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)

    def __post_init__(self):
        if not self.enabled:
            return
        if self.distillation_loss.loss_settings.use_topk and self.distillation_loss.topk is None:
            raise ValueError("distillation.distillation_loss.topk must be set for top-k distillation.")
