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
    topk: int | None = 32
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
        if self.loss_settings.use_topk and (
            isinstance(self.topk, bool) or not isinstance(self.topk, int) or self.topk < 1
        ):
            raise ValueError("distillation_loss.topk must be set when using a top-k distillation loss.")

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            logger.warning(
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
    """Configure losses for teachers managed by the reward model service."""

    enabled: bool = False
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
