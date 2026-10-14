from collections import defaultdict

import numpy as np
from omegaconf import DictConfig
from tensordict import TensorDict

from ..base import BaseBufferPostProcessor, BufferPostProcessorRegistry


@BufferPostProcessorRegistry.register("null")
class NoFilterProcessor(BaseBufferPostProcessor):
    """
    Buffer post-processor that performs no filtering.

    This processor simply returns all data unchanged.
    """

    def __call__(self, data: TensorDict) -> TensorDict | None:
        """
        Return data unchanged.

        Args:
            data (TensorDict): The buffer data to be processed.

        Returns:
            TensorDict: The unchanged data.
        """
        return data


@BufferPostProcessorRegistry.register("dynamic_sampling_filter")
class DynamicSamplingFilterProcessor(BaseBufferPostProcessor):
    """
    Buffer post-processor that filters data based on reward variance.

    This processor filters out data buffer where the reward variance is zero,
    which indicates that all samples in the buffer have the same reward value.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the buffer post-processor with system configuration.

        Args:
            config (DictConfig): System configuration object that may be used
                                in the __call__ method for processing decisions.
        """
        super().__init__(config)

        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n

    def __call__(self, data: TensorDict) -> TensorDict | None:
        """
        Filter data based on reward variance.

        Args:
            data (TensorDict): The grouped data to be processed.

        Returns:
            Optional[TensorDict]: The data if variance > 0, None otherwise.
        """
        metric_name = self.config.algorithm.filter_groups.metric
        if metric_name == "seq_final_reward":
            data["seq_final_reward"] = data["token_level_rewards"].sum(dim=-1).numpy()
        elif metric_name == "seq_reward":
            data["seq_reward"] = data["token_level_scores"].sum(dim=-1).numpy()

        # Collect the sequence reward for each trajectory
        pid2metric_vals = defaultdict(list)
        pid_list = []
        for uid, metric_val in zip(
            data["uid"],
            data[metric_name],
            strict=True,
        ):
            pid = uid // self.rollout_n
            pid2metric_vals[pid].append(metric_val)
            pid_list.append(pid)

        pid2metric_std = {}
        for pid, metric_vals in pid2metric_vals.items():
            pid2metric_std[pid] = np.std(metric_vals)

        kept_pids = {pid for pid, std in pid2metric_std.items() if std > 0 or len(pid2metric_vals[pid]) == 1}
        kept_traj_idxs = []
        for idx, pid in enumerate(pid_list):
            if pid in kept_pids:
                kept_traj_idxs.append(idx)
        filtered_data = data[kept_traj_idxs] if kept_traj_idxs else None
        return filtered_data
