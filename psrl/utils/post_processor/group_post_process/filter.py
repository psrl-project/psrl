import torch
from tensordict import TensorDict

from ..base import BaseGroupPostProcessor, GroupPostProcessorRegistry


@GroupPostProcessorRegistry.register("null")
class NoFilterProcessor(BaseGroupPostProcessor):
    """
    Group post-processor that performs no filtering.

    This processor simply returns all data unchanged.
    """

    def __call__(self, data: TensorDict) -> TensorDict | None:
        """
        Return data unchanged.

        Args:
            data (TensorDict): The grouped data to be processed.

        Returns:
            TensorDict: The unchanged data.
        """
        return data


@GroupPostProcessorRegistry.register("dynamic_sampling_filter")
class DynamicSamplingFilterProcessor(BaseGroupPostProcessor):
    """
    Group post-processor that filters data based on reward variance.

    This processor filters out data groups where the reward variance is zero,
    which indicates that all samples in the group have the same reward value.
    """

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
        rewards = data[metric_name]

        if torch.tensor(rewards, dtype=torch.float).std() > 0.0:
            return data
        return None
