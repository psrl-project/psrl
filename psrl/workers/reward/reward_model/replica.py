from verl.workers.config import HFModelConfig

from psrl.workers.config import RolloutConfig
from psrl.workers.gen.vllm_async_server import GenInterface, PSRL_vLLMReplica


class RewardModelReplica(PSRL_vLLMReplica):
    """
    Serve frozen reward or pooling models without parameter-server synchronization.

    The rollout configuration must set `runner: pooling`.
    """

    def __init__(
        self,
        replica_rank: int,
        local_replica_rank: int,
        psrl_config,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gen_interface: GenInterface,
        reward_model_name: str,
        gpus_per_node: int = 8,
        tag: str = "reward",
    ):
        """
        Initialize a PSRL reward-model replica.

        Args:
            replica_rank (int): Global replica rank (used for naming).
            local_replica_rank (int): Index of this replica inside one worker group.
            psrl_config: Top-level PSRL configuration.
            config (RolloutConfig): Rollout configuration. Must have ``runner: pooling``.
            model_config (HFModelConfig): HuggingFace model configuration.
            gen_interface (GenInterface): Interface for status reporting.
                ``ps_manager_handle`` must be ``None`` for reward models (no PS sync).
                ``status_endpoint`` may also be ``None`` when ZMQ status collection is disabled.
            reward_model_name (str): Human-readable name for the reward model.
            gpus_per_node (int): Number of GPUs per node for this replica.
            tag (str): Tag used in logging and actor naming.
        """
        # NOTE(linsh): Setting `is_reward_model=True` propagates reward-model stat
        # labelling to `PSRL_vLLMHttpServer`.
        super().__init__(
            replica_rank=replica_rank,
            local_replica_rank=local_replica_rank,
            psrl_config=psrl_config,
            config=config,
            model_config=model_config,
            gen_interface=gen_interface,
            gpus_per_node=gpus_per_node,
            is_reward_model=True,
            tag=tag,
        )
        self.reward_model_name = reward_model_name
