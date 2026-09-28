"""Dispatch-only agent loop manager for batch rollout."""

import logging
import os

from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from psrl.workers.agent_loop.manager_base import AgentLoopManagerBase

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class BatchRolloutAgentLoopManager(AgentLoopManagerBase):
    """Feed prompts to the workers, with no staleness and no parameter server.

    Everything this needs is in `AgentLoopManagerBase`: the bounded queue that
    provides backpressure, round-robin dispatch that co-locates the children of one
    prompt, and the worker lifecycle. Both dispatch hooks stay near their defaults
    because a fixed checkpoint has no version to wait for and no request status to
    advance.
    """

    def __init__(self, config, data_queue_size: int, agent_loop_workers: list):
        """
        Args:
            config (DictConfig): Composed batch rollout configuration.
            data_queue_size (int): Prompt queue capacity, which bounds in-flight work.
            agent_loop_workers (list): `BatchRolloutAgentLoopWorker` handles.
        """
        super().__init__(
            config=config,
            data_queue_size=data_queue_size,
            agent_loop_workers=agent_loop_workers,
            log_prefix="BatchRolloutAgentLoopManager",
        )
        self.rollout_n = config.gen_actor_rollout_ref.rollout.n

    async def _before_dispatch(self, data: TensorDict) -> None:
        """Stamp the static checkpoint version onto every request.

        A fixed checkpoint has one version, so there is nothing to wait for. The tag
        is still carried, because SMG filters candidates on it and the record is
        stamped with it.
        """
        tu.assign_non_tensor_stack(data, "version_tag", [0] * len(data))
