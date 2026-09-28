"""Agent loop worker that records trajectories instead of training on them."""

import logging
import os

import ray
from omegaconf import DictConfig
from tensordict import TensorDict

from psrl.batch_rollout.record import build_failure_record, build_records
from psrl.batch_rollout.serving.base import BackendHandle
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.agent_loop.worker_base import AgentLoopWorkerBase
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class BatchRolloutAgentLoopWorker(AgentLoopWorkerBase):
    """Run agent loops and hand each result to the output writer.

    Differs from `PSRL_AgentLoopWorker` only in what happens after an episode: no
    parameter server to advance, no TransferQueue to commit to, and no group to
    refill. A failure is recorded and the run moves on.
    """

    def __init__(
        self,
        config: DictConfig,
        backend_handle: BackendHandle,
        output_writer: ray.actor.ActorHandle,
        worker_id: int = 0,
        worker_num: int = 1,
    ):
        """
        Args:
            config (DictConfig): Composed batch rollout configuration.
            backend_handle (BackendHandle): Where the served model lives.
            output_writer (ray.actor.ActorHandle): `RolloutOutputWriter` to append to.
            worker_id (int): Unique identifier for this worker instance.
            worker_num (int): Total number of worker instances.
        """
        self.backend_handle = backend_handle
        self.output_writer = output_writer
        self.dump_tokens = bool(config.batch_rollout.dump_tokens) and backend_handle.supports_token_capture
        if config.batch_rollout.dump_tokens and not backend_handle.supports_token_capture:
            psrl_logger.warning(
                "dump_tokens is set but the %s backend returns no token ids, so the "
                "`tokens` field is omitted rather than reconstructed.",
                config.batch_rollout.serving.name,
            )

        super().__init__(
            config=config,
            rollout_gateway_url=backend_handle.rollout_gateway_url,
            session_router_url=backend_handle.session_router_url or "",
            worker_id=worker_id,
            worker_num=worker_num,
            log_prefix="BatchRolloutAgentLoopWorker",
        )

    async def _handle_output(
        self,
        output: TokenOutput | list[TokenOutput],
        batch: TensorDict,
        terminate_reason: TerminateReason,
    ) -> None:
        """Write one episode's trajectories to the output file."""
        records = build_records(
            output,
            batch,
            terminate_reason,
            model_name=self.backend_handle.model_name,
            dump_tokens=self.dump_tokens,
        )
        await self.output_writer.append.remote(records)

    async def _handle_failure(
        self,
        batch: TensorDict,
        terminate_reason: TerminateReason,
    ) -> None:
        """Record an episode that produced nothing, so it is not silently lost."""
        records = build_failure_record(
            batch,
            terminate_reason,
            model_name=self.backend_handle.model_name,
        )
        await self.output_writer.append.remote(records)
