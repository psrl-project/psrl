"""Prompt feed for batch rollout, reusing the RL data processor."""

import logging
import os

import ray
from verl.utils import tensordict_utils as tu

from psrl.utils.dataset.data_processor import DataProcessorBase, DatasetType

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class BatchRolloutDataProcessor(DataProcessorBase):
    """Stream the dataset once, with no parameter server and no TransferQueue.

    `DataProcessorBase` already does everything batch rollout needs from a prompt feed:
    it streams batches from a `StatefulDataLoader` rather than materializing the
    dataset, expands each prompt into `rollout_n` children with a shared
    `parent_id`, allocates non-colliding uids, and blocks on a bounded queue when
    the consumer falls behind. That last property is the flow control: with
    `data_queue_size` set from the per-worker episode cap, the feed stalls instead
    of building an unbounded backlog of Docker containers.

    Only two things are dropped: registering requests with the parameter server,
    and the TransferQueue bootstrap.
    """

    def __init__(self, config, tokenizer, processor, completed_uids: set | None = None):
        """
        Args:
            config (DictConfig): Composed batch rollout configuration.
            tokenizer: Tokenizer used to build prompts.
            processor: Multimodal processor, or `None`.
            completed_uids (set | None): Prompt uids already present in the output
                file, skipped so a resumed run does not repeat them.
        """
        self.completed_uids = completed_uids or set()
        self.feed_finished = False
        super().__init__(config=config, tokenizer=tokenizer, processor=processor, ps_manager_handle=None)

    def is_feed_finished(self) -> bool:
        """Report whether the dataset has been walked to exhaustion."""
        return self.feed_finished

    @property
    def needs_validation_data(self) -> bool:
        """Collection has one dataset and no validation round."""
        return False

    def _resolve_total_training_steps(self) -> int | None:
        """Collection walks the dataset once, so there is no step budget."""
        return None

    def _init_data_plane(self) -> None:
        """Skip the TransferQueue bootstrap: payloads go to disk, not to TQ."""

    def _register_requests(self, request_ids: list, is_validate: bool = False) -> None:
        """Skip parameter server registration: there is no staleness inventory."""

    def _process_data(self):
        """Stream the dataset once, dropping prompts an earlier run already collected.

        Replaces the RL loop, which is driven by `total_epochs` and
        `total_training_steps`. Collection instead walks the dataset exactly once
        and stops, sending the END sentinel so the dispatcher drains.

        Resume keys on `uid`, which `get_train_sample_ids` allocates sequentially
        over the dataloader order. That order is reproducible only when the dataset
        files, `data.seed`, and `data.shuffle` are unchanged, which is why the
        driver records them in `summary.json` and why resuming onto a different
        dataset is not supported.
        """
        self.train_dataloader_iters = [iter(dataloader) for dataloader in self.train_dataloaders]
        skipped = 0

        while not self.stop_data_process:
            try:
                batch = self.get_single_controller_batch(DatasetType.train, return_meta=False)
            except StopIteration:
                psrl_logger.info("Dataset exhausted after one pass, ending the prompt feed.")
                break
            except Exception as e:
                psrl_logger.error(f"Exception in batch rollout data feed: {e}", exc_info=True)
                break

            if self.completed_uids:
                keep = [i for i, uid in enumerate(tu.get(batch, "uid")) if uid not in self.completed_uids]
                skipped += len(batch) - len(keep)
                if not keep:
                    continue
                batch = batch[keep]

            psrl_logger.debug("Feeding %d request(s) to the dispatcher.", len(batch))
            # Blocks once the queue is full, which is what bounds in-flight episodes.
            ray.get(self.agent_loop_manager_handle.put_data.remote(batch))
            self.global_steps += 1

        if skipped:
            psrl_logger.info("Skipped %d request(s) already present in the output file.", skipped)
        psrl_logger.info("Prompt feed finished, sending the END sentinel.")
        self.agent_loop_manager_handle.put_data.remote(None)
        # Set last: the driver reads this to know no further prompt will arrive, so
        # in-flight episodes are all that remain.
        self.feed_finished = True
