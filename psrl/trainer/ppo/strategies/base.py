"""
Define training step strategies and stage metadata.

`STAGE_META` distinguishes chunk-safe, batch-coupled, and optimizer stages.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer

from psrl.utils.logger import EventType, log_dual_events

if TYPE_CHECKING:
    from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

STAGE_META: dict[str, str] = {
    "old_log_prob": "per_sample",
    "ref_log_prob": "per_sample",
    "values": "per_sample",
    "reward": "per_sample",
    "advantage": "batch_coupled",
    "update_actor": "optimizer_step",
    "update_critic": "optimizer_step",
}


class StepStrategy(ABC):
    """Base class for training step orchestration.

    `run_step(buffer_id, metrics, timing_raw)` is the single entry point.
    Each strategy acquires the batch, pipelines the phase methods, and
    returns the final full `KVBatchMeta`.

    The trainer reference gives access to all phase methods and config
    without copying state.
    """

    def __init__(self, trainer: PSRL_RayPPOTrainer) -> None:
        self.trainer = trainer

    @abstractmethod
    def run_step(self, buffer_id: int, metrics: dict, timing_raw: dict):
        """Execute all training phases for one step.

        Args:
            buffer_id: Training buffer slot index (``global_steps - 1``).
            metrics: Accumulator dict updated in-place by phase methods.
            timing_raw: Timing dict for ``marked_timer`` instrumentation.

        Returns:
            KVBatchMeta: The final full batch (used for metrics and cleanup).
        """

    def _run_ckpt_and_validate(
        self,
        batch,
        metrics: dict,
        timing_raw: dict,
        actor_updated: bool = True,
    ) -> None:
        """Shared checkpoint, rollout dump, and validation tail.

        Args:
            batch: Final full batch after all training phases.
            metrics: Accumulator dict updated with validation metrics.
            timing_raw: Timing dict passed to ``marked_timer`` contexts.
            actor_updated: Whether the actor was updated this step.
                Checkpoint save is gated on this flag to match the original
                ``ray_trainer.fit()`` behavior where saving only happened
                after the actor update. Rollout dump and validation run
                regardless.
        """
        t = self.trainer
        is_last_step = t.global_steps == t.total_training_steps
        esi_close = should_save_ckpt_esi(
            max_steps_duration=t.max_steps_duration,
            redundant_time=t.config.trainer.esi_redundant_time,
        )
        if (
            actor_updated
            and t.config.trainer.save_freq > 0
            and (is_last_step or t.global_steps % t.config.trainer.save_freq == 0 or esi_close)
        ):
            if esi_close:
                print("Force saving checkpoint: ESI instance expiration approaching.")
            with marked_timer("save_checkpoint", timing_raw, color="green"):
                with log_dual_events(
                    "Save checkpoint",
                    psrl_logger,
                    event_type=EventType.OTHER,
                ):
                    t._save_checkpoint()

        rollout_data_dir = t.config.trainer.get("rollout_data_dir", None)
        if rollout_data_dir:
            t._log_rollout_data(batch, timing_raw, rollout_data_dir)

        if t.config.trainer.test_freq > 0 and (is_last_step or t.global_steps % t.config.trainer.test_freq == 0):
            with marked_timer("testing", timing_raw, color="green"):
                with log_dual_events("Validate", psrl_logger, event_type=EventType.VAL):
                    val_metrics: dict = t._validate()
                    if is_last_step:
                        t._last_val_metrics = val_metrics
            metrics.update(val_metrics)


def build_step_strategy(cfg, trainer: PSRL_RayPPOTrainer) -> StepStrategy:
    """Return the appropriate ``StepStrategy`` based on ``psrl.fine_grain_overlap``."""
    if cfg is None or str(getattr(cfg, "granularity", "none")) == "none":
        from psrl.trainer.ppo.strategies.full_batch import FullBatchStepStrategy

        return FullBatchStepStrategy(trainer)

    from psrl.trainer.ppo.strategies.fine_grain_overlap import FineGrainOverlapStrategy

    return FineGrainOverlapStrategy(trainer, cfg)
