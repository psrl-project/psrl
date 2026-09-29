"""
Overlap chunk-safe training stages with ongoing rollout.

`recompute` applies advantages and updates to the full batch. `pre_step` updates
each mini-batch chunk and is exact only for group-local GRPO normalization.
"""

import ray
from transfer_queue.metadata import KVBatchMeta
from verl.utils.debug import marked_timer

from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer
from psrl.trainer.ppo.strategies.base import StepStrategy, psrl_logger
from psrl.utils.config import resolve_fine_grain_chunk_size
from psrl.utils.logger import EventType, log_dual_events


class FineGrainOverlapStrategy(StepStrategy):
    """
    Overlap training stages with rollout using chunk-level pipelining.

    `recompute` updates the concatenated batch. `pre_step` updates each
    mini-batch chunk.

    Session-mean-token-mean loss (``actor.loss_agg_mode=session-mean-token-mean``)
    requires window-level normalization, so it is only allowed with
    ``overlap_scope=recompute`` (pre_step's chunk-local normalization would change
    the per-session denominator per chunk).
    """

    def __init__(self, trainer: PSRL_RayPPOTrainer, cfg) -> None:
        """Initialize and register the chunk size with the manager.

        Args:
            trainer: The ``PSRL_RayPPOTrainer`` instance.
            cfg: The ``psrl.fine_grain_overlap`` config sub-tree (OmegaConf node).
        """
        super().__init__(trainer)
        t = trainer
        dp_size = t._get_dp_size(t.actor_wg, "actor")
        self.effective_granularity, self.chunk_groups = resolve_fine_grain_chunk_size(t.config, dp_size)
        self.overlap_scope = str(cfg.get("overlap_scope", "recompute"))

        if self.overlap_scope == "pre_step" and self.effective_granularity != "mini_batch":
            raise ValueError(
                f"pre_step scope requires mini_batch granularity (micro_batch pre_step is Phase 4, "
                f"not yet implemented). Got effective_granularity={self.effective_granularity!r}. "
                "Use overlap_scope=recompute with micro_batch, or reduce multiplier so chunk "
                "clamps to mini_batch."
            )

        # Session-mean-token-mean loss computes session weights once per scheduled batch, so the
        # chunk-local denominator of pre_step would change them. Require the recompute scope.
        loss_agg_mode = t.config.train_actor_rollout_ref.actor.get("loss_agg_mode", "token-mean")
        if self.overlap_scope == "pre_step" and loss_agg_mode == "session-mean-token-mean":
            raise ValueError(
                "overlap_scope=pre_step is not compatible with actor.loss_agg_mode="
                "session-mean-token-mean: pre_step applies chunk-local per-session normalization, "
                "which is inconsistent with window-level session normalization. "
                "Use overlap_scope=recompute (advantage + updates run on the "
                "concatenated full batch; per-sample stages still overlap) or set "
                "psrl.fine_grain_overlap.granularity=none for the full-batch path."
            )

        ray.get(t.agent_loop_manager.set_chunk_size.remote(self.chunk_groups))
        psrl_logger.info(
            "FineGrainOverlapStrategy initialized: granularity=%r chunk_groups=%d overlap_scope=%r.",
            self.effective_granularity,
            self.chunk_groups,
            self.overlap_scope,
        )

    def run_step(self, buffer_id: int, metrics: dict, timing_raw: dict):
        """Pipeline per_sample stages over chunks, then run full-batch updates.

        For `recompute` scope, per-sample stages run per chunk. Advantage
        and optimizer updates run once on the concatenated full batch.

        For `pre_step + mini_batch` scope, advantage and one optimizer step
        also run per chunk immediately after per_sample stages.

        Args:
            buffer_id: Training buffer ID (``global_steps - 1``).
            metrics: Accumulator dict updated in-place by phase methods.
            timing_raw: Timing dict for ``marked_timer`` instrumentation.

        Returns:
            KVBatchMeta: The concatenated full batch after all phases complete.
        """
        t = self.trainer

        # Switch before GPU work so chunks can arrive while rollout progresses.
        t.switch_to_trainer_mode()

        chunks: list[KVBatchMeta] = []
        chunk_idx = 0

        while True:
            # Use warning so these show under default PSRL_LOGGING_LEVEL=WARN.
            psrl_logger.warning(
                "FineGrainOverlap: waiting for buffer=%d chunk=%d",
                buffer_id,
                chunk_idx,
            )
            chunk_meta, is_last = ray.get(t.agent_loop_manager.wait_for_training_chunk.remote(buffer_id, chunk_idx))
            psrl_logger.warning(
                "FineGrainOverlap: got buffer=%d chunk=%d size=%d is_last=%s. Sampling replay buffer",
                buffer_id,
                chunk_idx,
                len(chunk_meta),
                is_last,
            )

            t.replay_buffer.sample(chunk_meta.keys, chunk_meta.partition_id)
            psrl_logger.warning(
                "FineGrainOverlap: replay sample ready for buffer=%d chunk=%d",
                buffer_id,
                chunk_idx,
            )

            if t.config.trainer.balance_batch:
                chunk_meta = t._balance_batch(chunk_meta, metrics=metrics)

            chunk_meta.extra_info["temperature"] = t.config.gen_actor_rollout_ref.rollout.temperature
            chunk_meta.extra_info["global_steps"] = t.global_steps

            # --- per_sample stages (overlap with rollout generating the next chunk) ---
            with marked_timer("old_log_prob", timing_raw, color="orange"):
                chunk_meta = t._compute_old_log_prob(chunk_meta, metrics=metrics)

            if t.use_reference_policy:
                with marked_timer("ref", timing_raw, color="olive"):
                    chunk_meta = t._compute_ref_log_prob(chunk_meta, metrics=metrics)

            if t.use_critic:
                with marked_timer("values", timing_raw, color="cyan"):
                    chunk_meta = t._compute_values(chunk_meta, metrics=metrics)

            if t.config.reward.launch_reward_fn_async:
                with marked_timer("async_reward_get", timing_raw, color="yellow"):
                    chunk_meta = ray.get(t.reward_manager.wait_for_reward_of_requests.remote(chunk_meta))
            else:
                chunk_meta = ray.get(t.reward_manager.normalize_reward.remote(chunk_meta))

            # --- pre_step + mini_batch: per-chunk advantage + optimizer step ---
            if self.overlap_scope == "pre_step" and self.effective_granularity == "mini_batch":
                with marked_timer("adv", timing_raw, color="brown"):
                    with log_dual_events(
                        "Compute advantage",
                        psrl_logger,
                        event_type=EventType.OTHER,
                    ):
                        chunk_meta = t._compute_advantage(chunk_meta, metrics=metrics)

                if t.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        with log_dual_events(
                            "Update critic",
                            psrl_logger,
                            event_type=EventType.TRAIN,
                        ):
                            chunk_meta = t._update_critic(
                                chunk_meta,
                                metrics=metrics,
                                finalize_step=is_last,
                            )

                if t.config.trainer.critic_warmup <= t.global_steps:
                    # Push only the final chunk because advancing the PS version can delete this buffer early.
                    with marked_timer("update_actor", timing_raw, color="red"):
                        with log_dual_events(
                            f"Update actor (chunk {chunk_idx}, push={is_last})",
                            psrl_logger,
                            event_type=EventType.TRAIN,
                        ):
                            chunk_meta = t._update_actor(
                                chunk_meta,
                                metrics=metrics,
                                finalize_step=is_last,
                            )
                            # Dispatch-local lifecycle controls must not survive into
                            # `KVBatchMeta.concat` with different values per window.
                            chunk_meta.extra_info.pop("advance_lr_scheduler", None)
                            chunk_meta.extra_info.pop("push_model", None)

            chunks.append(chunk_meta)
            chunk_idx += 1
            if is_last:
                break

        full_batch = KVBatchMeta.concat(chunks)

        # Prefix Match Rate of the full global batch (after chunk concat),
        # logged under pmr/*.
        self.maybe_collect_pmr(full_batch, metrics, timing_raw)

        # --- recompute scope: advantage + updates run on the full batch ---
        if self.overlap_scope == "recompute":
            with marked_timer("adv", timing_raw, color="brown"):
                with log_dual_events(
                    "Compute advantage",
                    psrl_logger,
                    event_type=EventType.OTHER,
                ):
                    full_batch = t._compute_advantage(full_batch, metrics=metrics)

            if t.use_critic:
                with marked_timer("update_critic", timing_raw, color="pink"):
                    with log_dual_events(
                        "Update critic",
                        psrl_logger,
                        event_type=EventType.TRAIN,
                    ):
                        full_batch = t._update_critic(full_batch, metrics=metrics)

            if t.config.trainer.critic_warmup <= t.global_steps:
                with marked_timer("update_actor", timing_raw, color="red"):
                    with log_dual_events(
                        "Update actor",
                        psrl_logger,
                        event_type=EventType.TRAIN,
                    ):
                        full_batch = t._update_actor(full_batch, metrics=metrics)

        self._run_ckpt_and_validate(
            full_batch,
            metrics,
            timing_raw,
            actor_updated=(t.config.trainer.critic_warmup <= t.global_steps),
        )

        return full_batch
