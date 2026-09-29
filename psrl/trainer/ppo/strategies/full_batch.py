"""FullBatchStepStrategy: standard full-batch training step.

Waits for the entire global batch to accumulate, then runs all phases
sequentially. This is the default strategy (``fine_grain_overlap.granularity:
none``) and is numerically identical to the original ``ray_trainer.fit()``
step body.
"""

import ray
from verl.utils.debug import marked_timer

from psrl.trainer.ppo.strategies.base import StepStrategy, psrl_logger
from psrl.utils.logger import EventType, log_dual_events


class FullBatchStepStrategy(StepStrategy):
    """Standard full-batch PPO step with no overlap.

    ``run_step`` blocks on ``wait_for_training_batch`` (measured under the
    ``wait_for_gen`` timer), switches to trainer mode, then runs the phase
    pipeline on the whole batch in sequence.
    """

    def run_step(self, buffer_id: int, metrics: dict, timing_raw: dict):
        t = self.trainer

        with marked_timer("wait_for_gen", timing_raw, color="gray"):
            with log_dual_events(
                f"Wait for training batch {buffer_id}",
                psrl_logger,
                event_type=EventType.WAIT,
            ):
                batch = ray.get(t.agent_loop_manager.wait_for_training_batch.remote(buffer_id))
                t.replay_buffer.sample(batch.keys, batch.partition_id)
            with log_dual_events(
                "Switch to trainer mode",
                psrl_logger,
                event_type=EventType.SWITCH,
            ):
                t.switch_to_trainer_mode()

        # Prefix Match Rate of the full global batch (before balancing /
        # mini-batch splitting), logged under pmr/*.
        self.maybe_collect_pmr(batch, metrics, timing_raw)

        if t.config.trainer.balance_batch:
            batch = t._balance_batch(batch, metrics=metrics)

        batch.extra_info["temperature"] = t.config.gen_actor_rollout_ref.rollout.temperature
        batch.extra_info["global_steps"] = t.global_steps

        with marked_timer("old_log_prob", timing_raw, color="orange"):
            with log_dual_events(
                "Recompute log_prob on training side",
                psrl_logger,
                event_type=EventType.OTHER,
            ):
                batch = t._compute_old_log_prob(batch, metrics=metrics)

        if t.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                with log_dual_events(
                    "Compute reference log_prob",
                    psrl_logger,
                    event_type=EventType.OTHER,
                ):
                    batch = t._compute_ref_log_prob(batch, metrics=metrics)

        if t.use_critic:
            with marked_timer("values", timing_raw, color="cyan"):
                with log_dual_events(
                    "Compute critic values",
                    psrl_logger,
                    event_type=EventType.OTHER,
                ):
                    batch = t._compute_values(batch, metrics=metrics)

        if t.config.reward.launch_reward_fn_async:
            with marked_timer("async_reward_get", timing_raw, color="yellow"):
                with log_dual_events(
                    "Wait for async reward model score",
                    psrl_logger,
                    event_type=EventType.OTHER,
                ):
                    batch = ray.get(t.reward_manager.wait_for_reward_of_requests.remote(batch))
        else:
            with log_dual_events(
                "Normalize reward",
                psrl_logger,
                event_type=EventType.OTHER,
            ):
                batch = ray.get(t.reward_manager.normalize_reward.remote(batch))

        with marked_timer("adv", timing_raw, color="brown"):
            with log_dual_events(
                "Compute advantage",
                psrl_logger,
                event_type=EventType.OTHER,
            ):
                batch = t._compute_advantage(batch, metrics=metrics)

        if t.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                with log_dual_events(
                    "Update critic",
                    psrl_logger,
                    event_type=EventType.TRAIN,
                ):
                    batch = t._update_critic(batch, metrics=metrics)

        if t.config.trainer.critic_warmup <= t.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                with log_dual_events(
                    "Update actor",
                    psrl_logger,
                    event_type=EventType.TRAIN,
                ):
                    batch = t._update_actor(batch, metrics=metrics)

        self._run_ckpt_and_validate(
            batch,
            metrics,
            timing_raw,
            actor_updated=(t.config.trainer.critic_warmup <= t.global_steps),
        )

        return batch
