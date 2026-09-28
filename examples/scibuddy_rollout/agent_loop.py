"""
Run SciBuddy tasks through Harbor and return PSRL training data.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading

from examples.scibuddy_rollout.config import SciBuddyRuntimeConfig, build_runtime_config
from examples.scibuddy_rollout.runner import HarborEpisodeResult, run_harbor_episode
from psrl.utils.agent.thinking import MULTI_TRAJ, select_trajectories
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason, register

psrl_logger = logging.getLogger("psrl.scibuddy_rollout.agent_loop")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_ABORT_MARKER = "Request aborted by PS Manager"


class _HarborLoopThread:
    """Single dedicated thread running its own asyncio event loop for Harbor jobs."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def _start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="scibuddy-harbor-loop",
        )
        self._thread.start()

    def run(self, coro) -> asyncio.Future:
        self._start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)


_harbor_loop = _HarborLoopThread()

_episode_gate: asyncio.Semaphore | None = None
_episode_gate_limit = 0


async def _acquire_episode_slot(limit: int) -> asyncio.Semaphore:
    """Return the shared episode gate, building it on first use."""
    global _episode_gate, _episode_gate_limit
    if _episode_gate is None or _episode_gate_limit != limit:
        _episode_gate = asyncio.Semaphore(limit)
        _episode_gate_limit = limit
    await _episode_gate.acquire()
    return _episode_gate


@register("scibuddy")
class SciBuddyAgentLoop(SessionAgentLoop):
    """
    Agent loop for SciBuddy-style Harbor research tasks.

    Unlike sciaccel (code repair), SciBuddy tasks ask the model to produce a
    structured JSON results file by running analysis code inside a CPU-only
    Harbor container. The verifier checks the results file's schema and numeric
    consistency. The reward is binary: 0 or 1 from /logs/verifier/reward.txt.

    Reuses the full sciaccel runner and Harbor infrastructure. The only
    task-specific logic is reward extraction: these tasks write a scalar reward
    to reward.txt rather than a multi-key dict.
    """

    def __init__(self, context: AgentLoopContext, **kwargs):
        super().__init__(context=context)
        runtime_kwargs = {k: kwargs[k] for k in ("harbor", "task_timeout_sec", "verifier_timeout_sec") if k in kwargs}
        self.runtime_config: SciBuddyRuntimeConfig = build_runtime_config(runtime_kwargs)
        self.thinking_template: str = context.config.psrl.agentic_rl.get("thinking_template", MULTI_TRAJ)

    async def run(
        self,
        request: dict,
    ) -> tuple[object | None, TerminateReason]:
        """Run one SciBuddy Harbor research episode and collect TITO training data.

        Args:
            request: Batch element from the parquet, carrying `extra_info`.

        Returns:
            Tuple of (outputs, terminate_reason) matching the SessionAgentLoop contract.
        """
        extra_info = request.get("extra_info") or {}
        if isinstance(extra_info, str):
            import json

            try:
                extra_info = json.loads(extra_info)
            except (json.JSONDecodeError, TypeError):
                extra_info = {}

        task_path = extra_info.get("task_path", "")
        reward_key = extra_info.get("reward_key", "reward")
        uid = request.get("uid", "?")

        if not task_path:
            psrl_logger.error("[uid=%s] task_path missing from extra_info.", uid)
            return None, TerminateReason.ROLLOUT_ERROR

        session_id: str | None = None
        try:
            session_id = await self.create_session(request)
            model_base_url = self.session_api_url(session_id)
            model_name = self.model_config.path

            psrl_logger.info(
                "[uid=%s] Starting Harbor episode: session=%s task=%s model_url=%s.",
                uid,
                session_id,
                task_path,
                model_base_url,
            )

            harbor_result = await self._run_harbor_in_thread(
                task_path=task_path,
                model_base_url=model_base_url,
                model_name=model_name,
                session_id=session_id,
                actor_id=getattr(self, "actor_id", ""),
            )

            if harbor_result.exception:
                psrl_logger.warning(
                    "[uid=%s] Harbor episode exception: %s (rewards=%s).",
                    uid,
                    harbor_result.exception,
                    harbor_result.rewards,
                )
            else:
                psrl_logger.info(
                    "[uid=%s] Harbor episode completed: reward=%.3f rewards=%s.",
                    uid,
                    harbor_result.reward,
                    harbor_result.rewards,
                )

            if harbor_result.exception and _ABORT_MARKER in harbor_result.exception:
                psrl_logger.info("[uid=%s] Episode aborted by PSManager.", uid)
                return None, TerminateReason.ABORTED

            training_data = select_trajectories(
                self.thinking_template,
                await self.get_training_data(session_id),
            )
            num_turns = sum(item["num_turns"] for item in training_data)
            num_tokens = sum(len(item.get("response_ids", [])) for item in training_data)
            psrl_logger.info(
                "[uid=%s] TITO data (%s): %d trajector%s num_turns=%d response_tokens=%d.",
                uid,
                self.thinking_template,
                len(training_data),
                "y" if len(training_data) == 1 else "ies",
                num_turns,
                num_tokens,
            )

            if num_turns == 0:
                psrl_logger.warning("[uid=%s] Zero turns in TITO session. Failing for refill.", uid)
                return None, TerminateReason.ROLLOUT_ERROR

            outputs = [
                self.build_token_output(
                    item,
                    extra_fields={
                        "harbor_rewards": harbor_result.rewards,
                        "reward_key": reward_key,
                        "task_name": harbor_result.task_name,
                        "exception": harbor_result.exception or "",
                        "exception_type": harbor_result.exception_type or "",
                    },
                )
                for item in training_data
            ]

            scored_output = await self.compute_reward_score(outputs if len(outputs) > 1 else outputs[0], **request)
            if scored_output is None:
                psrl_logger.warning("[uid=%s] compute_reward_score returned None.", uid)
                return None, TerminateReason.ROLLOUT_ERROR

            terminate_reason = (
                TerminateReason.MAX_TURNS_EXCEEDED
                if harbor_result.exception and "max_turns" in str(harbor_result.exception).lower()
                else TerminateReason.FINISHED
            )
            psrl_logger.info(
                "[uid=%s] Episode done: terminate=%s reward=%.3f.",
                uid,
                terminate_reason.value,
                harbor_result.reward,
            )
            return scored_output, terminate_reason

        except Exception as exc:
            psrl_logger.warning("[uid=%s] Harbor episode raised: %s.", uid, exc, exc_info=True)
            return None, TerminateReason.ROLLOUT_ERROR
        finally:
            if session_id:
                try:
                    await self.delete_session(session_id)
                except Exception:
                    pass

    async def _run_harbor_in_thread(
        self,
        task_path: str,
        model_base_url: str,
        model_name: str,
        session_id: str,
        actor_id: str = "",
    ) -> HarborEpisodeResult:
        """Run the Harbor Job on the dedicated Harbor event loop thread.

        All Harbor Jobs share one event loop to avoid asyncio.subprocess races.
        The episode is admitted through a semaphore released only after
        run_harbor_episode tears its containers down.

        Args:
            task_path: Absolute path to the Harbor task directory.
            model_base_url: Session URL for the model endpoint.
            model_name: Model name passed to the agent container.
            session_id: Job directory name for traceability.
            actor_id: Ray actor ID label attached to the Compose service.

        Returns:
            HarborEpisodeResult with reward and exception information.
        """
        limit = int(self.runtime_config.harbor.max_concurrent_episodes)
        max_model_len = int(getattr(self.rollout_config, "max_model_len", 40960) or 40960)
        max_turns_cfg = getattr(getattr(self.rollout_config, "multi_turn", None), "max_turns", None)

        async def _gated() -> HarborEpisodeResult:
            gate = await _acquire_episode_slot(limit)
            try:
                return await run_harbor_episode(
                    task_path=task_path,
                    model_base_url=model_base_url,
                    model_name=model_name,
                    config=self.runtime_config,
                    session_id=session_id,
                    max_model_len=max_model_len,
                    max_turns=max_turns_cfg,
                    actor_id=actor_id,
                    thinking_template=self.thinking_template,
                )
            finally:
                gate.release()

        future = _harbor_loop.run(_gated())
        return await asyncio.wrap_future(future)
