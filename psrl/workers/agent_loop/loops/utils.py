from enum import Enum
from importlib import import_module

from omegaconf import DictConfig

AGENT_LOOP_REGISTRY: dict[str, dict] = {}

# Lookup imports the module so `@register` runs without pulling optional loops at worker start.
_BUILTIN_AGENT_LOOP_MODULES = {
    "generate_only_agent": "psrl.workers.agent_loop.loops.generate_agent_loop",
    "multi_turn_agent": "psrl.workers.agent_loop.loops.multi_turn_agent_loop",
    "multi_turn_completion_agent": "psrl.workers.agent_loop.loops.multi_turn_completion_agent_loop",
    "mini_swe_agent": "psrl.workers.agent_loop.loops.mini_swe_agent_loop_v1",
    "mlgym_agent": "psrl.workers.agent_loop.loops.mlgym_agent_loop",
}


def register(agent_name: str):
    """Register an agent loop class with the given name.

    Args:
        agent_name (str): Name to register the agent loop under.

    Returns:
        function: Decorator function for registering the agent loop class.
    """
    from psrl.workers.agent_loop.loops.base_agent_loop import AgentLoopBase

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        AGENT_LOOP_REGISTRY[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


def get_agent_loop_config(agent_name: str) -> dict:
    """
    Return the Hydra config for a named agent loop.

    Built-in loops are imported on first lookup so `@register` can populate
    `AGENT_LOOP_REGISTRY` without loading optional implementations at import time.

    Args:
        agent_name (str): Registered agent loop name.

    Returns:
        dict: Hydra config containing `_target_` and any YAML overlays.

    Raises:
        ValueError: If the name is not a built-in loop and was not loaded from YAML.
    """
    if agent_name not in AGENT_LOOP_REGISTRY:
        module_name = _BUILTIN_AGENT_LOOP_MODULES.get(agent_name)
        if module_name is not None:
            import_module(module_name)
    if agent_name not in AGENT_LOOP_REGISTRY:
        raise ValueError(
            f"Unregistered agent loop: name={agent_name!r}, available={list(AGENT_LOOP_REGISTRY.keys())!r}."
        )
    return AGENT_LOOP_REGISTRY[agent_name]


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class TerminateReason(Enum):
    """Why an agent-loop trajectory stopped."""

    FINISHED = "finished"
    MAX_RESPONSE_LENGTH_EXCEEDED = "max_response_length_exceeded"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"
    # Wall-clock exhaustion is distinct from the configured turn cap in metrics.
    AGENT_TIMEOUT = "agent_timeout"
    # The agent finished normally but grading failed or timed out, so the trajectory is
    # complete and trainable while carrying no verifier reward.
    VERIFIER_ERROR = "verifier_error"
    # An environment step exceeded `agent.env.step_timeout`. Both call sites build a
    # trajectory from the turns completed before the timeout, so this keeps its data.
    ENV_TIMEOUT = "env_timeout"
    TRAJECTORY_TIMEOUT = "trajectory_timeout"
    ABORTED = "aborted"
    UNKNOWN = "unknown"
    ROLLOUT_ERROR = "rollout_error"

    @property
    def is_successful(self) -> bool:
        """Return whether the trajectory has usable training content.

        Every member here is a *budget* that ran out (context window, turn cap, wall
        clock, env step) or a post-run grading failure. In each case the turns produced
        before the limit are valid on-policy data, so the trajectory is truncated and
        trained rather than discarded.
        """
        return self in (
            TerminateReason.FINISHED,
            TerminateReason.MAX_TURNS_EXCEEDED,
            TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
            TerminateReason.AGENT_TIMEOUT,
            TerminateReason.VERIFIER_ERROR,
            TerminateReason.ENV_TIMEOUT,
        )

    @property
    def is_ungraded(self) -> bool:
        """Return whether the episode ran but never received a verifier score.

        The trajectory is real on-policy data, so `is_successful` keeps it, but no
        grader ever looked at it. A reward of 0 here is the absence of a measurement,
        not a measured failure, and nothing downstream can tell the two apart once the
        empty reward dict has defaulted to 0.0.

        Training it as a zero is worse than dropping it: siblings of the same task in
        the same GRPO group split 0.0 against 1.0 purely on whether a container timed
        out, which is a gradient of pure infrastructure noise pointing in a direction
        the policy cannot influence.
        """
        return self is TerminateReason.VERIFIER_ERROR

    @property
    def is_budget_truncated(self) -> bool:
        """Return whether a harness budget cut the episode off mid-work.

        These trajectories are still valid on-policy data, so `is_successful` keeps
        them, but their reward reports the cutoff rather than the quality of the
        model's choices. Grading a run that was never allowed to finish as a failure
        makes the group-relative advantage penalise every token in it, and under
        `token-mean` a long truncated trajectory outweighs many short ones. The
        cheapest way for the policy to shed that penalty is to emit fewer tokens per
        turn, which spends the turn cap faster and truncates more often.

        Measured over 1983 episodes of GRPO-sciaccel-Qwen35-4B-v2_repair-L1: tokens
        per turn fell 1125 to 327, `max_turns_exceeded` rose from 29% to 39%, and the
        score collapsed from 0.573 at step 11 to 0.078 at step 16.

        `AGENT_TIMEOUT` and `ENV_TIMEOUT` are excluded deliberately, because they are
        infrastructure faults rather than budget exhaustion. `VERIFIER_ERROR` is
        excluded too and handled by `is_ungraded`, which masks it for a different
        reason: its tokens are honest, but its reward was never measured.
        """
        return self in (
            TerminateReason.MAX_TURNS_EXCEEDED,
            TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED,
        )

    @property
    def is_timeout(self) -> bool:
        """Return whether a timeout stopped the trajectory before it produced data.

        Only `TRAJECTORY_TIMEOUT` qualifies: it fires from
        `run_with_termination_handling`, which owns no partial output. `AGENT_TIMEOUT`
        and `ENV_TIMEOUT` are timeouts too, but their loops return a finalized
        trajectory, and this property feeds `needs_worker_retry` -- re-running an
        episode whose turns were already accepted would duplicate them.
        """
        return self is TerminateReason.TRAJECTORY_TIMEOUT

    @property
    def is_error(self) -> bool:
        """Return whether the slot was wasted by a transient error."""
        return self in (TerminateReason.ROLLOUT_ERROR, TerminateReason.UNKNOWN)

    @property
    def is_aborted(self) -> bool:
        """Return whether PSRL intentionally aborted the trajectory."""
        return self is TerminateReason.ABORTED

    def needs_worker_retry(self) -> bool:
        """Return whether the worker should re-run the episode in place.

        Only reasons that produced no usable data qualify, since `worker.py` nulls the
        output for anything this returns True for. Retrying a data-bearing reason would
        both discard the trajectory and duplicate the work.

        Note this is inert at the default `rollout.agent.retry_limit=1`, which yields a
        single attempt.
        """
        return self.is_timeout or self.is_error

    def needs_manager_retry(self) -> bool:
        """Return whether manager must refill a wasted buffer slot.

        A train buffer entry is all-or-nothing: `AgentLoopManager` occupies it only once
        all `alg_rollout_n` trajectories arrive, and `PSManager.abort_requests` clears the
        whole entry when fewer remain. So every reason that reaches the worker without
        usable data must refill the group, or the surviving siblings wait forever.

        `ABORTED` is the sole exception: PSManager raised it after already clearing the
        entry, so requesting another refill would double-count the failure.
        """
        return not self.is_successful and not self.is_aborted
