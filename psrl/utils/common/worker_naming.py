"""
Worker naming utilities for PSRL.

This module is the single source of truth for all worker name construction
logic used across the trainer, PS, gen, and train worker layers. No other
module should construct worker name strings directly.

NIXL name string constants are imported from psrl.utils.common.nixl_names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from psrl.utils.common.nixl_names import (
    NIXL_GEN_CLIENT_PREFIX,
    NIXL_PS_CLIENT_PREFIX,
    NIXL_TRAIN_CLIENT_PREFIX,
)

WorkerRole = Literal["rollout", "actor", "validate"]


@dataclass(frozen=True)
class WorkerKey:
    """
    Structured identifier for a single PSRL worker.

    Usable as a dict key (frozen dataclass generates __hash__ and __eq__).

    Attributes:
        role (WorkerRole): One of 'rollout', 'actor', or 'validate'.
        instance_id (int): 0-based instance index. Actor workers always use 0.
        rank (int): Tensor-parallel rank within the instance.
            For actor workers this is the global rank.
    """

    role: WorkerRole
    instance_id: int
    rank: int

    def __post_init__(self) -> None:
        """Validate field invariants at construction time."""
        if self.role == "actor":
            assert self.instance_id == 0, f"Actor workers must have instance_id=0, got: {self.instance_id!r}."

    def to_trainer_name(self) -> str:
        """
        Return the trainer-layer string identifier for this worker.

        This string is used only for logging and debugging. All dict keys
        in ray_trainer.py use WorkerKey directly.

        Returns:
            str: Human-readable name, e.g. 'rollout_I0_R2'.
        """
        return f"{self.role}_I{self.instance_id}_R{self.rank}"

    def to_nixl_client_name(self, n_rollout_instances: int = 0) -> str:
        """
        Return the NIXL client name registered with the meta server.

        For validate workers, the NIXL instance_id is offset by
        n_rollout_instances to avoid collision with rollout workers.
        This offset is encapsulated here. Callers must not compute it
        themselves.

        Args:
            n_rollout_instances (int): Total number of rollout instances.
                Only used when role='validate' and ignored for 'rollout' and 'actor'. Defaults to 0.

        Returns:
            str: NIXL client name, e.g. 'NIXLGenClient_I2_R0'.
        """
        if self.role == "rollout":
            return f"{NIXL_GEN_CLIENT_PREFIX}_I{self.instance_id}_R{self.rank}"
        elif self.role == "validate":
            nixl_instance_id = n_rollout_instances + self.instance_id
            return f"{NIXL_GEN_CLIENT_PREFIX}_I{nixl_instance_id}_R{self.rank}"
        elif self.role == "actor":
            return f"{NIXL_TRAIN_CLIENT_PREFIX}_{self.rank}"
        else:
            raise AssertionError(
                f"to_nixl_client_name: unhandled role {self.role!r}. Expected 'rollout', 'actor', or 'validate'."
            )


# --- PS naming helpers ---


def ps_agent_name(rank: int) -> str:
    """
    Return the NIXL agent name for a PS storage worker.

    Args:
        rank (int): PS worker rank.

    Returns:
        str: Agent name, e.g. 'NIXLPSClient_0'.
    """
    return f"{NIXL_PS_CLIENT_PREFIX}_{rank}"


def ps_client_push_name(rank: int) -> str:
    """
    Return the NIXL client name for the push-side PS storage client.

    Args:
        rank (int): PS worker rank.

    Returns:
        str: Push client name, e.g. 'NIXLPSClient_0_for_push'.
    """
    return f"{NIXL_PS_CLIENT_PREFIX}_{rank}_for_push"


def ps_client_pull_name(rank: int) -> str:
    """
    Return the NIXL client name for the pull-side PS storage client.

    Args:
        rank (int): PS worker rank.

    Returns:
        str: Pull client name, e.g. 'NIXLPSClient_0_for_pull'.
    """
    return f"{NIXL_PS_CLIENT_PREFIX}_{rank}_for_pull"


# --- Gen / train worker naming helpers ---


def gen_client_name(instance_id: int, rank: int) -> str:
    """
    Return the NIXL client name for a gen worker.

    Gen workers receive a pre-computed instance_id that already includes
    any validate offset. They call this helper directly rather than
    constructing a WorkerKey, since they do not know their own role.

    Args:
        instance_id (int): Final NIXL instance_id (offset already applied
            by the caller for validate workers).
        rank (int): Worker rank within the instance.

    Returns:
        str: Gen client name, e.g. 'NIXLGenClient_I1_R0'.
    """
    return f"{NIXL_GEN_CLIENT_PREFIX}_I{instance_id}_R{rank}"


def train_client_name(rank: int) -> str:
    """
    Return the NIXL client name for a train (actor) worker.

    Args:
        rank (int): Global rank of the train worker.

    Returns:
        str: Train client name, e.g. 'NIXLTrainClient_3'.
    """
    return f"{NIXL_TRAIN_CLIENT_PREFIX}_{rank}"
