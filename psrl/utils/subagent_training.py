"""Subagent training policy shared by the session router and agent loops."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any

_MODES = frozenset({"off", "main_only", "all", "selected"})
_SHARED_PREFIX = frozenset({"owner", "context"})
_REWARD_MODES = frozenset({"session", "per_agent"})


def _normalize_mode(value: Any, default: str) -> str:
    mode = str(value or default).strip().lower()
    if mode not in _MODES:
        raise ValueError(f"subagent_training mode must be one of {sorted(_MODES)}, got {value!r}.")
    return mode


def _normalize_agents(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class SubagentTrainingPolicy:
    """Session-scoped policy for selecting and capturing agent trajectories."""

    mode: str = "main_only"
    train_agents: tuple[str, ...] = ()
    shared_prefix: str = "owner"
    reward: str = "session"
    loss_weight_main: float = 1.0
    loss_weight_subagent: float = 1.0
    require_primary_capture: bool = True
    val_mode: str = "main_only"

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"subagent_training mode must be one of {sorted(_MODES)}, got {self.mode!r}.")
        if self.val_mode not in _MODES:
            raise ValueError(f"subagent_training val_mode must be one of {sorted(_MODES)}, got {self.val_mode!r}.")
        if self.shared_prefix not in _SHARED_PREFIX:
            raise ValueError(
                f"subagent_training shared_prefix must be one of {sorted(_SHARED_PREFIX)}, "
                f"got {self.shared_prefix!r}."
            )
        if self.reward not in _REWARD_MODES:
            raise ValueError(f"subagent_training reward must be one of {sorted(_REWARD_MODES)}, got {self.reward!r}.")
        if self.reward != "session":
            raise ValueError("subagent_training reward='per_agent' is not implemented; use 'session'.")
        if self.mode == "selected" and not self.train_agents:
            raise ValueError("subagent_training mode='selected' requires at least one train_agents entry.")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> SubagentTrainingPolicy:
        """
        Build a policy from a resolved config or request mapping.

        Args:
            mapping (Mapping[str, Any] | None): Policy fields. Missing fields keep defaults.

        Returns:
            SubagentTrainingPolicy: The resolved policy.
        """
        mapping = dict(mapping or {})
        loss_weight = mapping.get("loss_weight") or {}
        if not isinstance(loss_weight, Mapping):
            loss_weight = {}
        return cls(
            mode=_normalize_mode(mapping.get("mode"), "main_only"),
            train_agents=_normalize_agents(mapping.get("train_agents")),
            shared_prefix=str(mapping.get("shared_prefix") or "owner").strip().lower(),
            reward=str(mapping.get("reward") or "session").strip().lower(),
            loss_weight_main=float(loss_weight.get("main", 1.0)),
            loss_weight_subagent=float(loss_weight.get("subagent", 1.0)),
            require_primary_capture=bool(mapping.get("require_primary_capture", True)),
            val_mode=_normalize_mode(mapping.get("val_mode"), "main_only"),
        )

    def overrides(self, mapping: Mapping[str, Any] | None) -> SubagentTrainingPolicy:
        """
        Return a copy with the given fields replaced.

        Args:
            mapping (Mapping[str, Any] | None): Fields to override.

        Returns:
            SubagentTrainingPolicy: The updated policy.
        """
        mapping = dict(mapping or {})
        if not mapping:
            return self
        merged = self.to_dict()
        for key, value in mapping.items():
            if key == "loss_weight" and isinstance(value, Mapping):
                merged["loss_weight"] = {**merged["loss_weight"], **dict(value)}
            else:
                merged[key] = value
        return SubagentTrainingPolicy.from_mapping(merged)

    def to_dict(self) -> dict[str, Any]:
        """Return the wire form used in the session creation body."""
        return {
            "mode": self.mode,
            "train_agents": list(self.train_agents),
            "shared_prefix": self.shared_prefix,
            "reward": self.reward,
            "loss_weight": {
                "main": self.loss_weight_main,
                "subagent": self.loss_weight_subagent,
            },
            "require_primary_capture": self.require_primary_capture,
            "val_mode": self.val_mode,
        }

    def is_trainable(
        self,
        thread_source: str,
        agent_id: str | None,
        *,
        validate: bool = False,
        main_agent_id: str | None = None,
    ) -> bool:
        """
        Return whether one trajectory may produce a training row.

        Args:
            thread_source (str): `main`, `subagent`, `auxiliary`, or `compaction`.
            agent_id (str | None): Stable agent identity.
            validate (bool): Use `val_mode` instead of `mode`.
            main_agent_id (str | None): Identity of the session's parent agent. A
                compaction call belongs to the parent when it carries this id, no id,
                or the `main` fallback.

        Returns:
            bool: Whether the trajectory is trainable.
        """
        mode = self.val_mode if validate else self.mode
        if mode == "off":
            return False
        role = str(thread_source or "main").strip().lower()
        if role == "compaction":
            parent_ids = {None, "", "main"}
            if main_agent_id is not None:
                parent_ids.add(main_agent_id)
            role = "main" if agent_id in parent_ids else "subagent"
        if role == "main":
            return True
        if role == "auxiliary":
            return False
        if role != "subagent":
            return False
        if mode == "main_only":
            return False
        if mode == "all":
            return True
        if agent_id is None:
            return False
        return any(fnmatch(agent_id, pattern) for pattern in self.train_agents)

    def capture_for(
        self,
        thread_source: str,
        agent_id: str | None,
        *,
        validate: bool = False,
        main_agent_id: str | None = None,
    ) -> str:
        """
        Return `full` for a trainable trajectory and `tokens` otherwise.

        Args:
            thread_source (str): `main`, `subagent`, `auxiliary`, or `compaction`.
            agent_id (str | None): Stable agent identity.
            validate (bool): Use `val_mode` instead of `mode`.
            main_agent_id (str | None): Identity of the session's parent agent.

        Returns:
            str: `full` or `tokens`.
        """
        trainable = self.is_trainable(
            thread_source,
            agent_id,
            validate=validate,
            main_agent_id=main_agent_id,
        )
        return "full" if trainable else "tokens"

    def loss_weight(
        self,
        thread_source: str,
        agent_id: str | None = None,
        *,
        main_agent_id: str | None = None,
    ) -> float:
        """
        Return the loss weight for one trajectory role.

        Args:
            thread_source (str): `main`, `subagent`, `auxiliary`, or `compaction`.
            agent_id (str | None): Stable agent identity, used to resolve compaction.
            main_agent_id (str | None): Identity of the session's parent agent.

        Returns:
            float: The configured loss weight.
        """
        role = str(thread_source or "main").strip().lower()
        if role == "compaction":
            parent_ids = {None, "", "main"}
            if main_agent_id is not None:
                parent_ids.add(main_agent_id)
            role = "main" if agent_id in parent_ids else "subagent"
        if role == "main":
            return self.loss_weight_main
        return self.loss_weight_subagent
