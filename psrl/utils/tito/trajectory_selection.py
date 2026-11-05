"""Select the TITO trajectories that may produce training rows.

The session snapshot is the authority for roles and capture. The trainer's current
configuration is only used when a snapshot predates the role metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

from psrl.utils.subagent_training import SubagentTrainingPolicy

CAPTURE_FULL = "full"
CAPTURE_TOKENS = "tokens"
_ROLE_FIELDS = ("thread_source", "capture", "parent_trajectory_id")


@dataclass(frozen=True)
class TrajectorySelection:
    """One trajectory that should become a rollout row."""

    trajectory: dict
    trainable_trajectory_id: int | None
    is_primary: bool
    thread_source: str
    agent_id: str | None
    parent_trajectory_id: int | None
    capture: str
    loss_weight: float


def _has_role_metadata(trajectory: dict) -> bool:
    return any(field in trajectory for field in _ROLE_FIELDS)


def _is_primary_role(trajectory: dict, main_agent_id: str) -> bool:
    role = str(trajectory.get("thread_source") or "main").strip().lower()
    if role == "main":
        return True
    if role != "compaction":
        return False
    agent_id = trajectory.get("agent_id")
    return agent_id in (None, "", "main", main_agent_id)


def _owned_records(trajectory: dict) -> list[dict]:
    trajectory_id = trajectory["trajectory_id"]
    owned = []
    for record in trajectory.get("records", []) or []:
        owner = record.get("owner_trajectory_id")
        if owner is None or int(owner) == trajectory_id:
            owned.append(record)
    return owned


def _live_trajectories(trajectories: list[dict], main_agent_id: str) -> list[dict]:
    """Keep the latest leaf for each identified main or subagent.

    A retry or a compaction can leave an earlier leaf in the snapshot. The later
    trajectory is the live one for that agent. A trajectory without an agent id is
    kept as-is: two unidentified parallel branches are distinct agents, and the
    tree cannot tell a retry from a second branch.
    """
    latest: dict[tuple[str, str], dict] = {}
    for trajectory in trajectories:
        role = str(trajectory.get("thread_source") or "main").strip().lower()
        agent_id = trajectory.get("agent_id")
        if role in ("main", "compaction"):
            is_main = role == "main" or agent_id in (None, "", main_agent_id)
            key = ("main", str(agent_id or main_agent_id)) if is_main else ("subagent", str(agent_id))
        elif role == "subagent" and agent_id:
            key = ("subagent", str(agent_id))
        else:
            continue
        current = latest.get(key)
        if current is None or trajectory["trajectory_id"] > current["trajectory_id"]:
            latest[key] = trajectory
    live_ids = {trajectory["trajectory_id"] for trajectory in latest.values()}
    return [
        trajectory
        for trajectory in trajectories
        if trajectory["trajectory_id"] in live_ids
        or str(trajectory.get("thread_source") or "main").strip().lower() == "auxiliary"
        or (
            str(trajectory.get("thread_source") or "main").strip().lower() == "subagent"
            and trajectory.get("agent_id") is None
        )
    ]


def annotate_session_contribution(
    session_data: dict,
    main_agent_id: str | None = None,
    primary: dict | None = None,
) -> int | None:
    """
    Mark each turn with whether it can contribute to the session reward.

    The main-close boundary is the storage sequence of the primary main leaf's
    last record. A subagent or subagent-compaction turn at or before that
    sequence may contribute; a turn after it is post-main and is masked. Main and
    main-compaction turns always contribute. A record without a sequence keeps the
    legacy coarse behavior and is treated as contributing.

    Args:
        session_data (dict): The expanded session snapshot. Mutated in place.
        main_agent_id (str | None): The parent agent identity, if known.
        primary (dict | None): The primary main trajectory, if already selected.

    Returns:
        int | None: The main-close sequence, or None for a legacy snapshot or a
            main trajectory with no sequenced records.
    """
    trajectories = session_data.get("trajectories")
    if not isinstance(trajectories, list):
        return None
    main_agent_id = str(main_agent_id or session_data.get("main_agent_id") or "main")
    if not any(_has_role_metadata(trajectory) for trajectory in trajectories):
        for trajectory in trajectories:
            for record in trajectory.get("records", []) or []:
                record.setdefault("contributes_to_session", True)
        session_data["main_close_sequence"] = None
        return None

    if primary is None:
        primary = max(
            (
                trajectory
                for trajectory in _live_trajectories(trajectories, main_agent_id)
                if _is_primary_role(trajectory, main_agent_id)
            ),
            key=lambda trajectory: trajectory["trajectory_id"],
            default=None,
        )
    sequences = (
        []
        if primary is None
        else [
            int(record["sequence"])
            for record in _owned_records(primary)
            if record.get("sequence") is not None
        ]
    )
    boundary = max(sequences) if sequences else None
    session_data["main_close_sequence"] = boundary

    for trajectory in trajectories:
        role = str(trajectory.get("thread_source") or "main").strip().lower()
        agent_id = trajectory.get("agent_id")
        is_main_role = role == "main" or (
            role == "compaction" and agent_id in (None, "", "main", main_agent_id)
        )
        is_auxiliary = role == "auxiliary"
        for record in trajectory.get("records", []) or []:
            if boundary is None or is_main_role:
                continues = True
            elif is_auxiliary:
                continues = False
            else:
                sequence = record.get("sequence")
                continues = True if sequence is None else int(sequence) <= boundary
            record["contributes_to_session"] = continues
    return boundary


def select_trainable_trajectories(
    session_data: dict,
    policy: SubagentTrainingPolicy,
    *,
    validate: bool = False,
) -> list[TrajectorySelection]:
    """
    Return the trajectories that may become training rows.

    A legacy snapshot has no role metadata and keeps the linear behavior: every
    trajectory is trainable and the last one is the session primary. A snapshot with
    role metadata uses the session policy and marks the latest main-role trajectory as
    primary.

    Args:
        session_data (dict): The TITO session snapshot.
        policy (SubagentTrainingPolicy): The session policy.
        validate (bool): Select with `val_mode` and skip training-only capture checks.

    Returns:
        list[TrajectorySelection]: Trainable trajectories, ordered by trajectory id.

    Raises:
        ValueError: If the snapshot has no primary, a trainable row lacks a full
            capture, or the policy selects no row at all.
    """
    trajectories = session_data.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError("TITO session snapshot has no trajectories list.")
    if not trajectories:
        return []

    if not any(_has_role_metadata(trajectory) for trajectory in trajectories):
        last = len(trajectories) - 1
        return [
            TrajectorySelection(
                trajectory=trajectory,
                trainable_trajectory_id=None,
                is_primary=index == last,
                thread_source="main",
                agent_id=None,
                parent_trajectory_id=None,
                capture=CAPTURE_FULL,
                loss_weight=1.0,
            )
            for index, trajectory in enumerate(trajectories)
        ]

    main_agent_id = str(session_data.get("main_agent_id") or "main")
    live_trajectories = _live_trajectories(trajectories, main_agent_id)
    primary = max(
        (trajectory for trajectory in live_trajectories if _is_primary_role(trajectory, main_agent_id)),
        key=lambda trajectory: trajectory["trajectory_id"],
        default=None,
    )
    if primary is None:
        raise ValueError(f"TITO session {session_data.get('session_id')!r} has no main trajectory.")
    annotate_session_contribution(session_data, main_agent_id, primary)

    selections: list[TrajectorySelection] = []
    for trajectory in live_trajectories:
        role = str(trajectory.get("thread_source") or "main").strip().lower()
        agent_id = trajectory.get("agent_id")
        capture = str(trajectory.get("capture") or CAPTURE_FULL).strip().lower()
        is_primary = trajectory["trajectory_id"] == primary["trajectory_id"]
        if not policy.is_trainable(
            role,
            agent_id,
            validate=validate,
            main_agent_id=main_agent_id,
        ):
            continue
        owned = _owned_records(trajectory)
        contributing = [record for record in owned if record.get("contributes_to_session", True)]
        if not contributing:
            if is_primary:
                raise ValueError(
                    f"TITO session {session_data.get('session_id')!r} has no contributing turn owned by "
                    f"its main trajectory {trajectory['trajectory_id']}."
                )
            continue
        if not validate and policy.require_primary_capture:
            if capture != CAPTURE_FULL:
                raise ValueError(
                    f"TITO trajectory {trajectory['trajectory_id']} is trainable but the router "
                    f"captured it as {capture!r}; expected 'full'."
                )
            token_only = [
                record
                for record in contributing
                if str(record.get("capture") or CAPTURE_FULL).strip().lower() != CAPTURE_FULL
            ]
            if token_only:
                raise ValueError(
                    f"TITO trajectory {trajectory['trajectory_id']} is trainable but has "
                    f"{len(token_only)} token-only contributing turn(s)."
                )
        selections.append(
            TrajectorySelection(
                trajectory=trajectory,
                trainable_trajectory_id=trajectory["trajectory_id"],
                is_primary=is_primary,
                thread_source=role,
                agent_id=agent_id,
                parent_trajectory_id=trajectory.get("parent_trajectory_id"),
                capture=capture,
                loss_weight=policy.loss_weight(role, agent_id, main_agent_id=main_agent_id),
            )
        )

    if not selections:
        raise ValueError(
            f"TITO session {session_data.get('session_id')!r} produced no trainable trajectory "
            f"under subagent_training mode {policy.mode!r}."
        )
    if primary["trajectory_id"] not in {selection.trajectory["trajectory_id"] for selection in selections}:
        raise ValueError(
            f"TITO session {session_data.get('session_id')!r} has a main trajectory that the "
            "subagent training policy does not select."
        )
    return selections
