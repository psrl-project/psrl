"""Tests for the subagent training policy and trajectory selection."""

import pytest
from psrl.utils.subagent_training import SubagentTrainingPolicy
from psrl.utils.tito.trajectory_selection import select_trainable_trajectories


def _session(*trajectories: dict) -> dict:
    return {
        "session_id": "sid",
        "main_agent_id": "main",
        "subagent_training": SubagentTrainingPolicy().to_dict(),
        "trajectories": list(trajectories),
    }


def _trajectory(
    trajectory_id: int,
    *,
    role: str = "main",
    agent_id: str | None = None,
    capture: str = "full",
    parent_trajectory_id: int | None = None,
    output_token_ids: list[int] | None = None,
    record_capture: str = "full",
) -> dict:
    return {
        "trajectory_id": trajectory_id,
        "thread_source": role,
        "agent_id": agent_id,
        "parent_trajectory_id": parent_trajectory_id,
        "capture": capture,
        "accumulated_token_ids": [1, 2, 3],
        "records": [
            {
                "prompt_token_count": 2,
                "output_token_ids": output_token_ids or [3],
                "output_logprobs": [[-0.1, 3]],
                "owner_trajectory_id": trajectory_id,
                "capture": record_capture,
            }
        ],
    }


def test_policy_defaults_to_main_only():
    policy = SubagentTrainingPolicy()

    assert policy.is_trainable("main", "main")
    assert not policy.is_trainable("subagent", "sub-1")
    assert not policy.is_trainable("auxiliary", "aux-1")
    assert policy.capture_for("main", "main") == "full"
    assert policy.capture_for("subagent", "sub-1") == "tokens"


def test_all_selects_every_subagent():
    policy = SubagentTrainingPolicy.from_mapping({"mode": "all"})

    assert policy.is_trainable("subagent", "sub-1")
    assert policy.is_trainable("subagent", None)
    assert not policy.is_trainable("auxiliary", "aux-1")


def test_selected_matches_ids_and_patterns():
    policy = SubagentTrainingPolicy.from_mapping({"mode": "selected", "train_agents": ["reviewer", "worker-*"]})

    assert policy.is_trainable("subagent", "reviewer")
    assert policy.is_trainable("subagent", "worker-2")
    assert not policy.is_trainable("subagent", "researcher")


def test_selected_requires_a_pattern():
    with pytest.raises(ValueError, match="train_agents"):
        SubagentTrainingPolicy.from_mapping({"mode": "selected"})


def test_compaction_follows_the_agent_being_compacted():
    policy = SubagentTrainingPolicy.from_mapping({"mode": "main_only"})

    assert policy.is_trainable("compaction", "main", main_agent_id="main")
    assert policy.is_trainable("compaction", None, main_agent_id="main")
    assert not policy.is_trainable("compaction", "sub-1", main_agent_id="main")
    selected = SubagentTrainingPolicy.from_mapping({"mode": "selected", "train_agents": ["sub-1"]})
    assert selected.is_trainable("compaction", "sub-1", main_agent_id="main")


def test_validation_uses_val_mode():
    policy = SubagentTrainingPolicy.from_mapping({"mode": "main_only", "val_mode": "all"})

    assert not policy.is_trainable("subagent", "sub-1")
    assert policy.is_trainable("subagent", "sub-1", validate=True)
    assert policy.capture_for("subagent", "sub-1", validate=True) == "full"


def test_per_agent_reward_is_not_implemented():
    with pytest.raises(ValueError, match="per_agent"):
        SubagentTrainingPolicy.from_mapping({"reward": "per_agent"})


def test_invalid_policy_fields_are_rejected():
    with pytest.raises(ValueError, match="mode"):
        SubagentTrainingPolicy.from_mapping({"mode": "unknown"})
    with pytest.raises(ValueError, match="shared_prefix"):
        SubagentTrainingPolicy.from_mapping({"shared_prefix": "unknown"})


def test_policy_wire_form_round_trips():
    policy = SubagentTrainingPolicy.from_mapping(
        {
            "mode": "selected",
            "train_agents": ["worker-*"],
            "loss_weight": {"main": 2.0, "subagent": 0.5},
            "val_mode": "all",
        }
    )

    assert SubagentTrainingPolicy.from_mapping(policy.to_dict()) == policy


def test_legacy_snapshot_keeps_every_trajectory_trainable():
    session = {
        "session_id": "sid",
        "trajectories": [
            {"trajectory_id": 0, "accumulated_token_ids": [1], "records": [{"prompt_token_count": 0}]},
            {"trajectory_id": 1, "accumulated_token_ids": [2], "records": [{"prompt_token_count": 0}]},
        ],
    }

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy())

    assert [selection.trajectory["trajectory_id"] for selection in selections] == [0, 1]
    assert [selection.is_primary for selection in selections] == [False, True]
    assert all(selection.trainable_trajectory_id is None for selection in selections)


def test_main_only_emits_only_the_main_row():
    session = _session(
        _trajectory(0, agent_id="main"),
        _trajectory(1, role="subagent", agent_id="sub-1", capture="tokens", parent_trajectory_id=0),
    )

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy())

    assert [selection.trajectory["trajectory_id"] for selection in selections] == [0]
    assert selections[0].is_primary
    assert selections[0].loss_weight == 1.0


def test_all_emits_main_and_subagent_rows_with_the_main_primary():
    session = _session(
        _trajectory(0, agent_id="main"),
        _trajectory(1, role="subagent", agent_id="sub-1", capture="full", parent_trajectory_id=0),
    )

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy.from_mapping({"mode": "all"}))

    assert [selection.trajectory["trajectory_id"] for selection in selections] == [0, 1]
    assert [selection.is_primary for selection in selections] == [True, False]


def test_latest_main_role_leaf_is_primary():
    session = _session(
        _trajectory(0, agent_id="main"),
        _trajectory(1, role="compaction", agent_id="main", parent_trajectory_id=0),
    )

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy())

    assert [selection.trajectory["trajectory_id"] for selection in selections] == [1]
    assert selections[0].is_primary


def test_token_only_trainable_trajectory_fails_the_sample():
    session = _session(_trajectory(0, agent_id="main", record_capture="tokens"))

    with pytest.raises(ValueError, match="token-only"):
        select_trainable_trajectories(session, SubagentTrainingPolicy())


def test_validation_does_not_require_full_capture():
    session = _session(_trajectory(0, agent_id="main", capture="tokens", record_capture="tokens"))

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy(), validate=True)

    assert len(selections) == 1
    assert selections[0].is_primary


def test_session_without_a_main_trajectory_is_rejected():
    session = _session(_trajectory(0, role="subagent", agent_id="sub-1"))

    with pytest.raises(ValueError, match="no main trajectory"):
        select_trainable_trajectories(session, SubagentTrainingPolicy.from_mapping({"mode": "all"}))


def test_off_selects_no_row():
    session = _session(_trajectory(0, agent_id="main"))

    with pytest.raises(ValueError, match="no trainable trajectory"):
        select_trainable_trajectories(session, SubagentTrainingPolicy.from_mapping({"mode": "off"}))


def test_compaction_loss_weight_follows_the_compacted_agent():
    policy = SubagentTrainingPolicy.from_mapping(
        {"mode": "all", "loss_weight": {"main": 2.0, "subagent": 0.5}}
    )

    assert policy.loss_weight("compaction", "main", main_agent_id="main") == 2.0
    assert policy.loss_weight("compaction", "sub-1", main_agent_id="main") == 0.5


def test_overrides_merge_loss_weight_instead_of_replacing_it():
    base = SubagentTrainingPolicy.from_mapping({"loss_weight": {"main": 2.0, "subagent": 1.0}})

    overridden = base.overrides({"loss_weight": {"subagent": 0.25}})

    assert overridden.loss_weight_main == 2.0
    assert overridden.loss_weight_subagent == 0.25


def test_a_retried_agent_keeps_only_its_latest_live_leaf():
    session = _session(
        _trajectory(0, agent_id="main"),
        _trajectory(1, role="subagent", agent_id="sub-1", capture="full", parent_trajectory_id=0),
        _trajectory(2, role="subagent", agent_id="sub-1", capture="full", parent_trajectory_id=0),
    )

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy.from_mapping({"mode": "all"}))

    assert [selection.trajectory["trajectory_id"] for selection in selections] == [0, 2]


def test_a_compaction_replaces_the_pre_compaction_main_leaf():
    session = _session(
        _trajectory(0, agent_id="main"),
        _trajectory(1, role="compaction", agent_id="main", parent_trajectory_id=0),
    )

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy())

    assert [selection.trajectory["trajectory_id"] for selection in selections] == [1]
    assert selections[0].is_primary


def test_parallel_subagents_with_distinct_ids_are_both_kept():
    session = _session(
        _trajectory(0, agent_id="main"),
        _trajectory(1, role="subagent", agent_id="sub-1", capture="full", parent_trajectory_id=0),
        _trajectory(2, role="subagent", agent_id="sub-2", capture="full", parent_trajectory_id=0),
    )

    selections = select_trainable_trajectories(session, SubagentTrainingPolicy.from_mapping({"mode": "all"}))

    assert {selection.trajectory["trajectory_id"] for selection in selections} == {0, 1, 2}
