"""Compact TITO snapshot reconstruction and per-turn contribution marking."""

import pytest
from psrl.utils.subagent_training import SubagentTrainingPolicy
from psrl.utils.tito.snapshot import normalize_session_snapshot
from psrl.utils.tito.training_data import build_training_data
from psrl.utils.tito.trajectory_selection import select_trainable_trajectories


def _record(prompt: int, token: int, sequence: int, owner: int) -> dict:
    return {
        "prompt_token_count": prompt,
        "output_token_ids": [token],
        "output_logprobs": [[-0.1, token]],
        "owner_trajectory_id": owner,
        "capture": "full",
        "sequence": sequence,
    }


def _compact_session() -> dict:
    return {
        "session_id": "sid",
        "max_trim_tokens": 0,
        "format": "compact",
        "nodes": [
            {
                "id": 0,
                "hash": "root",
                "stored": True,
                "parent": None,
                "num_tokens": 2,
                "token_start": 0,
                "token_ids": [1, 2],
                "sequence": 0,
                "trajectory_ids": [0, 1],
                "owner_trajectory_id": 0,
                "capture": "full",
                "turn_record": _record(1, 2, 0, 0),
            },
            {
                "id": 1,
                "hash": "a",
                "stored": True,
                "parent": 0,
                "num_tokens": 4,
                "token_start": 2,
                "token_ids": [3, 4],
                "sequence": 1,
                "trajectory_ids": [0],
                "owner_trajectory_id": 0,
                "capture": "full",
                "turn_record": _record(3, 4, 1, 0),
            },
            {
                "id": 2,
                "hash": "b",
                "stored": True,
                "parent": 0,
                "num_tokens": 4,
                "token_start": 2,
                "token_ids": [5, 6],
                "sequence": 2,
                "trajectory_ids": [1],
                "owner_trajectory_id": 1,
                "capture": "full",
                "turn_record": _record(3, 6, 2, 1),
            },
        ],
        "leaves": [
            {
                "trajectory_id": 0,
                "node_id": 1,
                "path_node_ids": [0, 1],
                "thread_source": "main",
                "agent_id": "main",
                "capture": "full",
            },
            {
                "trajectory_id": 1,
                "node_id": 2,
                "path_node_ids": [0, 2],
                "thread_source": "subagent",
                "agent_id": "sub-1",
                "parent_agent_id": "main",
                "capture": "full",
            },
        ],
        "subagent_training": SubagentTrainingPolicy.from_mapping({"mode": "all"}).to_dict(),
        "main_agent_id": "main",
    }


def test_compact_snapshot_reconstructs_expanded_trajectories():
    expanded = normalize_session_snapshot(_compact_session())

    assert expanded["format"] == "expanded"
    assert expanded["source_format"] == "compact"
    assert expanded["trajectories"][0]["accumulated_token_ids"] == [1, 2, 3, 4]
    assert expanded["trajectories"][1]["accumulated_token_ids"] == [1, 2, 5, 6]
    assert [record["sequence"] for record in expanded["trajectories"][0]["records"]] == [0, 1]
    assert len(expanded["tree"]["nodes"]) == 3
    assert len(expanded["tree"]["leaves"]) == 2


def test_compact_snapshot_truncates_a_divergent_parent_tail():
    session = {
        "session_id": "sid",
        "max_trim_tokens": 1,
        "format": "compact",
        "nodes": [
            {
                "id": 0,
                "hash": "root",
                "stored": True,
                "parent": None,
                "num_tokens": 4,
                "token_start": 0,
                "token_ids": [1, 2, 3, 4],
                "sequence": 0,
                "turn_record": _record(1, 4, 0, 0),
            },
            {
                "id": 1,
                "hash": "trimmed",
                "stored": True,
                "parent": 0,
                "num_tokens": 5,
                "token_start": 3,
                "token_ids": [9, 5],
                "sequence": 1,
                "turn_record": _record(3, 5, 1, 1),
            },
        ],
        "leaves": [
            {
                "trajectory_id": 0,
                "node_id": 0,
                "path_node_ids": [0],
                "thread_source": "main",
                "agent_id": "main",
                "capture": "full",
            },
            {
                "trajectory_id": 1,
                "node_id": 1,
                "path_node_ids": [0, 1],
                "thread_source": "subagent",
                "agent_id": "sub-1",
                "parent_agent_id": "main",
                "capture": "full",
            },
        ],
    }

    expanded = normalize_session_snapshot(session)

    assert expanded["trajectories"][0]["accumulated_token_ids"] == [1, 2, 3, 4]
    assert expanded["trajectories"][1]["accumulated_token_ids"] == [1, 2, 3, 9, 5]


def test_compact_snapshot_rejects_a_start_past_the_path():
    session = _compact_session()
    session["nodes"][1]["token_start"] = 99

    with pytest.raises(ValueError, match="starts at token"):
        normalize_session_snapshot(session)


def test_unknown_snapshot_format_is_rejected():
    with pytest.raises(ValueError, match="Unsupported TITO snapshot format"):
        normalize_session_snapshot({"format": "future", "trajectories": []})


def test_an_absent_format_is_expanded():
    expanded = normalize_session_snapshot({"session_id": "sid", "trajectories": [], "tree": {}})

    assert expanded["format"] == "expanded"


def test_mixed_trajectory_marks_only_post_main_turns():
    session = {
        "session_id": "sid",
        "main_agent_id": "main",
        "subagent_training": SubagentTrainingPolicy.from_mapping({"mode": "all"}).to_dict(),
        "trajectories": [
            {
                "trajectory_id": 0,
                "thread_source": "main",
                "agent_id": "main",
                "capture": "full",
                "accumulated_token_ids": [1, 2, 3],
                "records": [_record(1, 2, 0, 0), _record(2, 3, 2, 0)],
            },
            {
                "trajectory_id": 1,
                "thread_source": "subagent",
                "agent_id": "sub-1",
                "parent_agent_id": "main",
                "capture": "full",
                "accumulated_token_ids": [1, 2, 9, 4, 5],
                "records": [_record(1, 2, 0, 0), _record(3, 4, 1, 1), _record(4, 5, 3, 1)],
            },
        ],
    }
    selections = select_trainable_trajectories(
        session,
        SubagentTrainingPolicy.from_mapping({"mode": "all"}),
    )
    subagent = next(selection for selection in selections if selection.thread_source == "subagent")
    built = build_training_data(
        subagent.trajectory["accumulated_token_ids"],
        subagent.trajectory["records"],
        trainable_trajectory_id=subagent.trainable_trajectory_id,
    )

    assert session["main_close_sequence"] == 2
    assert [record["contributes_to_session"] for record in subagent.trajectory["records"]] == [
        True,
        True,
        False,
    ]
    assert built["num_turns"] == 2
    assert built["trainable_turns"] == 1
    assert built["response_mask"] == [1, 0]


def test_a_fully_post_main_trajectory_is_not_selected():
    session = {
        "session_id": "sid",
        "main_agent_id": "main",
        "subagent_training": SubagentTrainingPolicy.from_mapping({"mode": "all"}).to_dict(),
        "trajectories": [
            {
                "trajectory_id": 0,
                "thread_source": "main",
                "agent_id": "main",
                "capture": "full",
                "accumulated_token_ids": [1, 2],
                "records": [_record(1, 2, 0, 0)],
            },
            {
                "trajectory_id": 1,
                "thread_source": "subagent",
                "agent_id": "sub-1",
                "parent_agent_id": "main",
                "capture": "full",
                "accumulated_token_ids": [1, 2, 3],
                "records": [_record(1, 2, 0, 0), _record(3, 3, 2, 1)],
            },
        ],
    }

    selections = select_trainable_trajectories(
        session,
        SubagentTrainingPolicy.from_mapping({"mode": "all"}),
    )

    assert [selection.agent_id for selection in selections] == ["main"]


def test_post_main_token_only_turn_does_not_fail_a_mixed_trajectory():
    session = {
        "session_id": "sid",
        "main_agent_id": "main",
        "subagent_training": SubagentTrainingPolicy.from_mapping(
            {"mode": "all", "require_primary_capture": True}
        ).to_dict(),
        "trajectories": [
            {
                "trajectory_id": 0,
                "thread_source": "main",
                "agent_id": "main",
                "capture": "full",
                "accumulated_token_ids": [1, 2, 3],
                "records": [_record(1, 2, 0, 0), _record(2, 3, 2, 0)],
            },
            {
                "trajectory_id": 1,
                "thread_source": "subagent",
                "agent_id": "sub-1",
                "parent_agent_id": "main",
                "capture": "full",
                "accumulated_token_ids": [1, 2, 9, 3, 4],
                "records": [
                    _record(1, 2, 0, 0),
                    {**_record(3, 3, 1, 1), "capture": "full"},
                    {**_record(4, 4, 3, 1), "capture": "tokens", "output_logprobs": None},
                ],
            },
        ],
    }

    selections = select_trainable_trajectories(
        session,
        SubagentTrainingPolicy.from_mapping({"mode": "all", "require_primary_capture": True}),
    )

    assert [selection.agent_id for selection in selections] == ["main", "sub-1"]
