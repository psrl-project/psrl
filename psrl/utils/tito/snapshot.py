"""Normalize TITO session snapshots from SMG.

The expanded form is the historical per-trajectory view. The compact form stores
each tree node once; this module reconstructs the expanded view so every
downstream consumer sees one shape.
"""

from __future__ import annotations

from typing import Any

FORMAT_EXPANDED = "expanded"
FORMAT_COMPACT = "compact"
SUPPORTED_FORMATS = (FORMAT_EXPANDED, FORMAT_COMPACT)


def _require_mapping(value: Any, what: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"TITO compact snapshot {what} must be an object.")
    return value


def _require_list(value: Any, what: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"TITO compact snapshot {what} must be a list.")
    return value


def _expand_compact(session_data: dict) -> dict:
    """Reconstruct trajectories and tree metadata from compact nodes/leaves.

    Each node carries its own segment starting at the common prefix with its
    parent. A tolerated trimmed boundary can therefore shorten the accumulated
    parent prefix before the segment is appended.
    """
    raw_nodes = _require_list(session_data.get("nodes"), "nodes")
    raw_leaves = _require_list(session_data.get("leaves"), "leaves")
    nodes: dict[int, dict] = {}
    for node in raw_nodes:
        node = _require_mapping(node, "node")
        node_id = int(node["id"])
        if node_id in nodes:
            raise ValueError(f"TITO compact snapshot repeats node id {node_id}.")
        nodes[node_id] = node

    trajectories: list[dict] = []
    for leaf in raw_leaves:
        leaf = _require_mapping(leaf, "leaf")
        path_ids = [int(node_id) for node_id in leaf.get("path_node_ids", [])]
        accumulated: list[int] = []
        records: list[dict] = []
        for node_id in path_ids:
            node = nodes.get(node_id)
            if node is None:
                raise ValueError(f"TITO compact leaf path references unknown node {node_id}.")
            if not node.get("stored"):
                continue
            start = int(node.get("token_start", 0))
            if start > len(accumulated):
                raise ValueError(
                    f"TITO compact node {node_id} starts at token {start}, "
                    f"but its path has {len(accumulated)} tokens."
                )
            # A trimmed boundary may make this node diverge from the stored
            # parent tail. The node segment starts at the common prefix, so
            # drop the parent's divergent suffix before appending it.
            del accumulated[start:]
            accumulated.extend(int(token_id) for token_id in node.get("token_ids", []) or [])
            turn_record = node.get("turn_record")
            if turn_record is not None:
                records.append(dict(turn_record))
        leaf_id = leaf.get("node_id")
        if leaf_id is not None:
            expected = nodes.get(int(leaf_id), {}).get("num_tokens")
            if expected is not None and len(accumulated) != int(expected):
                raise ValueError(
                    f"TITO compact leaf {leaf.get('trajectory_id')} reconstructs "
                    f"{len(accumulated)} tokens but its leaf node declares {expected}."
                )
        trajectories.append(
            {
                "trajectory_id": int(leaf["trajectory_id"]),
                "accumulated_token_ids": accumulated,
                "records": records,
                "thread_source": leaf.get("thread_source", "main"),
                "agent_id": leaf.get("agent_id"),
                "parent_trajectory_id": leaf.get("parent_trajectory_id"),
                "parent_agent_id": leaf.get("parent_agent_id"),
                "capture": leaf.get("capture", "full"),
            }
        )

    tree_nodes = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        tree_nodes.append(
            {
                key: node.get(key)
                for key in (
                    "id",
                    "hash",
                    "stored",
                    "parent",
                    "finish_reason",
                    "truncated",
                    "num_tokens",
                    "trajectory_ids",
                    "owner_trajectory_id",
                    "capture",
                    "sequence",
                )
            }
        )
    tree_leaves = [
        {
            key: leaf.get(key)
            for key in (
                "trajectory_id",
                "node_id",
                "parent",
                "path_node_ids",
                "thread_source",
                "agent_id",
                "parent_trajectory_id",
                "parent_agent_id",
                "capture",
            )
        }
        for leaf in raw_leaves
    ]

    expanded = dict(session_data)
    expanded.update(
        {
            "format": FORMAT_EXPANDED,
            "source_format": FORMAT_COMPACT,
            "trajectories": trajectories,
            "tree": {"nodes": tree_nodes, "leaves": tree_leaves},
        }
    )
    expanded.pop("nodes", None)
    expanded.pop("leaves", None)
    return expanded


def normalize_session_snapshot(session_data: dict) -> dict:
    """
    Return one expanded snapshot shape.

    A missing `format` is the historical expanded form. An unknown format is a
    contract error, not a silent fallback.
    """
    if not isinstance(session_data, dict):
        raise ValueError("TITO session snapshot must be an object.")
    snapshot_format = str(session_data.get("format") or FORMAT_EXPANDED).strip().lower()
    if snapshot_format == FORMAT_COMPACT:
        return _expand_compact(session_data)
    if snapshot_format != FORMAT_EXPANDED:
        raise ValueError(
            f"Unsupported TITO snapshot format {snapshot_format!r}; expected one of {SUPPORTED_FORMATS}."
        )
    expanded = dict(session_data)
    expanded["format"] = FORMAT_EXPANDED
    if not isinstance(expanded.get("trajectories"), list):
        raise ValueError("TITO session snapshot has no trajectories list.")
    if not isinstance(expanded.get("tree"), dict):
        expanded["tree"] = {"nodes": [], "leaves": []}
    return expanded
