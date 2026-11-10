"""
Utility helpers for working with TransferQueue's KVBatchMeta.
"""

from collections import Counter
from enum import Enum
from typing import Any

import transfer_queue as tq
from transfer_queue import KVBatchMeta


class PayloadState(Enum):
    """Lifecycle state for payloads stored in TransferQueue."""

    PENDING_GROUP = "pending_group"
    OCCUPIED = "occupied"
    READY = "ready"
    DROPPED = "dropped"
    CONSUMED = "consumed"

    @property
    def is_terminal(self) -> bool:
        """Whether payload ownership has ended and its TQ storage may be cleared."""
        return self in {PayloadState.DROPPED, PayloadState.CONSUMED}


def request_payload_keys(request_id: int, n_trajectory: int) -> list[str]:
    """Reconstruct the TQ payload keys for one request."""
    if n_trajectory < 1:
        raise ValueError(f"n_trajectory must be positive, got {n_trajectory}.")
    if n_trajectory == 1:
        return [str(request_id)]
    return [f"{request_id}_{i}" for i in range(n_trajectory)]


def _require_terminal_state(state: PayloadState) -> None:
    if not state.is_terminal:
        raise ValueError(f"Cannot clear payload in non-terminal state {state.value!r}.")


def _existing_payload_keys(
    keys: list[str] | str,
    partition_id: str,
    partitions: dict[str, dict[str, Any]],
) -> list[str]:
    """Return only keys currently registered in the target partition."""
    requested_keys = [keys] if isinstance(keys, str) else keys
    partition = partitions.get(partition_id, {})
    return [key for key in requested_keys if key in partition]


def clear_payload(
    keys: list[str] | str,
    partition_id: str,
    state: PayloadState,
) -> None:
    """Synchronously clear any present payload after entering a terminal state."""
    _require_terminal_state(state)
    existing_keys = _existing_payload_keys(keys, partition_id, tq.kv_list(partition_id=partition_id))
    if existing_keys:
        tq.kv_clear(keys=existing_keys, partition_id=partition_id)


async def async_clear_payload(
    keys: list[str] | str,
    partition_id: str,
    state: PayloadState,
) -> None:
    """Asynchronously clear any present payload after entering a terminal state."""
    _require_terminal_state(state)
    partitions = await tq.async_kv_list(partition_id=partition_id)
    existing_keys = _existing_payload_keys(keys, partition_id, partitions)
    if existing_keys:
        await tq.async_kv_clear(keys=existing_keys, partition_id=partition_id)


def validate_ready_payload(
    keys: list[str],
    partition_id: str,
    partitions: dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Validate that every key in a prospective READY batch is committed."""
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    partition = partitions.get(partition_id, {})
    missing_keys = [key for key in keys if key not in partition]
    non_success_keys = [
        (key, partition[key].get("status"))
        for key in keys
        if key in partition and partition[key].get("status") != "success"
    ]
    if duplicate_keys or missing_keys or non_success_keys:
        raise RuntimeError(
            "Cannot publish READY payload: "
            f"partition_id={partition_id!r}, total={len(keys)}, "
            f"duplicate={len(duplicate_keys)}, missing={len(missing_keys)}, "
            f"non_success={len(non_success_keys)}, "
            f"duplicate_keys[:20]={duplicate_keys[:20]!r}, "
            f"missing_keys[:20]={missing_keys[:20]!r}, "
            f"non_success_keys[:20]={non_success_keys[:20]!r}."
        )


def kv_batch_meta_update_tags(batch: KVBatchMeta, key: str, value) -> KVBatchMeta:
    """Return a new KVBatchMeta with ``key`` set to ``value`` in every tag.

    If ``value`` is a list it must have the same length as ``batch``, and
    each element is assigned to the corresponding tag. Otherwise the scalar
    ``value`` is broadcast to all tags.

    Args:
        batch: Source ``KVBatchMeta``.
        key: Tag field name to set / overwrite.
        value: Scalar (broadcast) or list (per-sample) new value.

    Returns:
        A new ``KVBatchMeta`` instance with updated tags.
    """
    n = len(batch)
    if isinstance(value, list):
        if len(value) != n:
            raise ValueError(
                f"kv_batch_meta_update_tags: value list length {len(value)} does not match batch size {n}."
            )
        new_tags = [{**tag, key: value[i]} for i, tag in enumerate(batch.tags)]
    else:
        new_tags = [{**tag, key: value} for tag in batch.tags]

    return KVBatchMeta(
        keys=batch.keys,
        tags=new_tags,
        partition_id=batch.partition_id,
        fields=batch.fields,
        extra_info=batch.extra_info,
    )
