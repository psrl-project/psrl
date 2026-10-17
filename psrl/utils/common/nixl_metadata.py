from collections.abc import Iterable
from typing import TypeAlias, TypedDict


class NIXLClientUpdate(TypedDict):
    info: bytes
    temp_mapping: dict[tuple[str, tuple[int, ...]], bytes]


ClientInfoUpdates: TypeAlias = dict[str, NIXLClientUpdate]


def merge_client_updates(
    groups: Iterable[ClientInfoUpdates],
    expected_client_names: Iterable[str],
) -> ClientInfoUpdates:
    """
    Require one metadata update from every client participating in a storage transition.
    """
    updates: ClientInfoUpdates = {}
    for group in groups:
        duplicates = updates.keys() & group.keys()
        if duplicates:
            raise ValueError(f"Duplicate NIXL client updates: {sorted(duplicates)!r}.")
        updates.update(group)
    expected = set(expected_client_names)
    missing = expected - updates.keys()
    unexpected = updates.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"Incomplete NIXL client updates: missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}."
        )
    return updates
