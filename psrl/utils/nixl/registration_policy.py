"""Which clients must register memory with NIXL, derived from the comm plan (design 3.7.3.0, 3.7.4).

RDMA registration exists for exactly one reason: to let a remote NIC reach this process's
memory without going through the CPU. So the necessary and sufficient condition is that a
*NIXL transfer has an edge here*:

    register  <=>  as a source: some reader is on a different node
               or  as a source: some reader pulls a shard this process cannot export
               or  as a reader: some source is on a different node, or a shard it pulls
                   from a same-node source has no IPC handle

The tempting criterion is a deployment flag -- "colocate: don't register, disaggregate:
register" -- and it is wrong, because it misses a cell that really occurs: under
disaggregate, a trainer's own GPU is also switched colocate-style for eval, so its remote
rollout readers still read it over RDMA. The flag says "disaggregate", the truth is per
edge. Deriving the answer from the plan instead costs nothing and covers the mixed case for
free: one trainer can serve same-node readers over CUDA IPC and remote readers over RDMA,
and each read gets the transport it actually needs.

The reader direction matters as much as the source direction. A NIXL READ needs the
*destination* registered, and in pure colocate the rollout wake was paying five seconds to
register weights no NIC could ever reach. `all_sources_copyable` answers whether every
shard a reader pulls is same-node and backed by an IPC handle, which is what lets that
registration go. `missing_exports_block_readers` answers the mirror question on the source
side: which readers would fall back to NIXL because of an allocation it could not export.

**Why it matters.** The client registers every local tensor on every
`register_local_tensors` call, and the colocate switch calls that on each cycle
(`nixl_sleep` -> `nixl_wake_up`). In a pure colocate run *none* of those registrations can
ever be used -- every edge is same-node -- so the whole cost is paid for nothing. It is
also not free: one `nixlRegDList` plus `get_xfer_descs`/`get_serialized_descs` per shard,
which at MoE scale is the same O(#shards) cost design §3.5.5 measures in the tens of
thousands.

**Conservatism is not optional here.** Being wrong in one direction costs performance;
being wrong in the other costs correctness -- a remote reader of an unregistered shard
fails, or worse, reads a stale descriptor. So every unknown answers "register":
a source with no recorded node, a reader missing from `client_infos`, a client reading
from itself. The only way to get `False` is to positively observe that every edge is
same-node and exportable. A reader that guesses wrong is recovered at transfer time by
`ensure_rdma_registered`, which pays the deferred registration instead of failing.

**Deliberately ignored: peer-diffusion delegation on the source side.** A same-node
follower reads its designate's copy rather than the upstream source, so the plan
over-counts readers. Not modelling it can only add readers, i.e. only ever answer
"register" -- which is the safe direction, and the case it would matter for (a follower
delegating a *remote* read) keeps the remote source registered anyway. The reader side
does model the designate edge, because there an omit would answer "skip" wrongly.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from .nixl_spec import NIXLClientType

if TYPE_CHECKING:  # pragma: no cover
    from .comm_plan import NIXLCommPlan
    from .nixl_spec import NIXLClientInfo

#: The three plan directions, all shaped `{reader: {key: {source: [shards]}}}`.
#: Read together rather than one at a time because a client can be a source in more than
#: one of them -- a trainer pushes to the PS *and* is read directly by rollouts.
_PLAN_FIELDS = ("push_to_ps_plan", "rollout_pull_from_ps_plan", "train_pull_from_ps_plan")


def readers_by_source(plan: "NIXLCommPlan") -> dict[str, dict[str, set[str]]]:
    """Invert the plan: ``source -> key -> readers``.

    Built once and shared across every source in a cluster, because inverting three
    nested dicts per client would repeat the same O(#clients x #keys) walk N times.
    """
    inverted: dict[str, dict[str, set[str]]] = {}
    for field_name in _PLAN_FIELDS:
        per_reader = getattr(plan, field_name, None) or {}
        for reader, key_plans in per_reader.items():
            for key, per_source in key_plans.items():
                for source, shards in per_source.items():
                    if not shards:
                        continue
                    inverted.setdefault(source, {}).setdefault(key, set()).add(reader)
    return inverted


def descriptor_readers_by_source(plan: "NIXLCommPlan") -> dict[str, dict[str, set[str]]]:
    """Who needs ``source``'s full descriptors: `readers_by_source` plus peer-diffusion readers.

    A same-node follower reads its designate's tensors directly, and that read does not
    appear in any of the three plan directions -- it lives in `peer_groups`. Loading only
    connection info for the designate (the E4 two-stage join) then leaves the follower
    unable to prepare a transfer: `prep_xfer_dlist` raises NIXL_ERR_NOT_FOUND, on the
    second pull, because the first pull never exercised the peer path.

    Kept separate from `readers_by_source` on purpose. That function answers "does a
    remote NIC need this memory registered", and the module docstring ignores delegation
    there because over-counting readers is the safe direction. This one answers "who must
    hold descriptors", where leaving a reader out is the *unsafe* direction.
    """
    inverted = {source: {key: set(readers) for key, readers in keys.items()} for source, keys in readers_by_source(plan).items()}
    for client, per_key in (getattr(plan, "peer_groups", None) or {}).items():
        for key, per_shard in per_key.items():
            for shard, group in per_shard.items():
                designate = group[0]
                if client != designate:
                    inverted.setdefault(designate, {}).setdefault(key, set()).add(client)
    return inverted


def keys_read_across_nodes(
    plan: "NIXLCommPlan",
    client_infos: Mapping[str, "NIXLClientInfo"],
    source: str,
    readers: dict[str, dict[str, set[str]]] | None = None,
) -> set[str]:
    """Keys of ``source`` that at least one reader on another node will read.

    An empty set is the colocate case: nothing this client publishes can be reached by a
    remote NIC, so nothing needs registering.
    """
    readers = readers_by_source(plan) if readers is None else readers
    source_node = _node_of(client_infos, source)
    if source_node is None:
        # Without our own node there is nothing to compare against, so every key that
        # anyone reads must be assumed remote.
        return set(readers.get(source, {}))

    cross_node: set[str] = set()
    for key, key_readers in readers.get(source, {}).items():
        for reader in key_readers:
            if reader == source:
                continue
            reader_node = _node_of(client_infos, reader)
            if reader_node is None or reader_node != source_node:
                cross_node.add(key)
                break
    return cross_node


def needs_rdma_registration(
    plan: "NIXLCommPlan",
    client_infos: Mapping[str, "NIXLClientInfo"],
    source: str,
    readers: dict[str, dict[str, set[str]]] | None = None,
) -> bool:
    """Whether ``source`` must register anything at all.

    `client_infos` is required rather than optional: the node comparison is the whole
    criterion, and a caller that cannot supply node identities cannot answer the
    question. Making it optional would let a caller silently get "register everything"
    and mistake it for a measurement.
    """
    return bool(keys_read_across_nodes(plan, client_infos, source, readers))


def sources_by_reader(plan: "NIXLCommPlan", reader: str) -> dict[str, dict[str, list[tuple[int, ...]]]]:
    """Invert the pull directions for one reader: ``key -> source -> shards``.

    A peer-diffusion follower reads its designate instead of the upstream source for
    the specific shards that have one, so the designate replaces the plan's source for
    just those shards -- other shards of the same key keep whichever upstream source
    the plan named, since a key's shards can now have different (or no) designates.
    That is exactly what the pull engine asks each source for. The edge decides the
    follower's transport: a local designate can serve by IPC/peer copy even when the
    upstream trainer is remote.
    """
    out: dict[str, dict[str, list[tuple[int, ...]]]] = {}
    for field_name in ("rollout_pull_from_ps_plan", "train_pull_from_ps_plan"):
        per_key = getattr(plan, field_name, None) or {}
        for key, per_source in per_key.get(reader, {}).items():
            for source, shards in per_source.items():
                if shards and source != reader:
                    out.setdefault(key, {}).setdefault(source, []).extend(shards)
    for key, per_shard in (getattr(plan, "peer_groups", None) or {}).get(reader, {}).items():
        for shard, group in per_shard.items():
            if not group or group[0] == reader:
                continue
            designate = group[0]
            shard = tuple(shard)
            # Move this one shard from wherever the upstream plan put it to the
            # designate it actually reads from -- a per-shard override, not the old
            # blanket key replace, since different shards of the same key can now
            # have different designates (or no designate, if nobody else wants them).
            for source, source_shards in list(out.get(key, {}).items()):
                if shard in source_shards:
                    source_shards.remove(shard)
                    if not source_shards:
                        del out[key][source]
                    break
            out.setdefault(key, {}).setdefault(designate, []).append(shard)
    return out


def all_sources_copyable(
    plan: "NIXLCommPlan",
    client_infos: Mapping[str, "NIXLClientInfo"],
    reader: str,
) -> tuple[bool, str]:
    """Whether every byte ``reader`` pulls can arrive by IPC or peer copy.

    That is the condition under which the reader may skip its own NIXL registration: a
    NIXL READ needs the destination registered, so the registration is pure overhead only
    when no read will ever go through NIXL. Being wrong here is not a performance problem
    -- the first NIXL read fails -- so every unknown answers False:

    * an unknown node, source, or plan entry;
    * a source on another node, or a PS client (a different data plane);
    * a shard of this reader's whose source allocation has no IPC handle.

    The last check is per shard, not per source: a vLLM allocator can refuse to export a
    few buffers while every shard this reader wants is exportable, and a source-wide
    "complete" bit would make every same-node reader register for nothing.

    Returns ``(copyable, reason)``; the reason is for logs and is empty on success.
    """
    node = _node_of(client_infos, reader)
    if node is None:
        return False, "reader node unknown"
    sources = sources_by_reader(plan, reader)
    if not sources:
        return False, "no pull plan entry"
    for key, per_source in sources.items():
        for source, shards in per_source.items():
            info = client_infos.get(source)
            if info is None:
                return False, f"source {source} unknown"
            if getattr(info, "type", None) not in (NIXLClientType.PUSH_SIDE, NIXLClientType.PULL_SIDE):
                # A PS client is a different data plane; its transfers are not pull-engine
                # reads and cannot be served by IPC/peer copy.
                return False, f"source {source} is not a data-plane source"
            if _node_of(client_infos, source) != node:
                return False, f"source {source} is on another node"
            tensor_info = info.tensor_infos.get(key)
            if tensor_info is None or not shards:
                return False, f"source {source} publishes no {key!r}"
            positions = {tuple(s): i for i, s in enumerate(tensor_info.sharding.shard_indices)}
            read_bases = tensor_info.read_bases or ()
            for shard in shards:
                pos = positions.get(tuple(shard))
                base = read_bases[pos] if pos is not None and pos < len(read_bases) else None
                if base is None:
                    return False, f"source {source} exposes no bytes for {key!r} shard {tuple(shard)}"
                if base not in info.ipc_regions:
                    return False, f"source {source} cannot export {key!r} shard {tuple(shard)}"
    return True, ""


def missing_exports_block_readers(
    plan: "NIXLCommPlan | None",
    source: str,
    tensor_infos: Mapping[str, object],
    ipc_missing: frozenset[int],
    readers: dict[str, dict[str, set[str]]] | None = None,
) -> bool:
    """Whether any same-node reader pulls a shard of ``source`` that has no IPC handle.

    This is the source-side half of `all_sources_copyable`, and it decides whether a
    source that skipped NIXL registration must register after all: a reader whose shard
    cannot be exported will register and read through NIXL, so its source must be
    reachable too. A source nobody reads -- a validate instance, for example -- answers
    False even when its own allocator cannot export anything (3.7.4).

    Unknowns answer True: without a plan we cannot tell which shards are read.
    """
    if not ipc_missing:
        return False
    if plan is None:
        return True
    readers = descriptor_readers_by_source(plan) if readers is None else readers
    per_key = readers.get(source)
    if not per_key:
        return False
    for key, key_readers in per_key.items():
        info = tensor_infos.get(key)
        if info is None:
            continue
        positions = {tuple(s): i for i, s in enumerate(info.sharding.shard_indices)}
        read_bases = getattr(info, "read_bases", None) or ()
        for reader in key_readers:
            if reader == source:
                continue
            planned = (
                (getattr(plan, "rollout_pull_from_ps_plan", None) or {}).get(reader, {}).get(key, {}).get(source)
                or (getattr(plan, "train_pull_from_ps_plan", None) or {}).get(reader, {}).get(key, {}).get(source)
            )
            # A peer-diffusion follower is not in the plan: it takes every shard of the
            # key this source holds.
            shards = [tuple(s) for s in planned] if planned else [tuple(s) for s in positions]
            for shard in shards:
                pos = positions.get(shard)
                base = read_bases[pos] if pos is not None and pos < len(read_bases) else None
                if base is None or base in ipc_missing:
                    return True
    return False


def registration_report(
    plan: "NIXLCommPlan",
    client_infos: Mapping[str, "NIXLClientInfo"],
    source: str,
) -> dict[str, object]:
    """Human-readable account of one client's verdict, for logs and assertions."""
    readers = readers_by_source(plan)
    cross = keys_read_across_nodes(plan, client_infos, source, readers)
    all_keys = set(readers.get(source, {}))
    return {
        "source": source,
        "node": _node_of(client_infos, source),
        "published_keys": len(all_keys),
        "cross_node_keys": len(cross),
        "needs_registration": bool(cross),
        # The saving, stated as what would have been registered and cannot be read
        # remotely. Equal to `published_keys - cross_node_keys`.
        "skippable_keys": len(all_keys - cross),
    }


def _node_of(client_infos: Mapping[str, "NIXLClientInfo"], client: str) -> str | None:
    info = client_infos.get(client)
    if info is None:
        return None
    node_ip = getattr(info, "node_ip", None)
    return str(node_ip) if node_ip else None
