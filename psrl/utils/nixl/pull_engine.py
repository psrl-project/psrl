"""
Rollout side weight pull: plan every read once, then post a handful of handles per pull.

A pull reads each planned `(source, key)` as byte runs paired between the source's read
layout and the local tensor. Runs that are adjacent on both sides merge, so a peer that
holds an identical tensor costs one descriptor per key. A side that is fragmented is read
through a bounded staging pool and scattered locally instead of paying a descriptor per
row. Same node peers form diffusion groups: per pull the first participating member of a
group reads the key upstream and the others read it from that member over NVLink.
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from psrl.utils.nixl.device_ipc import local_device
from psrl.utils.nixl.network_topology import get_local_ip
from psrl.utils.nixl.nixl_spec import NIXLClientType
from psrl.utils.nixl.read_geometry import coalesce, pair_runs, view_runs

if TYPE_CHECKING:
    from psrl.utils.nixl.client import NIXLStorageClient

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))

# Cost model used only to rank strategies. Posting one descriptor costs a few microseconds.
SECONDS_PER_DESCRIPTOR = 3e-6
BYTES_PER_SECOND = 25e9
# A key whose merged direct read needs at most this many descriptors per shard is read in place.
DIRECT_DESCRIPTORS_PER_SHARD = 64
# Staged items start on this alignment so every dtype view of the pool is legal.
_ALIGN = 256
# Signal chunk carried by per-pull notifications. Wave `w` is ready as chunk `_READY + w`.
_READY = 1

_NIXL_MEM = {"cuda": "VRAM", "cpu": "DRAM", "VRAM": "VRAM", "DRAM": "DRAM"}


@dataclass
class _Staged:
    """One shard read into the staging pool, then copied into its local view."""

    key: str
    nbytes: int
    remote: np.ndarray
    rel: np.ndarray
    lengths: np.ndarray
    local_view: torch.Tensor
    view_shape: tuple[int, ...]
    view_strides: tuple[int, ...]
    #: Reader-side quantization: the staging bytes are `source_dtype`, and the
    #: scatter step quantizes them into `local_view` (FP8) plus its `_scale_inv`.
    source_dtype: torch.dtype | None = None
    quantize: bool = False
    shard: tuple[int, ...] | None = None


@dataclass
class _Unit:
    """A rollout parameter whose wire views are quantized as one tensor.

    A fused parameter can be smaller than one scale tile, so a checkpoint view cuts a
    tile and has no scale block of its own. The reader stages every member's bytes in
    destination order and quantizes the parameter once, which produces exactly the bytes
    and scale the engine would derive from the parameter itself.
    """

    key: str
    scale_key: str
    scale_view: torch.Tensor
    cols: int
    #: `(key, shard) -> (first row, row count)` in the unit.
    members: dict[tuple[str, tuple[int, ...]], tuple[int, int]]
    rows: int
    buffer: torch.Tensor | None = None
    staged: int = 0
    dirty: bool = False


@dataclass
class _Spec:
    """Everything one `(source, key)` read moves, independent of when it runs."""

    source: str
    key: str
    direct_local: np.ndarray
    direct_remote: np.ndarray
    direct_len: np.ndarray
    direct_region: np.ndarray
    staged: list[_Staged]
    versions: set[int]
    nbytes: int
    # `(destination view, source view)` pairs for a source reachable without a NIC: the
    # same device, or another GPU of this node. NIXL costs 7.7x a device copy, so these
    # are copied, not transferred (3.7.2.1, 3.7.4).
    ipc: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


@dataclass
class _Handle:
    source: str
    handle: object
    descriptors: int
    nbytes: int
    #: Keys whose bytes this request covers, so a preemption can classify what completed.
    keys: frozenset[str] = frozenset()
    posted_at: float = 0.0
    #: A staged round reads into the staging pool, so the pool must outlive the request.
    reads_pool: bool = False


@dataclass
class _Phase:
    """Posting order for one set of specs: direct handles, then staging rounds on two slots."""

    direct: list[_Handle] = field(default_factory=list)
    rounds: list[list[_Handle]] = field(default_factory=list)
    round_items: list[list[tuple[_Staged, int]]] = field(default_factory=list)
    #: The staging pool the round handles read into. Growth retires it, not frees it, while a read may use it.
    pool: torch.Tensor | None = None
    #: Every key this phase reads, including the ones served by same-device copies.
    keys: frozenset[str] = frozenset()
    versions: set[int] = field(default_factory=set)
    descriptors: int = 0
    nbytes: int = 0
    staged_bytes: int = 0
    ipc: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    ipc_bytes: int = 0


class ChunkUnavailable(RuntimeError):
    """
    Some keys have no live train source. Raised before any byte moves, so the rollout's weights are untouched.

    The caller keeps generating at its current version (design 3.5.7, D6).
    """

    def __init__(self, client: str, keys: list[str], dead: set[str]):
        self.keys, self.dead = keys, dead
        super().__init__(
            f"{client}: no live source holds {len(keys)} key(s), first {keys[0]!r}. "
            f"Unavailable sources: {sorted(dead)}."
        )


@dataclass(frozen=True)
class PullVerdict:
    """What one pull left the instance holding (design §3.4 version equality).

    Returned across the `collective_rpc` boundary, so it holds primitives rather than a
    tensor or a client. A caller distinguishes three ends because the response differs.
    `landed` means one version arrived and is still valid. `torn` means shards came from
    more than one version, which only another whole pull repairs. Neither means no byte
    was written (D6), so the instance is consistent at the version it already had.
    """

    landed: bool
    versions: frozenset[int]
    #: Why this pull did not land: "torn", "invalidated", "unavailable", or "preempted".
    reason: str | None = None
    #: The pull was preempted by an optimizer write and completed by re-reading its
    #: unfinished keys from certified same-node holders of the same version.
    repaired: bool = False

    @property
    def torn(self) -> bool:
        return self.reason == "torn"


class TornPull(RuntimeError):
    """
    A source failed mid pull and no replica holds what it was serving.

    Earlier waves have already written the new version, so the weights mix two versions and
    the instance must not generate until a later pull completes. `nixl_pull_model_core`
    turns this into a `PullVerdict`, so the caller can re-pull instead of failing.
    """


class PullPreempted(RuntimeError):
    """The writer is about to overwrite the trainer bytes this pull was reading.

    Carries what the reader needs to decide between repairing from certified holders of
    the same version and giving up until the next version:

    * ``completed_keys`` were read before the acknowledgement, which the writer waits for,
      so they are safe;
    * ``suspect_keys`` are every other key of the pull (in flight, not yet posted, or lost
      to an abandoned source) and must be re-read or invalidated;
    * ``holders`` are same-node instances the coordinator certified as holding ``version``.

    ``repaired`` is set when the exception is raised *after* a successful repair attempt is
    impossible; the caller must not serve the local bytes.
    """

    def __init__(
        self,
        epoch: int,
        version: int,
        completed_keys: frozenset[str],
        suspect_keys: frozenset[str],
        holders: tuple[str, ...],
        repaired: bool = False,
        reason: str | None = None,
    ) -> None:
        self.epoch = epoch
        self.version = version
        self.completed_keys = completed_keys
        self.suspect_keys = suspect_keys
        self.holders = holders
        self.repaired = repaired
        self.reason = reason
        super().__init__(
            f"pull preempted at epoch {epoch}: {len(suspect_keys)} suspect key(s), "
            f"{len(holders)} holder(s), repaired={repaired}"
            + (f", reason={reason}" if reason else "")
        )


class WeightRefreshFailed(RuntimeError):
    """Bounded re-pulls did not leave the instance holding one live version.

    Raised only when bytes have already been overwritten, so the weights are a mixture of
    publications. The sync coordinator quarantines the replica and the next sync re-pulls.
    """


@dataclass
class PullReport:
    """What one pull did, for logs and for tests that pin the traffic it generated."""

    seconds: float
    upstream_seconds: float
    peer_seconds: float
    upstream_keys: int
    peer_keys: int
    fallback_keys: int
    upstream_bytes: int
    peer_bytes: int
    descriptors: int
    staged_bytes: int
    followers: int
    versions: set[int]
    # Per source `(bytes, first post, last completion)` of the upstream and peer reads.
    sources: dict[str, tuple[int, float, float]] = field(default_factory=dict)
    # Keys re-read from a DP replica because their planned source was unavailable or failed.
    takeover_keys: int = 0
    failed_sources: set[str] = field(default_factory=set)
    # Bytes copied from a source that shares this device, rather than transferred (3.7.2.1).
    ipc_bytes: int = 0
    # Bytes copied from a source on another device of this node, over NVLink/PCIe (3.7.4).
    ipc_peer_bytes: int = 0
    #: The pull was preempted by an optimizer write and finished by re-reading its
    #: unfinished keys from certified same-node holders of the same version.
    repaired: bool = False


class NIXLPullEngine:
    """
    Pull engine bound to one rollout `NIXLStorageClient` after its handshake.

    Args:
        client (NIXLStorageClient): A PULL_SIDE client whose handshake has completed.
        staging_bytes (int): Size of the staging pool, split into two alternating slots.
        peer_timeout_s (float): How long a follower waits for its designate before reading upstream.
        waves (int): Upstream waves per pull. Peer reads of one wave overlap the next wave's upstream.
        read_timeout_s (float): How long one wave's reads may take before their sources count as failed.
    """

    def __init__(
        self,
        client: NIXLStorageClient,
        staging_bytes: int,
        peer_timeout_s: float,
        waves: int = 4,
        read_timeout_s: float = 300.0,
    ):
        self.client = client
        self.waves = max(1, int(waves))
        self.source_trace: dict[str, tuple[int, float, float]] = {}
        self.read_timeout_s = float(read_timeout_s)
        # Sources known to be gone (told by the coordinator, or learned from a failed read).
        self.unavailable: set[str] = set()
        # Sources that failed during the current pull.
        self.failed: set[str] = set()
        self.agent = client.agent
        #: This node's identity, resolved once: `_ipc_cells` asks it per source and a
        #: hostname lookup on that path would repeat hundreds of times per pull.
        self._local_ip = get_local_ip()
        self.staging_bytes = int(staging_bytes)
        self.peer_timeout_s = float(peer_timeout_s)
        self._specs: dict[tuple[str, str], _Spec] = {}
        self._phases: dict[tuple, _Phase] = {}
        self._pool: torch.Tensor | None = None
        self._pool_reg = None
        # Handles posted and not yet observed complete. The phase cache may be invalidated
        # while one of these runs (a pool growth rebuilds it), so release must skip them.
        self._in_flight: dict[int, _Handle] = {}
        # Handles whose phase was invalidated while they were in flight. Released once complete.
        self._orphans: list[_Handle] = []
        # Pools a growth replaced. Kept alive until no orphaned handle can still read them.
        self._retired_pools: list[tuple[torch.Tensor, object]] = []
        self._copy_stream: torch.cuda.Stream | None = None
        self._dlists: list[object] = []
        # (local allocation, remote allocation) to a small label, so coalescing never crosses one.
        self._region_labels: dict[tuple[int, int], int] = {}
        # --- v2 write-gate state, reset at every pull ---
        #: Writer epoch recorded by the coordinator when this pull was admitted.
        self._pull_epoch: int | None = None
        #: Called with the epoch the first time a preemption notice is observed.
        self._on_preempt_ack = None
        self._preempt_epoch: int | None = None
        self._preempt_version: int = -1
        self._preempt_holders: tuple[str, ...] = ()
        self._completed_keys: set[str] = set()
        self._pull_keys: set[str] = set()
        self._key_outstanding: dict[str, int] = {}
        #: Keys whose staged bytes still need a quantize/scatter before they are complete.
        self._quantize_keys: set[str] = set()
        self._quantize_pending: dict[str, int] = {}
        self._accounted_phases: set[int] = set()
        self._next_control_poll = 0.0
        #: True while re-reading suspect keys from holders; peer reads must not re-raise.
        self._repairing = False
        self._pull_started = 0.0
        #: Preemption notices are polled at control-plane latency, not per 50 µs transfer poll.
        self._control_poll_interval_s = 5e-4
        #: Bytes copied from a same-node source on another GPU (diagnostics, 3.7.4).
        self._ipc_peer_bytes = 0
        #: Rollout parameters whose wire views are quantized together, built on first use.
        self._unit_index: dict[str, _Unit] | None = None
        self._unit_of_key: dict[str, str] = {}

    # ------------------------------------------------------------------ geometry

    def _local_view(self, key: str, shard: tuple[int, ...]) -> torch.Tensor:
        view = self.client.get_original_tensor_mapping().get((key, shard))
        assert view is not None, f"{self.client.client_name} holds no local view for {key!r} shard {shard}."
        return view

    def _spec(self, source: str, key: str, shards: list[tuple[int, ...]]) -> _Spec:
        cached = self._specs.get((source, key))
        if cached is not None:
            return cached
        remote_info = self.client.all_client_infos[source].tensor_infos[key]
        positions = {s: i for i, s in enumerate(remote_info.sharding.shard_indices)}
        cells = []
        quantized = []
        for shard in shards:
            pos = positions[shard]
            meta = remote_info.shard_meta_infos[pos]
            local = self._local_view(key, shard)
            address = remote_info.read_addrs[pos] if remote_info.read_addrs else None
            assert address is not None, f"{source} exposes no readable bytes for {key!r} shard {shard}."
            quantize = (
                self.client._reader_quantization
                and self.client._quantizer is not None
                and meta.dtype != local.dtype
                and local.dtype == self.client._quantizer.weight_dtype
            )
            if not quantize and self._unit_of(key) is not None:
                raise RuntimeError(
                    f"{key!r} feeds a quantization unit, so it must arrive in the source dtype, "
                    f"but {source} publishes {meta.dtype}. A unit's scale spans every member, so "
                    "assembling it from already quantized bytes would leave it stale."
                )
            if quantize:
                assert tuple(meta.shape) == tuple(local.shape), (
                    f"{key!r} shard {shard}: source {meta.dtype} {tuple(meta.shape)} at {source} "
                    f"does not match the rollout target {local.dtype} {tuple(local.shape)}."
                )
                source_itemsize = torch.empty((), dtype=meta.dtype).element_size()
                runs = view_runs(address, tuple(meta.shape), tuple(remote_info.read_strides[pos]), source_itemsize)
                quantized.append((shard, meta, local, runs))
                continue
            assert meta.dtype == local.dtype and tuple(meta.shape) == tuple(local.shape), (
                f"{key!r} shard {shard} is {meta.dtype} {tuple(meta.shape)} at {source} "
                f"but {local.dtype} {tuple(local.shape)} locally."
            )
            itemsize = local.element_size()
            local_runs = view_runs(local.data_ptr(), tuple(local.shape), local.stride(), itemsize)
            remote_runs = view_runs(address, tuple(local.shape), tuple(remote_info.read_strides[pos]), itemsize)
            region = self._region_labels.setdefault(
                (local.untyped_storage().data_ptr(), remote_info.read_bases[pos]), len(self._region_labels)
            )
            cells.append((local, local_runs, remote_runs, tuple(remote_info.read_strides[pos]), meta.version, region))

        if quantized:
            staged = []
            versions = set()
            nbytes = 0
            for shard, meta, local, runs in quantized:
                item = _Staged(
                    key,
                    runs.total_bytes,
                    runs.starts,
                    np.arange(runs.count, dtype=np.int64) * runs.run_bytes,
                    np.full(runs.count, runs.run_bytes, dtype=np.int64),
                    local,
                    tuple(meta.shape),
                    tuple(_dense(meta.shape)),
                    source_dtype=meta.dtype,
                    quantize=True,
                    shard=shard,
                )
                staged.append(item)
                versions.add(meta.version)
                nbytes += item.nbytes
            empty = np.zeros(0, dtype=np.int64)
            spec = _Spec(source, key, empty, empty, empty, empty, staged, versions, nbytes)
            self._specs[(source, key)] = spec
            return spec

        shared = self._ipc_cells(source, remote_info, shards, positions, cells)
        if shared is not None:
            empty = np.zeros(0, dtype=np.int64)
            spec = _Spec(
                source,
                key,
                empty,
                empty,
                empty,
                empty,
                [],
                {c[4] for c in cells},
                sum(c[1].total_bytes for c in cells),
                ipc=shared,
            )
            self._specs[(source, key)] = spec
            return spec

        # Whole-storage fast path (FINDINGS §2, design item D1).
        #
        # Peer diffusion (designate → follower) assigns byte-identical shards: both sides
        # hold the same gen-TP rank and lay the shard out in the same contiguous storage.
        # When local *and* remote are both contiguous we can issue one descriptor per shard
        # rather than one per contiguous run of each strided cell.  For a row-parallel key
        # cut by finest-mesh TP that collapses from O(hidden) descriptors to O(1).
        #
        # Safety: `meta.is_contiguous` is set by the source at registration time and means
        # the readable bytes form a single flat allocation starting at `read_addrs[pos]`.
        # `local.is_contiguous()` means the local destination is equally flat.  If either
        # fails the code falls through to the general strided path, which is always correct.
        if cells and all(
            c[0].is_contiguous()
            and remote_info.shard_meta_infos[positions[shard]].is_contiguous
            for c, shard in zip(cells, shards)
        ):
            local_arr = np.array([c[0].data_ptr() for c in cells], dtype=np.int64)
            remote_arr = np.array([remote_info.read_addrs[positions[shard]] for shard in shards], dtype=np.int64)
            lengths_arr = np.array([c[1].total_bytes for c in cells], dtype=np.int64)
            regions_arr = np.array([c[5] for c in cells], dtype=np.int64)
            spec = _Spec(
                source, key, local_arr, remote_arr, lengths_arr, regions_arr,
                [], {c[4] for c in cells}, int(lengths_arr.sum()),
            )
            self._specs[(source, key)] = spec
            return spec

        pairs = [pair_runs(c[1], c[2]) for c in cells]
        empty = np.zeros(0, dtype=np.int64)
        local = np.concatenate([p[0] for p in pairs]) if pairs else empty
        remote = np.concatenate([p[1] for p in pairs]) if pairs else empty
        lengths = np.concatenate([np.full(p[0].shape, p[2], dtype=np.int64) for p in pairs]) if pairs else empty
        regions = (
            np.concatenate([np.full(p[0].shape, c[5], dtype=np.int64) for p, c in zip(pairs, cells)])
            if pairs
            else empty
        )
        local, remote, lengths, regions = _coalesce(local, remote, lengths, regions)
        nbytes = int(lengths.sum())
        spec = _Spec(source, key, local, remote, lengths, regions, [], {c[4] for c in cells}, nbytes)
        if local.shape[0] > DIRECT_DESCRIPTORS_PER_SHARD * max(len(cells), 1):
            self._restage(spec, cells)
        self._specs[(source, key)] = spec
        return spec

    def _ipc_cells(self, source, remote_info, shards, positions, cells) -> list | None:
        """
        `(destination, source)` view pairs when `source` is reachable without a NIC, else None.

        Two topologies qualify: the source is on this device (a plain device copy) or on
        another device of this node with peer access (`cudaMemcpyPeer`, NVLink/PCIe). The
        source views are strided over the mapped region exactly as the remote descriptors
        would have addressed it, so the copy moves the same bytes the transfer would have.
        Anything else -- another node, an unexportable allocation, no peer path -- returns
        None and the read goes through NIXL.
        """
        info = self.client.all_client_infos[source]
        if not info.ipc_regions or self.client.device.type != "cuda":
            return None
        mapper = self.client.region_mapper()
        local_index = self.client.device.index or 0
        same_device = info.device_uuid == mapper.uuid
        if same_device or info.node_ip == self._local_ip:
            # A same-node source is reachable, including one this process cannot see:
            # per-instance `CUDA_VISIBLE_DEVICES` hides the other GPUs of the node, but
            # `cudaIpcOpenMemHandle` still maps them with lazy peer access, and the copy
            # runs at NVLink speed (measured bit-exact at 390 GB/s with the source GPU
            # invisible). The imported storage must be attributed to a *local* ordinal
            # either way -- the exporter's numbering means a different physical GPU here
            # whenever the two processes were pinned differently.
            target = local_device(info.device_uuid) if not same_device else None
            target = torch.device("cuda", local_index) if target is None else target
            if target.index != local_index and not torch.cuda.can_device_access_peer(local_index, target.index):
                return None
        else:
            return None  # another node: only NIXL can reach it
        if not mapper.knows(source) and not mapper.adopt(source, info.ipc_regions, target):
            return None
        pairs, nbytes = [], 0
        for shard, (local, local_runs, remote_runs, remote_strides, _version, _region) in zip(shards, cells):
            pos = positions[shard]
            address = remote_info.read_addrs[pos]
            itemsize = local.element_size()
            # The span the view addresses, not just the bytes it moves: a strided shard
            # reaches past its covered byte count, and `locate` bounds-checks the span.
            span = (
                int(remote_runs.starts[-1]) + remote_runs.run_bytes - address
                if remote_runs.count
                else 0
            )
            found = mapper.resolve(source, address, span)
            if found is None or found[1] % itemsize:
                return None  # unmapped, or a view that would not start on an element boundary
            region, offset = found
            typed = region.view(local.dtype)
            view = torch.as_strided(typed, tuple(local.shape), tuple(remote_strides), offset // itemsize)
            pairs.append((local, view))
            nbytes += local_runs.total_bytes
        self._ipc_peer_bytes += 0 if same_device else nbytes
        return pairs

    def _restage(self, spec: _Spec, cells: list) -> None:
        """Pick, per shard, the cheapest of a direct read, a staged read, or a staged span read."""
        keep_local, keep_remote, keep_len, keep_region = [], [], [], []
        for local, lr, rr, remote_strides, _version, region in cells:
            nbytes = lr.total_bytes
            direct = nbytes // min(lr.run_bytes, rr.run_bytes)
            options = [(direct * SECONDS_PER_DESCRIPTOR + nbytes / BYTES_PER_SECOND, "direct")]
            if lr.count > 1:
                options.append((rr.count * SECONDS_PER_DESCRIPTOR + nbytes / BYTES_PER_SECOND, "stage"))
            span = int(rr.starts[-1] - rr.starts[0]) + rr.run_bytes
            if rr.count > 1:
                options.append((SECONDS_PER_DESCRIPTOR + span / BYTES_PER_SECOND, "span"))
            kind = min(options)[1]
            if kind == "direct":
                lo, ro, piece = pair_runs(lr, rr)
                keep_local.append(lo)
                keep_remote.append(ro)
                keep_len.append(np.full(lo.shape, piece, dtype=np.int64))
                keep_region.append(np.full(lo.shape, region, dtype=np.int64))
            elif kind == "stage":
                rel = np.arange(rr.count, dtype=np.int64) * rr.run_bytes
                spec.staged.append(
                    _Staged(
                        spec.key,
                        nbytes,
                        rr.starts,
                        rel,
                        np.full(rr.count, rr.run_bytes, dtype=np.int64),
                        local,
                        tuple(local.shape),
                        tuple(_dense(local.shape)),
                    )
                )
            else:
                spec.staged.append(
                    _Staged(
                        spec.key,
                        span,
                        rr.starts[:1].copy(),
                        np.zeros(1, dtype=np.int64),
                        np.array([span], dtype=np.int64),
                        local,
                        tuple(local.shape),
                        remote_strides,
                    )
                )
        if keep_local:
            spec.direct_local, spec.direct_remote, spec.direct_len, spec.direct_region = _coalesce(
                np.concatenate(keep_local),
                np.concatenate(keep_remote),
                np.concatenate(keep_len),
                np.concatenate(keep_region),
            )
        else:
            empty = np.zeros(0, dtype=np.int64)
            spec.direct_local, spec.direct_remote, spec.direct_len, spec.direct_region = empty, empty, empty, empty

    # ------------------------------------------------------------------ handles

    def _ensure_pool(self, need: int) -> None:
        size = max(self.staging_bytes, 2 * need)
        size = -(-size // (2 * _ALIGN)) * (2 * _ALIGN)
        if self._pool is not None and self._pool.numel() >= size:
            return
        # Every cached prepared handle embeds addresses from this pull's geometry, and the
        # staged ones point into the pool being replaced. Release them, but not the ones a
        # running read still holds: a pull with peer reads can need a bigger item in a
        # later wave, at which point the peer handles are already in flight. Freeing one
        # would make the next status poll read a freed request (NIXL: "BAD_STATUS").
        self._release_handles()
        if self._pool is not None:
            # Not deregistered yet: an orphaned round may still be writing into it.
            self._retired_pools.append((self._pool, self._pool_reg))
        self._pool = torch.empty(size, dtype=torch.uint8, device=self.client.device)
        self._pool_reg = self.agent.register_memory([self._pool])

    def _make_handle(
        self,
        source: str,
        local: np.ndarray,
        remote: np.ndarray,
        lengths: np.ndarray,
        remote_mem: str,
        remote_dev: int,
        keys: frozenset[str] = frozenset(),
        reads_pool: bool = False,
    ) -> _Handle:
        count = int(local.shape[0])
        # A colocate reader skips NIXL registration because every source is copyable. If a
        # read still lands here, one source was not; register now rather than fail the pull.
        self.client.ensure_rdma_registered()
        dev = self.client.device.index or 0
        local_mem = "VRAM" if self.client.device.type == "cuda" else "DRAM"
        local_rows = np.stack([local, lengths, np.full(count, dev, dtype=np.int64)], axis=1).astype(np.uint64)
        remote_rows = np.stack([remote, lengths, np.full(count, remote_dev, dtype=np.int64)], axis=1).astype(np.uint64)
        agent_name = self.client.agent_name_of(source)
        local_dlist = self.agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_rows, mem_type=local_mem)
        remote_dlist = self.agent.prep_xfer_dlist(agent_name, remote_rows, mem_type=_NIXL_MEM[remote_mem])
        self._dlists.extend([local_dlist, remote_dlist])
        indices = np.arange(count, dtype=np.int32)
        handle = self.agent.make_prepped_xfer(
            "READ", local_dlist, indices, remote_dlist, indices, f"pull:{self.client.client_name}".encode()
        )
        return _Handle(source, handle, count, int(lengths.sum()), keys=keys, reads_pool=reads_pool)

    def _phase(self, specs: list[_Spec]) -> _Phase:
        signature = tuple(sorted((s.source, s.key) for s in specs))
        cached = self._phases.get(signature)
        if cached is not None:
            return cached
        phase = _Phase(keys=frozenset(spec.key for spec in specs))
        by_source: dict[str, list[_Spec]] = defaultdict(list)
        for spec in specs:
            by_source[spec.source].append(spec)
            phase.versions |= spec.versions
        staged: list[tuple[str, _Staged]] = []
        for spec in specs:
            phase.ipc.extend(spec.ipc)
            if spec.ipc:
                phase.ipc_bytes += spec.nbytes
        for source, group in sorted(by_source.items()):
            staged.extend((source, item) for s in group for item in s.staged)

        if staged:
            # Before any handle exists: this may invalidate the phase cache, which holds
            # handles other phases of this same pull already posted.
            self._ensure_pool(max(_aligned(item.nbytes) for _, item in staged))
            phase.pool = self._pool

        for source, group in sorted(by_source.items()):
            info = self.client.all_client_infos[source].tensor_infos[group[0].key]
            local, remote, lengths, _ = _coalesce(
                np.concatenate([s.direct_local for s in group]),
                np.concatenate([s.direct_remote for s in group]),
                np.concatenate([s.direct_len for s in group]),
                np.concatenate([s.direct_region for s in group]),
            )
            if local.shape[0]:
                phase.direct.append(
                    self._make_handle(
                        source,
                        local,
                        remote,
                        lengths,
                        info.mem_type,
                        info.device_id,
                        keys=frozenset(s.key for s in group),
                    )
                )

        if staged:
            assert phase.pool is not None, "a staged phase has no staging pool"
            half = phase.pool.numel() // 2
            rounds: list[list[tuple[str, _Staged, int]]] = [[]]
            used = 0
            for source, item in staged:
                if used + _aligned(item.nbytes) > half:
                    rounds.append([])
                    used = 0
                rounds[-1].append((source, item, used))
                used += _aligned(item.nbytes)
            base = self._pool.data_ptr()
            for index, members in enumerate(rounds):
                slot_base = base + (index % 2) * half
                handles = []
                per_source: dict[str, list[tuple[_Staged, int]]] = defaultdict(list)
                for source, item, offset in members:
                    per_source[source].append((item, offset))
                for source, items in sorted(per_source.items()):
                    info = self.client.all_client_infos[source].tensor_infos[by_source[source][0].key]
                    local = np.concatenate([slot_base + offset + item.rel for item, offset in items])
                    remote = np.concatenate([item.remote for item, _ in items])
                    lengths = np.concatenate([item.lengths for item, _ in items])
                    handles.append(
                        self._make_handle(
                            source,
                            local,
                            remote,
                            lengths,
                            info.mem_type,
                            info.device_id,
                            keys=frozenset(item.key for item, _ in items),
                            reads_pool=True,
                        )
                    )
                phase.rounds.append(handles)
                phase.round_items.append([(item, (index % 2) * half + offset) for _, item, offset in members])
                phase.staged_bytes += sum(item.nbytes for _, item, _ in members)
        phase.descriptors = sum(h.descriptors for h in phase.direct) + sum(
            h.descriptors for handles in phase.rounds for h in handles
        )
        phase.nbytes = sum(s.nbytes for s in specs)
        self._phases[signature] = phase
        return phase

    # ------------------------------------------------------------------ execution

    def _poll_pull_control(self) -> None:
        """Observe a write-gate preemption and acknowledge it immediately.

        Called from the transfer wait loops. The acknowledgement is what tells the writer
        that reads completed from here on are past its boundary, so it is sent as soon as
        the notice is seen -- it does not wait for any transfer to drain. Reads already
        posted are classified as suspect by the caller and cancelled before repair.
        """
        if self._pull_epoch is None or self._preempt_epoch is not None:
            return
        now = time.time()
        if now < self._next_control_poll:
            return
        self._next_control_poll = now + self._control_poll_interval_s
        entry = self.client.drain_pull_preempt(self.client.client_name)
        if entry is None:
            return
        epoch = int(entry.get("epoch", -1))
        if epoch <= self._pull_epoch:
            return
        self._preempt_epoch = epoch
        self._preempt_version = int(entry.get("version", -1))
        self._preempt_holders = tuple(entry.get("holders", ()))
        if self._on_preempt_ack is not None:
            self._on_preempt_ack(epoch)

    def _note_completed(self, handle: _Handle) -> None:
        """Mark a handle's keys complete once every read that covers them finished."""
        for key in handle.keys:
            if key in self._quantize_keys:
                # The source bytes are in the staging pool, but the target is not written
                # until `_quantize_scatter`. A preemption must not certify it before then.
                continue
            remaining = self._key_outstanding.get(key, 0) - 1
            self._key_outstanding[key] = remaining
            if remaining <= 0:
                self._completed_keys.add(key)

    def _check_preempted(self) -> None:
        if self._preempt_epoch is not None and not self._repairing:
            raise PullPreempted(
                self._preempt_epoch,
                self._preempt_version,
                frozenset(self._completed_keys),
                frozenset(self._pull_keys - self._completed_keys),
                self._preempt_holders,
            )

    def _post(self, handles: list[_Handle]) -> list[_Handle]:
        posted = []
        for h in handles:
            h.posted_at = time.time()
            if h.source in self.failed:
                continue
            try:
                state = self.agent.transfer(h.handle)
            except Exception as exc:  # the transport may raise rather than return ERR for a lost peer
                psrl_logger.warning(f"{self.client.client_name}: posting a read from {h.source} raised {exc!r}.")
                state = "ERR"
            if state == "ERR":
                psrl_logger.warning(f"{self.client.client_name}: posting a read from {h.source} failed.")
                self.failed.add(h.source)
                continue
            self._in_flight[id(h)] = h
            for key in h.keys:
                self._key_outstanding[key] = self._key_outstanding.get(key, 0) + 1
            posted.append(h)
        return posted

    def _wait(self, handles: list[_Handle], timeout: float | None = None, copies=()) -> None:
        """
        Wait for every handle, polling them together and recording when each source finished.

        A handle that errors or outlives the read timeout marks its source failed rather than
        raising, so the other handles still complete and the caller can re-read that source's
        keys from a replica. A late completion from a merely slow source writes the same
        version's bytes, so it cannot corrupt the re-read.

        `copies` are `(event, keys)` pairs for IPC/peer copies this phase issued. They are
        polled alongside the transfers so a preemption sees exactly which copied keys had
        drained before the writer was about to overwrite their source.
        """
        timeout = self.read_timeout_s if timeout is None else timeout
        start = time.time()
        pending = list(handles)
        copies = list(copies)
        while pending or copies:
            self._poll_pull_control()
            remaining_copies = []
            for event, keys in copies:
                if event.query():
                    self._note_copied(keys)
                else:
                    remaining_copies.append((event, keys))
            copies = remaining_copies
            self._check_preempted()
            still = []
            for h in pending:
                state = self.agent.check_xfer_state(h.handle)
                if state == "DONE":
                    self._in_flight.pop(id(h), None)
                    self._note_completed(h)
                    done = time.time()
                    bytes_, first, last = self.source_trace.get(h.source, (0, done, done))
                    self.source_trace[h.source] = (bytes_ + h.nbytes, min(first, h.posted_at), done)
                    continue
                if state == "ERR":
                    self._in_flight.pop(id(h), None)
                    psrl_logger.warning(f"{self.client.client_name}: read of {h.nbytes} bytes from {h.source} failed.")
                    self.failed.add(h.source)
                    continue
                still.append(h)
            pending = still
            if pending or copies:
                if time.time() - start > timeout:
                    if pending:
                        late = sorted({h.source for h in pending})
                        psrl_logger.warning(f"{self.client.client_name}: reads from {late} exceeded {timeout}s.")
                        self.failed.update(late)
                    return
                time.sleep(5e-5)

    def _scatter(self, items: list[tuple[_Staged, int]], pool: torch.Tensor) -> None:
        with torch.cuda.stream(self._copy_stream) if self._copy_stream is not None else nullcontext():
            for item, offset in items:
                if item.quantize:
                    assert item.source_dtype is not None and item.shard is not None
                    typed = pool.view(item.source_dtype)
                    source = torch.as_strided(
                        typed,
                        item.view_shape,
                        item.view_strides,
                        offset // typed.element_size(),
                    )
                    unit = self._unit_of(item.key)
                    if unit is None:
                        self._quantize_scatter(item, source)
                    else:
                        self._stage_unit_member(unit, item, source)
                    continue
                typed = pool.view(item.local_view.dtype)
                source = torch.as_strided(
                    typed, item.view_shape, item.view_strides, offset // item.local_view.element_size()
                )
                item.local_view.copy_(source, non_blocking=True)

    def _unit_of(self, key: str) -> _Unit | None:
        """The unit `key` feeds, or None for a key quantized on its own."""
        return self._units().get(self._unit_of_key.get(key, ""))

    def _units(self) -> dict[str, _Unit]:
        """Rollout parameters whose views share one scale, built from the registered layout.

        A member view is a contiguous row range of its parameter's FP8 storage, so the unit
        a view feeds is its destination rows. The scale is registered whole by the rollout
        and is never pulled.
        """
        if self._unit_index is not None:
            return self._unit_index
        units: dict[str, _Unit] = {}
        quantizer = self.client._quantizer
        if quantizer is None:
            self._unit_index = units
            return units
        mapping = self.client.get_original_tensor_mapping()
        infos = self.client.local_client_info.tensor_infos
        scale_of: dict[str, str] = {}
        cells: dict[str, list[tuple[str, tuple[int, ...], torch.Tensor]]] = defaultdict(list)
        for key, info in infos.items():
            unit_id = getattr(info.sharding, "quant_unit", None)
            if unit_id is None:
                continue
            if quantizer.base_of_scale_key(key) is not None:
                scale_of[unit_id] = key
                continue
            for shard in info.sharding.shard_indices:
                view = mapping.get((key, tuple(shard)))
                if view is not None and view.numel():
                    cells[unit_id].append((key, tuple(shard), view))
        for unit_id, members in cells.items():
            scale_key = scale_of.get(unit_id)
            assert scale_key is not None, f"{self.client.client_name}: unit {unit_id!r} has no local scale."
            scale_shards = infos[scale_key].sharding.shard_indices
            assert len(scale_shards) == 1, (
                f"{self.client.client_name}: unit scale {scale_key!r} has {len(scale_shards)} local shards."
            )
            scale_view = mapping.get((scale_key, tuple(scale_shards[0])))
            assert scale_view is not None, f"{self.client.client_name}: unit scale {scale_key!r} is not registered."
            first = members[0][2]
            assert first.dim() == 2 and first.stride(1) == 1 and first.stride(0) == first.shape[1], (
                f"{self.client.client_name}: unit {unit_id!r} member {members[0][0]!r} is not a dense 2D row view."
            )
            row_bytes = first.stride(0) * first.element_size()
            base = min(view.data_ptr() for _, _, view in members)
            end = max(view.data_ptr() + view.shape[0] * row_bytes for _, _, view in members)
            unit = _Unit(unit_id, scale_key, scale_view, first.shape[1], {}, (end - base) // row_bytes)
            for key, shard, view in members:
                assert view.shape[1] == unit.cols and (view.data_ptr() - base) % row_bytes == 0, (
                    f"{self.client.client_name}: unit {unit_id!r} member {key!r} does not sit on the unit's row grid."
                )
                unit.members[(key, shard)] = ((view.data_ptr() - base) // row_bytes, view.shape[0])
                self._unit_of_key[key] = unit_id
            units[unit_id] = unit
        self._unit_index = units
        return units

    @torch.no_grad()
    def _stage_unit_member(self, unit: _Unit, item: _Staged, source: torch.Tensor) -> None:
        """Copy a unit member's source bytes to its destination rows, to quantize them together."""
        assert item.source_dtype is not None
        if unit.buffer is None:
            unit.buffer = torch.empty((unit.rows, unit.cols), dtype=item.source_dtype, device=item.local_view.device)
        start, rows = unit.members[(item.key, item.shard)]
        assert item.local_view.shape == (rows, unit.cols) and item.source_dtype == unit.buffer.dtype, (
            f"{item.key!r} shard {item.shard}: view {tuple(item.local_view.shape)} {item.source_dtype} "
            f"does not match unit {unit.key!r} row {start} ({rows}, {unit.cols})."
        )
        unit.buffer[start : start + rows].copy_(source)
        unit.staged += 1
        unit.dirty = True

    @torch.no_grad()
    def _flush_units(self) -> None:
        """Quantize every unit whose members landed, and publish its scale."""
        quantizer = self.client._quantizer
        if quantizer is None:
            return
        mapping = self.client.get_original_tensor_mapping()
        for unit in self._units().values():
            if not unit.dirty:
                continue
            if unit.staged < len(unit.members):
                psrl_logger.warning(
                    f"{self.client.client_name}: unit {unit.key!r} received {unit.staged} of "
                    f"{len(unit.members)} member shards this pull, leaving it at the previous version."
                )
                unit.buffer, unit.staged, unit.dirty = None, 0, False
                continue
            q, s = quantizer.quantize(unit.buffer)
            assert tuple(unit.scale_view.shape) == tuple(s.shape), (
                f"{unit.scale_key}: derived scale {tuple(s.shape)} does not match the registered "
                f"{tuple(unit.scale_view.shape)}."
            )
            unit.scale_view.copy_(s)
            for (key, shard), (start, rows) in unit.members.items():
                mapping[(key, shard)].copy_(q[start : start + rows])
                self._quantize_pending.pop(key, None)
                self._quantize_keys.discard(key)
                self._completed_keys.add(key)
            unit.buffer, unit.staged, unit.dirty = None, 0, False

    @torch.no_grad()
    def _quantize_scatter(self, item: _Staged, source: torch.Tensor) -> None:
        """Quantize a staged BF16/FP32 source shard into the rollout's FP8 view and scale.

        The scale is derived from the same source tile as the weight. A finer rollout
        scale grid receives the coarse 128x128 scale repeated over the cells it covers,
        matching what the trainer-side publisher wrote.

        The bytes are written straight into the registered view. Quantizing into a
        temporary and copying it in would read and write every FP8 byte twice.
        """
        quantizer = self.client._quantizer
        assert quantizer is not None
        block = quantizer.plan.block
        assert block is not None
        target = item.local_view
        scale_key = quantizer.scale_key_of(item.key)
        scale_view = self.client.get_original_tensor_mapping().get((scale_key, item.shard))
        scale_dest = scale_view
        refine = None
        if scale_view is not None:
            scale_tile = getattr(
                self.client.local_client_info.tensor_infos[scale_key].sharding,
                "scale_tile",
                None,
            ) or block
            repeat = tuple(int(b // t) for b, t in zip(block, scale_tile))
            if repeat != (1, 1):
                # The weight still lands in its registered view. Only the coarse scale
                # needs a temporary, and it is 1/16384 of the weight.
                scale_dest = torch.empty(
                    quantizer.scale_shape(source),
                    dtype=quantizer.scale_dtype,
                    device=source.device,
                )
                refine = repeat
            else:
                assert tuple(scale_view.shape) == tuple(quantizer.scale_shape(source)), (
                    f"{scale_key} shard {item.shard}: registered scale {tuple(scale_view.shape)} "
                    f"does not match the tile grid {tuple(quantizer.scale_shape(source))}."
                )
        if scale_dest is None:
            scale_dest = torch.empty(
                quantizer.scale_shape(source),
                dtype=quantizer.scale_dtype,
                device=source.device,
            )

        def refined_scale(coarse: torch.Tensor) -> torch.Tensor:
            """Repeat the coarse grid, then crop to the shard's actual tile count."""
            assert refine is not None
            refined = quantizer.refine_scale(coarse, refine)
            if tuple(refined.shape) == tuple(scale_view.shape):
                return refined
            assert all(r >= v for r, v in zip(refined.shape, scale_view.shape)), (
                f"{scale_key} shard {item.shard}: refined scale {tuple(refined.shape)} is smaller "
                f"than the registered {tuple(scale_view.shape)}."
            )
            return refined[tuple(slice(0, n) for n in scale_view.shape)]

        if quantizer.can_quantize_into(source, target, scale_dest):
            quantizer.quantize_into(source, target, scale_dest)
            if refine is not None:
                scale_view.copy_(refined_scale(scale_dest))
        else:
            # A destination the fused kernel cannot address is still writable by `copy_`.
            q, s = quantizer.quantize(source)
            target.copy_(q)
            if scale_view is not None:
                if refine is not None:
                    s = refined_scale(s)
                scale_view.copy_(s)
        pending = self._quantize_pending.get(item.key, 1) - 1
        if pending <= 0:
            self._quantize_pending.pop(item.key, None)
            self._quantize_keys.discard(item.key)
            self._completed_keys.add(item.key)
        else:
            self._quantize_pending[item.key] = pending

    def _run(self, phase: _Phase) -> None:
        handles, copy = self._start(phase)
        self._finish(handles, copy)

    def _finish(self, direct: list[_Handle], copy=None) -> None:
        copies = [] if copy is None else ([copy] if isinstance(copy, tuple) else copy)
        self._wait(direct, copies=copies)
        for event, keys in copies:
            event.synchronize()
            self._note_copied(keys)
        if self._copy_stream is not None:
            self._copy_stream.synchronize()

    def _note_copied(self, keys: frozenset[str] | set[str]) -> None:
        """Mark keys served only by IPC/peer copies complete, once the copy stream has run them."""
        for key in keys:
            if self._key_outstanding.get(key, 0) <= 0:
                self._completed_keys.add(key)

    def _start(self, phase: _Phase) -> tuple[list[_Handle], tuple[torch.cuda.Event, frozenset[str]] | None]:
        """Post the direct reads, run the staged rounds, and return the handles to wait on.

        Also returns, when this phase copied keys instead of transferring them, the event
        after which those copies are known complete. Preemption classification has to see
        that event: a peer copy from the trainer is reading the same pages the optimizer is
        about to write, exactly like a NIXL read, so calling the keys complete before the
        copy hardware drains would certify torn bytes.
        """
        if self._copy_stream is None and self.client.device.type == "cuda":
            self._copy_stream = torch.cuda.Stream(device=self.client.device)
        if id(phase) not in self._accounted_phases:
            self._accounted_phases.add(id(phase))
            for members in phase.round_items:
                for item, _offset in members:
                    if item.quantize:
                        self._quantize_keys.add(item.key)
                        self._quantize_pending[item.key] = self._quantize_pending.get(item.key, 0) + 1
        direct = self._post(phase.direct)
        copy = None
        if phase.ipc:
            transferred = {key for handle in phase.direct for key in handle.keys} | {
                key for handles in phase.rounds for handle in handles for key in handle.keys
            }
            copy_keys = frozenset(phase.keys - transferred)
            # Copies are issued on the copy stream that `_finish` waits for. The views are
            # detached because a shard is often an output of `chunk`, which autograd refuses
            # to let anyone write into; the transfer path wrote raw addresses and never met
            # that rule. Weights are data here, so stepping outside autograd is the point.
            with (
                torch.no_grad(),
                torch.cuda.stream(self._copy_stream) if self._copy_stream is not None else nullcontext(),
            ):
                for destination, origin in phase.ipc:
                    destination.detach().copy_(origin.detach(), non_blocking=True)
                if copy_keys and self._copy_stream is not None:
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                    copy = (event, copy_keys)
            if copy is None:
                self._note_copied(copy_keys)
        slot_free: list[torch.cuda.Event | None] = [None, None]
        in_flight: list[tuple[int, list[_Handle]]] = []

        def finish_oldest() -> None:
            index, handles = in_flight.pop(0)
            self._wait(handles)
            assert phase.pool is not None, "a staged round has no staging pool"
            self._scatter(phase.round_items[index], phase.pool)
            if self._copy_stream is not None:
                event = torch.cuda.Event()
                event.record(self._copy_stream)
                slot_free[index % 2] = event

        for index, handles in enumerate(phase.rounds):
            if len(in_flight) == 2:
                finish_oldest()
            event = slot_free[index % 2]
            if event is not None:
                event.synchronize()
            in_flight.append((index, self._post(handles)))
        while in_flight:
            finish_oldest()
        return direct, copy

    # ------------------------------------------------------------------ pull

    def mark_unavailable(self, sources) -> None:
        """Exclude train sources from every later pull, for example after the coordinator saw them die."""
        self.unavailable.update(sources)

    def _replica_of(self, key: str, shards: list, load: dict[str, int]) -> str | None:
        """
        A live train source holding every shard in `shards` of `key`, least loaded first.

        DP replicas of a train rank declare the same shard indices, so any of them serves the
        same bytes at the same version.
        """
        wanted = set(map(tuple, shards))
        candidates = []
        for name, info in self.client.all_client_infos.items():
            if name in self.unavailable or name in self.failed or info.type is not NIXLClientType.PUSH_SIDE:
                continue
            tensor_info = info.tensor_infos.get(key)
            if tensor_info is None or not wanted <= set(map(tuple, tensor_info.sharding.shard_indices)):
                continue
            candidates.append((load.get(name, 0), name))
        return min(candidates)[1] if candidates else None

    def _reroute(self, keys: dict[str, dict[str, list]]) -> tuple[dict[str, dict[str, list]], int, list[str]]:
        """
        Replace every unavailable or failed source in `keys` by a replica.

        Returns the rerouted reads, how many keys moved, and the keys no live source holds.
        """
        dead = self.unavailable | self.failed
        load: dict[str, int] = {}
        out: dict[str, dict[str, list]] = {}
        moved, missing = 0, []
        for key, per_source in keys.items():
            if not dead.intersection(per_source):
                out[key] = per_source
                continue
            rerouted: dict[str, list] = {}
            for source, shards in per_source.items():
                target = source
                if source in dead:
                    target = self._replica_of(key, shards, load)
                    if target is None:
                        missing.append(key)
                        break
                    load[target] = load.get(target, 0) + 1
                rerouted.setdefault(target, []).extend(shards)
            else:
                out[key] = {src: sorted(map(tuple, shards)) for src, shards in rerouted.items()}
                moved += 1
        return out, moved, missing

    def _prepare_pool(self, upstream: dict[str, dict[str, list]], peer: dict[str, dict[str, list]]) -> None:
        """Allocate the staging pool for the largest item this pull can plan.

        Building every planned spec up front is cheap (they are cached by `(source, key)`)
        and makes the pool geometry constant for the whole pull. A group that only appears
        through mid-pull take-over can still ask for more; that path keeps the in-flight
        handle protection in `_ensure_pool`.
        """
        need = 0

        def consider(source: str, key: str, shards: list) -> None:
            nonlocal need
            spec = self._spec(source, key, shards)
            for item in spec.staged:
                need = max(need, _aligned(item.nbytes))

        # `peer` has the same `{key: {source: shards}}` shape as `upstream` now that
        # designates are elected per shard rather than per whole key, so one loop
        # considers both -- and, critically, both read the shard list from the mapping
        # itself rather than re-deriving it, which is what keeps every call site that
        # can reach the same `(source, key)` pair (`_spec`'s cache key) agreeing on
        # exactly the same shards for that pair.
        for mapping in (upstream, peer):
            for key, per_source in mapping.items():
                for source, shards in per_source.items():
                    consider(source, key, shards)
        if need:
            self._ensure_pool(need)

    def _pick_holder(self, key: str, holders: tuple[str, ...]) -> str | None:
        """First same-node holder whose declared shards cover this client's for ``key``."""
        local = self.client.local_client_info.tensor_infos.get(key)
        if local is None:
            return None
        shards = list(local.sharding.shard_indices)
        wanted = set(map(tuple, shards))
        quantizer = self.client._quantizer
        unit = self._unit_of(key) is not None
        for name in holders:
            if name == self.client.client_name or name in self.failed or name in self.unavailable:
                continue
            info = self.client.all_client_infos.get(name)
            if info is None:
                continue
            tensor = info.tensor_infos.get(key)
            if tensor is None:
                continue
            if unit and quantizer is not None and tensor.shard_meta_infos:
                meta = tensor.shard_meta_infos[0]
                if meta is not None and meta.dtype == quantizer.weight_dtype:
                    # A unit must be re-quantized from the trainer's bytes: the holder's FP8
                    # output is only meaningful together with the scale it was made with.
                    continue
            if wanted <= set(map(tuple, tensor.sharding.shard_indices)) and self._source_readable(
                name, key, shards
            ):
                return name
        return None

    def _abandon_inflight(self) -> None:
        """Cancel this pull's outstanding requests and drop the prepared-handle cache.

        Called only on preemption. A cancelled read may have written part of its
        destination, which is exactly why it is re-read from a holder: the repair
        overwrites the whole key. Waiting for the transfers instead would let a large
        upstream wave delay the writer, which is what the preemption exists to avoid.
        """
        self._release_handles()
        for handle in self._orphans:
            try:
                self.agent.release_xfer_handle(handle.handle)
            except Exception as exc:  # the transport may already have finished it
                psrl_logger.warning(f"{self.client.client_name}: abandoning preempted read raised {exc!r}.")
        self._orphans.clear()
        self._in_flight.clear()

    def _repair_after_preempt(self, preempted: PullPreempted) -> PullReport:
        """Finish a preempted pull by re-reading its unfinished keys from certified holders.

        The holders were chosen by the coordinator from the committed-version ledger and
        are pinned for the write window, so their bytes are an immutable copy of the
        version the reader is completing. If any suspect key has no such holder, the pull
        cannot land one version and the caller must wait for the next one.
        """
        self._abandon_inflight()
        suspect = set(preempted.suspect_keys)
        if preempted.version < 0:
            raise PullPreempted(
                preempted.epoch,
                preempted.version,
                preempted.completed_keys,
                frozenset(suspect),
                preempted.holders,
                repaired=False,
                reason="preempt_notice_carried_no_version",
            )
        tensors = self.client.local_client_info.tensor_infos
        specs: list[_Spec] = []
        missing: list[str] = []
        for key in sorted(suspect):
            holder = self._pick_holder(key, preempted.holders)
            if holder is None:
                missing.append(key)
                continue
            local = tensors[key]
            specs.append(self._spec(holder, key, list(local.sharding.shard_indices)))
        if missing:
            raise PullPreempted(
                preempted.epoch,
                preempted.version,
                preempted.completed_keys,
                frozenset(suspect),
                preempted.holders,
                repaired=False,
                reason=f"no same-node holder for {len(missing)} key(s)",
            )

        peer_bytes = sum(spec.nbytes for spec in specs)
        phase: _Phase | None = None
        if specs:
            self._repairing = True
            try:
                phase = self._phase(specs)
                self._run(phase)
            finally:
                self._repairing = False
        if self.client.device.type == "cuda":
            torch.cuda.synchronize(self.client.device)
        self._flush_units()
        done = time.time()
        psrl_logger.info(
            f"{self.client.client_name}: pull repaired after preemption: "
            f"{len(suspect)} key(s) re-read from {len({s.source for s in specs})} holder(s) "
            f"({peer_bytes / 1e9:.2f} GB), version {preempted.version}."
        )
        return PullReport(
            seconds=done - self._pull_started,
            upstream_seconds=0.0,
            peer_seconds=done - self._pull_started,
            upstream_keys=0,
            peer_keys=len(suspect),
            fallback_keys=0,
            upstream_bytes=0,
            peer_bytes=peer_bytes,
            descriptors=phase.descriptors if phase is not None else 0,
            staged_bytes=phase.staged_bytes if phase is not None else 0,
            followers=0,
            versions={preempted.version},
            sources=dict(self.source_trace),
            takeover_keys=0,
            failed_sources=set(self.failed),
            repaired=True,
        )

    def _elect(
        self,
        me: str,
        plan,
        participates,
        group_key: str,
        shard: tuple[int, ...],
        read_key: str,
    ) -> tuple[str | None, list[str]]:
        """Live, readable designate for one shard's peer group, and the live members.

        `group_key` is the key the peer group is attached to (a derived scale key defers
        to its weight key's group at the same shard). `read_key` is the key
        `_source_readable` checks readability against -- the key whose bytes are
        actually about to be read from the candidate, which differs from `group_key`
        exactly for a scale key (the group lives under the weight key; the read is of
        the scale key). Returns ``(None, [])`` when there are fewer than 2 live
        members, so the caller must fall back to reading this shard upstream itself.
        """
        live = [w for w in plan.get_peer_group(me, group_key, shard) if w == me or participates(w)]
        if len(live) < 2:
            return None, []
        readable = [w for w in live if w == me or self._source_readable(w, read_key, [shard])]
        return (readable[0] if readable else None), live

    def _assign(self, participates) -> tuple[dict[str, dict[str, list]], dict[str, dict[str, list]], set[str]]:
        """Split this client's keys into upstream reads and reads from a designate.

        A peer is only eligible as the designate when this client holds its **full**
        metadata. Peers are sent conn-info-only metadata when the handshake plan does not
        make them readers (E4), and a sync over a participant subset can elect a different
        designate from the same peer group. Electing one whose addresses this client never
        loaded produces a read with no target and must fall back upstream instead.

        Designates are elected per individual finest-mesh shard (phase 2's canonical
        chunk), not per whole key: two readers whose whole shard sets for a key never
        match (different TP/EP layouts) can still share a designate for the individual
        shards they happen to overlap on. `peer`'s shape mirrors `upstream`'s exactly
        (`{key: {source: shards}}`) so one key's shards can come from a mix of upstream
        and one or more different peer designates in the same assignment.
        """
        me = self.client.client_name
        plan = self.client.comm_plan
        assert plan is not None, f"{me} has no communication plan."
        tensors = self.client.local_client_info.tensor_infos
        upstream: dict[str, dict[str, list]] = {}
        peer: dict[str, dict[str, list]] = {}
        followers: set[str] = set()
        reader_quantization = self.client._reader_quantization
        quantizer = self.client._quantizer
        for key, per_source in plan.rollout_pull_from_ps_plan.get(me, {}).items():
            local = tensors.get(key)
            base = quantizer.base_of_scale_key(key) if (reader_quantization and quantizer is not None) else None
            unit = getattr(getattr(local, "sharding", None), "quant_unit", None)
            if local is None:
                if base is None:
                    upstream[key] = per_source
                continue
            shard_source = {shard: source for source, shards in per_source.items() for shard in shards}
            if unit is not None:
                # A unit's scale spans every member view, so a replica can only produce it
                # after all its own bytes land. Members are read upstream by every replica
                # rather than from a designate, whose output is incomplete until then.
                if base is not None:
                    continue
                for shard in local.sharding.shard_indices:
                    upstream.setdefault(key, {}).setdefault(shard_source[shard], []).append(shard)
                continue
            if base is not None:
                # `_scale_inv` has no upstream source. It is computed by the reader
                # that quantizes the weight. A scale shard shards exactly like its
                # weight (the server deep-copies the weight's already-unified sharding
                # onto the scale key), so it defers entirely to the weight's own
                # designate election at that shard coordinate -- no independent
                # `_source_readable` call or ordering of its own.
                for shard in local.sharding.shard_indices:
                    designate, _live = self._elect(me, plan, participates, base, shard, key)
                    if designate is not None and designate != me:
                        peer.setdefault(key, {}).setdefault(designate, []).append(shard)
                    # No live peer: `me` produces this shard's scale locally while
                    # quantizing its own weight shard. Nothing to add to upstream or peer.
                continue
            # An ordinary weight (or non-reader-quantization) key: the designate reads
            # this shard upstream (and quantizes it, if applicable); a follower reads
            # the designate's output instead.
            psrl_logger.debug(f"{me}: peer shards for {key!r} are {plan.get_peer_shards(me, key)!r}.")
            for shard in local.sharding.shard_indices:
                designate, live = self._elect(me, plan, participates, key, shard, key)
                if designate is not None:
                    if designate != me:
                        peer.setdefault(key, {}).setdefault(designate, []).append(shard)
                        continue
                    followers.update(w for w in live if w != me)
                upstream.setdefault(key, {}).setdefault(shard_source[shard], []).append(shard)
        return upstream, peer, followers

    def _source_readable(self, source: str, key: str, shards: list) -> bool:
        """Whether ``source`` exposes an address for every one of ``shards`` of ``key``.

        A conn-info-only or meta-only ``NIXLClientInfo`` has no read addresses; planning a
        read against it would only fail later in `_spec`, after other waves had moved.
        """
        info = self.client.all_client_infos.get(source)
        if info is None:
            psrl_logger.debug(f"{self.client.client_name}: {source!r} has no entry in all_client_infos.")
            return False
        tensor = info.tensor_infos.get(key)
        if tensor is None or not tensor.read_addrs:
            psrl_logger.debug(
                f"{self.client.client_name}: {source!r} has no read_addrs for {key!r} "
                f"(tensor is None: {tensor is None})."
            )
            return False
        positions = {shard: index for index, shard in enumerate(tensor.sharding.shard_indices)}
        return all(
            (index := positions.get(tuple(shard))) is not None and tensor.read_addrs[index] is not None
            for shard in shards
        )

    def _upstream_specs(self, keys: dict[str, dict[str, list]]) -> list[_Spec]:
        return [self._spec(src, key, shards) for key, per_source in keys.items() for src, shards in per_source.items()]

    def _waves_of(self, keys: list[str]) -> dict[str, int]:
        """
        Wave of every key, cut in model order into byte balanced runs.

        Group members hold identical keys, so a designate and its followers derive the same waves.
        A reader-derived scale key shares its weight key's wave: the designate computes
        the scale while it quantizes that weight, and followers may only read it after the
        wave is signalled.
        """
        info = self.client.local_client_info.tensor_infos
        quantizer = self.client._quantizer if self.client._reader_quantization else None

        def base_key(key: str) -> str:
            if quantizer is None:
                return key
            base = quantizer.base_of_scale_key(key)
            return base if base is not None and base in info else key

        grouped: dict[str, list[str]] = defaultdict(list)
        for key in keys:
            grouped[base_key(key)].append(key)
        sizes = [
            (base, sum(info[base].get_shard_size_bytes(i) for i in range(info[base].num_local_shards)))
            for base in grouped
        ]
        total = max(sum(n for _, n in sizes), 1)
        waves, acc = {}, 0
        for base, nbytes in sorted(sizes):
            wave = min(self.waves - 1, acc * self.waves // total)
            for key in grouped[base]:
                waves[key] = wave
            acc += nbytes
        return waves

    def pull(
        self,
        epoch: int,
        participants: set[str] | None = None,
        version_holders: frozenset[str] | None = None,
        target_version: int = -1,
        pull_epoch: int | None = None,
        on_preempt_ack=None,
    ) -> PullReport:
        """
        Read this client's weights for one sync.

        Upstream reads go in waves. After wave `w` lands the client tells its followers, and
        reads its own peer keys of wave `w - 1` over NVLink while wave `w` crosses the NIC.

        Args:
            epoch (int): Identifier of this sync, shared by every participant and never reused.
            participants (set[str] | None): Pull clients syncing together. None means all of them.
            version_holders (frozenset[str] | None): Trainer client names that have published
                ``target_version``.  When provided, upstream sources for each key are filtered to
                this set before any byte moves (D1 version-aware source selection).  Sources not
                in the set are skipped; if filtering would leave a key with no source the original
                source list is kept as a fallback so the pull never deadlocks.  None means no
                filtering (backward-compatible default for non-direct-supply modes).
            target_version (int): The model version this pull is acquiring. Used only for logging
                when version_holders is set.  -1 means unknown / not applicable.
            pull_epoch (int | None): Writer epoch recorded when the pull was admitted. When
                provided, a preemption notice for a newer epoch interrupts the pull; None
                disables the write-gate control path (non-direct modes).
            on_preempt_ack: Called once with the epoch when a preemption is observed. It
                must send the acknowledgement over the notification channel.
        """
        # A pool growth releases every cached prepared handle, and on a fresh engine that
        # first growth can happen after a peer wave has posted. Reap what an earlier pull
        # left in that state before deciding what the cache may release.
        self._reap_orphans()
        try:
            try:
                return self._pull(
                    epoch, participants, version_holders, target_version, pull_epoch, on_preempt_ack
                )
            except PullPreempted as preempted:
                # The writer is inside its parameter update. Every byte read before the
                # acknowledgement is safe; re-read the rest from certified holders of the
                # same version, or fail so the caller waits for the next version.
                return self._repair_after_preempt(preempted)
        finally:
            self._repairing = False
            self._reap_orphans()

    def _pull(
        self,
        epoch: int,
        participants: set[str] | None,
        version_holders: frozenset[str] | None,
        target_version: int,
        pull_epoch: int | None = None,
        on_preempt_ack=None,
    ) -> PullReport:
        start = time.time()
        self._pull_started = start
        self.source_trace = {}
        self.failed = set()
        self._pull_epoch = pull_epoch
        self._on_preempt_ack = on_preempt_ack
        self._preempt_epoch = None
        self._preempt_version = -1
        self._preempt_holders = ()
        self._completed_keys = set()
        self._pull_keys = set()
        self._key_outstanding = {}
        self._next_control_poll = 0.0
        self._repairing = False
        self._ipc_peer_bytes = 0
        self._quantize_keys = set()
        self._quantize_pending = {}
        self._accounted_phases = set()
        upstream, peer, followers = self._assign(lambda c: participants is None or c in participants)
        self._pull_keys = set(upstream) | set(peer)
        # D1 version-aware source selection: keep only trainer sources that have published
        # target_version.  An empty holder set (startup, or non-direct-supply mode) is treated
        # as "no filter" so the caller never has to check whether filtering is applicable.
        if version_holders:
            filtered_upstream: dict[str, dict[str, list]] = {}
            for key, per_source in upstream.items():
                allowed = {s: shards for s, shards in per_source.items() if s in version_holders}
                if allowed:
                    filtered_upstream[key] = allowed
                else:
                    # All sources for this key are at an older version — keep original and warn.
                    filtered_upstream[key] = per_source
                    psrl_logger.warning(
                        f"{self.client.client_name}: no source for key {key!r} holds "
                        f"target_version={target_version}; falling back to unfiltered sources "
                        f"{sorted(per_source)}."
                    )
            upstream = filtered_upstream
        # Resolve known dead sources before any byte moves, so a missing key leaves the weights untouched.
        upstream, takeover, missing = self._reroute(upstream)
        if missing:
            raise ChunkUnavailable(self.client.client_name, missing, set(self.unavailable))
        publisher, waiter = self.client.chunk_completion_channel()
        all_keys = list(set(upstream) | set(peer))
        wave_of = self._waves_of(all_keys)
        up_waves = [{k: v for k, v in upstream.items() if wave_of[k] == w} for w in range(self.waves)]
        peer_waves = [{k: v for k, v in peer.items() if wave_of[k] == w} for w in range(self.waves)]
        # Size the staging pool before posting anything. The old lazy allocation could
        # grow the pool in a later wave, which invalidated the prepared-handle cache while
        # peer reads were in flight; pre-sizing makes that a take-over-only path.
        self._prepare_pool(upstream, peer)

        versions: set[int] = set()
        stats = {
            "upstream_bytes": 0,
            "peer_bytes": 0,
            "descriptors": 0,
            "staged": 0,
            "fallback": 0,
            "takeover": takeover,
            "ipc": 0,
        }

        def take_over(keys: dict[str, dict[str, list]]) -> None:
            """Re-read from replicas the keys of `keys` whose sources failed during this pull."""
            affected = {k: v for k, v in keys.items() if self.failed.intersection(v)}
            if not affected:
                return
            rerouted, moved, lost = self._reroute(affected)
            if lost:
                raise TornPull(
                    f"{self.client.client_name}: sources {sorted(self.failed)} failed mid pull and no replica "
                    f"holds {len(lost)} key(s), first {lost[0]!r}. The weights now mix two versions."
                )
            psrl_logger.warning(
                f"{self.client.client_name}: re-reading {moved} key(s) of failed sources {sorted(self.failed)} "
                f"from DP replicas in epoch {epoch}."
            )
            stats["takeover"] += moved
            phase = self._phase(self._upstream_specs(rerouted))
            versions.update(phase.versions)
            self._run(phase)
            take_over(rerouted)

        upstream_done = start

        def peer_wave(w: int) -> tuple[list[_Handle], list]:
            """Wait for this wave's designates, start reading them, and read any late designate's keys upstream."""
            wanted = peer_waves[w]  # {key: {source: shards}}
            if not wanted:
                return [], []
            designates = {source for per_source in wanted.values() for source in per_source}
            ready = self._await(waiter, epoch, _READY + w, designates)
            # Read shards straight from `wanted`, not re-derived from this client's own
            # full shard list for the key: a key's shards can now come from more than
            # one designate, and `_spec` is cached by `(source, key)` alone, so every
            # call site touching the same pair must agree on exactly the same shards.
            specs = [
                self._spec(source, key, shards)
                for key, per_source in sorted(wanted.items())
                for source, shards in per_source.items()
                if source in ready
            ]
            handles, copies = [], []
            if specs:
                phase = self._phase(specs)
                stats["peer_bytes"] += phase.nbytes
                stats["descriptors"] += phase.descriptors
                stats["ipc"] += phase.ipc_bytes
                handles, copy = self._start(phase)
                if copy is not None:
                    copies.append(copy)
            late = designates - ready
            if late:
                psrl_logger.warning(
                    f"{self.client.client_name}: designates {sorted(late)} were not ready after "
                    f"{self.peer_timeout_s}s in epoch {epoch}, reading their keys upstream."
                )
                plan = self.client.comm_plan.rollout_pull_from_ps_plan[self.client.client_name]
                # Whole-key re-fetch when any of its designates is late (rather than
                # precise per-shard-only rerouting): simple and safe, at the cost of
                # possibly re-reading a shard whose own designate was in fact ready.
                keys = {key: plan[key] for key, per_source in wanted.items() if late & set(per_source)}
                stats["fallback"] += len(keys)
                fallback = self._phase(self._upstream_specs(keys))
                versions.update(fallback.versions)
                self._run(fallback)
            return handles, copies

        pending_peer: list[_Handle] = []
        pending_copies: list = []
        for w in range(self.waves):
            phase = self._phase(self._upstream_specs(up_waves[w])) if up_waves[w] else None
            direct, copy = self._start(phase) if phase is not None else ([], None)
            if w > 0:
                handles, copies = peer_wave(w - 1)
                pending_peer += handles
                pending_copies += copies
            self._finish(direct, copy)
            if phase is not None:
                versions.update(phase.versions)
                stats["upstream_bytes"] += phase.nbytes
                stats["descriptors"] += phase.descriptors
                stats["staged"] += phase.staged_bytes
                stats["ipc"] += phase.ipc_bytes
                # Before followers hear this wave is ready, so they never read a hole.
                take_over(up_waves[w])
            if self.client.device.type == "cuda":
                torch.cuda.synchronize(self.client.device)
            if followers:
                publisher.publish(epoch, _READY + w, sorted(followers))
        upstream_done = time.time()
        handles, copies = peer_wave(self.waves - 1)
        pending_peer += handles
        pending_copies += copies
        self._finish(pending_peer, pending_copies)
        lost_designates = self.failed.intersection(
            {source for per_source in peer.values() for source in per_source}
        )
        if lost_designates:
            plan = self.client.comm_plan.rollout_pull_from_ps_plan[self.client.client_name]
            # Whole-key re-fetch, same scoping as the late-designate path in `peer_wave`.
            keys = {key: plan[key] for key, per_source in peer.items() if lost_designates & set(per_source)}
            psrl_logger.warning(
                f"{self.client.client_name}: designates {sorted(lost_designates)} failed mid read, "
                f"reading their {len(keys)} key(s) upstream."
            )
            stats["fallback"] += len(keys)
            rerouted, moved, lost = self._reroute(keys)
            if lost:
                raise TornPull(
                    f"{self.client.client_name}: no live source holds {lost[0]!r} after its designate failed."
                )
            stats["takeover"] += moved
            fallback = self._phase(self._upstream_specs(rerouted))
            versions.update(fallback.versions)
            self._run(fallback)
        if self.client.device.type == "cuda":
            torch.cuda.synchronize(self.client.device)
        # Units are quantized once every member landed, whichever wave each came in on.
        self._flush_units()
        # _DONE was removed: it notified peers "I finished reading your buffer" but did not
        # unlock any operation on either side. Followers finish their own peer reads
        # (self._finish(pending_peer)) before waiting for _DONE, so the signal had no
        # consumer. Buffer lifetime is controlled by the external sync scheduler, not by
        # an in-pull barrier, so _DONE never protected against overwrites either.
        waiter.forget_versions_before(epoch)
        end = time.time()
        return PullReport(
            seconds=end - start,
            upstream_seconds=upstream_done - start,
            peer_seconds=end - upstream_done,
            upstream_keys=len(upstream),
            peer_keys=len(peer),
            fallback_keys=stats["fallback"],
            upstream_bytes=stats["upstream_bytes"],
            peer_bytes=stats["peer_bytes"],
            descriptors=stats["descriptors"],
            staged_bytes=stats["staged"],
            followers=len(followers),
            versions=versions,
            sources=dict(self.source_trace),
            takeover_keys=stats["takeover"],
            ipc_bytes=stats["ipc"],
            ipc_peer_bytes=self._ipc_peer_bytes,
            failed_sources=set(self.failed),
        )

    def _await(self, waiter, epoch: int, chunk: int, sources: set[str]) -> set[str]:
        """Wait up to the peer timeout and return the sources that signalled."""
        deadline = time.time() + self.peer_timeout_s
        while True:
            waiter.absorb()
            self._poll_pull_control()
            self._check_preempted()
            arrived = {s for s in sources if waiter.seen(epoch, chunk, s)}
            if arrived == sources or time.time() > deadline:
                return arrived
            time.sleep(2e-4)

    def _release_handles(self) -> None:
        """
        Drop the prepared-handle cache, keeping handles a running read still holds.

        Every cached handle embeds addresses from the geometry it was prepared for, so a
        cache invalidation (registration change, pool growth) must replace all of them.
        A pull can, however, have handles in flight when a later wave needs a bigger
        staging item. Those are moved to `_orphans` and released once they complete:
        releasing one now would free a request NIXL is still polling.
        """
        for phase in self._phases.values():
            for h in phase.direct + [h for handles in phase.rounds for h in handles]:
                if id(h) in self._in_flight:
                    self._orphans.append(h)
                else:
                    self.agent.release_xfer_handle(h.handle)
        for dlist in self._dlists:
            # Safe while a request is in flight: `make_prepped_xfer` copies the descriptor
            # values into the request, so the dlist itself is no longer referenced.
            self.agent.release_dlist_handle(dlist)
        self._phases.clear()
        self._dlists.clear()

    def _reap_orphans(self) -> None:
        """Release orphaned handles that finished, and retire pools once none can read them."""
        keep: list[_Handle] = []
        for h in self._orphans:
            if id(h) in self._in_flight:
                try:
                    state = self.agent.check_xfer_state(h.handle)
                except Exception:
                    state = "ERR"
                if state in ("DONE", "ERR"):
                    self._in_flight.pop(id(h), None)
            if id(h) in self._in_flight:
                keep.append(h)
                continue
            self.agent.release_xfer_handle(h.handle)
        self._orphans = keep
        if any(h.reads_pool for h in self._orphans):
            # A still-running staged read may write into a retired pool after this pull
            # returned. Keep the pools until it is reaped, then free them.
            return
        for _pool, reg in self._retired_pools:
            if reg is not None:
                self.agent.deregister_memory(reg)
        self._retired_pools.clear()

    def release(self) -> None:
        """Drop every prepared handle and the staging pool. Required before memory is deregistered."""
        self._release_handles()
        # The engine is going away: abort whatever is still running so its request does
        # not outlive the descriptors and pool it reads.
        for h in self._orphans:
            self.agent.release_xfer_handle(h.handle)
        self._orphans.clear()
        self._in_flight.clear()
        for _pool, reg in self._retired_pools:
            if reg is not None:
                self.agent.deregister_memory(reg)
        self._retired_pools.clear()
        self._specs.clear()
        if self._pool_reg is not None:
            self.agent.deregister_memory(self._pool_reg)
        self._pool = None
        self._pool_reg = None


def _coalesce(local, remote, lengths, regions):
    """`coalesce` that also returns the region label of every merged segment."""
    if local.shape[0] <= 1:
        return local, remote, lengths, regions
    order = np.lexsort((remote, local))
    merged = coalesce(local, remote, lengths, regions)
    heads = np.searchsorted(local[order], merged[0])
    return (*merged, regions[order][heads])


def _aligned(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


def _dense(shape) -> list[int]:
    strides, acc = [], 1
    for extent in reversed(tuple(shape)):
        strides.append(acc)
        acc *= int(extent)
    return list(reversed(strides))
