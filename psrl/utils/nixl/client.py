import bisect
import logging
import math
import os
import pickle
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import nixl._bindings as nixlBind
import numpy as np
import ray
import torch
from nixl._api import nixl_agent, nixl_agent_config
from omegaconf import DictConfig

from psrl.utils.common.patch_utils import apply_tms_patch
from psrl.utils.logger import DualOutputHandler, get_worker_info, log_tensor
from psrl.utils.nixl.chunk_notify import ChunkCompletionPublisher, ChunkCompletionWaiter
from psrl.utils.nixl.comm_plan import NIXLCommPlan
from psrl.utils.nixl.device_ipc import RegionMapper, device_uuid, export_region
from psrl.utils.nixl.handshake_notif import NotifKind, decode, encode
from psrl.utils.nixl.leave_protocol import encode_leave_ack, revoke_descriptors
from psrl.utils.nixl.meta_buffer import MetaBuffer
from psrl.utils.nixl.moe_grouping import GroupedShards, entry_is_empty, local_shards_of
from psrl.utils.nixl.network_topology import get_local_gpu_id, get_local_ip
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
    NIXLShardMetaInfo,
    NIXLTensorInfo,
)
from psrl.utils.nixl.partial_md import conn_info_only
from psrl.utils.nixl.read_geometry import contiguous_strides, run_count
from psrl.utils.nixl.strided_desc import (
    STRIDED_ROW_BUDGET,
    STRIDED_US_PER_DESCRIPTOR,
    StridedLayout,
    describe_shard,
    strided_is_cheaper_than_staging,
)
from psrl.utils.quantization import QuantizationPlan, quantizer_for_plan

if TYPE_CHECKING:
    from torch_memory_saver import torch_memory_saver
else:
    try:
        from torch_memory_saver import torch_memory_saver
    except ImportError:
        torch_memory_saver = None  # type: ignore

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


# Utility function to combine tag, shard_idx, src_client, target_client, key
def _expandable_segments_enabled() -> bool:
    """Whether the caching allocator reserves VA on demand instead of mapping it whole.

    An expandable segment's pages appear and disappear as blocks are allocated and
    freed, so a page-locking registration of the whole range would pin memory that the
    allocator may later unmap. Segment-granularity registration is disabled there.
    """
    for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        conf = os.environ.get(var, "")
        for part in conf.split(","):
            if part.strip().replace(" ", "") == "expandable_segments:True":
                return True
    return False


def _is_quantized(tensor) -> bool:
    """
    Whether `tensor` holds the FP8 bytes of a per tensor scaled FP8 parameter (Megatron `fp8_param`).

    The Megatron converter emits such a parameter as its bytes, a torch FP8 tensor carrying
    `psrl_fp8_scale = (scale_inv getter, nominal dtype)`. Readers get a publish copy rather than the bytes.
    """
    return getattr(tensor, "psrl_fp8_scale", None) is not None


class _QuantizedCell:
    """One cell of a per tensor scaled FP8 parameter: its FP8 bytes, which carry its geometry, and its scale."""

    def __init__(self, data: torch.Tensor, scale_inv, dtype: torch.dtype):
        self.data, self.scale_inv, self.dtype = data, scale_inv, dtype

    def __getattr__(self, name):
        return getattr(self.data, name)

    def dequantize(self) -> torch.Tensor:
        return (self.data.float() * self.scale_inv()).to(self.dtype)


def _plain(tensor: torch.Tensor) -> torch.Tensor:
    """`tensor` in its nominal dtype, dequantised if it holds the bytes of an FP8 parameter."""
    if not _is_quantized(tensor):
        return tensor
    scale_inv, dtype = tensor.psrl_fp8_scale
    return (tensor.float() * scale_inv()).to(dtype)


def _local_shards_named(key: str, tensor, sharding) -> list:
    """`local_shards_of`, with the key and the block FP8 tile rule in any refusal."""
    try:
        if _is_quantized(tensor):
            scale_inv, dtype = tensor.psrl_fp8_scale
            return [_QuantizedCell(cell, scale_inv, dtype) for cell in local_shards_of(tensor, sharding)]
        if isinstance(tensor, GroupedShards) and any(_is_quantized(t) for t in tensor.shards):
            raise ValueError("grouped FP8 parameter experts are not supported yet.")
        return local_shards_of(tensor, sharding)
    except ValueError as e:
        hint = ""
        if key.endswith("_scale_inv"):
            hint = " A block FP8 scale shards like its weight, so each rank must hold whole 128 x 128 tiles."
        raise ValueError(f"{key}: {e}{hint}") from e


def make_xfer_tag(
    tag: str,
    src_client: str,
    target_client: str | None,
    key: str,
    shard_idx: tuple[int, ...] | None = None,
) -> bytes:
    """
    Combine tag, shard_idx, src_client, target_client, and key into a unique bytes object as message identifier.

    Encodes as UTF-8 with a NUL separator rather than pickling. NUL cannot appear in
    client names, keys, or decimal shard indices, so the join is injective. The
    single-element shard tuple gets a dedicated f-string path because that is the
    overwhelmingly common case (one shard index per descriptor) and it avoids both
    the list build and ``map(str, ...)``.

    Measured over 140k descriptors (MoE scale): pickle 55ms, this 20ms. The gain is
    real but modest -- the dominant per-descriptor cost at that scale is NIXL's own
    descriptor handling, not tag construction.

    ``target_client`` is Optional because `wait()` accepts it that way; None and the
    empty string must not collide, hence the explicit sentinel.
    """
    tgt = "\1" if target_client is None else target_client
    if not shard_idx:
        return f"{tag}\0{src_client}\0{tgt}\0{key}".encode()
    if len(shard_idx) == 1:
        return f"{tag}\0{src_client}\0{tgt}\0{key}\0{shard_idx[0]}".encode()
    return "\0".join((tag, src_client, tgt, key, ",".join(map(str, shard_idx)))).encode()


class NIXLStorageClient:
    def __init__(
        self,
        client_name: str,
        server_name: str,
        use_gpu: bool,
        client_type: NIXLClientType,
        nixl_config: DictConfig,
        replica_idx: int = 0,
        worker_index: int = 0,
        binded_agent: nixl_agent | None = None,
        client_group_id: int = -1,  # -1 is the default client group
        logging_path: str | None = None,
        enable_prog_thread: bool = True,
        pairwise_descriptors: bool = True,
        segment_registration: bool | None = None,
    ):
        self.client_name = client_name
        self.server_name = server_name
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
        # self.device = torch.device("cuda:0" if use_gpu else "cpu")
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if use_gpu else "cpu")
        self.client_type = client_type
        # Pairwise PS transfers (`client_read`/`client_write`) address a remote peer by
        # serialized descriptors, so that path needs them. Direct supply does not: its
        # pull engine derives remote descriptors from `read_addrs` + strides + shard
        # metadata. Skipping the serialization keeps client infos small, which is what
        # the switch-time broadcast and the meta server exchange carry.
        self._pairwise_descriptors = bool(pairwise_descriptors)
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        self.max_pinned_temp_memory_slots = (
            nixl_config.max_pinned_temp_memory_slots
        )  # None means no pinned temp memory
        self.client_group_id = client_group_id
        self.enable_tms_for_temp_buffers = nixl_config.enable_tms_for_temp_buffers and use_gpu

        self.merge_contiguous_xfer = bool(nixl_config.get("merge_contiguous_xfer", True))
        self.enable_prepared_dlist = bool(nixl_config.get("enable_prepared_dlist", False))
        self.capture_telemetry = bool(nixl_config.get("capture_telemetry", False))
        self.enable_nixl_telemetry = bool(nixl_config.get("enable_nixl_telemetry", False))
        if self.enable_tms_for_temp_buffers:
            if torch_memory_saver is None:
                raise ImportError("torch_memory_saver is required when nixl_config.enable_tms_for_temp_buffers=True")
            apply_tms_patch()

            psrl_logger.info(f"NIXLStorageClient {self.client_name} enabled TMS for temporary buffers.")

        # Initialize NIXL agent
        if binded_agent is None:
            from psrl.utils.nixl.port_scanner import get_port_scanner

            worker_ip = get_worker_info()[0]
            port_scanner = get_port_scanner(worker_ip)
            self.client_port = ray.get(port_scanner.find_free_port.remote())
            self.agent = nixl_agent(
                self.client_name,
                nixl_agent_config(
                    enable_prog_thread=enable_prog_thread,
                    enable_listen_thread=True,
                    listen_port=self.client_port,
                    capture_telemetry=self.capture_telemetry,
                ),
            )
        else:
            self.agent = binded_agent

        self.local_client_info: NIXLClientInfo | None = None
        self.xfer_handles: dict[bytes, Any] = {}  # xfer_tag -> handle
        self._is_connected = False

        # Prepared descriptor-list cache for the opt-in prepared-transfer path.
        # Cache key -> `(prepared_dlist_handle, shard_indices_in_dlist_order)`.
        self._prepared_dlists: dict[tuple, tuple] = {}

        # Original tensor mapping for contiguous tensors
        self._original_tensor_mapping: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}  # (key, shard_idx) -> original_tensor
        # Temporary memory management for non-contiguous tensors
        self._temp_tensor_mapping: dict[
            tuple[str, tuple[int, ...]], torch.Tensor
        ] = {}  # (key, shard_idx) -> contiguous_tensor
        self._temp_desc_bytes_mapping: dict[tuple[str, tuple[int, ...]], bytes] = {}  # (key, shard_idx) -> desc_bytes
        self._temp_meta_mapping: dict[
            tuple[str, tuple[int, ...]], NIXLShardMetaInfo
        ] = {}  # (key, shard_idx) -> meta_info
        # If we use pinned memory, we need to record the mapping
        # from the uncontiguous tensor to the index of the pinned memory
        self._temp_pinned_idx_mapping: dict[tuple[str, tuple[int, ...]], int] = {}  # (key, shard_idx) -> pinned_idx
        self._pinned_slot_running_write_xfer: dict[
            tuple[torch.Size, torch.dtype, int], tuple
        ] = {}  # (shape, dtype, pinned_idx) -> (key, tag, op_type, target_client)
        self._pinned_slot_running_read_xfer: dict[
            tuple[torch.Size, torch.dtype, int], tuple
        ] = {}  # (shape, dtype, pinned_idx) -> (key, tag, op_type, target_client)
        self._pinned_memory: dict[tuple[torch.Size, torch.dtype], list[torch.Tensor]] | None = None
        self._read_contiguous_event_cache: dict[
            tuple[str, tuple[int, ...]], torch.cuda.streams.Event
        ] = {}  # (key, shard_idx) -> cudaEvent
        self._write_contiguous_event_cache: dict[
            tuple[str, tuple[int, ...]], torch.cuda.streams.Event
        ] = {}  # (key, shard_idx) -> cudaEvent

        # Registry for all local registrations (desc_bytes -> desc object)
        self._registered_descs: dict[bytes, Any] = {}
        # Source side contiguous copies as `(key, local_pos) -> (view, copy)`, refreshed at every publish.
        self._publish_copies: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        # Quantizer for published weights and the keys that carry auxiliary tensors.
        self._quantizer = None
        #: True on a rollout that quantizes BF16 read bytes into its own FP8 parameters.
        #: The trainer side of the same mode publishes no weight copies; it only derives
        #: the scale metadata that the unified sharding needs.
        self._reader_quantization = False
        self._quantized_pairs: dict[str, tuple[object, NIXLSharding, object, tuple[int, int]]] = {}
        #: Keys this client computes and keeps, such as a quantization unit's scale,
        #: removed from the sharding the server plans over.
        self._local_only_shardings: dict[str, NIXLSharding] = {}
        self.publish_copy_bytes = 0
        # Train dtype to rollout dtype. Shards of a cast dtype are published as copies in the rollout dtype.
        self._publish_casts: dict[torch.dtype, torch.dtype] = {}
        # Bytes that copies of merely fragmented shards may take. Readers span read the rest. None is unbounded.
        budget_mb = getattr(nixl_config, "publish_copy_budget_mb", None)
        self._publish_copy_budget = None if budget_mb is None else int(budget_mb) << 20
        # Built on first pull. Its prepared handles embed addresses, so registration changes drop it.
        self._pull_engine = None
        self._pull_staging_bytes = int(getattr(nixl_config, "pull_staging_bytes", 256 << 20))
        self._peer_timeout_s = float(getattr(nixl_config, "peer_timeout_s", 120.0))
        self._pull_waves = int(getattr(nixl_config, "pull_waves", 4))
        self._read_timeout_s = float(getattr(nixl_config, "read_timeout_s", 300.0))

        # Prepped dlists for the hot path (item A1). Holds raw addresses, so it must be
        # cleared whenever registered memory changes -- see deregister_all_local_tensors.
        # Matches the meta server's budget. These were 1200s against the server's 600s,
        # so the server gave up first and the clients were still waiting when it did --
        # which is why a slow start reports "Timeout waiting for agents" on the server and
        # nothing useful on the clients.
        self._handshake_timeout = float(getattr(nixl_config, "handshake_timeout_s", 600.0))

        # mem_type -> [(base_addr, nbytes, device_id, mem_type)] tuples for storage registration
        self._mtype_to_reg_region_lists: dict[str, list[tuple[int, int, int, str]]] = {}
        self._ipc_regions: dict[int, tuple[object, int]] = {}
        #: Why the first few readable allocations could not be exported, for the report.
        self._ipc_export_errors: list[str] = []
        self._region_mapper: RegionMapper | None = None
        self._reg_regions: set[tuple[int, int, int, str]] = set()
        #: Whether this registration pass actually handed regions to NIXL. False after a
        #: topology-based skip; a later NIXL read then registers on demand (3.7.4).
        self._rdma_registered = True

        # Register the containing CUDA allocator segment once instead of each tensor
        # allocation. `auto` disables it where the allocator does not map its VA range
        # whole (expandable segments), because registering such a range would pin pages
        # the allocator may unmap. Callers that know their allocator mode (the vLLM
        # server, which decides whether to enable expandable segments) pass it in.
        if segment_registration is None:
            segment_mode = getattr(nixl_config, "segment_registration", "auto")
            if isinstance(segment_mode, str) and segment_mode == "auto":
                segment_mode = not _expandable_segments_enabled()
            segment_registration = bool(segment_mode)
        self._segment_registration = bool(segment_registration) and use_gpu
        #: (base, size) of every caching-allocator segment, refreshed per registration.
        self._segment_extents: list[tuple[int, int]] = []
        #: Every distinct tensor allocation seen, for the registration report.
        self._storage_bases: set[int] = set()

        # Mapping from (key, shard_idx) to registered desc slice info
        self.contig_desc_slice_map: dict[tuple[str, tuple[int, ...]], tuple[int, int, int, str]] = {}
        # Shards that are non-contiguous but regularly strided (item A2): addressed
        # in place via an Nx5 descriptor instead of being staged through a copy.
        self.strided_desc_slice_map: dict[tuple[str, tuple[int, ...]], tuple[StridedLayout, str]] = {}
        self.temp_desc_slice_map: dict[tuple[str, tuple[int, ...]], tuple[int, int, int, str]] = {}

        # Meta data
        self._target_client_connected: dict[str, bool] = {}  # target_client -> connected
        #: Targets whose full descriptor metadata (rkeys) this agent has loaded.
        self._full_meta_loaded: set[str] = set()
        self._unified_sharding_dict: dict[str, NIXLSharding] | None = None  # key -> sharding
        self._unified_sharding_dict_fetched = False
        self._all_client_infos: dict[str, NIXLClientInfo] = {}  # name -> ClientInfo
        self._all_client_infos_fetched = False
        self._comm_plan: NIXLCommPlan | None = None  # Communication plan
        self._all_temp_mappings: dict[str, dict[tuple[str, int], bytes]] = {}  # client_name -> temp_desc_mapping
        #: Canonical shard-metadata instances, so equal descriptions are one object on
        #: the wire rather than N copies. See `_intern_shard_meta`.
        self._shard_meta_interner: dict[tuple, NIXLShardMetaInfo] = {}
        #: Per-chunk completion signalling, built lazily by `chunk_completion_channel`.
        self._chunk_channel: tuple[ChunkCompletionPublisher, ChunkCompletionWaiter] | None = None
        #: Notifications drained by one subsystem that belong to another. Held rather
        #: than dropped -- the stream is shared -- and re-offered by `_drain_notifs`.
        self._foreign_notifs: dict[str, list[bytes]] = {}
        #: Clients this one has been told to forget, newest last. Kept for assertions
        #: and for the log: a pull that fails on a revoked source should say so rather
        #: than look like a transport fault.
        self._revoked_clients: list[str] = []
        #: Other clients sharing this NIXL agent, set by `NIXLMultiStorageClients`.
        #: Revocation is agent-level, so one client's eviction invalidates all of their
        #: caches; without this the siblings keep believing they are connected and post
        #: transfers against a remote the agent has already forgotten.
        self._agent_siblings: list[NIXLStorageClient] = []
        #: Microseconds spent in each phase of `client_read`'s per-shard loop.
        #: Three separate attempts to reduce the measured post cost (drop the wave
        #: barrier, merge descriptors, pre-post every wave) each left it the same or
        #: worse, because each treated it as submission cost without checking. These
        #: counters say which call inside the loop the time is actually in.
        self._read_phase_us: dict[str, float] = {
            "deserialize": 0.0,
            "align": 0.0,
            "initialize": 0.0,
            "transfer": 0.0,
        }

        # logging
        if logging_path is not None:
            self.log_prefix = "NIXLStorageClient_" + self.client_name
            psrl_logger.addHandler(DualOutputHandler(logging_path, self.log_prefix))
            psrl_logger.info(f"NIXLStorageClient {self.client_name} initialized.")

    def release_temp_memory(self):
        """Release all temporary memory and deregister descriptors.

        NOTE(linsh): deregistration is done globally. Here we just clear the local mappings.
        """
        # Clear all temporary mappings
        self._temp_tensor_mapping = {}
        self._temp_desc_bytes_mapping = {}
        self._temp_meta_mapping = {}
        self._pinned_memory = None
        self._temp_pinned_idx_mapping = {}
        self.temp_desc_slice_map = {}
        self.strided_desc_slice_map = {}

    def _intern_shard_meta(self, meta: NIXLShardMetaInfo) -> NIXLShardMetaInfo:
        """Return a canonical instance for metadata equal to ``meta``.

        Pickle memoises by object identity, not by equality, so N equal-but-distinct
        metadata objects cost N copies on the wire. Returning a shared instance collapses
        them to one back-reference each -- the difference between a 2.5x and a 10.7x
        handshake reduction on a 128-expert model.

        The cache key is every field, so any genuine difference still yields a distinct
        object. What is deliberately *not* in the metadata is the address: that lives in
        the descriptor, which is why experts of one projection can share this at all.
        """
        cache_key = (
            meta.dtype,
            str(meta.device),
            tuple(meta.shape),
            meta.stride,
            meta.is_contiguous,
            meta.is_strided_expressible,
        )
        return self._shard_meta_interner.setdefault(cache_key, meta)

    def _get_local_original_tensor(self, key: str, shard_idx: tuple[int, ...]) -> torch.Tensor | None:
        """Get original tensor mapping for non-contiguous shard"""
        return self._original_tensor_mapping.get((key, shard_idx))

    def _get_local_temp_tensor(self, key: str, shard_idx: tuple[int, ...]) -> torch.Tensor | None:
        """Get temporary tensor mapping for non-contiguous shard"""
        return self._temp_tensor_mapping.get((key, shard_idx))

    def set_publish_cast(self, train_dtype: torch.dtype, rollout_dtype: torch.dtype) -> None:
        """
        Publish shards held in `train_dtype` as copies in `rollout_dtype`. Call before registration.

        Only the train dtype is cast, so tensors a model keeps in another precision, such as a
        float32 router bias in a bfloat16 model, are published as they are.
        """
        assert self.client_type is NIXLClientType.PUSH_SIDE, "Only a train client publishes casts."
        if train_dtype != rollout_dtype:
            self._publish_casts[train_dtype] = rollout_dtype

    def set_publish_quantization(self, plan: QuantizationPlan, *, reader_quantization: bool = False) -> None:
        """
        Publish weights in the plan's wire format. Call before registration.

        The scale keys come from the unified sharding, so the rollout's own parameters
        decide which weights are quantized. The trainer supplies the bytes.

        With ``reader_quantization=True`` the trainer supplies **BF16** bytes in place
        and allocates no weight publish copy. The plan is still used to derive the scale
        metadata the unified sharding needs; the rollout quantizes the BF16 reads into
        its own FP8 parameters and scales.
        """
        assert self.client_type is NIXLClientType.PUSH_SIDE, "Only a train client publishes quantized weights."
        self._reader_quantization = bool(reader_quantization)
        self._quantizer = quantizer_for_plan(plan)

    def set_reader_quantization(self, plan: QuantizationPlan) -> None:
        """Quantize BF16 source shards into this rollout's FP8 parameters.

        Call before registration. The rollout's local state already carries the FP8
        weights and `_scale_inv`; the trainer has no scale source in this mode, so the
        pull engine derives the scale from the same BF16 tile as the weight.
        """
        assert self.client_type is NIXLClientType.PULL_SIDE, "Only a rollout client quantizes on land."
        self._reader_quantization = True
        self._quantizer = quantizer_for_plan(plan)

    def _derive_aux_parameters(self, state_dict: dict, sharding_dict: dict) -> dict:
        """Add a local auxiliary tensor for each scale key the server assigned to this client."""
        self._quantized_pairs = {}
        if self._quantizer is None:
            return state_dict
        block = self._quantizer.plan.block
        assert block is not None, "A quantized plan has a block."
        derived = dict(state_dict)

        # Pool auxiliary tensors into one allocation per device. One UCX region per
        # scale makes registration grow with weight count, which dominates a wake at 235B.
        specs = []
        elements_per_device: dict[torch.device, int] = defaultdict(int)
        for scale_key, sharding in sharding_dict.items():
            base = self._quantizer.base_of_scale_key(scale_key)
            if base is None or scale_key in state_dict or base not in state_dict or sharding.is_empty:
                continue
            entry = state_dict[base]
            tensors = entry.shards if isinstance(entry, GroupedShards) else (entry,)
            scale_tile = tuple(getattr(sharding, "scale_tile", None) or block)
            assert all(b % s == 0 for b, s in zip(block, scale_tile)), (
                f"{scale_key}: the rollout's scale tile {scale_tile} does not divide the quantisation tile {block}."
            )
            for t in tensors:
                self._quantizer.validate_weight_shard(base, t)
            shapes = [self._quantizer.scale_shape_with_tile(t, scale_tile) for t in tensors]
            specs.append((scale_key, base, sharding, entry, tensors, scale_tile, shapes))
            for t, shape in zip(tensors, shapes):
                elements_per_device[t.device] += math.prod(shape)

        pools = {
            device: torch.empty(elements, dtype=self._quantizer.scale_dtype, device=device)
            for device, elements in elements_per_device.items()
        }
        offsets: dict[torch.device, int] = defaultdict(int)

        for scale_key, base, sharding, entry, tensors, scale_tile, shapes in specs:
            scales = []
            for t, shape in zip(tensors, shapes):
                pool, offset = pools[t.device], offsets[t.device]
                elements = math.prod(shape)
                scales.append(pool[offset : offset + elements].view(shape))
                offsets[t.device] = offset + elements
            scales = tuple(scales)
            self._assert_whole_tiles(scale_key, scales, sharding, isinstance(entry, GroupedShards))
            scale_entry = (
                GroupedShards(scales, entry.shard_indices, entry.shard_mesh)
                if isinstance(entry, GroupedShards)
                else scales[0]
            )
            derived[scale_key] = scale_entry
            self._quantized_pairs[base] = (entry, sharding_dict[base], scale_entry, scale_tile)
        return derived

    @staticmethod
    def _assert_whole_tiles(scale_key: str, scales, sharding: NIXLSharding, grouped: bool) -> None:
        """Every cell of the unified mesh must hold whole quantization tiles."""
        local = sharding._local_shard_mesh
        for scale in scales:
            for dim, count in sharding.shard_mesh.items():
                position = dim - 1 if grouped else dim
                if position < 0:
                    continue
                cells_here = local.get(dim, 1)
                assert scale.shape[position] % cells_here == 0, (
                    f"{scale_key}: a cell along dim {dim} holds a fraction of a quantization tile "
                    f"({scale.shape[position]} tiles over {cells_here} cells of a {count} way cut). "
                    "Some side cuts this weight finer than its scale tile."
                )

    def region_mapper(self) -> RegionMapper:
        """Regions imported from sources this process can reach without a NIC (3.7.2.1, 3.7.4)."""
        if self._region_mapper is None:
            self._region_mapper = RegionMapper(self.device)
        return self._region_mapper

    def pull_engine(self):
        """The rollout pull engine for this client, built against the current registrations."""
        if self._pull_engine is None:
            from psrl.utils.nixl.pull_engine import NIXLPullEngine

            self._pull_engine = NIXLPullEngine(
                self,
                self._pull_staging_bytes,
                self._peer_timeout_s,
                self._pull_waves,
                self._read_timeout_s,
            )
        return self._pull_engine

    def _drop_pull_engine(self) -> None:
        if self._pull_engine is not None:
            self._pull_engine.release()
            self._pull_engine = None

    def agent_name_of(self, client: str) -> str:
        """NIXL agent name that serves `client`, loading its metadata on first use."""
        self._ensure_client_info_fetched(client)
        name = self._all_client_infos[client].agent_name
        assert name, f"{client} did not publish its NIXL agent name."
        return name

    @property
    def comm_plan(self) -> NIXLCommPlan | None:
        return self._comm_plan

    @property
    def all_client_infos(self) -> dict[str, NIXLClientInfo]:
        return self._all_client_infos

    def chunk_completion_channel(self) -> tuple[ChunkCompletionPublisher, ChunkCompletionWaiter]:
        """Build the per-chunk signalling pair bound to this client's NIXL agent (item D1).

        Both ends are created together because a client is usually both: a designate for
        one set of keys and a follower for another. Handing out one without the other
        invites a caller to publish without ever absorbing, which strands its own peers.

        Foreign payloads are pushed back into the agent's own pending list rather than
        dropped. The notification stream is shared with the handshake, and this channel
        runs *after* the handshake but concurrently with other clients still finishing
        theirs -- so a drain here can legitimately pick up another subsystem's message.
        """
        if self._chunk_channel is None:
            publisher = ChunkCompletionPublisher(
                sender_name=self.client_name,
                send=self._send_chunk_notif,
            )
            waiter = ChunkCompletionWaiter(
                drain=self._drain_notifs,
                stash_foreign=self._stash_foreign_notif,
            )
            self._chunk_channel = (publisher, waiter)
        return self._chunk_channel

    def _send_chunk_notif(self, follower: str, payload: bytes) -> None:
        """Send a chunk signal, ensuring the follower's metadata is loaded first.

        A follower this client never *reads* from has never been through
        `_ensure_client_info_fetched`, so `send_notif` would fail on an unknown agent.
        Diffusion makes that the normal case: the designate notifies downhill while the
        bytes flow the other way, so the notification direction has peers the data
        direction does not.
        """
        self._ensure_client_info_fetched(follower)
        self.agent.send_notif(follower, payload)

    def _stash_foreign_notif(self, agent_name: str, payload: bytes) -> None:
        """Hold a drained payload that belongs to another subsystem."""
        self._foreign_notifs.setdefault(agent_name, []).append(payload)

    def take_foreign_notifs(self) -> dict[str, list[bytes]]:
        """Hand back notifications the chunk channel drained but does not own.

        Exposed rather than internal because the subsystem that wants them is whichever
        handshake waiter is running; returning them here lets that code merge them into
        its own loop instead of losing them to a drain it did not make.
        """
        taken, self._foreign_notifs = self._foreign_notifs, {}
        return taken

    def _drain_notifs(self) -> dict[str, list[bytes]]:
        """Every pending notification: new ones plus whatever a previous drain stashed.

        `get_new_notifs` is destructive, so a subsystem that drains and discards what it
        does not recognise destroys another's mail. Stashing was already in place for the
        chunk channel, but nothing ever re-offered the stash -- `take_foreign_notifs` had
        no callers, so a stashed payload was merely lost later rather than sooner. Making
        every drain go through here closes that: whoever polls next sees it.

        This matters for leaves specifically. A MEMBER_LEAVE can land while a client sits
        in any other waiter, and dropping it would leave that client holding descriptors
        of a member whose memory is about to be freed -- the exact failure the message
        exists to prevent.
        """
        merged = self.take_foreign_notifs()
        for agent_name, payloads in self.agent.get_new_notifs().items():
            merged.setdefault(agent_name, []).extend(payloads)
        return merged

    def drain_pull_preempt(self, client_name: str) -> dict | None:
        """Return a pending write-gate preemption for ``client_name``, if one arrived.

        The pull engine polls this from its wait loops. Everything that is not ours is
        pushed back into the foreign stash rather than dropped: `get_new_notifs` is
        destructive, and handshake or membership mail may land while a pull is running.
        A preemption already stashed by the chunk waiter is re-offered through
        `take_foreign_notifs`, so ordering between the two drains does not matter.
        """
        found: dict | None = None

        def _consume(agent_name: str, raw: bytes) -> bool:
            nonlocal found
            notif = decode(raw)
            if notif is not None and notif.kind is NotifKind.PULL_PREEMPT:
                entry = notif.entries.get(client_name)
                if entry is not None:
                    found = entry
                    return True
            return False

        for agent_name, payloads in self.agent.get_new_notifs().items():
            for raw in payloads:
                if not _consume(agent_name, raw):
                    self._stash_foreign_notif(agent_name, raw)
        for agent_name, payloads in self.take_foreign_notifs().items():
            for raw in payloads:
                if not _consume(agent_name, raw):
                    self._stash_foreign_notif(agent_name, raw)
        if found is not None:
            psrl_logger.info(
                f"{client_name}: preempted for the write at epoch {found.get('epoch')}; "
                f"{len(found.get('holders', ()))} same-node holder(s) available for repair."
            )
        return found

    def send_pull_preempt_ack(self, epoch: int) -> None:
        """Acknowledge a write-gate preemption over the notification channel.

        A notification rather than a Ray call: the writer is a `PSManager` method that is
        itself waiting for these acknowledgements, so a queued actor call would deadlock
        behind it.
        """
        self.agent.send_notif(
            self.server_name,
            encode(NotifKind.PULL_PREEMPT_ACK, {self.client_name: {"epoch": int(epoch)}}),
        )

    def handle_membership_notif(self, agent_name: str, raw: bytes) -> bool:
        """Consume ``raw`` if it is a membership message. True when it was ours.

        Offered by every drain loop so that a leave is acted on wherever it arrives,
        rather than only in whichever loop happens to be the membership one. Returning a
        bool instead of raising on a foreign payload keeps the caller's fallback (stash
        it) a normal path.
        """
        notif = decode(raw)
        if notif is None or notif.kind is not NotifKind.MEMBER_LEAVE:
            return False
        for leaver, payload in notif.entries.items():
            self._revoke_member(leaver, str(payload.get("agent", "")), ack_to=agent_name)
        return True

    def _revoke_member(self, leaver: str, leaver_agent: str, ack_to: str) -> None:
        """Forget ``leaver``'s descriptors and confirm it (E4 shrink, step three).

        The eviction is agent-level in NIXL, so this also drops the cached connection
        flag and the pull engine: both hold state derived from descriptors that are
        about to stop existing, and a pull engine rebuilt from stale handles would fail
        at post time with a NOT_FOUND that reads like a network fault.

        The ack is sent even when there was nothing to forget. "This agent holds no
        descriptors of the leaver" is the state the shrink is waiting for, and treating
        silence as the answer for that case would stall every shrink involving a member
        some peers never read from -- which, under a comm plan, is most of them.
        """
        agent_name = leaver_agent or getattr(self._all_client_infos.get(leaver), "agent_name", "")
        # Whether this peer was held is read from our own cache, not from NIXL: the
        # eviction call reports nothing (see `revoke_descriptors`). Cleared before the
        # eviction so the two cannot disagree if the eviction raises.
        #
        # Across *every* client on this agent, because `remove_remote_agent` is an
        # agent-level operation: a sibling left believing it is still connected skips
        # `_ensure_client_info_fetched` and posts against a forgotten remote.
        held = False
        for client in self._agent_siblings or (self,):
            held = client._target_client_connected.pop(leaver, False) or held
            client._drop_pull_engine()
        if agent_name:
            revoke_descriptors(self.agent, agent_name)
        self._revoked_clients.append(leaver)
        psrl_logger.info(
            f"{self.client_name}: revoked descriptors of departing member {leaver} "
            f"(agent {agent_name or 'unknown'}, was_loaded={held})"
        )
        try:
            self.agent.send_notif(ack_to, encode_leave_ack(leaver, held))
        except Exception as e:
            # The server re-sends on timeout, so a lost ack costs a retry rather than
            # correctness. Raising here would instead fail the *reader*, which has
            # already done the one thing that mattered.
            psrl_logger.warning(f"{self.client_name}: could not ack revocation of {leaver}: {e}")

    def absorb_membership_notifs(self) -> int:
        """Drain once, acting on membership messages. Returns how many were consumed.

        The entry point a client needs because every other drain is inside a waiter,
        and a leave does not arrive during a wait -- it arrives while the client is
        working. Cheap enough to call on a step boundary: one `get_new_notifs`, and
        nothing else unless a message is actually there.

        Payloads belonging to other subsystems are stashed, not dropped, so calling
        this never costs another consumer its mail.
        """
        consumed = 0
        for agent_name, payloads in self._drain_notifs().items():
            for payload in payloads:
                if self.handle_membership_notif(agent_name, payload):
                    consumed += 1
                else:
                    self._stash_foreign_notif(agent_name, payload)
        return consumed

    @property
    def revoked_clients(self) -> list[str]:
        """Members this client has been told to forget, in order."""
        return list(self._revoked_clients)

    def read_phase_us(self) -> dict[str, float]:
        """Microseconds spent per phase inside `client_read`, and reset the counters.

        Reset on read so a caller can attribute one pull round rather than a running
        total across rounds, which would hide a per-round regression behind warm-up.
        """
        taken = dict(self._read_phase_us)
        for name in self._read_phase_us:
            self._read_phase_us[name] = 0.0
        return taken

    def _get_temp_desc_bytes(self, client_name: str, key: str, shard_idx: tuple[int, ...]) -> bytes | None:
        """Get temporary descriptor for non-contiguous shard"""
        assert client_name in self._all_temp_mappings, f"Client {client_name} not found in temp mappings."
        # Currently, temp mappings are only used locally
        assert client_name == self.client_name, f"Client {client_name} is not the current client."
        return self._all_temp_mappings[client_name].get((key, shard_idx), None)

    def get_original_tensor_mapping(
        self,
    ) -> dict[tuple[str, tuple[int, ...]], torch.Tensor]:
        """Get original tensor mapping"""
        return self._original_tensor_mapping

    def _track_registered_desc(self, desc) -> bytes:
        """Cache registered desc object and return serialized bytes."""
        desc_bytes = self.agent.get_serialized_descs(desc)
        self._registered_descs[desc_bytes] = desc
        return desc_bytes

    def _allocator_segment_report(self) -> str:
        """How many CUDA allocator segments cover the registered regions, best effort.

        NIXL registration costs one MR per allocation. If many tensors come from a few
        allocator segments, registering each segment once is the zero-copy way to cut
        that count. `memory_snapshot` does not see CuMem-pool mappings (sleep mode), so
        a base outside every reported segment is called out rather than miscounted.
        """
        if self.device.type != "cuda":
            return "allocator segments: n/a (non-CUDA)"
        if self._segment_registration:
            return (
                f"allocator segments: {len(self._segment_extents)} hold "
                f"{len(self._storage_bases)} allocation(s)"
            )
        try:
            snapshot = torch.cuda.memory_snapshot()
        except Exception as exc:  # pragma: no cover - depends on the driver
            return f"allocator segments: unavailable ({exc!r})"
        segments = [(b["address"], b["total_size"]) for b in snapshot if b.get("total_size")]
        bases = [
            region[0]
            for reg_list in self._mtype_to_reg_region_lists.values()
            for region in reg_list
        ]

        def segment_of(addr: int) -> int | None:
            for base, size in segments:
                if base <= addr < base + size:
                    return base
            return None

        found = {segment_of(base) for base in bases}
        outside = None in found
        found.discard(None)
        matched = len(bases) - (1 if outside else 0)
        detail = (
            " plus at least one region outside the caching allocator (CuMem pool)"
            if outside
            else ""
        )
        return f"allocator segments: {len(found)} cover {matched} region(s){detail}"

    def _refresh_segment_extents(self) -> None:
        """Snapshot the caching allocator's segments for this registration round.

        A segment is a cudaMalloc allocation the allocator carves tensors out of. Its
        address range is fully mapped (unlike an expandable segment, whose pages are
        mapped on demand), so one NIXL region can cover every tensor in it.
        """
        self._segment_extents = []
        if not self._segment_registration:
            return
        try:
            snapshot = torch.cuda.memory_snapshot()
        except Exception as exc:  # pragma: no cover - depends on the driver
            psrl_logger.warning(
                f"{self.client_name}: memory_snapshot unavailable ({exc!r}); "
                "registering each allocation."
            )
            self._segment_registration = False
            return
        extents: list[tuple[int, int]] = []
        for segment in snapshot:
            base = int(segment.get("address", 0) or 0)
            size = int(segment.get("total_size", 0) or 0)
            if base <= 0 or size <= 0:
                continue
            if str(segment.get("segment_type", "")).startswith("expandable"):
                # The VA range is reserved but pages are mapped on demand; registering
                # it whole would pin pages the allocator later unmaps.
                psrl_logger.warning(
                    f"{self.client_name}: expandable segments detected; registering each allocation."
                )
                self._segment_registration = False
                return
            extents.append((base, size))
        extents.sort()
        self._segment_extents = extents

    def _registration_extent(self, base: int, nbytes: int) -> tuple[int, int]:
        """The extent to register for an allocation: its containing segment, or itself."""
        extents = self._segment_extents
        if not extents:
            return base, nbytes
        index = bisect.bisect_right(extents, (base, float("inf"))) - 1
        if index >= 0:
            segment_base, segment_size = extents[index]
            if segment_base <= base and base + nbytes <= segment_base + segment_size:
                return segment_base, segment_size
        return base, nbytes

    def _record_region_registration(self, tensor: torch.Tensor) -> tuple[int, int, int, str]:
        """Record the registration covering a tensor view.

        Returns the **allocation** key ``(base_addr, nbytes, device_id, mem_type)``:
        `read_bases` and IPC handles are per allocation and must keep pointing at the
        tensor's own storage. The extent actually handed to NIXL may be larger -- the
        whole allocator segment -- and is deduplicated separately.
        """
        storage = tensor.untyped_storage()
        base_addr = storage.data_ptr()
        nbytes = storage.nbytes()
        device_id = tensor.get_device() if tensor.is_cuda else 0
        mem_type = "cuda" if tensor.is_cuda else "cpu"
        self._storage_bases.add(base_addr)

        reg_base, reg_nbytes = self._registration_extent(base_addr, nbytes)
        region = (reg_base, reg_nbytes, device_id, mem_type)
        if region not in self._reg_regions:
            self._reg_regions.add(region)
            self._mtype_to_reg_region_lists.setdefault(mem_type, []).append((reg_base, reg_nbytes, device_id, ""))
        # One IPC handle per allocation, so a reader on this device or another device of
        # this node copies rather than reading through NIXL. The size travels with the
        # handle because the reader locates a span before it imports it. Export is best
        # effort: an allocation the allocator cannot share falls back to NIXL.
        if mem_type == "cuda" and base_addr not in self._ipc_regions:
            handle, error = export_region(tensor, explain=True)
            if handle is not None:
                self._ipc_regions[base_addr] = (handle, nbytes)
            elif len(self._ipc_export_errors) < 3:
                self._ipc_export_errors.append(error)
        return base_addr, nbytes, device_id, mem_type

    def _deserialize_to_xfer_descs(self, desc_bytes: bytes):
        """Deserialize desc bytes and ensure xfer descriptors are returned."""
        descs = self.agent.deserialize_descs(desc_bytes)
        if isinstance(descs, nixlBind.nixlRegDList):
            return descs.trim()
        return descs

    def _align_desc_counts(self, local_desc, remote_desc):
        """Split whichever side has one descriptor so both sides have the same count.

        NIXL pairs transfer descriptors element-wise, so a strided shard -- which
        expands to one descriptor per row (item A2) -- cannot be paired against a single
        contiguous descriptor on the other side. The transfer is rejected with
        NIXL_ERR_INVALID_PARAM rather than being split automatically.

        The contiguous side is cut into the same number of equal, consecutive pieces.
        That is exactly the layout the strided rows land in, because the rows carry
        equal `row_bytes` and the receiver's buffer is dense: row i of the source
        corresponds to bytes [i*row_bytes, (i+1)*row_bytes) of the destination.

        Returns the pair unchanged when the counts already agree, so the contiguous
        path keeps its single-descriptor fast case.
        """
        n_local, n_remote = local_desc.descCount(), remote_desc.descCount()
        if n_local == n_remote:
            return local_desc, remote_desc

        if n_local == 1:
            single, count, other = local_desc, n_remote, "local"
        elif n_remote == 1:
            single, count, other = remote_desc, n_local, "remote"
        else:
            raise RuntimeError(
                f"{self.client_name}: cannot pair {n_local} local descriptors with "
                f"{n_remote} remote ones; neither side is a single contiguous descriptor"
            )

        addr, length, dev_id = single[0]
        if length % count != 0:
            raise RuntimeError(
                f"{self.client_name}: contiguous {other} descriptor of {length} bytes does "
                f"not divide into {count} equal rows, so it cannot be paired with a "
                "strided descriptor"
            )
        row = length // count
        rows = [(addr + i * row, row, dev_id) for i in range(count)]
        split = nixlBind.nixlXferDList(single.getType(), rows)
        return (split, remote_desc) if n_local == 1 else (local_desc, split)

    def _deregister_all_descs(self):
        """Deregister all cached descriptors in one pass."""
        if not self._registered_descs:
            return
        descs = list(self._registered_descs.values())
        self._registered_descs = {}
        for desc in descs:
            self.agent.deregister_memory(desc)

    def _register_memory(self, mem_type: str, reg_list: list[tuple[int, int, int, str]]) -> nixlBind.nixlRegDList:
        """Register memory with NIXL."""
        dlist = np.zeros((len(reg_list), 3), dtype=np.uint64)
        for i, (base_addr, nbytes, device_id, _) in enumerate(reg_list):
            dlist[i, 0] = base_addr
            dlist[i, 1] = nbytes
            dlist[i, 2] = device_id
        descs = nixlBind.nixlRegDList(self.agent.nixl_mems[mem_type], dlist)
        return self.agent.register_memory(descs)

    def ensure_rdma_registered(self) -> None:
        """
        Pay the registration a topology-based skip deferred, if a NIXL read now needs it.

        The pull engine calls this before preparing a transfer. In pure colocate it never
        runs; it exists so a source that unexpectedly cannot be exported (a CuMem pool, an
        allocation from outside the caching allocator) degrades into the slow path instead
        of failing the pull. The region list is the same one the skipped pass built.
        """
        if self._rdma_registered:
            return
        _t0 = time.time()
        for mem_type, reg_list in self._mtype_to_reg_region_lists.items():
            if not reg_list:
                continue
            reg_descs = self._register_memory(mem_type, reg_list)
            self._track_registered_desc(reg_descs)
        self._rdma_registered = True
        psrl_logger.warning(
            f"{self.client_name}: deferred RDMA registration ran because a read fell back to NIXL "
            f"({time.time() - _t0:.3f}s). A source could not be served by IPC/peer copy."
        )

    # NOTE(lhy): High level registration amortizes the expensive low level NIXL API
    # while preserving complete tensor coverage.
    def _serialize_strided_descs(self, tensor_infos, shard_pos_cache) -> int:
        """Serialize strided shards into `desc_bytes_list`, returning how many were written.

        A strided shard is written as the Nx3 rows it expands to, not as one Nx5 row:
        `nixlXferDList` accepts only Nx3 in every NIXL version, and Nx5 is reachable
        solely through `prep_xfer_dlist` (design 3.6.3). The win is still real -- no
        staging buffer is allocated and no copy-back happens -- and the prepped hot path
        (item A1) collapses these rows into a single strided descriptor when it preps the
        dlist.
        """
        written = 0
        for (key, shard_idx), (layout, mem_type) in self.strided_desc_slice_map.items():
            rows = layout.to_nx3_rows()
            xfer_descs = self.agent.get_xfer_descs(rows, mem_type=mem_type)
            assert xfer_descs.descCount() == len(rows), (
                f"{self.client_name}: get_xfer_descs returned {xfer_descs.descCount()} descs "
                f"for {len(rows)} strided rows of {key} shard {shard_idx}."
            )
            desc_bytes = self.agent.get_serialized_descs(xfer_descs)
            local_pos = shard_pos_cache[key][shard_idx]
            tensor_infos[key].desc_bytes_list[local_pos] = desc_bytes
            written += 1
        return written

    def _ensure_all_tensor_registered_high_level(self):
        """Check if all tensors are covered by our registered regions (no NIXL query)."""
        # Build (base, size) list from current registered regions
        registered: list[tuple[int, int]] = []
        for reg_list in self._mtype_to_reg_region_lists.values():
            for base_addr, nbytes, _device_id, _ in reg_list:
                registered.append((base_addr, nbytes))

        def _is_covered(ptr: int, size: int) -> bool:
            for base, sz in registered:
                if ptr >= base and ptr + size <= base + sz:
                    return True
            return False

        for (key, shard_idx), slice_info in self.contig_desc_slice_map.items():
            slice_addr, slice_len, _device_id, _mem_type = slice_info
            if not _is_covered(slice_addr, slice_len):
                raise RuntimeError(
                    f"{self.client_name}: tensor {key} shard {shard_idx} is a contiguous tensor but not registered"
                )

        for (key, shard_idx), slice_info in self.temp_desc_slice_map.items():
            slice_addr, slice_len, _device_id, _mem_type = slice_info
            if not _is_covered(slice_addr, slice_len):
                raise RuntimeError(
                    f"{self.client_name}: tensor {key} shard {shard_idx} is a temp tensor but not registered"
                )

        # A strided shard is registered through its own storage, so the whole span --
        # gaps included -- has to fall inside one region. Checking only the first row
        # would pass for a shard whose last row runs past the end.
        for (key, shard_idx), (layout, _mem_type) in self.strided_desc_slice_map.items():
            if not _is_covered(layout.address, layout.span_bytes):
                raise RuntimeError(
                    f"{self.client_name}: tensor {key} shard {shard_idx} is a strided tensor "
                    "but its span is not covered by a registered region"
                )

    def _serialize_descriptors_if_pairwise(self, tensor_infos, shard_pos_cache) -> None:
        """Serialize pairwise descriptors only on runs that can use them."""
        if not self._pairwise_descriptors:
            psrl_logger.info(
                f"{self.client_name}: direct-supply pull path; descriptors are derived by readers"
            )
            return
        self._serialize_pairwise_descriptors(tensor_infos, shard_pos_cache)

    def _serialize_pairwise_descriptors(self, tensor_infos, shard_pos_cache) -> None:
        """Serialize the descriptors a pairwise PS transfer addresses its peer by.

        Direct supply derives remote descriptors on the reader from `read_addrs` +
        strides + shard metadata, so it never needs these bytes on the wire; only
        `client_read` / `client_write` do. Serializing a strided shard expands it to
        one descriptor per row, which is what makes an update-info payload grow with
        the parameter count and its shape.
        """
        # --- contig_desc_slice_map ---
        contig_by_memtype: dict[str, list[tuple]] = defaultdict(list)
        for (key, shard_idx), slice_info in self.contig_desc_slice_map.items():
            slice_addr, slice_len, device_id, mem_type = slice_info
            contig_by_memtype[mem_type].append((key, shard_idx, slice_addr, slice_len, device_id))

        for mem_type, entries in contig_by_memtype.items():
            batch_tuples = [(addr, length, dev) for (_, _, addr, length, dev) in entries]
            xfer_descs = self.agent.get_xfer_descs(batch_tuples, mem_type=mem_type)
            assert xfer_descs.descCount() == len(entries), (
                f"{self.client_name}: get_xfer_descs returned {xfer_descs.descCount()} descs "
                f"for {len(entries)} contig inputs (mem_type={mem_type})."
            )
            desc_type = xfer_descs.getType()
            for i, (key, shard_idx, _, _, _) in enumerate(entries):
                single = nixlBind.nixlXferDList(desc_type, [xfer_descs[i]])
                desc_bytes = self.agent.get_serialized_descs(single)
                local_pos = shard_pos_cache[key][shard_idx]
                tensor_infos[key].desc_bytes_list[local_pos] = desc_bytes

        # --- temp_desc_slice_map ---
        temp_by_memtype: dict[str, list[tuple]] = defaultdict(list)
        for (key, shard_idx), slice_info in self.temp_desc_slice_map.items():
            slice_addr, slice_len, device_id, mem_type = slice_info
            temp_by_memtype[mem_type].append((key, shard_idx, slice_addr, slice_len, device_id))

        for mem_type, entries in temp_by_memtype.items():
            batch_tuples = [(addr, length, dev) for (_, _, addr, length, dev) in entries]
            xfer_descs = self.agent.get_xfer_descs(batch_tuples, mem_type=mem_type)
            assert xfer_descs.descCount() == len(entries), (
                f"{self.client_name}: get_xfer_descs returned {xfer_descs.descCount()} descs "
                f"for {len(entries)} temp inputs (mem_type={mem_type})."
            )
            desc_type = xfer_descs.getType()
            for i, (key, shard_idx, _, _, _) in enumerate(entries):
                single = nixlBind.nixlXferDList(desc_type, [xfer_descs[i]])
                desc_bytes = self.agent.get_serialized_descs(single)
                self._temp_desc_bytes_mapping[(key, shard_idx)] = desc_bytes
                local_pos = shard_pos_cache[key][shard_idx]
                tensor_infos[key].temp_desc_bytes_list[local_pos] = desc_bytes

        n_strided = self._serialize_strided_descs(tensor_infos, shard_pos_cache)
        if n_strided:
            # Report the descriptor count, not just the shard count. A strided
            # shard expands to one descriptor per row, and `postXferReq` charges
            # ~2.1-4us *per descriptor* -- so this line is the difference between
            # a transfer that costs microseconds and one that costs seconds.
            #
            # The old message said only "no staging copy", which reads as a win
            # and hid the cost entirely: locating it took several cluster runs
            # when one log line would have done. Avoiding the staging copy is
            # genuinely cheaper only while the row count stays small.
            rows = [layout.row_count for layout, _mem_type in self.strided_desc_slice_map.values()]
            total_descs = sum(rows)
            worst = max(rows, default=0)
            est_ms = total_descs * STRIDED_US_PER_DESCRIPTOR / 1e3
            psrl_logger.info(
                f"{self.client_name}: {n_strided} strided shard(s) registered in place "
                f"(no staging copy): {total_descs} descriptor(s), worst shard "
                f"{worst} rows, est. submit cost {est_ms:.1f}ms at "
                f"{STRIDED_US_PER_DESCRIPTOR}us/descriptor"
            )
            if worst > STRIDED_ROW_BUDGET:
                # A shard is sharded on its inner (contiguous) dimension, which is
                # what makes it strided at all. Row-parallel layers (`o_proj`,
                # `down_proj`) are sharded on the input dimension and so hit this;
                # column-parallel layers never do. Matching the two sides on that
                # dimension removes the re-shard and the expansion with it.
                psrl_logger.warning(
                    f"{self.client_name}: worst strided shard expands to {worst} "
                    f"descriptors (budget {STRIDED_ROW_BUDGET}), costing about "
                    f"{worst * STRIDED_US_PER_DESCRIPTOR / 1e3:.1f}ms to submit for "
                    f"bytes that transfer in microseconds. This comes from "
                    f"re-sharding a row-parallel weight on its input dimension; "
                    f"aligning the input-dimension shard count between train and "
                    f"rollout removes it."
                )


    def register_local_tensors(
        self,
        state_dict: dict[str, torch.Tensor],
        sharding_dict: dict[str, NIXLSharding] | None = None,
        binded_meta_tensor_mapping: (dict[tuple[str, tuple[int, ...]], torch.Tensor] | None) = None,
        meta_only: bool = False,
    ):
        """
        Register local tensors with NIXL. Build key->desc mapping.
        Currently only support all tensors are within
        binded_meta_tensor_mapping or not within binded_meta_tensor_mapping.

        Args:
            state_dict: {key: torch.Tensor}
            sharding_dict: {key: NIXLSharding}
            binded_meta_tensor_mapping: {(key, shard_idx): torch.Tensor}
            meta_only: whether to skip registering real tensors
        """

        self._drop_pull_engine()
        # Re-registration changes the addresses/rkeys behind local descriptors, so
        # cached prepared descriptor lists are invalid.
        self._release_prepared_dlists(side="local")
        if not meta_only:
            # The allocator's segment map must be current before any region is recorded:
            # the map is what maps an allocation to the extent registered for it.
            self._refresh_segment_extents()
            self._ipc_export_errors = []
        if (
            self.enable_tms_for_temp_buffers
            and self.local_client_info is not None
            and self.local_client_info.is_registered
        ):
            # Re-register local tensors (skip if all readers are same-node; no remote NIC needs these pages).
            assert self._mtype_to_reg_region_lists is not None, "No registered regions found."
            if self._should_register_rdma():
                for mem_type, reg_list in self._mtype_to_reg_region_lists.items():
                    if not reg_list:
                        continue
                    reg_descs = self._register_memory(mem_type, reg_list)
                    self._track_registered_desc(reg_descs)
                self._rdma_registered = True
            else:
                self._rdma_registered = False
                psrl_logger.info(f"{self.client_name}: skipping RDMA registration (no edge needs NIXL)")

            # Rebuild descriptors by memory type to reduce get_xfer_descs round-trips to O(mem_types).
            # Precompute shard positions for O(1) lookup instead of repeated O(S) list.index calls.
            reregister_shard_pos_cache: dict[str, dict] = {
                key: {s: i for i, s in enumerate(ti.sharding.shard_indices)}
                for key, ti in self.local_client_info.tensor_infos.items()
            }

            # --- contig_desc_slice_map ---
            contig_by_memtype: dict[str, list[tuple]] = defaultdict(list)
            for (key, shard_idx), slice_info in self.contig_desc_slice_map.items():
                slice_addr, slice_len, device_id, mem_type = slice_info
                contig_by_memtype[mem_type].append((key, shard_idx, slice_addr, slice_len, device_id))

            for mem_type, entries in contig_by_memtype.items():
                batch_tuples = [(addr, length, dev) for (_, _, addr, length, dev) in entries]
                xfer_descs = self.agent.get_xfer_descs(batch_tuples, mem_type=mem_type)
                assert xfer_descs.descCount() == len(entries), (
                    f"{self.client_name}: re-register get_xfer_descs returned {xfer_descs.descCount()} descs "
                    f"for {len(entries)} contig inputs (mem_type={mem_type})."
                )
                desc_type = xfer_descs.getType()
                for i, (key, shard_idx, _, _, _) in enumerate(entries):
                    single = nixlBind.nixlXferDList(desc_type, [xfer_descs[i]])
                    desc_bytes = self.agent.get_serialized_descs(single)
                    local_pos = reregister_shard_pos_cache[key][shard_idx]
                    self.local_client_info.tensor_infos[key].desc_bytes_list[local_pos] = desc_bytes

            # --- temp_desc_slice_map ---
            temp_by_memtype: dict[str, list[tuple]] = defaultdict(list)
            for (key, shard_idx), slice_info in self.temp_desc_slice_map.items():
                slice_addr, slice_len, device_id, mem_type = slice_info
                temp_by_memtype[mem_type].append((key, shard_idx, slice_addr, slice_len, device_id))

            for mem_type, entries in temp_by_memtype.items():
                batch_tuples = [(addr, length, dev) for (_, _, addr, length, dev) in entries]
                xfer_descs = self.agent.get_xfer_descs(batch_tuples, mem_type=mem_type)
                assert xfer_descs.descCount() == len(entries), (
                    f"{self.client_name}: re-register get_xfer_descs returned {xfer_descs.descCount()} descs "
                    f"for {len(entries)} temp inputs (mem_type={mem_type})."
                )
                desc_type = xfer_descs.getType()
                for i, (key, shard_idx, _, _, _) in enumerate(entries):
                    single = nixlBind.nixlXferDList(desc_type, [xfer_descs[i]])
                    desc_bytes = self.agent.get_serialized_descs(single)
                    self._temp_desc_bytes_mapping[(key, shard_idx)] = desc_bytes
                    local_pos = reregister_shard_pos_cache[key][shard_idx]
                    self.local_client_info.tensor_infos[key].temp_desc_bytes_list[local_pos] = desc_bytes
            self._serialize_strided_descs(self.local_client_info.tensor_infos, reregister_shard_pos_cache)
            self._all_temp_mappings[self.client_name] = self._temp_desc_bytes_mapping
            return

        tms_ctx = torch_memory_saver.region(tag="nixl") if self.enable_tms_for_temp_buffers else nullcontext()
        with tms_ctx:
            # Group noncontiguous tensors by shape and dtype, then assign each group
            # round robin across its bounded contiguous slot pool.
            if sharding_dict is None:
                sharding_dict = {}
            if self._local_only_shardings:
                sharding_dict = {**sharding_dict, **self._local_only_shardings}
            _uncontiguous_tensor_mapping: dict[
                tuple[Any, Any], list[tuple[str, tuple[int, ...], torch.Tensor]]
            ] = {}  # (shape, dtype) -> [key, shard_idx, uncontiguous_tensor]
            if self.max_pinned_temp_memory_slots is not None:
                # Scan the state_dict and find all the tensors that are not contiguous
                for key, tensor in state_dict.items():
                    assert key in sharding_dict, f"Key {key} not found in sharding_dict."
                    if entry_is_empty(tensor):
                        continue
                    sharding = sharding_dict[key]
                    shard_indices = sharding.shard_indices
                    local_sharded_tensors = _local_shards_named(key, tensor, sharding)
                    for local_pos, local_sharded_tensor in enumerate(local_sharded_tensors):
                        if not local_sharded_tensor.is_contiguous():
                            if (
                                local_sharded_tensor.shape,
                                local_sharded_tensor.dtype,
                            ) not in _uncontiguous_tensor_mapping:
                                _uncontiguous_tensor_mapping[
                                    (local_sharded_tensor.shape, local_sharded_tensor.dtype)
                                ] = []
                            _uncontiguous_tensor_mapping[
                                (local_sharded_tensor.shape, local_sharded_tensor.dtype)
                            ].append((key, shard_indices[local_pos], local_sharded_tensor))
                # Find all types of uncontiguous tensor and allocate pinned memory for them
                if _uncontiguous_tensor_mapping:
                    self._pinned_memory = {}
                    for (
                        shape,
                        dtype,
                    ), uncontiguous_tensor_list in _uncontiguous_tensor_mapping.items():
                        self._pinned_memory[(shape, dtype)] = []
                        for i, (key, shard_idx, uncontiguous_tensor) in enumerate(uncontiguous_tensor_list):
                            self._temp_pinned_idx_mapping[(key, shard_idx)] = i % self.max_pinned_temp_memory_slots
                        if meta_only:
                            continue

                        # Optimization: allocate a big pinned memory tensor and chunk it
                        # to reduce the number of registration calls
                        pinned_memory = torch.empty(
                            (self.max_pinned_temp_memory_slots, *shape),
                            dtype=dtype,
                            device=self.device,
                            requires_grad=False,
                        )
                        self._record_region_registration(pinned_memory)
                        memory_slots = torch.chunk(pinned_memory, self.max_pinned_temp_memory_slots, dim=0)
                        assert len(memory_slots) == self.max_pinned_temp_memory_slots, (
                            f"Expected {self.max_pinned_temp_memory_slots} memory slots, but got {len(memory_slots)}."
                        )
                        for slot in memory_slots:
                            memory_slot = slot.squeeze(0)
                            self._pinned_memory[(shape, dtype)].append(memory_slot)

            # Preallocate one buffer per dtype so meta tensor views share one registration.
            meta_buffer: MetaBuffer | None = None
            if binded_meta_tensor_mapping is None:
                entries: list[tuple[tuple[Any, Any], tuple[int, ...], torch.dtype]] = []
                for key, tensor in state_dict.items():
                    assert key in sharding_dict, f"Key {key} not found in sharding_dict."
                    if (
                        not isinstance(tensor, GroupedShards)
                        and tensor.device != torch.device("meta")
                        and tensor.untyped_storage().nbytes() == 0
                    ):
                        tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
                    sharding = sharding_dict[key]
                    shard_indices = sharding.shard_indices
                    local_sharded_tensors = _local_shards_named(key, tensor, sharding)
                    for local_pos, local_sharded_tensor in enumerate(local_sharded_tensors):
                        if local_sharded_tensor.device == torch.device("meta"):
                            entries.append(
                                (
                                    (key, shard_indices[local_pos]),
                                    local_sharded_tensor.shape,
                                    local_sharded_tensor.dtype,
                                )
                            )
                if entries:
                    meta_buffer = MetaBuffer(self.device)
                    meta_buffer.allocate(entries)
                    for buf in meta_buffer.buffers():
                        self._record_region_registration(buf)

            tensor_infos = {}
            # Train shards change only at publish, so copies refreshed there stay exact. Others are read in place.
            is_upstream_source = self.client_type is NIXLClientType.PUSH_SIDE
            if is_upstream_source and not meta_only:
                state_dict = self._derive_aux_parameters(state_dict, sharding_dict)
            publish_requests: list[tuple[str, int, torch.Tensor, torch.dtype]] = []
            self._publish_copies = {}
            self.publish_copy_bytes = 0
            for key, tensor in state_dict.items():
                assert key in sharding_dict, f"Key {key} not found in sharding_dict."
                if (
                    not isinstance(tensor, GroupedShards)
                    and tensor.device != torch.device("meta")
                    and tensor.untyped_storage().nbytes() == 0
                ):
                    tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
                sharding = sharding_dict[key]
                shard_indices = sharding.shard_indices
                # assert sharding.is_contiguous_sharding(), "Only contiguous sharding is supported for now."
                # Split registration
                desc_bytes_list = []
                temp_desc_bytes_list = []
                shard_meta_info_list = []
                read_addrs: list[int | None] = []
                read_strides: list[tuple[int, ...] | None] = []
                read_bases: list[int | None] = []
                tensor_mem_type, tensor_device_id = "cuda" if self.device.type == "cuda" else "cpu", 0

                # A grouped MoE key (item C1) has no single tensor to slice: its experts
                # are separate storages, so the shards arrive pre-split.
                local_sharded_tensors = _local_shards_named(key, tensor, sharding)

                for local_pos, tbd_local_sharded_tensor in enumerate(local_sharded_tensors):
                    # Store the original tensor mapping
                    # If the tensor is on meta device, allocate on-the-fly or binded from the external tensor
                    if tbd_local_sharded_tensor.device == torch.device("meta"):
                        # Case 1: use pre-allocated slice (from early scan) or binded from the external tensor
                        if binded_meta_tensor_mapping is None and meta_buffer is not None:
                            local_sharded_tensor = meta_buffer.get_tensor((key, shard_indices[local_pos]))
                        elif binded_meta_tensor_mapping is not None:
                            assert (key, shard_indices[local_pos]) in binded_meta_tensor_mapping, (
                                f"Key {key} shard {shard_indices[local_pos]} not found in binded_meta_tensor_mapping."
                            )
                            local_sharded_tensor = binded_meta_tensor_mapping[(key, shard_indices[local_pos])]
                        else:
                            local_sharded_tensor = tbd_local_sharded_tensor
                    else:
                        local_sharded_tensor = tbd_local_sharded_tensor
                    if not meta_only:
                        # Local sharded tensor should be on a real device now
                        assert local_sharded_tensor.device == self.device, (
                            f"Local sharded tensor {key} shard {shard_indices[local_pos]} is not "
                            f"on device {self.device}, but on {local_sharded_tensor.device}, "
                            f"torch current device is {torch.cuda.current_device()}, "
                            f"CUDA_VISIBLE_DEVICES is {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}"
                        )
                    self._original_tensor_mapping[(key, shard_indices[local_pos])] = local_sharded_tensor
                    # A TE FP8 parameter has no address of its own. Its FP8 bytes stand in for it below,
                    # and readers only ever read its publish copy, dequantised in the nominal dtype.
                    quantized_source = (
                        local_sharded_tensor if isinstance(local_sharded_tensor, _QuantizedCell) else None
                    )
                    nominal_dtype = local_sharded_tensor.dtype
                    if quantized_source is not None:
                        local_sharded_tensor = quantized_source.data

                    # Create meta info for shards
                    is_contiguous = local_sharded_tensor.is_contiguous()
                    # Computed once here and carried in the metadata: the peer needs to
                    # know this shard is sendable without staging, and it cannot work
                    # that out from shape and stride alone without repeating the
                    # analysis `describe_shard` already did.
                    strided_layout = None if is_contiguous else describe_shard(local_sharded_tensor)
                    # Only shards that the train side writes into a PS may be staged. Everything else is read from.
                    must_be_readable = self.client_type is not NIXLClientType.PS_FOR_PUSH
                    stage_despite_strided = strided_layout is not None and not (
                        strided_is_cheaper_than_staging(strided_layout, must_be_readable=must_be_readable)
                    )
                    if stage_despite_strided:
                        strided_layout = None
                    # Reuse one metadata object across shards that describe themselves
                    # identically. Pickle memoises by identity, so for a grouped MoE key
                    # -- whose 128 experts differ only in address, which lives in the
                    # descriptor rather than here -- this is what turns the handshake
                    # saving from 2.5x into 10.7x (measured on Qwen3-30B-A3B). Equal-but-
                    # distinct objects would serialise 128 times over.
                    if key in self._quantized_pairs:
                        # Reader-side quantization keeps the live BF16 bytes readable.
                        # The rollout quantizes them; the trainer allocates no FP8 copy.
                        publish_dtype = nominal_dtype if self._reader_quantization else self._quantizer.weight_dtype
                    elif (
                        self._quantizer is not None
                        and self._quantizer.base_of_scale_key(key) in self._quantized_pairs
                    ):
                        publish_dtype = nominal_dtype  # auxiliary tensors keep their declared dtype
                    else:
                        publish_dtype = self._publish_casts.get(nominal_dtype, nominal_dtype)
                    meta_info = self._intern_shard_meta(
                        NIXLShardMetaInfo(
                            dtype=publish_dtype,
                            device=local_sharded_tensor.device,
                            shape=local_sharded_tensor.shape,
                            stride=local_sharded_tensor.stride(),
                            is_contiguous=is_contiguous,
                            is_strided_expressible=strided_layout is not None,
                        )
                    )
                    shard_meta_info_list.append(meta_info)

                    if meta_only:
                        read_addrs.append(None)
                        read_strides.append(None)
                        read_bases.append(None)
                    else:
                        storage_key = self._record_region_registration(local_sharded_tensor)
                        tensor_device_id, tensor_mem_type = storage_key[2], storage_key[3]
                        fragmented = (
                            run_count(tuple(local_sharded_tensor.shape), local_sharded_tensor.stride())
                            > STRIDED_ROW_BUDGET
                        )
                        # A block scale is tiny but has many short rows (one per tile row), so a reader's
                        # column cut of it would cost a descriptor per row. Its copy costs 1/16384 of the weight.
                        is_scale = (
                            self._quantizer is not None
                            and self._quantizer.base_of_scale_key(key) in self._quantized_pairs
                        )
                        if is_upstream_source and not self._reader_quantization and (
                            fragmented or is_scale or publish_dtype != nominal_dtype or quantized_source is not None
                        ):
                            # Readers get a contiguous copy in their dtype, so any cut of it is one descriptor.
                            source = quantized_source if quantized_source is not None else local_sharded_tensor
                            publish_requests.append((key, local_pos, source, publish_dtype))
                            # Filled by `_allocate_publish_copies`, or by `_within_copy_budget` past the budget.
                            read_addrs.append(None)
                            read_strides.append(contiguous_strides(tuple(local_sharded_tensor.shape)))
                            read_bases.append(None)
                        else:
                            read_addrs.append(local_sharded_tensor.data_ptr())
                            read_strides.append(tuple(local_sharded_tensor.stride()))
                            read_bases.append(storage_key[0])

                    # Check if the shard is contiguous
                    if not meta_only:
                        if is_contiguous:
                            # Contiguous shard: batch register
                            storage_key = self._record_region_registration(local_sharded_tensor)
                            slice_addr = local_sharded_tensor.data_ptr()
                            slice_len = local_sharded_tensor.numel() * local_sharded_tensor.element_size()
                            assert (
                                slice_addr >= storage_key[0]
                                and slice_addr + slice_len <= storage_key[0] + storage_key[1]
                            ), (
                                f"{self.client_name}: key {key} shard {shard_indices[local_pos]} is contiguous, "
                                f"but contiguous slice address {slice_addr} is not within the registered "
                                f"region from {storage_key[0]} through {storage_key[0] + storage_key[1]}."
                            )
                            device_id = local_sharded_tensor.get_device() if local_sharded_tensor.is_cuda else 0
                            mem_type = "cuda" if local_sharded_tensor.is_cuda else "cpu"
                            self.contig_desc_slice_map[(key, shard_indices[local_pos])] = (
                                slice_addr,
                                slice_len,
                                device_id,
                                mem_type,
                            )
                        elif strided_layout is not None:
                            # Non-contiguous but regularly strided (implementation item A2).
                            # An Nx5 descriptor addresses the shard where it already lives, so
                            # there is no staging copy to allocate, no slot to wait for, and
                            # nothing to copy back after a pull. The shard's own storage is
                            # registered instead.
                            storage_key = self._record_region_registration(local_sharded_tensor)
                            assert (
                                strided_layout.address >= storage_key[0]
                                and strided_layout.address + strided_layout.span_bytes
                                <= storage_key[0] + storage_key[1]
                            ), (
                                f"{self.client_name}: key {key} shard {shard_indices[local_pos]} is strided, "
                                f"but its span {strided_layout.address} through "
                                f"{strided_layout.address + strided_layout.span_bytes} escapes the registered "
                                f"region {storage_key[0]} through {storage_key[0] + storage_key[1]}."
                            )
                            self.strided_desc_slice_map[(key, shard_indices[local_pos])] = (
                                strided_layout,
                                storage_key[3],
                            )
                        else:
                            if self.max_pinned_temp_memory_slots is None:
                                # Non-contiguous shard: create temporary contiguous memory
                                # Create a new contiguous tensor with the same shape and dtype
                                contiguous_tensor = torch.empty_like(
                                    local_sharded_tensor, device=self.device, memory_format=torch.contiguous_format
                                )
                                storage_key = self._record_region_registration(contiguous_tensor)
                            else:
                                assert self._pinned_memory is not None, (
                                    f"Pinned memory is not initialized for {key} shard {shard_indices[local_pos]}, "
                                    f"max_pinned_temp_memory_slots is {self.max_pinned_temp_memory_slots} and "
                                    f"_uncontiguous_tensor_mapping is {_uncontiguous_tensor_mapping}."
                                )
                                assert (
                                    local_sharded_tensor.shape,
                                    local_sharded_tensor.dtype,
                                ) in self._pinned_memory, (
                                    f"Pinned memory does not have slot for {key} shard {shard_indices[local_pos]}."
                                )
                                assert (
                                    key,
                                    shard_indices[local_pos],
                                ) in self._temp_pinned_idx_mapping, (
                                    f"Pinned memory does not have slot for {key} shard {shard_indices[local_pos]}."
                                )
                                # Non-contiguous shard: map to pinned memory
                                pinned_slot = self._pinned_memory[
                                    (local_sharded_tensor.shape, local_sharded_tensor.dtype)
                                ][self._temp_pinned_idx_mapping[(key, shard_indices[local_pos])]]
                                assert pinned_slot.dtype == local_sharded_tensor.dtype, (
                                    f"Pinned slot {self._temp_pinned_idx_mapping[(key, shard_indices[local_pos])]} "
                                    f"has {pinned_slot.dtype} dtype, but the {key} shard {shard_indices[local_pos]} "
                                    f"requires {local_sharded_tensor.dtype} dtype."
                                )
                                assert pinned_slot.shape == local_sharded_tensor.shape, (
                                    f"Pinned slot {self._temp_pinned_idx_mapping[(key, shard_indices[local_pos])]} "
                                    f"has {pinned_slot.shape} shape, but the {key} shard {shard_indices[local_pos]} "
                                    f"requires {local_sharded_tensor.shape} shape."
                                )
                                contiguous_tensor = pinned_slot
                                storage_key = self._record_region_registration(contiguous_tensor)

                            # Build the contiguous meta info
                            contiguous_meta_info = NIXLShardMetaInfo(
                                dtype=contiguous_tensor.dtype,
                                device=contiguous_tensor.device,
                                shape=contiguous_tensor.shape,
                                stride=contiguous_tensor.stride(),
                                is_contiguous=True,
                            )
                            # Store temporary mappings
                            temp_slice_addr = contiguous_tensor.data_ptr()
                            temp_slice_len = contiguous_tensor.numel() * contiguous_tensor.element_size()
                            assert (
                                temp_slice_addr >= storage_key[0]
                                and temp_slice_addr + temp_slice_len <= storage_key[0] + storage_key[1]
                            ), (
                                f"{self.client_name}: key {key} shard {shard_indices[local_pos]} is non-contiguous, "
                                f"but temporary slice address {temp_slice_addr} is not within the registered "
                                f"region from {storage_key[0]} through {storage_key[0] + storage_key[1]}."
                            )
                            self.temp_desc_slice_map[(key, shard_indices[local_pos])] = (
                                temp_slice_addr,
                                temp_slice_len,
                                storage_key[2],
                                storage_key[3],
                            )
                            self._temp_tensor_mapping[(key, shard_indices[local_pos])] = contiguous_tensor
                            self._temp_meta_mapping[(key, shard_indices[local_pos])] = contiguous_meta_info

                    # Placeholder for desc bytes, will be filled after global registration
                    desc_bytes_list.append(None)
                    temp_desc_bytes_list.append(None)

                # Create the tensor descriptor info
                tensor_infos[key] = NIXLTensorInfo(
                    desc_bytes_list=desc_bytes_list,
                    temp_desc_bytes_list=temp_desc_bytes_list,
                    sharding=sharding,
                    shard_meta_infos=shard_meta_info_list,
                    read_addrs=read_addrs,
                    read_strides=read_strides,
                    read_bases=read_bases,
                    mem_type=tensor_mem_type,
                    device_id=tensor_device_id,
                )

            if publish_requests:
                publish_requests = self._within_copy_budget(publish_requests, tensor_infos)
                self._allocate_publish_copies(publish_requests, tensor_infos)

            if binded_meta_tensor_mapping is not None:
                assert self.temp_desc_slice_map == {}, (
                    "Expected temp_desc_slice_map to be empty when binded_meta_tensor_mapping is provided, "
                    f"but got {self.temp_desc_slice_map}."
                )

            # Batch register all tensors and cache the reg list once.
            # Skip when every edge this client participates in is same-node and copyable:
            # no NIC needs these pages, and the registration cost (O(#shards)) is pure
            # overhead at MoE scale. A later NIXL read registers on demand.
            register_rdma = not meta_only and self._should_register_rdma()
            self._rdma_registered = register_rdma or meta_only or not self._mtype_to_reg_region_lists
            if not register_rdma and not meta_only:
                psrl_logger.info(f"{self.client_name}: skipping RDMA registration (no edge needs NIXL)")
            if register_rdma and self._mtype_to_reg_region_lists:
                _t_registration = time.time()
                n_regions = 0
                for mem_type, reg_list in self._mtype_to_reg_region_lists.items():
                    if not reg_list:
                        continue
                    # NOTE(lhy): Register each allocation separately because merging
                    # adjacent virtual regions is invalid for RDMA and UCX.
                    reg_descs = self._register_memory(mem_type, reg_list)
                    self._track_registered_desc(reg_descs)
                    n_regions += len(reg_list)
                self._rdma_registered = True
                # Registration is a page-lock + MR step whose cost tracks the number of
                # mapped bytes, not the shard count. It sits on the switch path whenever
                # a cast/FP8 publish copy was freed on sleep; separating it from the
                # descriptor work is what tells an operator which half to attack.
                psrl_logger.info(
                    f"{self.client_name}: [timing] registered {n_regions} region(s) for "
                    f"{len(self._storage_bases)} allocation(s) with NIXL in "
                    f"{time.time() - _t_registration:.3f}s "
                    f"(segment_registration={self._segment_registration}). "
                    f"{self._allocator_segment_report()}"
                )

                # Precompute {shard_idx: local_pos} once per key: O(1) lookup vs O(S) list.index()
                shard_pos_cache: dict[str, dict] = {
                    key: {s: i for i, s in enumerate(ti.sharding.shard_indices)} for key, ti in tensor_infos.items()
                }

                # Group by mem_type to batch get_xfer_descs calls: O(mem_types) round-trips instead of O(N_shards).
                psrl_logger.info(
                    f"{self.client_name}: [timing] register_local_tensors batch xfer_descs: "
                    f"contig_desc_slice_map={len(self.contig_desc_slice_map)}, "
                    f"temp_desc_slice_map={len(self.temp_desc_slice_map)}"
                )
                _t_xfer_descs = time.time()

                self._serialize_descriptors_if_pairwise(tensor_infos, shard_pos_cache)
                psrl_logger.info(
                    f"{self.client_name}: [timing] register_local_tensors batch xfer_descs done in "
                    f"{time.time() - _t_xfer_descs:.3f}s"
                )

            # Which readable addresses have no IPC handle. Readers decide per shard they
            # actually pull; `missing_exports_block_readers` decides the source side
            # (design 3.7.4).
            ipc_bases = {
                base for info in tensor_infos.values() for base in info.read_bases if base is not None
            }
            ipc_missing = frozenset(base for base in ipc_bases if base not in self._ipc_regions)
            ipc_complete = bool(ipc_bases) and not ipc_missing
            if ipc_missing:
                psrl_logger.info(
                    f"{self.client_name}: {len(ipc_missing)}/{len(ipc_bases)} readable allocation(s) have no IPC "
                    f"handle; same-node readers of those shards register with NIXL. "
                    f"First error: {self._ipc_export_errors[0] if self._ipc_export_errors else 'unknown'}"
                )
            if not meta_only and not self._rdma_registered and ipc_missing:
                from psrl.utils.nixl.registration_policy import missing_exports_block_readers  # noqa: PLC0415

                # Registering because *any* allocation failed to export would undo the
                # reader-side skip for clients nobody reads -- every vLLM instance's
                # allocator refuses IPC export. Only a reader that will actually pull a
                # missing shard needs this side registered (3.7.4).
                if missing_exports_block_readers(self._comm_plan, self.client_name, tensor_infos, ipc_missing):
                    psrl_logger.info(
                        f"{self.client_name}: a reader pulls a shard with no IPC handle; "
                        "registering RDMA for the same-node fallback"
                    )
                    self.ensure_rdma_registered()

            # Create the client info
            self.local_client_info = NIXLClientInfo(
                name=self.client_name,
                node_ip=get_local_ip(),
                node_gpu_id=get_local_gpu_id(),
                type=self.client_type,
                tensor_infos=tensor_infos,
                meta=self.agent.get_agent_metadata(),
                # conn_info_meta is the same identity without region descriptors (~401 B,
                # constant). Non-readers load only this; readers load this then `meta`.
                # See design §9c-c (E4) and `_ensure_client_info_fetched`.
                conn_info_meta=conn_info_only(self.agent),
                client_group_id=self.client_group_id,
                is_registered=not meta_only,
                agent_name=self.agent.name,
                device_uuid=device_uuid(self.device),
                ipc_regions=dict(self._ipc_regions),
                ipc_complete=ipc_complete,
                ipc_missing=ipc_missing,
            )
            psrl_logger.debug(
                f"Local client info is built, "
                f"temp pinned idx mapping is: {self._temp_pinned_idx_mapping}, "
                f"temp meta mapping is: {self._temp_meta_mapping}"
            )

            if binded_meta_tensor_mapping is None:
                self._ensure_all_tensor_registered_high_level()

            # Refresh local temp descriptors so later reads use current buffers.
            self._all_temp_mappings[self.client_name] = self._temp_desc_bytes_mapping

            psrl_logger.info(
                f"{self.client_name} all local tensors are registered. "
                f"node_ip={self.local_client_info.node_ip} gpu_id={self.local_client_info.node_gpu_id}"
            )

    def _within_copy_budget(
        self,
        requests: list[tuple[str, int, torch.Tensor, torch.dtype]],
        tensor_infos: dict[str, NIXLTensorInfo],
    ) -> list[tuple[str, int, torch.Tensor, torch.dtype]]:
        """
        Keep cast copies, then fragmented copies in order of runs saved per byte until the budget is spent.

        A shard left out stays readable in place, where a reader pays either one descriptor per run
        or one span read of the shard's extent, whichever its cost model prefers.
        """
        if self._publish_copy_budget is None:
            return requests
        casts = [r for r in requests if r[3] != r[2].dtype]
        fragmented = sorted(
            (r for r in requests if r[3] == r[2].dtype),
            key=lambda r: -run_count(tuple(r[2].shape), r[2].stride()) / max(r[2].numel(), 1),
        )
        kept, spent = list(casts), 0
        for request in fragmented:
            nbytes = request[2].numel() * request[2].element_size()
            if spent + nbytes > self._publish_copy_budget:
                key, local_pos, view, _ = request
                info = tensor_infos[key]
                info.read_addrs[local_pos] = view.data_ptr()
                info.read_strides[local_pos] = tuple(view.stride())
                info.read_bases[local_pos] = view.untyped_storage().data_ptr()
                continue
            spent += nbytes
            kept.append(request)
        return kept

    def _allocate_publish_copies(
        self,
        requests: list[tuple[str, int, torch.Tensor, torch.dtype]],
        tensor_infos: dict[str, NIXLTensorInfo],
    ) -> None:
        """
        Give each requested source shard a contiguous copy, in its publish dtype, inside one flat buffer per dtype.

        One buffer per dtype keeps registration to a handful of regions however many shards
        need copies. The copies hold valid bytes only after `refresh_publish_copies`.
        """
        by_dtype: dict[torch.dtype, list[tuple[str, int, torch.Tensor]]] = defaultdict(list)
        for key, local_pos, view, dtype in requests:
            by_dtype[dtype].append((key, local_pos, view))
        _t_copies = time.time()
        for dtype, entries in by_dtype.items():
            flat = torch.empty(sum(view.numel() for _, _, view in entries), dtype=dtype, device=self.device)
            self._record_region_registration(flat)
            offset = 0
            for key, local_pos, view in entries:
                copy = flat[offset : offset + view.numel()].view(view.shape)
                offset += view.numel()
                tensor_infos[key].read_addrs[local_pos] = copy.data_ptr()
                tensor_infos[key].read_bases[local_pos] = flat.data_ptr()
                self._publish_copies[(key, local_pos)] = (view, copy)
            self.publish_copy_bytes += flat.numel() * flat.element_size()
        casts = sum(1 for _, _, view, dtype in requests if view.dtype != dtype)
        psrl_logger.info(
            f"{self.client_name}: {len(requests)} shard(s) get publish copies ({casts} cast, "
            f"{len(requests) - casts} fragmented), {self.publish_copy_bytes / 2**20:.1f} MiB resident."
        )
        self.refresh_publish_copies()
        psrl_logger.info(
            f"{self.client_name}: publish copies allocated and refreshed in {time.time() - _t_copies:.3f}s"
        )

    @torch.no_grad()
    def refresh_publish_copies(self) -> None:
        """Copy the live shards into their publish copies. Must run while no reader is in flight."""
        if not self._publish_copies:
            return
        # Quantise first, since scale tensors written here may themselves have plain copies.
        for key, (entry, sharding, scale_entry, scale_tile) in self._quantized_pairs.items():
            self._quantize_into_copies(key, entry, sharding, scale_entry, scale_tile)
        plain = [pair for (key, _), pair in self._publish_copies.items() if key not in self._quantized_pairs]
        quantized = [(view, copy) for view, copy in plain if isinstance(view, _QuantizedCell)]
        plain = [(view, copy) for view, copy in plain if not isinstance(view, _QuantizedCell)]
        if plain:
            views, copies = zip(*plain)
            torch._foreach_copy_(list(copies), list(views))
        for view, copy in quantized:
            copy.copy_(view.dequantize())
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _quantize_into_copies(self, key: str, entry, sharding: NIXLSharding, scale_entry, scale_tile) -> None:
        """
        Quantise one key a stored tensor at a time, writing its scales and the FP8 copies of its cells.

        Quantisation runs on the publish tile. A rollout whose scale grid is finer gets each scale
        repeated over the finer cells it covers, as vLLM's loader upsamples a checkpoint's scales.
        """
        block = self._quantizer.plan.block
        assert block is not None
        repeat = tuple(b // s for b, s in zip(block, scale_tile))
        grouped = isinstance(entry, GroupedShards)
        tensors = entry.shards if grouped else (entry,)
        scales = scale_entry.shards if grouped else (scale_entry,)
        plain = tuple(_plain(t) for t in tensors)
        fused = self._quantizer.can_quantize_in_place(plain[0])
        if repeat == (1, 1) and fused:
            # Cells hold whole tiles (`_assert_whole_tiles`), so quantizing cell by cell gives the
            # same bytes and scales as quantizing the stored tensor. Each cell's bytes go straight
            # into its publish copy, which is one pass over the train shard and no temporaries.
            source = GroupedShards(plain, entry.shard_indices, entry.shard_mesh) if grouped else plain[0]
            cells = local_shards_of(source, sharding)
            scale_cells = local_shards_of(scale_entry, sharding)
            for local_pos, (cell, scale_cell) in enumerate(zip(cells, scale_cells)):
                self._quantizer.quantize_into(
                    cell,
                    self._publish_copies[(key, local_pos)][1],
                    scale_cell,
                )
            return
        # A refined scale grid may cut a quantization tile across cells, so quantize each stored
        # tensor whole and then cut. The transient is one stored tensor.
        quantized = []
        for tensor, scale in zip(plain, scales):
            q, s = self._quantizer.quantize(tensor)
            s = self._quantizer.refine_scale(s, repeat)
            scale.copy_(s)
            quantized.append(q)
        q_entry = GroupedShards(tuple(quantized), entry.shard_indices, entry.shard_mesh) if grouped else quantized[0]
        for local_pos, cell in enumerate(local_shards_of(q_entry, sharding)):
            self._publish_copies[(key, local_pos)][1].copy_(cell)

    def deregister_local_tensors(self):
        """Deregister all local tensors"""
        assert self.local_client_info is not None, "Local client info not registered."
        # Release transfer/prepared handles before their backing memory is gone.
        self._release_all_xfer_handles("deregister_local_tensors")
        self._release_prepared_dlists(side="local")
        self._drop_pull_engine()
        # An imported mapping outliving its exporter's allocation is the same hazard
        # `registration_epoch` guards against remotely, so drop them with the registrations.
        self._region_mapper = None
        self._ipc_regions = {}
        self._ipc_export_errors = []
        self._rdma_registered = False
        # Deregister all registered regions.
        self._deregister_all_descs()
        # Publish copies and derived FP8 scales are GPU tensors this client owns, rebuilt
        # by the next `register_local_tensors`. Holding them across a colocate switch
        # kept the publish copies -- 20 GB per rank at 235B FP8 -- out of the rollout's KV
        # cache while nothing could read them, since their registrations are gone.
        self._publish_copies = {}
        self.publish_copy_bytes = 0
        self._quantized_pairs = {}
        if not self.enable_tms_for_temp_buffers:
            self.release_temp_memory()

            # Clear all mappings
            self.local_client_info = None
            self._original_tensor_mapping = {}
            self._pinned_slot_running_read_xfer = {}
            self._pinned_slot_running_write_xfer = {}
            self._read_contiguous_event_cache = {}
            self._write_contiguous_event_cache = {}
            self._reg_regions = set()
            self._mtype_to_reg_region_lists = {}
            self._segment_extents = []
            self._storage_bases = set()
            self.contig_desc_slice_map = {}
            self.strided_desc_slice_map = {}

        psrl_logger.debug(f"{self.client_name} deregistered all local tensors.")

    def connect_to_server(self, timeout: float | None = None):
        """
        Connect to the storage/meta server.
        """
        assert not self._is_connected, "Already connected to server"
        self.agent.fetch_remote_metadata(self.server_name, self.server_ip, self.server_port)
        self.agent.send_local_metadata(self.server_ip, self.server_port)
        # `None` selects the configured budget, keeping client and server aligned.
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        ready = False
        while not ready:
            ready = self.agent.check_remote_metadata(self.server_name)
            if time.time() - start > budget:
                raise TimeoutError("Timeout waiting for server metadata to be fetched and connected.")
            time.sleep(0.1)
        self._is_connected = True

    def publish_version(self, version: int) -> None:
        """Stamp every locally registered shard with the version now readable (§3.4).

        O(1) per shard and touches no tensor data -- which is the point. The alternative,
        hashing content, measured ~61 s for an 8B model against a 1.4 s pull, so it could
        only run sampled and off by default; a stamp costs 0.06 ms for the same model and
        does not move when the model or its dtype changes.

        **The stamp only reaches readers through a metadata exchange.** Readers cache
        remote `NIXLClientInfo` at the handshake (`_ensure_client_info_fetched`) and
        refresh it only via `wait_for_update_infos`, which production runs on colocate
        switches rather than every step. So this is useful at points where metadata is
        about to be distributed -- bootstrap, and a rejoin -- and a *per-step* version
        check would need a metadata broadcast per step, which costs more than the fault it
        catches. Measured: stamping after the handshake leaves readers seeing `-1`, which
        is how this was found.
        """
        start = time.perf_counter()
        self.refresh_publish_copies()
        self.last_publish_seconds = time.perf_counter() - start
        for tensor_info in self.local_client_info.tensor_infos.values():
            for meta in tensor_info.shard_meta_infos:
                if meta is not None:
                    meta.version = version

    def expect_version(self, version: int | None) -> None:
        """Declare which version subsequent reads must find at their sources.

        For callers that know `v` up front -- notably peer assignment, which §3.4 names
        explicitly ("index 指派 peer 时，校验 `version(peer) == v`"). `None` disables the
        strict check, which is the default: a rollout learns its version *from* the pull,
        so demanding it beforehand would cost a round trip the design's C2 went out of its
        way to remove.
        """
        self._expected_version = version

    def begin_version_observation(self) -> None:
        """Start recording which versions the next reads find at their sources.

        The reader's half of the invariant when `v` is not known in advance. Collecting is
        a set insert per shard, so it stays on; the caller then asks two questions that
        together cover the same ground as a pre-known expectation:

        * did every shard come from the same version (a torn read, or one stale source,
          shows up as more than one);
        * does that version equal the one the manager reports (a *uniformly* stale pull
          agrees with itself and would otherwise pass).
        """
        self._observed_versions: set[int] = set()

    def observed_versions(self) -> set[int]:
        """Versions seen since `begin_version_observation`, or empty if not recording."""
        return set(getattr(self, "_observed_versions", set()))

    def plan_sharding(self, sharding_dict: dict[str, NIXLSharding]) -> dict[str, NIXLSharding]:
        """
        The sharding the transfer plan is built over.

        A quantization unit's scale is computed by the rollout from its members' bytes and
        never travels, so it stays out of the plan and is merged back for local registration.
        Every other key is planned as usual.
        """
        local_only = {
            key: sharding for key, sharding in sharding_dict.items() if getattr(sharding, "local_only", False)
        }
        self._local_only_shardings.update(local_only)
        return {key: sharding for key, sharding in sharding_dict.items() if key not in local_only}

    def send_local_sharding(self, sharding_dict: dict[str, NIXLSharding]):
        """
        Send local sharding to the server.
        """
        assert self._is_connected, "Not connected to server"
        announced = self.plan_sharding(sharding_dict)
        self.agent.send_notif(self.server_name, encode(NotifKind.SHARDING, {self.client_name: announced}))

    def send_local_info(self):
        """
        Send local client info to the server.
        For storage_server mode, notify the server that the client is ready.
        For meta_server mode, send the local client info to the server.
        """
        assert self._is_connected, "Not connected to server"
        if self.local_client_info is None:
            raise RuntimeError("Local client info not registered.")
        serialized = self.local_client_info.serialize()
        psrl_logger.info(f"{self.client_name}: client info {len(serialized) / 1e6:.3f} MB.")
        self.agent.send_notif(
            self.server_name,
            encode(NotifKind.CLIENT_INFO, {self.client_name: serialized}),
        )

    def send_local_temp_mapping(self):
        """Send local temporary mappings to the server"""
        assert self._is_connected, "Not connected to server"
        self.agent.send_notif(
            self.server_name,
            encode(NotifKind.TEMP_MAPPING, {self.client_name: self._temp_desc_bytes_mapping}),
        )

    def wait_for_server_sharding(self, timeout: float | None = None):
        """
        Wait for the server sharding to be fetched.
        """
        assert self._is_connected, "Not connected to server"
        if self._unified_sharding_dict_fetched:
            return
        # `None` selects the configured budget, keeping client and server aligned.
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs and notifs[self.server_name]:
                client_sharding_dicts = pickle.loads(notifs[self.server_name][0])
                assert isinstance(client_sharding_dicts, dict) and len(client_sharding_dicts) == 1, (
                    f"Expected a dict with one client sharding dict, but got {client_sharding_dicts}"
                )
                self._unified_sharding_dict = next(iter(client_sharding_dicts.values()))
                break
            if time.time() - start > budget:
                raise TimeoutError("Timeout waiting for server sharding notification.")
            time.sleep(0.1)
        self._unified_sharding_dict_fetched = True
        return self._unified_sharding_dict

    def wait_for_server_info(self, timeout: float | None = None):
        """
        Wait for the server info to be fetched.
        For storage_server mode, wait for the storage server info to be fetched.
        For meta_server mode, wait for all client infos (stored in the server) to be fetched.
        """
        assert self._is_connected, "Not connected to server"
        # Wait for all client infos (stored in the server) to be fetched
        if self._all_client_infos_fetched:
            return
        # `None` selects the configured budget, keeping client and server aligned.
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs and notifs[self.server_name]:
                notification_bytes = notifs[self.server_name][0]
                _t0 = time.time()
                notification_data = pickle.loads(notification_bytes)
                _t1 = time.time()
                # Process client infos
                if isinstance(notification_data, dict) and "client_infos" in notification_data:
                    # New format: includes communication plan
                    all_client_infos = notification_data["client_infos"]
                    for client_name, info_bytes in all_client_infos.items():
                        info = NIXLClientInfo.deserialize(info_bytes)
                        self._all_client_infos[client_name] = info
                        self._release_prepared_dlists(side="remote", target_client=client_name)
                    # Process communication plan
                    if notification_data.get("comm_plan"):
                        self._comm_plan = NIXLCommPlan.deserialize(notification_data["comm_plan"])
                    else:
                        self._comm_plan = None
                else:
                    # Old format: only client infos
                    all_client_infos = notification_data
                    for client_name, info_bytes in all_client_infos.items():
                        info = NIXLClientInfo.deserialize(info_bytes)
                        self._all_client_infos[client_name] = info
                        self._release_prepared_dlists(side="remote", target_client=client_name)
                        self._comm_plan = None
                _t2 = time.time()
                psrl_logger.info(
                    f"{self.client_name}: [timing] wait_for_server_info deserialization: "
                    f"pickle.loads={_t1 - _t0:.3f}s, "
                    f"NIXLClientInfo.deserialize={_t2 - _t1:.3f}s, "
                    f"n_infos={len(all_client_infos)}, "
                    f"payload_bytes={len(notification_bytes)}"
                )
                break
            if time.time() - start > budget:
                raise TimeoutError("Timeout waiting for meta server client infos.")
            time.sleep(0.1)
        self._all_client_infos_fetched = True

    def wait_for_server_temp_mappings(self, timeout: float | None = None):
        """Wait for the server temporary mappings to be fetched."""
        assert self._is_connected, "Not connected to server"
        # `None` selects the configured budget, keeping client and server aligned.
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs and notifs[self.server_name]:
                self._all_temp_mappings = pickle.loads(notifs[self.server_name][0])
                break
            if time.time() - start > budget:
                raise TimeoutError("Timeout waiting for temp mappings.")
            time.sleep(0.1)

    def send_local_info_to(self, dst_agent_names: list[str]):
        """Send local client info to specified destination agents.

        Args:
            dst_agent_names: List of destination agent names.
        """
        assert self._is_connected, "Not connected to server"
        serialized = self.local_client_info.serialize()
        payload_dict = {
            self.client_name: {
                "info": serialized,
                "temp_mapping": self._temp_desc_bytes_mapping,
            }
        }
        payload = encode(NotifKind.UPDATE_INFO, payload_dict)
        psrl_logger.info(
            f"{self.client_name}: client info {len(serialized) / 1e6:.3f} MB "
            f"(payload {len(payload) / 1e6:.3f} MB) to {len(dst_agent_names)} agent(s)."
        )
        for dst_agent_name in dst_agent_names:
            self.agent.send_notif(dst_agent_name, payload)

    def wait_for_update_infos(self, expected_agents: int, timeout: float | None = None):
        """Wait for updated client infos from other clients.

        Args:
            expected_agents: Number of expected client infos to be updated.
            timeout: Timeout in seconds.
        """
        assert self._is_connected, "Not connected to server"
        psrl_logger.info(f"{self.client_name}: Waiting for {expected_agents} updated client infos...")
        # `None` selects the configured budget, keeping client and server aligned.
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self._drain_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    # Route on the tag rather than on "a dict arrived while we were
                    # waiting". Peer-diffusion chunk notices share this stream, and the
                    # old shape check accepted them as client infos.
                    parsed = decode(notif)
                    if parsed is None or parsed.kind is not NotifKind.UPDATE_INFO:
                        # Not ours. A membership message is acted on here rather than
                        # deferred -- this loop runs concurrently with live traffic, so
                        # it is the most likely place for a leave to land -- and
                        # anything else is kept for its own consumer. Dropping was the
                        # old behaviour and silently destroyed both.
                        if not self.handle_membership_notif(agent_name, notif):
                            self._stash_foreign_notif(agent_name, notif)
                        continue
                    for client_name, info_and_temp_mapping in parsed.entries.items():
                        self._all_client_infos[client_name] = NIXLClientInfo.deserialize(info_and_temp_mapping["info"])
                        # A re-registered source has new allocations, so any mapping of the
                        # old ones points at freed memory. Drop the catalog with the info it
                        # came from; the next read imports the new handles (design 3.7.4).
                        if self._region_mapper is not None:
                            self._region_mapper.forget(client_name)
                        # Prepared descriptor lists embed the old remote addresses too.
                        self._release_prepared_dlists(side="remote", target_client=client_name)
                        # Re-registered memory carries new keys, so the remote metadata must be loaded again.
                        if self._target_client_connected.pop(client_name, False):
                            self.agent.remove_remote_agent(self._all_client_infos[client_name].agent_name)
                        self._full_meta_loaded.discard(client_name)
                        self._drop_pull_engine()
                        self._all_temp_mappings[client_name] = info_and_temp_mapping["temp_mapping"]
                    already_recved_agents.add(agent_name)
                    psrl_logger.info(f"Already received {len(already_recved_agents)} agents: {already_recved_agents}")
            if time.time() - start > budget:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.1)

    def broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        """Broadcast updated client infos to specified destination agents.

        Args:
            dst_agent_names: List of destination agent names.
            update_client_names: List of client names whose infos are to be broadcasted.
        """
        payload_dict = {}
        for client_name in update_client_names:
            client_info = self._all_client_infos[client_name]
            client_temp_mapping = self._all_temp_mappings[client_name]
            payload_dict[client_name] = {"info": client_info.serialize(), "temp_mapping": client_temp_mapping}
        payload = encode(NotifKind.UPDATE_INFO, payload_dict)
        for dst_agent_name in dst_agent_names:
            self.agent.send_notif(dst_agent_name, payload)

    def _should_register_rdma(self) -> bool:
        """Whether this client must register its tensors with NIXL.

        Registration exists to let a NIC reach these pages. It is needed when this client
        is on either end of a cross-node transfer, and it is pure overhead when every edge
        it participates in is same-node and can be served by IPC/peer copy (design 3.7.4).
        Registering anyway costs one `nixlRegDList` call plus `get_xfer_descs` /
        `get_serialized_descs` per region, which at MoE scale is the O(#shards) cost of
        §3.5.5; in colocate it was the largest item of the validate switch.

        The reader direction is the expensive one to get wrong in the other way. NIXL
        requires the local side of a transfer to be registered, so a reader may skip only
        when *every shard it pulls* comes from a same-node source allocation that has an
        IPC handle. One unexportable buffer a reader never touches does not force it to
        register. If a skip turns out wrong at transfer time, `ensure_rdma_registered`
        pays the deferred registration rather than failing.

        PS involvement always registers: the PS transfer path needs serialized
        descriptors, and its clients are a different data plane.

        Falls back to True when the plan or the client infos are not yet populated:
        without them we cannot prove every edge is copyable, so the safe answer is to
        register.
        """
        if self._comm_plan is None or not self._all_client_infos:
            return True
        plan = self._comm_plan
        # A client that moves bytes through NIXL's PS plans needs its own descriptors
        # serialized and its memory registered even when no one reads it remotely. The
        # train client is exactly that case in PS mode -- its shards may be read only by
        # a same-node PS worker, but it still writes them to the PS every step and pulls
        # them back after a wake. Skipping registration left `desc_bytes_list` unset and
        # made the first trainer pull fall into the temp-buffer path with no mapping
        # (KeyError on embed_tokens).
        if plan.push_to_ps_plan.get(self.client_name) or plan.train_pull_from_ps_plan.get(self.client_name):
            return True
        from psrl.utils.nixl.registration_policy import (  # noqa: PLC0415
            all_sources_copyable,
            needs_rdma_registration,
        )

        if needs_rdma_registration(plan, self._all_client_infos, self.client_name):
            return True
        if self.client_type is not NIXLClientType.PULL_SIDE:
            return False
        if getattr(self, "device", None) is None or self.device.type != "cuda":
            return True  # IPC and peer copy are CUDA primitives; anything else needs NIXL
        copyable, reason = all_sources_copyable(plan, self._all_client_infos, self.client_name)
        if copyable and not self._sources_peer_reachable(plan):
            copyable, reason = False, "a same-node GPU is not peer-reachable"
        if not copyable:
            psrl_logger.info(f"{self.client_name}: registering RDMA ({reason})")
        return not copyable

    def _sources_peer_reachable(self, plan) -> bool:
        """Whether every source's GPU is visible here and connected by a peer path.

        `all_sources_copyable` is pure plan arithmetic; this is the runtime half of the
        same question. On a CPU-only process there is nothing to check -- the caller has
        already refused to skip for non-CUDA clients -- so no objection is reported.
        """
        if not torch.cuda.is_available():
            return True
        from psrl.utils.nixl.device_ipc import local_device  # noqa: PLC0415
        from psrl.utils.nixl.registration_policy import sources_by_reader  # noqa: PLC0415

        local = self.device.index or 0
        for per_source in sources_by_reader(plan, self.client_name).values():
            for source in per_source:
                info = self._all_client_infos.get(source)
                if info is None:
                    return False
                if not info.device_uuid:
                    continue  # no CUDA identity to check; `ipc_complete` already vouched
                target = local_device(info.device_uuid)
                if target is None:
                    # Same-node (checked by `all_sources_copyable`) but hidden by this
                    # process's CUDA_VISIBLE_DEVICES. IPC still maps it with lazy peer
                    # access, so visibility is not a reason to register (3.7.4).
                    continue
                if target.index != local and not torch.cuda.can_device_access_peer(local, target.index):
                    return False
        return True

    def _is_reader_of(self, source: str) -> bool:
        """Whether this client reads from ``source`` according to the comm plan.

        Used to decide whether to load full descriptors or only connection info
        (E4, design §9c-c). Falls back to True when no plan is present so every
        unknown caller gets the full metadata and no transfer silently fails.
        """
        if self._comm_plan is None:
            return True
        from psrl.utils.nixl.registration_policy import descriptor_readers_by_source  # noqa: PLC0415

        # Peer-diffusion followers read their designate too, and `readers_by_source` does not
        # count them, so a follower would hold only connection info for its designate.
        rbs = descriptor_readers_by_source(self._comm_plan)
        for key_readers in rbs.get(source, {}).values():
            if self.client_name in key_readers:
                return True
        return False

    def _ensure_client_info_fetched(self, target_client: str):
        """Ensure connection to target client is established.

        Two-stage load (E4, design §9c-c): every client loads only the
        connection-info bytes (~401 B, constant) to establish the link; only
        actual readers of `target_client` follow up with the full descriptor
        bytes. This keeps join cost O(members) instead of O(members × regions).

        Falls back to the full metadata when `conn_info_meta` is empty (old
        clients or meta-only registrations) so the new code is backward-
        compatible with existing traffic.
        """
        if target_client in self._target_client_connected:
            return
        assert target_client in self._all_client_infos, (
            f"Target client {target_client} not found in client infos: {self._all_client_infos.keys()}"
        )
        info = self._all_client_infos[target_client]
        if info.conn_info_meta:
            # Stage 1: every client loads the link bytes.
            try:
                self.agent.add_remote_agent(info.conn_info_meta)
            except Exception as e:
                psrl_logger.error(f"Error adding remote agent (conn-info) {target_client}: {e}")
                raise e
            # Stage 2: only readers load the full descriptor bytes.
            if self._is_reader_of(target_client):
                try:
                    self.agent.add_remote_agent(info.meta)
                except Exception as e:
                    psrl_logger.error(f"Error adding remote agent (descriptors) {target_client}: {e}")
                    raise e
                self._full_meta_loaded.add(target_client)
        else:
            # Backward-compatible path: old client or meta-only registration.
            try:
                self.agent.add_remote_agent(info.meta)
            except Exception as e:
                psrl_logger.error(f"Error adding remote agent {target_client}: {e}")
                raise e
            self._full_meta_loaded.add(target_client)
        self._target_client_connected[target_client] = True

    def _ensure_full_metadata(self, target_client: str) -> None:
        """Load the target's full descriptor metadata when this client addresses it.

        The two-stage join loads only the connection bytes for non-readers, because
        most clients neither read nor write a given peer. A WRITE addresses the
        target's memory exactly as a READ does, and PS broadcast is the case the comm
        plan does not describe at all (PS-to-PS), so the writer was left with a link
        but no rkeys and NIXL answered NIXL_ERR_NOT_FOUND.
        """
        loaded = getattr(self, "_full_meta_loaded", None)
        if loaded is None:
            loaded = self._full_meta_loaded = set()
        if target_client in loaded:
            return
        info = self._all_client_infos[target_client]
        if info.meta:
            self.agent.add_remote_agent(info.meta)
        loaded.add(target_client)

    def _release_xfer_handle(self, handle, context: str) -> None:
        """Best-effort release of a NIXL transfer handle that is no longer needed."""
        if handle is None:
            return
        try:
            self.agent.release_xfer_handle(handle)
        except Exception as e:
            psrl_logger.warning(f"{self.client_name}: Failed to release xfer handle ({context}): {e}.")

    def _release_all_xfer_handles(self, context: str) -> None:
        """Release every tracked transfer handle.

        Callers wait before reaching the sync points that invoke this, so a
        still-pending handle means a call site forgot to wait. It is released on
        a best-effort basis so one stuck request cannot block teardown.
        """
        if not self.xfer_handles:
            return
        for handle_key, handle in list(self.xfer_handles.items()):
            self._release_xfer_handle(handle, f"{context}: {handle_key!r}")
        self.xfer_handles = {}

    def _release_prepared_dlists(self, side: str | None = None, target_client: str | None = None) -> None:
        """Release cached prepared descriptor-list handles, optionally filtered.

        Relies on the sync-point contract: no transfer built from a released list
        is in flight when this runs.
        """
        for cache_key in list(self._prepared_dlists.keys()):
            if side is not None and cache_key[0] != side:
                continue
            if target_client is not None and cache_key[1] != target_client:
                continue
            handle, _shard_idxs = self._prepared_dlists.pop(cache_key)
            try:
                self.agent.release_dlist_handle(handle)
            except Exception as e:
                psrl_logger.warning(f"{self.client_name}: Failed to release prepared dlist {cache_key!r}: {e}.")

    def _await_and_release(
        self,
        handle_key: bytes,
        key: str,
        tag: str,
        op_type: str,
        target_client: str | None,
        shard_idx: tuple[int, ...] | None,
        info: NIXLTensorInfo,
        timeout: float,
    ) -> None:
        """Wait for a posted transfer and release its handle.

        ``shard_idx`` is ``None`` for a merged group transfer and the concrete
        shard index for a per-shard (non-contiguous) transfer. READ transfers
        through a temporary buffer are copied back to the original tensor here.
        """
        handle = self.xfer_handles.pop(handle_key, None)
        if handle is None:
            return
        start = time.time()
        try:
            while True:
                try:
                    state = self.agent.check_xfer_state(handle)
                except Exception as e:
                    raise RuntimeError(
                        f"Checking transfer state for ({key!r}, {tag!r}, {op_type}, shard {shard_idx!r}) "
                        f"from {self.client_name!r} to {target_client!r} failed: {e}."
                    ) from e
                if state == "ERR":
                    raise RuntimeError(
                        f"Transfer error for ({key!r}, {tag!r}, {op_type}, shard {shard_idx!r}) "
                        f"from {self.client_name!r} to {target_client!r}."
                    )
                if state == "DONE":
                    break
                if time.time() - start > timeout:
                    raise TimeoutError(
                        f"Timed out waiting for transfer ({key!r}, {tag!r}, {op_type}, shard {shard_idx!r}) "
                        f"from {self.client_name!r} to {target_client!r}."
                    )
                time.sleep(0.001)  # 1ms backoff to avoid CPU starvation on PS nodes at large scale

            # For non-contiguous shards, sync data back to the original tensor after READ.
            if op_type == "READ" and shard_idx is not None:
                local_pos = info.sharding.shard_indices.index(shard_idx)
                if info.desc_bytes_list[local_pos] is None:
                    original_tensor = self._get_local_original_tensor(key, shard_idx)
                    if original_tensor is None:
                        raise RuntimeError(f"No original tensor mapping found for key {key!r} shard {shard_idx!r}.")
                    contiguous_tensor = self._get_local_temp_tensor(key, shard_idx)
                    if contiguous_tensor is None:
                        raise RuntimeError(f"No temporary tensor mapping found for key {key!r} shard {shard_idx!r}.")
                    self._read_contiguous_event_cache[(key, shard_idx)] = torch.cuda.Event()
                    original_tensor.data.copy_(contiguous_tensor)
                    self._read_contiguous_event_cache[(key, shard_idx)].record()
                    psrl_logger.debug(
                        f"Copied temporary contiguous tensor back to the original non-contiguous tensor "
                        f"for key {key!r} shard {shard_idx!r}."
                    )

            if self.enable_nixl_telemetry:
                # Telemetry needs `NIXL_TELEMETRY_ENABLE=true`, so a missing exporter must not fail the transfer.
                try:
                    telem = self.agent.get_xfer_telemetry(handle)
                    psrl_logger.info(
                        f"[nixl_telemetry] {self.client_name} key {key!r} to {target_client} shard {shard_idx}: "
                        f"{telem.totalBytes} bytes, {telem.descCount} descs, "
                        f"post {telem.postDuration}us, xfer {telem.xferDuration}us."
                    )
                except Exception as e:
                    psrl_logger.debug(f"{self.client_name}: NIXL telemetry unavailable for key {key!r}: {e}.")
        finally:
            self._release_xfer_handle(handle, f"({key}, {tag}, {op_type}, shard {shard_idx})")

    def _post_contiguous_group(
        self,
        op_type: str,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        shards_to_transfer: list[tuple[int, ...]],
        local_info: NIXLTensorInfo,
        remote_info: NIXLTensorInfo,
    ) -> set[tuple[int, ...]]:
        """Post one NIXL request covering all contiguous shards of a key.

        Returns the shards covered so the caller transfers the remaining
        non-contiguous shards individually through their temporary buffers.
        """
        local_pos_map = {s: i for i, s in enumerate(local_info.sharding.shard_indices)}
        remote_pos_map = {s: i for i, s in enumerate(remote_info.sharding.shard_indices)}
        entries: list[tuple[tuple[int, ...], Any, Any]] = []
        for shard_idx in shards_to_transfer:
            assert shard_idx in local_pos_map and shard_idx in remote_pos_map, (
                f"Shard {shard_idx!r} not found in local or remote shards for key {key!r}."
            )
            local_pos = local_pos_map[shard_idx]
            remote_pos = remote_pos_map[shard_idx]
            local_desc_bytes = local_info.desc_bytes_list[local_pos]
            if local_desc_bytes is None:
                continue
            assert local_info.shard_meta_infos[local_pos].can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                f"Shard meta info mismatch for key {key!r} shard {shard_idx!r}: "
                f"{local_info.shard_meta_infos[local_pos]!r} != {remote_info.shard_meta_infos[remote_pos]!r}."
            )
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                raise RuntimeError(
                    f"{self.client_name}: Remote descriptor must be contiguous for client transfer, "
                    f"but key {key!r} shard {shard_idx!r} in {target_client!r} is non-contiguous."
                )
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), (
                f"Shard size mismatch for key {key!r} shard {shard_idx!r}: "
                f"{local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}."
            )
            entries.append(
                (
                    shard_idx,
                    self._deserialize_to_xfer_descs(local_desc_bytes),
                    self._deserialize_to_xfer_descs(remote_desc_bytes),
                )
            )
        if not entries:
            return set()
        local_type = entries[0][1].getType()
        remote_type = entries[0][2].getType()
        merged: list[tuple[tuple[int, ...], Any, Any]] = []
        for entry in entries:
            _shard_idx, local_desc, remote_desc = entry
            # A shard whose pair needs a descriptor split (the per-shard path's
            # `_align_desc_counts`) or that spans a different memory type cannot join a
            # single merged request. It stays with the caller's per-shard loop, which is
            # the correct general path; merging is the fast path, not a requirement.
            if local_desc.descCount() != 1 or remote_desc.descCount() != 1:
                continue
            if local_desc.getType() != local_type or remote_desc.getType() != remote_type:
                continue
            merged.append(entry)
        if not merged:
            return set()
        local_desc_tuples = [local_desc[0] for _, local_desc, _ in merged]
        remote_desc_tuples = [remote_desc[0] for _, _, remote_desc in merged]
        group_shard_idxs = [shard_idx for shard_idx, _, _ in merged]
        group_tag = make_xfer_tag(tag, self.client_name, target_client, key)
        try:
            if group_tag not in self.xfer_handles:
                if self.enable_prepared_dlist:
                    handle = self._make_prepped_group_xfer(
                        op_type,
                        target_agent,
                        target_client,
                        key,
                        group_shard_idxs,
                        local_type,
                        remote_type,
                        group_tag,
                    )
                else:
                    merged_local = nixlBind.nixlXferDList(local_type, local_desc_tuples)
                    merged_remote = nixlBind.nixlXferDList(remote_type, remote_desc_tuples)
                    handle = self.agent.initialize_xfer(op_type, merged_local, merged_remote, target_agent, group_tag)
                self.xfer_handles[group_tag] = handle
            handle = self.xfer_handles[group_tag]
        except Exception as e:
            psrl_logger.warning(
                f"{self.client_name}: merged {op_type} to {target_client!r} for key {key!r} could not be "
                f"prepared ({e!r}); falling back to per-shard transfers."
            )
            self.xfer_handles.pop(group_tag, None)
            return set()
        if not handle:
            psrl_logger.warning(
                f"{self.client_name}: merged {op_type} to {target_client!r} for key {key!r} produced no handle; "
                "falling back to per-shard transfers."
            )
            self.xfer_handles.pop(group_tag, None)
            return set()
        try:
            state = self.agent.transfer(handle)
        except Exception as e:
            psrl_logger.warning(
                f"{self.client_name}: posting merged {op_type} to {target_client!r} for key {key!r} failed "
                f"({e!r}); falling back to per-shard transfers."
            )
            self.xfer_handles.pop(group_tag, None)
            return set()
        if state == "ERR":
            psrl_logger.warning(
                f"{self.client_name}: posting merged {op_type} to {target_client!r} for key {key!r} returned ERR; "
                "falling back to per-shard transfers."
            )
            self.xfer_handles.pop(group_tag, None)
            return set()
        psrl_logger.debug(
            f"{self.client_name}: Posted client {op_type} group transfer to {target_client!r} "
            f"for key {key!r} with {len(merged)} merged shards."
        )
        return set(group_shard_idxs)

    def _make_prepped_group_xfer(
        self,
        op_type: str,
        target_agent: str,
        target_client: str,
        key: str,
        shard_indices: list[tuple[int, ...]],
        local_type: int,
        remote_type: int,
        notif_msg: bytes,
    ):
        """Create a transfer from cached prepared descriptor lists for both sides."""
        local_handle, local_shards = self._get_or_create_prepared_dlist(
            "local",
            target_agent,
            "",
            key,
            local_type,
            self.local_client_info.get_tensor_info(key),
        )
        remote_handle, remote_shards = self._get_or_create_prepared_dlist(
            "remote",
            target_agent,
            target_client,
            key,
            remote_type,
            self._all_client_infos[target_client].get_tensor_info(key),
        )
        try:
            local_indices = [local_shards.index(s) for s in shard_indices]
            remote_indices = [remote_shards.index(s) for s in shard_indices]
        except ValueError as e:
            raise RuntimeError(
                f"{self.client_name}: Prepared dlist for key {key!r} is missing a requested shard: {e}."
            ) from e
        return self.agent.make_prepped_xfer(
            op_type, local_handle, local_indices, remote_handle, remote_indices, notif_msg
        )

    def _get_or_create_prepared_dlist(
        self,
        side: str,
        target_agent: str,
        target_client: str,
        key: str,
        mem_type: int,
        info: NIXLTensorInfo,
    ):
        """Return `(prepared_dlist_handle, shard_indices)` for one side, building it on first use."""
        cache_key = (side, target_client, key, mem_type)
        cached = self._prepared_dlists.get(cache_key)
        if cached is not None:
            return cached
        desc_tuples, shard_idxs = [], []
        for pos, shard_idx in enumerate(info.sharding.shard_indices):
            desc_bytes = info.desc_bytes_list[pos]
            if desc_bytes is None:
                continue
            desc = self._deserialize_to_xfer_descs(desc_bytes)
            if desc.getType() != mem_type:
                continue
            if desc.descCount() != 1:
                # Served by the per-shard path, not by a prepared group dlist.
                continue
            desc_tuples.append(desc[0])
            shard_idxs.append(shard_idx)
        if not desc_tuples:
            raise RuntimeError(
                f"{self.client_name}: No {side} descriptors of memory type {mem_type!r} found for key {key!r}."
            )
        xfer_list = nixlBind.nixlXferDList(mem_type, desc_tuples)
        agent_name = "NIXL_INIT_AGENT" if side == "local" else target_agent
        handle = self.agent.prep_xfer_dlist(agent_name, xfer_list)
        entry = (handle, shard_idxs)
        self._prepared_dlists[cache_key] = entry
        return entry

    def client_read(
        self,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
    ) -> list[tuple[int, ...]]:
        """Read from another client, supports shard alignment and communication plan.

        All contiguous shards of the key are posted as a single merged NIXL
        request. Non-contiguous shards fall back to one request per shard
        through their temporary buffers.

        Args:
            target_agent: NIXL agent name of the remote worker to read from.
            target_client: Client name on the remote agent that holds the tensor.
            key: Parameter name identifying the tensor to transfer.
            tag: Opaque string used to track the transfer handle (must be unique
                per concurrent in-flight transfer for the same key).
            comm_plan: Optional explicit communication plan overriding
                ``self._comm_plan``. When ``None`` the stored plan is used.
                A supplied plan takes precedence.
        """
        plan = comm_plan or self._comm_plan
        self._ensure_client_info_fetched(target_client)
        self._ensure_full_metadata(target_client)
        # A source legitimately may not hold this key: under direct supply (item D3) the
        # sources are train ranks, and pipeline parallelism gives each rank only its own
        # stage's keys. Reading `remote_info` before consulting the plan raised KeyError
        # for every key the rank does not have. Against a PS this never arose -- every PS
        # worker holds every key -- so the ordering looked safe while it was merely
        # untested.
        remote_client_info = self._all_client_infos[target_client]
        if key not in remote_client_info.tensor_infos:
            return []
        remote_info = remote_client_info.get_tensor_info(key)
        local_info = self.local_client_info.get_tensor_info(key)
        shards_to_transfer = []
        if plan and self.client_type == NIXLClientType.PULL_SIDE:
            pull_plan = plan.get_rollout_pull_plan(self.client_name, key)
            if target_client in pull_plan:
                shards_to_transfer = pull_plan[target_client]
        elif plan and self.client_type == NIXLClientType.PUSH_SIDE:
            push_plan = plan.get_train_pull_plan(self.client_name, key)
            if target_client in push_plan:
                shards_to_transfer = push_plan[target_client]
        else:
            # Default behavior: align shards
            for shard_idx in local_info.sharding.shard_indices:
                if shard_idx in remote_info.sharding.shard_indices:
                    shards_to_transfer.append(shard_idx)
        handled_by_group: set[tuple[int, ...]] = set()
        if self.merge_contiguous_xfer:
            handled_by_group = self._post_contiguous_group(
                "READ", target_agent, target_client, key, tag, shards_to_transfer, local_info, remote_info
            )

        for shard_idx in shards_to_transfer:
            if shard_idx in handled_by_group:
                continue
            assert (
                shard_idx in local_info.sharding.shard_indices and shard_idx in remote_info.sharding.shard_indices
            ), f"Shard {shard_idx} not found in local or remote shards for key {key}"
            local_pos = local_info.sharding.shard_indices.index(shard_idx)
            remote_pos = remote_info.sharding.shard_indices.index(shard_idx)

            # §3.4 "版本相等性": the source must hold the version this reader was told to
            # pull. Checked here because this is where the remote shard's metadata is
            # already in hand, so the invariant costs a comparison rather than a round
            # trip -- which is what lets it stay on rather than being a debug mode.
            #
            # A stale shard is the fault a missed publish produces, and without this it is
            # silent: the read succeeds and returns last version's bytes.
            remote_version = remote_info.shard_meta_infos[remote_pos].version
            expected_version = getattr(self, "_expected_version", None)
            if expected_version is not None:
                assert remote_version == expected_version, (
                    f"version mismatch reading {key} shard {shard_idx} from "
                    f"{target_client}: expected v{expected_version}, source holds "
                    f"v{remote_version}" + (" (never published)" if remote_version == -1 else "")
                )
            observed = getattr(self, "_observed_versions", None)
            if observed is not None:
                observed.add(remote_version)

            # For non-contiguous shards, record the running key and shard idx
            running_key, running_shard_idx = None, None
            _phase_t0 = time.time()
            # Get local descriptor (check if it's a temporary one)
            local_desc_bytes = local_info.desc_bytes_list[local_pos]
            if local_desc_bytes is not None:
                assert local_info.shard_meta_infos[local_pos].can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{local_info.shard_meta_infos[local_pos]} != {remote_info.shard_meta_infos[remote_pos]}"
                )
            else:
                meta_info = self._temp_meta_mapping[(key, shard_idx)]
                assert meta_info.can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Temporary shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{meta_info} != {remote_info.shard_meta_infos[remote_pos]} "
                    f"during client_read from {self.client_name} to {target_client}"
                )
                # Use temporary descriptor for non-contiguous shard
                local_desc_bytes = self._get_temp_desc_bytes(self.client_name, key, shard_idx)
                if local_desc_bytes is None:
                    raise RuntimeError(f"No temporary descriptor found for key {key} shard {shard_idx}")
                # Wait for the pinned slot to be available
                if self.max_pinned_temp_memory_slots is not None:
                    pinned_idx = self._temp_pinned_idx_mapping[(key, shard_idx)]
                    slot_key = (meta_info.shape, meta_info.dtype, pinned_idx)
                    if slot_key in self._pinned_slot_running_read_xfer:
                        (
                            running_key,
                            running_tag,
                            running_op_type,
                            running_target_client,
                            running_shard_idx,
                        ) = self._pinned_slot_running_read_xfer[slot_key]
                        self.wait(
                            running_key,
                            running_tag,
                            running_op_type,
                            target_client=running_target_client,
                            shard_idx=running_shard_idx,
                        )
                    self._pinned_slot_running_read_xfer[slot_key] = (
                        key,
                        tag,
                        "READ",
                        target_client,
                        shard_idx,
                    )

            # Get remote descriptor (check if it's a temporary one)
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                raise RuntimeError(
                    f"Remote descriptor must be contiguous for client read, "
                    f"but found key {key} shard {shard_idx} in {target_client} is non-contiguous"
                )

            # Double check the shard size
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), (
                f"Shard size mismatch for key {key} shard {shard_idx}: "
                f"{local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}"
            )
            local_desc = self._deserialize_to_xfer_descs(local_desc_bytes)
            remote_desc = self._deserialize_to_xfer_descs(remote_desc_bytes)
            self._read_phase_us["deserialize"] += (time.time() - _phase_t0) * 1e6
            _phase_t0 = time.time()
            local_desc, remote_desc = self._align_desc_counts(local_desc, remote_desc)
            self._read_phase_us["align"] += (time.time() - _phase_t0) * 1e6
            # Real xfer
            _phase_t0 = time.time()
            try:
                if running_key is not None and running_shard_idx is not None:
                    assert (
                        running_key,
                        running_shard_idx,
                    ) in self._read_contiguous_event_cache, (
                        f"Running key {running_key} shard {running_shard_idx} not found in contiguous event cache"
                    )
                    self._read_contiguous_event_cache[(running_key, running_shard_idx)].synchronize()
                    self._read_contiguous_event_cache.pop((running_key, running_shard_idx))
                xfer_tag = make_xfer_tag(tag, self.client_name, target_client, key, shard_idx)
                if xfer_tag not in self.xfer_handles:
                    self.xfer_handles[xfer_tag] = self.agent.initialize_xfer(
                        "READ",
                        local_desc,
                        remote_desc,
                        target_agent,
                        xfer_tag,
                    )
                handle = self.xfer_handles[xfer_tag]
            except Exception as e:
                raise RuntimeError(
                    f"{self.client_name} creating client READ transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}: {e}"
                ) from e
            if not handle:
                raise RuntimeError(
                    f"{self.client_name} creating client READ transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
            self._read_phase_us["initialize"] += (time.time() - _phase_t0) * 1e6
            _phase_t0 = time.time()
            try:
                state = self.agent.transfer(handle)
            except Exception as e:
                raise RuntimeError(
                    f"{self.client_name} posting client READ transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}: {e}"
                ) from e
            self._read_phase_us["transfer"] += (time.time() - _phase_t0) * 1e6
            if state == "ERR":
                raise RuntimeError(
                    f"{self.client_name} posting client READ transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
        return shards_to_transfer

    def client_write(
        self,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
        use_comm_plan: bool = True,
    ) -> list[tuple[int, ...]]:
        """Write to another client, supports shard alignment and communication plan.

        All contiguous shards of the key are posted as a single merged NIXL
        request. Non-contiguous shards fall back to one request per shard
        through their temporary buffers.

        Args:
            target_agent: NIXL agent name of the remote worker to write to.
            target_client: Client name on the remote agent that holds the tensor.
            key: Parameter name identifying the tensor to transfer.
            tag: Opaque string used to track the transfer handle (must be unique
                per concurrent in-flight transfer for the same key).
            comm_plan: Optional explicit communication plan overriding
                ``self._comm_plan``. When ``None`` the stored plan is used.
                A supplied plan takes precedence.
            use_comm_plan: If ``False``, ignore both ``comm_plan`` and the stored
                ``self._comm_plan`` and fall back to default shard-alignment.
                Useful when writing to a target that is not covered by the
                existing comm plan (e.g. PS-to-PS broadcast transfers).
        """
        plan = (comm_plan or self._comm_plan) if use_comm_plan else None
        self._ensure_client_info_fetched(target_client)
        self._ensure_full_metadata(target_client)
        # Symmetric with `client_read`: a target that does not hold the key has nothing
        # to receive. Unreachable today, since every push target is a PS worker holding
        # every key -- but leaving the two paths asymmetric is how the read side came to
        # have this bug in the first place.
        remote_client_info = self._all_client_infos[target_client]
        if key not in remote_client_info.tensor_infos:
            return []
        remote_info = remote_client_info.get_tensor_info(key)
        local_info = self.local_client_info.get_tensor_info(key)
        shards_to_transfer = []
        if plan and self.client_type == NIXLClientType.PUSH_SIDE:
            push_plan = plan.get_push_plan(self.client_name, key)
            if target_client in push_plan:
                shards_to_transfer = push_plan[target_client]
        else:
            # Default behavior: align shards
            for shard_idx in local_info.sharding.shard_indices:
                if shard_idx in remote_info.sharding.shard_indices:
                    shards_to_transfer.append(shard_idx)
        handled_by_group: set[tuple[int, ...]] = set()
        if self.merge_contiguous_xfer:
            handled_by_group = self._post_contiguous_group(
                "WRITE", target_agent, target_client, key, tag, shards_to_transfer, local_info, remote_info
            )

        for shard_idx in shards_to_transfer:
            if shard_idx in handled_by_group:
                continue
            assert (
                shard_idx in local_info.sharding.shard_indices and shard_idx in remote_info.sharding.shard_indices
            ), f"Shard {shard_idx} not found in local or remote shards for key {key}"
            local_pos = local_info.sharding.shard_indices.index(shard_idx)
            remote_pos = remote_info.sharding.shard_indices.index(shard_idx)

            # Check if local shard is non-contiguous and needs data copying
            local_desc_bytes = local_info.desc_bytes_list[local_pos]
            if local_desc_bytes is not None:
                assert local_info.shard_meta_infos[local_pos].can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{local_info.shard_meta_infos[local_pos]} != {remote_info.shard_meta_infos[remote_pos]}"
                )
            else:
                meta_info = self._temp_meta_mapping[(key, shard_idx)]
                assert meta_info.can_xfer_to(remote_info.shard_meta_infos[remote_pos]), (
                    f"Temporary shard meta info mismatch for key {key} shard {shard_idx}: "
                    f"{meta_info} != {remote_info.shard_meta_infos[remote_pos]}"
                )
                # Non-contiguous shard: copy data to temporary contiguous memory
                original_tensor = self._get_local_original_tensor(key, shard_idx)
                if original_tensor is None:
                    raise RuntimeError(f"No original tensor mapping found for key {key} shard {shard_idx}")
                contiguous_tensor = self._get_local_temp_tensor(key, shard_idx)
                if contiguous_tensor is None:
                    raise RuntimeError(f"No temporary tensor mapping found for key {key} shard {shard_idx}")
                # Wait for the pinned slot to be available
                if self.max_pinned_temp_memory_slots is not None:
                    pinned_idx = self._temp_pinned_idx_mapping[(key, shard_idx)]
                    slot_key = (meta_info.shape, meta_info.dtype, pinned_idx)
                    if slot_key in self._pinned_slot_running_write_xfer:
                        (
                            running_key,
                            running_tag,
                            running_op_type,
                            running_target_client,
                            running_shard_idx,
                        ) = self._pinned_slot_running_write_xfer[slot_key]
                        start_time = time.time()
                        self.wait(
                            running_key,
                            running_tag,
                            running_op_type,
                            target_client=running_target_client,
                            shard_idx=running_shard_idx,
                        )
                        end_time = time.time()
                        psrl_logger.debug(
                            f"{self.client_name} write uncontiguous {(key, shard_idx)}, "
                            f"pinned slot {pinned_idx} is available, time: {end_time - start_time}s"
                        )
                    self._pinned_slot_running_write_xfer[slot_key] = (
                        key,
                        tag,
                        "WRITE",
                        target_client,
                        shard_idx,
                    )
                # Copy data from original non-contiguous tensor to temporary contiguous tensor
                self._write_contiguous_event_cache[(key, shard_idx)] = torch.cuda.Event()
                contiguous_tensor.copy_(original_tensor.detach())
                self._write_contiguous_event_cache[(key, shard_idx)].record()
                # Use temporary descriptor
                local_desc_bytes = self._get_temp_desc_bytes(self.client_name, key, shard_idx)
                if local_desc_bytes is None:
                    raise RuntimeError(
                        f"No temporary descriptor found for key {key} shard {shard_idx} "
                        f"in {self.client_name}'s temp descs"
                    )

            # Get remote descriptor (check if it's a temporary one)
            remote_desc_bytes = remote_info.desc_bytes_list[remote_pos]
            if remote_desc_bytes is None:
                raise RuntimeError(
                    f"Remote descriptor must be contiguous for client write, "
                    f"but found key {key} shard {shard_idx} in {target_client} is non-contiguous"
                )

            # Double check the shard size
            assert local_info.get_shard_size_bytes(local_pos) == remote_info.get_shard_size_bytes(remote_pos), (
                f"Shard size mismatch for key {key} shard {shard_idx}: "
                f"{local_info.get_shard_size_bytes(local_pos)} != {remote_info.get_shard_size_bytes(remote_pos)}"
            )
            local_desc = self._deserialize_to_xfer_descs(local_desc_bytes)
            remote_desc = self._deserialize_to_xfer_descs(remote_desc_bytes)
            local_desc, remote_desc = self._align_desc_counts(local_desc, remote_desc)
            # Real xfer
            try:
                if (key, shard_idx) in self._write_contiguous_event_cache:
                    self._write_contiguous_event_cache[(key, shard_idx)].synchronize()
                    self._write_contiguous_event_cache.pop((key, shard_idx))
                xfer_tag = make_xfer_tag(tag, self.client_name, target_client, key, shard_idx)
                if xfer_tag not in self.xfer_handles:
                    self.xfer_handles[xfer_tag] = self.agent.initialize_xfer(
                        "WRITE", local_desc, remote_desc, target_agent, xfer_tag
                    )
                handle = self.xfer_handles[xfer_tag]
            except Exception as e:
                raise RuntimeError(
                    f"{self.client_name} creating client WRITE transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}: {e}"
                ) from e
            if not handle:
                raise RuntimeError(
                    f"{self.client_name} creating client WRITE transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
            try:
                state = self.agent.transfer(handle)
            except Exception as e:
                raise RuntimeError(
                    f"{self.client_name} posting client WRITE transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}: {e}"
                ) from e
            if state == "ERR":
                raise RuntimeError(
                    f"{self.client_name} posting client WRITE transfer to {target_client} failed for "
                    f"key {key} shard {shard_idx}."
                )
        return shards_to_transfer

    def clear_intermediate_cached_data(self):
        """Clear per-cycle intermediate state and release any leftover handles."""
        self._pinned_slot_running_read_xfer.clear()
        self._pinned_slot_running_write_xfer.clear()
        self._read_contiguous_event_cache.clear()
        self._write_contiguous_event_cache.clear()
        self._release_all_xfer_handles("clear_intermediate_cached_data")

    def wait(
        self,
        key: str,
        tag: str,
        op_type: str,
        target_client: str | None = None,
        shard_idx: tuple[int, ...] | None = None,
        timeout: float = 1800.0,
    ):
        """Wait for the transfer(s) of a key and release their handles.

        With ``shard_idx=None`` this waits for the merged contiguous group and
        every per-shard non-contiguous transfer of the key. With a concrete
        ``shard_idx`` it waits only for that shard's request (used for pinned
        temp-buffer slot reuse).
        """
        info = self.local_client_info.get_tensor_info(key)
        if shard_idx is not None:
            handle_key = make_xfer_tag(tag, self.client_name, target_client, key, shard_idx)
            if handle_key in self.xfer_handles:
                self._await_and_release(handle_key, key, tag, op_type, target_client, shard_idx, info, timeout)
            else:
                psrl_logger.debug(
                    f"Transfer ({key}, {tag}, {op_type}, shard {shard_idx}) "
                    f"from {self.client_name} to {target_client} not found, continue"
                )
            return

        # Whole-key wait: the merged contiguous group first, then any per-shard transfers.
        group_key = make_xfer_tag(tag, self.client_name, target_client, key)
        if group_key in self.xfer_handles:
            self._await_and_release(group_key, key, tag, op_type, target_client, None, info, timeout)
        for shard in info.sharding.shard_indices:
            handle_key = make_xfer_tag(tag, self.client_name, target_client, key, shard)
            if handle_key in self.xfer_handles:
                self._await_and_release(handle_key, key, tag, op_type, target_client, shard, info, timeout)

    def load_state_dict_into_registered_tensors(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        """
        Copy state dictionary weights into registered NIXL buffers.

        Source tensors are reshaped to match the registered sharding before copying.

        Args:
            state_dict: Mapping from parameter names to checkpoint tensors.
        """
        assert self.local_client_info is not None, (
            "load_state_dict_into_registered_tensors: local_client_info is None. Call register_local_tensors() first."
        )
        assert self.local_client_info.is_registered, (
            "load_state_dict_into_registered_tensors: local_client_info.is_registered is False. "
            "register_local_tensors() must have been called with meta_only=False."
        )

        tensor_infos = self.local_client_info.tensor_infos
        for key, src_tensor in state_dict.items():
            if key not in tensor_infos:
                continue

            tensor_info = tensor_infos[key]
            sharding = tensor_info.sharding

            # Reconstruct the full shape when registered and checkpoint tensor ranks differ.
            dst_tensor_sample = None
            for shard_idx_sample in sharding.shard_indices:
                dst_sample = self._original_tensor_mapping.get((key, shard_idx_sample))
                if dst_sample is not None:
                    dst_tensor_sample = dst_sample
                    break
            # A grouped MoE source is already shard-shaped, so there is no full tensor to
            # reconstruct and no rank mismatch to repair.
            if (
                dst_tensor_sample is not None
                and not isinstance(src_tensor, GroupedShards)
                and dst_tensor_sample.ndim != src_tensor.ndim
            ):
                # Reconstruct full unsharded 3D shape from the registered shard shape.
                full_shape = list(dst_tensor_sample.shape)
                for dim, count in sharding.shard_mesh.items():
                    full_shape[dim] *= count
                src_tensor_full = src_tensor.reshape(full_shape)
                # Use the global mesh because the local mesh counts only owned shards.
                shard_dims = list(sharding.shard_mesh.keys())
                shard_counts = list(sharding.shard_mesh.values())
                src_shards = []
                for shard_idx in sharding.shard_indices:
                    shard = src_tensor_full
                    for i_dim, (dim, count) in enumerate(zip(shard_dims, shard_counts)):
                        shard_size = shard.shape[dim] // count
                        shard = shard.narrow(dim, shard_idx[i_dim] * shard_size, shard_size)
                    src_shards.append(shard)
            else:
                # Produce the same shards that were registered.
                # For the default (no-op) sharding this returns [src_tensor] as-is.
                # A grouped MoE key is already split across storages (item C1).
                src_shards = local_shards_of(src_tensor, sharding)

            assert len(src_shards) == len(sharding.shard_indices), (
                f"[{self.client_name}] key={key!r}: sharding produced {len(src_shards)} shards "
                f"but shard_indices has {len(sharding.shard_indices)} entries."
            )

            for shard_idx, src_shard in zip(sharding.shard_indices, src_shards):
                dst_tensor = self._original_tensor_mapping.get((key, shard_idx))
                if dst_tensor is None:
                    psrl_logger.warning(
                        f"[{self.client_name}] key={key!r} shard_idx={shard_idx}: "
                        f"not found in _original_tensor_mapping. Skipping."
                    )
                    continue

                # Shape must match the registered shard's shape.
                assert src_shard.shape == dst_tensor.shape, (
                    f"[{self.client_name}] key={key!r} shard_idx={shard_idx}: "
                    f"src_shard.shape={tuple(src_shard.shape)} != "
                    f"dst_tensor.shape={tuple(dst_tensor.shape)}. "
                    f"src full shape={tuple(src_tensor.shape)}, "
                    f"sharding={sharding.shard_mesh}."
                )
                # numel sanity (catches dtype/element-size confusion early)
                assert src_shard.numel() == dst_tensor.numel(), (
                    f"[{self.client_name}] key={key!r} shard_idx={shard_idx}: "
                    f"src_shard.numel()={src_shard.numel()} != dst_tensor.numel()={dst_tensor.numel()}."
                )
                # copy_ handles device and dtype cast automatically.
                dst_tensor.copy_(src_shard, non_blocking=False)

    def log_shard_info(self, label: str = "", max_elements: int = 8):
        """
        Print detailed shard information for debugging push/pull precision issues.

        For each (key, shard_idx) known to this client, logs:
          - shard type: "contiguous" (from _original_tensor_mapping) or "temp" (from _temp_tensor_mapping)
          - shard name (key) and shard indices
          - tensor stats: shape, dtype, min/max/mean/norm and the first `max_elements` values

        Args:
            label: A descriptive label printed at the start (e.g. "BEFORE_PUSH").
            max_elements: How many scalar values to print per shard.
        """
        all_keys: set[tuple[str, tuple[int, ...]]] = set(self._original_tensor_mapping.keys()) | set(
            self._temp_tensor_mapping.keys()
        )
        psrl_logger.info(
            f"[{self.client_name}] === log_shard_info [{label}] === "
            f"total shards: {len(all_keys)} "
            f"(uncontiguous: {len(self._temp_tensor_mapping)})"
        )
        for key, shard_idx in sorted(all_keys, key=lambda x: (x[0], x[1])):
            # Determine type and pick the tensor
            if (key, shard_idx) in self._temp_tensor_mapping:
                shard_type = "uncontiguous"
                temp_tensor = self._temp_tensor_mapping[(key, shard_idx)]
            else:
                shard_type = "contiguous"
                temp_tensor = None
            assert (key, shard_idx) in self._original_tensor_mapping, (
                f"[{self.client_name}][{label}]: key={key!r}, shard_idx={shard_idx}, "
                f"type={shard_type}, not found in _original_tensor_mapping"
            )
            tensor = self._original_tensor_mapping[(key, shard_idx)]

            log_tensor(
                tensor=tensor,
                psrl_logger=psrl_logger,
                log_prefix=f"{self.client_name}][{label}",
                name="shard",
                max_elements=max_elements,
                key=key,
                shard_idx=shard_idx,
                shard_type=shard_type,
                tensor_variant="original",
            )
            if temp_tensor is not None:
                log_tensor(
                    tensor=temp_tensor,
                    psrl_logger=psrl_logger,
                    log_prefix=f"{self.client_name}][{label}",
                    name="shard",
                    max_elements=max_elements,
                    key=key,
                    shard_idx=shard_idx,
                    shard_type=shard_type,
                    tensor_variant="temp",
                )

    def shutdown(self):
        # Release handles before their backing memory is deregistered, so no
        # stale-generation handle can outlive the remote metadata teardown.
        self._release_all_xfer_handles("shutdown")
        self._release_prepared_dlists()
        self._deregister_all_descs()


class NIXLMultiStorageClients:
    """
    Multiple NIXLStorageClient instances can be registered to the same NIXL agent.
    This is useful for multi-precision (e.g., train use fp32, gen use bf16).
    """

    def __init__(
        self,
        agent_name: str,
        multi_client_names: list[str],
        server_name: str,
        use_gpu: bool,
        multi_client_types: list[NIXLClientType],
        nixl_config: DictConfig,
        replica_idx: int = 0,
        worker_index: int = 0,
        client_group_id: int = -1,  # -1 is the default client group
        logging_path: str | None = None,
    ):
        self.agent_name = agent_name
        self.multi_client_names = multi_client_names
        self.server_name = server_name
        if use_gpu:
            assert torch.cuda.is_available(), "CUDA is not available."
        self.device = torch.device("cuda:0" if use_gpu else "cpu")
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        # Same budget as the single client and the meta server; without it the handshake
        # waits below raise AttributeError rather than timing out.
        self._handshake_timeout = float(getattr(nixl_config, "handshake_timeout_s", 600.0))
        # Initialize NIXL agent
        from psrl.utils.nixl.port_scanner import get_port_scanner

        worker_ip = get_worker_info()[0]
        port_scanner = get_port_scanner(worker_ip)
        self.client_port = ray.get(port_scanner.find_free_port.remote())
        self.agent = nixl_agent(
            self.agent_name,
            nixl_agent_config(
                enable_prog_thread=True,
                enable_listen_thread=True,
                listen_port=self.client_port,
                capture_telemetry=bool(nixl_config.get("capture_telemetry", False)),
            ),
        )

        # Initialize multi clients
        self.multi_clients: list[NIXLStorageClient] = []
        for client_name, client_type in zip(multi_client_names, multi_client_types):
            self.multi_clients.append(
                NIXLStorageClient(
                    client_name,
                    server_name,
                    use_gpu,
                    client_type,
                    nixl_config,
                    replica_idx=replica_idx,
                    worker_index=worker_index,
                    binded_agent=self.agent,
                    client_group_id=client_group_id,
                    logging_path=logging_path,
                )
            )

        # They share one NIXL agent, so an eviction by any of them applies to all.
        for client in self.multi_clients:
            client._agent_siblings = self.multi_clients

        self._is_connected = False
        self._multi_unified_sharding_dicts_fetched = False

    def get_client_by_name(self, client_name: str) -> NIXLStorageClient:
        for client in self.multi_clients:
            if client.client_name == client_name:
                return client
        raise ValueError(f"Client {client_name} not found")

    def connect_to_server(self, timeout: float | None = None):
        assert not self._is_connected, "Already connected to server"
        self.agent.fetch_remote_metadata(self.server_name, self.server_ip, self.server_port)
        self.agent.send_local_metadata(self.server_ip, self.server_port)
        # `None` selects the configured budget, keeping client and server aligned.
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        ready = False
        while not ready:
            ready = self.agent.check_remote_metadata(self.server_name)
            if time.time() - start > budget:
                raise TimeoutError("Timeout waiting for server metadata to be fetched and connected.")
            time.sleep(0.1)
        self._is_connected = True
        for client in self.multi_clients:
            client._is_connected = True

    def send_local_sharding(self, multi_sharding_dicts: dict[str, dict[str, NIXLSharding]]):
        assert self._is_connected, "Not connected to server"
        # multi_sharding_dicts: {client_name: {key: NIXLSharding}}
        self.agent.send_notif(self.server_name, encode(NotifKind.SHARDING, multi_sharding_dicts))

    def send_local_info(self):
        assert self._is_connected, "Not connected to server"
        for client in self.multi_clients:
            assert client.local_client_info is not None, "Local client info not registered"
        self.agent.send_notif(
            self.server_name,
            encode(
                NotifKind.CLIENT_INFO,
                {client.client_name: client.local_client_info.serialize() for client in self.multi_clients},
            ),
        )

    def send_local_temp_mapping(self):
        assert self._is_connected, "Not connected to server"
        for client in self.multi_clients:
            assert client._temp_desc_bytes_mapping is not None, "Temp desc bytes mapping not registered"
        self.agent.send_notif(
            self.server_name,
            encode(
                NotifKind.TEMP_MAPPING,
                {client.client_name: client._temp_desc_bytes_mapping for client in self.multi_clients},
            ),
        )

    def wait_for_server_sharding(self, timeout: float | None = None):
        assert self._is_connected, "Not connected to server"
        # `None` selects the configured budget, keeping client and server aligned.
        budget = self._handshake_timeout if timeout is None else timeout
        start = time.time()
        if self._multi_unified_sharding_dicts_fetched:
            return
        while True:
            notifs = self.agent.get_new_notifs()
            if self.server_name in notifs and notifs[self.server_name]:
                client_sharding_dicts = pickle.loads(notifs[self.server_name][0])
                assert isinstance(client_sharding_dicts, dict), (
                    f"Expected a dict of client sharding dicts, but got {client_sharding_dicts}"
                )
                for client_name, sharding_dict in client_sharding_dicts.items():
                    assert client_name in self.multi_client_names, (
                        f"Client {client_name} not found in {self.multi_client_names}"
                    )
                    self.multi_clients[
                        self.multi_client_names.index(client_name)
                    ]._unified_sharding_dict = sharding_dict
                break
            if time.time() - start > budget:
                raise TimeoutError("Timeout waiting for server sharding notification.")
            time.sleep(0.1)
        self._multi_unified_sharding_dicts_fetched = True
        return {client.client_name: client._unified_sharding_dict for client in self.multi_clients}

    def wait_for_server_info(self, timeout: float | None = None):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].wait_for_server_info(timeout)
        if len(self.multi_clients) > 1:
            for client in self.multi_clients[1:]:
                client._all_client_infos = self.multi_clients[0]._all_client_infos
                client._comm_plan = self.multi_clients[0]._comm_plan
                client._all_client_infos_fetched = True
                client._release_prepared_dlists(side="remote")

    def wait_for_server_temp_mappings(self, timeout: float | None = None):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].wait_for_server_temp_mappings(timeout)
        if len(self.multi_clients) > 1:
            for client in self.multi_clients[1:]:
                client._all_temp_mappings = self.multi_clients[0]._all_temp_mappings

    def wait_for_update_infos(self, expected_agents: int, timeout: float | None = None):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].wait_for_update_infos(expected_agents, timeout)
        if len(self.multi_clients) > 1:
            for client in self.multi_clients[1:]:
                client._all_client_infos = self.multi_clients[0]._all_client_infos
                client._all_temp_mappings = self.multi_clients[0]._all_temp_mappings
                client._release_prepared_dlists(side="remote")

    def broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        assert self._is_connected, "Not connected to server"
        self.multi_clients[0].broadcast_update_client_infos(dst_agent_names, update_client_names)

    def client_read(
        self,
        cur_client: str,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
    ):
        assert self._is_connected, "Not connected to server"
        client = self.get_client_by_name(cur_client)
        client.client_read(target_agent, target_client, key, tag, comm_plan)

    def client_write(
        self,
        cur_client: str,
        target_agent: str,
        target_client: str,
        key: str,
        tag: str,
        comm_plan: NIXLCommPlan | None = None,
        use_comm_plan: bool = True,
    ):
        assert self._is_connected, "Not connected to server"
        client = self.get_client_by_name(cur_client)
        client.client_write(target_agent, target_client, key, tag, comm_plan, use_comm_plan=use_comm_plan)

    def wait(
        self,
        cur_client: str,
        key: str,
        tag: str,
        op_type: str,
        target_client: str | None = None,
        shard_idx: tuple[int, ...] | None = None,
        timeout: float = 1200.0,
    ):
        assert self._is_connected, "Not connected to server"
        client = self.get_client_by_name(cur_client)
        client.wait(key, tag, op_type, target_client=target_client, shard_idx=shard_idx, timeout=timeout)

    def log_shard_info(self, label: str = "", max_elements: int = 8):
        """Call log_shard_info on every sub-client."""
        for client in self.multi_clients:
            client.log_shard_info(label=label, max_elements=max_elements)

    def shutdown(self):
        # TODO(lhy): Avoid duplicate releases when clients share memory.
        for client in self.multi_clients:
            client.shutdown()
