"""
The pull engine moves the right bytes for every train and gen layout, with or without peers.

Real clients register real tensors and the real planner builds the plan. Only the transport
is replaced by host memcpy (`memcpy_agent.py`), so every descriptor the engine posts is
executed and the layout judge checks the result element by element.
"""

from __future__ import annotations

import copy
import ctypes
import threading
from collections import OrderedDict

import pytest
import torch
from omegaconf import OmegaConf
from psrl.utils.common.worker_naming import gen_client_name, model_role_of, train_client_name
from psrl.utils.nixl.block_fp8 import quantize_blockwise, scale_shape
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.comm_plan import CommunicationPlanner
from psrl.utils.nixl.handshake_notif import NotifKind
from psrl.utils.nixl.nixl_spec import NIXLClientType, NIXLSharding
from psrl.utils.nixl.pull_engine import ChunkUnavailable, NIXLPullEngine
from psrl.utils.nixl.server import NIXLMetaServer
from psrl.utils.quantization import QuantizationPlan, VllmFp8BlockQuantizer

from tests.nixl.coord_judge import check_by_layout, check_fp8_exact, fill_by_layout, fill_fp8_exact
from tests.nixl.memcpy_agent import MemcpyAgent, NotifBus

pytestmark = pytest.mark.cpu_test

HIDDEN, INTER = 128, 256
# key -> (global shape, dim cut by tensor parallelism or None)
MODEL = {
    "layers.0.q_proj.weight": ((INTER, HIDDEN), 0),
    "layers.0.o_proj.weight": ((HIDDEN, INTER), 1),
    "layers.0.down_proj.weight": ((HIDDEN, INTER), 1),
    "layers.0.norm.weight": ((HIDDEN,), None),
}

def _train_ranks_remote(clients: dict) -> None:
    """Give train ranks a distinct node when a test exercises diffusion.

    Diffusion now collapses *cross-node* upstream reads only; a same-node upstream is
    served by a direct device copy and deliberately forms no groups. The loopback
    transport in these tests does not care where the source lives, but the planner reads
    node identities, so a same-node fixture would correctly plan no groups at all.
    """
    for name, client in clients.items():
        if name.startswith("NIXLTrainClient"):
            client.local_client_info.node_ip = "train-node"

CONFIG = OmegaConf.create(
    {
        "server_ip": "127.0.0.1",
        "server_port": 0,
        "max_pinned_temp_memory_slots": None,
        "enable_tms_for_temp_buffers": False,
    }
)


def _sharding(dim, tp, rank) -> NIXLSharding:
    if dim is None:
        return NIXLSharding.default()
    return NIXLSharding(shard_mesh=OrderedDict([(dim, tp)]), shard_indices=[(rank,)])


# Keeps the arenas behind `_locals` alive for the life of the test process.
_ARENAS: list[torch.Tensor] = []


def _locals(model: dict, tp: int, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """
    Carve local tensors back to back from one buffer, each over its own storage like a caching allocator.

    Adjacent storages are what tempts a coalescer to merge a descriptor across two registrations.
    """
    shapes = {}
    for key, (shape, dim) in model.items():
        shape = list(shape)
        if dim is not None:
            shape[dim] //= tp
        shapes[key] = shape
    arena = torch.zeros(sum(torch.Size(s).numel() for s in shapes.values()), dtype=dtype)
    itemsize = arena.element_size()
    _ARENAS.append(arena)
    out, offset = {}, 0
    for key, shape in shapes.items():
        n = torch.Size(shape).numel()
        window = (ctypes.c_char * (n * itemsize)).from_address(arena[offset:].data_ptr())
        out[key] = torch.frombuffer(window, dtype=dtype).view(shape)
        offset += n
    return out


class Cluster:
    model = MODEL

    def __init__(
        self,
        train_tp: int,
        gen_tp: int,
        instances: int,
        peer_diffusion: bool,
        train_dtype=torch.bfloat16,
        model: dict | None = None,
    ):
        self.bus = NotifBus()
        self.model = MODEL if model is None else model
        self.declared: dict[str, dict] = {}
        self.states: dict[str, dict] = {}
        kinds = {}
        for r in range(train_tp):
            name = f"NIXLTrainClient_{r}"
            kinds[name] = NIXLClientType.PUSH_SIDE
            self.states[name] = _locals(self.model, train_tp, train_dtype)
            self.declared[name] = {k: _sharding(d, train_tp, r) for k, (s, d) in self.model.items()}
        for i in range(instances):
            for r in range(gen_tp):
                name = f"NIXLGenClient_I{i}_R{r}"
                kinds[name] = NIXLClientType.PULL_SIDE
                self.states[name] = _locals(self.model, gen_tp)
                self.declared[name] = {k: _sharding(d, gen_tp, r) for k, (s, d) in self.model.items()}
        for name in self.states:
            if name.startswith("NIXLTrainClient"):
                fill_by_layout(self.states[name], self.declared[name], version=7, wire_dtype=torch.bfloat16)

        unified = {name: copy.deepcopy(d) for name, d in self.declared.items()}
        for key in self.model:
            finest = NIXLSharding.find_finest_shard_mesh([unified[n][key].shard_mesh for n in unified])
            for name in unified:
                unified[name][key].refactor_based_on_finer_shard_mesh(finest)

        self.clients: dict[str, NIXLStorageClient] = {}
        for name, kind in kinds.items():
            client = NIXLStorageClient(
                client_name=name,
                server_name="server",
                use_gpu=False,
                client_type=kind,
                nixl_config=CONFIG,
                binded_agent=MemcpyAgent(name, self.bus),
            )
            if kind is NIXLClientType.PUSH_SIDE:
                client.set_publish_cast(train_dtype, torch.bfloat16)
            client.register_local_tensors(self.states[name], unified[name])
            self.clients[name] = client
        if peer_diffusion:
            _train_ranks_remote(self.clients)
        infos = {name: c.local_client_info for name, c in self.clients.items()}
        plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
        for client in self.clients.values():
            client._all_client_infos = dict(infos)
            client._comm_plan = plan
        self.plan = plan
        self.gen = [n for n in self.clients if n.startswith("NIXLGenClient")]
        self.engines = {
            n: NIXLPullEngine(self.clients[n], staging_bytes=1 << 16, peer_timeout_s=5.0) for n in self.gen
        }

    def pull(self, epoch: int, participants: set[str] | None = None) -> dict:
        names = sorted(participants) if participants is not None else self.gen
        reports, errors = {}, []

        def run(name):
            try:
                reports[name] = self.engines[name].pull(epoch, participants)
            except Exception as exc:  # surfaced below
                errors.append((name, exc))

        threads = [threading.Thread(target=run, args=(n,)) for n in names]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        return reports

    def judge(self, names, version: int) -> None:
        for name in names:
            judged, failure = check_by_layout(self.states[name], self.declared[name], list(self.model), version=version)
            assert failure is None, f"{name}: {failure}"
            assert judged == len(self.model)


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (4, 2), (2, 4), (1, 4), (4, 1)])
def test_every_layout_pulls_exact_bytes_without_peers(train_tp, gen_tp):
    cluster = Cluster(train_tp, gen_tp, instances=1, peer_diffusion=False)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.peer_keys == 0 and report.versions == {-1}


def test_fragmented_sources_get_publish_copies_and_readers_one_descriptor_per_shard():
    cluster = Cluster(train_tp=2, gen_tp=4, instances=1, peer_diffusion=False)
    assert all(cluster.clients[f"NIXLTrainClient_{r}"].publish_copy_bytes > 0 for r in range(2))
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    # q is split on dim 0 on both sides, o and down come from contiguous copies, norm is one run.
    for report in reports.values():
        assert report.descriptors <= len(MODEL), report


def test_fragmented_readers_stage_through_the_pool():
    cluster = Cluster(train_tp=4, gen_tp=1, instances=1, peer_diffusion=False)
    report = cluster.pull(epoch=1)["NIXLGenClient_I0_R0"]
    cluster.judge(cluster.gen, version=7)
    assert report.staged_bytes > 0 and report.descriptors < 64


LARGE_LATE_MODEL = {
    "layers.0.a_proj.weight": ((HIDDEN, INTER), 1),
    "layers.0.b_proj.weight": ((HIDDEN, INTER), 1),
    "layers.0.c_proj.weight": ((HIDDEN, INTER), 1),
    "layers.0.d_proj.weight": ((HIDDEN, INTER), 1),
    # Last in wave order and four times a small key's bytes, so its staged cell neither
    # fixes the pool size nor lands in the first wave.
    "layers.0.z_big.weight": ((HIDDEN, INTER * 4), 1),
}


def test_the_staging_pool_is_sized_before_any_read_is_posted():
    """A later wave needing a bigger staging item must not resize the pool mid-pull.

    With peer diffusion the first staged wave runs while peer reads are in flight. The
    engine now plans every source's specs up front and sizes the pool before posting, so
    the phase cache is never invalidated under a running read. The old lazy allocation
    freed in-flight handles here, which NIXL reports as `BAD_STATUS` (the strict
    `MemcpyAgent` raises it too).
    """
    cluster = Cluster(train_tp=4, gen_tp=1, instances=2, peer_diffusion=True, model=LARGE_LATE_MODEL)
    grew: list[tuple[str, int]] = []
    grew_in_flight: list[str] = []
    for name, engine in cluster.engines.items():
        original = engine._ensure_pool

        def ensure(need, _engine=engine, _name=name, _original=original):
            in_flight = bool(_engine._in_flight)
            before = _engine._pool.numel() if _engine._pool is not None else None
            _original(need)
            after = _engine._pool.numel() if _engine._pool is not None else None
            if after != before:
                grew.append((_name, need))
                if in_flight:
                    grew_in_flight.append(_name)

        engine._ensure_pool = ensure

    cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    assert grew, "the scenario never needed a staging pool"
    assert not grew_in_flight, f"the pool was resized while reads were posted: {grew_in_flight}"


def test_a_pool_resize_keeps_a_running_reads_handle():
    """Handle ownership under a resize, independent of pull geometry.

    A take-over phase can still need a bigger staging item after reads were posted. The
    resize must not release a request NIXL is still polling: the handle is orphaned, the
    old pool is retired, and both are reclaimed once the read completes.
    """
    cluster = Cluster(train_tp=4, gen_tp=1, instances=1, peer_diffusion=False)
    engine = cluster.engines["NIXLGenClient_I0_R0"]
    engine.pull(epoch=1)

    phase = next(p for p in engine._phases.values() if p.direct)
    assert phase.direct, "the scenario needs a direct handle to keep in flight"
    handle = phase.direct[0]
    engine._post([handle])
    assert id(handle) in engine._in_flight

    engine._ensure_pool(1 << 20)

    assert not handle.handle.released, "a resize released a request that was still posted"
    assert any(orphan is handle for orphan in engine._orphans)
    assert engine._retired_pools, "the replaced pool was freed instead of retired"

    engine._reap_orphans()
    assert handle.handle.released, "the orphaned handle was never released after completion"
    assert not engine._retired_pools


def test_a_conn_info_only_peer_is_never_elected_as_designate():
    """A peer without read addresses must not be planned as a source.

    Peers receive conn-info-only metadata when the handshake plan does not make them
    readers. A sync over a participant subset can elect a different designate from the
    same group, and reading through an entry that never loaded addresses fails after
    other waves have already moved. The reader must fall back to a readable peer or to
    the trainer instead.
    """
    cluster = Cluster(train_tp=4, gen_tp=1, instances=2, peer_diffusion=True)
    reader = "NIXLGenClient_I0_R0"
    holder = "NIXLGenClient_I1_R0"
    engine = cluster.engines[reader]
    # Strip the addresses exactly as a conn-info-only NIXLClientInfo would have none.
    other = cluster.clients[reader]._all_client_infos[holder]
    for tensor in other.tensor_infos.values():
        tensor.read_addrs = [None] * len(tensor.read_addrs) if tensor.read_addrs else []

    upstream, peer, _followers = engine._assign(lambda client: True)
    local = cluster.clients[reader].local_client_info.tensor_infos
    for source, keys in peer.items():
        assert source != holder
        for key in keys:
            shards = list(local[key].sharding.shard_indices)
            assert engine._source_readable(source, key, shards)

    report = engine.pull(epoch=1)
    assert report.versions == {-1} or report.versions
    cluster.judge([reader], version=7)


@pytest.mark.parametrize("train_tp,gen_tp", [(4, 2), (2, 4)])
def test_peers_read_their_designate_and_trainer_egress_is_one_copy_per_group(train_tp, gen_tp):
    cluster = Cluster(train_tp, gen_tp, instances=3, peer_diffusion=True)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    upstream = sum(r.upstream_bytes for r in reports.values())
    model_bytes = sum(torch.Size(s).numel() * 2 for s, _ in MODEL.values())
    # Each byte leaves the trainer once per node, whichever instance happens to be its designate.
    assert upstream == model_bytes, (upstream, model_bytes)
    # Upstream work is spread over the instances rather than concentrated on one.
    assert sum(1 for r in reports.values() if r.upstream_keys) > gen_tp
    for report in reports.values():
        assert report.fallback_keys == 0
        assert report.peer_keys == 0 or report.peer_bytes > 0


def test_peer_reads_are_one_descriptor_per_key():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=2, peer_diffusion=True)
    reports = cluster.pull(epoch=1)
    for name, report in reports.items():
        if report.peer_keys:
            tensor_infos = cluster.clients[name].local_client_info.tensor_infos

            def is_designate(k):
                shard = tensor_infos[k].sharding.shard_indices[0]
                group = cluster.plan.get_peer_group(name, k, shard)
                return not group or group[0] == name

            peer_only = (
                report.descriptors
                - cluster.engines[name]
                ._phase(
                    cluster.engines[name]._upstream_specs(
                        {
                            k: v
                            for k, v in cluster.plan.rollout_pull_from_ps_plan[name].items()
                            if is_designate(k)
                        }
                    )
                )
                .descriptors
            )
            assert peer_only == report.peer_keys, (name, report)


def test_a_subset_of_participants_diffuses_among_itself_only():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=3, peer_diffusion=True)
    subset = {"NIXLGenClient_I1_R0", "NIXLGenClient_I1_R1", "NIXLGenClient_I2_R0", "NIXLGenClient_I2_R1"}
    cluster.pull(epoch=1, participants=subset)
    cluster.judge(sorted(subset), version=7)
    # The instance that did not sync was never read into.
    assert all(int(t.abs().sum()) == 0 for t in cluster.states["NIXLGenClient_I0_R0"].values())


def test_a_second_version_replaces_every_byte_and_stale_signals_do_not_leak():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)
    for r in range(4):
        name = f"NIXLTrainClient_{r}"
        fill_by_layout(cluster.states[name], cluster.declared[name], version=8, wire_dtype=torch.bfloat16)
        cluster.clients[name].publish_version(8)
    cluster.pull(epoch=2)
    cluster.judge(cluster.gen, version=8)


def test_a_missing_designate_falls_back_to_upstream():
    cluster = Cluster(train_tp=4, gen_tp=2, instances=2, peer_diffusion=True)
    for engine in cluster.engines.values():
        engine.peer_timeout_s = 0.5
    # Claim both instances participate but run only one, so its designates never answer.
    everyone = set(cluster.gen)
    lone = ["NIXLGenClient_I1_R0", "NIXLGenClient_I1_R1"]
    reports = {}
    threads = [
        threading.Thread(target=lambda n=n: reports.__setitem__(n, cluster.engines[n].pull(1, everyone))) for n in lone
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    cluster.judge(lone, version=7)
    assert sum(r.fallback_keys for r in reports.values()) > 0


def test_descriptors_never_span_two_registered_allocations():
    """Back to back storages on both sides must still be read as separate descriptors."""
    cluster = Cluster(train_tp=2, gen_tp=2, instances=1, peer_diffusion=False)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.descriptors == len(MODEL), report


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (2, 4), (4, 2)])
def test_a_float32_trainer_publishes_bfloat16_copies(train_tp, gen_tp):
    cluster = Cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, train_dtype=torch.float32)
    for r in range(train_tp):
        client = cluster.clients[f"NIXLTrainClient_{r}"]
        assert client.publish_copy_bytes == sum(t.numel() * 2 for t in cluster.states[client.client_name].values())
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.descriptors <= 2 * len(MODEL), report


def test_past_the_copy_budget_readers_span_read_in_place():
    CONFIG.publish_copy_budget_mb = 0
    try:
        cluster = Cluster(train_tp=2, gen_tp=4, instances=1, peer_diffusion=False)
    finally:
        del CONFIG["publish_copy_budget_mb"]
    assert all(cluster.clients[f"NIXLTrainClient_{r}"].publish_copy_bytes == 0 for r in range(2))
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    for report in reports.values():
        assert report.staged_bytes > 0 and report.descriptors <= 2 * len(MODEL), report


def test_the_pull_path_makes_no_control_plane_calls():
    """Source selection is a lookup in the plan received at the handshake, never a Ray call per pull."""
    import inspect

    from psrl.utils.nixl import pull_engine
    from psrl.utils.nixl.client import NIXLStorageClient

    source = inspect.getsource(pull_engine) + inspect.getsource(NIXLStorageClient.client_read)
    assert ".remote(" not in source and "ray.get" not in source
    assert "comm_plan" in inspect.getsource(NIXLStorageClient.wait_for_server_info)


BLOCK = (32, 32)
FP8_PLAN = QuantizationPlan(
    scheme="fp8",
    block=BLOCK,
    scale_suffix="_scale_inv",
    scale_dtype="float32",
    weight_dtype="float8_e4m3fn",
)
FP8_KEYS = ["layers.0.q_proj.weight", "layers.0.o_proj.weight", "layers.0.down_proj.weight"]


def _fp8_cluster(
    train_tp,
    gen_tp,
    instances,
    peer_diffusion,
    train_dtype=torch.bfloat16,
    scale_tile=BLOCK,
    train_fp8=False,
    reader_quantization=False,
):
    """Build a cluster whose rollouts are FP8, reusing `Cluster` with the gen state swapped before registration."""
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states = {}, {}
    kinds = {}
    for r in range(train_tp):
        name = f"NIXLTrainClient_{r}"
        kinds[name] = NIXLClientType.PUSH_SIDE
        cluster.states[name] = _locals(MODEL, train_tp, train_dtype)
        cluster.declared[name] = {k: _sharding(d, train_tp, r) for k, (s, d) in MODEL.items()}
    for i in range(instances):
        for r in range(gen_tp):
            name = f"NIXLGenClient_I{i}_R{r}"
            kinds[name] = NIXLClientType.PULL_SIDE
            state = _locals(MODEL, gen_tp)
            declared = {k: _sharding(d, gen_tp, r) for k, (s, d) in MODEL.items()}
            for key in FP8_KEYS:
                state[key] = torch.zeros(state[key].shape, dtype=torch.float8_e4m3fn)
                padded_tile = (1,) * (state[key].dim() - 2) + tuple(scale_tile)
                state[key + "_scale_inv"] = torch.zeros(
                    scale_shape(state[key].shape, padded_tile), dtype=torch.float32
                )
                declared[key + "_scale_inv"] = copy.deepcopy(declared[key])
                declared[key + "_scale_inv"].scale_tile = scale_tile
            cluster.states[name], cluster.declared[name] = state, declared

    def tile_of(t):
        return (1,) * (t.dim() - 2) + BLOCK

    for r in range(train_tp):
        name = f"NIXLTrainClient_{r}"
        fill_fp8_exact(cluster.states[name], cluster.declared[name], FP8_KEYS, tile_of, version=7, uniform=train_fp8)
        rest = {k: v for k, v in cluster.states[name].items() if k not in FP8_KEYS}
        fill_by_layout(rest, cluster.declared[name], version=7, wire_dtype=torch.bfloat16)
        if train_fp8:
            # A Megatron `fp8_param` trainer: per tensor scaled FP8 bytes, as the converter emits them.
            for key in FP8_KEYS:
                values = cluster.states[name][key]
                scale = torch.tensor(2.0**-7)
                fp8 = (values.float() / scale).to(torch.float8_e4m3fn)
                fp8.psrl_fp8_scale = (lambda scale=scale: scale, train_dtype)
                cluster.states[name][key] = fp8

    # The server derives the trainer's scale shardings, so run its unification rather than a copy of it.
    from psrl.utils.nixl.server import NIXLMetaServer

    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in cluster.declared.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    from psrl.utils.nixl.handshake_notif import NotifKind

    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(cluster.declared)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        if kind is NIXLClientType.PUSH_SIDE:
            client.set_publish_cast(train_dtype, torch.bfloat16)
            client.set_publish_quantization(FP8_PLAN, reader_quantization=reader_quantization)
        elif reader_quantization:
            client.set_reader_quantization(FP8_PLAN)
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    if peer_diffusion:
        _train_ranks_remote(cluster.clients)
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = plan
    cluster.plan = plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=5.0) for n in cluster.gen
    }
    return cluster


def _fp8_cluster_mixed_layout(
    train_tp,
    layouts,
    peer_diffusion,
    train_dtype=torch.bfloat16,
    scale_tile=BLOCK,
    reader_quantization=True,
):
    """Like `_fp8_cluster`, but `layouts` gives one gen_tp per instance group instead of
    a single uniform `gen_tp` -- the phase 2 scenario where different target layouts
    for the same model share a node and must still find a designate per shard, even
    though their whole shard sets never match."""
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states = {}, {}
    kinds = {}
    for r in range(train_tp):
        name = f"NIXLTrainClient_{r}"
        kinds[name] = NIXLClientType.PUSH_SIDE
        cluster.states[name] = _locals(MODEL, train_tp, train_dtype)
        cluster.declared[name] = {k: _sharding(d, train_tp, r) for k, (s, d) in MODEL.items()}
    for group, gen_tp in enumerate(layouts):
        for r in range(gen_tp):
            name = f"NIXLGenClient_L{group}_R{r}"
            kinds[name] = NIXLClientType.PULL_SIDE
            state = _locals(MODEL, gen_tp)
            declared = {k: _sharding(d, gen_tp, r) for k, (s, d) in MODEL.items()}
            for key in FP8_KEYS:
                state[key] = torch.zeros(state[key].shape, dtype=torch.float8_e4m3fn)
                padded_tile = (1,) * (state[key].dim() - 2) + tuple(scale_tile)
                state[key + "_scale_inv"] = torch.zeros(
                    scale_shape(state[key].shape, padded_tile), dtype=torch.float32
                )
                declared[key + "_scale_inv"] = copy.deepcopy(declared[key])
                declared[key + "_scale_inv"].scale_tile = scale_tile
            cluster.states[name], cluster.declared[name] = state, declared

    def tile_of(t):
        return (1,) * (t.dim() - 2) + BLOCK

    for r in range(train_tp):
        name = f"NIXLTrainClient_{r}"
        fill_fp8_exact(cluster.states[name], cluster.declared[name], FP8_KEYS, tile_of, version=7)
        rest = {k: v for k, v in cluster.states[name].items() if k not in FP8_KEYS}
        fill_by_layout(rest, cluster.declared[name], version=7, wire_dtype=torch.bfloat16)

    from psrl.utils.nixl.server import NIXLMetaServer

    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in cluster.declared.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    from psrl.utils.nixl.handshake_notif import NotifKind

    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(cluster.declared)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        if kind is NIXLClientType.PUSH_SIDE:
            client.set_publish_cast(train_dtype, torch.bfloat16)
            client.set_publish_quantization(FP8_PLAN, reader_quantization=reader_quantization)
        elif reader_quantization:
            client.set_reader_quantization(FP8_PLAN)
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    if peer_diffusion:
        _train_ranks_remote(cluster.clients)
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = plan
    cluster.plan = plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=5.0) for n in cluster.gen
    }
    return cluster


def _judge_fp8(cluster, version, uniform=False):
    for name in cluster.gen:
        judged, failure = check_fp8_exact(
            cluster.states[name], cluster.declared[name], FP8_KEYS, BLOCK, version, uniform=uniform
        )
        assert failure is None, f"{name}: {failure}"
        assert judged == len(FP8_KEYS)
        rest = [k for k in MODEL if k not in FP8_KEYS]
        judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], rest, version=version)
        assert failure is None and judged == len(rest), (name, failure)


@pytest.mark.parametrize("train_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (4, 2), (2, 4), (1, 2)])
def test_a_trainer_publishes_block_fp8_with_scales(train_dtype, train_tp, gen_tp):
    cluster = _fp8_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, train_dtype=train_dtype)
    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (2, 16)])
def test_a_finer_rollout_scale_grid_gets_each_tile_scale_repeated(train_tp, gen_tp):
    # vLLM refines an FP8 MoE scale grid when a TP cut splits a tile. At gen TP 16 each rank holds
    # 16 of a 32 wide tile, which only the refined 16 x 16 grid can express.
    cluster = _fp8_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, scale_tile=(16, 16))
    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (4, 2), (1, 4)])
def test_an_fp8_trainer_publishes_block_fp8_from_its_per_tensor_bytes(train_tp, gen_tp):
    cluster = _fp8_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, train_fp8=True)
    for name in cluster.clients:
        if name.startswith("NIXLTrainClient"):
            assert all(cluster.states[name][k].dtype == torch.float8_e4m3fn for k in FP8_KEYS)
    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7, uniform=True)


def test_a_new_fp8_version_requantises_at_publish():
    cluster = _fp8_cluster(4, 2, instances=2, peer_diffusion=True)
    cluster.pull(epoch=1)

    def tile_of(t):
        return (1,) * (t.dim() - 2) + BLOCK

    for r in range(4):
        name = f"NIXLTrainClient_{r}"
        fill_fp8_exact(cluster.states[name], cluster.declared[name], FP8_KEYS, tile_of, version=9)
        rest = {k: v for k, v in cluster.states[name].items() if k not in FP8_KEYS}
        fill_by_layout(rest, cluster.declared[name], version=9, wire_dtype=torch.bfloat16)
        cluster.clients[name].publish_version(9)
    cluster.pull(epoch=2)
    _judge_fp8(cluster, version=9)


@pytest.mark.parametrize("waves", [1, 2, 7])
def test_any_wave_count_moves_the_same_bytes(waves):
    cluster = Cluster(train_tp=4, gen_tp=2, instances=3, peer_diffusion=True)
    for engine in cluster.engines.values():
        engine.waves = waves
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    model_bytes = sum(torch.Size(s).numel() * 2 for s, _ in MODEL.values())
    assert sum(r.upstream_bytes for r in reports.values()) == model_bytes


def _replicated_cluster(train_tp, dp, gen_tp, instances, peer_diffusion=False):
    """A cluster whose train ranks come in `dp` identical replicas, as Megatron's DP groups do."""
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states = {}, {}
    kinds = {}
    for d in range(dp):
        for r in range(train_tp):
            name = f"NIXLTrainClient_{d * train_tp + r}"
            kinds[name] = NIXLClientType.PUSH_SIDE
            cluster.states[name] = _locals(MODEL, train_tp)
            cluster.declared[name] = {k: _sharding(dim, train_tp, r) for k, (s, dim) in MODEL.items()}
            fill_by_layout(cluster.states[name], cluster.declared[name], version=7, wire_dtype=torch.bfloat16)
    for i in range(instances):
        for r in range(gen_tp):
            name = f"NIXLGenClient_I{i}_R{r}"
            kinds[name] = NIXLClientType.PULL_SIDE
            cluster.states[name] = _locals(MODEL, gen_tp)
            cluster.declared[name] = {k: _sharding(dim, gen_tp, r) for k, (s, dim) in MODEL.items()}

    unified = {name: copy.deepcopy(d) for name, d in cluster.declared.items()}
    for key in MODEL:
        finest = NIXLSharding.find_finest_shard_mesh([unified[n][key].shard_mesh for n in unified])
        for name in unified:
            unified[name][key].refactor_based_on_finer_shard_mesh(finest)

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    if peer_diffusion:
        _train_ranks_remote(cluster.clients)
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = plan
    cluster.plan = plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=2.0, read_timeout_s=2.0)
        for n in cluster.gen
    }
    return cluster


def _planned_source(cluster, train_tp):
    """A train rank some reader is planned to read from, with its DP twin."""
    for reader in cluster.gen:
        for per_source in cluster.plan.rollout_pull_from_ps_plan[reader].values():
            for source in per_source:
                rank = int(source.rsplit("_", 1)[1])
                twin = f"NIXLTrainClient_{(rank + train_tp) % (2 * train_tp)}"
                if twin in cluster.clients:
                    return source, twin
    raise AssertionError("no reader was planned against any train rank")


def _kill(cluster, sources):
    """Make every reader's transport fail for `sources`, as it does once a train rank is gone."""
    for name in cluster.gen:
        cluster.clients[name].agent.dead_sources.update(sources)


def test_a_known_dead_source_is_replaced_before_the_pull_starts():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=1)
    victim, _twin = _planned_source(cluster, train_tp=2)
    dead = {victim}
    _kill(cluster, dead)
    for engine in cluster.engines.values():
        engine.mark_unavailable(dead)
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    assert all(dead.isdisjoint(r.sources) for r in reports.values()), "a dead source still served bytes"
    assert sum(r.takeover_keys for r in reports.values()) > 0


def test_a_source_that_fails_mid_pull_is_taken_over_by_its_replica():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=1)
    victim, _twin = _planned_source(cluster, train_tp=2)
    _kill(cluster, {victim})  # not announced: the engine must learn it from the read
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)
    # Only the readers whose plan named that rank see it fail; they must all recover from a replica.
    hit = [r for r in reports.values() if r.failed_sources]
    assert hit, "no reader was planned against the failed source"
    assert all(r.failed_sources == {victim} and r.takeover_keys > 0 for r in hit)


def test_losing_a_whole_dp_group_is_refused_before_any_byte_moves():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=1)
    # Both replicas of one rank's shards: nothing holds those keys any more.
    victim, twin = _planned_source(cluster, train_tp=2)
    dead = {victim, twin}
    _kill(cluster, dead)
    for engine in cluster.engines.values():
        engine.mark_unavailable(dead)
    reader = next(
        r
        for r in cluster.gen
        if dead.intersection(src for per in cluster.plan.rollout_pull_from_ps_plan[r].values() for src in per)
    )
    before = {k: v.clone() for k, v in cluster.states[reader].items()}
    with pytest.raises(ChunkUnavailable):
        cluster.engines[reader].pull(epoch=1)
    for key, value in before.items():
        assert torch.equal(value, cluster.states[reader][key]), f"{key} was modified by a refused pull"


def test_a_designate_that_dies_sends_its_followers_upstream():
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=2, peer_diffusion=True)
    reports = cluster.pull(epoch=1)
    designates = {src for r in reports.values() for src in r.sources if src.startswith("NIXLGenClient")}
    assert designates, "no peer reads were planned"
    cluster.judge(cluster.gen, version=7)
    _kill(cluster, designates)
    reports = cluster.pull(epoch=2)
    cluster.judge(cluster.gen, version=7)
    assert sum(r.fallback_keys for r in reports.values()) > 0


def test_a_mid_pull_failure_is_promoted_to_unavailable_for_the_next_pull():
    # D7 production wiring: report.failed_sources from a mid-pull failure must be promoted
    # into mark_unavailable so the next pull skips the dead source rather than
    # re-discovering the failure through another timed-out read.
    #
    # This test validates the contract that nixl_pull_model_core relies on: calling
    # mark_unavailable(report.failed_sources) after a pull causes subsequent pulls to
    # skip the dead source entirely, without any reads against it.
    cluster = _replicated_cluster(train_tp=2, dp=2, gen_tp=2, instances=1)
    victim, _twin = _planned_source(cluster, train_tp=2)

    # Kill but do not announce: the engine learns the failure from the read itself.
    _kill(cluster, {victim})
    reports = cluster.pull(epoch=1)
    cluster.judge(cluster.gen, version=7)

    hit = [r for r in reports.values() if r.failed_sources]
    assert hit, "no reader was planned against the failed source"
    assert all(victim in r.failed_sources for r in hit)

    # Simulate what nixl_pull_model_core does: promote failed_sources to unavailable.
    # Only engines that were planned against the dead source encounter the failure.
    for name, report in reports.items():
        if report.failed_sources:
            cluster.engines[name].mark_unavailable(report.failed_sources)

    # Engines that encountered the failure must have the dead source in unavailable.
    readers_that_hit = [name for name, report in reports.items() if victim in report.failed_sources]
    assert readers_that_hit, "no engine recorded the failed source"
    for name in readers_that_hit:
        assert victim in cluster.engines[name].unavailable, (
            f"{name}: dead source {victim!r} not in unavailable after promotion"
        )

    # Epoch 2 must still succeed (served by the twin) and must not touch the dead source.
    reports2 = cluster.pull(epoch=2)
    cluster.judge(cluster.gen, version=7)
    assert all(victim not in r.sources for r in reports2.values()), (
        "dead source was used in epoch 2 despite being in unavailable"
    )


def _multi_role_cluster(roles, train_tp, gen_tp):
    """One cluster carrying several models at once, their keys named identically."""
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states, cluster.roles = {}, {}, {}
    kinds = {}
    for role in roles:
        for r in range(train_tp):
            name = train_client_name(r, role)
            kinds[name] = NIXLClientType.PUSH_SIDE
            cluster.states[name] = _locals(MODEL, train_tp)
            cluster.declared[name] = {k: _sharding(d, train_tp, r) for k, (s, d) in MODEL.items()}
            # Each model fills at its own version, so bytes of one are never valid for the other.
            fill_by_layout(
                cluster.states[name], cluster.declared[name], version=roles[role], wire_dtype=torch.bfloat16
            )
        for r in range(gen_tp):
            name = gen_client_name(0, r, role)
            kinds[name] = NIXLClientType.PULL_SIDE
            cluster.states[name] = _locals(MODEL, gen_tp)
            cluster.declared[name] = {k: _sharding(d, gen_tp, r) for k, (s, d) in MODEL.items()}
        cluster.roles[role] = [n for n in kinds if model_role_of(n) == role and n.startswith("NIXLGenClient")]

    # The real server unifies and plans, which is what must keep the models apart.
    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in cluster.declared.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(cluster.declared)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    server.client_infos = infos
    server._agents_reported[NotifKind.CLIENT_INFO] = set(infos)
    server._peer_diffusion = False
    server.comm_plan = None
    server.make_comm_plan()
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = server.comm_plan
    cluster.plan = server.comm_plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=2.0) for n in cluster.gen
    }
    return cluster


def test_two_models_publish_and_pull_without_reading_each_other():
    roles = {"actor": 7, "critic": 11}
    cluster = _multi_role_cluster(roles, train_tp=2, gen_tp=2)
    reports = cluster.pull(epoch=1)
    for role, version in roles.items():
        for name in cluster.roles[role]:
            judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], list(MODEL), version)
            assert failure is None, f"{name} ({role}): {failure}"
            assert judged == len(MODEL)
    # Every byte a reader took came from its own model's ranks.
    for name, report in reports.items():
        role = model_role_of(name)
        assert all(model_role_of(src) == role for src in report.sources), f"{name} read {sorted(report.sources)}"


def test_one_model_republishing_does_not_disturb_the_other():
    roles = {"actor": 7, "critic": 11}
    cluster = _multi_role_cluster(roles, train_tp=2, gen_tp=2)
    cluster.pull(epoch=1)
    for r in range(2):
        name = train_client_name(r, "actor")
        fill_by_layout(cluster.states[name], cluster.declared[name], version=9, wire_dtype=torch.bfloat16)
        cluster.clients[name].publish_version(9)
    cluster.pull(epoch=2)
    for role, version in {"actor": 9, "critic": 11}.items():
        for name in cluster.roles[role]:
            judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], list(MODEL), version)
            assert failure is None, f"{name} ({role}) after actor republished: {failure}"
            assert judged == len(MODEL)


def test_an_instance_that_joins_later_pulls_correct_bytes():
    """A rollout started after the handshake is admitted, replanned for, and reads the right bytes."""
    roles = {"actor": 7}
    cluster = _multi_role_cluster(roles, train_tp=2, gen_tp=2)
    cluster.pull(epoch=1)
    incumbents = list(cluster.gen)

    # A second instance registers after everyone else has been planned for.
    joiners = []
    for r in range(2):
        name = gen_client_name(1, r)
        cluster.states[name] = _locals(MODEL, 2)
        cluster.declared[name] = {k: _sharding(d, 2, r) for k, (s, d) in MODEL.items()}
        joiners.append(name)

    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in cluster.declared.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(cluster.declared)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    for name in joiners:
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        client.register_local_tensors(cluster.states[name], unified[name])
        cluster.clients[name] = client
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    server.client_infos = infos
    server._agents_reported[NotifKind.CLIENT_INFO] = set(infos)
    server._peer_diffusion = False
    server.comm_plan = None
    server.make_comm_plan()
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = server.comm_plan
    cluster.gen = incumbents + joiners
    cluster.engines.update(
        {n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=2.0) for n in joiners}
    )

    # The replan must cover the joiners and leave the incumbents able to pull as before.
    assert all(name in server.comm_plan.rollout_pull_from_ps_plan for name in joiners)
    cluster.pull(epoch=2, participants=set(cluster.gen))
    for name in cluster.gen:
        judged, failure = check_by_layout(cluster.states[name], cluster.declared[name], list(MODEL), version=7)
        assert failure is None, f"{name}: {failure}"
        assert judged == len(MODEL)


@pytest.mark.parametrize("train_tp,gen_tp", [(1, 2), (2, 2), (4, 2), (2, 4)])
def test_reader_quantization_pulls_bf16_and_quantizes_in_place(train_tp, gen_tp):
    """The trainer keeps BF16 and allocates no publish copy; the rollout quantizes."""
    cluster = _fp8_cluster(
        train_tp,
        gen_tp,
        instances=2,
        peer_diffusion=True,
        reader_quantization=True,
    )
    for rank in range(train_tp):
        trainer = cluster.clients[f"NIXLTrainClient_{rank}"]
        assert trainer.publish_copy_bytes == 0, trainer.publish_copy_bytes

    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)

    # A second version must replace every FP8 byte and scale, not just the weights.
    for rank in range(train_tp):
        name = f"NIXLTrainClient_{rank}"
        fill_fp8_exact(cluster.states[name], cluster.declared[name], FP8_KEYS, lambda t: (1,) * (t.dim() - 2) + BLOCK, version=8)
        rest = {k: v for k, v in cluster.states[name].items() if k not in FP8_KEYS}
        fill_by_layout(rest, cluster.declared[name], version=8, wire_dtype=torch.bfloat16)
        cluster.clients[name].publish_version(8)
    cluster.pull(epoch=2)
    _judge_fp8(cluster, version=8)


class _RegisteredDestinationQuantizer(VllmFp8BlockQuantizer):
    """Refuse the allocating path, so a pull must quantize into the registered views."""

    def __init__(self, plan: QuantizationPlan):
        super().__init__(plan)
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def quantize(self, source: torch.Tensor):
        raise AssertionError("quantized into a temporary tensor instead of the registered views.")

    def quantize_into(self, source, destination, scale):
        self.calls.append((destination, scale))
        super().quantize_into(source, destination, scale)


def _require_registered_destinations(cluster) -> None:
    """Swap every reader's quantizer for one that only accepts registered destinations."""
    for name in cluster.gen:
        cluster.clients[name]._quantizer = _RegisteredDestinationQuantizer(FP8_PLAN)


@pytest.mark.parametrize("train_tp,gen_tp", [(2, 2), (4, 2)])
def test_reader_quantization_quantizes_into_the_registered_fp8_views(train_tp, gen_tp):
    """No FP8 temporary and no second copy when the scale grid is the publish tile."""
    cluster = _fp8_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True, reader_quantization=True)
    _require_registered_destinations(cluster)

    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)

    for name in cluster.gen:
        mapping = cluster.clients[name].get_original_tensor_mapping()
        weights = {id(view) for (key, _), view in mapping.items() if not key.endswith("_scale_inv")}
        scales = {id(view) for (key, _), view in mapping.items() if key.endswith("_scale_inv")}
        calls = cluster.clients[name]._quantizer.calls
        assert calls, name
        assert all(id(destination) in weights for destination, _ in calls), name
        assert all(id(scale) in scales for _, scale in calls), name


def test_reader_quantization_writes_the_registered_weight_with_a_refined_scale_grid():
    """A finer rollout grid still quantizes the weight into its registered view.

    Gen TP 16 cuts the 32-row tile, so the coarse scale repeats past the registered
    grid and has to be cropped to the shard's own tile count.
    """
    cluster = _fp8_cluster(
        2, 16, instances=2, peer_diffusion=True, scale_tile=(16, 16), reader_quantization=True
    )
    _require_registered_destinations(cluster)

    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)

    for name in cluster.gen:
        mapping = cluster.clients[name].get_original_tensor_mapping()
        weights = {id(view) for (key, _), view in mapping.items() if not key.endswith("_scale_inv")}
        calls = cluster.clients[name]._quantizer.calls
        assert calls, name
        assert all(id(destination) in weights for destination, _ in calls), name


def test_reader_quantization_falls_back_when_the_registered_view_is_not_addressable(monkeypatch):
    """A destination the fused kernel cannot flatten still lands through `copy_`."""
    cluster = _fp8_cluster(2, 2, instances=2, peer_diffusion=True, reader_quantization=True)
    monkeypatch.setattr(
        VllmFp8BlockQuantizer,
        "can_quantize_into",
        lambda self, source, destination, scale: False,
    )

    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)


def test_reader_quantization_peer_diffusion_shares_training_reads_across_separate_instances():
    """Four separate TP1 rollout instances on one node share one designate per key.

    Each instance is its own NIXL client wanting the identical, unsharded tensor --
    the deployment shape measured on real hardware in the GPU validation run, where
    this grouping did not trigger. The planner is verified correct here; a live run
    is needed to confirm what differs in the actual deployment (see the "Measured
    status" note in docs/design/reader_quantization_impl.md).
    """
    cluster = _fp8_cluster(4, 1, instances=4, peer_diffusion=True, reader_quantization=True)
    reports = cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)
    assert sum(r.peer_keys for r in reports.values()) > 0
    assert all(r.followers or r.peer_keys for r in reports.values())


def test_reader_quantization_peer_diffusion_shares_training_reads():
    """One designate per key group reads BF16 and quantizes; followers read FP8."""
    cluster = _fp8_cluster(2, 2, instances=2, peer_diffusion=True, reader_quantization=True)
    reports = cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)
    # One designate per (key, shard-set) group. With gen TP2: three FP8 weights
    # have two shard sets each, and the norm has one => 7 upstream keys. Scales are derived.
    assert sum(r.upstream_keys for r in reports.values()) == 7
    assert sum(r.peer_keys for r in reports.values()) > 0


def test_reader_quantization_shares_across_different_gen_layouts_on_one_node():
    """Phase 2: a gen_tp=4 instance's one finest shard is reused by gen_tp=2 instances'
    matching half, even though the layouts' whole shard sets never match.

    train_tp=4 makes the finest mesh 4 shards per key. Three same-node layout groups
    (TP4, TP2, TP2) each read their own shards independently without peer diffusion --
    3x the model's worth of upstream bytes, one full copy per layout. With peer
    diffusion, every one of the 4 distinct finest shards is read upstream exactly once
    (by whichever rank, across any layout, is elected its designate) and shared with
    every other rank -- in any layout -- that also needs it: 1x, not 3x.
    """
    cluster = _fp8_cluster_mixed_layout(
        train_tp=4, layouts=[4, 2, 2], peer_diffusion=True, reader_quantization=True
    )
    reports = cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)
    assert sum(r.peer_keys for r in reports.values()) > 0
    # At least one coarser-layout (L1/L2, gen_tp=2) instance got part of a key shared
    # with a finer-layout (L0, gen_tp=4) or the other coarse layout's instance -- the
    # whole point of phase 2, since L1/L2's whole shard sets never match L0's.
    coarse_names = [n for n in cluster.gen if not n.startswith("NIXLGenClient_L0")]
    assert any(reports[n].peer_keys > 0 for n in coarse_names)

    def _single_layout_upstream_bytes(gen_tp):
        single = _fp8_cluster(4, gen_tp, instances=1, peer_diffusion=False, reader_quantization=True)
        return sum(r.upstream_bytes for r in single.pull(epoch=1).values())

    independent_bytes = sum(_single_layout_upstream_bytes(gt) for gt in [4, 2, 2])
    actual_bytes = sum(r.upstream_bytes for r in reports.values())
    assert actual_bytes < independent_bytes


@pytest.mark.parametrize("train_tp,gen_tp", [(1, 2), (2, 2), (4, 2), (2, 4)])
def test_reader_quantization_tiles_a_grouped_moe_shaped_key(monkeypatch, train_tp, gen_tp):
    """A 3-D grouped-expert tensor quantizes bit for bit, same as a dense 2-D one.

    `VllmFp8BlockQuantizer.tile_of` pads leading dims with 1, so each expert's
    trailing 2-D slice gets its own independent block tile -- quantizing a
    `(experts, out, in)` tensor is the same per-slice math as `(out, in)`, just
    repeated down an unsharded leading axis. This is the production reader-side
    path (`pull_engine._quantize_scatter` -> `quantize_into` -> `tile_of`), not
    the test fixture's own tiling helper, so it exercises the real code.
    """
    moe_key = "layers.0.experts.gate_up_proj.weight"
    model = dict(MODEL)
    model[moe_key] = ((4, HIDDEN, INTER), 2)
    monkeypatch.setattr(f"{__name__}.MODEL", model)
    monkeypatch.setattr(f"{__name__}.FP8_KEYS", FP8_KEYS + [moe_key])
    cluster = _fp8_cluster(train_tp, gen_tp, instances=1, peer_diffusion=True, reader_quantization=True)
    cluster.pull(epoch=1)
    _judge_fp8(cluster, version=7)


# --- A fused parameter whose checkpoint views cut a quantization tile ---------

#: Two 24-row halves of one 48-row parameter. Against BLOCK's 32-row tile, neither half
#: is a whole tile, so the rollout can only quantize them together.
UNIT_MEMBER_ROWS = 24
UNIT_MEMBERS = {
    "layers.0.gdn.in_proj_b.weight": (UNIT_MEMBER_ROWS, HIDDEN),
    "layers.0.gdn.in_proj_a.weight": (UNIT_MEMBER_ROWS, HIDDEN),
}
UNIT_ID = "layers.0.gdn.in_proj_ba.weight"
UNIT_SCALE = UNIT_ID + "_scale_inv"


def _unit_cluster(train_tp, gen_tp, instances=2, peer_diffusion=True):
    """A rollout whose fused parameter is quantized as one unit.

    The checkpoint stores `in_proj_b` and `in_proj_a` separately, so each wire view is
    half of the 64-row parameter and neither is a whole 128-row tile. The trainer keeps
    both BF16; the rollout assembles them and quantizes the parameter once.
    """
    cluster = Cluster.__new__(Cluster)
    cluster.bus = NotifBus()
    cluster.declared, cluster.states = {}, {}
    cluster.units, cluster.expected = {}, {}
    kinds = {}
    torch.manual_seed(11)
    global_b = torch.randn(UNIT_MEMBER_ROWS, HIDDEN).to(torch.bfloat16)
    global_a = torch.randn(UNIT_MEMBER_ROWS, HIDDEN).to(torch.bfloat16)
    trainer_rows = UNIT_MEMBER_ROWS // train_tp
    for r in range(train_tp):
        name = f"NIXLTrainClient_{r}"
        kinds[name] = NIXLClientType.PUSH_SIDE
        cluster.states[name] = {
            "layers.0.gdn.in_proj_b.weight": global_b[r * trainer_rows : (r + 1) * trainer_rows].clone(),
            "layers.0.gdn.in_proj_a.weight": global_a[r * trainer_rows : (r + 1) * trainer_rows].clone(),
        }
        cluster.declared[name] = {key: _sharding(0, train_tp, r) for key in UNIT_MEMBERS}
    gen_rows = UNIT_MEMBER_ROWS // gen_tp
    for i in range(instances):
        for r in range(gen_tp):
            name = f"NIXLGenClient_I{i}_R{r}"
            kinds[name] = NIXLClientType.PULL_SIDE
            fused = torch.zeros((2 * gen_rows, HIDDEN), dtype=torch.float8_e4m3fn)
            state = {
                "layers.0.gdn.in_proj_b.weight": fused[:gen_rows],
                "layers.0.gdn.in_proj_a.weight": fused[gen_rows:],
                UNIT_SCALE: torch.zeros(scale_shape(fused.shape, BLOCK), dtype=torch.float32),
            }
            declared = {key: _sharding(0, gen_tp, r) for key in UNIT_MEMBERS}
            for sharding in declared.values():
                sharding.quant_unit = UNIT_ID
            declared[UNIT_SCALE] = NIXLSharding.default()
            declared[UNIT_SCALE].quant_unit = UNIT_ID
            declared[UNIT_SCALE].local_only = True
            cluster.states[name], cluster.declared[name] = state, declared
            cluster.units[(i, r)] = fused
            lo, hi = r * gen_rows, (r + 1) * gen_rows
            cluster.expected[(i, r)] = torch.cat([global_b[lo:hi], global_a[lo:hi]])

    from psrl.utils.nixl.server import NIXLMetaServer

    cluster.clients = {}
    for name, kind in kinds.items():
        client = NIXLStorageClient(
            client_name=name,
            server_name="server",
            use_gpu=False,
            client_type=kind,
            nixl_config=CONFIG,
            binded_agent=MemcpyAgent(name, cluster.bus),
        )
        if kind is NIXLClientType.PUSH_SIDE:
            client.set_publish_quantization(FP8_PLAN, reader_quantization=True)
        else:
            client.set_reader_quantization(FP8_PLAN)
        cluster.clients[name] = client
    # The server plans over what each client announces, which leaves the unit's
    # reader-computed scale out of the plan.
    announced = {n: c.plan_sharding(cluster.declared[n]) for n, c in cluster.clients.items()}

    server = NIXLMetaServer.__new__(NIXLMetaServer)
    server.client_sharding_dicts = {n: copy.deepcopy(d) for n, d in announced.items()}
    server.client_unified_sharding_dicts, server.finest_shard_meshes = {}, {}
    server._agents_reported = {kind: set() for kind in NotifKind}
    server._agents_reported[NotifKind.SHARDING] = set(announced)
    server.make_unified_sharding()
    unified = server.client_unified_sharding_dicts

    for name, client in cluster.clients.items():
        client.register_local_tensors(cluster.states[name], unified[name])
    if peer_diffusion:
        _train_ranks_remote(cluster.clients)
    infos = {name: c.local_client_info for name, c in cluster.clients.items()}
    plan = CommunicationPlanner(peer_diffusion=peer_diffusion).make_comm_plan(infos)
    for client in cluster.clients.values():
        client._all_client_infos = dict(infos)
        client._comm_plan = plan
    cluster.plan = plan
    cluster.gen = [n for n in cluster.clients if n.startswith("NIXLGenClient")]
    cluster.engines = {
        n: NIXLPullEngine(cluster.clients[n], staging_bytes=1 << 16, peer_timeout_s=5.0) for n in cluster.gen
    }
    return cluster


def _judge_units(cluster):
    for (i, r), fused in cluster.units.items():
        name = f"NIXLGenClient_I{i}_R{r}"
        expected_q, expected_s = quantize_blockwise(cluster.expected[(i, r)], BLOCK)
        assert torch.equal(fused.view(torch.uint8), expected_q.view(torch.uint8)), name
        assert torch.equal(cluster.states[name][UNIT_SCALE], expected_s), name


@pytest.mark.parametrize("train_tp,gen_tp", [(1, 1), (4, 1), (2, 2), (4, 2), (1, 2), (4, 4)])
def test_reader_quantization_quantizes_a_fused_parameter_as_one_unit(train_tp, gen_tp):
    """A view that cuts a tile is quantized with its siblings, over the engine's own tile.

    The scale is derived from the whole parameter, so it equals what vLLM's online
    quantizer produces for the same local tensor, at every train/rollout TP split.
    """
    cluster = _unit_cluster(train_tp, gen_tp)
    cluster.pull(epoch=1)
    _judge_units(cluster)


def test_a_unit_member_is_read_upstream_by_every_replica():
    """A follower cannot read a designate's unit bytes: the designate quantizes at pull end."""
    cluster = _unit_cluster(4, 1, instances=2, peer_diffusion=True)
    reports = cluster.pull(epoch=1)
    _judge_units(cluster)
    assert all(report.peer_keys == 0 for report in reports.values())
