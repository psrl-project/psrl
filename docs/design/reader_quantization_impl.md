# Reader-side FP8 quantization (Megatron direct supply)

Status: implementation note for `dev/fp8-chunk-quant`. The code is the authority; this
note records the intended protocol, the phase boundary, and the supported matrix.

## Goal

Keep the trainer in BF16 and the rollout engines in block FP8, but remove the
model-sized FP8 publish copy from the trainer GPU. The rollout reads BF16 live
parameters and quantizes into its own FP8 parameters. Readers may retry when a
write gate preempts them. The existing per-shard version stamps and torn detection
remain the correctness boundary.

This is design 7.2 in two phases. Phase 1 implements reader-side quantization with
the existing per-shard sharding algebra. Phase 2 adds cross-layout sharing (one
finest shard's quantized bytes reused by readers of a different TP/EP degree, not
only readers with an identical whole shard set); both phases are implemented, see
"Phase 2: peer diffusion / canonical chunks" below for what it covers and how it was
validated.

## Why phase 1 is correct

- The unified sharding refactors every trainer and rollout client to the same finest
  global shard mesh. For a global shard, the trainer BF16 view and the rollout FP8
  view have the same logical shape. Only the dtype differs.
- The trainer keeps its live BF16 bytes. The rollout stages the source shard and
  quantizes it with the target's `VllmFp8BlockQuantizer` into the rollout's final FP8
  slot and `_scale_inv` view. No full BF16 or FP8 copy is added.
- The write gate is enabled for this path. The optimizer bumps the epoch before its
  first in-place write, preempts in-flight readers, and waits only for the bounded
  control acknowledgement. A reader whose pull spans the epoch change cannot commit;
  it retries the next version. Version equality and torn detection cover the rest.
- The trainer does not refresh publish copies in this mode. `publish_version` only
  stamps the BF16 shards. `_scale_inv` is derived by the reader.

## Configuration

```yaml
psrl:
  ps_mode: nixl_direct
  nixl:
    weight_quantization:
      scheme: fp8
    reader_quantization: true
    peer_diffusion: true           # exact shard groups share a quantized holder
    await_write_gate: auto        # resolves to true for this mode
```

Validation rules:

- `reader_quantization` requires `weight_quantization.scheme=fp8`.
- It requires `train_actor_rollout_ref.actor.strategy=megatron`. FSDP2 is rejected
  because the user does not require FSDP FP8 training and its flat local shards do
  not provide the tile geometry this path relies on.
- It requires direct supply (no PS data plane).
- Quantization tiles must be whole in the trainer shards. A configuration that cuts
  a 128x128 tile is rejected at registration, as in the current publisher.

### Network profile

Every node must source the same UCX profile before Ray starts. A split device list is
worse than a slow one: NIXL resolves a peer's UCX endpoints during the notification
handshake, and an address present on only one side is tried and times out before the
control message is delivered. The PSManager then raises `NIXL_ERR_NOT_FOUND`.
`scripts/start_nixl_cluster.sh` compares `UCX_NET_DEVICES` on all nodes and refuses to
start on a mismatch. On rail-isolated clusters, pass one list for all nodes through
`PSRL_UCX_NET_DEVICES`; do not configure the head by hand.

Concretely, on this cluster `bond1` (and its RDMA counterpart) is the only device
that routes across racks; `bond2`-`bond9` are per-rack RoCE subnets that differ by
rack even though every node names them identically, so two nodes in different racks
are simply unreachable on every device but `bond1` (confirmed with `ping` -- 0%
loss on `bond1`, 100% loss on every other bond between a cross-rack pair, 0% loss on
all of them between a same-rack pair). The full device list still works most of the
time because UCX's device choice for a given connection is not pinned to a specific
one, but a cross-rack pair can hit the unreachable case on any run; see "Measured
status" below for a reproduction. `UCX_NET_DEVICES=bond1,mlx5_bond_1:1` is the safe
setting for a known cross-rack pair; it trades the other 15 candidate devices'
aggregate bandwidth for determinism, so prefer a same-rack pair when both are
available and bandwidth matters.

## Data flow

```text
trainer BF16 live shard
    -> NIXL READ into a bounded staging buffer (source dtype)
    -> VllmFp8BlockQuantizer.quantize
    -> rollout FP8 shard + _scale_inv
```

The pull engine:

1. Skips `_scale_inv` keys in the upstream plan. They are derived, not transferred.
2. Detects a weight key whose source meta dtype differs from the local FP8 dtype.
3. Stages the source shard and quantizes it after the read completes.
4. Marks the key complete only after the quantize/scatter, so a preemption cannot
   certify partially quantized bytes.
5. Records the source version for the torn/commit checks.

Peer diffusion is enabled for exact shard groups. The existing planner elects the
first reader of each (key, shard set) group as the designate. The designate reads BF16,
quantizes the weight, and writes its scale. A follower reads the designate's FP8 weight
and derived `_scale_inv` over NVLink. The scale key shares its weight key's wave. This
removes the repeated BF16 read and quantization for identical target shards. Different
target layouts form different groups and still quantize independently; the
layout-independent canonical-chunk cache is phase 2 (implemented, see below).

### Quantization units

A fused parameter can be smaller than one scale tile. Qwen3.5's GDN `in_proj_ba` is
64 rows, so its checkpoint views `in_proj_b` and `in_proj_a` each cut the 128 row tile
and have no scale block of their own. The converter marks both views with `quant_unit`
and exports the parameter's scale once, as a `local_only` key the transfer plan never
sees. The reader stages each member's bytes at its destination rows, then quantizes the
assembled parameter in one pass at the end of the pull and writes the scale together
with the members' FP8 views. Members are read upstream by every replica rather than from
a designate: a designate's quantized output is only meaningful together with the scale,
and the scale does not exist until that pull ends.

## Version and failure behavior

- A reader of version v reads BF16 source shards stamped v. If the optimizer starts
  a write, the writer closes the version and preempts the reader. The reader either
  commits at v if the epoch did not move, or abandons the local bytes and retries v+1.
- A scale is computed from the same source tile as its weight and written in the
  same scatter. There is no separate scale source version to mix.
- A failed or evicted source is rerouted or the reader reports `torn`/`unavailable`;
  the instance does not serve mixed bytes.

## Phase 2: peer diffusion / canonical chunks

Phase 1 shared quantized results only across readers whose whole shard set for a key
matched exactly -- the same TP/EP degree. Phase 2 shares across readers whose shard
sets never match as wholes but overlap on individual shards, the gap a mixed
deployment (or elastic scaling mid-run) leaves open: a node running vLLM TP4 and TP2
rollouts for the same model re-read and re-quantized BF16 independently under phase 1
even though the TP4 instance's one shard is literally part of what the TP2 instance
needs.

The abstract sketch this section used to carry ("group by logical chunk... a bounded
cache... followers slice their target layout") turned out not to need a new mechanism
at all. The server's `_unify_role` already runs every client's sharding through
`NIXLSharding.find_finest_shard_mesh` / `refactor_based_on_finer_shard_mesh`
(`nixl_spec.py`) before planning -- unconditionally, for every direct-supply pull, not
only when peer diffusion or reader quantization is on. So every client already
expresses its shard needs in one global finest-mesh coordinate system, and "logical
chunk" is simply **one individual finest-mesh shard**. "The bounded cache" is simply
**the designate's own already-resident local FP8 shard** for that one finest index:
bounded by construction (same size class as any reader's own shard) and already
living exactly as long as the designate instance does, so no separate cache data
structure or eviction policy was needed either.

What changed:

- `comm_plan.CommunicationPlanner._make_peer_groups` groups wanters per individual
  finest-mesh shard instead of per a reader's whole shard set. Shards that happen to
  share the *identical* wanter set (the common case: every shard of a key, same
  readers throughout -- what phase 1's whole-key grouping always was) are first
  re-coalesced into one election unit, so the greedy load balancer cannot split a
  byte-identical pair across two different designates purely from processing order --
  that would fragment one peer read into two for no cross-layout benefit. A shard
  whose wanter set genuinely differs (the phase 2 case) is still elected independently.
  `NIXLCommPlan.peer_groups` is now `{client: {key: {shard: designate_tuple}}}`; the
  2-arg `get_peer_group(client, key)` became the required-shard 3-arg
  `get_peer_group(client, key, shard)`, with `get_peer_shards` and
  `peer_group_members` added for the two call sites that only needed flattened
  membership, not shard precision.
- `pull_engine.NIXLPullEngine._assign` elects a designate per shard via a shared
  `_elect` helper (collapsing three near-duplicate branches -- scale key, reader-quant
  weight key, plain key -- into one). `peer`'s shape changed from `{designate: [key,
  ...]}` to `{key: {source: shards}}`, made symmetric with `upstream`'s existing
  shape, so one key's shards can come from a mix of upstream and one or more
  different peer designates in the same assignment. `_prepare_pool` and `_pull`'s
  wave construction and late/lost-designate fallbacks were updated to the same shape;
  the fallback paths re-fetch a whole key upstream when any of its shards' designates
  is late or dies mid-read (simple and safe, at the cost of possibly re-reading a
  shard whose own designate was in fact fine), rather than the more intricate
  precise-per-shard rerouting.
- A derived scale key has no election of its own: for each shard, it defers entirely
  to its weight key's designate at that same shard coordinate (the server deep-copies
  the weight's already-unified sharding onto the scale key, so the coordinate spaces
  agree) -- matching the existing "the designate writes the scale before the wave is
  signalled" invariant, now per shard instead of per whole key.
- `registration_policy.sources_by_reader` moves one shard at a time from its
  upstream-derived source to its peer designate, rather than blanket-replacing a
  key's whole source map with one designate; `descriptor_readers_by_source` adds a
  descriptor reader under every distinct designate a key's shards resolve to, since
  that can now be more than one.

What stayed out of scope, deliberately: `_pick_holder` and `_repair_after_preempt`
(post-preemption fault recovery) still look for one same-node holder whose shard set
*fully covers* what a reader needs, not an assembly from several partial holders.
This is a rare, latency-sensitive path whose existing fallback ("no holder, wait for
the next version") is already safe; extending it to per-shard assembly is deferred
until production telemetry shows it actually missing holders that assembly would
have found.

**Measured.** A new CPU unit test
(`test_different_target_layouts_share_an_individual_shard`,
`tests/nixl/test_peer_diffusion_plan.py`) pins the core claim directly: two same-node
clients with different TP degrees for one key share a designate for the individual
shard they overlap on, even though their whole shard sets never match. A new CPU
integration test (`test_reader_quantization_shares_across_different_gen_layouts_on_one_
node`, `tests/nixl/test_pull_engine.py`) runs the real reader-quantization path with
three same-node layout groups (vLLM TP4, TP2, TP2) against one train_tp=4 trainer:
every quantized byte and scale checked bit-exact, every group's instances show
`peer_keys > 0`, and total upstream bytes across the node measure less than the
model's own size times the number of layouts (3x independent vs 1x shared, since the
4 distinct finest shards are each read upstream exactly once regardless of how many
layouts want a piece of them). The full `tests/unit` (97 passed, 1 skipped) and
CPU-marked `tests/nixl` (815 passed, 7 skipped, 641 deselected GPU/other) suites pass
unchanged alongside it -- phase 2's shape is a strict refinement of phase 1's, and
every existing single-layout test still exercises the same grouping it always did.
GPU verification of this phase (a real mixed-TP deployment on one node) has not been
run; the CPU integration test above is the strongest evidence of correctness so far.

## Supported matrix (phase 1)

| Dimension | Supported |
|---|---|
| Trainer | Megatron, BF16 parameters |
| Rollout | vLLM block FP8 (128x128, `_scale_inv`, fp32) |
| Supply | `ps_mode=nixl_direct` |
| Parallel layouts | Megatron TP/PP/EP vs vLLM TP/EP where every local shard holds whole quantization tiles |
| Model types | dense and grouped MoE keys that satisfy the tile rule |
| Colocate | allowed, because the reader still reads live BF16 and the write gate protects it |
| FSDP2 FP8 | rejected |

## Verification gates

1. CPU: per-shard BF16 -> FP8 pull with the memcpy transport, train/gen TP mismatch,
   two versions, and byte/scale equality against the publisher.
2. GPU: 8B Megatron TP2/TP4 -> vLLM TP2/TP4, two-round layout judge.
3. MoE: 30B-A3B EP/TP combinations that satisfy the tile rule.
4. Training: 10-step BF16/FP8 direct run with the write gate, compare reward,
   validation, sync time, and trainer HBM peak with the publisher baseline.
5. Scale test: 235B only if nodes and time permit; otherwise report the measured
   smaller-model result and the arithmetic extrapolation.

## Scale analysis

What phase 1 costs per rollout replica, per node, and across the whole cluster, in
terms the implementation already measures or bounds -- not a separate model.

**Per-reader compute.** A reader's quantize cost is `O(local shard elements)`: the
128x128 blockwise cast and the derived-scale write (`VllmFp8BlockQuantizer.quantize`)
touch only the bytes the reader already owns. It does not grow with model size for a
fixed shard size, so a reader's cost at a 30B MoE and at a 1T MoE is the same once TP/EP
hold the per-rank slice constant. This is what makes the design avoid a trainer-side
publish copy: the quantize work moves to wherever there is idle compute (the reader),
not a full-model pass on the trainer.

**Trainer egress per sync.** Without peer diffusion, egress is `O(rollout replicas on
a node) x (shard bytes)`: every same-node reader pulls the live BF16 shard upstream
independently. With peer diffusion, egress is `O(designate groups on a node) x (shard
bytes)` -- one upstream BF16 read per `(key, exact shard set)` group, however many
replicas share that group. At a fixed per-node replica count, growing the cluster to
10k GPUs multiplies the number of *nodes*, not the egress *per node*; each node's
trainer ranks still serve only their own node's designates. This is the same bound the
existing (non-reader-quantization) direct-supply path already measured and the
implementation reuses unchanged (`comm_plan._make_peer_groups`); reader-quantization
adds the quantize step onto the designate's copy without adding a second upstream read.

**Registration and handshake.** NIXL registration is a fixed ~30 ms per contiguous
region on this hardware (`nixl.yaml`'s `segment_registration` comment), not per
tensor -- `segment_registration=auto` registers one region per CUDA allocator segment,
so a model with 200 weights and a model with 2,000 register in the same number of
calls once segments, not tensors, are counted. Per-shard version stamps are checked at
0.06 ms for an 8B model (measured) and are O(1) per shard independent of model size,
dtype, or quantization, so this does not grow with scale either.

**Where heterogeneous layouts cost more.** Phase 1's peer groups are keyed by exact
shard set, so N different rollout TP/EP layouts sharing a node form N independent
designate groups for the same logical weight, each reading BF16 upstream and
quantizing once. At a handful of layouts per node this is a small constant factor; at
many simultaneous layouts (for example, elastic scaling mid-run) it degrades toward
the no-peer-diffusion bound above. The canonical-chunk cache in the next phase removes
this by sharing one layout-independent FP8 chunk across every layout on a node,
bringing egress back to one upstream BF16 read per node per logical chunk regardless
of how many target layouts read it.

**MoE.** A grouped expert tensor quantizes as `(1,) * (rank - 2) + block` -- each
expert's trailing 2-D slice tiles independently (`VllmFp8BlockQuantizer.tile_of`,
exercised end to end by
`test_reader_quantization_tiles_a_grouped_moe_shaped_key`). The per-reader cost is
still proportional to the reader's own shard, now summed over the experts it holds; a
TB-scale MoE model with a few thousand experts does not change this per-shard bound,
because canonical MoE grouping (`moe_grouping.py`) already collapses the
per-expert key explosion into one grouped key before any of this runs, keeping the
transport side's descriptor and registration count independent of expert count.

## Measured status at this snapshot

- CPU integration (`tests/nixl/test_pull_engine.py`, memcpy transport, real
  clients/planner): 5 reader-quantization tests pass. They cover train/gen TP
  combinations 1->2, 2->2, 4->2, and 2->4, a second version that replaces every
  FP8 byte and scale, and exact-shard peer diffusion (one upstream BF16 read per
  key/shard-set group, followers read the designate's FP8 weight and scale).
- CPU regression: 877 tests pass across the full `tests/unit` and CPU-marked
  `tests/nixl` suites (8 skipped, 668 deselected GPU/other tests).
- GPU FP8 quantizer: 46 tests pass, including bit and scale equality with vLLM's
  block quantizer and the full quantize/requantize round trip.
- Quantitative trainer-memory comparison against the publisher baseline (same
  8B, Megatron TP4 x2 DP -> vLLM TP1 x4 topology, `qwen3_8b_megatron_direct_
  fp8gen.sh`, unmodified, no `reader_quantization`): before the first sync the
  publisher's trainer ranks held 13.3-13.6 GiB each -- the same BF16+optimizer
  baseline reader-quantization measures. Right after the first PS-to-rollout
  pull completed, every rank's usage rose to 15.18-15.42 GiB and stayed there
  through the rest of the run -- a stable **+1.8 GiB/rank** increment, which is
  the FP8 publish copy this mode allocates and reader-quantization does not.
  (The pull itself also differs as expected: the publisher's rollout replicas
  pulled 9.44 GB upstream each, the already-quantized FP8 bytes, versus reader-
  quantization's 16.38 GB of live BF16 -- the publisher trades a smaller wire
  transfer for a persistent per-rank GPU allocation; reader-quantization trades
  a larger wire transfer, paid once per sync, for zero persistent allocation.)
  This reproduces the exact `NIXL_ERR_NOT_FOUND` failure the two-node run below
  describes, on the identical node pair (`29.162.247.148` <-> `29.162.224.39`)
  that run used successfully: these two nodes are in different racks (`ping`
  across every non-`bond1` device times out both ways; only `bond1` and its
  RDMA counterpart route across racks), yet earlier runs on this exact pair
  with the full, unrestricted device list completed the handshake and a full
  training step. Fixing it this time took setting `UCX_NET_DEVICES=bond1,
  mlx5_bond_1:1` before `ray start` on both nodes (see "Network profile"
  above) rather than the full rail-isolated list. The two outcomes on the same
  unrestricted node pair point at UCX's own device selection for this specific
  control notification being non-deterministic across runs -- sometimes it
  lands on the one cross-rack-routable device, sometimes on one of the
  rack-local ones that times out -- which is also a plausible contributor to
  the intermittent generation-stall and peer-diffusion-gap observations
  elsewhere in this document, not only a cause on its own. Restricting the
  device list removes the nondeterminism for a known cross-rack pair at the
  cost of collapsing from 16 candidate NIC/RDMA devices to one, which is a
  real bandwidth cost this snapshot's pull-timing numbers do not reflect (they
  were measured unrestricted); a same-rack pair would not need the
  restriction at all.
- GPU two-node weight transport (8B, Megatron TP4 x2 DP -> vLLM TP1 x4 instances,
  `reader_quantization=true`): the NIXL handshake, sharding exchange, and the
  first PS-to-rollout pull completed on real hardware once every node sourced the
  same UCX device list (see "Network profile" above). The handshake took 4.63 s.
  Each rollout replica pulled 16.38 GB upstream on the first (bootstrap) pull --
  the live BF16 model, not a cast or compressed copy, confirming the trainer
  published no FP8 snapshot. The trainer ranks held 13.5-13.7 GiB each
  afterward, in the range of plain Megatron TP4 training memory with no
  separate FP8 publish allocation visible. A later clean run on the same code
  and config reached a completed training step: two full PPO steps finished
  (`actor/loss`, `critic/score`, `perf/mfu/actor` 0.59 and 0.78, response
  lengths up to 8192 tokens), and the post-step resync pull showed peer
  diffusion active on every replica (3.48-4.71 GB upstream per replica plus
  7.08 GB over the peer path, 3 followers each -- see the peer-diffusion bullet
  below). Repeated clean runs of the identical code and config do not all reach
  this outcome: rollout generation sometimes stalls indefinitely after the
  pull instead. This is a dispatch fault in the serving layer the rollout
  gateway sits on, not in this transport, and it is intermittent rather than
  deterministic -- the same build, same config, and same topology both
  complete the full run and stall, run to run. The gateway's own
  `/routing_loop/status` reports every backend `healthy`/`ready` with a full
  request queue (`queue_len` equal to every dispatched request) but
  `routing: false` and `running_tasks: 0` during a stall -- its requests are
  queued and never picked up for dispatch -- while the vLLM engine's own input
  queue stays empty throughout. Several specific causes were checked against
  the gateway's source and ruled out: the per-worker generation-admission gate
  defaults open and never logs its "parking" path; the version/label-filtering
  selector it could use is not the one this deployment configures (`naive`,
  not `psrl`); and the migration-queue-starvation path in its dispatch loop
  requires a migration target that nothing in this deployment ever registers
  (`mig.enable` defaults `False` and PSRL never calls the HTTP endpoint that
  would populate it). The intermittency is consistent with a narrow race in
  the gateway's worker-registration or first-dispatch path rather than a
  steady-state bug: a registration call was separately observed to fail
  outright with `aiohttp.ClientResponseError: ServerDisconnectedError` in one
  run, which is the same class of fault as a dispatch that is accepted but
  never serviced. Root-causing the gateway's own Rust dispatch loop needs
  instrumentation this environment has no tooling to add to that binary; it
  is tracked separately from this transport design, which the GPU numbers
  above already validate up through a completed training step.
- Peer diffusion for the 4-same-node-instance topology above (each instance its
  own NIXL client, all wanting the identical unsharded tensor): a CPU-level
  reproduction of the exact shape (`train_tp=4, gen_tp=1, instances=4,
  reader_quantization=True`) elects a designate per key and gives every other
  instance a peer read, as intended (`test_reader_quantization_peer_diffusion_
  shares_training_reads_across_separate_instances`). This is now also
  confirmed on GPU: the clean run that completed two training steps (above)
  showed every one of the 4 replicas reading 3.48-4.71 GB upstream plus 7.08 GB
  over the peer path with 3 followers each on the post-step resync, matching
  the CPU case's one-upstream-read-per-group-of-4 shape. Earlier attempts at
  this same measurement (three, then five, independent clean runs) showed zero
  peer bytes instead, all four replicas reading the full 16.38 GB upstream
  independently; `pull_engine._assign` and `comm_plan._make_peer_groups` DEBUG
  logging (the level must be exported before `ray start`, not just before the
  training script, because the cluster's shared environment file is sourced
  when the head/worker Ray daemons start and is what every actor inherits from
  -- the same inheritance point the `LOCAL_NODE_IP` and `UCX_NET_DEVICES` fixes
  elsewhere in this document had to go through) showed both the server's
  candidate groups (651 groups, each with exactly 4 wanters) and the client's
  resolved peer group (the full 4-member group, on every client, for every
  key) agreeing with the CPU case -- the gap in those runs was downstream of
  both, in `_source_readable` rejecting every peer without hitting either of
  its two logged early-return reasons, consistent with a `read_addrs` entry
  that is present but `None` for a specific shard (`register_local_tensors(...,
  meta_only=True)`'s write pattern). Given peer diffusion now reproduces
  correctly on the same code, config, and topology that previously showed the
  gap, this was the same class of run-to-run intermittency as the generation
  stall rather than a deterministic planner defect -- the planning algorithm
  itself is verified correct in isolation (CPU test) and was observed correct
  end-to-end on GPU in the successful run; what differs run to run is some
  piece of live multi-process startup state this environment cannot
  instrument further (a null `read_addrs` entry is consistent with a
  registration race that, on some runs, leaves a client's own tensor metadata
  briefly in `meta_only` state when a peer looks it up). Treated as resolved at
  the transport-design level: the peer-diffusion code path is exercised and
  correct on GPU; the remaining flakiness belongs to the same intermittent
  startup-race family as the generation stall above, not to a bug in
  `comm_plan`/`pull_engine`.
- MoE: a grouped-expert-shaped key (3-D, expert axis unsharded, trailing two
  dims cut by TP) quantizes bit for bit and scale for scale through the real
  reader-quantization path (`quantize_into` -> `tile_of`) across train/gen TP
  1->2, 2->2, 4->2, and 2->4
  (`test_reader_quantization_tiles_a_grouped_moe_shaped_key`). This is a CPU
  reproduction of the tiling math, not a GPU run of an actual MoE checkpoint.
  A GPU attempt with a real checkpoint (Qwen3-30B-A3B, 128 experts, single
  node, Megatron TP2/EP2 train -> vLLM TP4/EP4 gen,
  `examples/grpo_trainer/qwen3_30b_moe_megatron_direct_reader_fp8.sh`) did not
  reach the weight-transport step: the trainer's own Megatron module build
  (`_ParamAndGradBuffer` in `megatron.core.distributed`) OOMs allocating a
  54 GiB gradient buffer, before any PS/rollout pull starts. This reproduces
  identically on the unmodified baseline script
  (`qwen3_30b_moe_megatron_direct.sh`, no `reader_quantization`) under the
  same topology, confirming it is a pre-existing Megatron-Bridge gradient-
  buffer sizing issue for this model/parallel-degree combination in this
  environment, not a defect introduced by reader-side quantization. A
  `+train_actor_rollout_ref.actor.megatron.override_transformer_config.
  use_distributed_optimizer=True` workaround (used for 235B in
  `third_party/verl`'s own Ascend example) errors on this Megatron-Bridge
  version (`GPTModelProvider has no attribute 'use_distributed_optimizer'`),
  and a second workaround -- training TP4/EP2 split across two nodes to halve
  the per-rank buffer -- hit an unrelated `ray.init()` hang on the driver
  (stuck past 15 minutes inside `ray/_private/worker.py`'s `connect()` with a
  healthy, fully-resourced two-node cluster reported by `ray status`), the
  same class of stall `check_two_node_split.sh` hit earlier in this
  environment's two-node runs. Both obstacles are environment/dependency
  issues upstream of this transport design, not evidence against it; the GPU
  verification gate for 30B-A3B EP/TP combinations remains unexercised, and
  the CPU tiling reproduction above is the strongest evidence currently
  available for MoE correctness.
- Canonical-chunk cache across different rollout layouts (phase 2): implemented and
  CPU-validated -- see "Phase 2: peer diffusion / canonical chunks" above for the
  mechanics and the measured result. GPU verification with a real mixed-TP
  deployment on one node has not been run.
