# Configuration

PSRL uses [Hydra](https://hydra.cc/) with [OmegaConf](https://omegaconf.readthedocs.io/)
for hierarchical, composable configuration management.

## Configuration System

All configuration lives under `psrl/trainer/config/`. Hydra composes a single merged
config from multiple YAML files at runtime, allowing you to:

- Override individual parameters from the command line
- Swap entire config groups (e.g., switch from FSDP to Megatron backend)
- Use variable interpolation (e.g., `${psrl.staleness}`) across config files

### File Layout of the `psrl` Group

`psrl/psrl.yaml` is a Hydra defaults list that composes one file per feature. All of
them live in `psrl/trainer/config/psrl/`:

| File | Merged at | Covers |
|---|---|---|
| `psrl.yaml` | `psrl` | Core settings and the defaults list |
| `deployment.yaml` | `psrl.deployment` | Cluster sizing, heterogeneous rollout, elastic RM |
| `fine_grain_overlap.yaml` | `psrl.fine_grain_overlap` | Rollout/training compute overlap |
| `agentic_rl.yaml` | `psrl.agentic_rl` | Multi-turn agent loop toggles |
| `server_rollout.yaml` | `psrl.server_rollout` | Optional external HTTP rollout service |
| `profile.yaml` | `psrl.profile` | Profiling/analysis switches |
| `log_prob.yaml` | `psrl.log_prob` | Rollout-engine log-prob toggle |
| `status_collection.yaml` | `psrl.status_collection` | Engine status telemetry |
| `memory_logger.yaml` | `psrl.memory_logger` | Periodic GPU memory logging |
| `nixl.yaml` | `psrl.nixl` | RDMA weight-sync transport |
| `broadcast_init.yaml` | `psrl.broadcast_init` | Rank-0 checkpoint broadcast |
| `checkpoint.yaml` | `psrl.checkpoint` | Megatron save/load strategy |
| `group_post_process.yaml` | `psrl.group_post_process` | Streaming group post-processing |
| `buffer_post_process.yaml` | `psrl.buffer_post_process` | Batch buffer post-processing |
| `tms.yaml` | `psrl.tms` | torch_memory_saver scope |
| `lmcache.yaml` | `psrl.lmcache` | KV offloading and P2P transfer |
| `rollout_gateway.yaml` | `psrl.rollout_gateway` | SMG gateway process |
| `rollout_coordination/*.yaml` | `psrl.rollout_coordination.*` | Online generation coordination (six files) |

Each file is merged into the node shown above, so a field's key path is that node
plus the field name. For example, `enable` in `lmcache.yaml` is addressed as
`psrl.lmcache.enable`.

## Top-Level Config

The entry point is `psrl/trainer/config/ppo_trainer.yaml`, which composes the following
groups via Hydra defaults:

| Group | Config Key | Config Path | Description |
|---|---|---|---|
| `psrl` | `psrl` | `psrl/psrl.yaml` | PSRL-specific settings (staleness, deployment, routing) |
| `model_engine` | Hydra selection | `dp` by default | Selects DP/FSDP-compatible or Megatron component groups |
| `actor` | `train_actor_rollout_ref.actor` | `${model_engine}_actor` (veRL) | Actor model training config |
| `rollout` (train) | `train_actor_rollout_ref.rollout` | `rollout/psrl_rollout.yaml` | PSRL-extended validation/training-side rollout config |
| `rollout` (gen) | `gen_actor_rollout_ref.rollout` | `rollout/psrl_rollout.yaml` | PSRL-extended generation-cluster rollout config |
| `data` | `data` | `data/data.yaml` | Dataset and dataloader config |
| `ref` | `train_actor_rollout_ref.ref` | `${model_engine}_ref` (veRL) | Reference model config |
| `model` (train) | `train_actor_rollout_ref.model` | `model/hf_model.yaml` (veRL) | HuggingFace model loading config |
| `critic` | `critic` | `${model_engine}_critic` (veRL) | Critic model config. Its `model` sub-tree is `model/hf_model.yaml` |
| `reward` | `reward` | `reward/reward.yaml` | Reward model and reward-manager config |
| `algorithm` | `algorithm` | *(inline in ppo_trainer.yaml)* | Algorithm hyperparameters (PPO/GRPO/DAPO) |
| `algorithm` (correction) | `algorithm.rollout_correction` | `algorithm/rollout_correction.yaml` (veRL) | Rollout importance-sampling correction |
| `distillation` | `distillation` | `distillation/distillation.yaml` (veRL) | On-policy distillation config |
| `trainer` | `trainer` | *(inline in ppo_trainer.yaml)* | Training loop settings (epochs, logging, checkpoints) |
| `transfer_queue` | `transfer_queue` | *(inline in ppo_trainer.yaml)* | Sample storage and transport backend |

The generation cluster does not load its own model group: `gen_actor_rollout_ref.model`
is an interpolation of `train_actor_rollout_ref.model`, so both clusters always agree on
the HuggingFace model definition.

Groups marked *(veRL)* are not duplicated in this repository. A Hydra search-path
plugin appends `pkg://verl.trainer.config` as a fallback, so any group PSRL does not
define locally resolves against the installed veRL package. Dropping a same-named
YAML into PSRL's own `config/` directory overrides the veRL version. For Megatron
training, pass `model_engine=megatron`.

:::{note}
The **rollout (train)** group, `train_actor_rollout_ref.rollout`, exists only to
stay aligned with veRL's config layout. Except for the handful of parameters that
affect **validation** and **recompute log-prob** on the training cluster, its fields
are unused and act as a placeholder. In practice, keep it aligned with
`gen_actor_rollout_ref.rollout` (the generation-cluster rollout config that actually
drives online generation). This config split is a known rough edge that we plan to
streamline in a future release.
:::

## veRL-Managed Configuration

Most of the config groups above (`train_actor_rollout_ref`, `data`, `algorithm`,
`trainer`, `rollout`, `critic`, `reward_model`, etc.) are inherited directly from
[veRL](https://github.com/volcengine/verl) with minimal changes.

:::{tip}
For the full reference of veRL-managed config groups, including actor training
hyperparameters, optimizer settings, rollout sampling parameters, data loading,
algorithm coefficients, and trainer loop settings, refer to the official veRL
configuration documentation:

**→ [veRL Configuration Reference](https://verl.readthedocs.io/en/latest/examples/config.html)**
:::

Given the note above, the PSRL-introduced surface is small: only **two** parts of the
tree are genuinely new. The `psrl` config group (described below) and the
`gen_actor_rollout_ref` sub-tree that configures the decoupled generation cluster
separately from the training cluster. Everything else mirrors veRL.

---

## PSRL Config Reference

The primary PSRL configuration file is `psrl/trainer/config/psrl/psrl.yaml`. Below is a
categorized reference of all parameter groups.

### Core Settings

`ps_manager_ip`
: IP address of the Parameter Server Manager process. All inter-component communication
  (NIXL, LMCache Controller, reward service) defaults to this address.
  **Default:** `127.0.0.1`

`reward_service_ip`
: IP address of the reward scoring service.
  **Default:** `${psrl.ps_manager_ip}`

`logging_path`
: Base directory for all PSRL log outputs (trajectory dumps, profiling files, etc.).
  **Default:** `~/psrl_logs`

`staleness`
: Maximum version gap between generation and training. `0` = fully synchronous
  (generation blocks until training consumes). Values `>0` allow that many rollout
  buffers to be generated ahead of training consumption.
  **Default:** `0`

`staleness_buffer_entries`
: Number of prompts in each staleness buffer (effective training batch size).
  Each buffer must be fully filled before it can be consumed by training.
  **Default:** `512`

`rollout_n`
: Number of responses generated per prompt. Set `>1` for GRPO/DAPO group sampling
  (e.g., 8).
  **Default:** `1`

`ps_mode`
: Parameter server weight synchronization mode.
  - `cpu_ref`: CPU-based reference model (simpler setup, no NIXL required)
  - `nixl_cpu`: GPU-direct RDMA via NIXL (recommended for production)

  **Default:** `cpu_ref`

`retry_bound`
: Buffer bound for the retry mechanism. `-1` means unbounded.
  **Default:** `-1`

`retry_ratio`
: Ratio applied on top of `retry_bound`.
  **Default:** `1.0`

:::{note}
`retry_bound` and `retry_ratio` are declared placeholders: no code path reads them
today. They are reserved for the buffer retry mechanism and setting them has no
effect on the current training loop.
:::

### Fine-Grain Overlap

Overlaps training-side compute with ongoing rollout generation by releasing training
buffers in smaller chunks instead of waiting for a full global batch to accumulate.
When enabled, the trainer starts running per-sample forward passes (old log-prob,
reference log-prob, values, reward) on completed prompt groups as they arrive, so the
per-sample GPU work for chunk N runs while rollout continues generating chunk N+1.

`fine_grain_overlap.granularity`
: Granularity of each training chunk.

  - `none`: disabled. The trainer waits for the full global batch before starting any
    work.
  - `mini_batch`: each chunk contains one PPO mini-batch worth of prompt groups
    (`ppo_mini_batch_size` prompts, times `rollout_n` trajectories each). Set
    `multiplier` to combine multiple mini-batches into one chunk.
  - `micro_batch`: each chunk is derived from `ppo_micro_batch_size_per_gpu * dp_size`
    samples. Only `overlap_scope: recompute` is supported at this granularity. The
    `pre_step` scope for micro-batch overlap is not yet implemented.

  When `multiplier` causes the computed chunk size to exceed the next level (a
  micro-batch chunk exceeding one mini-batch, or a mini-batch chunk exceeding the full
  batch), the granularity is automatically clamped to that level.

  **Default:** `none`

`fine_grain_overlap.multiplier`
: Multiplier applied to the base chunk size. The effective chunk is
  `base_unit_samples * multiplier`, clamped at the full global batch.
  **Default:** `1`

`fine_grain_overlap.overlap_scope`
: What computation runs inside each chunk.

  - `recompute`: only the per-sample forward stages (old log-prob, reference log-prob,
    values, reward) run per chunk. Advantage computation and the optimizer update run
    once on the full concatenated batch after all chunks arrive. The training math is
    numerically identical to `granularity: none` for all advantage estimators.
  - `pre_step`: in addition to per-sample stages, advantage and one optimizer step
    also run per chunk. Each chunk with `granularity: mini_batch` becomes a complete
    PPO mini-batch update. This scope requires `ppo_epochs: 1`. With
    `adv_estimator: grpo`, results are exact because GRPO normalizes per prompt group.
    With `adv_estimator: gae` or `reinforce_plus_plus`, the per-chunk whitening scope
    differs from the full-batch scope and results are approximate.

  **Default:** `recompute`

:::{note}
`overlap_scope: pre_step` is only supported with `granularity: mini_batch`.
Combining `pre_step` with `granularity: micro_batch` raises a `ValueError` at startup.
:::

**Compatibility constraints enforced at startup:**

- `overlap_scope: pre_step` requires `ppo_epochs: 1`.
- `granularity: micro_batch` with `use_dynamic_bsz: True` raises an error because the
  chunk size cannot be determined without a static `ppo_micro_batch_size_per_gpu`.
- `granularity: micro_batch` with `overlap_scope: pre_step` and
  `actor.strategy: megatron` is not supported.

**Example: recompute scope (safe starting point)**

```yaml
fine_grain_overlap:
  granularity: mini_batch
  multiplier: 1
  overlap_scope: recompute
```

**Example: pre_step scope with GRPO (maximum overlap)**

```yaml
fine_grain_overlap:
  granularity: mini_batch
  multiplier: 1
  overlap_scope: pre_step
# also set:
# train_actor_rollout_ref.actor.ppo_epochs: 1
# algorithm.adv_estimator: grpo
```

### Deployment

Resource allocation for training and rollout clusters.

`deployment.n_rollout_instances`
: Number of independent rollout (generation) vLLM instances.
  **Default:** `1`

`deployment.n_validate_instances`
: Number of validation rollout instances (typically colocated with training).
  **Default:** `1`

`deployment.rollout_nnodes_per_instance`
: Nodes allocated to each rollout instance.
  **Default:** `1`

`deployment.rollout_ngpus_per_node_per_instance`
: GPUs per node for each rollout instance.
  **Default:** `1`

`deployment.validate_nnodes_per_instance`
: Nodes allocated to each validation instance.
  **Default:** `1`

`deployment.validate_ngpus_per_node_per_instance`
: GPUs per node for each validation instance.
  **Default:** `1`

`deployment.train_nnodes`
: Nodes allocated to the training cluster.
  **Default:** `1`

`deployment.train_ngpus_per_node`
: GPUs per node in the training cluster.
  **Default:** `1`

`deployment.total_nnodes`
: Total nodes in the job. When set, excess nodes are blocked from scheduling to
  prevent colocated validate workers from spilling onto idle nodes. Set to match
  `NNODES` in your launch script.
  **Default:** `null`

`deployment.heterogeneous_rollout`
: Enable per-instance configuration of rollout resources. When `enable: True`, each
  rollout instance can be individually configured:

  | Sub-field | Description |
  |---|---|
  | `enable` | Master switch. **Default:** `False` |
  | `n_rollout_instances` | Mirrors `psrl.deployment.n_rollout_instances` |
  | `rollout_nnodes_per_instance` | List of per-instance node counts (length = `n_rollout_instances`) |
  | `rollout_ngpus_per_node_per_instance` | List of per-instance GPU counts |
  | `tensor_model_parallel_size_per_instance` | List of per-instance TP sizes |
  | `pipeline_model_parallel_size_per_instance` | List of per-instance PP sizes |

`deployment.elastic_rm`
: Policy-driven resource sharing between rollout and named generative reward-model
  instances. The `ElasticExecutor` monitor loop samples per-instance KV-cache
  utilization and queue depth, then sleeps and wakes whole inference instances
  through their coordinators so the two roles can trade GPUs at runtime.

  **Resource pool and monitor loop**

  | Sub-field | Description | Default |
  |---|---|---|
  | `enable` | Master switch for rollout/reward auto-scaling. | `False` |
  | `shared_nnodes` | Nodes in the shared resource pool. | `1` |
  | `shared_ngpus_per_node` | GPUs per node in the shared pool. | `8` |
  | `enable_policy` | Let the monitor loop make policy-driven scaling decisions. With `False` the executor only reports state. | `True` |
  | `monitor_interval_ms` | Monitor loop tick interval. | `1000` |
  | `coordinator_sync_timeout_s` | Per-tick cap on coordinator Ray RPCs (engine snapshot + router backlog), so a blocked coordinator cannot silently hang the monitor. `<= 0` disables the cap. | `10` |
  | `coordinator_command_timeout_s` | Per-command cap on each `SLEEP`/`WAKE_UP`/`ABORT` sent to a coordinator. `0` waits indefinitely. | `300` |
  | `decision_execution_abandon_stall_ticks` | Consecutive ticks blocked on an unfinished scale decision before the in-flight state is cleared so new decisions can run. `0` never abandons. Wall time is roughly `ticks * monitor_interval_ms / 1000`. | `180` |

  **Scaling policy**

  | Sub-field | Description | Default |
  |---|---|---|
  | `theta_low` | KV-cache utilization below which an instance may cede its GPUs. | `0.1` |
  | `theta_max` | KV-cache utilization above which an instance counts as full-load. | `0.8` |
  | `full_load_mode` | Whether `all` instances of a role must be full-load before scaling up, or `any` single one suffices. | `any` |
  | `cooldown_ms` | Cooldown after each scaling action, to damp oscillation. | `10000` |
  | `hysteresis` | Minimum bottleneck-throughput gain required to justify a one-step transfer. | `0.05` |
  | `min_awake_per_role` | Instances each role always keeps awake. `0` lets one side sleep entirely. | `1` |
  | `max_waiting_queue_for_scale_down` | Scale-down guard: even below `theta_low`, do not shrink a candidate whose local waiting queue exceeds this. `0` requires an empty queue. | `64` |
  | `post_scale_up_abort_waiting_ratio` | Fraction `[0, 1]` of each instance's waiting queue aborted after a scale-up so the freshly woken instance can pick the work up. Aborts are FIFO. `0` disables and `1` aborts all. | `0.8` |
  | `lambda_ewma_alpha` | EWMA smoothing factor for the arrival-rate estimate. | `0.2` |

  **Throughput model**

  | Sub-field | Description | Default |
  |---|---|---|
  | `throughput_model_dir` | Directory holding `{model_name}_token.json` fitted throughput formulas. | `psrl/config/throughput_model` |
  | `throughput_model_output_len` | Output-length bucket to read from the token-throughput fit. | `1024` |
  | `profile_paths` | Optional `{model_name: profile_json_path}` map using the newer profile schema. | `{}` |

  The policy resolves throughput in priority order: fitted formula file, then
  `profile_paths` entry, then the measured runtime `generation_throughput`.

### Colocate & Fuse Settings

`colocate_validate_and_train`
: Whether to colocate validation and training workers on the same nodes.
  **Default:** `True`

`fuse_rollout_with_validate`
: Whether to dispatch validation requests to the rollout instance pool as well,
  effectively using the rollout instances as extra validation capacity (validation
  and rollout share one instance pool instead of validation having a dedicated pool).
  Must be `True` when `colocate_validate_and_train=False`.
  **Default:** `True`

### Status Collection

The rollout coordinator collects real-time engine statistics to enable smart routing
and sync decisions.

`status_collection.enable`
: Whether to enable engine status collection.
  **Default:** `True`

`status_collection.engine_sync_interval_in_ms`
: How often each vLLM engine pushes its status to the rollout coordinator (ms).
  **Default:** `100`

`status_collection.coordinator_sync_interval_in_ms`
: How often the coordinator aggregates engine statuses and pushes to the router (ms).
  **Default:** `100`

`status_collection.dump_logging_to_file_level`
: Granularity of status logs written to disk. Options: `none`, `partial_rollout`,
  `prompt`, `generation`, `all`.
  **Default:** `all`

`status_collection.dump_logging_to_file_interval_in_ms`
: File logging flush interval (ms).
  **Default:** `500`

`status_collection.stats_recorder.enable`
: Periodically write per-replica JSONL snapshots to `psrl.logging_path`, one file per
  `(replica_id, dp_rank)` pair.
  **Default:** `True`

`status_collection.stats_recorder.interval_in_s`
: Snapshot interval (seconds).
  **Default:** `1.0`

### Rollout Coordination

`psrl.rollout_coordination.*` groups all the online generation-coordination
strategies together. It is composed from six Hydra sub-groups (**partial rollout**,
**redundant rollout**, **routing strategy**, **sync & migration**, **proactive
filter**, and **session strategy**), each documented below. Together they decide how
requests are dispatched, when weights are synced, how stragglers and over-capacity
sessions are handled, and how much work is over-provisioned.

```{seealso}
{doc}`../design/flexible_rollout` covers the design of partial rollout, redundant
rollout, intelligent routing, and migration, and how they interact with the staleness
system.
```

#### Partial Rollout

Allows generation to be interrupted and resumed, preventing long sequences from
blocking the training pipeline.

`rollout_coordination.partial_rollout.enable`
: Whether to enable partial rollout interruption.
  **Default:** `True`

`rollout_coordination.partial_rollout.interrupt_as_prompt`
: If `True`, interrupted trajectories are treated as new prompts (the partial
  generation becomes part of the next prompt). If `False`, the SMG path keeps the
  request active and continues it through partial-rollout routing loopback.
  **Default:** `False`

#### Redundant Rollout

Generates more trajectories than needed for training, allowing the system to select the
best subset and discard redundant or slow samples.

`rollout_coordination.redundant_rollout.enable`
: Whether to enable redundant rollout generation.
  **Default:** `False`

`rollout_coordination.redundant_rollout.alg_global_batch_size`
: Required batch size for the algorithm (buffers are considered ready at this size).
  **Default:** `${psrl.staleness_buffer_entries}`

`rollout_coordination.redundant_rollout.alg_rollout_n`
: Number of responses required by the algorithm per prompt.
  **Default:** `${psrl.rollout_n}`

`rollout_coordination.redundant_rollout.redundant_global_batch_size`
: Actual number of trajectory *prompts* generated (must be ≥ `alg_global_batch_size`).
  **Default:** `${psrl.staleness_buffer_entries}`

`rollout_coordination.redundant_rollout.redundant_rollout_n`
: Actual number of responses generated per prompt (must be ≥ `alg_rollout_n`).
  **Default:** `${psrl.rollout_n}`

#### Routing Strategy

Controls how generation requests are dispatched across rollout instances.

`rollout_coordination.routing_strategy.method`
: Routing algorithm. Options:
  - `random`: uniform random assignment
  - `round_robin`: cyclic assignment
  - `request_num_balance`: route to the instance with fewest active requests
  - `throughput_optimal`: maximize global throughput using a cost model
  - `throughput_optimal_with_budget`: throughput-optimal with per-request token budget
  - `cache_aware`: SMG's native prefix-cache-aware routing (single-tier: GPU-resident
    prefix hits, plus shortest-queue load balancing)
  - `cache_aware_v1`: PSRL's optimized, **multi-tier** cache-aware variant. On top of
    SMG's native behaviour it also scores off-GPU (LMCache CPU-tier) prefix hits,
    using the `cache_aware_policy` weights below. Prefer this when LMCache offload is
    enabled.

  **Default:** `request_num_balance`

`rollout_coordination.routing_strategy.cache_aware_policy`
: Hyperparameters for the SMG cache-aware router. Only used when
  `method` is `cache_aware` or `cache_aware_v1`. The multi-tier fields
  (`lmcache_overlap_weight`) only take effect under `cache_aware_v1`.

  | Sub-field | Description | Default |
  |---|---|---|
  | `cache_threshold` | Min cache-hit ratio for the approximate radix-tree fallback path (used only when KV events are unavailable, since event-driven scoring ignores it). Ranges from 0.0 to 1.0. | `0.0` |
  | `gpu_overlap_weight` | Weight for GPU-resident prefix hits in the multi-tier overlap score. A GPU hit costs ~zero reload. | `1.0` |
  | `lmcache_overlap_weight` | Weight for off-GPU (LMCache) prefix hits (`cache_aware_v1` only). Cheaper than re-prefill but not free, so keep `gpu >= lmcache`. `0` scores GPU hits only, so raise it once LMCache offload is enabled. | `0` |
  | `balance_abs_threshold` | Shortest-queue load balancing triggers when BOTH the absolute and relative request-count thresholds are met. | `16` |
  | `balance_rel_threshold` | Relative request-count spread threshold for the load-balancing trigger. | `1.5` |
  | `balance_token_usage_threshold` | KV-utilization (token usage) level that triggers load balancing (`>= 1.0` disables). | `0.75` |
  | `overload_token_usage_threshold` | KV-utilization level above which an instance is treated as overloaded (`>= 1.0` disables, which is the default). | `1.0` |
  | `eviction_interval_secs` | Approximate radix-tree maintenance interval (fallback when KV events are unavailable). | `60` |
  | `max_tree_size` | Max size of the approximate prefix tree. | `67108864` (2^26) |
  | `block_size` | KV block size used for event-driven routing. | `16` |
  | `kv_capacity_threshold` | `cache_aware_v1` only. Admission gate rejects an instance when `effective_kv_used + new_tokens > kv_capacity_threshold * max_model_len`. Values below `1.0` reserve headroom for decode-time growth (`0.85` leaves 15%). The shipped default of `2.0` is permissive and effectively disables the check. | `2.0` |

  The request-count spread (`balance_abs_threshold` / `balance_rel_threshold`) is
  computed from in-flight worker load, whereas the two token-usage triggers read the
  backend engine's `token_usage` snapshot.

`rollout_coordination.routing_strategy.kv_transfer`
: When re-routing a request to a different instance, optionally transfer its
  accumulated KV cache via LMCache P2P to avoid re-prefill. Requires
  `lmcache.enable` and `lmcache.enable_p2p`.

  | Sub-field | Description |
  |---|---|
  | `enable` | Master switch. **Default:** `False` |
  | `transfer_mode` | `async` (fire-and-forget), `sync` (await, no pin), `pin_sync` (pin→await→unpin). **Default:** `async` |
  | `transfer_timeout_ms` | Timeout for `sync`/`pin_sync` modes before falling back to re-prefill. **Default:** `5000` |
  | `stats_log_interval_s` | Interval (s) between periodic KV-transfer stats log lines on each source instance. `0` suppresses stats even when transfer is enabled. **Default:** `30` |

`rollout_coordination.routing_strategy.cost_model_path`
: Path to a JSON cost model file (required for `throughput_optimal` methods).
  **Default:** `null`

`rollout_coordination.routing_strategy.request_sort_indicator`
: How to prioritize requests within a routing cycle. Options: `short_length`,
  `long_length`, `small_id`.
  **Default:** `small_id`

`rollout_coordination.routing_strategy.candidate_sort_indicator`
: How to sort candidate instances. Options: `version`, `reserve_capability`.
  **Default:** `version`

`rollout_coordination.routing_strategy.enable_multi_priority_queue`
: Use separate queues for different request priorities.
  **Default:** `False`

`rollout_coordination.routing_strategy.enable_group_sticky`
: Pin all rollout requests sharing a `prompt_id` to the same rollout instance so
  their KV-cache prefixes are reused.
  **Default:** `False`

`rollout_coordination.routing_strategy.enable_trajectory_sticky`
: Pin all generation calls within a single trajectory (subsequent turns) to the
  same rollout instance that served the first turn, reusing the per-trajectory
  KV-cache prefix. This is the trajectory-affinity knob for multi-turn agentic RL.
  **Default:** `False`

`rollout_coordination.routing_strategy.logging_interval_in_ms`
: Interval for routing-loop log lines (ms). Currently a declared placeholder that no
  code path reads.
  **Default:** `2000`

`rollout_coordination.routing_strategy.delta_throughput_threshold`
: Stop routing new requests to an instance when its marginal throughput contribution
  drops below this fraction.
  **Default:** `0.5`

`rollout_coordination.routing_strategy.request_budget`
: Estimated token budget per request, used by `throughput_optimal_with_budget` to
  predict response length.
  **Default:** `1024`

`rollout_coordination.routing_strategy.snapshot_staleness_threshold_in_ms`
: Age limit for an engine-status snapshot before it is considered stale, measured as
  the gap between the last recorded timestamp and the snapshot time. Currently a
  declared placeholder that no code path reads.
  **Default:** `1000`

`rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance`
: Cap on concurrent sequences per instance. This value serves double duty: it is the
  admission gate's in-flight request cap **and** it is forwarded to vLLM as
  `max_num_seqs`. Lower it to bound per-instance concurrency. `0` means no cap.
  **Default:** `1024`

`rollout_coordination.routing_strategy.check_interval_in_ms`
: Polling interval for the routing loop (ms).
  **Default:** `500`

**Admission control**

The admission gate is always on and has no master switch. It decides whether a
selected instance may accept a request based on in-flight count, KV capacity
(`cache_aware_policy.kv_capacity_threshold`), and the waiting-queue rule below.

`rollout_coordination.routing_strategy.admission_reject_on_waiting`
: When `True`, the gate only admits a request to an instance whose engine waiting
  queue is empty, which is the strict setting that keeps queueing at the router
  rather than inside the engine.
  **Default:** `True`

`rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption`
: Forwarded to vLLM as `preemption_notification_threshold`. The engine notifies the
  gateway once its waiting queue exceeds this many requests after a preemption, and
  those preempted requests are then looped back to the SMG router for global
  re-scheduling instead of staying queued on the local instance. This is a
  **notification** threshold, unrelated to admission despite the similar name.
  **Default:** `1024`

:::{important}
With the default SMG gateway, set `rollout_coordination.routing_strategy.method` to
`cache_aware` or `cache_aware_v1` to enable PSRL's vLLM KV-event publisher and prefix
reuse. Use `cache_aware_v1` when LMCache offload is enabled so off-GPU prefix hits are
scored too.
:::

#### Sync & Migration

Controls when model weights are synchronized and when rollout requests are migrated
between instances for load balancing.

`rollout_coordination.sync_and_mig_strategy.method`
: Strategy for deciding sync/migration timing.
  - `status_based`: use instance status indicators to decide when to sync
  - `greedy`: sync as soon as a new version is available

  **Default:** `greedy`

`rollout_coordination.sync_and_mig_strategy.check_interval_in_ms`
: Polling interval for the sync/migration loop (ms).
  **Default:** `100`

`rollout_coordination.sync_and_mig_strategy.sync.indicator`
: Metric that triggers weight sync. Options: `request_num`, `throughput`, `kv_cache`,
  `hypothesis_test`.
  **Default:** `request_num`

`rollout_coordination.sync_and_mig_strategy.sync.threshold`
: Workload threshold below which model sync is triggered. Interpretation depends on
  the `indicator` (count, tokens/s, or utilization fraction).
  **Default:** `null`

`rollout_coordination.sync_and_mig_strategy.sync.check_req_before_sync`
: Before syncing, verify that no routeable requests are pending for this instance.
  **Default:** `True`

`rollout_coordination.sync_and_mig_strategy.sync.seamless_train_version`
: All model versions ≤ this value are guaranteed to have a ready buffer waiting, so
  training can proceed immediately after weight pull without stalling.
  **Default:** `0`

`rollout_coordination.sync_and_mig_strategy.mig.enable`
: Whether to enable coordinator-side request migration between rollout instances.
  When enabled, the coordinator aborts requests on overloaded instances so they loop
  back to the router.
  **Default:** `False`

`rollout_coordination.sync_and_mig_strategy.mig.indicator`
: Metric used to identify imbalanced instances for migration. Options: `request_num`,
  `throughput`, `kv_cache`.
  **Default:** `request_num`

`rollout_coordination.sync_and_mig_strategy.mig.threshold`
: Relative imbalance ratio (`max_indicator / min_indicator`) that triggers migration.
  **Default:** `null`

`rollout_coordination.sync_and_mig_strategy.mig.stop_indicator`
: Metric used to decide when to stop migrating.
  **Default:** `request_num`

`rollout_coordination.sync_and_mig_strategy.mig.stop_threshold`
: Threshold on `stop_indicator` below which migration halts.
  **Default:** `null`

#### Proactive Filter

Handles situations where a buffer is nearly ready but a few remaining requests are
straggling.

`rollout_coordination.proactive_filter_strategy.method`
: Strategy for handling straggling requests.
  - `null`: disabled (wait indefinitely)
  - `retry`: abort and re-dispatch straggling requests
  - `truncate`: mark buffer as ready with fewer entries

  **Default:** `null`

`rollout_coordination.proactive_filter_strategy.threshold`
: Number of remaining reserved entries below which the filter strategy activates.
  **Default:** `0`

```{seealso}
{doc}`../design/staleness_control`, How proactive filtering integrates with the
Reserve/Occupy/Consume staleness protocol.
```

#### Session Strategy

Session hang/continue scheduling for multi-turn TITO sessions, ported from
**ThunderAgent**. When enabled, the RolloutCoordinator periodically **hangs** whole
sessions pinned to over-KV-capacity instances (blocking their next turn at the
SessionRouter without aborting in-flight work) and **continues** them once the pinned
instance frees capacity. This subsystem is under active development. ThunderAgent is
the currently integrated strategy, adapted from the original capacity-based
pause/resume design in the
[ThunderAgent project](https://github.com/ThunderAgent-org/ThunderAgent).

`rollout_coordination.session_strategy.thunder_agent.enable`
: Master switch for session hang/continue scheduling.
  **Default:** `False`

`rollout_coordination.session_strategy.thunder_agent.check_interval_in_ms`
: Scheduler tick interval (ms).
  **Default:** `1000`

`rollout_coordination.session_strategy.thunder_agent.env_token_weight`
: Reservation coefficient for env-status (between-turns) session tokens. Such a
  session's KV has already been freed from the engine pool, so it is absent from the
  measured `used_tokens`, so this coefficient adds it back as a predictive reservation
  for when the session returns from the environment. Values below `1.0` assume not
  all env sessions come back at once.
  **Default:** `1.0`

`rollout_coordination.session_strategy.thunder_agent.buffer_per_session`
: Decode headroom (tokens) reserved per running session.
  **Default:** `100`

`rollout_coordination.session_strategy.thunder_agent.continue_scope`
: Which instance a hung session is readmitted on.

  - `bucketed`: readmit the session on the instance it already occupies. This is the
    only valid choice under trajectory sticky routing, which `validate_config`
    enforces.
  - `global`: run a global Best-Fit-Decreasing pass across all instances, which may
    relocate the session onto an emptier one, matching ThunderAgent's
    `_greedy_resume`.

  **Default:** `bucketed`

`rollout_coordination.session_strategy.thunder_agent.continue_force_pin`
: Whether to force-pin the chosen continue instance for the readmitted session's next
  turn, sent as a one-shot pin that SMG clears on the first loopback. When `False`,
  only the session id is sent on continue and SMG routes the next turn normally.
  **Default:** `False`

`continue_scope` and `continue_force_pin` are independent, giving four combinations.

```{seealso}
{doc}`../design/router_tito`, SessionRouter, TITO session capture, and how hang/continue
interacts with sticky routing.
```

### NIXL

Configuration for RDMA-based weight synchronization via NIXL (used when
`ps_mode: nixl_cpu`).

`nixl.server_ip`
: NIXL server IP address.
  **Default:** `${psrl.ps_manager_ip}`

`nixl.server_port`
: NIXL server port.
  **Default:** `23456`

`nixl.max_pinned_temp_memory_slots`
: Number of pinned temporary memory slots for non-contiguous tensor transfers.
  Increase if you hit registration contention with many concurrent PS workers.
  **Default:** `16`

`nixl.enable_tms_for_temp_buffers`
: Manage NIXL temporary buffers with TMS, simplifying re-registration after memory
  is resumed.
  **Default:** `${psrl.tms.enable_nixl}`

### Checkpoint

Controls the Megatron checkpoint save/load strategy. Relevant only when using the
Megatron training backend.

`checkpoint.use_dcp_save`
: Whether to use verl's default DCP (Distributed Checkpointing) for save/load.

  - **`False`**: Use PSRL's per-rank `torch.save` (saves `rank_N.pt` +
    `parallel_config.json` per rank). This path is UCX-safe: it avoids DCP's
    `FullyParallelSaveStrategyWrapper` which calls `all_gather_object` on all shard
    metadata, causing a large temporary allocation that can corrupt NIXL's UCX
    endpoint memory under high memory pressure (manifests as
    `addr_version assertion` SIGABRT). The NIXL background UCX progress thread
    (`enable_prog_thread`) is kept **enabled** in this mode.

  - **`True` (default)**: Use verl's DCP path. Two patches are applied automatically:
    1. **NCCL no-fork patch**: DCP's async writer normally forks child processes
       that inherit NCCL communicators. When the child exits, `ncclCommAbort`
       corrupts the parent's NCCL state, causing a 600-second timeout then SIGABRT. The patch
       replaces the forking multiproc writer with a sequential in-process version.
    2. **NIXL prog_thread disabled**: `enable_prog_thread=False` is passed to the
       NIXL agent to prevent the UCX background thread from racing with DCP's
       `all_gather_object` memory activity.

  **Default:** `True`

### LMCache

KV cache offloading and cross-instance P2P transfer for reducing re-prefill overhead
in multi-turn workloads.

`lmcache.enable`
: Master switch for LMCache KV offloading in vLLM.
  **Default:** `False`

`lmcache.backend`
: Storage backend for offloaded KV blocks.
  - `cpu`: host memory (fast, limited by DRAM)
  - `disk`: filesystem-backed (large capacity, slower)
  - `remote`: reserved for a remote KV server, not yet implemented

  **Default:** `cpu`

`lmcache.offload_size_gb`
: Total offload budget in GiB, divided automatically across TP ranks. Do **not**
  set `LMCACHE_MAX_LOCAL_CPU_SIZE` as an env var, that would apply the full budget
  to every rank.
  **Default:** `100.0`

`lmcache.chunk_size`
: Token chunk size for hash-based KV indexing (must divide the block size).
  **Default:** `256`

`lmcache.clear_on_weight_update`
: Evict all cached KV entries after each model weight pull from the PS. This prevents
  stale-weight KV from being reused in the next generation round, but it is a blunt
  instrument, it discards every reusable prefix in the offload backend once per
  weight update. The default is `False` because the
  shipped configuration relies on `multi_version_kv` instead, which is the
  finer-grained mechanism and the one P2P requires.
  **Default:** `False`

`lmcache.multi_version_kv`
: Tag cached KV entries with the model version that produced them, so a request
  running under version N can never structurally hit an entry produced under version
  M, instead of clearing the whole cache on every update. Stale entries then age out
  naturally through ordinary LRU eviction as new-version entries fill the cache. This
  is the shipped default because under `psrl.staleness > 0` different rollout
  instances can legitimately sit at different model versions at the same time, so clearing the
  whole cache on every pull would throw away prefixes that a still-behind instance
  could still use. Required when `enable_p2p: True` (the shared P2P backend has no
  clear operation and relies entirely on version tags), in which case
  `clear_on_weight_update` must be `False`.
  **Default:** `True`

`lmcache.reserve_local_cpu_size`
: GiB of CPU memory to keep free and never use for KV offloading (headroom for other
  processes on the same node).
  **Default:** `0.0`

`lmcache.save_decode_cache`
: Also cache KV from decode steps (not just prefill). Increases memory usage but
  improves multi-turn prefix reuse.
  **Default:** `True`

`lmcache.save_unfull_chunk`
: Persist a chunk even when it is not completely filled, which helps prompts shorter
  than `chunk_size`.
  **Default:** `False`. Currently has a known bug, do **not** enable.

`lmcache.cache_policy`
: Eviction policy: `LRU` or `FIFO`.
  **Default:** `LRU`

`lmcache.enable_async_loading`
: Overlap KV cache retrieval with prefill computation to reduce time-to-first-token.
  **Default:** `False`. Currently has a known bug, do **not** enable.

`lmcache.config_file`
: Path to a full LMCache YAML config. When set, **overrides all individual fields
  above**.
  **Default:** `null`

**Disk backend (when `backend: disk`)**

`lmcache.disk_path`
: Filesystem path for disk-backed KV storage. Required when `backend: disk`.
  **Default:** `null`

`lmcache.max_disk_size_gb`
: Maximum disk usage for KV storage (GiB).
  **Default:** `1000.0`

**Remote backend (when `backend: remote`)**

`lmcache.remote_url`
: URL of the remote LMCache server, for example `redis://host:6379`. The remote
  backend is not implemented yet.
  **Default:** `null`

**P2P cross-instance transfer**

`lmcache.enable_p2p`
: Enable cross-instance KV cache transfer via a shared LMCache Controller process.
  Required when `routing_strategy.kv_transfer.enable: True`.
  **Default:** `False`

`lmcache.p2p_transfer_channel`
: Transport for P2P KV transfer.
  - `nixl`: UCX-based (RDMA on multi-node, shared memory on same node). Recommended.
  - `tcp`: fallback when UCX is unavailable.

  **Default:** `nixl`

`lmcache.controller_host`
: Host where the LMCache Controller runs.
  **Default:** `${psrl.ps_manager_ip}`

`lmcache.controller_base_port`
: Base HTTP port for the LMCache Controller's REST API (`/move`, `/lookup`, etc.).
  The actual port is selected via `find_available_port()` starting here.
  **Default:** `9000`

`lmcache.controller_pull_port`
: ZMQ PULL port where the Controller listens for worker registrations and heartbeats.
  **Default:** `8300`

`lmcache.controller_reply_port`
: ZMQ REPLY port for Controller → worker task dispatch.
  **Default:** `8400`

`lmcache.controller_health_timeout_s`
: Seconds to wait for the Controller's HTTP API to become healthy before failing
  init. The Controller imports torch and vLLM at startup and runs on the busy
  `ps_manager` node, so under cluster CPU or filesystem contention it can take
  considerably longer than a standalone launch.
  **Default:** `3000`

`lmcache.gpu_pin_block_budget`
: Max number of GPU KV blocks PSRL may hold pinned simultaneously, used by
  `routing_strategy.kv_transfer.transfer_mode == "pin_sync"`. When exceeded, the
  oldest-pinned trajectory is unpinned (PSRL-side LRU). `0` means no limit.
  **Default:** `0`

```{seealso}
{doc}`../design/kv_cache`, KV cache management architecture, LMCache Controller
process, and cache eviction behavior.
```

### TMS (torch_memory_saver)

GPU memory management that transparently swaps idle tensors to CPU, enabling colocated
workloads to share GPU memory.

`tms.range`
: Scope of TMS management.
  - `null`: disabled
  - `train`: manage training worker memory only
  - `all`: manage both rollout and training worker memory

  **Default:** `null`

`tms.enable_cuda_graph`
: Release CUDA graphs via TMS when not in use. Requires `range: all`.
  **Default:** `False`

`tms.enable_nixl`
: Manage NIXL temporary buffers with TMS (simplifies re-registration after resume).
  **Default:** `False`

### Agentic RL

Settings for multi-turn agent training loops (tool-use, code generation, SWE-agent).

`agentic_rl.refill_failure_threshold`
: How many rollout groups may fail back to back, with none completing in between,
  before training aborts. A train buffer entry is all-or-nothing, so a failed group
  is purged and replaced by one fresh prompt. When the fault is deterministic (an
  unreachable registry, a broken task image) every replacement fails the same way
  and the run churns refills at zero progress instead of failing. On reaching the
  threshold the manager latches a diagnosis naming the dominant `TerminateReason`
  and raises it from `wait_for_training_batch` and `wait_for_training_chunk`, which
  are the calls the driver blocks on. Any group that reaches `OCCUPIED` clears the
  count, so sporadic failures over a long run never accumulate into a trip. A refill
  that dispatches nothing aborts immediately regardless of the count, because the
  slot is already lost at that point.
  **Default:** `32`

`agentic_rl.trajectory_output.enable`
: Whether every agent loop writes a per-trajectory text dump via `TrajectoryWriter`.
  Files land at `<dir>/v{version}/{uid}.txt`, one per rollout trajectory.
  **Default:** `True`

`agentic_rl.trajectory_output.dir`
: Output directory for the per-trajectory dumps. Empty string falls back to
  `<psrl.logging_path>/trajectories`.
  **Default:** `""`

### Broadcast Init

When loading a large model checkpoint, each PS worker normally reads from disk
independently. `broadcast_init` instead has rank-0 read the checkpoint and broadcast
weights to other PS workers via NIXL, reducing filesystem load at scale.

`broadcast_init.enabled`
: Enable rank-0 broadcast initialization.
  **Default:** `False`

`broadcast_init.algorithm`
: Broadcast algorithm. Currently only `binary_tree` is supported.
  **Default:** `binary_tree`

### Group & Buffer Post-Processing

Post-processors can filter, re-weight, or transform trajectory groups before they are
submitted to the staleness buffer.

`group_post_process.enable`
: Enable streaming group-level post-processing.
  **Default:** `False`

`group_post_process.name`
: Registered post-processor name. Options: `dynamic_sampling_filter`, `no_filter`.
  When using `dynamic_sampling_filter`, requires `algorithm.filter_groups.metric` to
  be set.
  **Default:** `null`

`buffer_post_process.enable`
: Enable batch-level buffer post-processing (applied when a full buffer is ready).
  **Default:** `False`

`buffer_post_process.name`
: Same options as `group_post_process.name`.
  **Default:** `null`

### Log Probability

`log_prob.enable_rollout_engine_log_prob`
: Whether to request token log-probabilities from the vLLM rollout engine (used for
  importance sampling corrections). Disable to reduce generation overhead when
  log-probs are not needed.
  **Default:** `True`

### Server Rollout

An optional HTTP gateway that exposes PSRL's rollout service externally (useful for
serving agent loops from non-PSRL clients).

`server_rollout.enable`
: Enable the server rollout HTTP gateway.
  **Default:** `False`

`server_rollout.gateway.router_ip`
: Bind address for the gateway process.
  **Default:** `${psrl.ps_manager_ip}`

`server_rollout.gateway.router_port`
: HTTP port for the gateway.
  **Default:** `18080`

`server_rollout.server_concurrency`
: Max concurrent HTTP connections per rollout server.
  **Default:** `64`

### Rollout Gateway (SMG)

The rollout gateway is the mandatory online request path. A Ray `RolloutGateway` actor
starts SMG and SessionRouter subprocesses, and rollout replicas register as gRPC workers.

`rollout_gateway.server_max_concurrency`
: Maximum HTTP generation concurrency per active rollout server. The shared client
  budget is this value multiplied by active rollout and colocated validation
  instances.
  **Default:** `256`

`rollout_gateway.use_distributed_post`
: Route AgentLoopWorker POST requests through a round-robin Ray actor pool to spread
  HTTP client work across nodes.
  **Default:** `False`

`rollout_gateway.post_actor_num_per_node`
: Number of distributed POST actors placed on each alive Ray node when the pool is
  enabled.
  **Default:** `8`

`rollout_gateway.rust_log_filter`
: Per-module Rust log filter for the SMG gateway process, in `RUST_LOG` tracing
  directive syntax. Overrides the gateway's default `warn` level. An empty string
  means no override. The shipped value keeps SMG quiet while leaving PSRL's
  `route_trace` and `score_trace` targets at `info` so routing decisions stay
  visible.
  **Default:**
  `"warn,smg::routers::grpc::kv_transfer=warn,smg::routers::grpc::common::stages::worker_selector::psrl=warn,route_trace=info,score_trace=info"`

`rollout_gateway.grpc_registration_health_timeout_s`
: Total seconds a replica waits for its local `VllmEngine.HealthCheck` to pass
  before registering itself with the SMG gateway.
  **Default:** `300`

`rollout_gateway.grpc_registration_health_poll_interval_s`
: Interval between those health-check polls (seconds).
  **Default:** `1.0`

`rollout_gateway.grpc_registration_health_rpc_timeout_s`
: Per-RPC timeout for a single health-check call (seconds).
  **Default:** `5.0`

`rollout_gateway.enable_kv_event_replay`
: Serve `SubscribeKvEvents` through a buffering `KvEventReplayHub` that can replay
  missed sequence numbers after a gap, instead of the inline per-subscription ZMQ
  loop.
  **Default:** `False`. Keep it disabled: the gateway's KV-event monitor accepts
  sequence gaps monotonically, so it does not depend on the hub's replay guarantee,
  and the inline path is simpler with no ingester thread and measures at roughly
  100% cache-overlap routing. The hub path is opt-in and currently exhibits a
  zero-overlap indexing bug under investigation.

SMG uses `worker_selection_strategy=psrl`, gRPC worker connections, the
routing loop, and TITO. See {doc}`../design/router_tito`.

### TransferQueue

TransferQueue configuration is a top-level block in `ppo_trainer.yaml`.

`transfer_queue.enable`
: Runtime integration flag. `main_ppo.py` enables it for the current PSRL training
  flow.
  **Default in YAML:** `False`

`transfer_queue.metrics.enabled`
: Expose Prometheus-style metrics on an HTTP `/metrics` endpoint. When `False`,
  metrics are reported through the logger only.
  **Default:** `False`

`transfer_queue.metrics.port`
: Port for that endpoint. `0` auto-assigns a free port.
  **Default:** `0`

`transfer_queue.controller.sampler`
: Metadata sampling strategy.
  **Default:** `SequentialSampler`

`transfer_queue.controller.polling_mode`
: Enable polling-mode controller behavior.
  **Default:** `False`

`transfer_queue.backend.storage_backend`
: Storage implementation: `SimpleStorage` or experimental `MooncakeStore`.
  **Default:** `SimpleStorage`

`transfer_queue.backend.SimpleStorage.total_storage_size`
: Maximum number of experience samples across storage units.
  **Default:** `100000`

`transfer_queue.backend.SimpleStorage.num_data_storage_units`
: Distributed storage-unit count. Use at least twice the node count for larger
  deployments.
  **Default:** `8`

`transfer_queue.backend.MooncakeStore.*`
: Experimental Mooncake metadata/master addresses, local host, TCP/RDMA protocol,
  memory sizes, and NIC selection. See {doc}`../design/transfer_queue`.

### Memory Logger

`memory_logger.enable`
: Enable periodic GPU memory logging for debugging memory pressure.
  **Default:** `False`

`memory_logger.interval_seconds`
: Logging interval (seconds).
  **Default:** `30`

### Profile

Switches for profiling and analysis. `disable_attn` and `fix_weight` deliberately
break training correctness. Keep them at `False` for real runs. The `time_split`
and `prefill_composition` switches are observability-only and safe to enable during
normal training.

`profile.disable_attn`
: Disable attention in the rollout engine, propagated to vLLM as `VLLM_DISABLE_ATTN`
  and to `rollout.disable_attn`. Useful for isolating attention cost when profiling.
  **Default:** `False`

`profile.fix_weight`
: Skip the weight-load step after a parameter pull, so rollout instances keep serving
  their initial weights. Useful for measuring sync overhead without the load cost.
  **Default:** `False`

`profile.time_split.enable`
: Enable periodic per-replica prefill/decode time-split logging to
  `<psrl.logging_path>/TimeSplit_I{replica_idx}.log`. Each line reports cumulative
  prefill and decode wall-time, prefill fraction, token counts, and wall time since
  engine start.
  **Default:** `True`

`profile.time_split.interval_in_s`
: Interval between time-split log entries, in seconds.
  **Default:** `60.0`

`profile.prefill_composition.enable`
: Enable per-step prefill composition logging to
  `<psrl.logging_path>/Prefill_I{replica_idx}.log`. Every engine step that runs a
  prefill kernel emits one entry: which sequences were batched (new request, chunked
  prefill, or co-scheduled decode), how many tokens each contributed as cache hit vs.
  actual compute, and the host-side step duration. No sampling. High log volume on
  long runs (potentially GB per replica).
  **Default:** `False`

---

## Overrides

Override any parameter from the command line using Hydra syntax:

```bash
python -m psrl.trainer.main_ppo \
    +psrl.staleness=3 \
    psrl.rollout_coordination.routing_strategy.method=throughput_optimal \
    psrl.deployment.n_rollout_instances=4 \
    psrl.lmcache.enable=True \
    transfer_queue.backend.storage_backend=SimpleStorage
```

Key syntax rules:

- `key=value`: Override an existing key
- `+key=value`: Add a new key not present in the default config
- `~key`: Remove a key from the config
- Use dot notation for nested keys: `psrl.rollout_coordination.routing_strategy.method=...`

:::{tip}
For complex experiments, create a separate YAML file with your overrides and pass it
with `--config-path`:

```bash
python -m psrl.trainer.main_ppo \
    --config-path=/path/to/my_overrides \
    --config-name=my_experiment
```
:::

## Example: Advanced 7B FSDP Config

Here is a representative override pattern for 4-node DAPO training from
`examples/dapo_trainer/advanced_qwen2.5_7b_fsdp.sh`:

```bash
python -m psrl.trainer.main_ppo \
    --config-path=./config --config-name='ppo_trainer' \
    psrl.staleness=2 \
    psrl.staleness_buffer_entries=64 \
    psrl.ps_mode=nixl_cpu \
    psrl.rollout_n=8 \
    psrl.deployment.n_rollout_instances=16 \
    psrl.deployment.train_nnodes=2 \
    psrl.deployment.total_nnodes=4 \
    psrl.rollout_coordination.partial_rollout.enable=True \
    psrl.rollout_coordination.redundant_rollout.enable=True \
    psrl.rollout_coordination.routing_strategy.method=throughput_optimal \
    psrl.rollout_coordination.routing_strategy.enable_multi_priority_queue=True \
    psrl.rollout_coordination.sync_and_mig_strategy.method=status_based \
    psrl.rollout_coordination.sync_and_mig_strategy.sync.indicator=kv_cache \
    psrl.rollout_coordination.sync_and_mig_strategy.mig.enable=True \
    psrl.rollout_coordination.proactive_filter_strategy.method=retry \
    psrl.rollout_coordination.proactive_filter_strategy.threshold=4
```

```{seealso}
- {doc}`quickstart`: Minimal working example with DAPO
- {doc}`../design/staleness_control`: Staleness control design
- {doc}`../design/flexible_rollout`: Routing and rollout coordination
- {doc}`../design/kv_cache`: KV cache management
- {doc}`../design/router_tito`: SMG, SessionRouter, and TITO
- {doc}`../design/transfer_queue`: sample data plane
```
