# Elastic Configuration

Elastic deployment is configured under `psrl.deployment.elastic`. Use
{doc}`../examples/elastic` for a complete launch recipe and
{doc}`resource_elasticity` for how the settings affect execution.

## Configuration Layout

The deployment group composes three files. **Policy names contain hyphens, while
configuration sections use underscores.**

| File under `psrl/trainer/config/psrl/deployment/elastic/` | Configuration prefix |
|--------------------------------------------------------|----------------------|
| `elastic_base.yaml` | `psrl.deployment.elastic` |
| `rule_based.yaml` | `psrl.deployment.elastic.rule_based` |
| `throughput_optimal.yaml` | `psrl.deployment.elastic.throughput_optimal` |

For example, select `scaling_policy_variant=throughput-optimal`, then tune
`psrl.deployment.elastic.throughput_optimal.min_gain`. Values in the tables below
are **composed YAML defaults**. The launch scripts override some of them.

## Required Settings

This fragment belongs in a complete training configuration with model, dataset,
reward-manager, and inference settings. It does not define an entire experiment.

```yaml
psrl:
  ps_mode: nixl_cpu
  colocate_validate_and_train: true
  fuse_rollout_with_validate: false
  fine_grain_overlap:
    granularity: none
  deployment:
    elastic:
      enable: true
      enable_policy: true
      enable_trainer_pool: true
      scaling_policy_variant: throughput-optimal
  tms:
    range: all
    enable_nixl: true
train_actor_rollout_ref:
  actor:
    fsdp_config:
      optimizer_offload: true
```

For Megatron, use
`train_actor_rollout_ref.actor.megatron.optimizer_offload=true` instead of the
FSDP setting. **Optimizer offload is required whenever the trainer can sleep**:
restoring model weights does not restore discarded optimizer state.

Elastic deployments require `psrl.ps_mode=nixl_cpu` or `nixl_gpu`, dedicated
validation replicas on training GPUs, `n_validate_instances>0`, and
`psrl.fine_grain_overlap.granularity=none`. The supplied recipes use `nixl_cpu`.
`psrl.deployment.heterogeneous_rollout.enable` cannot be combined with elastic
deployment. Different rollout/reward replica sizes are a separate supported
case, controlled by the heterogeneous candidate setting below.

Request-level evaluation additionally requires rollout routing
`psrl.rollout_coordination.routing_strategy.method=throughput_optimal`, reward
routing `itl`, and `router_waiting_top_t != 0`. Keep partial rollout enabled when
using interruption and redispatch. The supplied recipes configure these together.

## Resource Pools

| Field | YAML default | Meaning |
|-------|--------------|---------|
| `elastic.enable` | `false` | Initialize shared elastic placements and executor |
| `elastic.shared_nnodes` | `1` | Number of nodes in the shared inference pool |
| `elastic.shared_ngpus_per_node` | `8` | GPU count per shared node, or a per-node list |
| `elastic.enable_trainer_pool` | `false` | Allow inference replicas to borrow sleeping trainer GPUs |
| `elastic.enable_policy` | `true` | Enable automatic scaling decisions |
| `elastic.scaling_policy_variant` | `throughput-optimal` | `ScalingPolicy` implementation |

Fields in this table are relative to `psrl.deployment`. Training pool size comes
from `train_nnodes` and `train_ngpus_per_node`. `total_nnodes` optionally restricts
the GPU nodes available to the job. Shared and training pools reserve distinct
devices. A shared GPU specification such as `[4,8]` requires `shared_nnodes=2`.

Inference replica world size is `TP * PP * DP`. The runtime derives the maximum
rollout and reward replica counts by tiling each eligible pool independently.
**Do not use `n_rollout_instances` or a reward manager's `num_replicas` to set the
awake elastic allocation.** They are replaced with placement-derived limits.
Use the policy and minimum awake count to control the live allocation.

Set `psrl.deployment.mode` before allocating workers. A null mode selects
`elastic_rl` when `elastic.enable=true`, otherwise `disaggregated`.

| Mode | GPU ownership |
|------|---------------|
| `disaggregated` | Independent rollout and reward pools |
| `colocated` | Rollout, each reward model, validation, and training take turns on the trainer pool |
| `rollout_rm_colocated` | Rollout and each reward model take turns on the shared pool. Training and validation use the trainer pool |
| `trainer_pool_only` | Independent main pools plus fixed borrowers on idle trainer GPUs |
| `elastic_rl` | Placement aware policy decisions across eligible pools |

For `trainer_pool_only`, set `trainer_pool_idle_rollout_instances` and
`trainer_pool_idle_rm_instances`. The reward count applies to each active model.
All borrowers must fit on disjoint trainer GPU bundles. They sleep before training
or validation reclaims the pool.

Phased modes admit concurrent requests for one role or reward model, then drain
those users before another role wakes. Fully colocated training waits for its
batch's reward fields before taking GPU ownership. External agent sessions retain
their generation lease until the external agent returns.

## Lifecycle and Monitor

These fields are relative to `psrl.deployment.elastic`.

| Field | YAML default | Effect |
|-------|--------------|--------|
| `monitor_interval_ms` | `1000` | Delay between monitor cycles |
| `cooldown_ms` | `20000` | Minimum policy interval after an action |
| `min_awake_per_role` | `1` | Minimum awake inference instances retained by the policy |
| `trainer_ready_grace_s` | `5` | Keep an awake trainer briefly while checking the next batch |
| `coordinator_sync_timeout_s` | `10` | Timeout when collecting coordinator snapshots, non-positive disables the cap |
| `coordinator_command_timeout_s` | `180` | Command acknowledgement timeout, zero waits indefinitely |
| `decision_execution_abandon_stall_ticks` | `360` | Report a long-running decision without releasing its transition lease |

The executor keeps waiting for completion after a command acknowledgement timeout.
Increasing or reducing that timeout does not cancel the underlying operation.
The YAML includes `wakeup_immunity_ms=10000`, but the executor and scaling policies
do not consume it. Use cooldown to control repeated sleep and wake transitions.

TMS settings live under a separate prefix, `psrl.tms`:

| Field | YAML default | Effect |
|-------|--------------|--------|
| `range` | `null` | `train` manages training allocations, `all` covers supported inference paths too |
| `enable_cuda_graph` | `false` | Enable CUDA graph memory management on supported paths |
| `enable_nixl` | `false` | Include NIXL temporary buffers in TMS management |

Select TMS scope together with pool configuration. TMS does not enable colocation
or the scaling policy by itself.

## Scaling Policies

For the decision algorithms and a comparison of the two main policies, see
{doc}`resource_elasticity_scaling_policies`.

| `scaling_policy_variant` | Behavior |
|--------------------------|----------|
| `throughput-optimal` | Compare throughput gain against transition penalty, including physical GPU conflicts |
| `rule-based` | Expand onto free placements under KV pressure or router backlog, shrink under low KV usage |
| `none` | Select `DummyScalingPolicy`, which still expands demand onto free GPUs and shrinks empty replicas |

**`none` does not disable scaling.** Set `enable_policy=false` to stop automatic
policy decisions. Trainer and validation ownership transitions still occur.

### Rule-Based Policy

Select `scaling_policy_variant=rule-based`. Fields below are relative to
`psrl.deployment.elastic.rule_based` and measure KV cache utilization.

| Field | YAML default | Effect |
|-------|--------------|--------|
| `scale_down_threshold` | `0.1` | Consider shrinking a fresh instance strictly below this utilization |
| `scale_up_threshold` | `0.8` | Consider expanding a group when a fresh instance is strictly above this utilization |

Router backlog can also trigger expansion. The policy checks backlog first,
then high KV utilization, then low KV utilization. Scale-up requires a free
placement. Each decision changes at most one instance and respects cooldown and
the minimum awake count. Existing requests stay in place after scale-up.

Keep `0 <= scale_down_threshold < scale_up_threshold <= 1`. The YAML also
contains load-metric, waiting-queue, and full-load knobs, but
`RuleBasedScalingPolicy` does not use them to change these decisions.

### Throughput-Optimal Policy

Select `scaling_policy_variant=throughput-optimal`. Fields below are relative to
`psrl.deployment.elastic.throughput_optimal`.

| Field | YAML default | Effect |
|-------|--------------|--------|
| `decision_window_s` | `60.0` | Horizon over which estimated throughput gain pays for a sleep and wake transition |
| `migrate_time_s_initial` | `5.0` | Initial sleep and wake duration estimate |
| `migrate_time_ewma_alpha` | `0.2` | Weight of a new transition-duration observation |
| `min_gain` | `0.05` | Minimum absolute net benefit `L`, not a percentage |
| `max_scale_instances_per_action` | `-1` | Cap on same-role changes, `-1` removes the cap |
| `enable_heterogeneous_parallelism_candidates` | `false` | Enable bundle planning for different power-of-two replica sizes |
| `throughput_objective` | `sum` | Sum per-instance role throughput, or use `balanced_min` balanced-load estimation |
| `vllm_current_queue_scope` | `running` | Count `running` requests or `running_waiting` in engine load estimates |
| `router_waiting_top_t` | `-1` | Include all router waiting requests, positive values cap the count, zero excludes them |
| `current_state_include_router_waiting` | `false` | Include waiting load in aggregate current-state estimates |
| `role_throughput_weight_enable` | `true` | Weight rollout/reward throughput by outstanding demand |
| `role_throughput_weight_basis` | `request_count` | Use request counts, `token_count`, or `none` |
| `role_throughput_weight_mode` | `raw` | Raw pressure weights, or normalized `share` weights |
| `harmonic_denominator_use_max` | `false` | Use the sum of weighted reciprocal throughputs, `true` selects their maximum |

With request-level evaluation enabled, baseline and candidate simulations consume
their router snapshots directly. `current_state_include_router_waiting` controls
the aggregate estimator and should not be used to hide requests from the simulator.

Start tuning by checking cost-model accuracy and snapshot freshness. A longer
decision window makes predicted gains more likely to cover transition cost. A larger
`min_gain` or cooldown makes scaling more conservative. Enable heterogeneous
candidates for the 7B/30B pair's TP1/TP4 layout.

#### Candidate Evaluation and Redispatch

These fields use the same `throughput_optimal` prefix.

| Field | YAML default | Effect |
|-------|--------------|--------|
| `enable_request_level_candidate_evaluation` | `true` | Simulate per-request scheduling for baseline and candidate plans |
| `candidate_evaluation_backend` | `python` | `python` thread pool or persistent `cpp` subprocess |
| `candidate_evaluation_max_workers` | `64` | Maximum candidate-role evaluation workers |
| `candidate_evaluation_cpp_binary` | `null` | Resolve to `build/elastic_simulator/elastic_simulator` under the repository |
| `candidate_evaluation_cpp_timeout_s` | `30.0` | C++ evaluator response timeout, must be positive |
| `rebalance_after_scale_up` | `true` | Allow post-scale-up interruption and redispatch |
| `rm_router_enable` | `true` | Enable shared ITL reward routing |
| `rm_router_max_itl` | `null` | Optional ITL threshold for queueing reward requests |

Setting `rebalance_after_scale_up=false` makes the executor skip post-scale-up
interruption. It does not disable the evacuation needed to release a replica
during scale-down. Candidate simulation still evaluates primary scale-up
rebalancing, so keep the default when matching planned and executed migrations.

#### Cost Models

The scaling policy's `throughput_optimal.cost_model_path` defaults to
`psrl/trainer/config/cost_model`. It accepts a directory of model-named JSON files
or one JSON file. `model_params={}` accepts inline coefficient overrides.
The matching inference TP/PP bucket is selected from the model settings.

Configure routing cost models independently:

- Rollout: `psrl.rollout_coordination.routing_strategy.cost_model_path`.
- Reward: `reward.reward_loops.<manager>.routing_strategy.cost_model_path`.

The supplied JSON files contain `TP1_PP1`, `TP4_PP1`, or other fitted buckets. The
request-snapshot adapter requires the exact bucket and the routing fields
`other_threshold`, `other_latency_b`, `other_latency_k`, `attn_latency_b`, and
`attn_latency_k`. The scaling policy maps these to its ITL coefficients. Refit for a new
hardware or parallelism setup rather than reusing coefficients blindly.

## Recipe Overrides

The shared launcher sets `enable_trainer_pool=true`, `min_awake_per_role=0`,
`cooldown_ms=10000`, throughput-optimal policy, and partial rollout. The 1.5B and
7B recipes select C++ evaluation and exclusive migration dispatch. The 7B recipes
also enable heterogeneous candidates. These are launcher choices, not YAML
defaults. Inspect a recipe with `--cfg job --resolve` before running a modified
configuration.
