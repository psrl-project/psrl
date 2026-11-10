# Elastic Scaling Policies

A scaling policy decides which inference replicas to sleep or wake. The
{doc}`resource_elasticity` runtime carries out the decision and enforces physical
GPU ownership. This page describes the two main policies:
`RuleBasedScalingPolicy` and `ThroughputOptimalScalingPolicy`.

| Property | `rule-based` | `throughput-optimal` |
|----------|--------------|----------------------|
| Decision signal | KV utilization and router backlog | Modeled throughput gain minus transition cost |
| Scale-up placement | Already free GPUs | Free GPUs or GPUs released by another role |
| Decision size | At most one instance | Placement-feasible batches |
| Cost model | Not required | Fitted per-model and parallelism coefficients |
| Post-scale-up migration | Disabled | Planned request redistribution when enabled |

Select the policy under `psrl.deployment.elastic.scaling_policy_variant`. Both
share the monitor, coordinator sleep/wake protocol, cooldown, and minimum awake
count. Training reclaims its entire pool through its batch lifecycle.

## Shared Decision Contract

`ScalingPolicy.decide()` receives `InstanceSignal` observations containing role, model,
instance ID, awake state, KV utilization, request counts, token occupancy,
throughput, timestamp, and physical GPU keys. Router backlog is supplied
separately. The throughput policy can additionally receive immutable per-request
snapshots and a trainer-pool availability hint.

The result is a `ScalingDecision`. Each `ScalingAction` identifies its role/model,
action type, target count, and preferred instance IDs. A throughput decision can
also include exact conflicting donors, companion wake targets, and planned request
migrations. These are transitions between initialized placements.

`ScalingAction.action_type` is `scale_up` or `scale_down`. `ElasticExecutor`
applies these decisions through `CommandType.SLEEP` and `CommandType.WAKE_UP`.
Scale-up may first sleep conflicting donors before waking the selected replicas.

Both policies skip decisions while disabled, during cooldown, or while a
previous decision is executing. The executor excludes unavailable training-pool
placements before invoking the policy. Minimum awake counts limit ordinary
scaling decisions.

Set `enable_policy=false` to stop automatic policy decisions. The factory's
`none` variant selects `DummyScalingPolicy`, which can still wake free capacity
or sleep empty replicas. It is not a disabling switch.

## Rule-Based Policy

`RuleBasedScalingPolicy` is a threshold controller. It groups instances by
role/model and applies the following priorities:

1. **Router backlog:** if demand is queued and a compatible sleeping placement is
   free, wake one instance. If several roles qualify, prefer the larger backlog.
2. **High KV utilization:** if any fresh awake instance is strictly above the high
   threshold, wake one free placement for its group. Prefer the group with the
   highest observed utilization.
3. **Low KV utilization:** if expansion was not selected, sleep the fresh instance
   with the lowest utilization strictly below the low threshold, provided the
   minimum awake count remains satisfied. A group above the high threshold does
   not shrink in the same decision.

Only one instance changes per decision. Scale-up does not evict another role,
and `rebalance_after_scale_up` is disabled for this policy. New requests can use
the added replica, while existing requests stay on their current replicas.
Scale-down still uses the runtime's interruption and release path.

### Configuration

This overlay selects the policy in a complete elastic training configuration:

```yaml
psrl:
  deployment:
    elastic:
      enable: true
      enable_policy: true
      scaling_policy_variant: rule-based
      cooldown_ms: 20000
      min_awake_per_role: 1
      rule_based:
        scale_down_threshold: 0.1
        scale_up_threshold: 0.8
```

Thresholds measure KV cache utilization and must satisfy
`0 <= scale_down_threshold < scale_up_threshold <= 1`. These values are the
composed YAML defaults. The class uses KV thresholds directly. Other fields in
`rule_based.yaml`, such as `load_threshold_metric` and
`max_waiting_queue_for_scale_down`, do not change this implementation's decisions.

Use this policy when a threshold controller without fitted latency models is
sufficient. If all compatible placements are occupied, high KV utilization or
backlog alone cannot trigger a cross-role GPU transfer. Another role must first
release capacity.

Source: `psrl/workers/elastic/scaling_policy/rule_based.py`.

## Throughput-Optimal Policy

`ThroughputOptimalScalingPolicy` compares placement-feasible allocations using
a fitted decoding cost model. Its candidate construction and benefit evaluation
follow the allocation design discussed in
[Section 7 of the paper](https://arxiv.org/html/2610.16866#S7).

### Throughput and Transition Cost

For instance $i$, the fitted inter-token latency and throughput are:

$$
\operatorname{ITL}_i = A S_i + \max(B, C N_i) + D,
\qquad T_i = \frac{N_i}{\operatorname{ITL}_i}.
$$

$S_i$ is token occupancy. $N_i$ is the request count selected by
`vllm_current_queue_scope`. The coefficients depend on the model, hardware, and
inference parallelism. Routing and planning use separately configured cost-model
paths. See {doc}`../examples/elastic` for preparing cost models.

With the supplied `throughput_objective=sum`, each role's throughput is the sum
of its active instances. When both roles have demand, the default system
objective is the demand-weighted harmonic mean:

$$
F = \frac{w_{\mathrm{rollout}} + w_{\mathrm{reward}}}
{w_{\mathrm{rollout}} / T_{\mathrm{rollout}}
+ w_{\mathrm{reward}} / T_{\mathrm{reward}}}.
$$

By default, weights use outstanding request counts, including router demand.
If only one role has demand, the implementation uses its loaded-role objective
so an idle downstream or upstream stage does not prevent useful scaling.

A candidate's net benefit is:

$$
L = H \left(F_{\mathrm{candidate}} - F_{\mathrm{current}}\right) - \Phi.
$$

`decision_window_s` supplies horizon $H$. The transition penalty $\Phi$ estimates
tokens lost while donor requests are interrupted, using donor load and estimated
sleep and wake duration. Observed transition durations update the estimate through an
EWMA. `ThroughputOptimalScalingPolicy` accepts the highest-scoring candidate only if
`L > min_gain`. **`min_gain` is an absolute net-benefit threshold**, not a
percentage or relative speedup.

### Candidate Construction and Simulation

The planning cycle proceeds as follows:

1. Identify bottleneck inference roles using outstanding demand and modeled
   throughput. Respect the trainer's current pool reservation.
2. Construct wake/sleep combinations from registered placements, including
   conflicting replicas that must release GPUs.
3. Keep a bounded set of candidate allocations. Equal-size replicas use ordered
   prefixes. Heterogeneous bundle planning groups physical conflicts and uses
   dynamic programming to retain low-penalty plans for each wake count.
4. Estimate the resulting workload distribution and throughput. With request-level
   evaluation enabled, simulate both baseline and candidates from the same
   immutable engine and router snapshots.
5. Choose the candidate with the largest net benefit above `min_gain`, or keep
   the current allocation.

A larger receiver can require multiple donors. For example, a reward TP4 replica
can conflict with four rollout TP1 replicas. Enable
`enable_heterogeneous_parallelism_candidates` for supported layouts with
different power-of-two replica sizes. A donor shared by several wake targets is
charged once. Every candidate must fit the actual GPU sets and awake-count bounds.

Request simulation preserves sequence lengths, queue order, version eligibility,
and routing cost parameters. The selected scale-up action carries its simulated
request migrations for execution. Python thread-pool and persistent C++ process
backends use the same snapshot and result contract.

### Configuration

This overlay selects the supplied throughput settings in a complete elastic
training configuration:

```yaml
psrl:
  deployment:
    elastic:
      enable: true
      enable_policy: true
      scaling_policy_variant: throughput-optimal
      throughput_optimal:
        throughput_objective: sum
        decision_window_s: 60.0
        migrate_time_s_initial: 5.0
        migrate_time_ewma_alpha: 0.2
        min_gain: 0.05
        cost_model_path: psrl/trainer/config/cost_model
        enable_request_level_candidate_evaluation: true
        candidate_evaluation_backend: python
        rebalance_after_scale_up: true
```

The request-level path requires rollout routing `throughput_optimal`, reward
routing `itl`, matching cost-model buckets, complete request snapshots, and
`router_waiting_top_t != 0`. Keep partial rollout enabled for policy interruption
and redispatch. The supplied elastic launchers configure these together.

Keep `rebalance_after_scale_up=true` when executing the migration assumptions
used by candidate simulation. Setting it to false suppresses post-scale-up
interruption, but scale-down still has to release donor replicas. Existing reward
requests are marked non-movable by the live snapshot adapter, as described in
{doc}`resource_elasticity`.

Start with cost-model accuracy and snapshot completeness before adjusting
`min_gain`. A longer horizon makes a throughput improvement more likely to repay
transition cost. A larger threshold or longer cooldown makes decisions more
conservative. Use `ScalingPolicy.log` to inspect `F`, `Phi`, `L`, selected
placements, and no-action reasons. The complete field reference is in
{doc}`resource_elasticity_configuration`.

Source: `psrl/workers/elastic/scaling_policy/thoughput_optimal.py`,
`psrl/utils/elastic/cost_model.py`, and
`psrl/workers/elastic/scaling_policy/evaluation/`.
