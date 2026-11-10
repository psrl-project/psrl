# Flexible Rollout Coordination

PSRL employs several complementary techniques to maximize generation throughput and minimize idle GPU time during rollout. Each one targets a different source of imbalance in the rollout workload.

```{figure} /_static/img/rollout_techniques.svg
:alt: Rollout Coordination Techniques
:width: 100%
:align: center

Different complementary rollout coordination techniques in PSRL.
```

All of these strategies live under a single config group, `psrl.rollout_coordination.*`,
composed from six Hydra sub-groups: `partial_rollout`, `redundant_rollout`,
`routing_strategy`, `sync_and_mig_strategy`, `proactive_filter_strategy`, and
`session_strategy`. Each is described below (proactive filter is covered in
{doc}`staleness_control`, while session strategy is summarized here and detailed in
{doc}`router_tito`).

---

## Partial Rollout

**Config**: `psrl.rollout_coordination.partial_rollout.*`

### Concept

When a rollout instance needs to sync to a new model version (triggered by the sync strategy), any in-progress generations present a dilemma: discarding them wastes all the compute spent so far, while waiting for them to finish delays the sync and increases staleness for future requests.

**Partial rollout** interrupts at the version boundary while preserving completed
tokens. On the SMG path, the routing loop drains the aborted gRPC stream, accumulates
tokens/log-probabilities, and loops the request back through worker selection until
it reaches a terminal result.

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable` | bool | `True` | Master switch for partial rollout |
| `interrupt_as_prompt` | bool | `False` | Treat interrupted trajectories as new prompts (prefix reuse mode) |

```yaml
psrl:
  rollout_coordination:
    partial_rollout:
      enable: true
      interrupt_as_prompt: false
```

### When to Use

:::{tip}
**Always enable** partial rollout for asynchronous training (`staleness > 0`). It avoids wasting compute when a sync fires in the middle of long-running generations. The only reason to disable it is for debugging or when running fully synchronously.
:::

- **`interrupt_as_prompt: false`** (default): preserve the partial-rollout state and
  continue the active request through routing loopback.
- **`interrupt_as_prompt: true`**: expose the interrupted prefix as a prompt-style
  continuation where supported by the selected path.

---

## Redundant Rollout

**Config**: `psrl.rollout_coordination.redundant_rollout.*`

### Concept

A training step needs exactly `B` prompts × `N` responses per prompt. Generation latency, however, is **highly variable**: even more so in agentic RL, where different trajectories may differ greatly in number of tool calls and turns. The slowest few trajectories dominate the step time.

**Redundant rollout** over-provisions generation work and then drops the stragglers as soon as enough trajectories have finished. PSRL supports two orthogonal levels of redundancy that can be combined:

- **Batch-level (prompt-level) redundancy**: launch generation for more prompts than the algorithm needs (`B' > B`). Once `B` prompts have all their responses ready, the remaining `B' - B` prompts are aborted.
- **Group-level (per-prompt) redundancy**: within each prompt's response group, generate more responses than the algorithm needs (`N' > N`). Once `N` responses for a prompt have finished, the remaining `N' - N` responses for that prompt are aborted.

Both levels cut tail latency by paying extra compute on aborted work.

### Key Configuration

| Field | Type | Description |
|-------|------|-------------|
| `alg_global_batch_size` | int | Prompts per step required by the algorithm (`B`) |
| `redundant_global_batch_size` | int | Prompts actually launched (`B' ≥ B`), batch-level redundancy |
| `alg_rollout_n` | int | Responses per prompt required by the algorithm (`N`) |
| `redundant_rollout_n` | int | Responses per prompt actually launched (`N' ≥ N`), group-level redundancy |

```yaml
psrl:
  rollout_coordination:
    redundant_rollout:
      enable: true
      # Batch-level: launch 80 prompts, keep the first 64 that fully complete.
      alg_global_batch_size: 64
      redundant_global_batch_size: 80
      # Group-level: launch 10 responses per prompt, keep the first 8 per prompt.
      alg_rollout_n: 8
      redundant_rollout_n: 10
```

The **redundancy ratio** at each level is `redundant / alg`. The two ratios are independent, total over-provisioning is their product (e.g. `1.25 × 1.25 ≈ 1.56×` extra trajectories launched in the example above).

### When to Use

Enable redundancy when **both** of the following hold:

1. The workload has a **significant long tail** in generation latency (a small fraction of trajectories takes much longer than the median).
2. **Dropping those tail trajectories does not meaningfully hurt convergence.**

The second condition is the important one. For example, early in training the longest trajectories are often the model repeating itself in a loop until hitting `max_response_length`, they are low-value (or outright noisy) and discarding them is essentially free. In contrast, for long chain-of-thought workloads the longest trajectories are precisely the hardest, most informative problems, systematically dropping them biases the training signal and can hurt convergence. In that case keep the redundancy ratio close to `1.0`, or disable redundant rollout entirely.

---

## Routing Strategy

**Config**: `psrl.rollout_coordination.routing_strategy.*`

SMG is the default Router. Its routing policy ranks load/cache candidates only after
the PSRL worker selector has enforced model-version eligibility, partial/sticky
instance hints, prompt-group affinity, and PSManager reservation constraints.

### Method Options

| Method | Description |
|--------|-------------|
| `random` | Random instance selection. Simple baseline with no state tracking. |
| `round_robin` | Cycle through instances sequentially. Ensures even distribution over time. |
| `request_num_balance` | Route to the instance with the fewest currently queued requests. |
| `throughput_optimal` | Cost-model-based routing that estimates per-instance throughput using the Waterfall model. |
| `throughput_optimal_with_budget` | Same as `throughput_optimal` plus token budget estimation for generation length prediction. |
| `cache_aware` | SMG's **native** prefix-cache-aware routing. Scores GPU-resident prefix overlap and applies shortest-queue load balancing (single-tier). |
| `cache_aware_v1` | PSRL's **optimized, multi-tier** variant. On top of the native behaviour it also scores off-GPU (LMCache CPU-tier) prefix hits via `cache_aware_policy.lmcache_overlap_weight`. Prefer this when LMCache offload is enabled. |

```yaml
psrl:
  rollout_coordination:
    routing_strategy:
      method: throughput_optimal
```

### Cost Model (for `throughput_optimal`)

The cost model estimates the decoding throughput of each instance:

$$
T_i = \frac{n_i}{k_1 \cdot \text{kv\_cache} + \max(k_2, k_3 \cdot n) + k_4}
$$

Where:
- $n_i$ = number of running requests on instance $i$
- $\text{kv\_cache}$ = current KV cache utilization (0-1)
- $k_1, k_2, k_3, k_4$ = fitted coefficients (calibrated per hardware/model)

The Router selects the instance with maximum estimated marginal gain of $T_i$.

:::{admonition} Cost Model Calibration
:class: note

The coefficients $k_1$-$k_4$ are fitted via profiling runs (see `psrl/trainer/config/cost_model/analyze.py`). They depend on:
- GPU type (H100, H20, A100)
- Model size and architecture
- Tensor parallelism degree
- Typical sequence lengths

Re-calibrate when changing hardware or model.
:::

### Multi-Level Queue (MLQ)

When `enable_multi_priority_queue=True`, requests are queued by their trajectory version (`V_traj`). Lower-version requests, those started under an older model version, are routed first, so older (more stale) trajectories drain ahead of fresher ones.

```yaml
psrl:
  rollout_coordination:
    routing_strategy:
      method: throughput_optimal
      enable_multi_priority_queue: true
```

### KV Transfer

**Config**: `psrl.rollout_coordination.routing_strategy.kv_transfer.*`

When SMG re-routes a request to a different instance, it can ask the LMCache transfer
path to move accumulated KV instead of recomputing it. The source worker performs the
data movement, while the shared LMCache Controller provides registration and fallback
control rather than carrying KV payloads itself.

| Field | Type | Description |
|-------|------|-------------|
| `enable` | bool | Enable KV transfer on re-routing |
| `transfer_mode` | str | Transfer strategy: `async`, `sync`, or `pin_sync` |

**Transfer modes:**
- **`async`**: Fire-and-forget. Start the KV transfer and begin generation on the target immediately, if the KV arrives late, the target falls back to re-prefill.
- **`sync`**: Wait for the KV transfer to complete before starting generation. No re-prefill, at the cost of added latency.
- **`pin_sync`**: Pin source KV (prevent eviction), wait for transfer, then unpin. Most reliable, highest overhead.

```yaml
psrl:
  rollout_coordination:
    routing_strategy:
      kv_transfer:
        enable: true
        transfer_mode: sync
```

:::{seealso}
{doc}`kv_cache` for details on LMCache integration and P2P transfer mechanisms.
:::

---

## Sync & Migration Strategy

**Config**: `psrl.rollout_coordination.sync_and_mig_strategy.*`

### Sync Strategy

Determines **when** a rollout instance pulls new weights from the Parameter Server. The goal is to sync at natural low points in the workload, when the instance is underutilized, instead of at arbitrary moments that interrupt peak throughput.

| Field | Type | Description |
|-------|------|-------------|
| `indicator` | str | Metric to monitor: `request_num`, `throughput`, `kv_cache`, `hypothesis_test` |
| `threshold` | float | Trigger sync when the indicator drops below this value |
| `check_req_before_sync` | bool | Only sync when no routable requests remain (i.e. the instance would otherwise starve) |

**Indicators:**
- `request_num`: Sync when the number of running requests drops below the threshold. Simple and effective.
- `throughput`: Sync when observed throughput drops below the threshold.
- `kv_cache`: Sync when KV cache utilization drops below the threshold.
- `hypothesis_test`: Hypothetically apply a sync and estimate its expected benefit, sync only if the estimate is positive.

```yaml
psrl:
  rollout_coordination:
    sync_and_mig_strategy:
      method: status_based
      sync:
        indicator: request_num
        threshold: 2
        check_req_before_sync: true
```

### Migration Strategy

Balances load across instances by **interrupting requests** on overloaded instances. Interrupted requests are automatically re-routed through the Router to a less-loaded instance. This is a coarse-grained but effective mechanism for handling sudden workload imbalance.

| Field | Type | Description |
|-------|------|-------------|
| `mig.enable` | bool | Enable migration-based load balancing |
| `mig.indicator` | str | Imbalance metric: `request_num`, `throughput` |
| `mig.threshold` | float | Relative imbalance ratio to trigger migration |
| `mig.stop_indicator` | str | Metric for stopping migration |
| `mig.stop_threshold` | float | When balance is restored below this threshold, stop migrating |

```yaml
psrl:
  rollout_coordination:
    sync_and_mig_strategy:
      mig:
        enable: true
        indicator: request_num
        threshold: 2.0       # Trigger when one instance has 2x the load of another
        stop_indicator: request_num
        stop_threshold: 1.2  # Stop when imbalance drops below 1.2x
```

:::{tip}
The sync and migration strategies work **together**, evaluated in order each tick: sync is considered first, and migration is only considered if no sync was triggered.
:::

---

## Session Strategy (Hang / Continue)

**Config**: `psrl.rollout_coordination.session_strategy.*`

The techniques above target single-shot generation. For **multi-turn TITO sessions**,
a trajectory alternates between generating on a pinned vLLM instance and calling its
environment, and it holds KV cache on that instance across the whole episode. When
many sticky sessions pile onto one instance, they can exceed its KV-cache capacity.

The **session strategy**, ported from **ThunderAgent**, adds capacity-based
**hang / continue** scheduling on top of routing:

- **Hang**: when a pinned instance is over KV capacity, the GlobalCoordinator hangs a
  whole session, blocking its *next* turn at the SessionRouter without aborting any
  in-flight turn. Env-status sessions (already off-GPU, cheapest to shed) are hung
  first, then generate-status sessions, smallest footprint first.
- **Continue**: hung sessions are readmitted (best-fit-decreasing) as soon as their
  pinned instance frees enough capacity.

This is currently the only integrated session strategy and is under active
development. It is disabled by default (`thunder_agent.enable: False`).

```yaml
psrl:
  rollout_coordination:
    session_strategy:
      thunder_agent:
        enable: true
        check_interval_in_ms: 1000
        env_token_weight: 1.0
        buffer_per_session: 100
```

:::{seealso}
{doc}`router_tito` for how hang/continue interacts with SessionRouter, sticky routing,
and TITO session capture.
:::
