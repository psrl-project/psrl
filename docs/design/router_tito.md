# Router, SessionRouter, and TITO

PSRL uses [SMG](https://github.com/psrl-project/smg) as its rollout gateway, configured and
extended with PSRL-aware worker selection, routing-loop state, partial-rollout
loopback, weight-version updates, KV transfer, and TITO training-data capture.

## Request Path

```{mermaid}
flowchart LR
    ALW["AgentLoopWorker"] -->|"HTTP generate/chat"| SMG["SMG RolloutGateway"]
    SALW["Session AgentLoop"] -->|"session-scoped OpenAI API"| SR["SessionRouter"]
    SR --> SMG
    SMG <-->|"gRPC admission + Reserve"| PSM["PSManager"]
    SMG -->|"gRPC"| V["PSRL vLLM replicas"]
    RC["RolloutCoordinator"] -->|"stats, pause/resume, weight version"| SMG
```

`RolloutGateway` is a Ray actor that launches two child processes:

- **SMG Router** on an automatically selected port starting at `8100`.
- **SessionRouter** on an automatically selected port starting at `8200`.

Rollout replicas register with SMG as gRPC workers. A single registered worker may
represent several data-parallel ranks, and routing and session affinity identify an
instance by `(base_worker_id, dp_rank)`.

## PSRL Changes to SMG

PSRL launches SMG with a routing loop and `worker_selection_strategy="psrl"`. The
PSRL selector performs a staged candidate selection instead of blindly applying a
load-balancing policy:

1. Filter unavailable workers and workers whose post-sync weight version cannot
   serve the request's `version_tag`.
2. Honor a partial-rollout or sticky-session instance hint when migration rules
   allow it.
3. Apply prompt-group affinity when group sampling is configured to stay on one
   instance.
4. Query PSManager for admission/reservation state and sort candidates by configured
   version or reserve capability.
5. Apply the selected SMG policy among valid candidates, record the chosen instance,
   and optionally initiate LMCache KV transfer when the request moved.

The RolloutCoordinator keeps this state current through SMG endpoints for worker
statistics, weight-version updates, routing-loop status, and pause/resume operations.
This allows weight synchronization to stop new dispatches before interrupting an
instance, then safely resume routing after the new version is installed.

For prefix-cache-aware routing, configure
`psrl.rollout_coordination.routing_strategy.method=cache_aware` (SMG's native,
single-tier policy) or `cache_aware_v1` (PSRL's multi-tier variant that also scores
off-GPU LMCache prefix hits).

## Partial Rollout in SMG

When a weight sync aborts an in-flight PSRL request, SMG's routing loop drains the
gRPC stream, preserves generated token IDs, log-probabilities, and optional
routed-expert metadata, and loops the request back through selection. The next
dispatch continues with the accumulated prefix rather than losing all completed
decode work.

This online loopback is different from storing a completed sample in TransferQueue:
the request remains an active rollout until it reaches a terminal result.

## SessionRouter

SessionRouter is a lightweight PSRL FastAPI proxy in front of SMG. It does not choose
workers itself. Its responsibilities are:

- Create, read, and delete SMG TITO sessions.
- Preserve immutable PSRL routing headers such as request ID, prompt ID, version,
  validation flag, and trajectory ID.
- Inject `x-smg-tito-session-id` on every session request.
- Record the **version tag** SMG pins on the first routed turn (`x-version-tag`) and
  carry it into every subsequent turn, so re-routes only pick instances whose weight
  version is at least as fresh as the one that first served the trajectory. This is
  how per-trajectory staleness control is enforced across multi-turn sessions.
- **Sticky affinity (opt-in)**: when trajectory-sticky routing is enabled
  (`psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky=True`, sent as
  the `x-is-sticky` header), capture the first successful `x-base-worker-id` and
  `x-target-dp-rank` and pin later turns to that rollout instance for KV-cache prefix
  reuse. When sticky is off, later turns are free to re-route.
- **Hang / continue**: track per-session status (`generate` while inferring on
  vLLM/SMG, `env` while calling the environment) and a hang state (`running` / `hung`).
  When the RolloutCoordinator's session strategy requests a hang, the session blocks at
  its next turn boundary (a mid-turn request is deferred and converted to hung once the
  in-flight turn returns, so no work is aborted). A continue releases it. See
  {doc}`flexible_rollout` (Session Strategy).
- Track in-flight requests so session deletion waits for active turns to drain.

The session-scoped endpoint is OpenAI compatible:

```text
POST /sessions
POST /sessions/{session_id}/v1/chat/completions
GET  /sessions/{session_id}
DELETE /sessions/{session_id}
```

This lets an external or third-party agent loop use its normal OpenAI client while
PSRL retains routing metadata and training observability.

## TITO

TITO is implemented inside SMG's gRPC chat pipeline. For a session, it incrementally
tracks the canonical token sequence and a per-turn record containing generated token
IDs, log-probabilities, finish reason, prompt boundaries, and optional routed-expert
metadata.

For session-based agent loops, the lifecycle is:

1. PSRL creates one TITO session and binds request/version metadata.
2. The agent sends normal chat-completion requests through SessionRouter.
3. SMG incrementally tokenizes matching prefixes and captures each assistant turn.
4. At episode completion, PSRL fetches the TITO session once.
5. `build_training_arrays()` converts the records into `prompt_ids`,
   `response_ids`, `response_mask`, rollout log-probabilities, turn counts, and
   optional routed-expert tensors.
6. AgentLoopWorker writes the canonical trajectory fields to TransferQueue and
   deletes the session.

TITO makes black-box agent integration practical: the agent can own conversation and
tool execution while SMG captures the exact model-side training sequence. It also
avoids repeatedly retokenizing a long conversation from scratch when the session
prefix is known.

## Agent Loop Impact

PSRL supports two agent-loop integration styles:

| Style | Request path | Best suited for |
|---|---|---|
| Native PSRL loop | AgentLoop directly calls rollout generation and constructs trajectory fields | Generic `Environment` + `AgentData` tasks and existing multi-turn loops |
| Session/TITO loop | Agent uses a session-scoped OpenAI API, and TITO reconstructs training arrays | Third-party or black-box agents such as mini-SWE-agent |

Session/TITO loops require the SMG rollout gateway. Native loops run on the same SMG
gateway.

## Configuration

```yaml
psrl:
  rollout_gateway:
    enable: true
    server_max_concurrency: 256
    use_distributed_post: false
    post_actor_num_per_node: 8

  rollout_coordination:
    routing_strategy:
      method: round_robin
      candidate_sort_indicator: version
      enable_group_sticky: true         # pin same-prompt_id requests to one instance
      enable_trajectory_sticky: true    # pin later turns of a trajectory (session sticky)
    session_strategy:
      thunder_agent:
        enable: false                   # capacity-based hang/continue (opt-in)
```

SMG enables TITO for the rollout gateway automatically. `tito_debug` and
`tito_gc_threshold` are accepted by the adapter when added as rollout-gateway
overrides. Debug validation adds CPU overhead and is intended for development.
`tito_snapshot_format=compact` (the default) stores shared tree nodes once and
lets PSRL reconstruct each trajectory; set it to `expanded` only for a gateway
that predates the compact snapshot.

```{seealso}
- {doc}`architecture`: complete PSRL data and control flow
- {doc}`flexible_rollout`: routing, partial rollout, and migration
- {doc}`../examples/agentic_rl/index`: agent-loop usage
```
