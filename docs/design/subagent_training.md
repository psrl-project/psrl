# Subagent-aware TITO training

Status: proposed

Scope: PSRL SessionRouter, TITO training data, agent loops, and the SMG gateway.

This page describes how one TITO session can hold a main agent and one or more
subagents, how each trajectory is marked trainable or context-only, and how a
shared prefix is trained exactly once.

## 1. Problem

A harness such as Codex can run a main agent and several subagents in one
conversation. The subagent calls currently share the session. TITO already
stores them as separate trajectories, but PSRL trains every trajectory in the
session with the same rule:

- Every assistant token on every trajectory is mask 1.
- Shared prefix turns appear in each trajectory's `turn_records`.
- The reward is broadcast to every output.
- There is no way to say "train the main agent, keep the subagent context, and
  do not train the subagent".

The goal is to make trainability a per-trajectory property while keeping one
session, one prefix tree, and one KV-cache lineage.

## 2. Current behavior

### 2.1 TITO snapshot

`third_party/smg/crates/tito/src/store.rs` already exposes:

```rust
pub struct Trajectory {
    pub trajectory_id: u64,
    pub accumulated_token_ids: Vec<u32>,
    pub turn_records: Vec<TurnRecord>,
}

pub struct TitoNodeMeta {
    pub id: u64,
    pub hash: String,
    pub stored: bool,
    pub parent: Option<u64>,
    pub finish_reason: String,
    pub truncated: bool,
    pub num_tokens: usize,
    pub trajectory_ids: Vec<u64>,
}

pub struct TitoLeafMeta {
    pub trajectory_id: u64,
    pub node_id: u64,
    pub parent: Option<u64>,
    pub path_node_ids: Vec<u64>,
}
```

`collect_records_for_leaf()` walks the leaf's parent chain and clones each
shared node's `TurnRecord` into every trajectory that passes through it.
`collect_tree_meta()` lists the trajectories that share a node in
`TitoNodeMeta::trajectory_ids`. There is no owner field, no capture mode, and no
role metadata.

### 2.2 PSRL extraction

`psrl/utils/tito/training_data.py::build_training_data()` treats one
trajectory as a linear stream:

- The first record's prompt becomes `prompt_ids`.
- Every later assistant output becomes `response_ids` with mask 1.
- Environment and tool tokens between turns become mask 0.
- `output_logprobs` is the only source of the generated token ids.

`psrl/workers/agent_loop/loops/session_agent_loop.py::get_training_data()`
builds one item per trajectory. `run()` returns a list of `TokenOutput` when the
session has more than one. `compute_advantage_for_multi_trajectories()` picks
the highest trajectory index in the session and broadcasts its advantage to all
rows.

### 2.3 Subagent detection

`psrl/workers/gen/session_router.py::is_subagent_request()` reads
`x-codex-turn-metadata` and checks `thread_source == "subagent"`. The router
uses that only to charge the turn budget separately. The header is forwarded to
SMG already because the router filters only hop-by-hop headers. SMG does not
store the role today.

### 2.4 Gaps

| Gap | Location |
|---|---|
| No role or agent identity on a trajectory | `Trajectory`, `TitoLeafMeta` |
| No owner on a shared node | `TitoNodeMeta`, `PrefixEntry` |
| No per-record capture mode | `TurnRecord` |
| Token-only records cannot be reconstructed without logprobs | `build_training_data()` |
| Shared prefix is trained once per trajectory | `build_training_data()` |
| Advantage picks the last trajectory, not the main one | `compute_advantage_for_multi_trajectories()` |
| No per-agent training policy | `psrl.agentic_rl` |

## 3. Goals and non-goals

### Goals

- Keep one TITO session for the main agent and its subagents.
- Mark each trajectory trainable or context-only.
- Keep the same prefix tree and KV lineage across agents.
- Train a shared prefix exactly once.
- Let an untrained subagent keep only the token ids needed for its next turn.
- Keep the session-level reward and the existing GRPO path.
- Preserve the current behavior when no role metadata exists.

### Non-goals

- Per-agent reward models or per-agent credit assignment beyond the session
  reward. This is a later extension.
- Changing how the harness spawns subagents. That remains a harness concern.
- Multi-session subagent routing. The design keeps one session.

## 4. Invariants

1. A session may contain many trajectories and many leaves. A trajectory is one
   root-to-leaf path.
2. Each stored node has exactly one `owner_trajectory_id`. The first successful
   store assigns it.
3. A node is trainable only when its owner is trainable and its capture is
   `Full`. A node without a training payload is context only.
4. A trainable trajectory trains its owned nodes. A node owned by another
   trajectory is context with mask 0.
5. A token is masked 1 at most once in a session. The owner's sample trains it
   and every other path masks it.
6. An untrained trajectory never becomes a training row. Its token ids remain in
   the session for inference.
7. The default mode trains only the main trajectory.
8. A session without role metadata keeps the current linear behavior.

## 5. Design

### 5.1 Data model in SMG

#### TurnRecord

```rust
pub enum TurnCapture {
    Full,
    Tokens,
}

pub struct TurnRecord {
    pub prompt_token_count: usize,
    pub output_token_ids: Option<Vec<u32>>,
    pub output_logprobs: Option<Vec<(f32, u32)>>,
    pub sampler_capture_width: u32,
    pub sampler_distribution: Option<SamplerDistribution>,
    pub routed_experts: Option<TurnRoutedExperts>,
    pub finish_reason: String,
    pub mismatch_report: Vec<MismatchEntry>,
    pub weight_version: Option<String>,

    pub owner_trajectory_id: u64,
    pub capture: TurnCapture,
    /// Session-wide storage order, assigned when the token stream is first
    /// stored. It is the causal boundary for post-main turns.
    pub sequence: u64,
}
```

`sequence` is monotonic within a session and is preserved across a later
payload upgrade. `contributes_to_session` is not stored in SMG; PSRL derives it
from `sequence` and the primary main leaf.

`output_token_ids` is populated on every capture path. It is the authoritative
source of generated ids for a token-only record. With `Full` it duplicates the
ids already present in `output_logprobs`, which keeps the extraction uniform and
lets the boundary match work without logprobs. The duplicate is 4 bytes per
token. A later optimization can omit it for `Full` records if the wire size
matters.

`capture == Tokens` sets:

- `output_logprobs = None`
- `sampler_capture_width = 0`
- `sampler_distribution = None`
- `routed_experts = None`
- `output_token_ids = Some(generated_ids)`
- `finish_reason` and `prompt_token_count` retained

The routed-experts prefix offset still advances on a token-only turn. vLLM's
`routed_experts_prompt_start` means "skip the prompt tokens whose routing was
already returned by earlier turns". A token-only turn adds its tokens to that
prefix even though no payload is stored, so the next full turn does not capture
them again. The extractor zero-fills the gap.

The turn also disables the sampler head request. SMG must not ask vLLM for
`sampler_top_logprobs` on a token-only turn, otherwise the capture is skipped
but the engine still pays for it.

#### PrefixEntry

```rust
pub struct PrefixEntry {
    pub token_ids: Arc<Vec<u32>>,
    pub reusable_prefix_token_ids: Option<Arc<Vec<u32>>>,
    pub parent_hash: Option<PrefixHash>,
    pub turn_record: TurnRecord,
    pub seq: u64,
    pub ends_tool_call: bool,

    pub owner_trajectory_id: u64,
    pub capture: TurnCapture,
}
```

`store_token_sequences()` sets `owner_trajectory_id = trajectory_id` when it
inserts a new node. If the same node is stored again:

- The first owner stays when its capture is `Full`.
- A `Tokens` owner may be upgraded to `Full` by a later trainable store, and the
  new owner and payload replace the old ones.
- A `Full` owner is never taken over by a `Tokens` store.

This keeps a trainable node trainable and prevents a later subagent call from
erasing its payload.

#### Trajectory metadata

```rust
pub struct TrajectoryMeta {
    pub thread_source: String,          // "main" or "subagent"
    pub agent_id: Option<String>,
    pub parent_trajectory_id: Option<u64>,
    pub capture: TurnCapture,
}
```

`SessionState` holds `trajectory_meta: HashMap<u64, TrajectoryMeta>`. The
preparation stage writes it when it resolves the trajectory id. The snapshot
copies it into each `Trajectory` and `TitoLeafMeta`:

```rust
pub struct Trajectory {
    pub trajectory_id: u64,
    pub accumulated_token_ids: Vec<u32>,
    pub turn_records: Vec<TurnRecord>,
    pub thread_source: String,
    pub agent_id: Option<String>,
    pub parent_trajectory_id: Option<u64>,
    pub capture: TurnCapture,
}
```

`parent_trajectory_id` comes from the normalized request header when the harness
provides one. Otherwise the snapshot derives it from the tree. For a leaf, walk
its path and find the nearest ancestor node whose `owner_trajectory_id` differs
from this leaf's id. That owner is the parent trajectory.

#### TitoNodeMeta

```rust
pub struct TitoNodeMeta {
    pub id: u64,
    pub hash: String,
    pub stored: bool,
    pub parent: Option<u64>,
    pub finish_reason: String,
    pub truncated: bool,
    pub num_tokens: usize,
    pub trajectory_ids: Vec<u64>,

    pub owner_trajectory_id: u64,
    pub capture: TurnCapture,
}
```

`trajectory_ids` keeps the sharing view. `owner_trajectory_id` identifies who
generated the node. A tool that joins snapshots by `hash` can use both.

#### Snapshot example

```json
{
  "session_id": "...",
  "max_trim_tokens": 1,
  "trajectories": [
    {
      "trajectory_id": 0,
      "thread_source": "main",
      "agent_id": "main",
      "parent_trajectory_id": null,
      "capture": "full",
      "accumulated_token_ids": [1, 2, 3, 4, 5],
      "records": [
        {
          "prompt_token_count": 2,
          "output_token_ids": [3],
          "output_logprobs": [[-0.1, 3]],
          "owner_trajectory_id": 0,
          "capture": "full",
          "sequence": 1,
          "contributes_to_session": true,
          "finish_reason": "tool_calls"
        }
      ]
    },
    {
      "trajectory_id": 1,
      "thread_source": "subagent",
      "agent_id": "codex-thread-7",
      "parent_trajectory_id": 0,
      "capture": "tokens",
      "accumulated_token_ids": [1, 2, 3, 6, 7],
      "records": [
        {
          "prompt_token_count": 3,
          "output_token_ids": [6],
          "output_logprobs": null,
          "owner_trajectory_id": 1,
          "capture": "tokens",
          "sequence": 2,
          "contributes_to_session": false,
          "finish_reason": "stop"
        }
      ]
    }
  ],
  "tree": {
    "nodes": [
      {"id": 0, "parent": null, "num_tokens": 3, "trajectory_ids": [0, 1],
       "owner_trajectory_id": 0, "capture": "full"},
      {"id": 1, "parent": 0, "num_tokens": 2, "trajectory_ids": [1],
       "owner_trajectory_id": 1, "capture": "tokens"}
    ],
    "leaves": [
      {"trajectory_id": 0, "node_id": 0, "path_node_ids": [0]},
      {"trajectory_id": 1, "node_id": 1, "path_node_ids": [0, 1]}
    ]
  }
}
```

In this example the subagent record starts at `prompt_token_count = 3`, so the
shared prefix node 0 is inside its prompt. The main trajectory owns node 0 and
trains it. The subagent owns node 1 but it is token-only, so it is context only.
`sequence` is stored by SMG; `contributes_to_session` is derived by PSRL at
snapshot normalization. Here the main-close sequence is 1; the subagent record
has sequence 2, so it is post-main and masked. A subagent record with sequence
<= 1 would contribute.

### 5.2 Request headers

Four turn-level headers are introduced. The session policy is not a header. It
travels in the `POST /sessions` JSON body. None of the headers is read while
training data is built. They matter during rollout, because SMG persists their
values into the TITO snapshot and the extractor reads that snapshot.

The session body is:

```json
{
  "subagent_training": {
    "mode": "main_only",
    "train_agents": [],
    "capture": "auto",
    "shared_prefix": "owner",
    "reward": "session",
    "loss_weight": {"main": 1.0, "subagent": 1.0}
  }
}
```

The SessionRouter parses it, stores it in `SessionState`, and does not forward it
to SMG. An empty body uses the configured default. A body is a better fit than a
header because the policy is structured and session scoped.

| Header | From | To | Scope | Meaning |
|---|---|---|---|---|
| `x-psrl-agent-role` | SessionRouter | SMG | turn | `main`, `subagent`, `auxiliary`, or `compaction` |
| `x-psrl-agent-id` | SessionRouter | SMG | turn | stable agent identity |
| `x-psrl-parent-agent-id` | SessionRouter | SMG | turn | parent agent identity, empty for main |
| `x-psrl-tito-capture` | SessionRouter | SMG | turn | `full` or `tokens` |

The role domain has four values. `auxiliary` covers harness calls that are not
part of the agent conversation, such as title generation and side replies.
`compaction` covers a main-context rewrite. It is not a subagent.

#### Why the identity headers exist

Each harness already marks its own calls in its own way. Codex uses
`x-codex-turn-metadata.thread_source`. Claude Code uses
`x-claude-code-request-class`, `x-claude-code-agent-id`, and
`x-claude-code-parent-agent-id`. A third-party harness may have no marker at
all.

`x-psrl-agent-role`, `x-psrl-agent-id`, and `x-psrl-parent-agent-id` are the
normalization layer. The SessionRouter is the only component that knows the
harness dialect. SMG and the extractor see one vocabulary. This keeps SMG from
depending on Codex or Claude Code, and it gives one place to add a new harness.

The headers are turn scoped because the role is a property of the call, not of
the session. A session can mix main, subagent, auxiliary, and compaction calls.
The identity is turn scoped for the same reason.

#### Why the capture header exists

`x-psrl-tito-capture` is the storage decision for one turn. It tells SMG which
part of the turn to keep:

- `full`: keep `output_logprobs`, `sampler_distribution`, and
  `routed_experts`, in addition to the token ids. The node can train.
- `tokens`: keep the token ids, prompt boundary, output ids, and finish reason.
  Drop the training payload. The node can still be used as context for the next
  turn, but it can never train.

The SessionRouter computes it from the session policy and the role. SMG does not
need to know the training policy. It only needs to know what to store.

The header is part of the first landing. The first version implements both
`full` and `tokens`. A trainable agent gets `full`, and untrained agents and
auxiliary calls get `tokens`. A correctness run may force `full` for everyone
before `tokens` is enabled in production.

#### Header lifecycle

1. The agent loop sends the `subagent_training` object once, in the
   `POST /sessions` body. The harness CLI does not send it.
2. The harness CLI sends its own native per-call metadata. The SessionRouter
   reads it and normalizes it into `x-psrl-agent-role`, `x-psrl-agent-id`, and
   `x-psrl-parent-agent-id`.
3. The SessionRouter computes `x-psrl-tito-capture` from the session policy and
   the role, and sends it to SMG with the turn.
4. SMG stores the role, agent id, parent, owner, and capture into the TITO
   session. The extractor reads only the snapshot.

The harness must carry its native marker on every model call. It does not need
to send the `x-psrl-*` names itself. If it has no native marker, the router
cannot distinguish its subagents, and the strict default attributes every call
to the main agent.

The SessionRouter overwrites any client-supplied `x-psrl-agent-role`,
`x-psrl-agent-id`, `x-psrl-parent-agent-id`, and `x-psrl-tito-capture` with its
own normalized values. Only the native harness markers are trusted from the
client. This keeps a harness from declaring an auxiliary call trainable.

#### Native harness markers

| Harness | Native input | Notes |
|---|---|---|
| Codex | `x-codex-turn-metadata` | `thread_source` is `user` or `subagent`. A stable `thread_id` and parent id are used when present. |
| Claude Code | `x-claude-code-request-class`, `x-claude-code-agent-id`, `x-claude-code-parent-agent-id`, `x-claude-code-agent-type` | The request class is `main`, `subagent`, `workflow`, `compaction`, `auxiliary`, or `side_reply`. Agent id and parent id are emitted for non-main agents. |
| Generic | `x-psrl-agent-role`, `x-psrl-agent-id`, `x-psrl-parent-agent-id` | A harness without native metadata may send the normalized headers directly. |

The Claude Code runtime is pinned to 2.1.295 or newer. That version emits
`x-claude-code-request-class`, `x-claude-code-agent-type`,
`x-claude-code-agent-id`, and `x-claude-code-parent-agent-id`.

Claude Code gates `x-claude-code-request-class` and
`x-claude-code-agent-type` behind `CLAUDE_CODE_GATEWAY_HINT_HEADERS` or a
first-party base URL. A custom session router is not first-party, so the Claude
Code adapter must set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` in the harness
environment.

Normalization rules:

- Codex `thread_source == "subagent"` maps to `subagent`. Any other value maps
  to `main`.
- Claude Code `request-class == "subagent"` or `"workflow"` maps to `subagent`.
  `"main"` maps to `main`. `"compaction"` maps to `compaction`.
  `"auxiliary"` and `"side_reply"` map to `auxiliary`.
- `agent_id` uses the native agent id or thread id when present. Otherwise the
  router uses `main` for the main thread and `subagent:<session turn index>` for
  a subagent.
- `parent_agent_id` uses the native parent id when present. Otherwise it uses
  the session's first main agent id.
- `x-psrl-tito-capture` comes from the session policy and the role. A trainable
  agent gets `full`. An untrained agent and every `auxiliary` call get `tokens`.
  A `compaction` call follows the main agent's trainability.
- A request without any role metadata is charged to the main agent, which is the
  stricter default.

SMG reads the headers in
`model_gateway/src/routers/grpc/regular/stages/chat/preparation.rs::try_tito()`.
The values live on `TitoRequestContext` and reach the response stage.

### 5.3 Capture policy

The SessionRouter owns the policy. It stores the session mode and the resolved
trainable agent ids in `SessionState`.

```text
role        = classify_request(request)   # main, subagent, auxiliary, compaction
agent_id    = normalized agent identity
trainable   = policy.is_trainable(role, agent_id)
capture     = "full" if trainable else "tokens"
```

Policy semantics:

- `off`: no trajectory is trained. This is an explicit diagnostic mode. A
  session with no trainable row is a setup error.
- `main_only`: the main agent is trainable. All subagents are `tokens`.
- `all`: the main agent and every subagent are `full`.
- `selected`: the main agent plus the configured agent ids or patterns are
  `full`. Everything else is `tokens`.
- `auxiliary` and `side_reply` calls are always `tokens` and never produce a
  training row.
- `compaction` calls follow the main agent's trainability. They are a context
  rewrite of the main path, not a separate agent.

The policy can be overridden per dataset row through
`extra_info.subagent_training`. The agent loop reads it and sends it in the
`POST /sessions` body when it creates the session.

The harness setting `harness.subagents_enabled` is separate. It controls whether
the CLI may spawn subagents. The training policy controls whether their
trajectories train.

### 5.4 Owner rule

A node's owner is the trajectory whose model call generated the node. The store
already receives the resolved `trajectory_id`, so it can write the owner at
insert time.

- New node: owner is the storing trajectory.
- Existing node with a `Full` owner: keep the owner and payload.
- Existing node with a `Tokens` owner and a new `Full` store: replace owner and
  payload with the trainable one.
- Existing node with a `Tokens` owner and a new `Tokens` store: keep the first
  owner.

A shared node's `trajectory_ids` list still contains every path that passes
through it. The owner is the one that trains it.

### 5.5 PSRL extraction

#### Selection

```python
def select_trainable_trajectories(session_data, policy) -> list[dict]:
    """Return the trajectories that may produce a training row."""
```

The function resolves the session's policy, reads each trajectory's
`thread_source`, `agent_id`, `parent_trajectory_id`, and `capture`, and returns
the trainable ones. The main trajectory is marked `session_primary=True`. If no
main trajectory exists, raise. If a trainable trajectory has `capture=tokens`,
raise when `require_primary_capture` is on.

Untrained trajectories stay in `session_snapshot` for the dump and for analysis.
They do not become `TokenOutput` rows.

#### Owner-aware builder

`build_training_data()` gains:

```python
def build_training_data(
    accumulated_token_ids: list[int],
    records: list[dict],
    *,
    trainable_trajectory_id: int,
    shared_prefix: str = "owner",
    ...
) -> dict:
```

For each record:

```text
owned = [i for i, r in enumerate(records)
         if r.owner_trajectory_id == trainable_trajectory_id]
if not owned:
    return empty
first = owned[0]
prompt_ids = accumulated_token_ids[:records[first].prompt_token_count]
process records[first:]
num_turns = len(owned)

env/user tokens between the previous cursor and this prompt end:
    response_ids += env_ids
    response_mask += 0
    logprobs += 0.0
    sampler_distribution += empty
    routed_experts += zero rows

output tokens:
    owner = record.get("owner_trajectory_id", trainable_trajectory_id)
    capture = record.get("capture", "full")
    shared = len(record.get("trajectory_ids", [])) > 1
    train_here = (
        owner == trainable_trajectory_id
        and capture == "full"
        and (shared_prefix == "owner" or not shared)
    )
    response_ids += record output ids
    response_mask += 1 if train_here else 0
    logprobs += record logprobs if train_here else zeros
    sampler_distribution += record distribution if train_here else empty
    routed_experts += record payload if train_here else zero rows
```

The prompt boundary is the first record owned by the trainable trajectory. The
shared prefix before it is prompt/context. The total context window
`prompt_length + response_length` is unchanged. Only the split changes.

`output ids` come from `record.output_token_ids` when present. The older
`output_logprobs[:, 1]` path remains as a fallback for snapshots without the new
field.

`shared_prefix="owner"` is the default and the only policy used by this design.
It trains every owned node, shared or not. `shared_prefix="context"` exists
only as a diagnostic fallback that trains only owned nodes that are not shared
with another trajectory. It is not recommended.

`routed_experts` needs one change. `_assemble_routed_experts()` currently
requires a segment on every record once any record has one. It becomes:

- A `Full` record contributes its segment at `routed_experts.prompt_start`.
- A `Tokens` record contributes zero rows for the positions it covers.
- If no record has a segment, the whole tensor is `None`.

The assembler must place each segment by its absolute `prompt_start` in the
canonical `(prompt_length + response_length, layers, top_k)` tensor. It must
not concatenate blindly, because a token-only turn leaves a gap. Overlapping
segments, out-of-range starts, and inconsistent shapes are contract errors.
The first `Full` segment defines `(num_layers, top_k)` and `dtype`. The trainer
ignores zero rows because their mask is 0.

#### TokenOutput metadata

Each trainable output carries:

```python
extra_fields = {
    "trajectory_id": ...,
    "agent_id": ...,
    "thread_source": ...,
    "parent_trajectory_id": ...,
    "trainable": True,
    "capture": "full",
    "session_primary": bool,
    "loss_weight": float,
    # v2: owned records that may train; num_turns remains all owned records.
    "trainable_turns": int,
}
```

#### Advantage

`compute_advantage_for_multi_trajectories()` prefers the row with
`session_primary=True` for the session advantage. When several rows carry the
same marker (legacy loops emit no marker), the latest trajectory index wins, as
before.

`session_primary` and `loss_weight` are emitted for every rollout row with
defaults `True` and `1.0`. The TransferQueue reads a field for the whole batch,
so a field that only some loops emit would fail the batch rather than fall back.
`ray_trainer.py` selects both next to `uid` and `parent_id`.

#### Loss and reward

- `reward=session` is the default. The harness computes one session reward and
  broadcasts it to every trainable row. A trainable subagent receives the same
  reward as the main agent.
- `loss_agg_mode=session-mean-token-mean` is recommended. `uid` identifies the
  session and `T_s` counts only mask 1 tokens. Shared prefix tokens appear in
  mask 1 only in the owner row, so they are counted once. The per-row
  `loss_weight` multiplies that row's share of the session mean.
- `group_baseline_valid_only` excludes zero-mask rows from the group baseline.
- `n_trajectory` is the number of trainable rows, not the number of leaves.

Per-agent reward is a later extension. It would call the reward function once
per trainable output and pass `agent_id`. It is out of scope for this version.

### 5.6 Configuration

```yaml
psrl:
  agentic_rl:
    subagent_training:
      # off | main_only | all | selected
      mode: main_only
      # Used only by selected. Exact agent ids or simple prefix patterns.
      train_agents: []
      # owner is the only supported value for training. context is diagnostic.
      shared_prefix: owner
      # session is the only implemented value. per_agent is reserved and rejected.
      reward: session
      # Applied to the session-weighted loss. Both default to 1.0.
      loss_weight:
        main: 1.0
        subagent: 1.0
      # Fail when a trainable trajectory has no training payload.
      require_primary_capture: true
      # Validation selection. Validation captures tokens only.
      val_mode: main_only
```

The capture mode is not a config field. It is derived per call: `full` for a
trainable agent, `tokens` for an untrained or auxiliary call, and `tokens` for
every validation call. `all` and `selected` put more than one row in one
session, so config validation requires `psrl.agentic_rl.batch_agg_mode=request`.

A dataset row may override the mode:

```json
{"extra_info": {"subagent_training": {"mode": "all"}}}
```

The agent loop sends the resolved policy in the `POST /sessions` body at session
creation. The same resolved policy is used at extraction time. If the two
disagree, `require_primary_capture` fails the sample instead of silently
training a context token.

### 5.7 Harness adaptation

The harness changes are additive. The session URL is already the only endpoint
the CLI talks to. The extra work is to enable subagents and to expose their
native role metadata.

#### Codex

- Enable subagent generation. The adapter writes `multi_agent_version = "v1"`
  when `subagents_enabled` is true and `null` when it is false, and adds
  `[features] multi_agent = true`. Codex 0.164 honors both; the live episode
  spawned a subagent with the namespaced v1 tool.
- Keep the provider on the session URL. Codex applies one `model_provider` to
  the main agent and its subagents, so no per-agent URL override is needed. A
  harness that lets a subagent select another provider must force the session
  provider.
- Keep `x-codex-turn-metadata` on every call. `thread_source` is the role
  marker. The router reads `thread_id` as the stable agent id and the parent
  thread id when present. The TITO tree is the fallback parent.
- The turn budget already counts subagent calls separately. No change.

#### Claude Code

- The harness runtime pins Claude Code 2.1.295 or newer. That version emits
  `x-claude-code-request-class`, `x-claude-code-agent-type`,
  `x-claude-code-agent-id`, and `x-claude-code-parent-agent-id`.
- Set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` in the harness environment. The
  Claude Code adapter already writes `settings.json` under the sandbox home, so
  this is one entry in that file. Without it, the request-class and agent-type
  headers are gated off on a custom base URL.
- `subagents_enabled` already controls the `Agent` tool. With the flag on, the
  CLI emits `x-claude-code-request-class = subagent` or `workflow`, plus
  `x-claude-code-agent-id`, `x-claude-code-parent-agent-id`, and
  `x-claude-code-agent-type`.
- Keep the Messages session URL for the main agent and its subagents.
- `x-claude-code-request-class = auxiliary` or `side_reply` marks calls outside
  the agent conversation. They are captured as `tokens` and never trained.
- `x-claude-code-request-class = compaction` is a main-context rewrite. It
  follows the main agent's trainability and is part of the main path.
- The CLI's `isSidechain` and `parent_tool_use_id` are transcript-level markers.
  The router does not need them.

#### Generic harnesses

- Send `x-psrl-agent-role`, `x-psrl-agent-id`, and `x-psrl-parent-agent-id` on
  every model call, or a native marker the SessionRouter knows.
- The SessionRouter sets `x-psrl-tito-capture` from the policy. The harness does
  not compute it.
- Use the same session URL for every agent. A subagent that talks to another
  endpoint is not captured.
- A harness that cannot set per-call metadata keeps the strict default. Every
  call is attributed to the main agent, so only `main_only` is meaningful.

## 6. Behavior matrix

| Mode | Main capture | Subagent capture | Auxiliary capture | Training rows |
|---|---|---|---|---|
| `off` | tokens | tokens | tokens | none, session is rejected |
| `main_only` | full | tokens | tokens | main only |
| `all` | full | full | tokens | main plus every subagent |
| `selected` | full | full for selected, tokens for others | tokens | main plus selected agents |

A `compaction` call follows the main agent's row. It is a context rewrite of the
main path, not a separate agent.

For `main_only` and `selected`, an untrained subagent still keeps its token ids
and turn boundaries. Its next turn can continue from the same prefix and KV
cache.

## 7. Example

```text
P (main prompt)
└── M1 (main assistant, owner=main, full)
    ├── M2 (main continues, owner=main, full)
    └── S1 (subagent assistant, owner=subagent, tokens)
        └── S2 (subagent continues, owner=subagent, tokens)
```

`main_only`:

- One training row. `P` mask 0, `M1` mask 1, `M2` mask 1.
- `S1` and `S2` exist in the TITO snapshot for inference. They are not rows.

`all`, with `S1` and `S2` captured as `full`:

- Main row: `P` mask 0, `M1` mask 1, `M2` mask 1.
- Subagent row: `P` mask 0, `M1` mask 0 because the owner is main, `S1` mask 1,
  `S2` mask 1.
- `M1` is trained once.

`all`, with `S1` token-only and a later trainable child:

- The subagent row has `S1` mask 0 and its own later tokens mask 1. `S1` has no
  payload, so it can never be trained.

## 8. Failure modes

| Condition | Action |
|---|---|
| A trainable trajectory has a `Tokens` owned node | Raise when `require_primary_capture` is on |
| Two `Full` stores disagree on an owner | Keep the first owner and log a warning |
| No main trajectory in the session | Raise, because the session has no primary |
| A subagent turn completes after the main-close sequence | Mask that turn; earlier contributing turns of the same trajectory may still train |
| The snapshot has no role or owner metadata | Use the legacy linear path |
| A `Tokens` record has no `output_token_ids` | Raise, because the boundary cannot be reconstructed |
| A shared node has `num_tokens` mismatch across snapshots | Raise, because the tree is inconsistent |
| `drop_dead_leaves` would drop a live subagent | Keep the leaf, because it is still pointed to by `trajectory_leaves` |
| A contributing owned record has token-only capture | `require_primary_capture` fails the sample |
| A record has no sequence | Treat `contributes_to_session` as true (legacy snapshot) |
| An identified agent has several leaves | Keep the highest trajectory id as the live leaf |
| The snapshot `format` is unknown | Error; an absent field means `expanded` |

## 9. Compatibility

- Old SMG snapshots have no role, owner, or capture fields. PSRL detects their
  absence and uses `build_training_data()` exactly as it does today. Every
  trajectory is trainable.
- Manual trajectory mode keeps working. The manual id is the owner, and the
  normalized headers still identify the role.
- Compaction branches are ordinary trajectories. The owner and capture fields
  apply to their nodes the same way.
- A harness that cannot mark subagent calls has all calls attributed to the main
  agent. This is the strict default and matches the current turn-budget rule.
- Claude Code auxiliary and side-reply calls are captured as `tokens` and never
  become rows. Compaction calls follow the main agent. The TITO tree already
  carries the compaction branch.
- The session API, the TITO tree shape, and the existing `x-codex-turn-metadata`
  input stay compatible.

## 10. Maturity, first-version fixes, and v2 design

The data model and the `main_only` path are ready for the first landing. The
`all/selected` path is ready in the data model but needs the fixes below. This
section separates what lands in the first version from what is designed for the
second.

### 10.1 Batch accounting: the sample is the session

Extra subagent rows do not add a sample. The existing multi-trajectory
compaction path already produces more than one row per request, and the chunk
manager treats the request as one unit:

- `tests/trainer/test_chunk_manager.py::test_multi_trajectory_request_is_not_split`
  asserts that a request with `n_trajectory=3` stays one chunk.
- `psrl/trainer/ppo/batch_schedule.py` groups by `uid` in request aggregation
  mode. `entries_per_update` is in requests, not rows.
- `psrl/utils/config.py` treats `train_batch_size * rollout_n` and
  `ppo_mini_batch_size * rollout_n` as request counts.

The extra subagent rows are part of the same session/request. They do not add a
sample. The requirements are:

- Use `psrl.agentic_rl.batch_agg_mode=request` for any run that may produce
  more than one trajectory per session. The harness recipes already do this.
- Keep `ppo_mini_batch_size` in sessions/requests, not rows.
- Keep a `uid`-aware loss such as `session-mean-token-mean` so the rows of one
  session are normalized together.
- Add a test with one main and several subagent rows under request aggregation
  that checks the chunk, advantage, and loss grouping.

`batch_agg_mode=trajectory` is not supported for multi-trajectory sessions. It
would treat the extra rows as separate samples and change the request count.

### 10.2 Subagent sample boundary and num_turns (first version)

Move the prompt boundary to the first record owned by the trainable trajectory:

- Find the first record with
  `owner_trajectory_id == trainable_trajectory_id`.
- Use `accumulated_token_ids[:record.prompt_token_count]` as `prompt_ids`.
- Start the response at that record's output.
- Count only owned records in `num_turns`.
- If no owned record exists, the trajectory is not trainable and produces no
  row.

The total context window `prompt_length + response_length` is unchanged. Only
the prompt/response split changes. The shared prefix is prompt context in the
subagent sample and is trained in the owner's sample.

v2 addendum: `num_turns` remains the count of owned records in the row.
`trainable_turns` counts only owned records with
`contributes_to_session=true`. A mixed trajectory therefore keeps its early
contributing turns and masks its post-main turns without lowering `num_turns`.

### 10.3 Routed-experts offset and alignment (first version)

vLLM's `routed_experts_prompt_start` skips the prompt tokens whose routing was
already returned by earlier turns. The scheduler reads the prompt routing from
that offset and the decode routing for the new tokens.

Therefore:

- A token-only turn still advances the trajectory's routed-experts offset. Its
  tokens enter the already-returned prefix even though no payload is stored.
  The next full turn starts after them and does not re-capture them.
- `TurnRoutedExperts.prompt_start` is the absolute start of the segment.
  `_assemble_routed_experts()` places each segment at that position in the
  canonical `(prompt_length + response_length, layers, top_k)` tensor and
  zero-fills the gap left by a token-only turn.
- Blind concatenation of per-turn segments is wrong when a gap exists. The
  current `canonicalize_routed_experts()` assumes contiguous rows from 0. Add
  an offset-aware assembler and reject overlapping, out-of-range, or
  inconsistent segments.
- Tests cover full, then tokens, then full, a tokens final turn, a tokens-only
  trajectory, and the normal contiguous multi-turn case. Each checks the
  canonical row positions, the zero gap, and the final sampled-token filler.

This is a first-version correctness fix because the first version includes the
`tokens` capture mode.

### 10.4 Parallel subagents: deterministic rules (first version rules)

Parallel branches are supported by the existing TITO auto trajectory
resolution. `resolve_trajectory_id()` reserves an in-flight id, so two
concurrent branches from the same prefix cannot claim the same trajectory.

The deterministic rules are:

- **Agent identity is request scoped.** Use the native agent id or thread id.
  If absent, derive a stable id from `(session_id, parent_agent_id, role,
  request sequence)`. Do not use the TITO trajectory id as the agent id. It is
  assigned later and is not stable under concurrency.
- **Parent is explicit or tree-derived.** Use the native parent id when
  present. Otherwise leave it empty and derive it at snapshot time from the
  nearest ancestor node whose owner differs from the leaf's owner. Do not fall
  back to the session main for a nested branch.
- **Trajectory id is SMG's authority.** The SessionRouter records the
  normalized metadata per request and lets SMG attach it to the resolved
  trajectory. It does not assign auto ids.
- **Primary is role-based.** The main trajectory is the one with
  `thread_source=main`, not the lowest or highest trajectory id.
- **Duplicate agent ids are retries.** If the same agent id appears on two
  leaves, the later leaf is live and the earlier leaf is protected or excluded
  by the dead-leaf rule. Extraction also keeps only the highest trajectory id
  for an identified main or subagent, so a retry or a compaction never emits two
  rows for the same agent.
- **Concurrent non-main threads.** Once a main call has been seen, a later
  main-labeled call with a different agent id is auxiliary unless it declares a
  parent (then it is a subagent). This prevents a second primary row.

The request aggregation keeps the rows of one session together. Parallel
branches add rows, not samples.

### 10.5 Nested subagents (first version metadata and rules)

- The native parent id is authoritative when present.
- Without it, `parent_trajectory_id` is the owner of the nearest ancestor node
  whose owner differs from the leaf's owner.
- `parent_agent_id` is the parent trajectory's `agent_id` from
  `TrajectoryMeta`, not necessarily the main.
- The owner rule applies per node. The nested subagent's shared prefix (main
  and parent subagent turns) is context in its sample and is trained in the
  owner's sample.
- `train_agents` may select an exact nested agent id or a role pattern.
- Auto trajectory resolution is role/agent-aware. It reuses an id only when the
  matched parent is that trajectory's current leaf and the request role and
  agent id are compatible. A subagent or nested branch from another agent's
  leaf gets a new id; a compaction starts a new context; a main continuation
  after compaction follows the compaction leaf.
- Test: main, then subagent A, then subagent B. B's prompt contains the main
  and A turns. B's response contains its own turns. A's tokens are mask 0 in
  B's sample and mask 1 only in A's sample.

### 10.6 Subagent compaction (first version rule)

`request-class=compaction` is a request class, not an agent role. The agent
identity is still `agent-id` and `parent-agent-id`.

- A compaction call carries `x-psrl-agent-role=compaction` plus the agent
  id/parent of the agent being compacted. The SessionRouter resolves
  trainability from the agent id, not from the word `compaction`.
- A main compaction follows the main agent. A subagent compaction follows the
  subagent. The capture mode follows the agent's trainability.
- The TITO tree represents a compaction as a branch with a phantom boundary.
  The live compaction leaf replaces the pre-compaction leaf for that agent.
  The dead pre-compaction leaf is excluded.
- Test: main compaction and subagent compaction. The live leaf is selected,
  the pre-compaction leaf is excluded, and the prompt/response split is
  correct.

If the four-value role enum is kept, `compaction` means "compaction of the
agent named by agent-id", never a fifth agent. A future `x-psrl-compaction`
flag can separate the two fields if the enum becomes ambiguous.

### 10.7 GC and dead leaves (first version protection)

`drop_dead_leaves` must never drop the latest live leaf of an agent.

- The TITO store keeps `trajectory_meta` and the latest leaf per agent id.
- A leaf is protected when it is the latest leaf for its agent id and parent,
  or when the SessionRouter reports it as in flight.
- `compute_dead_leaf_hashes()` skips protected leaves. Pre-compaction and
  aborted retry leaves remain eligible.
- The simplest first-version rule is to disable `tito_drop_dead_leaves` for a
  session that has subagents. Agent-level selection already picks the live
  leaf, so dead branches are context only. Store-side protection is the
  long-term rule.
- Test: a subagent leaf ending in a tool call with a later sibling. The latest
  subagent leaf is kept.

### 10.8 Validation (first version policy)

Validation may run subagents. Train and validation selection are separate:

- Add `subagent_training.val_mode` with default `main_only`.
- Validation scores the session primary (the main artifact) for accuracy and
  reward. Subagent rows are diagnostic unless `val_mode` is `all` or
  `selected`.
- `get_primary_training_data()` selects the primary trajectory instead of
  raising on multiple trajectories.
- The request-aggregated batch schedule handles extra rows in validation the
  same way as in training.
- Validation applies the same per-turn `contributes_to_session` gate. A
  post-main subagent turn is not scored even when `val_mode` selects its
  trajectory.
- Test: a validation session with main and subagents. Only the primary
  contributes to accuracy. Subagent rows are available for dump but are not
  scored.

### 10.9 Policy persistence (first version)

The session body policy must be visible to extraction and offline analysis.

- The SessionRouter stores the resolved `subagent_training` policy in
  `SessionState` and adds it to the `GET /sessions/{id}` view next to the turn
  accounting.
- `TitoSessionData`, or the PSRL session snapshot wrapper, carries the policy
  and the per-trajectory `TrajectoryMeta`.
- The session view also carries `main_agent_id`. It is session state rather than
  policy: compaction resolution needs the parent identity the session actually
  observed.
- Extraction uses the snapshot policy when present and falls back to the
  current config only for legacy sessions.
- Test: a snapshot with a policy different from the current config selects the
  rows from the snapshot.

### 10.10 Background and async subagents (v2 design)

Background subagents can outlive the main turn or the CLI process. They are not
part of the first landing.

The training question is causal, not temporal: a subagent may be trained only
when its output is part of the context or artifact that the session reward
scores. A subagent that finishes after the main agent's last turn cannot have
influenced that turn, so the session reward carries no credit for it. Training
it would reinforce arbitrary tokens.

The artifact boundary is part of that definition. Under logical isolation the
main collector reads only the main directory, so a post-main subagent's writes
to its own directory cannot enter the scored artifact. A deployment that lets a
post-main subagent write the scored artifact has a broken attribution boundary:
it must either exclude those writes or stop using the session reward for them.

Design:

- The harness episode summary reports subagent lifecycle: started, finished,
  background. Codex and Claude Code stream-json events are the source.
- The main sample is snapshotted as soon as the main agent's episode closes.
  The extractor does not wait for background subagents before that snapshot:
  once the main agent has made its last turn, no later subagent result can
  enter the main context or the main artifact, so waiting cannot change a main
  training row.
- At the main snapshot, the agent loop records only a count: the number of
  distinct subagent trajectories that are still live. A trajectory is live when
  the harness reports it started and not finished, or, when the harness reports
  no lifecycle, when it has an in-flight request at the snapshot. The count is
  attached to every emitted row as `background_subagent_count`; the current
  version does not collect the trajectories or their tokens.
- A live background subagent is not a whole-trajectory training unit. Its
  completed turns are handled per turn: a turn that completed after the
  main-close boundary is context/diagnostic only; a turn that completed before
  the boundary may still train when it is owned by a selected trajectory.
- A background result that the main agent consumes in a later turn is not
  post-main. The main context is the authority, so a main agent that resumes
  after a subagent completes can still make that subagent eligible. In the
  current one-episode-per-session flow the main sample closes when the harness
  episode ends; a turn arriving later is post-main by construction.
- Session deletion cancels or abandons remaining background work. It does not
  wait for it; waiting would only extend the session and hold the TITO store
  open. The generic `delete_session` drain is for requests on the main path;
  v2 must cancel or detach background in-flight turns instead of counting them
  in that drain. Otherwise the rollout still blocks on work that cannot
  contribute.
- A background subagent that writes to a separate session is out of scope. It
  must use the same session URL.
- Test: a harness reports two live background subagents at main close. The
  metric is 2, the main rows and snapshot latency are unchanged, session
  deletion does not wait for either, and only their pre-boundary owned turns
  can train.

#### Turn-level contribution boundary

A trajectory can straddle the main close: the main agent used an early result,
then sent a follow-up and ended while the same subagent was still working.
Whole-trajectory exclusion would drop the causal early turns; whole-trajectory
training would train the post-main turns. The boundary is therefore per turn.

- **Sequence.** TITO assigns a session-wide monotonic `sequence` to a turn when
  its token stream is first stored, not when its request starts. A later
  payload upgrade keeps the original sequence, because the output was already
  available. A turn that completes after main close gets a larger sequence even
  if it started earlier.
- **Main close.** `main_close_sequence` is the `sequence` of the last owned
  record on the primary main trajectory. The primary is the latest main-role
  leaf, the same rule the extractor already uses. If the main later resumes,
  its new last record moves the boundary later, so turns that were before the
  resumed main become eligible again.
- **Marking.** PSRL normalizes the snapshot before selection and sets
  `contributes_to_session` on every record:
  - main-role records and compactions of the main agent: true;
  - auxiliary records: false;
  - subagent records and compactions of a subagent: true when
    `sequence <= main_close_sequence`, false otherwise;
  - a record with no `sequence` (legacy snapshot): true, which preserves the
    old coarse behavior.
- **Selection.** A trajectory is selected only when it has at least one owned
  record with `contributes_to_session=true`. `require_primary_capture` checks
  capture only on those contributing owned records; a post-main token-only
  turn does not fail a mixed trajectory.
- **Extraction.** `build_training_data` sets `train_here` only for an owned,
  full-capture, contributing record. A post-main record is still part of the
  response with mask 0, zero logprobs, an empty sampler segment, and zero
  routed-expert rows. The row remains valid for its earlier contributing
  turns.
- **Turn counts.** `num_turns` keeps counting every owned record in the
  trajectory. `trainable_turns` counts the owned contributing records. A
  trajectory with `trainable_turns == 0` produces no row.
- **Compact snapshot.** The compact form carries the same turn sequence in its
  node payload; PSRL applies the same contribution marking after expanding it,
  so expanded and compact snapshots produce identical masks.

This boundary is still a time-order lower bound for credit: a subagent turn
that completed before main close but whose result the main never read remains
`true`. Detecting "spawned but unused" requires linking the subagent output to
the main messages; that is a later refinement, not part of this boundary.

The count is deliberately an observation, not a collection point. No timeout is
involved because nothing waits. If a deployment later wants post-main tokens
for diagnostics or a per-agent reward sample, that is a separate opt-in
collection path with its own lifetime, not part of the main snapshot.

Per-agent reward would let a post-main subagent be trained as its own sample
with its own task verifier. That is a different aggregation and reward path,
not the session-reward path. Until it is designed, post-main output is
diagnostic only.

### 10.11 Workspace isolation (v2 design)

TITO separates trajectories but not filesystem writes. Parallel subagents in
one sandbox can overwrite each other.

The v2 choice is **logical isolation**. It is a convention, not a security
boundary: one sandbox per session, a main task root that is the only collected
artifact, and a per-agent directory `/workspace/agents/<agent_id>/` for
subagent output. The harness prompt states the convention and the sandbox
exposes `PSRL_MAIN_WORKSPACE` and `PSRL_AGENT_WORKSPACE_ROOT` for tools that
honor it. The main collector reads only the main task root.

Nothing in the filesystem enforces the convention: a subagent can still write
to the main root or another agent's directory. That is the accepted trade-off
for keeping one sandbox, one shared filesystem, and no extra volume. A
deployment that needs a hard boundary selects sandbox isolation per task class.

The other options remain available when a deployment needs a hard boundary:

- **Sandbox isolation:** one sandbox per agent with a read-only copy of the
  task root and a per-agent output volume. The shared TITO session still works
  because model calls go through the same SessionRouter. This costs one sandbox
  per parallel subagent.
- **Worktree isolation:** for git tasks, one worktree per agent under
  `/workspace/.agents/<agent_id>/`. This is a special case of logical
  isolation.

Logical isolation is chosen because it keeps one sandbox per session, preserves
the shared filesystem the harness already expects, and costs no extra
container or volume. A deployment that cannot accept an advisory boundary
selects sandbox isolation.

Test: two subagents write the same filename. The per-agent directories keep
both artifacts and the main collector reads the main directory.

### 10.12 Compact tree snapshot (v2 design)

The current snapshot stores `accumulated_token_ids` and `turn_records` per
leaf, duplicating shared prefixes. A compact tree snapshot stores each node
once.

The snapshot carries a top-level `format` field. `format: "expanded"` is the
current per-trajectory view and the default when the field is absent, so old
SMG builds and old PSRL readers keep working. `format: "compact"` is the tree
form. The request selects it explicitly, and an unknown format is an error
rather than a silent fallback.

Design:

- `nodes[]` carries `id`, `parent`, its own `token_ids` segment with
  `token_start`, `num_tokens`, `owner_trajectory_id`, `capture`, the turn
  record, and `trajectory_ids`. `token_start` is the common-prefix length with
  the stored parent, so a tolerated trimmed boundary reconstructs exactly: the
  reader drops the parent's divergent suffix before appending the segment.
- `leaves[]` carries `trajectory_id`, `node_id`, `path_node_ids`,
  `thread_source`, `agent_id`, `parent_trajectory_id`, and `capture`.
- The extractor reconstructs a leaf's `accumulated_token_ids` and
  `turn_records` by walking the path. Shared nodes are read once.
- The per-trajectory view remains a derived adapter over the compact form or
  the current expanded form; training rows are identical either way.
- Each turn record carries its storage `sequence` and the derived
  `contributes_to_session` marker, so the compact form preserves the per-turn
  boundary.
- Tests: compact and expanded snapshots produce identical training rows. A
  shared prefix appears once. Phantom boundaries and dead leaves match the
  current behavior. A snapshot without `format` is read as `expanded`.

PSRL requests this form through
`psrl.rollout_gateway.tito_snapshot_format=compact` (the default). Set it to
`expanded` for a gateway that predates the compact form.

This is a v2 wire and gateway-serialization optimization: the saved bytes are
on the gateway-to-worker path and in the gateway's snapshot construction. PSRL
expands the compact form before selection, so its extraction memory matches the
expanded form. It does not change the training semantics.

### 10.13 Shared-prefix forward duplication (later, measure first)

Owner masks avoid duplicate gradients, but each trainable path still forwards
the shared prefix. This is a compute cost, not a correctness bug. Measure the
overhead on the target workload before designing tree-packed attention. If the
overhead is material, the v2 option is a packed tree sequence with a custom
attention mask and per-node loss masks. Do not build it speculatively.

### 10.14 Reward normalization (unchanged)

The session-level reward design keeps `RewardManager.normalize_reward()`
unchanged. It normalizes by `uid` and broadcasts the session reward. Per-agent
reward is deferred with credit assignment.

### 10.15 Subagent artifacts (out of scope)

The harness still collects the main artifact. Per-subagent artifacts and
per-agent rewards are out of scope until the `per_agent` reward path is
designed.

## 11. Implementation phases

| Phase | Scope | State | Evidence |
|---|---|---|---|
| 0 | Contract: all four headers, session body policy, SMG fields, SessionRouter normalization, snapshot dump, both capture modes. | Done | Rust tests cover the store fields and capture merge. A live Codex episode shows role, agent id, parent, owner, and capture in the snapshot. |
| 1 | First landing: `main_only`, four headers active, `tokens` for untrained subagents and auxiliary calls, owner-aware masks, token-only extraction, routed-expert zero-row padding, the first-version correctness fixes in section 10. | Done | A live subagent episode produces one main row. Its token-only subagent has no logprobs or routed experts, and its next turn still completes. |
| 2 | `all` and `selected`, `session_primary`, request-aggregated multi-row advantage and session loss weights. | Done | Live `all` episodes produce one row per trainable agent. Unit tests pin the advantage selection and the session-weighted loss. Config validation refuses `all`/`selected` outside request aggregation. |
| 3 | v2 landings: background/async live count with per-turn causal eligibility, logical workspace isolation, compact tree snapshot under a `format` field, role-aware trajectory resolution, per-agent reward and credit, shared-prefix forward if measurement justifies it. | Partial | The per-turn boundary, background count, compact `format`, logical isolation, and role-aware branch IDs are implemented and tested. Per-agent reward/credit and shared-prefix forward remain design-gated. |

Phase 1 is the first landing. Phase 2 is implemented in the same landing because
`all` and `selected` reuse the same extraction, selection, and policy plumbing;
the only new requirement is request aggregation, which the config validator now
enforces.

Phase 3 is the v2 design set. The concrete v2 items listed below are
implemented; per-agent reward/credit and the shared-prefix forward optimization
remain design-gated.

| v2 item | State | Evidence |
|---|---|---|
| Per-turn `contributes_to_session` boundary | Done | `TurnRecord.sequence`, primary-main boundary, mixed-row masking, `trainable_turns`, post-main exclusion tests. |
| Background subagent observation | Done | Main-close control, distinct live count, no wait/collection, delete drain excludes background. |
| Logical workspace isolation | Done | `workspace_isolation=logical` creates the agent root, states the convention in the prompt, and exports the workspace env; the main collector keeps its workdir. |
| Compact tree snapshot | Done | `format=compact` returns nodes/leaves; PSRL reconstructs the same trajectories and masks as `expanded`; live byte-size reduction measured. |
| Role-aware trajectory resolution | Done | A branch from another agent's live leaf gets a new id; main continuation reuses the main id, compaction creates a new context. |
| GC / live-leaf protection | Existing | `trajectory_leaves` protects every live leaf and its parent chain; subagent/retry tests pass. |
| Per-agent reward, credit assignment, subagent artifacts | Design-gated | `reward=per_agent` remains rejected until the semantics are defined. |
| Shared-prefix forward deduplication | Not started | Measure-first; no tree-packed attention until a workload measurement justifies it. |

### 11.1 Measured results

These are the checks that exercised the implementation. Conditions are kept
because a capture or extraction result is only meaningful with them.

The gateway changes are in the companion SMG branch `dev/subagent-training`.
Build and install that binding before running the PSRL branch; the stock
gateway does not know the four normalized headers or the per-turn capture
field.

**Test suites.** `cargo test -p smg-tito` passes 99 unit tests and 8 engine
tests. `cargo test -p smg --lib` passes 1350 tests. The Python suites
`tests/tito`, `tests/nixl/test_session_loss_agg.py`,
`tests/trainer/test_group_baseline.py`, `tests/trainer/test_chunk_manager.py`,
`tests/config/test_subagent_training_config.py`, and
`tests/agent_loop/test_harness.py` cover the extraction, policy, router,
advantage, loss, and adapter changes. That combined Python run passes 484 tests.
`tests/trainer/test_session_loss.py` adds 11 policy-loss, entropy, and KL tests
when the vendored verl checkout is present.

**Live stack, synthetic client.** Qwen2.5-0.5B-Instruct on one H20, a vLLM gRPC
worker, SMG with TITO, and the SessionRouter in process. The client drove the
subagent turns before the final main turn, so they are causally eligible.

| Case | Rows | Prompt | Response | Mask | Turns | Distribution |
|---|---|---|---|---|---|---|
| `main_only`, main | 1 | 40 | 57 | 43 | 2 | n/a (disabled sampler) |
| `all`, main | 1 | 40 | 38 | 24 | 2 | n/a |
| `all`, subagent | 1 | 69 | 41 | 27 | 2 | n/a |

The subagent prompt includes the shared main prefix; its response contains only
its own turns. The router sent `tokens` for the subagent in `main_only` and
`full` in `all`.

**Sampler capture.** The same stack with `SamplerCapture(mode="topk", budget=8)`
produced sampler distributions whose lengths matched the main (52 responses, 38
masked) and subagent (36 responses, 21 masked) rows exactly. `strict_validation`
accepted both.

**Capture cost and payload.** Six subagent turns on the same stack, with the
same seeds:

| Capture | Median turn latency | Snapshot size |
|---|---|---|
| `tokens` | 0.157 s | 6,484 bytes |
| `full` + top-8 sampler | 0.159 s | 19,747 bytes |

Token-only storage is 3.05x smaller on the wire for this six-turn subagent. The
turn latency is within noise on a dense 0.5B model with 16-token generations;
the engine still computes sampled-token logprobs. The benefit that scales is
the stored payload and the omitted sampler and routed-expert collection, not
short-turn decode time.

**Real harness episodes.** Codex 0.164.0 and Claude Code 2.1.295, each against
the same SMG and SessionRouter with a Qwen3-8B worker and a 32768-token context.
Both prompts asked the model to delegate through its subagent tool. Both
harnesses spawned one subagent.

| Harness | Main agent | Main rows | Main prompt / response / mask / turns | Subagent agent | Subagent prompt / response / mask / turns |
|---|---|---|---|---|---|
| Codex | catalog `multi_agent_version=v1` | 1 | 9195 / 1709 / 1009 / 3 | native `thread_id` | 7017 / 250 / 250 / 1 |
| Claude Code | `main` | 1 | 15425 / 2729 / 1493 / 3 | native `agent-id` | 9573 / 486 / 486 / 1 |

`main_only` selected one row and `all` selected two in both snapshots. The
subagent's `parent_agent_id` was the main agent in both. `parent_trajectory_id`
was absent because these subagents started a fresh context rather than sharing
the main prefix; the explicit parent agent id is what identifies them. The Codex
snapshot carried the main `thread_id` as its agent id, proving the native
metadata was normalized; the Claude Code subagent carried its native agent id,
proving `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1` exposed the request-class and
identity headers.

**Compact snapshot and per-turn gate.** The same live session requested both
wire formats. For `main_only` the expanded snapshot was 6,272 bytes and the
compact one 5,079 bytes; for `all` they were 6,091 and 5,097 bytes. The compact
and expanded trajectories, records, masks, and sequence values were identical.
A separate session drove a subagent after the main agent's final turn: the
subagent remained in the snapshot, every owned record had
`contributes_to_session=false`, and `all` selected only the main row.

**Extraction cost.** A synthetic 40-turn main trajectory with 2560 response
tokens took 0.56 ms in the owner-aware builder and 0.57 ms in the legacy path.
The builder is linear in tokens; the ownership checks add no measurable cost at
this scale.

**Not exercised live.** A full TransferQueue-to-optimizer training step was
not completed. The local stack reached actor and vLLM initialization, but this
environment has no NIXL plugins, so the worker failed before the first rollout;
the advantage, session-loss, and per-row weight math remain covered by the
unit and gradient tests. Routed-expert capture needs a model with experts; the
dense models above cannot produce it, and the assembler is covered by unit
tests for gaps, overlap, out-of-range starts, metadata mismatch, and zero-fill.
Compaction is covered by the classification, policy, and TITO branching tests,
but the live harness episodes did not force a compaction.

The live synthetic check is reproducible with `tests/tito/live_subagent_e2e.py`.
It expects a vLLM gRPC worker on port 51051 unless
`PSRL_TITO_E2E_START_VLLM=1` is set:

```bash
PSRL_TITO_E2E_MODEL=/models/Qwen3-8B \
PSRL_TITO_E2E_MAX_MODEL_LEN=32768 \
CUDA_VISIBLE_DEVICES=0 python tests/tito/live_subagent_e2e.py
```

## 12. Testing

The unit suites cover the behavior below. The live checks in section 11.1 cover
the end-to-end contract.

SMG:

- Owner assignment on a new node, on a repeated `Full` store, and on a `Tokens`
  to `Full` upgrade.
- Token-only records omit logprobs, sampler distribution, and routed experts but
  keep `output_token_ids`.
- A shared node appears in `trajectory_ids` for every path and has one owner.
- The snapshot round-trips the new fields, including the per-turn storage
  `sequence`.
- `drop_dead_leaves` keeps live subagent leaves. `trajectory_leaves` is the
  protection set, so a live leaf is never dead even when it is not the newest
  node in the tree.

PSRL:

- `main_only` returns one row and keeps the subagent in the snapshot.
- `all` returns one row per trainable agent. The shared prefix is mask 1 only in
  the owner row.
- `selected` filters by agent id.
- A contributing owned record without `Full` capture fails when
  `require_primary_capture` is set; a post-main token-only record does not.
- A token-only node is mask 0 in every sample.
- `session_primary` selects the main row for the advantage.
- A legacy snapshot without metadata matches the current extraction.
- `session-mean-token-mean` counts the shared prefix once, and the per-row
  `loss_weight` scales each session share.
- Request aggregation with one main and several subagent rows keeps the request
  as one chunk and one sample.
- A subagent sample uses the first owned record's prompt as `prompt_ids` and
  counts only owned records in `num_turns`.
- A mixed trajectory keeps its turn before `main_close_sequence` and masks the
  turn after it. `num_turns` counts both; `trainable_turns` counts one.
- A trajectory whose owned turns are all post-main produces no row.
- A main agent that resumes moves `main_close_sequence` later and makes an
  earlier subagent turn contributing again.
- A record with no `sequence` (legacy snapshot) is contributing.
- Expanded and compact snapshots produce the same per-turn masks.
- Routed experts are placed by `prompt_start`. A full, then tokens, then full
  sequence zero-fills the gap. A tokens final turn and a tokens-only trajectory
  are also covered.
- Parallel subagents get distinct trajectory ids and stable agent ids.
- A nested subagent's parent is the parent agent or the parent trajectory, not
  the session main.
- A subagent compaction follows the subagent's trainability, not the main's.
- `drop_dead_leaves` keeps the latest live leaf per agent.
- Validation scores the primary by default and can include subagents when
  `val_mode` selects them. Validation turns capture tokens only.
- A snapshot with a persisted policy selects rows from the snapshot, not from a
  different current config.

Integration:

- Run one Codex main and subagent episode against a local SMG gateway.
- Inspect `GET /tito/sessions/{id}` for the tree, roles, owners, capture, and
  the persisted policy.
- Run one training step with `main_only` and one with `all`.
- Run one parallel and one nested subagent episode.
- Run one compaction episode for the main and one for a subagent.
- Run one MoE episode to exercise live routed-expert capture.

## 13. Open questions

1. Resolved: Codex 0.164 emits a stable thread id and the parent thread id; the
   live Codex episode normalized both. A subagent that starts a fresh context
   has no tree-derived `parent_trajectory_id`; its explicit parent agent id is
   the authority.
2. Resolved: the harness runtime pins Claude Code 2.1.295 or newer. That version
   emits `x-claude-code-request-class`, `x-claude-code-agent-type`,
   `x-claude-code-agent-id`, and `x-claude-code-parent-agent-id`. The Claude
   Code adapter must set `CLAUDE_CODE_GATEWAY_HINT_HEADERS=1`.
3. Resolved: the session policy travels in the `POST /sessions` body, not in a
   header.
4. Resolved: the gateway is rebuilt from the modified SMG source before a run.
   The live checks used the rebuilt binding.
5. Should `output_token_ids` be emitted for `Full` records as well? This design
   emits it to keep the extraction uniform. It can be dropped for `Full` records
   later if wire size matters.
6. Resolved: Codex 0.164 honors the catalog's `multi_agent_version`; `v1` plus
   `[features] multi_agent = true` produced a live subagent call.
7. Resolved: there is no background drain or collection timeout. The main
   snapshot is immediate, background subagents are only counted, and session
   deletion does not wait for them.
8. Should the first version disable `tito_drop_dead_leaves` for subagent
   sessions, or implement the per-agent protected-leaf rule in the store?
   `trajectory_leaves` already protects every live leaf, including a subagent
   leaf and its parent chain; no change is needed for the first version.
9. Resolved: parallel subagents use logical isolation. One sandbox, a main task
   root that is the only collected artifact, per-agent
   `/workspace/agents/<agent_id>/` directories stated in the prompt and exposed
   through `PSRL_AGENT_WORKSPACE_ROOT` / `PSRL_MAIN_WORKSPACE`. The boundary is
   advisory; sandbox and worktree isolation remain deployment options for tasks
   that cannot accept the convention.
10. Resolved: the compact snapshot uses a top-level `format` field.
    `expanded` is the current view and the default when the field is absent;
    `compact` is the tree form, selected explicitly. An unknown format is an
    error.
11. Resolved: contribution is per turn, not per trajectory. TITO assigns a
    storage `sequence`; the boundary is the last record on the primary main
    trajectory; a subagent record after it is `contributes_to_session=false`.
    A mixed trajectory keeps its early contributing turns and masks the late
    ones. A turn that completed before the boundary but whose result main never
    read remains coarse session credit; detecting that is a later refinement.
    A fully post-main trajectory needs the per-agent reward path to train as its
    own sample with its own verifier.
