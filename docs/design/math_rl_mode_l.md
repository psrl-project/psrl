# Mode L: training on the Rethlas loop

Mode S trains one Codex run per episode and rewards the final proof. Mode L keeps the
same reward authority and replaces the one-shot run with the Rethlas generation loop:
the agent drafts, asks for verification, reads the critique, and repairs, up to a fixed
number of rounds. This document is the implementation plan for that environment.

## 1. Goal and success criteria

One episode runs the Rethlas generation agent inside a p3b sandbox, with an external
verifier reachable through a per-episode proxy. The agent may repair its proof between
rounds. The reward is the verifier verdict on the final proof, unchanged from Mode S.

Success for the first landing:

1. The inner loop runs end to end inside the sandbox and stops on a verified blueprint
   or on the round cap.
2. The proxy answers every in-loop verification, serves repeats from cache, and refuses
   calls beyond the per-episode cap.
3. The finalizer grades the collected proof with the same client and mapping Mode S uses.
4. A two-step smoke on the mock verifier trains without ungraded episodes, and the
   validation reward matches the verifier's correctness rate.

## 2. What Mode S already provides

Mode L reuses most of Mode S. Nothing in the reward path changes.

| Piece | Module | Role in Mode L |
| --- | --- | --- |
| Verifier client | `examples/math_rl/verifier.py` | The proxy and the finalizer both call it |
| Reward mapping | `examples/math_rl/reward.py` | Unchanged |
| Prompt and paths | `examples/math_rl/prompt.py` | The task prompt feeds the round-0 message |
| Harness loop | `examples/math_rl/agent_loop.py` | Gains a sibling loop for the loop environment |
| Image and config | `examples/math_rl/prepare`, `config/` | The image gains the Rethlas tree and venvs |

## 3. Architecture

### 3.1 The sandbox is the Rethlas workspace

The image layers the Rethlas checkout on top of the Mode S image. The sandbox workspace
holds `agents/generation`, `agents/verification`, `tools`, and the selected problem under
`data/`. `docker/run-task.sh` is not used. It creates a sibling container and mounts
host venvs, neither of which exists inside p3b. The inner entry point is the existing
`agents/generation/tests/run_example.sh`, which already implements the round loop:
`codex exec` for round 0 and `codex exec resume` for later rounds, stopping on
`results/<problem>/blueprint_verified.md`.

The venvs the inner entry needs are built into the image at build time, not on first
use, so an episode does not pay for `uv` installs. `MAX_ITERATIONS` comes from the loop
config, and the round budget is part of the episode contract.

### 3.2 One conversation, resumed

The round loop resumes one Codex session, so the whole episode is a single TITO
conversation. The session router needs no multiplexing, and the gateway sees one
trajectory per episode. The harness writes `model_provider psrl` into the Codex config
as it does today, with the session root URL as the base URL, so every round is served
by the engines at the pinned model version.

Subagent threads are the exception. The Rethlas generation config enables `multi_agent`
with a subgoal prover, and each subagent thread is a separate conversation that would
land on the same session router and interleave with the trained trajectory. The first
landing sets `multi_agent = false`. Re-enabling it requires session multiplexing and a
way to mark subagent tokens as environment calls.

### 3.3 The verifier sits behind a per-episode proxy

The generation agent calls an MCP tool, `verify_proof_service`, which posts to
`VERIFY_PROOF_URL`. In Rethlas that URL is the local verification service. In Mode L it
points at a proxy owned by the agent loop worker:

- The proxy runs in the worker process and binds an address the sandbox can reach.
- The sandbox URL carries the episode identity, `VERIFY_PROOF_URL=<proxy>/verify/<uid>`.
- The proxy keeps a per-episode cache keyed by the proof hash, so a repeated draft costs
  nothing and a repair that does not change the statement reuses the verdict.
- The proxy enforces a per-episode verification cap and a token budget, and turns a cap
  hit into a definite answer the agent can read rather than a hang.
- The proxy forwards to the external verifier with the same four-field request the Mode S
  client sends, and it shapes the answer into the Rethlas schema the agent expects:
  `verification_report.summary` from the verifier reason, `repair_hints` from the same
  text, and `verdict` mapped from the 0/1/6/7 scale to pass or fail.
- The finalizer reuses the same cache entry for the final proof, so the reward authority
  is the same verifier call the agent saw.

The verifier never joins TITO and never consumes training tokens. It stays the reward
authority, and the proxy is the only path from the sandbox to it.

### 3.4 The reward is the final proof

The finalizer reads `blueprint_verified.md` when present and otherwise the latest
`blueprint.md`. It hashes the exact bytes, grades the dataset problem statement with the
verifier client, and records the verdict, reason, receipt, and proof hash. A finished
episode without a proof is a graded zero. A verifier failure is ungraded and stays out of
the loss and the group baseline, exactly as in Mode S.

## 4. Work breakdown

Each phase is independently testable.

### P0. Episode verifier proxy

- New module `examples/math_rl/verify_proxy.py`: an aiohttp app with a route per episode,
  an LRU cache keyed by the proof hash, a per-episode call counter, and a token budget.
- It calls the existing client and maps the verdict into the Rethlas response schema.
- Tests: cache hit skips the client, cap rejects with a readable body, schema mapping
  round trips, a transport failure surfaces as a retryable error.

### P1. MCP adapter

- `agents/generation/mcp/server.py` reads `VERIFY_PROOF_URL` instead of a fixed local URL
  and forwards the proxy answer unchanged.
- Tests: the tool builds the request from the proxy URL, and a rejected call returns the
  proxy's message rather than raising.

### P2. Command harness kind

- A new harness kind that runs a script instead of a single CLI invocation, parses the
  JSONL event stream the CLI writes, and collects artifacts from the workspace.
- The existing Codex harness already writes the per-episode Codex config. The command
  harness reuses that config and only changes the entry point.
- Tests: the command line is built from config, a non-zero exit maps to a harness
  failure, and artifacts are collected from the expected paths.

### P3. Loop

- New `MathRethlasAgentLoop` in `examples/math_rl/agent_loop.py`: prepares the workspace,
  starts the proxy, stages the problem under `data/`, runs the inner entry, and collects
  the proof.
- Config block: `max_iterations`, round timeout, total episode timeout, proxy caps,
  verifier settings.
- Tests: preparation writes the problem and config, collect picks the verified proof
  first, and finalize reuses the verifier client.

### P4. Image, config, launcher

- `prepare/Dockerfile` adds the Rethlas tree and builds the generation venv at build time.
- `config/math_harness_config.yaml` gains the loop entry and the round caps.
- The launcher gains the loop switch, so Mode S and Mode L share one script.

### P5. Verification

- Unit and integration tests as listed above.
- A single-episode run inside one sandbox with the mock verifier, then a two-step smoke
  with train batch 4 and two rollouts per prompt.
- Record the reward mean, the round distribution, the proxy hit rate, and the ungraded
  count in `FINDINGS.md` beside the launcher command.

## 5. Cost control

Rounds multiply cost, so the caps are part of the design rather than tuning:

- Round cap per episode.
- Per-round generation budget through the existing response length limit.
- Proxy verification cap per episode, with cache hits free.
- Episode wall-clock timeout, aligned with the sandbox lifetime.

Measure the pass rate and the reward variance before scaling the round cap. A round cap
of one reduces Mode L to Mode S with the Rethlas prompt and tools.

## 6. Risks and open decisions

- Subagent threads. They are one conversation each, and they would interleave with the
  trained trajectory. The first landing disables them. Supporting them needs session
  multiplexing and a token attribution rule.
- Repair feedback quality. The external verifier returns a verdict and a reason, while
  the Rethlas verifier agent returns structured findings. The proxy maps the reason into
  the report and repair hints. If repair quality proves insufficient, the option is a
  private verifier deployment that returns structured findings.
- Verifier cost and latency. A loop can call the verifier several times per episode.
  The cap and the cache bound it, and the first smoke measures it.
- Episode length. Longer episodes raise staleness pressure and refill rates. The round
  cap and the trajectory timeout bound the tail.
- Image size. The Rethlas tree and its venvs add hundreds of megabytes. Build them into
  the image once, not per episode.
