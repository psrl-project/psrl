# Rethlas harness training: P1 and P2 implementation

This document is the implementation plan for the Rethlas loop environment (P1) and its
episode-level training semantics (P2). It builds on the Mode S verifier client and the
merged TITO subagent support, and it keeps every math-specific piece under
`examples/math_rl`.

## 1. Scope

P1 delivers a runnable Rethlas episode inside a p3b sandbox:

- a command harness that runs a workspace script instead of one CLI invocation,
- the Rethlas workspace staged into the sandbox from a host checkout,
- a sandbox-reachable verifier gateway that owns the problem, the budget and the cache,
- an MCP endpoint patch so the generation agent's repair loop calls the gateway,
- an artifact collector that reads the Rethlas result tree,
- OPP problem files materialized for `PROBLEM_FILE`.

P2 delivers the training semantics around that episode:

- primary-aware reward selection,
- per-episode verification budget,
- a step gate on the graded ratio,
- reward and verifier diagnostics carried through the existing reward path.

Non-goals for this change: a multi-replica Redis gateway, service-side idempotency or
result lookup, and a web-search policy change.

## 2. Architecture

```text
RethlasHarnessAgentLoop (worker)
  |
  +-- stage rethlas checkout, problem, runner script
  +-- register episode in VerifierGateway (worker HTTP endpoint)
  |       |
  |       +-- MCP verify_proof_service --> gateway --> external verifier
  |
  +-- CommandHarness runs run_rethlas.sh
  |       |
  |       +-- agents/generation/tests/run_example.sh
  |             codex exec / codex exec resume  (PSRL session)
  |
  +-- artifact collector: results/psrl/problem/blueprint*.md
  |
  +-- finalizer: VerifierClient.evaluate(problem, proof)
```

The gateway and the finalizer share one process, so the finalizer reuses the gateway's
cache entry for the final proof. Data is exchanged over the existing four-field external
contract.

## 3. Components

### 3.1 CommandHarness

`psrl/workers/agent_loop/harness/command.py`.

- `kind=command`, no executable probe against the runtime mount (the script reports its
  own failures), `_prepare` delegates the Codex config to `CodexHarness` when
  `codex_config` is on.
- `build_command` returns `[executable, *args]`. The prompt is not injected because
  the script owns its own first message.
- `build_env` copies the configured environment, the inherited proxy variables, and adds
  the session host to `NO_PROXY` so an in-cluster callback never goes through the proxy.
- `trajectory_format=plain_text`; turn counts come from TITO.

### 3.2 Rethlas harness adapter

`examples/math_rl/rethlas_harness.py`, registered as `kind=rethlas_command`.

- Extends `CommandHarness`.
- `_prepare` writes the Codex config (model provider, session URL, API key, catalog,
  web search disabled) and appends the generation agent's MCP server entry:
  `[mcp_servers.reasoning_agent]` with the venv interpreter and `agents/generation/mcp/server.py`.
- Writes the episode runner script into the harness state directory. The script:
  - exports the proxy from the staged file when present,
  - reads `VERIFY_PROOF_URL` and `VERIFIER_EPISODE_ID` from staged files,
  - sets `PROBLEM_FILE=data/psrl/problem.md`, `MAX_ITERATIONS`, `LLM_MODEL`, `CODEX_HOME`,
  - creates the generation venv with `uv` on first use,
  - activates it and runs `agents/generation/tests/run_example.sh`.

### 3.3 Rethlas episode loop

`examples/math_rl/rethlas_loop.py`, loop name `math_rethlas`.

- Extends `MathHarnessAgentLoop`.
- `prepare_harness_sandbox`:
  - packs the host Rethlas checkout (without `.git`) and extracts it at `/workspace/rethlas`,
  - writes `data/psrl/problem.md` from the dataset row,
  - patches the MCP server's `VERIFY_PROOF_URL` literal to the episode gateway URL,
  - writes the episode id, verifier URL and proxy files under the harness state directory,
  - registers the problem in the worker gateway.
- `collect_harness_artifact`: reads `results/psrl/problem/blueprint_verified.md`, falls back
  to `results/psrl/problem/blueprint.md`, then to the last assistant message in the run log.
- The inherited finalizer grades the collected proof against the dataset problem through
  the shared verifier client.

### 3.4 Verifier gateway

`examples/math_rl/verifier_gateway.py`, one instance per agent-loop worker process.

- `POST /verify/{episode}` accepts the Rethlas `{statement, proof}` shape, reads the
  problem from the episode registry, and calls the external service with the four-field
  contract.
- Returns the Rethlas verification schema: `verification_report`, `verdict` in
  `correct`/`wrong`, `repair_hints`.
- Verdict mapping: 7 and 6 are `correct`, 1 and 0 are `wrong`; the reason becomes the
  summary and the repair hint.
- Single-flight per proof hash, on top of the client cache.
- Per-episode call cap (`max_calls_per_episode`, default 4) and a token counter.
- `GET /health` and `GET /metrics` expose the external health, per-episode counters, the
  call count and the accumulated usage.

### 3.5 OPP problem materialization

`project_dataset.py --problems-root <dir>` writes `<dir>/<uid>/problem.md` and records
`problem_path` and `problem_sha256` in `extra_info`. The Rethlas loop reads the problem
from the dataset row, so the file is a staging aid rather than a second source of truth.

### 3.6 Primary-aware reward

`base_agent_loop.compute_reward_score` accepts `primary_index`. The harness loop passes
the index of the row marked `session_primary` by the TITO selection. The reward scalar is
still broadcast to every trainable row of the episode, so all of them share one GRPO
outcome, while the annotated reward metadata comes from the primary.

### 3.7 Step gate

`full_batch.run_step` computes the graded ratio from the per-row `terminate_reason` after
reward normalization. When it is below `algorithm.min_graded_ratio`, the step skips the
advantage computation and the actor update, records the metric, and logs the decision.
The default is zero, so existing recipes are unchanged.

## 4. Configuration

```yaml
- name: math_rethlas
  _target_: examples.math_rl.rethlas_loop.RethlasHarnessAgentLoop
  rethlas:
    repo_path: <host checkout>
    max_iterations: 3
    venv_python: "3.11"
  verifier: { ... same as math_codex ... }
  sandbox_config: { ... same as math_codex ... }
  harness:
    kind: rethlas_command
    executable: bash
    codex_config: true
    trajectory_format: plain_text
    args: ["<state dir>/run_rethlas.sh"]
```

The launcher selects the loop with `MATH_RL_LOOP=math_rethlas`.

## 5. Tests and acceptance

Unit:

- command harness command and env building, and the skipped executable probe,
- gateway payload mapping, verdict mapping, single flight, per-episode cap,
- Rethlas staging: problem file, runner script, MCP URL patch,
- artifact selection: verified, draft fallback, missing,
- dataset materialization and hashes,
- primary reward selection and the fallback,
- graded-ratio gate helper.

Integration:

- one episode with the mock verifier through the gateway: the MCP call is cached and the
  finalizer returns the same verdict,
- a two-step smoke with train batch 4 and two rollouts, `subagents_enabled=false`,
  `max_iterations=2`.

Acceptance: the final validation reward equals the verifier's correctness rate, no
verifier call appears in the training samples, and the graded ratio metric is recorded.
