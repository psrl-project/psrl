# Mode S full run on the real verifier

This document is the execution plan for the first full Mode S run: one Codex episode
per rollout, graded by the deployed Bench Verifier at harness `V3` with `glm-5.3`, over
the difficulty 1 and 2 slice of the Open Problem Pool. It covers the contract the service
actually exposes, the client work that contract implies, the run configuration, the
preflight gates, and the operating limits.

## 1. The service contract as deployed

`GET /health` on the deployment reports:

| Field | Value | Meaning |
| --- | --- | --- |
| `verifier_harness_version` | `V3` | Requests must name `V3` |
| `models.glm-5.3.configured` | `true` | The GLM credentials are present |
| `models.glm-5.3.reasoning_effort` | `max` | Default effort for the stage agents |
| `web_search` | `live` | Hosted web search is enabled |
| `network_access` | `true` | Stage sandboxes have general network access |

The request body has exactly four fields: `problem`, `solution_to_verify`,
`model_config`, and `verifier_harness_version`. `model_config` accepts `profile`,
`model`, `reasoning_effort`, and `timeout_seconds`. The body must stay under 16 MiB.
A successful response is `{"verdict": "0|1|6|7", "reason": "..."}` plus receipt headers:
the run id, the harness, prompt and skill digests, the duration, and a usage object.
An infrastructure failure answers 503 with no verdict, which the trainer must treat as
ungraded rather than as a zero.

`POST /verify/stream` is the same job over Server-Sent Events. It emits `progress`
events for `accepted`, `contract`, `audit`, and `grading`, then one terminal `result`
event carrying the same verdict, reason, and receipt, or an `error` event with the
stage that failed. There are no heartbeats, so silence does not mean progress.

The per-stage limit comes from `model_config.timeout_seconds`. The worker's total limit
is three times that value plus 120 seconds, and executor queueing adds further wait.

## 2. The network policy decision

The deployed service runs in `serve` mode, and `serve` forces `web_search="live"` and
sandbox network access on. There is no request flag that turns them off, and the offline
policy exists only in the `benchmark` command. That conflicts with the earlier decision
to keep data egress on a private deployment with web search disabled. Three paths:

1. Private deployment with a serve-mode offline switch. Run this same image on a private
   host and patch `versions.py` to honor an environment switch that forces the
   `benchmark` policy, so `serve` reports and enforces `web_search="disabled"` and
   `network_access=false`. This is a small, reviewable change, and it is the only path
   that satisfies the original constraint while keeping the HTTP service.
2. Use the public deployment as it is. Problems and proofs go to the provider either
   way, and the dataset is a public benchmark, so the added exposure is the verifier's
   web lookups. The cost is grading that can depend on live search results, which makes
   verdicts less reproducible and lets a public solution influence the grade.
3. Grade through the `benchmark` worker instead of the HTTP service. It enforces the
   offline policy, but it is a batch driver, not a per-episode service, so the trainer
   would need a new integration.

The plan below assumes path 1 for the full run and path 2 only for a small pilot if the
private deployment is not ready. The preflight asserts `/health` reports the policy that
was chosen, so the two cannot be confused at run time.

## 3. Client work

The current client already speaks the four-field contract, sends `timeout_seconds`, maps
verdicts, and treats 503 as ungraded. The full run needs five changes.

1. Streaming transport. Add an SSE path that reads `progress` events for stage tracking
   and takes the verdict from the terminal `result` event. Keep the blocking transport
   behind the same interface so a deployment without streaming still works.
2. Stage-aware timeouts. Bound the stream by the worker's total limit, not by the stage
   limit, and surface which stage was active when a timeout or disconnect happened.
3. Body cap. Enforce the 16 MiB limit on the whole request body, not just on the
   solution. Set the solution cap to 8 MiB so the problem and the JSON envelope fit.
4. Retry policy. Retry only a connection failure before the stream starts, and never a
   mid-stream error, because a retry re-runs three model stages and is billed again.
5. Usage capture. Parse the receipt from the terminal event and record duration, usage,
   and the stage timeline in the reward metadata, so cost and latency are measured per
   episode instead of estimated.

## 4. Run configuration

| Knob | Value | Reason |
| --- | --- | --- |
| `MATH_RL_VERIFIER_URL` | `/verify/stream` on the chosen host | Stage visibility |
| `MATH_RL_VERIFIER_VERSION` | `V3` | The served version |
| `MATH_RL_VERIFIER_MODEL` | `glm-5.3` | Requested model |
| `MATH_RL_VERIFIER_EFFORT` | `max` | Matches the server default |
| `MATH_RL_VERIFIER_STAGE_TIMEOUT_S` | 900 | 15 minutes per stage, 47 minutes total |
| `GRADING_TIMEOUT` | 2900 | Above the worker total limit |
| `MATH_RL_VERIFIER_MAX_CONCURRENCY` | 16 | Episode concurrency plus refills |
| `MATH_RL_VERIFIER_MAX_RETRIES` | 1 | One retry only on a pre-stream failure |
| `MATH_RL_REWARD_MODE` | `graded` | Keeps group variance |
| Train batch / rollouts | 4 / 4 | First full run, then widen |
| Staleness | 1 | Absorbs long episodes |
| Steps | 20 | Two epochs over 81 rows at batch 4 |

The training window stays at the smoke settings: prompt 4096, response 16384, dynamic
token batching, trainer TP 4 and PP 2. The rollout context is 20480 tokens, so a Codex
episode fits the training window without truncation.

## 5. Preflight gates

Run these before the first step, and abort the run if any fails.

1. `/health` names `V3`, reports `glm-5.3` as configured, and reports the chosen network
   policy.
2. Two known cases go through the real client: a correct proof returns 7 or 6, a
   deliberately broken proof returns 1 or 0, and both come back with a receipt and a
   usage object.
3. The stream path reports stage transitions in order and terminates with a `result`
   event whose verdict matches the blocking path for the same input.
4. The measured latency and usage from those calls feed the cost estimate, and the
   estimate stays under the run budget.
5. One episode runs end to end inside a sandbox with the real verifier, and the reward
   metadata carries the verdict, the run id, and the usage.

## 6. Scope and cost

The dataset is the difficulty 1 and 2 slice: 81 train rows and 21 validation rows.
Each training episode is one Codex run plus one verification, so a step at batch 4 and
four rollouts costs 16 verifications, and a 20-step run costs 320. Refilled groups add
to that, because an all-equal group is discarded and refilled. The pilot measures the
refill rate, and the cost model multiplies by it.

Verification latency dominates step time at batch 4. The pilot records the p50 and p95,
and the step count and batch size follow from the measured rate rather than the reverse.

## 7. Monitoring and abort criteria

Record per step: reward mean and histogram over verdicts, ungraded count, verifier
latency percentiles, verifier usage, refill and consecutive group failures, overlong
filtering, queue depth, and GPU memory.

Abort and inspect when any of these holds for a step:

- Ungraded episodes exceed two percent, which points at the service or the client.
- Verifier p95 latency exceeds 25 minutes, which makes the run uneconomical.
- The all-equal group rate exceeds half, which means the reward is not separating.
- Overlong filtering removes more than a quarter of episodes, which means the response
  budget is too small for the task.

## 8. Rollback and resumption

Checkpoints land every ten steps under the run directory, and the trainer resumes from
them. A verifier outage stops the run rather than filling the buffer with ungraded
episodes, because ungraded episodes are masked and the step would not advance. The
launcher keeps the mock verifier path, so a broken deployment can be isolated from a
broken environment.

## 9. Open decisions

- Which network policy path from section 2 the first full run takes.
- Whether `glm-5.3` at `max` effort stays affordable at the measured usage, or the
  effort drops to `high`.
- Whether the first full run uses the graded mapping or the binary mapping once the
  pilot shows the verdict distribution.
