# Math RL with the external verifier

Train Qwen3.8-27B on the low and medium difficulty slice of the Open Problem Pool
with GRPO and score centering. The agent runs Codex in a p3b sandbox, writes a proof
to `/workspace/solution.md`, and the external verifier returns one verdict in
`{0, 1, 6, 7}`. One verdict is one reward.

## Placeholders

| Placeholder | Meaning |
| --- | --- |
| `${PSRL_WORKSPACE}` | workspace that holds models and `env/env.sh` |
| `${OPP_DATASET_ROOT}` | checkout of the Open Problem Pool |
| `${MATH_RL_ASSETS_ROOT}` | directory that contains the Rethlas checkout |
| `${PSRL_P3B_ENDPOINT}` | p3b control plane, `host:port` |
| `${MATH_RL_VERIFIER_URL}` | verifier endpoint ending in `/verify` or `/verify/stream` |

## Quickstart

**1. Build the sandbox image.** The Rethlas base image already ships Codex, Python,
and the PDF tools. The derived image clears the Rethlas entrypoint so p3b can start
it with `sleep infinity`, and adds NumPy, SciPy, and SymPy.

```bash
export MATH_RL_ASSETS_ROOT=<directory that contains Rethlas>
bash examples/math_rl/prepare/build_image.sh
bash examples/math_rl/prepare/check_image.sh math-rl-codex:local
```

**2. Project the data.** The default keeps difficulty 1 and 2 and every status.
Status is metadata, not a filter, because the verifier is the reward authority.

```bash
export OPP_DATASET_ROOT=<checkout of the Open Problem Pool>
python -m examples.math_rl.project_dataset
```

Output is `examples/math_rl/data/train.parquet`, `val.parquet`, and `manifest.json`.
The script prints the manifest to stdout.

**3. Point at the verifier.** The recipe needs the endpoint, the version the
deployment serves, and a model or a profile.

```bash
export MATH_RL_VERIFIER_URL=http://<verifier-host>/verify
export MATH_RL_VERIFIER_VERSION=V3
export MATH_RL_VERIFIER_MODEL=glm-5.3
export MATH_RL_VERIFIER_EFFORT=max
```

The client follows `/verify/stream` by default, which reports the contract, audit,
and grading stages, and falls back to a blocking post when
`MATH_RL_VERIFIER_TRANSPORT=blocking`. A request body is capped at 16 MiB by the
service, so the solution limit is 8 MiB. Retries happen only when no stage started,
because a retry re-runs three model stages and is billed again.

A served deployment reports its network policy in `/health`. `serve` always enables
hosted web search and sandbox network access, and the offline policy exists only in
the `benchmark` command. Record the policy that was chosen before a run, because
verdicts from a live-search deployment can depend on current web results.

The launcher checks `/health` before training and stops when the served version
differs from `MATH_RL_VERIFIER_VERSION`, or when the requested model is missing or
not configured.

Run the preflight before a run. It checks the deployment and grades one correct and
one broken synthetic proof, so no dataset problem leaves the machine.

```bash
python -m examples.math_rl.verify_probe
```

For a smoke run without a deployment, start the mock verifier with
`MATH_RL_VERIFIER_MODE=mock`. The launcher then serves the same four field contract
from the driver.

**4. Train.**

```bash
export PSRL_P3B_ENDPOINT=<p3b control plane>
export MATH_RL_RUNTIME_IN_IMAGE=true
bash examples/math_rl/run_qwen3.8-27b_math_codex_megatron.sh
```

Run two steps first with `TOTAL_TRAINING_STEPS=2` and check `actor/sc_*` in the log
before starting a long run.

## Reward mapping

| Verdict | Meaning | Graded | Binary |
| --- | --- | --- | --- |
| 7 | Fully correct | 1.0 | 1.0 |
| 6 | Complete with a minor non-material defect | 0.8 | 1.0 |
| 1 | Meaningful progress with a major gap | 0.3 | 0.0 |
| 0 | No meaningful progress | 0.0 | 0.0 |

`MATH_RL_REWARD_MODE` selects `graded` or `binary`. Graded is the default because a
strict binary gate makes most groups on open problems all-zero, and a zero-variance
group is dropped and refilled rather than trained.

## Files

| Path | What it holds |
| --- | --- |
| `audit_opp.py` | Schema and RL readiness audit for the source JSONL |
| `project_dataset.py` | Filter, normalize, and write the training parquet |
| `prompt.py` | The single task prompt builder |
| `agent_loop.py` | `MathHarnessAgentLoop`: sandbox, proof collection, verifier call |
| `verifier.py` | External verifier client, receipt parsing, verdict mapping, process cache |
| `mock_verifier_server.py` | Local verifier for tests and smoke runs |
| `reward.py` | Projects the verdict metadata into the PSRL reward loop |
| `config/math_harness_config.yaml` | Codex, sandbox, and verifier settings |
| `prepare/` | Sandbox image and runtime checks |
| `run_qwen3.8-27b_math_codex_megatron.sh` | Training launcher |
| `tests/` | Projection, prompt, verifier, reward, and loop tests |

## What the recipe fixes

- **Batch.** 32 prompts per step, 8 rollouts per prompt, 32 prompts per optimizer update.
- **Algorithm.** GRPO with `group_baseline_valid_only`, plus score centering
  (`score_centering`, `bypass_mode`, `loss_type=reinforce`) and TIS at 2.0.
- **Zero-variance groups.** `dynamic_sampling_filter` drops and refills groups with no
  reward variance.
- **Harness.** Codex runs with `subagents_enabled: false` and web search disabled by the
  adapter. Compaction triggers at 75% of the context window.
- **Verifier.** Exactly four request fields. The problem comes from the dataset row,
  never from the agent. Receipts are parsed from response headers.
- **Ungraded.** A timeout, a 5xx, a rejected request, a malformed body, or a proof over
  the payload limit marks the episode ungraded. The response mask is zeroed and the
  group baseline excludes it. A returned 0 or 1 is a measured failure and stays graded.

## Gotchas

- **The verifier is slow.** A call can run for many minutes. `GRADING_TIMEOUT` is
  separate from the episode timeout and defaults to `3 x stage timeout + 180`.
- **The version must match.** The service serves one harness version per process and
  rejects a request that names another. Pin both sides to the same value.
- **The image is node local.** Building it on the driver does not distribute it. Build
  or load `math-rl-codex:local` on every sandbox node, or push it to a registry the
  nodes can pull from.
- **The Rethlas image entrypoint is not usable under p3b.** p3b starts a sandbox with
  `sleep infinity` and does not mount `/opt/rethlas-docker`. The derived image clears
  the entrypoint.
- **The source has no ground-truth solutions.** Every record has an empty `solution`.
  The reward is the verifier, not a reference answer.
- **Every p3b node needs the image.** Same as above. A missing image turns every
  create into a pull.
- **Codex in the image versus a mounted runtime.** If the image ships Codex, set
  `MATH_RL_RUNTIME_IN_IMAGE=true` and `MATH_RL_HARNESS_RUNTIME_MOUNT=/usr/local`. If it
  does not, set `MATH_RL_RUNTIME_IN_IMAGE=false` and
  `PSRL_HARNESS_RUNTIME_ROOT=<runtime root>`.
- **Score centering with the harness is a new combination.** Start with two steps and
  check `actor/sc_correction`, `actor/sc_sampler_head_mass`, and `actor/sc_train_head_mass`
  before committing to a long run.
