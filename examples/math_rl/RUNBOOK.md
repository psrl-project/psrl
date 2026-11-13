# Math RL training runbook

This runbook is the handoff for running the Qwen3.8-27B math RL recipes. It covers the
current state, the p3b sandbox service, dataset preparation, the verifier contract, and
the two training modes. It ends with the known limits, a troubleshooting table, and a
file map.

Values written as `<...>` are deployment values. Set them in your shell or in the
workspace `env.sh`. Do not commit real endpoints, proxy hosts, or cluster addresses.

## 1. Current state

Two modes are implemented under `examples/math_rl`.

| Mode | Loop | What it runs | Status |
| --- | --- | --- | --- |
| Mode S | `math_codex` | One Codex run per episode, then one verifier call on the collected `solution.md` | End-to-end validated with the mock verifier and exercised with the real verifier |
| Mode L | `math_rethlas` | The Rethlas generation loop with its MCP verification tool, then one verifier call on the final blueprint | Harness integration works, but the learning loop is not closed yet |

What works today:

- The external verifier client, the per-worker gateway, the preflight probe, and the
  finalizer are implemented.
- The Rethlas checkout is staged into the sandbox, the MCP tool is pointed at the
  episode gateway, the OPP references and literature review are staged, and the final
  artifact is graded.
- TITO stores multi-turn training sessions. The Qwen3.8 whole-slice fix is in the SMG
  `psrl-dev` branch and in the rebuilt extension.
- The verifier preflight graded a correct and a broken synthetic proof through the real
  deployment.

What is still missing for Mode L:

- The agent usually stops before writing `blueprint.md`. The finalizer records
  `missing_solution` and the reward is zero. The smoke uses `MAX_ITERATIONS=1`.
- A 64k actor update is not stable yet. See section 9.
- The subagent tools are enabled and the policy is `main_only`, but no model run has
  actually spawned a subagent, so the runtime capture has not been exercised.
- The verifier is slow and has no authentication, idempotency, or result lookup.

The math test suite is `pytest examples/math_rl/tests`. The SMG tests are the
`smg-tito`, `llm-tokenizer`, and `tito_e2e` suites in the SMG checkout.

## 2. What runs where

| Component | Where it runs | Role |
| --- | --- | --- |
| Training driver | One cluster node with access to the shared filesystem | Starts Ray jobs, the p3b client, and the verifier preflight |
| Ray cluster | The GPU nodes | Trainer, vLLM rollout engines, SMG gateway, reward loop |
| p3b control plane | Any host that can reach the sandbox nodes | Places sandbox creates on the fleet |
| p3b node agent | Every sandbox node | Drives the local Docker backend and creates containers |
| Sandbox image | Every p3b node | `math-rl-codex:local`, the Codex and Rethlas runtime |
| External verifier | A deployment endpoint | Grades a problem and proof through the four-field contract |
| Rethlas checkout | The driver filesystem, packed per episode | The generation loop, skills, and MCP server |
| Training environment | The Ray nodes | The PSRL Python environment with verl, vLLM, and SMG |

## 3. Prerequisites

- A Ray cluster with enough GPUs for the recipe. The reference layout is two nodes with
  eight GPUs each. The driver must reach the Ray head and the p3b control plane.
- The Qwen3.8-27B checkpoint and its tokenizer on the shared filesystem.
- The PSRL Python environment with editable `psrl`, `verl`, `vllm`, `vllm_patches`, and
  `smg`. The current cluster uses the base conda environment. The reproducible
  alternative is the GPU image in `docker/`.
- The p3b service and its Python SDK, started as described in section 4.
- `math-rl-codex:local` present on every p3b node.
- A verifier deployment with the model and version the recipe asks for.
- An HTTP proxy for verifier calls and package installs. Keep cluster hosts, the p3b
  endpoint, and the gateway host out of the proxy.
- Disk for logs, trajectories, and checkpoints. A full checkpoint is hundreds of
  gigabytes per save.

The dependency sources used by the current cluster are:

| Dependency | Path |
| --- | --- |
| verl | `<workspace>/psrl-wip/psrl/third_party/verl` |
| vllm | `<workspace>/psrl-wip/psrl/third_party/vllm-0.30.0` |
| vllm patches | `<workspace>/psrl-wip/psrl/patch/vllm/vllm_plugin/vllm_patches` |
| SMG | `<workspace>/psrl/third_party/smg-subagent` |
| PSRL | `<workspace>/psrl-wip/math-rl` for the recipe and `<workspace>/psrl-wip/psrl` for the shared install |

The verl and vllm code patches in the branch match the running checkouts. The SMG
extension must be built from a source that contains the Qwen3.8 TITO fix, otherwise
training sessions store only their first turn.

## 4. p3b sandbox service

The service is one control plane and one node agent per sandbox machine. The control
plane places creates, the node agents drive the Docker backend. The full configuration
reference is `p3b/README.md`.

### 4.1 Install

```bash
cd p3b
make install          # installs sandboxd into /usr/local/bin
make sdk-install      # installs the Python SDK used by the caller and the trainer
```

`make cluster` is the single-host example. A fleet uses separate planes.

### 4.2 Configure

Start from `p3b/example_node.json` and `p3b/example_control.json`. Every node config
uses `role: node`, a `node_listen` address, a node envelope, and the Docker backend.
The control config uses `role: control`, a `listen` address for the SDK, the fleet
ceiling, and a `fleet_nodes` list. The class names and backend declarations must agree
on both sides.

The node's Docker backend needs a running Docker daemon at the socket named in
`backends[].socket` and the sandbox image present locally.

### 4.3 Start the planes

Start every node agent before the control plane.

```bash
# on every sandbox node
sandboxd -preflight -config <node-config>
sandboxd -config <node-config>

# on the control host
sandboxd -preflight -config <control-config>
sandboxd -config <control-config>
```

For a long run, start each plane under systemd or `nohup` and keep its log. A preflight
failure names the missing piece, for example a Docker daemon that does not answer.

### 4.4 Verify the fleet

```bash
python p3b/verify_cluster.py <p3b-control-host>:9000 12
```

Or from Python:

```python
from sandboxd import SandboxClient

async with SandboxClient("<p3b-control-host>:9000") as client:
    print(await client.fleet())
```

The count must include every node you expect. The trainer reaches this endpoint through
`PSRL_P3B_ENDPOINT`.

### 4.5 Sandbox image

Build the base Rethlas image and the math derivative on a machine with Docker:

```bash
export MATH_RL_ASSETS_ROOT=<directory that contains Rethlas>
bash examples/math_rl/prepare/build_image.sh
```

The script builds `rethlas-codex:local` if it is absent, then builds
`math-rl-codex:local` from `examples/math_rl/prepare/Dockerfile`. The derivative adds
numpy, scipy, and sympy and clears the Rethlas entrypoint so p3b can start it with
`sleep infinity`.

The image is node local. Load it on every p3b node, for example by `docker save` and
`docker load`, or by pulling it from a registry the nodes can reach. The training
launcher does not copy the image to the fleet.

### 4.6 Clean stale sandboxes

The launcher runs one reclamation pass:

```bash
python -m psrl.sandbox.reclaimer --once
```

It is a warning when it fails. If it blocks because a dead session holds the service,
stop it and start the training run. The p3b lifetime is set by each create, so a caller
that dies cannot leave a sandbox running forever.

## 5. Dataset preparation

The source is an OPP JSONL file. The projection filters the easy slice, deduplicates
problems, writes the parquet, and materializes the problem and reference files.

### 5.1 Audit the source

```bash
python -m examples.math_rl.audit_opp \
  --input <opp-root>/datasets/omnibus_stable/omnibus_stable_1022.jsonl
```

Add `--json` for the full report. The audit checks the record schema and the fields the
projection needs.

### 5.2 Project the parquet

```bash
python -m examples.math_rl.project_dataset \
  --input <opp-root>/datasets/omnibus_stable/omnibus_stable_1022.jsonl \
  --output-dir examples/math_rl/data \
  --problems-root examples/math_rl/data/problems \
  --tokenizer <qwen3.8-27b-checkpoint>
```

Defaults:

- Keep difficulty 1 and 2.
- Keep every status.
- Deduplicate identical problem text.
- Cap the dataset prompt at 8192 tokens.
- Split 80 percent train and 20 percent validation with seed 42.

The command writes `train.parquet`, `val.parquet`, `manifest.json`, and
`problems/<uid>/problem.md` plus `problems/<uid>/references/references.md` and
`literature.md`. The manifest records the input hashes, filters, counts, and prompt
token summary. Repeat `--input` to combine sources. Add `--strict` to fail on an
invalid source record.

The `data/` directory is ignored by git. Regenerate it after changing the source or the
projection. The Rethlas loop reads the problem, the citations, and the full literature
review from the dataset row, so the materialized files are for provenance and audit.

### 5.3 Check the output

The reference omnibus projection keeps 102 rows and splits 81 train and 21 validation.
Check your manifest against the source you used. Every row must have `uid`, `problem`,
`difficulty`, and a non-empty problem. The reference files are not required for a row
with no citations and no review.

## 6. Verifier usage

The verifier is the reward authority. The client is `examples/math_rl/verifier.py`, the
per-worker gateway is `examples/math_rl/verifier_gateway.py`, and the preflight is
`examples/math_rl/verify_probe.py`. Nothing in PSRL core calls the verifier.

### 6.1 Environment

| Variable | Meaning | Reference value |
| --- | --- | --- |
| `MATH_RL_VERIFIER_URL` | External endpoint including `/verify` | `http://<verifier-host>/verify` |
| `MATH_RL_VERIFIER_VERSION` | Harness version the service serves | `V3` |
| `MATH_RL_VERIFIER_MODEL` | Model the service grades with | `glm-5.3` |
| `MATH_RL_VERIFIER_EFFORT` | Reasoning effort | `max` |
| `MATH_RL_VERIFIER_PROXY` | Proxy for the external call | `http://<http-proxy>` |
| `MATH_RL_VERIFIER_TRANSPORT` | `stream` or `blocking` | `stream` |
| `MATH_RL_VERIFIER_STAGE_TIMEOUT_S` | Per-stage service budget | `3600` for GLM-5.3 |
| `MATH_RL_VERIFIER_MAX_RETRIES` | Transport retries | `1` |
| `MATH_RL_VERIFIER_MAX_CONCURRENCY` | Per-process call cap | `16` |
| `MATH_RL_REWARD_MODE` | `graded` or `binary` | `graded` |

The launcher derives `GRADING_TIMEOUT` and `MATH_RL_RETHLAS_TOOL_TIMEOUT` from the stage
timeout as `3 x stage + 300` seconds. Raising the stage timeout raises both.

### 6.2 Preflight

```bash
export MATH_RL_VERIFIER_URL=http://<verifier-host>/verify
export MATH_RL_VERIFIER_VERSION=V3
export MATH_RL_VERIFIER_MODEL=glm-5.3
export MATH_RL_VERIFIER_EFFORT=max
export MATH_RL_VERIFIER_PROXY=http://<http-proxy>
export MATH_RL_VERIFIER_STAGE_TIMEOUT_S=3600

python -m examples.math_rl.verify_probe --case both
```

The probe checks `/health`, the served version, and the requested model, then grades one
correct and one broken synthetic proof. No dataset problem leaves the machine. A
non-zero exit means do not start.

For an offline smoke, set `MATH_RL_VERIFIER_MODE=mock`. The launcher then starts a local
mock verifier and uses the same four-field contract.

### 6.3 Contract and failure semantics

The request has exactly four fields:

```json
{
  "problem": "...",
  "solution_to_verify": "...",
  "model_config": {"model": "<model>", "reasoning_effort": "<effort>", "timeout_seconds": 3600},
  "verifier_harness_version": "V3"
}
```

The service returns verdict 7, 6, 1, or 0. The gateway maps 7 and 6 to correct and 1 and
0 to wrong, and returns the reason as repair hints. A valid 0 or 1 is a graded failure.
A transport error, a timeout, a rejected request, or an oversized body is ungraded and
excluded from the group baseline.

The client caches by problem bytes, proof bytes, model configuration, and harness
version. The gateway adds single-flight and a per-episode call budget. The finalizer
sends the dataset problem, not the agent-written statement. The gateway owns the model,
version, and effort, so the sandbox cannot change them.

### 6.4 Cautions

- The real deployment is public. It has no request authentication, no idempotency key,
  and no result lookup. Retrying a lost stream can repeat the full cost. Training
  problems and proofs leave the machine. Get owner approval before a real run.
- Live web search is enabled on the service. The same proof can receive different
  verdicts. Pin the deployment revision and measure variance before trusting a reward
  curve.
- GLM-5.3 is slow. Measured calls took 277 seconds to 1067 seconds, and one grading stage
  ran past thirty minutes. Use a stage timeout of 3600 seconds and a trajectory timeout
  that covers a full verification plus drafting.
- The service also serves `gpt-6.1-sol` and `gpt-6-astra`. A same-day measurement on the
  same deployment put `gpt-6.1-sol` at about 91 seconds for the synthetic correct proof,
  versus more than thirty minutes for GLM-5.3. Use the faster model for bulk rollouts
  only if you accept that the reward authority changes.
- The cache is per process. Two workers do not share verdicts. A restart loses the
  cache.
- The verifier is math-only in this repository. Do not add it to PSRL core.

## 7. Mode S training on Codex

Mode S runs one Codex episode per rollout and grades the collected `solution.md`.

### 7.1 Environment

```bash
export RAY_ADDRESS=<ray-head>:<port>
export LOCAL_IP=<driver-cluster-ip>
export PSRL_WORKSPACE=<workspace>
export HF_MODEL_PATH=<qwen3.8-27b-checkpoint>
export MATH_RL_DATA_DIR=<data-directory>
export PSRL_P3B_ENDPOINT=<p3b-control-host>:9000
export MATH_RL_SANDBOX_IMAGE=math-rl-codex:local
export MATH_RL_RUNTIME_IN_IMAGE=true
export MATH_RL_LOOP=math_codex
export STALENESS=0
export WANDB_MODE=offline
export LD_LIBRARY_PATH=<conda-lib>:$LD_LIBRARY_PATH
```

Set the verifier variables from section 6. `PSRL_WORKSPACE` must contain the cluster
`env.sh` that sets the NCCL and transport variables. If it does not, source an
equivalent environment first.

### 7.2 Smoke

```bash
export TOTAL_TRAINING_STEPS=1
export TRAIN_BSZ=2
export ROLLOUT_N=2
export TRAIN_MINI_BSZ=2
export GROUP_POST_PROCESS_ENABLE=false

bash examples/math_rl/run_qwen3.8-27b_math_codex_megatron.sh
```

The group filter is off because an untrained model can put every rollout in a zero
reward group and the filter would refill forever.

### 7.3 Full run

```bash
export TOTAL_TRAINING_STEPS=200
export TRAIN_BSZ=32
export ROLLOUT_N=8
export TRAIN_MINI_BSZ=32
export GROUP_POST_PROCESS_ENABLE=true

bash examples/math_rl/run_qwen3.8-27b_math_codex_megatron.sh
```

The default recipe uses tensor parallelism 4 and pipeline parallelism 2 for the actor,
four rollout instances with two GPUs each, and a 64k token sequence window. The window
is controlled by `MAX_PROMPT_LENGTH` and `MAX_RESPONSE_LENGTH`, which default to 16384
and 49152.

## 8. Mode L training on Rethlas

Mode L stages a Rethlas checkout into the sandbox and runs its generation, verification,
and repair loop. Only the main agent trajectory is trained. The subagent tools are
enabled and the policy is `main_only`.

The entry point is `examples/math_rl/run_qwen3.8-27b_math_rethlas_megatron.sh`. It
selects `math_rethlas`, applies a 3600 second verifier stage timeout and a 7200 second
trajectory timeout, and delegates the rest to the shared launcher. The shared launcher
stays the single source of the batch, optimizer, and deployment settings.

### 8.1 Prerequisites

- A Rethlas checkout with `agents/generation/tests/run_example.sh`.
- The sandbox image present on every p3b node. The image must contain Codex, uv, and
  Python 3.11.
- The verifier variables from section 6.
- A package proxy for the sandbox. The first episode may need it to build the MCP
  wheelhouse on the worker.

### 8.2 Episode flow

For every rollout the loop does the following:

1. Packs the Rethlas checkout and extracts it at `/workspace/rethlas`.
2. Writes `data/psrl/problem.md` from the dataset row.
3. Writes `data/psrl/problem.refs/references.md` and `literature.md` from the OPP
   citations and the full literature review.
4. Links `/workspace/results`, `/workspace/.scratch`, and `/workspace/.memory` into the
   checkout, so the paths named by the Rethlas skills resolve to the collected tree.
5. Stages the MCP runtime wheelhouse under `agents/generation/.psrl-wheelhouse`. The
   runner creates the venv and installs from the wheelhouse without the package index.
6. Registers the episode in the worker verifier gateway and rewrites the MCP server URL
   to the episode gateway.
7. Runs `agents/generation/tests/run_example.sh` through the command harness. Codex
   drives the main agent, `verify_proof_service` reaches the gateway, and the gateway
   reaches the external verifier.
8. Collects `blueprint_verified.md`, then `blueprint.md`, then the last assistant
   message.
9. Calls the verifier once on the collected artifact with the dataset problem, records
   the verdict and receipt, and releases the episode.
10. TITO stores the main agent trajectory. Subagent turns are captured as tokens and do
    not carry a loss.

### 8.3 Environment

| Variable | Meaning | Reference value |
| --- | --- | --- |
| `MATH_RL_RETHLAS_REPO` | Host Rethlas checkout | required |
| `MATH_RL_RETHLAS_MAX_ITERATIONS` | Generation and repair rounds | `3`, use `1` for the first smoke |
| `MATH_RL_RETHLAS_MAX_CALLS` | Verifier calls per episode | `4`, use `2` for a smoke |
| `MATH_RL_RETHLAS_VENV_PYTHON` | Python version for the MCP venv | `3.11` |
| `MATH_RL_RETHLAS_PROXY` | Package proxy inside the sandbox | `http://<http-proxy>` |
| `MATH_RL_VERIFIER_STAGE_TIMEOUT_S` | Verifier stage budget | `3600` |
| `TRAJECTORY_TIMEOUT` | Episode budget | `7200` |

Use the same base environment as section 7.1. The wrapper sets `MATH_RL_LOOP` to
`math_rethlas` and the experiment name to the Rethlas run. Pass `--help` to the wrapper
for the full list.

### 8.4 Smoke

```bash
export MATH_RL_RETHLAS_REPO=<rethlas-checkout>
export MATH_RL_RETHLAS_PROXY=http://<http-proxy>
export MATH_RL_RETHLAS_MAX_ITERATIONS=1
export MATH_RL_RETHLAS_MAX_CALLS=2
export TOTAL_TRAINING_STEPS=1
export TRAIN_BSZ=2
export ROLLOUT_N=2
export TRAIN_MINI_BSZ=2
export GROUP_POST_PROCESS_ENABLE=false

bash examples/math_rl/run_qwen3.8-27b_math_rethlas_megatron.sh
```

In a healthy run the episodes finish, the TITO tree contains every turn that the agent
produced, and the log has no `merge_incremental failed` or `prefix validation failed`
line. With the current one-iteration smoke the agent usually writes no blueprint, so the
finalizer logs `No solution for task ...` and the reward is a graded zero. This is the
known Mode L blocker, not a harness failure.

### 8.5 Full run

```bash
export MATH_RL_RETHLAS_REPO=<rethlas-checkout>
export MATH_RL_RETHLAS_PROXY=http://<http-proxy>
export MATH_RL_RETHLAS_MAX_ITERATIONS=3
export TOTAL_TRAINING_STEPS=200
export TRAIN_BSZ=32
export ROLLOUT_N=8
export TRAIN_MINI_BSZ=32
export GROUP_POST_PROCESS_ENABLE=true

bash examples/math_rl/run_qwen3.8-27b_math_rethlas_megatron.sh
```

The wrapper keeps the shared 64k window and the actor layout. Do not start a long run
until the 64k training limit in section 9 is resolved.

### 8.6 Artifact contract and current blocker

The generation agent must write `blueprint.md` or `blueprint_verified.md` under the
problem results directory. The prompt points the agent at
`data/<problem>.refs/references.md` and `literature.md`. A completed run with no
blueprint is a graded failure. Raising `MATH_RL_RETHLAS_MAX_ITERATIONS` gives the agent
a repair round and is the first thing to try. If the finalizer still records
`missing_solution`, inspect the runner log and the trajectory for the output path the
agent used.

## 9. The 64k training limit

The rollout side runs at 64k. The actor update at 64k is not stable on the current
layout:

- With `TRAIN_TP=4`, `TRAIN_PP=2`, `TRAIN_CP=1`, the actor finishes its updates and the
  next CUDA synchronization during optimizer offload or checkpoint save reports
  `CUDA error: invalid argument`.
- With `OFFLOAD=false` the failure moves to checkpoint save.
- With `TRAIN_TP=2`, `TRAIN_PP=2`, `TRAIN_CP=2`, a 56k sequence trains, then the fp32
  conversion of the output logits runs out of memory.

Do not start a long 64k training job until this is fixed. The next steps are to enable
Megatron cross entropy loss fusion or to move to a layout that shards both the sequence
and the vocabulary. The knobs are `TRAIN_TP`, `TRAIN_PP`, `TRAIN_CP`, and `OFFLOAD`.

## 10. Monitoring

The launcher writes to `$PSRL_WORKSPACE/output/math_rl/<timestamp>`:

| Path | Content |
| --- | --- |
| `train.log` | Console log, metrics, and errors |
| `math_harness_config.yaml` | The composed agent config for this run |
| `psrl/` | Framework logs |
| `trajectories/v0/` | One `.txt` and `.tree.json` per captured trajectory |
| `checkpoints/` | Model and optimizer checkpoints |

Check for:

- `Training completed successfully!` at the end.
- `Final validation metrics` with the reward mean, accuracy, and grading failure count.
- `critic/score` and `critic/rewards` in the step metrics.
- `training/num_turns` and `termination/finished/fraction`.
- In a Rethlas run, the trajectory tree depth against the episode turn count. Every turn
  should be stored as a `full` node.
- In the verifier path, `Graded task` with a verdict and run id, or `No solution for
  task` when the agent produced nothing.

A quick check of a stored trajectory tree:

```bash
python - <<'PY'
import glob, json
for path in glob.glob('<run-dir>/trajectories/v0/*.tree.json'):
    nodes = json.load(open(path))['nodes']
    print(path, [(n['id'], n['num_tokens'], n['stored'], n['capture']) for n in nodes])
PY
```

## 11. Known limits and next steps

- Mode L needs a blueprint-producing episode before it can learn. The current smoke uses
  one iteration.
- The 64k actor update needs a memory and kernel fix.
- The subagent runtime path has unit tests but no live episode that spawns a subagent.
- The verifier is slow and public. A dedicated deployment, a batch queue, a shared
  cache, or a faster model is needed for the full 32 by 8 step.
- The finalizer calls the verifier client directly. The gateway only serves the in-loop
  MCP tool. The design that makes the finalizer use the gateway, with a shared cache and
  a proof hash echo, is not implemented.
- Long sessions and compaction are only partially exercised. The verified TITO depth is
  a few turns.
- The Docker Hub and Volcengine images predate the SMG TITO fix. Rebuild before using
  them for training.

## 12. Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Verifier health or version failure | Wrong endpoint, version, or model | Check `MATH_RL_VERIFIER_*` and run the probe |
| Verifier timeout | Stage timeout below the deployment latency | Raise `MATH_RL_VERIFIER_STAGE_TIMEOUT_S` to 3600 |
| `TITO merge_incremental failed` or `prefix validation failed` | SMG lacks the Qwen3.8 whole-slice fix | Rebuild the SMG extension from `psrl-dev` at the fix commit |
| `No solution for task` | The agent wrote no blueprint | Raise `MAX_ITERATIONS` and check the runner log |
| Group refill loop | Every rollout has the same zero reward | Use a smoke with `GROUP_POST_PROCESS_ENABLE=false`, or fix the curriculum |
| `CUDA error: invalid argument` after the actor update | 64k actor update on the current layout | See section 9, do not start a long job |
| `CUDA out of memory` at the output conversion | 56k fp32 logits at TP2 | Use TP4 or loss fusion |
| p3b create failure | Image missing on a node, Docker down, or the endpoint is wrong | Check the image, the daemon, and the control plane fleet |
| `libnixl_build.so` not found in a Ray worker | The runtime env lost the conda library path | Add the conda `lib` to `LD_LIBRARY_PATH` before launch |
| Codex cannot reach the session endpoint | A proxy is set for cluster traffic | Keep cluster hosts in `no_proxy` and unset the proxy before Codex |
| Rethlas cannot find the result | The agent wrote outside the collected tree | Check the workspace path links and the runner log |
| Reclaimer hangs before launch | A dead sandbox session | Stop it and start the run |

## 13. File map

| Path | What it holds |
| --- | --- |
| `examples/math_rl/README.md` | Recipe overview and knobs |
| `examples/math_rl/RUNBOOK.md` | This handoff |
| `examples/math_rl/agent_loop.py` | Codex harness loop, finalizer, reward metadata |
| `examples/math_rl/rethlas_loop.py` | Rethlas staging, artifact collection, episode gateway |
| `examples/math_rl/rethlas_harness.py` | Runner script and MCP server wiring |
| `examples/math_rl/verifier.py` | External verifier client and cache |
| `examples/math_rl/verifier_gateway.py` | Per-worker MCP gateway |
| `examples/math_rl/verify_probe.py` | Deployment preflight |
| `examples/math_rl/project_dataset.py` | OPP projection and file materialization |
| `examples/math_rl/audit_opp.py` | Source audit |
| `examples/math_rl/config/math_harness_config.yaml` | Codex, Rethlas, sandbox, and verifier config |
| `examples/math_rl/run_qwen3.8-27b_math_codex_megatron.sh` | Shared training launcher |
| `examples/math_rl/run_qwen3.8-27b_math_rethlas_megatron.sh` | Rethlas training entry point |
| `examples/math_rl/prepare/` | Sandbox image build and checks |
| `p3b/README.md` | Sandbox service configuration and operations |
| `docker/Dockerfile` | Reproducible training environment image |
| `docs/design/math_rl_mode_l.md` | Mode L design and reward semantics |
| `docs/design/math_rl_rethlas_training.md` | Rethlas adapter design |

## 14. Handoff checklist

Record these values with every run:

- The `dev/math-rl-v2` commit and the SMG `psrl-dev` commit.
- The image digest and the sandbox image digest.
- The dataset manifest and input hashes.
- The verifier endpoint, version, model, effort, and deployment revision.
- The p3b control and node configs.
- The run log directory and the final validation metrics.
