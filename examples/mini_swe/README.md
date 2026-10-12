# mini-SWE-agent Training Recipe

Train language models to solve software engineering tasks using reinforcement
learning. This recipe integrates [mini-SWE-agent](https://github.com/SWE-agent/mini-SWE-agent)
(v2) with PSRL's trainer, enabling models to learn from
interactive coding feedback in Docker-sandboxed environments.

Three training paths are supported:

| Path | Dataset | Reward | Use case |
|------|---------|--------|----------|
| **Toy** | Synthetic bugs in `python:3.11-slim` | Patch-text overlap | Smoke tests, fast iteration |
| **SWE-smith-py** | Real GitHub bugs (50k SWE problems, per-repo images) | F2P / P2P test execution | Full RL training |
| **SWE-Gym** | Real GitHub bugs (2438 SWE problems, `xingyaoww` images) | F2P / P2P test execution (pre-computed eval_script) | Full RL training |

The **harness mode** (Claude Code / Codex) can replace mini-SWE-agent's native
loop on either real path. `megatron_qwen_4b_swe_cc.sh` is the ready-to-run
Claude Code recipe (Megatron, Qwen3.5-4B, SWE-Gym-293); see
[Harness training](#harness-training-preprocessing--bake).

---

## Overview

The training loop works as follows:

1. **Data**: Each training sample contains a problem statement and grading metadata.
2. **Rollout**: For each sample, PSRL dispatches one task to a local black-box
   runner function. The runner owns mini-SWE-agent, Docker, and grading.
3. **Model Routing**: PSRL creates one SMG TITO session per episode and gives
   mini-swe-agent the session-scoped OpenAI-compatible URL. mini-swe-agent then
   runs through its normal inference path, while SMG captures training tokens,
   masks, logprobs, and routed experts.
4. **Grading** (SWE-smith / SWE-Gym path): After the agent submits a patch, a fresh Docker
   container runs the SWE problem's FAIL_TO_PASS and PASS_TO_PASS tests.
5. **Reward**: Score is computed from the grading result (or patch-text overlap for
   the toy path) and used for GRPO policy gradient updates.

```
┌─────────────────────────────────────────────────────┐
│               PSRL GRPO Trainer                     │
│  (actor, ref model, vLLM rollout, reward scoring)   │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │ MiniSWEAgentLoopV1.run()│
          │  (async event loop)     │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼──────────────────┐
     │                 │                  │
     ▼                 ▼                  ▼
┌──────────┐   ┌──────────────┐   ┌────────────────────┐
│  Docker  │   │ SMG Session  │◄──│ mini-swe runner    │
│container │   │ Router + TITO│   │ Python bindings    │
│(rollout) │   └──────┬───────┘   └────────────────────┘
└──────────┘          │
               ┌─────┴────────┐
               │ vLLM rollout │
               └─────┬────────┘
                      │
               ┌──────┴────────────────┐
               │ grade_fresh_container │  ← SWE-smith / SWE-Gym paths
               │ (fresh Docker, pytest)│
               └──────┬────────────────┘
                      │
               ┌──────┴───────┐
               │ compute_score│
               └──────────────┘
```

---

## Directory Structure

```
examples/mini_swe/
├── README.md                             # This file
├── config.py                             # Runtime config dataclasses
├── runner.py                             # Black-box mini-swe/Docker/grader runner
├── reward.py                             # Reward function (patch-overlap + test-execution)
├── swebench_grader.py                    # Fresh-container grader orchestration for SWE-smith / SWE-Gym / Verified (shared by training + eval)
├── grading/                              # Host-independent grading payload (frozen eval script + vendored parser registry + in-sandbox driver)
│   ├── schema.py                         # GradingPlan built from swe_problem.eval_script / log_parser
│   ├── driver.py                         # Stdlib-only driver executed inside the grading sandbox
│   ├── payload.py                        # Packages driver + _vendor into grader.zip
│   ├── runtime.py                        # Host orchestration: write payload, run, read scorecard
│   ├── freeze.py                         # Prepare-time SWE-smith eval_script / parser freezing
│   └── _vendor/                          # Vendored swebench parsers + grading logic (see PROVENANCE.md)
├── utils/                                # Helpers kept out of the example's core modules
│   ├── harness_task.py                   # Harness prompt + patch collection helpers
│   ├── integrity.py                      # Post-rollout integrity scan (format-dispatched: Claude stream-json / Codex JSONL)
│   └── git_sanitize.py                   # Runtime git-leak probe + fallback purge for harness sandboxes
├── fsdp_qwen_7b_dapo.sh                  # Launch script — toy dataset (FSDP, 7B)
├── fsdp_qwen_14b_dapo.sh                 # Launch script — toy / DAPO path (FSDP, 14B)
├── fsdp_qwen_7b_swe_smith.sh             # Launch script — SWE-smith-py (FSDP, 7B)
├── fsdp_qwen_7b_swe_gym.sh               # Launch script — SWE-Gym (FSDP, 7B)
├── megatron_qwen_4b_swe_cc.sh            # Launch script — Claude Code harness (Megatron, Qwen3.5-4B, SWE-Gym-293)
├── megatron_qwen_4b_swe_smith.sh         # Launch script — SWE-smith (Megatron, 4B, native loop)
├── megatron_qwen_7b_swe_gym.sh           # Launch script — SWE-Gym (Megatron, 7B)
├── megatron_qwen_8b_swe_smith.sh         # Launch script — SWE-smith (Megatron, 8B)
├── megatron_qwen_8b_swe_gym.sh           # Launch script — SWE-Gym (Megatron, 8B)
├── megatron_qwen_32b_swe_smith.sh        # Launch script — SWE-smith (Megatron, 32B)
├── test_perf.sh                          # Performance / throughput probe
├── config/
│   ├── simple_agent_config.yaml          # Agent config for toy path
│   ├── swebench_agent_config.yaml        # Agent config for SWE-smith / SWE-Gym / Verified
│   ├── swebench_agent_config_xml_fc.yaml # XML function-calling variant (newer models)
│   ├── swebench_agent_config_full_sweagent.yaml # Full SWE-agent format (SWE-agent-LM-7B)
│   ├── swebench_harness_config.yaml      # Claude Code / Codex harness agent-loop config
│   ├── qwen_no_think_strip.jinja         # Qwen3 chat template (strips thinking)
│   └── qwen35_no_think_strip.jinja       # Qwen3.5 chat template (used by the cc harness script)
├── eval/                                 # Standalone evaluation (serving is in psrl/eval/)
│   ├── README.md                         # Guide for gold-patch sanity, multi-node eval, serving your own checkpoint
│   ├── eval_swebench.py                  # Single-node eval entry point
│   └── eval_swebench_multinode.py        # Hash-sharded cross-host eval launcher
└── prepare/
    ├── README.md                         # Data preparation guide (Path A, B, and C)
    ├── prepare_simple_data.py            # Toy dataset generator
    ├── simple_cases_train.json           # Synthetic training bug-fix tasks
    ├── simple_cases_val.json             # Synthetic validation bug-fix tasks
    ├── prepare_swebench.py               # HF → parquet converter (smith / verified / lite)
    ├── prepare_swe_gym.py                # HF → parquet converter (SWE-Gym / SWE-Gym-Subset)
    ├── prepare_swe_gym_293.py            # HF → parquet converter SkyRL-v0-293
    ├── swebench_subsets.py               # Repo-balanced sampling helpers
    └── docker_scripts/                   # Docker image pre-fetch / fan-out helpers
        ├── bake_simple_repos.sh          # Bakes toy repos into a Docker image (Path A)
        ├── bake_harness_image.sh         # Per-image git-purged derivative (harness mode)
        ├── rebake_harness_image.sh       # Force re-bake after a bake-step change
        ├── build_harness_runtimes.sh     # Fetch native Claude Code / Codex runtime trees (no Node)
        ├── prefetch_images.sh            # Pull per-SWE-problem images (skopeo + multi-mirror + tar cache)
        ├── prefetch_example.sh           # Reference invocation chaining prefetch + load_all_nodes
        ├── swe_smith.sh                  # Convenience wrapper for SWE-smith images
        ├── swe_gym.sh                    # Convenience wrapper for SWE-Gym images
        ├── swe_gym_subset.sh             # Convenience wrapper for SWE-Gym-Subset images
        ├── swe_gym_293.sh                # Convenience wrapper: SWE-Gym-293 train + val images
        ├── swe_eval_subset.sh            # Convenience wrapper for SWE-bench eval subset images
        ├── probe_mirrors.sh              # Check which public Docker Hub mirrors serve a given image
        ├── load_all_nodes.sh             # pssh fan-out of `docker load` across the cluster
        └── migrate_docker_overlay2.sh    # Move the Docker data-root/overlay2 onto a larger disk

# Core integration modules inside psrl/
psrl/workers/agent_loop/loops/session_agent_loop.py       # Shared SessionRouter/TITO lifecycle
psrl/workers/agent_loop/loops/mini_swe_agent_loop_v1.py   # Session-router/TITO black-box loop (native mini-SWE)
psrl/workers/agent_loop/loops/harness_agent_loop.py       # Generic sandboxed-harness lifecycle
psrl/workers/agent_loop/loops/mini_swe_harness_agent_loop.py # Mini-SWE task hooks for harness mode
psrl/workers/agent_loop/harness/                          # Harness adapters (claude_code / codex) + config
psrl/workers/agent_loop/agent_data/mini_swe_agent_data.py # MiniSWEAgentData
psrl/environments/mini_swe_env.py                         # Task metadata → observation adapter
psrl/sandbox/                                             # Backend-neutral runtime abstraction
examples/mini_swe/harness_adapter.py                      # Third-party synchronous harness adapter
examples/mini_swe/runner.py                               # Episode orchestration + SandboxSpec mapping
```

---

## Prerequisites

### Hardware

- NVIDIA GPUs (tested on A100, H100; 8× GPUs per node recommended)
- Sufficient disk space for checkpoints (~50 GB each)
- Docker on every worker node
- For SWE-smith-py: additional ~500 GB–1 TB per node for Docker image cache

### Software

```bash
# 1. Create conda environment and install PSRL (from repo root)

# 2. mini-SWE-agent (used as a library)
python -m pip install mini-swe-agent

# 3. Grading deps — needed to PREPARE data only (swebench/swesmith are used to
#    freeze eval scripts + parser names; the training host never imports them).
python -m pip install swebench==4.1.0 swesmith
python -m examples.mini_swe.grading.vendor_parsers   # regenerate vendored parser registry after version bumps

# 4. Verify (preparation deps + the host-independent grading payload)
python -c "from minisweagent.agents.default import DefaultAgent; print('mini-swe-agent OK')"
python -c "import swebench; print('swebench', swebench.__version__)"   # prepare-time only
python -c "from swesmith.profiles import registry; print('swesmith profiles:', len(registry.data))"   # prepare-time only
python -c "from examples.mini_swe.grading.payload import grader_zip_bytes; print('grading payload', len(grader_zip_bytes()), 'bytes: OK')"
docker run --rm python:3.11-slim bash -c "echo Docker OK"
```

---

## Data Preparation

All dataset preparation (toy, SWE-smith-py, and SWE-Gym) is documented in
[`prepare/README.md`](prepare/README.md). That file covers:

- **Path A** — baking toy repos into Docker and generating parquets
- **Path B** — converting SWE-smith-py and SWE-bench Verified from HuggingFace,
  generating balanced subsets, and pre-fetching per-SWE-problem Docker images on
  every cluster node
- **Path C** — converting SWE-Gym (2438 problems), SWE-Gym-Subset (100
  problems), or SWE-Gym-293 / SkyRL-v0-293 (293 train + 23 val) from
  HuggingFace, with pre-computed eval scripts and `xingyaoww` / `swebench`
  Docker images

Read that file before running training for the first time.

**Harness mode (Claude Code / Codex) needs two extra host-side preparations**
on top of the data and image steps: building the native harness runtime trees
(bind-mounted read-only, no Node) and optionally baking a per-image derivative
that purges leaked git metadata.
See [Harness training: preprocessing & bake](#harness-training-preprocessing--bake).

---

## Training

### Sandbox regression acceptance

Before a long run, validate the exact node image and persistent Engine path:

```bash
ruff check .
pytest -q tests/sandbox
PSRL_RUN_DOCKER_INTEGRATION=1 pytest -q -s tests/sandbox/test_docker_live.py
python tests/sandbox/benchmark_docker_backend.py \
  --image python:3.11-slim --iterations 100 --concurrency 16 \
  | tee docker-sandbox-benchmark.json
```

Then enable one resource sample per episode and run the existing five-step
cluster smoke job (or the normal launch script for the target dataset):

```bash
export PSRL_SANDBOX_COLLECT_RESOURCE_METRICS=true
bash examples/mini_swe/test_perf.sh 1
```

Compare the same dataset/model/seed and concurrency on the baseline and this
branch. Acceptance requires equal task/patch/grader semantics, no remaining
`psrl.sandbox=true` containers after shutdown, no monotonically growing dockerd
RSS, and non-regressed p50/p95 create/exec/episode latency. Trajectory timing
contains `sandbox_create_s`, `sandbox_peak_memory_mib` and
`sandbox_cpu_total_s`; final worker logs contain aggregate operation counts,
failures and mean/max latency.

### Toy dataset

```bash
bash examples/mini_swe/fsdp_qwen_7b_dapo.sh
```

Requires `data/mini_swe_agent/train.parquet` to exist (see `prepare/README.md`
Path A).

### SWE-smith-py (real RL)

#### Step 1: Install preparation dependencies

`swebench`/`swesmith` are used only while *preparing* data (freezing the SWE-smith
eval script + parser name); the training host imports neither:

```bash
python -m pip install swebench==4.1.0 swesmith
python -m examples.mini_swe.grading.vendor_parsers
```

#### Step 2: Prepare data and pre-fetch images

Follow `prepare/README.md` Path B. The expected layout when done:

```
examples/mini_swe/data/
  swe_smith_py_1k/train.parquet          # 1 000 training SWE problems
  verified_subset_80/train.parquet       # 80 validation SWE problems (test_freq)
```

#### Step 3: Pre-flight check

```bash
python -c "from minisweagent.agents.default import DefaultAgent; print('mini-swe-agent OK')"
python -c "from examples.mini_swe.swebench_grader import grade_fresh_container; print('grader OK')"
python -c "from examples.mini_swe.grading.payload import grader_zip_bytes; print('grading payload', len(grader_zip_bytes()), 'bytes: OK')"
ray status | head -5
```

#### Step 4: Launch training

```bash
bash examples/mini_swe/fsdp_qwen_7b_swe_smith.sh
```

The script takes an optional positional argument for the PSRL staleness value
(default: `2`):

```bash
bash examples/mini_swe/fsdp_qwen_7b_swe_smith.sh 3
```

#### Step 5: Monitor in wandb

| Metric | Meaning |
|--------|---------|
| `train/score` | Shaped outcome reward: `+1.0` resolved, `-1.0` failed, `0.0` policy violation |
| `train/acc` | Binary resolve rate (0 or 1 per sample) — the primary progress indicator |

---

### SWE-Gym (real RL)

SWE-Gym provides 2438 real-world bugs with pre-computed eval scripts and Docker
images from the `xingyaoww/sweb.eval.x86_64.*` registry. It shares the same
agent config (`swebench_agent_config.yaml`) and grading infrastructure as
SWE-smith, with two key differences: no `git checkout HEAD~1` is needed (standard
repo layout), and eval scripts are already frozen in the parquet (`swe_problem.eval_script`),
so grading is host-independent without any extra preparation step.

#### Step 1: Install preparation dependencies

The full SWE-Gym dataset ships without `eval_script`, so `prepare_swe_gym.py`
needs the SWE-Bench-Fork to generate it. This is preparation-only — grading
itself imports nothing upstream:

```bash
python -m pip install swebench==4.1.0
```

(The SWE-Bench-Fork is only needed for _preparing_ the full dataset, not for
training/grading. See `prepare/README.md` Path C.)

#### Step 2: Prepare data and pre-fetch images

Follow `prepare/README.md` Path C. The expected layout when done:

```
examples/mini_swe/data/
  swe_gym_2438/train.parquet              # 2438 training SWE problems (full)
  swe_gym_subset_100/train.parquet        # 100 training SWE problems (quick start)
```

#### Step 3: Pre-flight check

```bash
python -c "from minisweagent.agents.default import DefaultAgent; print('mini-swe-agent OK')"
python -c "from examples.mini_swe.swebench_grader import grade_fresh_container; print('grader OK')"
python -c "from examples.mini_swe.grading.payload import grader_zip_bytes; print('grading payload', len(grader_zip_bytes()), 'bytes: OK')"
docker run --rm xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest true && echo "SWE-Gym image OK"
ray status | head -5
```

#### Step 4: Launch training

```bash
bash examples/mini_swe/fsdp_qwen_7b_swe_gym.sh
```

To replace mini-SWE-agent's native loop with Claude Code or Codex, prepare rows
whose `agent_name` selects the corresponding entry in
`config/swebench_harness_config.yaml`:

```bash
python -m examples.mini_swe.prepare.prepare_swe_gym \
  --dataset gym-subset \
  --agent-name mini_swe_claude_code \
  --output-dir examples/mini_swe/data/swe_gym_claude

python -m examples.mini_swe.prepare.prepare_swe_gym \
  --dataset gym-subset \
  --agent-name mini_swe_codex \
  --output-dir examples/mini_swe/data/swe_gym_codex
```

For a ready-to-run Megatron recipe, `examples/mini_swe/megatron_qwen_4b_swe_cc.sh`
already pins the `mini_swe_claude_code` loop and TITO auto IDs on the SWE-Gym-293
parquets — see [§4 Launching harness training](#4-launching-harness-training).

Launch either dataset through the same trainer. Harness mode must use TITO's
automatic prefix-tree trajectory IDs so sub-agents and context-compression
branches remain in one session without sharing a trajectory ID:

```bash
AGENT_LOOP_CONFIG_PATH="$(pwd)/examples/mini_swe/config/swebench_harness_config.yaml" \
TRAJECTORY_ID_STRATEGY=auto \
TRAIN_FILE="$(pwd)/examples/mini_swe/data/swe_gym_claude/train.parquet" \
TEST_FILE="$(pwd)/examples/mini_swe/data/swe_gym_claude/train.parquet" \
bash examples/mini_swe/fsdp_qwen_7b_swe_gym.sh
```

Before launching, complete the harness preprocessing in
[Harness training: preprocessing & bake](#harness-training-preprocessing--bake):
build the native runtime trees and export `PSRL_HARNESS_RUNTIME_ROOT` (required),
and optionally bake the per-image derivative **on every worker host that creates
sandboxes**. Without the bake the runtime git probe has to purge any leaked image
metadata — a missing bake never blocks training, it only costs throughput.
`callback_base_url` is only needed when a remote backend cannot reach the
worker's configured SessionRouter origin; local Docker rewrites loopback
through `host.docker.internal` automatically.

Lifecycle ordering is: create `session_id`, acquire the task sandbox, optionally
snapshot the clean filesystem, run the sandbox-init hook (git sanitization),
prepare and run the harness, fetch all TITO trajectories, destroy the agent
sandbox and session, then grade the captured patch in a separate clean sandbox.
Cancelling the loop cancels the active CLI exec and releases the sandbox lease;
SessionRouter deletion drains any in-flight inference request before removing
session state.

#### Extending harness training to another task

`HarnessAgentLoop` owns the protocol and resource lifecycle. A task integration
subclasses it and supplies only the following hooks:

| Hook | Task responsibility |
|------|---------------------|
| `prepare_harness_task` | Return prompt, rollout `SandboxSpec`, backend, and opaque task state |
| `prepare_harness_sandbox` | Optionally initialize the acquired sandbox (e.g. git sanitization) and add timing entries |
| `collect_harness_artifact` | Optionally collect a patch, answer file, or other result before sandbox deletion |
| `finalize_harness_task` | Optionally grade the artifact in a clean environment and return reward fields |
| `close_harness_task` | Release task-only resources such as environments or concurrency slots |

Only `prepare_harness_task` is abstract. Tasks without artifacts or an external
grader can use the other default implementations. The generic layer owns session
creation/deletion, sandbox leases, sandbox init timing, harness abort, TITO
collection, trajectory validation, response capping, reward dispatch, and
snapshot cleanup.

```python
class MyHarnessAgentLoop(HarnessAgentLoop):
    async def prepare_harness_task(self, request):
        state = await prepare_my_task(request)
        return HarnessTaskContext(
            state=state,
            prompt=state.prompt,
            sandbox_spec=state.sandbox_spec,
            backend=state.backend,
        )

    async def close_harness_task(self, task):
        await task.state.close()
```

Or with Megatron parallelism:

```bash
bash examples/mini_swe/megatron_qwen_7b_swe_gym.sh
```

---

### Script parameter reference (`fsdp_qwen_7b_swe_smith.sh`)

#### Cluster layout

| Variable | Default | Description |
|----------|---------|-------------|
| `NNODES` | `4` | Total nodes in the job |
| `NGPUS_PER_NODE` | `8` | GPUs per node |
| `GEN_NNODES` | `2` | Nodes dedicated to vLLM rollout |
| `TRAIN_NNODES` | `2` | Nodes dedicated to FSDP training |
| `GEN_TP` | `1` | Tensor-model parallelism for rollout vLLM |
| `GEN_PP` | `1` | Pipeline parallelism for rollout vLLM |
| `TRAIN_SP` | `2` | Ulysses sequence parallelism during training |
| `TRAIN_FSDP` | `8` | FSDP shard group size (number of GPUs per FSDP shard) |

`GEN_INSTANCES` and `VAL_INSTANCES` are derived automatically from the above.

#### Algorithm

| Variable | Default | Description |
|----------|---------|-------------|
| `adv_estimator` | `grpo` | Advantage estimator. `grpo` groups rollouts by prompt and normalises within the group. |
| `clip_ratio_low` | `0.2` | Lower clip bound for the PPO probability ratio (DAPO asymmetric clipping) |
| `clip_ratio_high` | `0.28` | Upper clip bound |
| `use_kl_in_reward` | `False` | Whether to add a KL-penalty term to the reward |
| `kl_coef` | `0.0` | KL coefficient when `use_kl_in_reward=True` |
| `use_kl_loss` | `False` | Whether to add a KL-divergence term to the policy loss |
| `kl_loss_coef` | `0.0` | Weight of the KL loss term |

#### Sequence lengths

| Variable | Default | Description |
|----------|---------|-------------|
| `max_turns` | `30` | Maximum agent turns (bash commands) per episode. Also passed as `step_limit` to `DefaultAgent`. |
| `max_prompt_length` | `2048` | Maximum number of tokens in the initial prompt fed to vLLM |
| `max_response_length` | `16384` | Maximum tokens generated per turn (applies to each individual assistant response) |

#### Training hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `actor_lr` | `1e-6` | Actor learning rate |
| `train_prompt_bsz` | `16` | Number of unique prompts per training step (each produces `n_resp_per_prompt` rollouts) |
| `n_resp_per_prompt` | `4` | Rollouts per prompt during generation. GRPO normalises reward within this group. |
| `n_resp_per_prompt_val` | `4` | Rollouts per prompt during validation |
| `train_prompt_mini_bsz` | `16` | Mini-batch size for PPO gradient steps |
| `loss_agg_mode` | `token-mean` | Loss aggregation: `token-mean` divides by total tokens, `seq-mean` divides by number of sequences |
| `overlong_buffer_len` | `10240` | Buffer of tokens beyond `max_response_length` before penalty kicks in |
| `overlong_penalty_factor` | `1.0` | Penalty per token in the overlong buffer (linear ramp from 0 to this value) |

#### Sampling

| Variable | Default | Description |
|----------|---------|-------------|
| `temperature` | `1.0` | Sampling temperature for rollout generation |
| `top_p` | `1.0` | Nucleus sampling threshold |
| `top_k` | `-1` | Top-k sampling (`-1` = disabled) |
| `val_top_p` | `1.0` | `top_p` used during validation rollouts |

#### Token Importance Sampling (TIS)

| Variable | Default | Description |
|----------|---------|-------------|
| `rollout_is` | `token` | Importance-sampling granularity: `token` weights by per-token probability ratio; `sequence` weights by sequence-level ratio |
| `rollout_is_threshold` | `2.0` | Rollouts whose IS weight exceeds this threshold are clipped / dropped. Prevents stale rollouts from dominating updates. |

The PSRL `staleness` parameter (first positional arg to the script, default `2`)
controls how many training steps a rollout may remain in the replay buffer before
being discarded. Lower values are more on-policy; higher values increase throughput.

#### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `use_dynamic_bsz` | `True` | Pack sequences into fixed-length chunks to avoid wasted padding |
| `packing_length` | `max_prompt_length + max_response_length` | Target chunk size for dynamic batching |
| `offload` | `False` | CPU offload for optimizer state. Saves GPU memory at the cost of throughput. |

---

## Harness training: preprocessing & bake

Canonical reference for running a **harness** (Claude Code or Codex) instead of
mini-SWE-agent's native loop. The `claude_code` harness is `mini_swe_claude_code`
and the Codex harness is `mini_swe_codex`, both defined in
`config/swebench_harness_config.yaml` and selected by the data's `agent_name`.

Harness mode needs everything the native path needs (task parquet + prefetched
per-problem images) plus two host-side preparations: the read-only harness
runtime trees (§1) and, optionally, the git-purged per-image derivative (§2).
Do them **once per host that creates sandboxes**; the runtime trees live on the
shared filesystem, the baked images are node-local.

| # | Step | Reference |
|---|------|-----------|
| 1 | Build the task parquet; set `--agent-name mini_swe_claude_code` / `mini_swe_codex` (`megatron_qwen_4b_swe_cc.sh` instead pins `default_agent_loop` at launch) | [`prepare/README.md`](prepare/README.md) Path C |
| 2 | Prefetch and fan out the per-problem base images to every worker node | [`prepare/README.md`](prepare/README.md) Path C Step 2 |
| 3 | Build the native harness runtime trees and export `PSRL_HARNESS_RUNTIME_ROOT` | §1 |
| 4 | Bake the per-image derivative (**git-leak purge** only; optional) | §2 |
| 5 | Launch with the harness agent-loop config and TITO auto trajectory IDs (`megatron_qwen_4b_swe_cc.sh` sets both) | §4 |

Step 4 is optional but recommended. A missing bake never blocks training: the
sandbox falls back to the runtime git probe, which purges only images that
actually leak (see §2).

### 1. Harness runtime trees (host-side, read-only mount)

The harness executables are **self-contained native binaries** — Claude Code
2.1.233 (linux-x64) and Codex 0.154.0 (linux-x64). There is no Node, no npm and
no in-sandbox installation. Each sandbox binds one runtime tree read-only; the
mount path is the harness config's `runtime_mount` (default `/opt/harness`) and
the executable resolves to `<runtime_mount>/bin/<executable>`.

| Host env var | Layout | In-sandbox mount |
|--------------|--------|------------------|
| `PSRL_HARNESS_RUNTIME_ROOT` | `<root>/<kind>/bin/<executable>` | `<harness.runtime_mount>` (default `/opt/harness`) |

Build it once on the shared filesystem (idempotent; verifies the pinned Claude
Code sha256 and runs `<bin> --version`):

```bash
export PSRL_HARNESS_RUNTIME_ROOT=/shared/psrl/harness-runtimes
bash examples/mini_swe/prepare/docker_scripts/build_harness_runtimes.sh
# -> $PSRL_HARNESS_RUNTIME_ROOT/claude_code/bin/claude
#    $PSRL_HARNESS_RUNTIME_ROOT/codex/bin/codex
```

`NPM_REGISTRY` defaults to `registry.npmjs.org`; point it at a mirror (e.g.
`https://mirrors.tencent.com/npm`) when the host cannot reach npmjs.org
directly. The root is propagated to Ray workers via `_HOST_RUNTIME_ENV_KEYS`
(`psrl/trainer/constants_ppo.py`).

Per trajectory the harness setup is a **single round trip** — create the state
directory and run `<runtime_mount>/bin/<executable> --version`. No install, no
network fetch, and no write to `/usr/local`: the task image's Node (if any) is
never touched.

Backend note: bind mounts require host-mount support. Docker has it; a microVM
backend (E2B / Cube / AgentEnv) does not, and the sandbox manager rejects the
spec — those backends must provision the runtime into their template instead.

### 2. Per-image bake (git purge only, optional)

Each SWE task uses its **own** per-problem base image (the parquet's
`sandbox_overrides.environment.image` — e.g. `swebench/swesmith.x86_64.*`), so
there is no single "harness image". `bake_harness_image.sh` derives one image
per base — `psrl/swebench-harness:<sha12(base)>` — that **purges leaked
git metadata** (remotes, refs, reflog, unreachable objects) so an agent can
never read a future fix commit from the image. The harness executable is **not**
baked: it is mounted read-only (§1).

Bake every unique image referenced by the train/validation parquets, **on each
worker host that creates sandboxes** (Docker images are node-local):

```bash
# every unique image in a parquet:
bash examples/mini_swe/prepare/docker_scripts/bake_harness_image.sh \
  --parquet examples/mini_swe/data/swe_gym_293/train.parquet
# a single image:
bash examples/mini_swe/prepare/docker_scripts/bake_harness_image.sh \
  swebench/swesmith.x86_64.foo:latest
```

What it does per image: start a container from the base image, run the git
purge, then `docker commit`. A per-image host lock serializes concurrent bakes
of the same image.

Verify a baked image is git-clean:

```bash
tag=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep '^psrl/swebench-harness:' | head -1)
docker run --rm "$tag" bash -lc \
  'cd /testbed && echo "remotes=[$(git remote)]" && echo "leaked=$(git rev-list --count --all --reflog --not HEAD)"'
# expected: remotes=[] and leaked=0
```

* `runner.py` selects the derivative automatically: for each task it computes
  `psrl/swebench-harness:<sha12(task-image)>` and uses it when present, else
  falls back to the task image + the runtime git probe — **a missing bake never
  blocks training**.
* **Re-baking after a bake change**: the tag keys on the base image alone, so a
  derivative that already exists is skipped. After editing the bake steps, run
  `rebake_harness_image.sh` (same arguments) to delete the existing derivative(s)
  and re-bake — the supported replacement for the old `BAKE_REVISION` bump:
  ```bash
  bash examples/mini_swe/prepare/docker_scripts/rebake_harness_image.sh \
    --parquet examples/mini_swe/data/swe_gym_293/train.parquet
  ```
  It also removes tags left by an older digest formula. `PSRL_REBAKE_VERIFY=1`
  probes each new image for git leaks, `PSRL_REBAKE_DRY_RUN=1` only prints the
  plan, and `PSRL_BAKE_IMAGE_GLOB` overrides which tags are deleted.
* **Runtime fallback**: on a host without the derivative, the harness loop runs
  a cheap git probe (`examples/mini_swe/utils/git_sanitize.py`) before the agent
  starts and purges only when it detects a leak. Probe/purge timings show up in
  the trajectory `[Time Breakdown]` as `sandbox_init` / `git_probe` / `git_purge`
  (raw metrics `git_leak_detected` / `git_sanitize_error` are on the reward
  info). A purge failure is logged and degraded, never fatal.
* **Knobs**: `PSRL_BAKE_WORKDIR` (default `/testbed`),
  `PSRL_BAKE_SKIP_GIT_CLEAN=1`, `PSRL_HARNESS_IMAGE_TAG` (single-image mode);
  `rebake_harness_image.sh` adds `PSRL_BAKE_IMAGE_GLOB`, `PSRL_REBAKE_VERIFY=1`
  and `PSRL_REBAKE_DRY_RUN=1`.
* Local-only images: on multi-host clusters run the bake on every host, or
  distribute with `docker save`/`docker load`, or `docker push` to a registry.

### 3. Differences vs. the mini-SWE-agent training prepare flow

| Aspect | mini-SWE-agent | `claude_code` harness |
|--------|----------------|------------------------|
| CLI / agent location | Host-side Python library (`pip install mini-swe-agent`), imported by the runner | **Sandbox-resident CLI**: a self-contained native binary (Claude Code 2.1.233 / Codex 0.154.0), bind-mounted read-only from the host runtime tree — no Node, no npm |
| Per-sandbox install | None (pure Python on the host) | **None** — a single `<runtime_mount>/bin/<executable> --version` probe; the executable is mounted read-only |
| Network dependency | None at rollout time | None at rollout time: the runtime tree is built ahead of time (`build_harness_runtimes.sh`) |
| System prompt / tools | Native mini-SWE-agent template | Keeps Claude Code's stock system prompt (`system_prompt_mode: none`) and full default tool catalog (`tools: null`); task + integrity rules ride in the user message, and safety is enforced post-hoc by the integrity scan |
| Permission mode | n/a | `permission_mode: acceptEdits` (Claude rejects `bypassPermissions` as root); headless-required tools are pre-approved via `permissions.allow` in `settings.json`, and `subagents_enabled: false` adds an `Agent` deny rule |
| Compaction | n/a | `compaction.compact_percent` (Claude `87.5`, Codex `75`) triggers auto-compact at that share of the rollout `max_model_len`; the same threshold drives the `x-smg-prompt-too-long-limit` header |
| Sandbox image | Per-task image as-is | Per-image baked derivative `psrl/swebench-harness:<sha12>` (git purge only) when present, else the per-task image + runtime git probe |
| Git hygiene | Image git state left as-is | The baked derivative purges remotes/refs/reflog/unreachable objects; an unbaked image gets a runtime probe + conditional purge before the harness starts |
| Integrity / anti-cheat | n/a | Trajectory scan dispatched on `trajectory_format` (Claude stream-json / Codex JSONL) **plus** an independent final-patch re-check, so a trajectory violation cannot hide a protected-path edit |
| Grading | Fresh container from the SWE problem image | Same fresh-container grader; additionally supports **clean-snapshot reuse** — the clean rollout sandbox is `docker commit`ted (`FILESYSTEM_SNAPSHOT` + `RESTORE` on the docker backend) and the grader restores from it, avoiding a fresh cold start |
| Prep instrumentation | n/a | Trajectory dump emits a fine-grained `prep` breakdown: `task` / `sandbox` / `snapshot` / `sandbox_init` / `git_probe` / `git_purge` / `harness_prepare` |

### 4. Launching harness training

**Ready-to-run (Megatron, Claude Code).** `megatron_qwen_4b_swe_cc.sh` is the
canonical harness recipe: Qwen3.5-4B on the SWE-Gym-293 parquets, trained with
Megatron (TP=4 / CP=2, 2 nodes × 8 GPUs). It pins everything harness mode needs:

- `agent_loop_config_path=config/swebench_harness_config.yaml`
- `default_agent_loop=mini_swe_claude_code` (so the parquet's `agent_name` is not required to select the harness)
- `psrl.rollout_gateway.trajectory_id_strategy=auto` (TITO prefix-tree IDs)
- `PSRL_HARNESS_RUNTIME_ROOT` (the read-only runtime trees from §1)

```bash
bash examples/mini_swe/megatron_qwen_4b_swe_cc.sh        # staleness defaults to 1
bash examples/mini_swe/megatron_qwen_4b_swe_cc.sh 2      # optional staleness arg
```

It expects `data/swe_gym_293/{train,val}.parquet` and their prefetched images
from [`prepare/README.md`](prepare/README.md) Path C, and it exports
`SWE_STRICT_NO_TEST_PATCH=1` / `SWE_TEST_PATCH_POLICY_SCOPE=all_tests`.

**Generic (any model/dataset, FSDP).** Point the trainer at the harness config
and a parquet whose `agent_name` selects the harness. Harness mode requires TITO
auto trajectory IDs so sub-agent / compaction branches stay in one session:

```bash
AGENT_LOOP_CONFIG_PATH="$(pwd)/examples/mini_swe/config/swebench_harness_config.yaml" \
TRAJECTORY_ID_STRATEGY=auto \
TRAIN_FILE="$(pwd)/examples/mini_swe/data/swe_gym_claude/train.parquet" \
TEST_FILE="$(pwd)/examples/mini_swe/data/swe_gym_claude/train.parquet" \
bash examples/mini_swe/fsdp_qwen_7b_swe_gym.sh
```

`TRAJECTORY_ID_STRATEGY` is read by `fsdp_qwen_7b_swe_gym.sh`; the Megatron
scripts pass the equivalent `psrl.rollout_gateway.trajectory_id_strategy`
config key directly. Both resolve to the same TITO strategy.

### 5. Gotchas

* **Native binary, not npm**: the `@anthropic-ai/claude-code` npm package is only
  a wrapper that downloads a platform binary as an `optionalDependencies` entry;
  PSRL fetches that platform binary directly (sha256-pinned) and mounts it, so no
  Node/npm ever runs in a sandbox.
* **Runtime tree must exist on every worker**: the mount source is resolved
  node-locally, so `PSRL_HARNESS_RUNTIME_ROOT` must point at a shared path every
  worker can read (same as the task images).
* **Snapshot reuse safety**: `clean_snapshot_compatible` only allows reuse when
  the rollout sandbox has no content-bearing bind mounts the grader relies on
  (the read-only harness runtime mount is excluded), so repo-bind-mount
  configurations fall back to a fresh grader.
* **Bake is per host**: derivatives live in the node-local Docker daemon. A node
  that was not baked still trains, but pays the runtime git probe every rollout.
* **Host-mounted repos are skipped by the runtime purge**: if a task bind-mounts
  a host checkout into the workdir, git sanitization is skipped to avoid mutating
  the host; rely on the baked image or the image contract instead.

---

## Evaluation and Serving

Standalone SWE-bench / SWE-smith evaluation and vLLM-based model serving live
under [`eval/`](eval/README.md). That guide covers gold-patch sanity checks,
single-node and multi-node eval, how to serve your own checkpoint (single
host with TP / PP / DP, or cross-host via a litellm proxy), the output-artefact
layout, and how PSRL's in-training validation differs from the standalone tool.

---

## Configuration

### Config hierarchy

```
MiniSWEAgentRuntimeConfig   (dataclass defaults in config.py)
  └── swebench_agent_config.yaml  or  simple_agent_config.yaml
       └── extra_info per SWE problem  (sandbox_overrides / agent_overrides)
```

`sandbox_config.environment` is the shared container base. The optional
`rollout_environment` and `grader_environment` mappings override that base for
their respective lifecycle; `env` entries are merged while scalar and list
fields replace the base value.

MiniSWE model traffic stays in the AgentLoopWorker process, so Docker bridge
networking does not sit between the model client and SessionRouter, SMG, vLLM,
or ModelProxy. Commands inside the container can reach node-local services at
`host.docker.internal`. The `mini_swe` Docker policy in
`psrl/trainer/config/rollout/psrl_rollout.yaml` creates that host-gateway mapping
and rewrites forwarded proxy URLs whose host is `localhost`, `127.0.0.1`, or
`::1`; remote proxy URLs pass through unchanged.

### Choosing the right config YAML

| YAML | Use with |
|------|---------|
| `config/simple_agent_config.yaml` | Toy path — single `python:3.11-slim` image, preexisting repos |
| `config/swebench_agent_config.yaml` | SWE-smith-py / SWE-Gym / Verified — per-SWE-problem images, `cwd=/testbed` |
| `config/swebench_agent_config_xml_fc.yaml` | Same as above but with XML function-calling format (for newer models) |
| `config/swebench_agent_config_full_sweagent.yaml` | Full SWE-agent prompt format (SWE-agent-LM-7B) |
| `config/swebench_harness_config.yaml` | Claude Code or Codex harness over the same SWE task and grader contracts |

The `environment.image` field in `swebench_agent_config.yaml` is intentionally set
to a sentinel value (`swebench-sentinel-override-per-instance`). The real image is
injected at rollout time from `extra_info.sandbox_overrides.environment.image`,
which is written by `prepare_swebench.py`.

### Key config fields

| Field | Category | Description |
|-------|----------|-------------|
| `rollout.multi_turn.enable` | Required | Must be `True` for mini-SWE-agent |
| `rollout.multi_turn.max_turns` | Required | Max LLM generation turns per episode; training reads `gen_actor_rollout_ref`, validation reads `train_actor_rollout_ref` |
| `sandbox_config.environment.image` | Data-affine | Docker image (overridden per-SWE-problem for SWE-smith path) |
| `sandbox_config.environment.template` | Data-affine | Provider template/snapshot ID for AgentEnv or CubeSandbox |
| `sandbox_config.environment.cwd` | Data-affine | Working directory inside container (`/testbed` for SWE-bench images) |
| `sandbox_config.environment.forward_env` | Infrastructure | Additional host environment names to forward; proxy variables are always included once |
| `sandbox_config.environment` | Infrastructure | Shared image/template, cwd, env, forwarded env and timeout base for rollout and grader |
| `sandbox_config.rollout_environment` | Infrastructure | Rollout-only overrides, including `memory` and command `timeout` |
| `sandbox_config.grader_environment` | Infrastructure | Fresh-verifier overrides; resources must match rollout to reuse a microVM snapshot |
| `sandbox_config.environment.container_timeout` | Infrastructure | Max container lifetime (default: `2h`) |
| `sandbox_config.backend` | Infrastructure | Optional backend name from PSRL's worker-local sandbox registry |
| `sandbox_config.policy_profile` | Infrastructure | Docker policy profile; set `null` for microVM providers |
| `sandbox_config.snapshot_verifier` | Infrastructure | Use a capability-gated clean verifier snapshot when its spec matches exactly |
| `sandbox_config.collect_resource_metrics` | Infrastructure | Sample per-trajectory memory/CPU once; disabled by default |
| `sandbox_config.sandbox_cpu_count` | Infrastructure | Per-container CPU request charged with the effective memory request against the node envelope |
| `agent.system_template` | Native required | Native mini-SWE-agent system prompt; harnesses use their adapter-specific system prompt |
| `agent.problem_template` | Native required | Native `instance_template`; harnesses preserve the same `<pr_description>` task boundary |
| `agent.cost_limit` | Optional | LiteLLM cost limit per episode (`0.0` = unlimited) |

Harness settings live under `harness` in `swebench_harness_config.yaml`:

| Field | Effect |
|-------|--------|
| `kind` | Harness adapter to run (`claude_code`, `codex`). |
| `tools` | `null` keeps the CLI's full default tool catalog; a string pins `--tools` and restricts it. |
| `allowed_permissions` | Headless pre-approval rules written to `settings.json` `permissions.allow`; does not restrict the catalog. |
| `permission_mode` | CLI permission mode (`acceptEdits`; Claude rejects `bypassPermissions` as root). |
| `system_prompt` / `system_prompt_mode` | `none` keeps the CLI's stock system prompt; `append` / `replace` inject or replace it. |
| `subagents_enabled` | `false` denies the Claude `Agent` tool and background tasks. |
| `compaction.compact_percent` | Share of the rollout `max_model_len` at which the CLI auto-compacts (same threshold drives the `x-smg-prompt-too-long-limit` header). |
| `trajectory_format` | Integrity-scan dispatch key: `claude_code_stream_json`, `codex_jsonl`, `auto`, or `plain_text`. |
| `runtime_mount` | In-sandbox path of the read-only runtime tree; executable resolves to `<runtime_mount>/bin/<executable>` (default `/opt/harness`). |

The context window is the effective rollout `max_model_len` — there is no
separate harness window knob. Training episodes read
`gen_actor_rollout_ref.rollout`, validation episodes read
`train_actor_rollout_ref.rollout`, so `max_model_len` and `multi_turn.max_turns`
can differ between the two.

---

## Architecture: How It Works

### Session-routed black-box execution

1. PSRL creates one TITO session and builds
   `/sessions/{session_id}/v1` as mini-swe-agent's API base.
2. The loop sends the session URL, task, sampling parameters, and runtime config
   to the local black-box `runner.py` function.
3. The runner uses mini-swe-agent's normal Python bindings. Session router
   injects immutable TITO/PSRL routing headers and forwards each request body
   unchanged to SMG.
4. After mini-swe-agent exits, PSRL fetches the session once and converts the
   captured trajectory into canonical training data.
5. PSRL releases generic sandbox leases and deletes the TITO session in
   `finally` cleanup. Docker containers are removed; capable microVM providers
   also clean up temporary verifier snapshots.

### Runner execution

The runner uses mini-swe-agent's official Python bindings and runs in a bounded,
dedicated worker-thread pool. This keeps Docker rollout distributed with PSRL's
AgentLoopWorkers, avoids a task-level HTTP hop or centralized agent-server
bottleneck, and leaves the default executor available for timeout cleanup.
The same path works with AgentEnv and CubeSandbox because the runner only sees
`SyncSandboxManager` and capability flags.

The Docker backend uses a worker-persistent Engine API connection pool; it does
not spawn a Docker CLI process for create/exec/file operations. See
[`psrl/sandbox/README.md`](../../psrl/sandbox/README.md) for live conformance,
benchmark and microVM snapshot/restore commands.

---

## Reward Function

### Toy path (`data_source = mini_swe_agent_simple` or `mini_swe_agent`)

| Condition | Score |
|-----------|-------|
| Exact patch match | `1.0` |
| Partial patch match (file + line overlap) | `0.10 – 0.85` |
| Patch on wrong files | `0.05` |
| No patch, edited correct file | `0.05` |
| No patch, ran tests or Python | `0.03` |
| No patch, made edits (wrong file) | `0.02` |
| No patch, explored correct file | `0.02` |
| No patch, explored code | `0.01` |
| Alignment failed / 0 turns / timeout | `0.0` |
| Long and fruitless (≥10 turns, no edits) | `-0.05` |
| Premature exit (≤2 turns, no tools) | `-0.1` |

### SWE-smith / SWE-Gym / Verified path (`data_source = swe_smith_py`, `swe_gym`, or `swebench_verified`)

Reward is based on whether the submitted patch resolves the SWE problem (all
FAIL_TO_PASS tests pass and all PASS_TO_PASS tests still pass):

| Condition | `score` (→ loss) | `acc` (→ wandb) |
|-----------|-----------------|-----------------|
| All F2P pass, no P2P regressions | `+1.0` | `1.0` |
| Patch modified test or config files | `0.0` (policy violation — not penalised) | `0.0` |
| Not resolved (patch failed, tests failed) | `-1.0` | `0.0` |
| No patch submitted / 0 turns (aborted) | `0.0` | `0.0` |

The `{-1, 0, +1}` convention. The `score` field drives the
policy gradient loss; the `acc` field is a separate metric for tracking resolve
rate. Both are visible in wandb as `train/score` and `train/acc`.

**Patch policy rules** (configurable via environment variables):

| Env var | Default | Effect |
|---------|---------|--------|
| `SWE_STRICT_NO_TEST_PATCH` | `1` | Reject patches that modify FAIL_TO_PASS / PASS_TO_PASS test files |
| `SWE_STRICT_NO_CONFIG_PATCH` | `1` | Reject patches that modify `pyproject.toml`, `setup.py`, etc. |
| `SWE_TEST_PATCH_POLICY_SCOPE` | `eval_tests_only` | `all_tests` to also reject changes to non-eval test files |

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `docker: Cannot connect` | Docker not running | `sudo systemctl start docker` |
| Rollout container fails immediately | Image not pulled | Run `prepare/docker_scripts/prefetch_images.sh` on this node first (or `load_all_nodes.sh` across the cluster) |
| `swebench-sentinel-override-per-instance` in error (sentinel was not replaced per SWE problem) | Wrong config YAML for toy path | Use `simple_agent_config.yaml` for toy, `swebench_agent_config.yaml` for SWE-smith/SWE-Gym |
| `xingyaoww/sweb.eval.x86_64.*` image not found | SWE-Gym images not pre-fetched | Run `prepare/docker_scripts/swe_gym.sh` (or `swe_gym_subset.sh`) |
| `eval_script missing` error in grader | SWE-Gym parquet missing eval_script | Re-run `prepare_swe_gym.py` (full dataset requires SWE-Bench-Fork 2.0.13) |
| Grader always returns `resolved=False` | Image pull failing silently | Check `grading.json` in the eval output dir for error messages |
| No patch found | Agent hit turn limit without submitting | Increase `max_turns` |
| OOM during rollout | The sandbox node envelope overlaps too much with co-located services | Set `gen_actor_rollout_ref.rollout.agent.sandbox.capacity.memory_mb` explicitly or lower its single `utilization` value; verify each rollout/grader memory request |
| `alignment_failed` every episode | Context truncation | Reduce `max_prompt_length` or increase `max_model_len` |

### Emergency cleanup

Stop all rollout and grader containers left behind by an aborted run:

```bash
# All PSRL-owned Docker sandboxes
docker ps -aq --filter "label=psrl.sandbox=true" | xargs -r docker rm -f

# Ray cluster
ray stop --force

# Training process
pkill -9 -f main_ppo
```

---

## Extending

### Custom Docker Image (toy path)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*
RUN python -m pip install pytest numpy
```

Set `sandbox_config.environment.image` in the agent config YAML.

### Custom Reward Functions

```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict | None = None,
    **kwargs,
) -> float | dict:
    """Return a float or {"score": float, "acc": float}."""
```

Pass `custom_reward_function.path` and `custom_reward_function.name` to `main_ppo`.

### Custom Prompt Templates

Override `agent.system_template` and `agent.problem_template` in the config YAML.
Templates use Jinja2; available variables: `{{ task }}`, `{{ cwd }}`,
`{{ system }}`, `{{ release }}`, `{{ version }}`, `{{ machine }}`.
