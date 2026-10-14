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
├── swebench_grader.py                    # Fresh-container grader for SWE-smith / SWE-Gym / Verified (shared by training + eval)
├── fsdp_qwen_7b_dapo.sh                  # Launch script — toy dataset (FSDP, 7B)
├── fsdp_qwen_14b_dapo.sh                 # Launch script — toy / DAPO path (FSDP, 14B)
├── fsdp_qwen_7b_swe_smith.sh             # Launch script — SWE-smith-py (FSDP, 7B)
├── fsdp_qwen_7b_swe_gym.sh               # Launch script — SWE-Gym (FSDP, 7B)
├── megatron_qwen_4b_swe_smith.sh         # Launch script — SWE-smith (Megatron, 4B)
├── megatron_qwen_7b_swe_gym.sh           # Launch script — SWE-Gym (Megatron, 7B)
├── megatron_qwen_8b_swe_smith.sh         # Launch script — SWE-smith (Megatron, 8B)
├── megatron_qwen_8b_swe_gym.sh           # Launch script — SWE-Gym (Megatron, 8B)
├── megatron_qwen_32b_swe_smith.sh        # Launch script — SWE-smith (Megatron, 32B)
├── test_perf.sh                          # Performance / throughput probe
├── config/
│   ├── simple_agent_config.yaml          # Agent config for toy path
│   ├── swebench_agent_config.yaml        # Agent config for SWE-smith / SWE-Gym / Verified
│   └── swebench_agent_config_xml_fc.yaml # XML function-calling variant (newer models)
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
    ├── swebench_subsets.py               # Repo-balanced sampling helpers
    └── docker_scripts/                   # Docker image pre-fetch / fan-out helpers
        ├── bake_simple_repos.sh          # Bakes toy repos into a Docker image (Path A)
        ├── prefetch_images.sh            # Pull per-SWE-problem images (skopeo + multi-mirror + tar cache)
        ├── prefetch_example.sh           # Reference invocation chaining prefetch + load_all_nodes
        ├── swe_smith.sh                  # Convenience wrapper for SWE-smith images
        ├── swe_gym.sh                    # Convenience wrapper for SWE-Gym images
        ├── swe_gym_subset.sh             # Convenience wrapper for SWE-Gym-Subset images
        ├── swe_eval_subset.sh            # Convenience wrapper for SWE-bench eval subset images
        ├── probe_mirrors.sh              # Check which public Docker Hub mirrors serve a given image
        └── load_all_nodes.sh             # pssh fan-out of `docker load` across the cluster

# Core integration modules inside psrl/
psrl/workers/agent_loop/loops/session_agent_loop.py       # Shared SessionRouter/TITO lifecycle
psrl/workers/agent_loop/loops/mini_swe_agent_loop_v1.py   # Session-router/TITO black-box loop
psrl/workers/agent_loop/agent_data/mini_swe_agent_data.py # MiniSWEAgentData
psrl/environments/mini_swe_env.py                         # MiniSWEEnvironment
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

# 3. Extra deps for SWE-smith-py / SWE-bench grading (skip for toy path)
python -m pip install swebench==4.1.0 swesmith

# 4. Verify
python -c "from minisweagent.agents.default import DefaultAgent; print('mini-swe-agent OK')"
python -c "import swebench; print('swebench', swebench.__version__)"
python -c "from swesmith.profiles import registry; print('swesmith profiles:', len(registry.data))"
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
- **Path C** — converting SWE-Gym (2438 problems) or SWE-Gym-Subset (100
  problems) from HuggingFace, with pre-computed eval scripts and `xingyaoww`
  Docker images

Read that file before running training for the first time.

---

## Training

### Toy dataset

```bash
bash examples/mini_swe/fsdp_qwen_7b_dapo.sh
```

Requires `data/mini_swe_agent/train.parquet` to exist (see `prepare/README.md`
Path A).

### SWE-smith-py (real RL)

#### Step 1: Install grading dependencies

```bash
python -m pip install swebench==4.1.0 swesmith
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
python -c "import swebench; print('swebench', swebench.__version__)"
python -c "from swesmith.profiles import registry; print('swesmith profiles:', len(registry.data))"
python -c "from examples.mini_swe.swebench_grader import grade_fresh_container; print('grader OK')"
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
repo layout), and eval scripts are pre-computed in the parquet rather than
generated at grading time.

#### Step 1: Install grading dependencies

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
python -c "import swebench; print('swebench', swebench.__version__)"
python -c "from examples.mini_swe.swebench_grader import grade_fresh_container; print('grader OK')"
docker run --rm xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7365:latest true && echo "SWE-Gym image OK"
ray status | head -5
```

#### Step 4: Launch training

```bash
bash examples/mini_swe/fsdp_qwen_7b_swe_gym.sh
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

### Choosing the right config YAML

| YAML | Use with |
|------|---------|
| `config/simple_agent_config.yaml` | Toy path — single `python:3.11-slim` image, preexisting repos |
| `config/swebench_agent_config.yaml` | SWE-smith-py / SWE-Gym / Verified — per-SWE-problem images, `cwd=/testbed` |
| `config/swebench_agent_config_xml_fc.yaml` | Same as above but with XML function-calling format (for newer models) |

The `environment.image` field in `swebench_agent_config.yaml` is intentionally set
to a sentinel value (`swebench-sentinel-override-per-instance`). The real image is
injected at rollout time from `extra_info.sandbox_overrides.environment.image`,
which is written by `prepare_swebench.py`.

### Key config fields

| Field | Category | Description |
|-------|----------|-------------|
| `rollout.multi_turn.enable` | Required | Must be `True` for mini-SWE-agent |
| `rollout.multi_turn.max_turns` | Required | Max LLM generation turns per episode |
| `sandbox_config.environment.image` | Data-affine | Docker image (overridden per-SWE-problem for SWE-smith path) |
| `sandbox_config.environment.cwd` | Data-affine | Working directory inside container (`/testbed` for SWE-bench images) |
| `sandbox_config.environment.container_timeout` | Infrastructure | Max container lifetime (default: `2h`) |
| `sandbox_config.max_parallel_tasks_per_worker` | Infrastructure | Concurrency limit per node (`0` = unlimited) |
| `agent.system_template` | **Required** | System prompt (no default) |
| `agent.problem_template` | **Required** | Per-SWE-problem prompt template; maps to `instance_template` in mini-swe-agent (no default) |
| `agent.cost_limit` | Optional | LiteLLM cost limit per episode (`0.0` = unlimited) |

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
5. PSRL deletes the TITO session and removes rollout Docker containers in
   `finally` cleanup.

### Runner execution

The runner uses mini-swe-agent's official Python bindings and runs in a bounded,
dedicated worker-thread pool. This keeps Docker rollout distributed with PSRL's
AgentLoopWorkers, avoids a task-level HTTP hop or centralized agent-server
bottleneck, and leaves the default executor available for timeout cleanup.

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
| OOM during rollout | Too many concurrent containers | Reduce `max_parallel_tasks_per_worker` or lower `train_prompt_bsz` |
| `alignment_failed` every episode | Context truncation | Reduce `max_prompt_length` or increase `max_model_len` |

### Emergency cleanup

Stop all rollout and grader containers left behind by an aborted run:

```bash
# Rollout containers
docker ps -q --filter "label=psrl.swe_task_id" | xargs -r docker stop

# Grader containers
docker ps -q --filter "label=psrl.grader_task_id" | xargs -r docker stop

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
