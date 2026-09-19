# ReTool Training Recipe

Train language models to strategically invoke a Python code interpreter while
solving hard math problems, using RL. This recipe is inspired by
[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)
(ByteDance-Seed, 2025) and uses
[SandboxFusion](https://github.com/bytedance/SandboxFusion) as the code
execution backend.

Three training paths are supported:

| Path | Backend | Model | Cluster | Use case |
|------|---------|-------|---------|----------|
| **FSDP 7B** | FSDP2 | `multiturn-sft-qwen-2.5-7b-instruct` | 4 nodes × 8 GPU | Default recipe, fast iteration |
| **Megatron 32B** | Megatron-LM | `ReTool-Qwen-32B-SFT` | 6 nodes × 8 GPU | Large-dense RL |
| **Megatron MoE 30B** | Megatron-LM | `Qwen3-30B-A3B` | 6+ nodes × 8 GPU | MoE recipe (EP / ETP) |

All three paths share the same dataset (`dapo-math-17k` train, `aime-2024` /
`aime-2025` val), the same tool (`CustomSandboxFusionTool`) and the same
reward (`compute_score` in [retool.py](retool.py)).

---

## Overview

The training loop works as follows:

1. **Data**: each row is a math problem (AIME-style) with a reference answer.
   `CustomRLHFDataset.map_fn` appends the `\boxed{}` answer-format instruction.
2. **Rollout**: PSRL's `MultiTurnAgentLoop` drives the model turn-by-turn.
   Every turn is a chat completion with tool calls in `hermes` format.
3. **Tool call**: when the model emits a `code_interpreter` tool call,
   `CustomSandboxFusionTool.async_forward` strips the enclosing
   ```` ```python ... ``` ```` block, auto-adds a trailing `print(...)`
   if the final statement isn't already a print, and POSTs the code to the
   SandboxFusion HTTP endpoint (`http://localhost:8080/run_code` — the Swarm
   service publishes port 8080 on every node in `mode=host`).
4. **Observation**: the sandbox's `stdout` / `stderr` / `return_code` comes
   back as the tool output and is appended to the conversation.
5. **Reward** (once the model emits `\boxed{...}`): `compute_score` verifies
   the boxed answer via `verl.utils.reward_score.math_dapo.compute_score`
   with `strict_box_verify=True`, and shapes the reward with a small
   tool-call bonus when the answer is wrong.
6. **GRPO update**: rewards are grouped by prompt, normalised within the
   group, and used for the PPO policy gradient step.

```
┌─────────────────────────────────────────────────────┐
│               PSRL GRPO / DAPO Trainer              │
│  (actor, vLLM rollout, reward scoring, TIS)         │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │ MultiTurnAgentLoop.run()│
          │   (tool_env + hermes)   │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────────┐
     │                 │                     │
     ▼                 ▼                     ▼
┌──────────┐    ┌────────────────┐   ┌─────────────────────────┐
│   vLLM   │    │ ToolEnvironment│   │ CustomSandboxFusionTool │
│ generate │    │  .step(action) │◄──│   .async_forward(code)  │
└──────────┘    └──────┬─────────┘   └───────────┬─────────────┘
                       │                         │ HTTP POST
                       │                         ▼
                       │              ┌──────────────────────┐
                       │              │ SandboxFusion service│
                       │              │  (Docker Swarm, 8080)│
                       │              └──────────────────────┘
                       │
               ┌───────┴──────────────┐
               │ compute_score        │
               │ (math_dapo +         │
               │  tool-call shaping)  │
               └──────────────────────┘
```

---

## Directory Structure

```
examples/retool/
├── README.md                             # This file
├── retool.py                             # CustomSandboxFusionTool + CustomRLHFDataset + compute_score
├── sandbox_fusion_tool_config.yaml       # Tool registry loaded at rollout time
├── fsdp_qwen_7b_dapo.sh                  # Launch script — FSDP2 Qwen2.5-7B (Path A)
├── megatron_qwen_32b_dapo.sh             # Launch script — Megatron ReTool-Qwen-32B (Path B)
├── megatron_qwen3_30b_dapo_small.sh      # Launch script — Megatron Qwen3-30B-A3B MoE (Path C)
├── docker_scripts/                       # Image bake + multi-node fan-out
│   ├── README.md                         # Image preparation guide (read this before deploying)
│   ├── docker_install.sh                 # Pull + save `code_sandbox:server` to a shared-FS tar
│   ├── docker_copy.sh                    # pssh fan-out: copy tar to every node and docker load
│   ├── docker_manager.sh                 # Cluster-wide dockerd start/stop/restart/status/logs
│   ├── docker_common.sh                  # Shared helpers (sourced by the above)
│   └── example.sh                        # End-to-end reference invocation
└── sandbox_fusion/                       # SandboxFusion service deployment (Docker Swarm)
    ├── README.md                         # Swarm deploy guide
    ├── launch_service.sh                 # Swarm init + `docker service create`
    └── clear_service.sh                  # Tear-down

# Core integration modules inside psrl/
psrl/workers/agent_loop/loops/multi_turn_agent_loop.py  # MultiTurnAgentLoop
psrl/workers/agent_loop/agent_data/tool_agent_data.py   # ToolAgentData
psrl/environments/tool_env.py                           # ToolEnvironment
psrl/tools/sandbox_fusion_tool.py                       # SandboxFusionTool base class
```

---

## Prerequisites

### Hardware

- NVIDIA GPUs (tested on A100, H100; 8× GPUs per node)
- 4–8 nodes depending on the path (see table above)
- Shared filesystem visible on every node — used to stage
  the SandboxFusion image tar and the model checkpoints
- Docker on every worker node, with a daemon reachable by the current user

### Software

```bash
# 1. Create conda environment and install PSRL (from repo root).
#    Full details: ../../CONTRIBUTING.md and the Installation guide.
conda create -n psrl python=3.12
conda activate psrl

bash scripts/install_basic.sh      # includes torch_memory_saver (TMS)
bash scripts/install_nixl.sh
bash scripts/install_megatron.sh   # required for megatron_*.sh paths only
bash scripts/install_lmcache.sh

pip install -e .

# 2. Cluster-ops tools (used by docker_scripts/ and sandbox_fusion/)
#    pssh for multi-node SSH; skopeo for proxy-friendly image pulls.
dnf -y install pssh skopeo --nobest   # or apt-get equivalent

# 3. Verify
python -c "from psrl.tools.sandbox_fusion_tool import SandboxFusionTool; print('sandbox tool OK')"
python -c "from verl.utils.reward_score import math_dapo; print('math_dapo OK')"
docker version
```

### Models

Download one of the following to `${PSRL_WORKSPACE}/models/`:

| Path | Model | HuggingFace | Notes |
|------|-------|-------------|-------|
| A | `multiturn-sft-qwen-2.5-7b-instruct` | ByteDance-Seed cold-start SFT | Used by `fsdp_qwen_7b_dapo.sh` |
| B | `ReTool-Qwen-32B-SFT` | ByteDance-Seed official ReTool SFT | Used by `megatron_qwen_32b_dapo.sh` |
| C | `Qwen3-30B-A3B` | `Qwen/Qwen3-30B-A3B` | Used by `megatron_qwen3_30b_dapo_small.sh` |

> **Important**: after downloading, edit `config.json` and set
> `max_position_embeddings=32768`. The launch scripts assume this (see the
> `gen_actor_rollout_ref.model.override_config.max_position_embeddings=32768`
> flag). Without it, long multi-turn rollouts will crash vLLM.

The Megatron paths additionally convert the HF checkpoint to a distributed
checkpoint via `scripts/convert_hf_to_mcore.py`; this happens automatically
inside the launch script the first time it runs.

---

## Data Preparation

Download the three DAPO math parquets into `${PSRL_WORKSPACE}/data/dapo/`:

```
${PSRL_WORKSPACE}/data/dapo/
├── dapo-math-17k.parquet    # training set (BytedTsinghua-SIA/DAPO-Math-17k)
├── aime-2024.parquet        # validation set (optional)
└── aime-2025.parquet        # validation set (default)
```

`CustomRLHFDataset._read_files_and_tokenize` caches the post-mapping dataset
under `cache_dir/processed_datasets/<md5(path)>.parquet`; the first run will
do the mapping, subsequent runs will hit the cache.

`map_fn` recognises three `data_source` tags based on the parquet path
(`.../dapo/aime-2024-raw.parquet`, `.../dapo/aime-2024.parquet`,
`.../dapo/aime-2025.parquet`) and routes to the correct
`(problem, answer)` extraction. Any other parquet falls through to
`map_fn2`, which only appends the `\boxed{...}` answer-format instruction.

Every row ends up with:

| Field | Value |
|-------|-------|
| `prompt` | `[{"role": "user", "content": problem + answer_format}]` |
| `data_source` | `aime_2024_raw` / `aime_2024` / `aime_2025` / original |
| `ability` | `"MATH"` |
| `reward_model.ground_truth` | Stringified reference answer |
| `agent_name` | `"multi_turn_agent"` — selects `MultiTurnAgentLoop` |

---

## Sandbox Infrastructure

Tool calls are served by a cluster-local SandboxFusion service listening on
port `8080` on every node. Bringing it up is a two-step process; both steps
are documented in detail in their own READMEs.

### Step 1 — Bake the SandboxFusion image onto every node

See [docker_scripts/README.md](docker_scripts/README.md).

In brief:

```bash
# (a) Build or pull code_sandbox:server into a shared-FS tar.
DOCKERHUB_MIRROR=docker.m.daocloud.io \
DOCKER_INSTALL_METHOD=skopeo \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_install.sh

# (b) Fan the tar out to every node and docker load + retag.
DOCKER_NODE_IPS="${NODE_IPS}" \
DOCKER_NODE_NUM=8 \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_copy.sh
```

### Step 2 — Deploy the SandboxFusion Swarm service

See [sandbox_fusion/README.md](sandbox_fusion/README.md).

```bash
SANDBOX_NODE_IPS="${NODE_IPS}" \
SANDBOX_NODE_NUM=8 \
  bash examples/retool/sandbox_fusion/launch_service.sh
```

`launch_service.sh` will:

- Start `dockerd` on every host that isn't already running it.
- `docker swarm init` on the first host, `docker swarm join` everyone else.
- Create an overlay network and `docker service create ... --publish
  published=8080,target=8080,mode=host --replicas N`, so every node has a
  sandbox replica bound to `localhost:8080`.

### Step 3 — Smoke test

On any node (or every node via pssh):

```bash
curl 'http://localhost:8080/run_code' \
    -H 'Content-Type: application/json' \
    --data-raw '{"code": "print(2+3)", "language": "python"}'
```

Expected output:

```json
{
  "status": "Success",
  "run_result": {"status": "Finished", "return_code": 0, "stdout": "5\n", "stderr": ""},
  ...
}
```

If this fails from a rollout node, the tool calls will all return errors and
every episode will collapse to `-0.6` reward.

---

## Training

### Path A — FSDP Qwen2.5-7B (default)

```bash
bash examples/retool/fsdp_qwen_7b_dapo.sh
```

Optional positional argument — PSRL staleness (default `2`):

```bash
bash examples/retool/fsdp_qwen_7b_dapo.sh 3
```

### Path B — Megatron ReTool-Qwen-32B

```bash
bash examples/retool/megatron_qwen_32b_dapo.sh
```

Positional arguments:

1. `staleness` (default `1`)
2. `fix_weight` (default `False`) — profiling hook
3. `disable_attn` (default `False`) — profiling hook

### Path C — Megatron Qwen3-30B-A3B (MoE)

```bash
bash examples/retool/megatron_qwen3_30b_dapo_small.sh
```

Same positional arguments as Path B, default `staleness=3`.

### Monitor in wandb

| Metric | Meaning |
|--------|---------|
| `train/score` | Shaped reward: `+1.0` correct; `-1.0` incorrect (capped at `-0.6` when tool-call bonus kicks in); `0.0` for malformed boxed answer |
| `train/acc` | Strict boxed-answer accuracy (0 or 1) — the primary progress indicator |
| `val/acc` | Same metric run on the validation parquets every `test_freq` steps |

---

## Script Parameter Reference (`fsdp_qwen_7b_dapo.sh`)

### Cluster layout

| Variable | Default | Description |
|----------|---------|-------------|
| `NNODES` | `4` | Total nodes in the job |
| `NGPUS_PER_NODE` | `8` | GPUs per node |
| `GEN_NNODES` | `2` | Nodes dedicated to vLLM rollout |
| `TRAIN_NNODES` | `2` | Nodes dedicated to FSDP training |
| `GEN_TP` | `1` | Tensor-model parallelism for rollout vLLM |
| `GEN_PP` | `1` | Pipeline parallelism for rollout vLLM |
| `VAL_TP` | `1` | Tensor-model parallelism for validation vLLM on the training nodes |
| `VAL_PP` | `1` | Pipeline parallelism for validation vLLM |
| `TRAIN_SP` | `2` | Ulysses sequence parallelism during training |
| `TRAIN_FSDP` | `8` | FSDP shard group size |

`GEN_INSTANCES` and `VAL_INSTANCES` are derived automatically from the above.

### Algorithm

| Variable | Default | Description |
|----------|---------|-------------|
| `adv_estimator` | `grpo` | Advantage estimator; GRPO groups rollouts per prompt and normalises within the group |
| `clip_ratio_low` | `0.2` | Lower clip bound for the PPO probability ratio (DAPO asymmetric clipping) |
| `clip_ratio_high` | `0.28` | Upper clip bound |
| `use_kl_in_reward` | `False` | Whether to add a KL-penalty term to the reward |
| `kl_coef` | `0.0` | KL coefficient when `use_kl_in_reward=True` |
| `use_kl_loss` | `False` | Whether to add a KL-divergence term to the policy loss |
| `kl_loss_coef` | `0.0` | Weight of the KL loss term |

### Sequence lengths

| Variable | Default | Description |
|----------|---------|-------------|
| `max_turns` | `64` | Maximum agent turns (tool calls + final answer) per episode |
| `max_prompt_length` | `2048` | Maximum number of tokens in the initial prompt |
| `max_response_length` | `16384` | Maximum tokens generated per turn |

### Training hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `actor_lr` | `1e-6` | Actor learning rate |
| `train_prompt_bsz` | `64` | Unique prompts per training step (× `n_resp_per_prompt` rollouts) |
| `n_resp_per_prompt` | `8` | Rollouts per prompt during generation — GRPO group size |
| `n_resp_per_prompt_val` | `8` | Rollouts per prompt during validation |
| `train_prompt_mini_bsz` | `64` | Mini-batch size for PPO gradient steps |
| `loss_agg_mode` | `token-mean` | `token-mean` divides by total tokens, `seq-mean` by number of sequences |
| `enable_overlong_buffer` | `True` | Penalise responses longer than `max_response_length` with a linear ramp |
| `overlong_buffer_len` | `10240` | Size of the linear-ramp buffer beyond `max_response_length` |
| `overlong_penalty_factor` | `1.0` | Max penalty applied at the end of the buffer |

### Sampling

| Variable | Default | Description |
|----------|---------|-------------|
| `temperature` | `1.0` | Rollout temperature |
| `top_p` | `1.0` | Nucleus sampling threshold |
| `top_k` | `-1` | Top-k sampling (`-1` = disabled; use `0` for HF rollout) |
| `val_top_p` | `1.0` | Validation `top_p` |

### Token Importance Sampling (TIS)

| Variable | Default | Description |
|----------|---------|-------------|
| `rollout_is` | `token` | IS granularity: `token` (per-token ratio) or `sequence` (per-sequence ratio) |
| `rollout_is_threshold` | `2.0` | Clip rollouts whose IS weight exceeds this |

The PSRL `staleness` argument (first positional arg to the script, default `2`)
bounds how many training steps a rollout may remain in the replay buffer
before being discarded.

### Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `use_dynamic_bsz` | `True` | Pack sequences into fixed-length chunks |
| `packing_length` | `max_prompt_length + max_response_length` | Target chunk size |
| `offload` | `False` | CPU offload for optimizer state. Disabled because NIXL-CPU param sync doesn't support it yet |

---

## Megatron Script Reference (deltas)

[`megatron_qwen_32b_dapo.sh`](megatron_qwen_32b_dapo.sh) and
[`megatron_qwen3_30b_dapo_small.sh`](megatron_qwen3_30b_dapo_small.sh) share
the same algorithm / reward / tool wiring as Path A but differ in:

| Area | Difference |
|------|------------|
| Trainer config | `--config-name='ppo_trainer' model_engine=megatron` instead of `'ppo_trainer'` |
| Checkpoint | Requires a distributed Megatron checkpoint; `convert_hf_to_mcore.py` is invoked at the top of the script |
| Parallelism | `TRAIN_TP`, `TRAIN_PP`, `TRAIN_CP` (plus `TRAIN_EP` / `TRAIN_ETP` for MoE) replace `TRAIN_FSDP` / `TRAIN_SP` |
| Offload | `offload=True` is supported (param / optimizer / grad offload) |
| Reward manager | Uses `reward_model.reward_manager=dapo` with the built-in overlong-buffer config, in addition to `custom_reward_function` |
| Sequence lengths | `max_turns=16`, `max_response_length=30720`; longer responses but fewer turns than Path A |
| Generation | `GEN_TP` raised to `4` (32B) to fit the weights |
| Sampling | `val_top_p=0.7` instead of `1.0` |
| Advanced features | `psrl.partial_rollout.enable=True` on Path B; `algorithm.filter_groups.metric=acc` (DAPO dynamic sampling filter) |

Path C additionally configures MoE-specific parallelism (`TRAIN_EP`,
`TRAIN_ETP`, `NUM_LAYERS_IN_FIRST_PIPELINE_STAGE`) — see the top of the
script for the full list.

---

## Tool Configuration

[`sandbox_fusion_tool_config.yaml`](sandbox_fusion_tool_config.yaml) registers
a single tool instance used by every rollout:

```yaml
tools:
  - path: retool.py           # resolved relative to this YAML file
    class_name: CustomSandboxFusionTool
    params:
      sandbox_fusion_url: "http://localhost:8080/run_code"
      memory_limit_mb: 1024
      default_timeout: 30
      default_language: "python"
      name: "code_interpreter"
      description: "A tool for executing code."
      type: native
```

Key fields:

| Field | Description |
|-------|-------------|
| `path` | Python file to import. Relative paths resolve against the YAML's directory |
| `class_name` | Must subclass `psrl.tools.base.Tool`. `CustomSandboxFusionTool` subclasses `SandboxFusionTool` |
| `sandbox_fusion_url` | Each rollout worker POSTs here. `localhost:8080` works because the Swarm service publishes `mode=host` on every node |
| `memory_limit_mb` | Per-execution memory cap enforced by the sandbox container |
| `default_timeout` | Per-execution wall-clock cap (seconds) |
| `name` | The tool name the model sees in its tool schema — leave as `code_interpreter` to match the SFT format |
| `type` | `native` uses PSRL's direct async call path; leave as `native` |

You can register additional tools (file operations, search, …) by adding
more entries to the `tools:` list; each entry must resolve to a
`Tool.register`-decorated class.

---

## Reward Function

The reward is defined by `compute_score` in [retool.py](retool.py):

```python
def compute_score(data_source, solution_str, ground_truth, extra_info):
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)
    if result["pred"] is None:
        result["pred"] = ""
    return result
```

- `math_dapo.compute_score(..., strict_box_verify=True)` extracts the last
  `\boxed{...}` from the response (within the final 100 characters, or
  since the last pause token) and compares against `ground_truth`. Returns
  `{"score": +1.0 | -1.0, "acc": bool, "pred": str | None}`.
- When the answer is wrong (`score < 0`), the policy gets a small
  tool-call bonus proportional to `num_turns`. The bonus is
  `(num_turns - 2) / 2 * 0.1`, added to `-1.0`, and the result is then
  **floored at `-0.6`** (via `min(-0.6, ...)`). This means:

  | `num_turns` | Raw | Clamped `score` |
  |-------------|-----|-----------------|
  | 0 — 2 | `-1.0 – -1.1` | `-0.6` (bonus ≤ 0 ignored by floor) |
  | 4 | `-1.0 + 0.1 = -0.9` | `-0.6` |
  | 10 | `-1.0 + 0.4 = -0.6` | `-0.6` |
  | 20 | `-1.0 + 0.9 = -0.1` | `-0.1` |

  In other words the bonus encourages the model to actually call the tool
  (at least a handful of times) when it's heading for a wrong answer.
- Correct answers retain `score = +1.0` unchanged.
- `result["pred"]` being `None` is normalised to `""` so the downstream
  reward-manager logging doesn't crash.

`result["acc"]` (unshaped 0/1) is what ends up in wandb as `train/acc` and
`val/acc` — it's the metric to watch for genuine progress. `train/score`
includes the tool-call shaping and is noisier.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on port 8080 at rollout time | SandboxFusion service not running on this node | Re-run `sandbox_fusion/launch_service.sh`; check `docker service ps sandbox-service` on the manager |
| Every episode ends with `score=-0.6` | Tool endpoint unreachable → empty tool output → wrong boxed answer | Smoke-test with `curl` from a rollout node; verify `code_sandbox:server` is present on every node (`docker images | grep code_sandbox`) |
| `code_sandbox:server` missing on some nodes | `docker_copy.sh` failed on those nodes | Re-run `docker_copy.sh` with just the affected `DOCKER_NODE_IPS`; see [docker_scripts/README.md](docker_scripts/README.md) |
| Trailing `print(...)` auto-wrap produces `SyntaxError` | Multi-line final expression (e.g. indented inside a function) — `async_forward` only wraps the last non-empty line | Have the model emit an explicit `print(...)` as its last line. Worst case, edit the regex in `CustomSandboxFusionTool.async_forward` |
| vLLM `AssertionError` on long rollouts | `max_position_embeddings` in `config.json` too small | Edit `config.json` on every model copy to `max_position_embeddings=32768` |
| OOM during rollout | vLLM `gpu_memory_utilization` too high given tool-call queue depth | Lower `gen_actor_rollout_ref.rollout.gpu_memory_utilization` (default `0.3` FSDP / `0.9` Megatron) |
| Swarm `docker service create` hangs | Worker nodes never joined the swarm | Check `launch_service.sh` output for `pssh` auth errors; run `ssh-copy-id` to every worker first |
| `skopeo not found` | Package missing | `dnf -y install skopeo --nobest` (Docker CE installs containerd.io, which conflicts without `--nobest`) |

### Emergency cleanup

```bash
# Stop the Swarm service on the manager
bash examples/retool/sandbox_fusion/clear_service.sh

# Ray / training process
ray stop --force
pkill -9 -f main_ppo
```

---

## Extending

### Custom languages

`CustomSandboxFusionTool` defaults to Python but `SandboxFusionTool` supports
30+ languages (C++, Go, Rust, TypeScript, Java, SQL, …). Pass `language="rust"`
through the tool-call arguments, or override `default_language` in
`sandbox_fusion_tool_config.yaml`.

### Custom reward function

Any callable matching the `compute_score(data_source, solution_str,
ground_truth, extra_info)` signature works. Point the launch script at it:

```bash
custom_reward_function.path=/path/to/my_reward.py
custom_reward_function.name=my_compute_score
```

The function must return either a `float` or a dict with `score` and `acc`
keys (and optionally `pred` for logging).

### Custom tools

1. Subclass `psrl.tools.base.Tool` (or `SandboxFusionTool`) and decorate it
   with `@Tool.register("my_tool")`.
2. Add an entry to `sandbox_fusion_tool_config.yaml`:

    ```yaml
    - path: my_tool.py
      class_name: MyTool
      params: { ... }
    ```

3. Make sure your tool's `name` field matches what the model was SFT-trained
   to emit; otherwise the `hermes` parser won't route the call.
