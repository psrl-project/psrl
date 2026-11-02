# Elastic RL

Use elastic deployment when policy rollout and generative reward inference have
changing GPU demand. PSRL sleeps and wakes pre-initialized replicas on shared GPUs and
can lend idle training GPUs to inference. Choose a launch path:

| Path | Use it when |
|------|-------------|
| [Run Existing Launch Scripts](#run-existing-launch-scripts) | Your model pair and GPU layout match a recipe under `examples/elastic/` |
| [Customize a Run](#customize-a-run) | You need different runtime settings, models, data, or GPU placement |

```{seealso}
See {doc}`../design/resource_elasticity_configuration` for configuration defaults and
{doc}`../design/resource_elasticity` for the runtime and scaling policy design.
```

## Run Existing Launch Scripts

Select a model-pair recipe, prepare its inputs, then run its smoke or full settings.

### Prerequisites

Complete {doc}`../tutorial/installation` on every node, including the patched vLLM and SMG,
NIXL, and `torch_memory_saver`. Megatron recipes also need the Megatron training
dependencies. Launch from the repository root with a running Ray cluster.

| Input | Requirement |
|-------|-------------|
| `PSRL_WORKSPACE` | Shared workspace containing the repository, models, data, and local environment setup |
| `env/psrl.sh` | Local activation script under the workspace, available on every node |
| Policy and reward checkpoints | Complete Hugging Face model directories, including `config.json` |
| Training data | Prepared parquet files at the paths below |
| Cost models | JSON files under `psrl/trainer/config/cost_model/` matching the model and TP/PP |
| Network | Working Ray and NIXL connectivity between the participating nodes |

`env/psrl.sh` is local configuration and is not committed. Have it activate the
installed PSRL environment and configure the transport libraries for your cluster.
The launcher sources this file itself.

### Choose a Recipe

These are the **launcher allocations**, not the paper's benchmark allocations.
Training GPUs may also serve inference while the trainer sleeps.

| Script under `examples/elastic/` | Policy / reward TP | Training pool | Shared pool |
|---------------------------------|--------------------|---------------|-------------|
| `qwen2.5_1.5b_glm_z1_9b.sh` | 1 / 1 | 1 node x 4 GPUs | 1 node x 8 GPUs |
| `qwen2.5_7b_qwen3_30b.sh` | 1 / 4 | 1 node x 8 GPUs | 3 nodes x 8 GPUs |
| `qwen2.5_7b_qwen3_30b_megatron.sh` | 1 / 4 | 1 node x 8 GPUs | 3 nodes x 8 GPUs |
| `qwen2.5_32b_qwen3_30b.sh` | 4 / 4 | 4 nodes x 8 GPUs | 4 nodes x 8 GPUs |
| `qwen2.5_72b_qwen3.5_122b.sh` | 8 / 8 | 4 nodes x 8 GPUs | 4 nodes x 8 GPUs |

The policy checkpoints are Qwen2.5-1.5B, Qwen2.5-7B, Qwen2.5-32B, and Qwen2.5-72B.
The reward checkpoints are GLM-Z1-9B-0414, Qwen3-30B-A3B-Thinking-2507, or
Qwen3.5-122B-A10B as indicated by the script. The Megatron 7B wrapper selects
TP2/PP2 training. Other recipes default to FSDP2.

### Prepare the Workspace

```bash
export PSRL_WORKSPACE="$PWD"
source env/psrl.sh
```

Put checkpoints under `models/<model-name>/`, or set
`PSRL_DEPLOY_MODEL_PATH` and `PSRL_DEPLOY_RM_MODEL_PATH` to their locations.
Every node must be able to read them.

| Dataset selection | Required parquet files under the workspace |
|-------------------|---------------------------------------------|
| `gsm8k` | `data/gsm8k/train.parquet`, `data/gsm8k/test.parquet` |
| `dapo` | `data/dapo/dapo-math-17k.parquet`, `data/dapo/aime-2024.parquet` |
| `mixed` | Both sets, sampled with equal training ratios |

The 32B and 72B recipes use `data/gsm8k_verl/` instead of `data/gsm8k/`.
Prepare the data as described in {doc}`../tutorial/quickstart`. The launcher attaches the
verifiable and generative reward definitions to each dataset. Its default dataset
selection is `mixed`.

Start Ray on the selected nodes using your cluster setup, then check membership:

```bash
ray status
```

### Launch a Smoke Run

The 1.5B recipe needs two GPU nodes with the allocation in the table above. This
command selects the Python evaluator, so no C++ binary is needed, and uses console
logging:

```bash
PSRL_DEPLOY_DATASET=gsm8k \
PSRL_CANDIDATE_EVALUATION_BACKEND=python \
bash examples/elastic/qwen2.5_1.5b_glm_z1_9b.sh 1 \
  trainer.logger='[console]' trainer.wandb_proxy=null
```

The first positional argument is the **smoke switch**, not staleness: `1` selects
the recipe's smoke settings and `0` selects its full settings. For the 1.5B recipe,
smoke mode uses 10 steps, 32 prompts per batch, 8 responses per prompt, and disables
periodic validation and checkpoint saving. Smoke lengths differ across recipes.

To inspect Hydra configuration without starting training, add `--cfg job --resolve`
to the same launch command. Model and environment checks in the shell launcher
still run.

### Run with C++ Candidate Evaluation

Build on the launch host before selecting `cpp`. CMake needs a C++17 compiler,
RapidJSON headers, and pthreads. The evaluator runs in the `ElasticExecutor` actor,
so its executable must also be available at the resolved path on the actor's node.

```bash
cmake -S tools/elastic_simulator -B build/elastic_simulator \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/elastic_simulator -j

PSRL_DEPLOY_DATASET=gsm8k \
PSRL_CANDIDATE_EVALUATION_BACKEND=cpp \
bash examples/elastic/qwen2.5_1.5b_glm_z1_9b.sh 1 \
  trainer.logger='[console]' trainer.wandb_proxy=null
```

Set `PSRL_CANDIDATE_EVALUATION_CPP_BINARY` for a different executable location.
Missing binaries and evaluator process failures are errors. They do not
automatically select Python.

### Full Runs, Staleness, and Validation

Use `0` as the first positional argument to select a recipe's full-run settings.
The full run uses the same workspace, models, and evaluator setup as its smoke run.

Use `STALENESS=0` for synchronous operation or a positive value for asynchronous
operation. Recipes default to `STALENESS=2`. This setting controls sample-version
staleness independently of elastic scaling.

Set `PSRL_NEED_VALIDATION=1` to enable validation every five steps, including smoke
runs. Dedicated validation replicas must remain configured even when periodic
validation is disabled. To use Megatron for the 7B pair, launch
`examples/elastic/qwen2.5_7b_qwen3_30b_megatron.sh` with the same arguments.

## Customize a Run

Use trailing Hydra overrides for runtime tuning. For a different model pair or GPU
topology, create a launcher from the closest recipe and keep the shared launch body
in `examples/elastic/_common_elastic.sh`.

### Override Runtime Settings

Pass Hydra overrides **after** the smoke switch. This example uses the 7B/30B
recipe's models and placement while changing the scaling cooldown:

```bash
PSRL_CANDIDATE_EVALUATION_BACKEND=python \
bash examples/elastic/qwen2.5_7b_qwen3_30b.sh 0 \
  psrl.deployment.elastic.cooldown_ms=20000 \
  psrl.deployment.elastic.throughput_optimal.decision_window_s=60 \
  trainer.logger='[console]' trainer.wandb_proxy=null
```

### Create a Launcher for Your Models and Topology

Copy a recipe into the same directory so it can source `_common_elastic.sh`:

```bash
cp examples/elastic/qwen2.5_7b_qwen3_30b.sh \
  examples/elastic/custom_run.sh
```

Edit the copy's exports before running it:

| Settings in `custom_run.sh` | What to match |
|-----------------------------|---------------|
| `PSRL_DEPLOY_MODEL_NAME`, `PSRL_DEPLOY_RM_MODEL_NAME` | Policy and reward checkpoint names |
| `PSRL_DEPLOY_ROLLOUT_TP`, `PSRL_DEPLOY_RM_TP`, `PSRL_DEPLOY_VAL_TP` | Each inference role's tensor parallelism |
| `PSRL_DEPLOY_NNODES`, `PSRL_DEPLOY_TRAIN_NNODES`, `PSRL_DEPLOY_TRAIN_NGPUS` | Total node limit and training pool |
| `PSRL_DEPLOY_SHARED_NNODES`, `PSRL_DEPLOY_SHARED_NGPUS` | Shared inference pool, sized for complete replica groups |
| `PSRL_DEPLOY_ROLLOUT_COST_MODEL`, `PSRL_DEPLOY_RM_COST_MODEL` | Cost-model JSON names with the matching TP/PP buckets |
| `PSRL_DEPLOY_GEN_GPU_MEMORY_UTILIZATION`, `PSRL_DEPLOY_RM_GPU_MEMORY_UTILIZATION` | Available memory on GPUs shared across roles |
| `PSRL_ENABLE_HETEROGENEOUS_PARALLELISM_CANDIDATES` | Enable when policy and reward replica sizes differ |

Set `PSRL_DEPLOY_MODEL_PATH` and `PSRL_DEPLOY_RM_MODEL_PATH` if checkpoints are
outside the workspace's `models/` directory. Select `PSRL_DEPLOY_DATASET=gsm8k`,
`dapo`, or `mixed` for the supplied data layouts. For another dataset, update the
dataset definitions in your launcher and its `data.train_datas` / `data.val_datas`
overrides together.

Set RoPE overrides under `rollout.engine_kwargs.vllm.hf_overrides.rope_parameters`,
using the checkpoint's RoPE settings and vLLM's parameter names. The GLM recipe
sets YaRN's `rope_type`, `factor`, and `original_max_position_embeddings` there.
vLLM validates the context length with these overrides during engine creation.

**Change topology through the launcher's `PSRL_DEPLOY_*` exports.**
Many model, node, TP, and memory values are assigned unconditionally by the model
script. The shared launch body derives instance sizes and validation counts from
them before applying trailing Hydra overrides. Changing only one final Hydra
field can leave these derived settings inconsistent.

### Prepare Cost Models

Each policy and reward model needs coefficients matched to its GPU type and
inference TP/PP. Reuse the JSON files in `psrl/trainer/config/cost_model/` when
those settings match. For another setup:

1. Collect decoding measurements with
   `examples/bench/rollout/run_rollout_test.sh`, sweeping batch sizes with attention
   both enabled and disabled on the target GPUs.
2. Fit the collected JSONL files with
   `examples/anaylsis/fit_cost_model_from_jsonl_dir.py`. Its per-model output is
   `<model-name>_cost_model_fit.json`, containing the fitted TP/PP buckets.
3. Save the per-model result under the intended cost-model filename and set both
   launcher cost-model exports. Configure the scaling policy's cost-model directory too
   if you store the files outside its default directory.

See {doc}`../design/resource_elasticity_configuration` for the separate scaling policy
and routing paths, the JSON fields, and required elastic settings.

### Inspect and Launch

Inspect the custom launcher's resolved configuration, then run its smoke settings:

```bash
PSRL_CANDIDATE_EVALUATION_BACKEND=python \
bash examples/elastic/custom_run.sh 1 --cfg job --resolve \
  trainer.logger='[console]' trainer.wandb_proxy=null

PSRL_CANDIDATE_EVALUATION_BACKEND=python \
bash examples/elastic/custom_run.sh 1 \
  trainer.logger='[console]' trainer.wandb_proxy=null
```

Confirm the models, resource pools, validation placement, and routing strategies
before switching the first argument to `0` for a full run.

### Monitor a Run

Both launch paths write runtime logs below `logs/psrl_elastic/<experiment-name>/` and a
combined launch log below `logs/`. `ScalingPolicy.log` records candidate gains,
penalties, and no-action reasons. `ElasticMonitor.log` records awake counts,
engine load, router backlog, and snapshot readiness. `ElasticExecutor.log` records
accepted plans and coordinator transitions.
