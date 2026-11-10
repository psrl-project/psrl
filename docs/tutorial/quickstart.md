# Quick Start

This guide walks you through running a **DAPO** (Decoupled Clip and Dynamic Sampling
Policy Optimization) training job, which is the simplest end-to-end example for PSRL.

## Goal

Train a **Qwen2.5-3B-Instruct** model on the GSM8K math reasoning dataset using the
DAPO algorithm with asynchronous staleness-3 training across 2 nodes (16 GPUs).

## Minimal Setup

| Resource | Requirement |
|---|---|
| Nodes | 2 |
| GPUs per node | 8 |
| Backend | FSDP |
| Network | InfiniBand / RoCE recommended |

## Choose your runtime

If you use Docker, run the following command from the repository root before following
the steps below.

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/home/psrl" \
  -it psrl:latest
```

If you installed from source, activate the environment created by the
{doc}`installation` guide and run the commands from the repository root.

### Cluster Layout

| Variable | Value | Description |
|---|---|---|
| `NNODES` | 2 | Total nodes in the cluster |
| `NGPUS_PER_NODE` | 8 | GPUs per node |
| `GEN_NNODES` | 1 | Nodes dedicated to generation (rollout) |
| `TRAIN_NNODES` | 1 | Nodes dedicated to training |

## Steps

### 1. Download Model

Download the `Qwen2.5-3B-Instruct` model from HuggingFace:

```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
    --local-dir ${PSRL_WORKSPACE}/models/Qwen2.5-3B-Instruct
```

The training script already overrides `max_position_embeddings` to 32768 at runtime via
`+train_actor_rollout_ref.model.override_config.max_position_embeddings=32768`, so no
manual edit of `config.json` is required.

### 2. Download Data

This example uses the **GSM8K** dataset, with both `train.parquet` and `test.parquet`
expected at `${PSRL_WORKSPACE}/data/gsm8k/`:

```
${PSRL_WORKSPACE}/data/gsm8k/
├── train.parquet
└── test.parquet
```

:::{tip}
PSRL reuses veRL's data preprocessing pipeline. For a step-by-step walkthrough of how
to download and convert GSM8K (and other datasets such as MATH, HellaSwag,
Full_hh_rlhf) into the expected parquet format, refer to the official veRL data
preparation documentation:

**→ [veRL: Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)**
:::

### 3. Set Up Ray Cluster

We recommend using the bundled `examples/ray/ray_start.sh` helper, which cleans up any
stale Ray processes and launches the head + worker nodes from a hostfile in one shot.

First prepare a hostfile listing all nodes (the **first** line is the head node, the
rest are workers):

```
# ${PSRL_WORKSPACE}/hosts/16GPUs
192.168.1.1
192.168.1.2
```

Replace these example hostnames with the actual hostnames or addresses of the nodes in your cluster.

Then start the cluster from the launch node:

```bash
bash examples/ray/ray_start.sh ${PSRL_WORKSPACE}/hosts/16GPUs
```

Verify the cluster is ready:

```bash
ray status
# Should show 2 nodes, 16 GPUs total
```

:::{note}
The script uses `pssh` to fan out, port `8887` for Ray node-to-node communication, and
exposes the dashboard on port `8265`. Override `PORT` / `DASHBOARD_PORT` inside the
script if those clash with existing services on your cluster.
:::

### 4. Launch Training

Run the DAPO training script with the default staleness of 3:

```bash
bash examples/dapo_trainer/qwen2.5_3b_fsdp.sh
```

To specify a different staleness value, pass it as the first argument:

```bash
# Set staleness=2
bash examples/dapo_trainer/qwen2.5_3b_fsdp.sh 2
```

:::{admonition} About Staleness
:class: tip

Staleness controls the maximum version gap between generation and training:

- `staleness=0`: fully synchronous (generation blocks until training catches up)
- `staleness=3` (default for this example): generation can run up to 3 versions ahead of training
- Higher values increase throughput but may degrade sample quality

Values of 2 to 3 provide a good balance between throughput and policy freshness for
most workloads.
:::

### 5. Monitor Training

PSRL logs to [Weights & Biases](https://wandb.ai) by default. The most useful metrics
to watch are the **critic** statistics that summarize reward quality, together with a
few actor / performance counters:

| Metric | Description |
|---|---|
| `critic/score/mean` | Mean raw task score on the training batch (e.g. GSM8K correctness, range [-1, 1]) |
| `critic/score/max` / `critic/score/min` | Best / worst raw score in the batch, useful for spotting reward collapse |
| `critic/rewards/mean` | Mean shaped reward (raw score + overlong penalty) fed into advantage estimation |
| `critic/advantages/mean` | Mean GRPO advantage used for the policy update |
| `critic/returns/mean` | Mean return target seen by the actor (equals advantage under GRPO) |
| `response_length/mean` | Mean generated response length (tokens), should stabilize as training converges |
| `actor/entropy` | Policy entropy, a slow decrease indicates healthy exploration → exploitation |
| `actor/grad_norm` | Gradient norm, sudden spikes often signal an unstable update |
| `perf/throughput` | End-to-end throughput in tokens/sec/GPU |
| `perf/mfu/actor` | Model FLOPs Utilization of the actor update |
| `training/global_step` | Current training step |

You should see `critic/score/mean` and `critic/rewards/mean` trend upward over the
first few hundred steps as the policy learns to solve more GSM8K problems.

## What Happens Under the Hood

1. **AgentLoopWorkers** drive rollout. For a DAPO math task each prompt runs as a
   **single-turn agent** (one prompt → one response, no tool calls). They send
   generation requests through the **SMG RolloutGateway**, whose PSRL worker
   selector checks version/admission state with PSManager before dispatching to a
   vLLM gRPC replica.
2. The selected **vLLM instances** (1 node here) generate rollouts from the current
   policy version.
3. Completed trajectories are written to **TransferQueue**, and the trainer receives a
   lightweight `KVBatchMeta` and reads/writes fields as the DAPO stages execute.
4. After each training step, training workers push updated weights to the
   **Parameter Server**. Rollout workers pull new versions when coordinated by the
   GlobalCoordinator.

```{seealso}
- {doc}`configuration`: Full parameter reference and override syntax
- {doc}`../design/staleness_control`: Deep dive into staleness control mechanisms
- {doc}`../design/flexible_rollout`: Flexible rollout coordination
```
