# AIRS-Bench RL Recipe

This recipe trains a language model with GRPO on the AIRS-Bench machine learning benchmark
suite. The agent is MLGym's autonomous ML researcher scaffold. It submits predictions to a
sandboxed AIRS-Bench grader and receives a normalized score between 0 (worst baseline) and 1
(SOTA) as the reward signal.

## What the Recipe Trains

The policy learns to perform autonomous machine learning: it is given a task description, a
training dataset, and a Python environment, and must produce a submission CSV by writing and
running code inside a Docker sandbox. The reward is the AIRS normalized score defined in
`reward.py`. The dataset has 14 training tasks and 6 held-out validation tasks. The split is
pinned in `prepare/split.json`.

## Prerequisites

1. A two-node Ray cluster. The compute node (`29.162.247.148`) hosts rollout and training GPUs.
   The env node (`28.49.16.220`) hosts the EnvWorker pool and runs containers.
2. The AIRS-Bench repository cloned locally (referred to as `AIRS_REPO`).
3. The Qwen3-4B model at `/jizhicfs/johnnyslin/models/Qwen3-4B`.
4. Shared filesystem at `/apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data`.

## Data Preparation

Run the three stages in order. Each stage is idempotent. Re-running skips completed work.

**Stages 1-3: Download and prepare AIRS-Bench datasets**

```bash
source ${PSRL_WORKSPACE}/env/psrl.sh
python -m examples.airs_bench.prepare.prepare_airs_data \
    --airs-repo "${AIRS_REPO}" \
    --data-root /apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data
```

This downloads the raw HuggingFace datasets (stage 1), runs each task's `prepare.py` and
`evaluate_prepare.py` to build the train, test, and test-with-labels splits (stage 2), and
rewrites the MLGym dataset and task YAML files to point at the prepared data (stage 3).

**Stage 4: Build the training parquet files**

```bash
python -m examples.airs_bench.prepare.build_parquet \
    --airs-repo "${AIRS_REPO}" \
    --data-root /apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data \
    --out-dir examples/airs_bench/data
```

This emits `examples/airs_bench/data/train.parquet` (14 rows) and
`examples/airs_bench/data/val.parquet` (6 rows). Each row is one task. The normalization
constants the reward function needs are embedded in each row's `extra_info` field.

## Container Image

**Build the sandbox image on one node:**

```bash
bash examples/airs_bench/docker/build_image.sh
```

**Fan out the image to all cluster nodes:**

```bash
HOSTS="28.49.16.220 29.162.247.148" bash examples/airs_bench/docker/fanout_image.sh
```

The image is saved to shared storage once, then loaded in parallel on each target node via
`pssh`. Verify with `docker images psrl/airs-bench-agent:latest` on both nodes.

## Cluster Layout

```
28.49.16.220   env node    EnvWorker pool only, no PSRL train or rollout GPUs
29.162.247.148 compute     GPU 0-3: rollout (vLLM, TP=2, 2 instances)
                           GPU 4-7: training (Megatron, TP=2)
```

The `total_nnodes=1` setting tells PSRL's excess-node reserver to fence the env node out of
the Ray resource pool, so train and rollout workers never land on it.

**To use colocated placement instead** (single-node, sandboxes share GPU nodes with PSRL):

```bash
bash examples/airs_bench/run_qwen3-4b.sh \
    psrl.env_worker.placement=colocated \
    ++psrl.env_worker.dedicated_node_ips=[]
```

Colocated placement puts env workers on every alive Ray node alongside train and rollout.
It is simpler to operate but the training node's CPU and memory compete with running sandboxes.

## Running the Training

Start Ray on both nodes, then launch:

```bash
bash examples/ray/ray_start.sh /tmp/airs_hosts.txt
bash examples/airs_bench/run_qwen3-4b.sh
```

Key environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HF_MODEL_PATH` | `/jizhicfs/johnnyslin/models/Qwen3-4B` | Model weights path |
| `AIRS_DATA_ROOT` | `.../airs_bench_data` | Shared data root |
| `SANDBOX_IMAGE` | `psrl/airs-bench-agent:latest` | Container image |
| `N_RESP_PER_PROMPT` | `4` | GRPO group size |
| `MAX_TURNS` | `40` | Episode turn limit |
| `TOTAL_EPOCHS` | `30` | Training epochs |

A short two-epoch smoke run:

```bash
TOTAL_EPOCHS=2 TEST_FREQ=1 bash examples/airs_bench/run_qwen3-4b.sh 2>&1 | tee /tmp/airs_e2e.log
```

Watch for these log lines in order to confirm healthy startup:

1. `Env worker pool ready with 1 worker(s) across 1 node(s) using placement 'dedicated'`
2. `Registered env worker 0 on 28.49.16.220 with 8 cpu slot(s)`
3. Rollout gateway and session router URLs logged
4. `AIRS-Bench episode for '<task>' finished with status 'submitted' after N turn(s)`
5. Non-zero reward mean in the step metrics

## Debugging with the Scripted Episode

The scripted episode runs one AIRS-Bench episode with fixed actions and no LLM. Any failure
here is infrastructure (container, shell protocol, data mounts, or grading), not policy.

```bash
source ${PSRL_WORKSPACE}/env/psrl.sh
python -m examples.airs_bench.scripted_episode \
    --task-id TextualClassificationSickAccuracy
```

### Episode Failure Table

| Symptom | Likely cause | Fix |
|---|---|---|
| Container not found | Image not loaded on env node | Run `fanout_image.sh` on env node |
| Shell probe timeout | Base image or login shell broken | Rebuild image, verify bash |
| MLGym workspace setup fails | Data not mounted at expected path | Check `AIRS_DATA_ROOT` mount |
| Grading returns zero | `test_with_labels` missing | Re-run stage 2 of data preparation |
| Sentinel parse error | Command emitted the sentinel pattern | Escape the output in the command |
| Idle sandbox reaped mid-episode | `idle_sandbox_timeout_s` too low | Increase to match episode wall-clock |

## Deliberate Deviations from Vanilla MLGym

These changes are required for PSRL correctness and are documented here so that future
MLGym version bumps do not silently reintroduce problems.

### 1. History processor is pinned to `DefaultHistoryProcessor`

Vanilla MLGym ships `Last5Observations` as the default history processor, which rewrites
earlier messages in the conversation by dropping them. PSRL's TITO system records training
data by matching message prefixes against the SMG session. Any processor that mutates or
removes earlier messages changes those prefixes and corrupts TITO's token alignment,
producing misaligned training tokens rather than an error.

`DefaultHistoryProcessor` appends each new observation without modification, which keeps the
prefix hash stable. The `assert_default_history_processor` guard in `mlgym_env.py` fails
loudly at episode start if a different processor is configured, so the corruption is caught
before training begins. Context growth is bounded by observation truncation (see below) and
by the `max_turns` limit instead.

### 2. Observations are truncated to `max_observation_chars` head and tail

Vanilla MLGym does not bound command output length. A single training run command can emit
megabytes of log text, which exceeds the model context in one turn and causes TITO to write
an oversized trajectory into TransferQueue.

The `truncate_observation` function in `shell.py` caps each observation to a configurable
character budget (default 8000 characters per side), keeping the head (which carries the
command echo and early errors) and the tail (which carries the final result). The elision
notice shows how many characters were omitted.

### 3. `--gpus` is replaced by device passthrough and driver bind-mounts

The NVIDIA Container Toolkit (`--gpus all`) is absent on this cluster. Instead, the sandbox
builder in `sandbox.py` exposes individual `/dev/nvidiaN` character devices and bind-mounts
the driver libraries (`libcuda.so.1`, `libnvidia-ml.so.1`) from the host. This was verified
to work on H20 nodes. The implementation is in `build_gpu_argv` and `build_docker_run_argv`
in `psrl/workers/env_worker/sandbox.py`.

AIRS-Bench grades on CPU (no GPU tasks in the current 20-task set), so `gpu_slots_per_worker`
is set to 0 in this recipe and no device passthrough occurs during normal training. The GPU
path is exercised by MLGym's native tasks and is ready for future use.

### 4. `cache_baseline_scores` is disabled

MLGym's baseline scorer commits the baseline model's predictions to a file inside the sandbox
image, so that the agent can see the baseline score at episode start. Enabling it would require
building a per-task image variant with committed predictions, which defeats the purpose of a
shared sandbox image.

Baseline caching is disabled in `agent_config.yaml`. Agents see the task description and
dataset but not the baseline score. This makes the task slightly harder but avoids image
proliferation and the accompanying fan-out cost.

## Known Limitations

The training split has 14 tasks. With 4 rollouts per prompt, each training step produces 56
trajectories from only 14 distinct task instances. This invites memorization: the model can
achieve reward by overfitting to these 14 tasks rather than learning general ML reasoning.
The 6 held-out validation tasks are the only out-of-distribution guard.

The episode wall-clock is the primary throughput risk. Each episode runs up to 40 turns of
real code execution, which can take tens of minutes for tasks with long training jobs. If the
rollout pipeline starves training, reduce `MAX_TURNS`, reduce `N_RESP_PER_PROMPT`, or raise
`psrl.staleness` to allow more off-policy data.
