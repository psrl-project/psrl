# SciAccel-RL: agentic RL on scientific-computing repair tasks

> [!IMPORTANT]
> **This example has moved to its own repository:
> [ScienceInfra](https://github.com/Gen-Verse/ScienceInfra).**
>
> Development continues there, as an installable `scienceinfra` package with a
> bundled demo task bank. This copy still runs, but it is the older version and
> is no longer the one being updated. Use ScienceInfra for new work.

Trains a model to fix injected defects in real scientific simulation codebases
(LAPS, MITgcm, Athena++). Each task is a containerized [Harbor](https://github.com/laude-institute/harbor)
episode: the agent gets a repository and an instruction, edits source, and a
verifier recompiles and compares numerical output against reference frames.

The reward is earned by making a simulation numerically correct again, not by
matching a diff. That makes it unusually hard to game and unusually slow to
grade: budget 8 to 15 minutes per episode for the image build alone.

## Results

Qwen3.5-4B, GRPO at hint level L1 on 3 nodes, trained separately on two
environments. Both are the `repair` category at the `easy` tier.

### `laps`: LAPS, MHD in Fortran

<p align="center">
  <img src="../../assets/sciaccel_rl_laps_L1_curves.png" alt="laps: reward, response length, entropy, and train-inference KL over 31 GRPO steps" width="100%">
</p>

Reward climbs from 0.49 to 0.77, peaking at 0.91. Response length falls from
~40k tokens to ~28k over the same window, so the gain is not bought by rambling:
the policy is finding the defect in fewer tokens.

### `mitgcm-biogeo`: MITgcm biogeochemistry

<p align="center">
  <img src="../../assets/sciaccel_rl_mitgcm_biogeo_L1_curves.png" alt="mitgcm-biogeo: reward, response length, entropy, and train-inference KL over 30 GRPO steps" width="100%">
</p>

Reward climbs from ~0.42 to ~0.53, peaking at 0.71, on a noisier curve than
laps. Response length moves the other way here, ~24k tokens to ~31k, so on this
bank the policy buys accuracy with more exploration rather than less.

In both runs entropy decays smoothly rather than collapsing, and
train-inference KL stays low and trends down, which is the check that the
rollout and trainer policies have not drifted apart. Reproduce either figure
with [`plot/plot.py`](plot/plot.py) against that run's log.

## Quickstart

```bash
# 1. Get the task bank (separate repo, authored on its own cadence).
git clone https://github.com/HHHHHejia/sciaccel-rl.git -b easy-rl ${SCIACCEL_REPO}

# 2. Compile tasks, resolve hint lines, build datasets, warm image caches.
bash examples/sciaccel_rl/prepare/prepare_all.sh \
    --repo ${SCIACCEL_REPO} --envs mitgcm-biogeo \
    --hosts 192.168.1.1,192.168.1.2

# 3. Prove the harness works before spending GPUs. oracle must score ~1.0.
bash examples/sciaccel_rl/eval/run_eval.sh --agent oracle \
    --dataset examples/sciaccel_rl/data/mitgcm-biogeo/repair_easy/all/L1.parquet \
    --output-dir examples/sciaccel_rl/outputs/anchor --skip-gpu-tasks -n 1

# 4. Train.
bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
```

Step 3 is not optional. A model number measured against a broken verifier is
worse than no number.

## Where things are

| Path | Purpose |
|------|---------|
| [`prepare/`](prepare/) | **Data preparation.** Compile tasks, build hinted datasets, warm caches |
| [`eval/`](eval/) | **Standalone eval**, independent of the training stack |
| [`fsdp_qwen35_4b.sh`](fsdp_qwen35_4b.sh) | GRPO training entry point |
| [`eval_qwen35_4b.sh`](eval_qwen35_4b.sh) | Score a checkpoint through PSRL, no training |
| [`agent_loop.py`](agent_loop.py) | PSRL agent loop, one Harbor episode per prompt |
| [`runner.py`](runner.py) | Black-box Harbor job runner |
| [`agent.py`](agent.py) | terminus-2 subclass with observation truncation |
| [`reward.py`](reward.py) | Verifier reward extraction |
| [`exceptions.py`](exceptions.py) | Harness failure to `TerminateReason` mapping |
| [`config.py`](config.py) | Runtime config dataclasses |
| [`config/`](config/) | Agent loop registration, chat template, compose overlays |
| [`plot/plot.py`](plot/plot.py) | Training curves from a run log |

## Conventions

| Placeholder | Meaning |
|---|---|
| `${PSRL_WORKSPACE}` | Your workspace root, holding `env/`, `models/`, `hosts/` |
| `${SCIACCEL_REPO}` | Your checkout of the task bank |
| `192.168.1.x` | Stand-in node addresses. Substitute your own |

Paths inside this repository are written relative to the repository root, so run
every command from there.

---

## The task bank

Five environments ship in the task bank, and every one has the same shape, so
nothing in this recipe is env-specific:

```
envs/<env>/
├── env/harbor/                 # Dockerfile and instruction templates
├── factory/harbor_spec.py      # What "densify" means for this codebase
└── tasks/<category>/<tier>/    # Authored task sources (sparse)
```

| env | Codebase | Easy repair tasks |
|---|---|---|
| `laps` | LAPS (MHD, Fortran) | 32 |
| `mitgcm-biogeo` | MITgcm biogeochemistry | 87 |
| `mitgcm-atmos` | MITgcm atmosphere | 123 |
| `athena-gr` | Athena++ general relativity | 104 |
| `athena-fft` | Pinned Athena++ source archive only, reused by `athena-gr` | n/a |

Authored tasks are **sparse** and must be compiled before Harbor can build them.
That, the L1/L2/L3 hint levels, the dataset layout, and node warming are all in
[`prepare/README.md`](prepare/README.md).

---

## Train

```bash
bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
```

Defaults to `mitgcm-biogeo/repair_easy` at `L1` on 3 nodes (8 generation + 16
training GPUs). Hydra overrides pass straight through, and any of these can be set
in the environment:

```bash
HF_MODEL_PATH=${PSRL_WORKSPACE}/models/Qwen3.5-4B \
DATA_DIR=examples/sciaccel_rl/data/athena-gr/repair_easy \
HINT_LEVEL=L1 \
AGENT_NODE_IPS=192.168.1.1,192.168.1.2 \
    bash examples/sciaccel_rl/fsdp_qwen35_4b.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `HF_MODEL_PATH` | `${PSRL_WORKSPACE}/models/Qwen3.5-4B` | Model to train |
| `DATA_DIR` | `data/mitgcm-biogeo/repair_easy` | Dataset directory. Becomes part of the experiment name. |
| `HINT_LEVEL` | `L1` | `L1`, `L2`, or `L3`. Selects `train/${HINT_LEVEL}.parquet`. |
| `VAL_FILES` | `eval/${HINT_LEVEL}.parquet` | Point at `eval/unhinted.parquet` to score unaided localization. |
| `AGENT_NODE_IPS` | empty | Nodes allowed to host containers. Empty means all alive nodes. |
| `OVERLONG_FILTERING` | `True` | DAPO overlong filtering. See below. |
| `GROUP_FILTER` | `False` | Drop zero-variance GRPO groups. |
| `MAX_TURNS` | `50` | Turn cap. Raising it needs `MAX_RESPONSE_LENGTH` raised too. |
| `OUTPUT_DIR` | `examples/sciaccel_rl` | Root for `ckpts/` and `psrl_logs/`. |

### Metrics that matter

Watch these rather than `critic/score/mean` alone, which blends populations that
move in opposite directions:

- `termination/finished/score_mean`: reward on episodes that actually finished
- `termination/verifier_error/fraction`: share whose verifier never produced a
  score. These are masked out of the gradient, but a rising number means the
  cluster is eating rollouts
- `termination/max_turns_exceeded/fraction`: share cut off by the turn cap
- `group/zero_variance_fraction`: share of GRPO groups producing no gradient
- `rollout_corr/rollout_is_eff_sample_size`: rollout-vs-trainer agreement. A drop
  here means a weight-transfer problem, not an RL one

`OVERLONG_FILTERING=True` zeroes the loss mask of episodes whose reward reports
the harness rather than the policy, while keeping that reward in the GRPO
baseline. Two cases qualify:

- **Budget-truncated**, cut off mid-work by the turn or length cap. Grading that
  as a policy failure makes `token-mean` reward shorter turns, which spends the
  turn cap faster still. Leaving this off has collapsed a run.
- **Ungraded**, where the verifier never ran. Its 0.0 is a missing measurement
  rather than a measured failure, so training it is pure infrastructure noise.

---

## Evaluate

### Through PSRL, reusing the training topology

```bash
EVAL_BASE=False \
CKPT_PATH=examples/sciaccel_rl/ckpts/sciaccel_rl_mit/<experiment>/global_step_30 \
    bash examples/sciaccel_rl/eval_qwen35_4b.sh
```

`val_only=True` returns straight after the initial validation, so nothing trains
and no optimizer step runs. `resume_mode=resume_path` loads the named checkpoint
rather than searching, which matters: `auto` would find nothing in the empty eval
directory and silently score the **base** model. `EVAL_BASE=True`, the default,
scores the untrained weights and is the baseline every checkpoint is measured
against.

The checkpoint is FSDP-sharded across 16 ranks, so it must be scored on the same
`TRAIN_NNODES x TRAIN_NGPUS_PER_NODE` topology that wrote it.

### Standalone, without PSRL

Use [`eval/`](eval/) when you want a number without the training stack: it serves
the model with vLLM, drives Harbor's own `terminus-2` harness, and aggregates by
category and family.

```bash
bash examples/sciaccel_rl/eval/run_eval.sh \
    --model ${PSRL_WORKSPACE}/models/Qwen3.5-4B \
    --dataset examples/sciaccel_rl/data/mitgcm-biogeo/repair_easy/eval/L1.parquet \
    --output-dir examples/sciaccel_rl/outputs/eval_biogeo_base \
    --skip-gpu-tasks
```

Two agents need no model at all and are the fastest way to test infrastructure:

| `--agent` | Does | Use for |
|---|---|---|
| `nop` | Builds the environment, edits nothing | Warming the image cache, and proving a node can build |
| `oracle` | Applies the known fix from `solution/` | Proving the verifier grades a correct patch |

`oracle` is the real end-to-end check, and it should score near 1.0. Anything
lower means the task or verifier is broken rather than the model. Budget time for
it: a single MITgcm task takes about 8 minutes to build and up to 15 more in the
verifier (`verifier_timeout_sec: 900`).

See [`eval/README.md`](eval/README.md) for filters, serving topology, and output
layout, and [`eval/FINDINGS.md`](eval/FINDINGS.md) for a measured Qwen3.5-9B
baseline.

---

## Gotchas worth knowing before you hit them

- **The verifier's 120 s reference-run cap is the top source of wasted episodes.**
  Under concurrency a 3 s simulation can overrun it, and the episode then carries
  no verifier score at all. Lower `harbor.max_concurrent_episodes` or raise
  `ROW_TIMEOUT_MAX` in the env's `factory/config.py`, which requires a recompile.
  Watch `termination/verifier_error/fraction`.
- **`/tmp` is node-local.** Put datasets on the shared filesystem before a
  cross-node run, or the remote node reads a stale copy and fails against paths
  that no longer exist.
- **Editing the `prompt` column does nothing.** Harbor re-reads `instruction.md`
  from disk. The hint reaches the model through `extra_info["hint"]`, which
  `runner.py` passes as Harbor `extra_instructions`. The `prompt` column is a
  record of the delivered text, not the delivery path.
- **`task_path` is absolute, baked at dataset-build time.** Moving or recompiling
  the task bank invalidates an existing parquet. Rebuild it.
- **`max_model_len` is the agent's budget, not slack.** It is forwarded to
  terminus-2 as `max_input_tokens`, so any headroom is headroom the agent will
  spend, and TITO then hands the trainer a response longer than
  `max_response_length`.
- **Dangling images accumulate over days** on fuse-overlayfs and eventually wedge
  the daemon. Prefer batched `docker rmi -f`, because `docker image prune -f`
  crawls once the daemon is already degraded.
