# SWE Data Preparation

Each training path requires converting raw datasets into PSRL training parquets and warming up Docker image caches on all cluster nodes.

Harness mode (Claude Code / Codex) additionally needs two host-side preparations — the read-only runtime trees and an optional per-image derivative. See [Harness Mode Preparation](#harness-mode-preparation).

---

## Data Paths Summary

| Path | Dataset | Prepare script | Docker images | Disk budget |
|------|---------|----------------|---------------|-------------|
| **Toy** | `simple_cases_train.json` (40 tasks) | `prepare_simple_data.py` | `python:3.11-slim` (baked as `psrl-mini-swe:latest`) | Minimal |
| **SWE-smith** | [SWE-bench/SWE-smith-py](https://huggingface.co/datasets/SWE-bench/SWE-smith-py) (50k) | `prepare_swebench.py` | `swebench/swesmith.x86_64.*` (~3 GB each) | 500 GB–1 TB shared FS |
| **SWE-Gym** | [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) (2438) | `prepare_swe_gym.py` | `xingyaoww/sweb.eval.x86_64.*` | 500–800 GB shared FS |
| **SWE-Gym-293** | [NovaSky-AI/SkyRL-v0-293-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data) (293 + 23) | `prepare_swe_gym_293.py` | `xingyaoww/*` (train) + `swebench/*` (val) | Same per-image cost as SWE-Gym |

---

## Path A: Toy Dataset

Quick iteration and smoke tests. All bugs are synthetic and baked into a single Docker image.

```bash
# 1. Bake repos into Docker image
bash examples/mini_swe/prepare/docker_scripts/bake_simple_repos.sh python:3.11-slim

# 2. Generate parquets
python examples/mini_swe/prepare/prepare_simple_data.py \
    --mode simple --train_size 64 --test_size 16 \
    --output_dir examples/mini_swe/data/mini_swe_agent
```

---

## Path B: SWE-smith-py

Real-world GitHub bugs with per-problem Docker images. Supports repo-balanced subsampling.

```bash
# 1. Generate training parquet (1k subset)
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith --split train \
    --total 1000 --per-repo-k 10 \
    --output-dir examples/mini_swe/data/swe_smith_py_1k

# 2. Generate validation parquet
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset verified --split test \
    --total 80 --repo-balanced \
    --output-dir examples/mini_swe/data/verified_subset_80 \
    --output-filename train.parquet

# 3. Pre-fetch Docker images to shared FS
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe --workers 4

# 4. Fan out to all cluster nodes
bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe
```

---

## Path C: SWE-Gym

Real-world bugs with pre-computed eval scripts. No `git checkout HEAD~1` is needed.

| Variant | HuggingFace dataset | Instances | `eval_script` |
|---------|---------------------|-----------|---------------|
| `gym-subset` | `SumanthRH/SWE-Gym-Subset` | 100 | Shipped in the dataset |
| `gym` | `SWE-Gym/SWE-Gym` | 2438 | Generated with SWE-Bench-Fork 2.0.13 |
| SWE-Gym-293 | `NovaSky-AI/SkyRL-v0-293-data` | 293 + 23 | Generated with SWE-Bench-Fork (own script, not a `--dataset` key) |

### Step 1: Generate parquets

```bash
# Quick start (100 instances, no Fork dependency)
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym-subset \
    --output-dir examples/mini_swe/data/swe_gym_subset_100

# Full dataset (2438 instances, requires SWE-Bench-Fork 2.0.13)
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym \
    --output-dir examples/mini_swe/data/swe_gym_2438
```

#### SWE-Gym-293 (SkyRL-v0-293)

A curated SWE-bench Verified subset that a Qwen3.5-4B model already solves part of, so GRPO gets a non-zero reward signal. `prepare_swe_gym_293.py` downloads the parquet directly from HuggingFace and generates each instance's `eval_script` inside an isolated SWE-Bench-Fork venv, leaving the main environment's `swebench` untouched.

```bash
python -m examples.mini_swe.prepare.prepare_swe_gym_293 \
    --output-dir examples/mini_swe/data/swe_gym_293 \
    --ensure-fork --fork-venv /tmp/swegym-fork-venv
```

Output: `data/swe_gym_293/train.parquet` (293 rows) and `val.parquet` (23 rows).

### Step 2: Pre-fetch images

```bash
# Convenience wrappers
bash examples/mini_swe/prepare/docker_scripts/swe_gym.sh          # full 2438
bash examples/mini_swe/prepare/docker_scripts/swe_gym_subset.sh   # 100-instance subset
```

#### SWE-Gym-293 images

`swe_gym_293.sh` prefetches **both** splits and fans them out to the worker nodes:

```bash
bash examples/mini_swe/prepare/docker_scripts/swe_gym_293.sh
```

:::{warning}
The validation split (23 instances) uses a **disjoint** set of legacy `swebench/*` images that never appear in `train.parquet`. Prefetching the training split alone makes validation fail at sandbox creation with `Docker Engine returned HTTP 404: No such image`.
:::

Env overrides: `SWE_GYM_293_TRAIN` / `SWE_GYM_293_VAL` (defaults under `data/swe_gym_293/`), `SWE_GYM_293_IMAGE_DIR` (shared tar cache) and `SWE_PREFETCH_WORKERS` (skopeo parallelism, default 64). The fan-out step uses `${PSRL_WORKSPACE}/hosts/16GPUs` when that file exists.

---

## Harness Mode Preparation

Only needed when a harness loop is selected — either the parquet's `agent_name` is `mini_swe_claude_code` / `mini_swe_codex`, or the launch script pins `default_agent_loop` (as `megatron_qwen_4b_swe_cc.sh` does). Both steps are host-side; Docker images stay node-local.

### 1. Build the harness runtime trees

The harnesses ship as self-contained native binaries — no Node, no npm, nothing installed inside the sandbox. Each sandbox bind-mounts one tree read-only at the harness config's `runtime_mount` (default `/opt/harness`).

```bash
export PSRL_HARNESS_RUNTIME_ROOT=/shared/psrl/harness-runtimes
bash examples/mini_swe/prepare/docker_scripts/build_harness_runtimes.sh
# -> <root>/claude_code/bin/claude, <root>/codex/bin/codex
```

Build once on the shared filesystem. The script is idempotent, verifies the pinned Claude Code sha256, and skips trees whose executable already answers `--version`. Point `NPM_REGISTRY` at a mirror when the host cannot reach npmjs.org directly.

### 2. Bake the git-purged derivative (optional)

Each task has its own base image, so `bake_harness_image.sh` derives one per-image derivative — `psrl/swebench-harness:<sha12(base)>` — that purges leaked git metadata (remotes, refs, reflog, unreachable objects) so an agent can never read a future fix commit. The harness executable is **not** baked; it is mounted read-only in step 1.

```bash
# every unique image in a parquet, on each worker host that creates sandboxes
bash examples/mini_swe/prepare/docker_scripts/bake_harness_image.sh \
    --parquet examples/mini_swe/data/swe_gym_293/train.parquet
```

The tag keys on the base image alone, so an existing derivative is skipped. After changing the bake steps, use `rebake_harness_image.sh` (same arguments) to delete the stale derivative and re-bake it.

A missing bake never blocks training: the runner falls back to the base image plus a runtime git probe that purges only images that actually leak — you just pay that cost on every rollout.

### 3. Launch the harness recipe

With the parquets and images ready, `examples/mini_swe/megatron_qwen_4b_swe_cc.sh` is the ready-to-run Claude Code recipe (Megatron, Qwen3.5-4B, SWE-Gym-293). It exports `PSRL_HARNESS_RUNTIME_ROOT` and sets `psrl.rollout_gateway.trajectory_id_strategy=auto` itself, so only steps 1–2 above are host-side prerequisites.

---

## Docker Image Workflow

The two-step workflow avoids hitting Docker Hub `N × nodes` times:

1. **Prefetch** (once): `prefetch_images.sh` pulls each unique image via `skopeo` to a shared-FS tar.
2. **Fan-out** (per cluster): `load_all_nodes.sh` does `docker load` on every node via `pssh`.

Both steps are idempotent — already-cached tars and already-loaded images are skipped automatically.

---

```{seealso}
Full instructions including mirror configuration, retry strategies, disk planning, and the harness re-bake / runtime-fallback details:
[`examples/mini_swe/prepare/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/mini_swe/prepare/README.md)
and
[`examples/mini_swe/README.md`](https://github.com/psrl-project/psrl/blob/main/examples/mini_swe/README.md#harness-training-preprocessing--bake).
```
