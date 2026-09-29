# Data Preparation for mini-SWE-agent RL Training

This directory contains everything needed to convert raw datasets into PSRL
training parquets and to warm up Docker image caches on cluster nodes.

Three independent data paths are supported:

- **Path A — Toy dataset**: 40–64 synthetic Python bug-fix tasks baked into
  a single `python:3.11-slim` image. Good for rapid iteration and smoke tests.
- **Path B — SWE-smith-py + SWE-bench Verified**: Real-world bugs from the
  SWE-smith-py collection (~51k SWE problems) for RL training, graded by running
  the actual test suite inside per-problem Docker images. SWE-bench Verified
  (500 SWE problems) is used for periodic validation.
- **Path C — SWE-Gym**: 2438 real-world bugs from the SWE-Gym dataset, graded
  by executing pre-computed eval scripts inside per-problem Docker images
  (`xingyaoww/sweb.eval.x86_64.*`). Supports a 100-problem subset for fast
  iteration and the full 2438-problem set for production training.

Harness mode (Claude Code / Codex) reuses any of these paths and adds two
host-side steps — the read-only runtime trees and an optional git-purged
per-image derivative. See
[Step 3: Harness runtime trees](#step-3-harness-runtime-trees--git-purged-derivatives-harness-mode-only);
the ready-to-run Claude Code launch script is
[`megatron_qwen_4b_swe_cc.sh`](../megatron_qwen_4b_swe_cc.sh) (SWE-Gym-293 data).

---

## Files

| File | Description |
|------|-------------|
| `prepare_simple_data.py` | Toy dataset generator — produces `train.parquet` / `test.parquet` from `simple_cases_*.json` |
| `simple_cases_train.json` | 40 synthetic training bug-fix tasks |
| `simple_cases_val.json` | 12 synthetic validation bug-fix tasks |
| `prepare_swebench.py` | HF → parquet converter for SWE-smith-py and SWE-bench Verified/Lite/Full |
| `prepare_swe_gym.py` | HF → parquet converter for SWE-Gym and SWE-Gym-Subset |
| `prepare_swe_gym_293.py` | HF → parquet converter for the SWE-Gym-293 (SkyRL-v0-293) dataset |
| `swebench_subsets.py` | Repo-balanced sampling helpers used by `prepare_swebench.py` and `prepare_swe_gym.py` |
| `docker_scripts/bake_simple_repos.sh` | Bakes toy repositories into a Docker image for Path A |
| `docker_scripts/build_harness_runtimes.sh` | Fetches the native Claude Code / Codex runtime trees (no Node) for harness mode |
| `docker_scripts/bake_harness_image.sh` | Per-image git-purged derivative (harness mode, optional) |
| `docker_scripts/rebake_harness_image.sh` | Force re-bake of existing derivatives after a bake-step change |
| `docker_scripts/prefetch_images.sh` | Pull per-SWE-problem images (skopeo-first, multi-mirror fallback, tar cache, `docker load`) |
| `docker_scripts/prefetch_example.sh` | Reference invocation that chains `prefetch_images.sh` + `load_all_nodes.sh` |
| `docker_scripts/swe_gym.sh` | Convenience wrapper: prefetch full SWE-Gym images (2438 problems) |
| `docker_scripts/swe_gym_293.sh` | Convenience wrapper: prefetch SWE-Gym-293 train + val images, then fan out |
| `docker_scripts/swe_gym_subset.sh` | Convenience wrapper: prefetch SWE-Gym-Subset images (100 problems) |
| `docker_scripts/swe_eval_subset.sh` | Convenience wrapper: prefetch SWE-bench eval subset images |
| `docker_scripts/swe_smith.sh` | Convenience wrapper: prefetch SWE-smith images |
| `docker_scripts/probe_mirrors.sh` | Quickly check which public Docker Hub mirrors can serve a given image (uses `skopeo inspect`, no download) |
| `docker_scripts/load_all_nodes.sh` | `pssh` fan-out: on every host listed in a file, `docker load` every `*.tar` in a shared-FS image dir, with per-node parallelism and skip-if-already-loaded |
| `docker_scripts/migrate_docker_overlay2.sh` | Move the Docker data-root / `overlay2` store onto a larger disk before loading a big image set |
| `_prefetch_logs/` | One log file per image (kept by `prefetch_images.sh`) — header `Already cached` when cached, or a full per-mirror/per-attempt log when pulled |
| `_load_logs/<timestamp>/` | `pssh` per-host stdout / stderr from `load_all_nodes.sh` |

---

## Path A — Toy dataset (quick iteration)

### Step 1: Bake repos into a Docker image

```bash
# From examples/mini_swe/
bash prepare/docker_scripts/bake_simple_repos.sh python:3.11-slim
```

This creates a local Docker image (tagged `psrl-mini-swe:latest` by default) with
all toy repositories pre-installed under `/<split>_<idx>/`. The script may need
proxy settings uncommented if your network requires them.

### Step 2: Generate parquet datasets

```bash
python prepare/prepare_simple_data.py \
    --mode simple \
    --train_size 64 \
    --test_size 16 \
    --output_dir examples/mini_swe/data/mini_swe_agent
```

Output: `data/mini_swe_agent/train.parquet` and `data/mini_swe_agent/test.parquet`.

Each row contains a synthetic problem statement, a reference patch, and
`sandbox_overrides.use_preexisting_repo = True` pointing to the pre-baked repo.

---

## Path B — SWE-smith-py + SWE-bench Verified (real RL)

### Prerequisites

Extra Python packages are required **at preparation time only**:

```bash
python -m pip install swebench==4.1.0 swesmith
```

Grading is host-independent: the eval script travels with the prepared row and
is parsed inside the grading sandbox by a vendored, stdlib-only driver. The
`prepare_*` scripts use `swebench`/`swesmith` to *freeze* that grading metadata,
but nothing on the training host imports them at rollout time. So these packages
are needed to prepare data, not to train on it.

If you bump the pinned `swebench`/`swesmith` versions, regenerate the vendored
parser registry so the frozen parser names stay resolvable:

```bash
python -m examples.mini_swe.grading.vendor_parsers
```

### Grading metadata

`prepare_swebench.py` records everything the grader needs under
`extra_info.swe_problem`:

| Dataset | `eval_script` | `log_parser` | What prepare does |
|---|---|---|---|
| `verified` / `lite` / `full` | shipped by the HF dataset | shipped by the HF dataset | copies them through unchanged |
| `smith` | **frozen here** | **frozen here** | builds the official-shaped eval script (`set -uxo pipefail` + `>>>>> Start/End Test Output`) from the repo profile and records the flattened parser name |

SWE-smith rows have neither field upstream, so the preparation step generates
them. Splits prepared before this change must be regenerated (or the eval script
backfilled) — otherwise `MiniSWEEnvironment.reset` fails fast with an explicit
"requires swe_problem.eval_script" error rather than silently grading on the
training host.

### Step 1: Generate SWE-smith-py training data

SWE-smith-py has 50,908 SWE problems, each with a pre-built per-problem Docker
image.  A repo-balanced 1,000-problem subset is a good starting point (roughly
10 bugs per repository, ~100 unique images to pull):

```bash
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith \
    --split train \
    --total 1000 \
    --per-repo-k 10 \
    --output-dir examples/mini_swe/data/swe_smith_py_1k
```

For a larger scale (more diversity, more disk space):

```bash
# 5 000 SWE problems, ~20 per repo
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith --split train \
    --total 5000 --per-repo-k 20 \
    --output-dir examples/mini_swe/data/swe_smith_py_5k
```

The script always applies repo-balanced round-robin sampling so no single
repository dominates. Pass `--no-repo-balanced` to disable and simply truncate
to `--total` instead.

### Step 2: Generate SWE-bench Verified validation data

Training uses a small repo-balanced subset for fast `test_freq` evaluation.
The full 500-problem set is used for leaderboard-quality final evaluation.

```bash
# 80-problem subset — used as test_files during training
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset verified \
    --split test \
    --total 80 \
    --repo-balanced \
    --output-dir examples/mini_swe/data/verified_subset_80 \
    --output-filename train.parquet

# Full 500 — for standalone eval or leaderboard submission
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset verified \
    --split test \
    --output-dir examples/mini_swe/data/swe_bench_verified
```

Other available datasets (`--dataset` values):

| Key | HuggingFace path | Split | Notes |
|-----|-----------------|-------|-------|
| `smith` | `SWE-bench/SWE-smith-py` | `train` | 50,908 Python bugs, pre-built images |
| `verified` | `SWE-bench/SWE-bench_Verified` | `test` | 500 human-verified issues |
| `lite` | `SWE-bench/SWE-bench_Lite` | `test` | 300 issues, common benchmark |
| `full` | `SWE-bench/SWE-bench` | `test` | 2,294 issues |

#### Subsampling and filtering

`prepare_swebench.py` (used in Step 1 and Step 2 above) also accepts two independent
filter/sample mechanisms:

- `--subset-spec`: Applied first. Accepts a Python slice (`"0:500"`) or a regex
  matched against each SWE problem's `instance_id` field (e.g. `"^django"`).
- `--total` + `--repo-balanced`: Applied after `--subset-spec`. Round-robins
  across repositories alphabetically until `--total` SWE problems are collected.
  `--per-repo-k` adds a hard cap per repository before round-robin.

Example — first 200 Django SWE problems only:

```bash
python -m examples.mini_swe.prepare.prepare_swebench \
    --dataset smith --split train \
    --subset-spec "^django" \
    --total 200 \
    --output-dir examples/mini_swe/data/django_200
```

### Step 3: Pre-fetch Docker images (shared-FS cache)

Each SWE-smith-py SWE problem uses a dedicated Docker image (~2–5 GB per
image, 3 GB median). Images must be reachable from **every cluster node** that
runs rollout workers. The recommended flow is:

1. Pull every unique image **once** to a shared-FS directory as a
   `docker-archive` tar (`prefetch_images.sh`, this step).
2. Fan `docker load` out to every node over pssh (`load_all_nodes.sh`,
   [Step 4](#step-4-fan-out-to-all-cluster-nodes)).

That way Docker Hub is hit once per image, not `num_nodes` times.

#### 3.1 — Why not just `docker pull`?

On many clusters the docker daemon itself cannot reach `registry-1.docker.io`
(user-space proxy env vars like `http_proxy` are **not** inherited by
`dockerd`). `prefetch_images.sh` sidesteps this by using **skopeo** — a
user-space tool that honours `$http_proxy` / `$https_proxy` from
`${PSRL_WORKSPACE}/env/psrl.sh` and writes straight to a local `docker-archive`
tar (or directly into `dockerd` via the Unix socket). No daemon proxy config
needed.

#### 3.2 — Dry-run

```bash
source ${PSRL_WORKSPACE}/env/psrl.sh   # sets http_proxy for skopeo
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
    --dry-run
```

Prints every unique `extra_info.sandbox_overrides.environment.image` referenced
by the parquet.

#### 3.3 — Full pull

```bash
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_smith_py_1k/train.parquet \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe \
    --workers 4 \
    --retries 5 \
    --mirrors docker.xuanyuan.me,docker.1ms.run,docker.1panel.live,hub.rat.dev,dockerproxy.net,proxy.vvvv.ee,docker.xiaogenban1993.com,lispy.org,registry.cyou \
    --no-direct-fallback
```

A reference invocation that chains the full pull + `docker load` fan-out lives
at `docker_scripts/prefetch_example.sh`.

What each flag does:

| Flag | Effect |
|------|--------|
| `--image-dir DIR` | Save each image as `docker-archive:<DIR>/<image>.tar`. Cached tars are reused on re-run. |
| `--load` | (optional) After each successful pull, also `docker load -i` into the local `dockerd`. Safe to skip if you plan to load later via [Step 4](#step-4-fan-out-to-all-cluster-nodes). |
| `--workers N` | Parallel pulls. `4` is sane; heavy networks can push to `8–16`. |
| `--retries N` | Per `(image, mirror)` retry count on transient failures (`unexpected EOF`, blob `404` from a mirror's cold cache). |
| `--mirrors a,b,c` | Ordered fallback list. A pull failing on `a` transparently retries on `b`, etc. Mirrors are applied via `apply_dockerhub_mirror` (same logic as `scripts/docker/docker_install.sh`). |
| `--no-direct-fallback` | Don't try `docker.io` directly after all mirrors fail — useful on clusters where `registry-1.docker.io` is firewalled. |
| `--force` | Ignore cache (even if the tar is structurally complete) and re-pull. |
| `--images FILE` / `--only A,B` | Alternatives to `--parquet` — feed a hand-written image list, one image per line (`#` comments OK). Useful for re-running a curated subset. |
| `--log-dir DIR` | Default is `<prepare>/_prefetch_logs/`, one `.log` per image. |

#### 3.4 — Integrity, idempotency, and the log directory

- Every tar is validated with `tar -tf | grep manifest.json` on every run
  (the `verify_docker_archive` helper, ~40 ms per 3 GB tar). Truncated/EOF'd
  tars are treated as **missing** and re-pulled automatically.
- Failed/aborted runs **do not** leave zombie tars: partial tars are
  `rm -f`'d before every retry, after every failed mirror, on SIGINT/SIGTERM
  (via `trap`), and again in the "all mirrors failed" branch.
- Every image gets a log file in `_prefetch_logs/`:
  - **Cached**: header `Already cached <image>` + verification timestamp.
  - **Pulled**: per-mirror, per-attempt output (`----- attempt N/M -----`).
  - **Failed**: final attempt's fatal error + note pointing at `.log` /
    `.log.load`.

#### 3.5 — Mirror hygiene

Public Docker Hub mirrors come and go almost monthly. The list above is the
one that actually works as of 2026-04. Before trusting a new mirror,
probe it first:

```bash
bash examples/mini_swe/prepare/docker_scripts/probe_mirrors.sh \
    swebench/swesmith.x86_64.paramiko_1776_paramiko.23f92003
```

Output:

```
MIRROR                            RESULT    MESSAGE
docker.xuanyuan.me                OK        manifest reachable
docker.1ms.run                    OK        manifest reachable
docker.1panel.live                FAIL      manifest unknown: ...
hub.rat.dev                       TIMEOUT   > 25s
...
```

`probe_mirrors.sh` only does `skopeo inspect` (manifest-level probe, ~seconds
per mirror) — no blobs are downloaded. Edit the `MIRRORS=(...)` array inside
the script to add/remove candidates.

Known-dead / do-not-use mirrors (as of 2026-04):

- `dockerpull.org` — GFW-blocked since 2025-12.
- `docker.hlmirror.com` — now paywalls pulls behind `mirror.houlang.cloud`.
- `docker.m.daocloud.io` — allow-list only; `swebench/*` is **not** in it.
- `docker.imgdb.de`, `hub.docker.io`, `aicarbon.xyz`, …  — abandoned.

#### 3.6 — Retrying only the images that failed

After a run, failing images are easy to list from the log dir:

```bash
cd examples/mini_swe/prepare/_prefetch_logs
for f in *.log; do
    tail -n 1 "$f" | grep -q FATA && \
        echo "${f%.log}" | sed 's|__|/|; s|__|:|'
done > /tmp/failed_images.txt
```

Feed the list back into `prefetch_images.sh` via `--images` (same flags as the
full pull, just swap `--parquet` for `--images`):

```bash
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --images /tmp/failed_images.txt \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe \
    --workers 8 \
    --retries 10 \
    --mirrors docker.xuanyuan.me,docker.1ms.run,docker.1panel.live,hub.rat.dev,dockerproxy.net,proxy.vvvv.ee,docker.xiaogenban1993.com,lispy.org,registry.cyou \
    --no-direct-fallback
```

Already-cached tars are skipped automatically via `verify_docker_archive`, so
retrying is cheap — only the genuinely missing/truncated ones get re-pulled.

> **Disk budget**: The 1k smith subset uses ~131 unique images, total
> ~330–500 GB of `docker-archive` tars on the shared FS. The 5k subset roughly
> doubles that. Plan `${PSRL_WORKSPACE}/docker_images/swe` capacity accordingly,
> and on each node reserve ~1–2× that again for `/var/lib/docker` after
> `docker load`.

---

### Step 4: Fan out to all cluster nodes

Once every image has a complete tar in `${PSRL_WORKSPACE}/docker_images/swe/`, use
`load_all_nodes.sh` to `docker load` them on every host **in parallel** over
`pssh`.

```bash
bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts     ${PSRL_WORKSPACE}/hosts/32GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe
```

Key defaults and flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--hosts FILE` | required | One IP (or IP:port) per line. `#` comments and blank lines ignored. |
| `--image-dir DIR` | required | Directory containing `*.tar` files (created by Step 3). |
| `--images-list FILE` | — | Roll out a subset. Accepts either image refs (`swebench/xxx:latest`) or tar basenames (`swebench__xxx`). |
| `--parallel-per-node N` | `2` | `xargs -P` concurrency **on each node**. `docker load` is I/O bound; 2–4 is the sweet spot. |
| `--skip-existing` / `--force` | skip | Before loading, `tar -xOf <tar> manifest.json` extracts the `RepoTags[0]`; if `docker image inspect <tag>` finds the image already present, the tar is skipped on that node. `--force` disables skipping. |
| `--timeout S` | `7200` | `pssh -t`. |
| `--user USER` | — | `pssh -l USER`. Uses your default ssh config if unset. |
| `--outdir DIR` | `_load_logs/<ts>/` | Per-host stdout / stderr collection directory. |
| `--dry-run` | off | Prints the planned command and the first 10 hosts/tars, does nothing. |

After the run, the script prints a per-host summary:

```
--- summary ---
  192.168.1.1                loaded=131  skipped=0    failed=0
  192.168.1.2                loaded=0    skipped=131  failed=0   # already had them
  192.168.1.3                loaded=130  skipped=0    failed=1
  192.168.1.4                loaded=131  skipped=0    failed=0
```

Full per-host output is in `_load_logs/<timestamp>/stdout/<ip>` and
`_load_logs/<timestamp>/stderr/<ip>`.

#### 4.1 — Rolling out only a new subset

After adding more training data (e.g. you went from 1k → 5k), compute the
delta and feed it to `--images-list`:

```bash
# Build a subset file with only the NEW images:
diff <(ls ${PSRL_WORKSPACE}/docker_images/swe/*.tar | xargs -n1 basename -s .tar | sort) \
     <(previous_deployed_list.txt) \
    | grep '^<' | sed 's/^< //' > /tmp/new_images.txt

bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts ${PSRL_WORKSPACE}/hosts/128GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe \
    --images-list /tmp/new_images.txt
```

Old images already on every node are untouched (thanks to
`--skip-existing`).

#### 4.2 — Operational notes

- **I/O planning**: 4 nodes × 2 concurrent loads × ~500 MB/s on the shared FS
  read side is already ~4 GB/s of NFS/Ceph read traffic. Tune
  `--parallel-per-node` down if the FS saturates.
- **Disk on each node**: `/var/lib/docker` needs room for *every* image
  you plan to use during training, not just the currently running ones. Run
  `docker system df` on one node post-load to sanity-check.
- **Skip logic uses tags, not digests**: If a remote mirror changed what
  `swebench/foo:latest` points to, `--skip-existing` will still skip. Use
  `--force` if you specifically need to refresh.
- **Partial failures**: If one node reports `failed=K`, you can re-run with
  `--hosts` pointing to just that node — `--skip-existing` makes the retry
  cheap:
  ```bash
  echo node-c > /tmp/one_host
  bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
      --hosts /tmp/one_host --image-dir ${PSRL_WORKSPACE}/docker_images/swe \
      --parallel-per-node 4
  ```

Running `docker_scripts/prefetch_images.sh` (Step 3) + `docker_scripts/load_all_nodes.sh` (Step 4) on a
new cluster is the complete image-warmup path. Containers that can't pull
their image at rollout time produce a zero-reward episode and waste the
rollout slot, so validate once with `docker run --rm <a-sample-image> true`
on every host before kicking off training.

#### 4.3 — Scaling beyond full replication

PSRL intentionally keeps one node-local Docker daemon per Ray node. Do not
point all workers at one remote daemon: every command and file transfer would
cross the network, and that daemon would become a scheduling and failure
bottleneck.

Full image replication is appropriate for small clusters or when any task can
land on any node. For larger clusters, reduce disk and warm-up cost in this
order:

1. generate an exact image manifest for the planned dataset shard and pass it
   through `--images-list`;
2. serve images from a cluster-local pull-through registry cache and pin tags
   to digests;
3. expose image-presence as a Ray node resource/label and route a task only to
   a warm node;
4. evaluate P2P distribution or a lazy-pull containerd snapshotter only after
   metrics show image transfer dominates rollout time.

Kubernetes is useful when the organization already needs multi-tenant policy
and cluster admission, but creating a Pod per agent turn/episode adds more
control-plane latency and memory than PSRL's node-local persistent Engine API.
It is not a performance upgrade by itself.

---

## Path C — SWE-Gym (real RL, pre-computed eval scripts)

SWE-Gym provides 2438 real-world GitHub issues with Docker images following the
`xingyaoww/sweb.eval.x86_64.*` naming convention. Unlike SWE-smith (which
removes F2P test files on HEAD), SWE-Gym images have a standard repository
layout — no `git checkout HEAD~1` is needed. Grading uses a pre-computed
`eval_script` embedded directly in the parquet.

Three variants are supported: the full 2438-instance set, the 100-instance
subset (fast iteration), and [SWE-Gym-293](#swe-gym-293-skyrl-v0-293) — a
curated SWE-bench Verified subset with a train/val split.

### Prerequisites

```bash
# Same grading deps as Path B
python -m pip install swebench==4.1.0

# Needed for the full SWE-Gym dataset (2438) and for SWE-Gym-293.
# The 100-instance subset ships with pre-computed eval_scripts and does NOT
# require the fork. For SWE-Gym-293 the fork is installed into an isolated venv
# by --ensure-fork, so this global install is not required there.
pip install git+https://github.com/SWE-Gym/SWE-Bench-Fork.git

# After generating parquets, restore swebench 4.1.0:
python -m pip install swebench==4.1.0
```

### Dataset variants (`prepare_swe_gym.py --dataset`)

| Key | HuggingFace path | Instances | `eval_script` source | Notes |
|-----|-----------------|-----------|---------------------|-------|
| `gym` | `SWE-Gym/SWE-Gym` | 2438 | Generated via `make_test_spec` (needs SWE-Bench-Fork 2.0.13) | Full training set |
| `gym-subset` | `SumanthRH/SWE-Gym-Subset` | 100 | Pre-computed in HF dataset column | Quick iteration, no Fork needed |

SWE-Gym-293 / SkyRL-v0-293 is **not** a `--dataset` key here; it has its own
converter, [`prepare_swe_gym_293.py`](#swe-gym-293-skyrl-v0-293).

### Step 1: Generate SWE-Gym training data

```bash
# SWE-Gym-Subset (100 instances) — quick start, no Fork dependency
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym-subset \
    --output-dir examples/mini_swe/data/swe_gym_subset_100

# Full SWE-Gym (2438 instances) — requires SWE-Bench-Fork 2.0.13
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym \
    --output-dir examples/mini_swe/data/swe_gym_2438

# Repo-balanced 500-instance subset for smaller experiments
python -m examples.mini_swe.prepare.prepare_swe_gym \
    --dataset gym \
    --total 500 \
    --repo-balanced \
    --output-dir examples/mini_swe/data/swe_gym_500
```

The script automatically skips instances for which `eval_script` cannot be
resolved (prints a count of skipped instances at the end). If many instances
are skipped, verify that SWE-Bench-Fork 2.0.13 is correctly installed.

#### SWE-Gym-293 (SkyRL-v0-293)

A **curated SWE-bench Verified subset** that a Qwen3.5-4B model already solves
part of — which is what produces a non-zero reward signal under GRPO, unlike the
unfiltered `swe_smith_py_1k` slice. `prepare_swe_gym_293.py` downloads the
parquet directly from HuggingFace and generates each instance's `eval_script`
with the SWE-Bench-Fork inside an isolated venv (`--fork-venv`), so the main
environment's swebench 4.1.0 is never disturbed. Re-run with `--force-fork` to
rebuild the venv from scratch.

```bash
python -m examples.mini_swe.prepare.prepare_swe_gym_293 \
    --output-dir examples/mini_swe/data/swe_gym_293 \
    --ensure-fork --fork-venv /tmp/swegym-fork-venv
```

Output: `data/swe_gym_293/train.parquet` (293 rows) and `val.parquet` (23 rows).

These are the parquets used by the ready-to-run Claude Code harness script
[`megatron_qwen_4b_swe_cc.sh`](../megatron_qwen_4b_swe_cc.sh). It pins
`default_agent_loop=mini_swe_claude_code` at launch, so the default
`agent_name` tag is fine; for other harness recipes add
`--agent-name mini_swe_claude_code` (or `mini_swe_codex`) so the rows select the
harness themselves.

### Step 2: Pre-fetch Docker images

SWE-Gym images follow the naming convention:
```
xingyaoww/sweb.eval.x86_64.{instance_id.replace("__", "_s_").lower()}:latest
```

Use the convenience wrappers or invoke `prefetch_images.sh` directly:

```bash
# Convenience wrapper (full dataset)
source ${PSRL_WORKSPACE}/env/psrl.sh
bash examples/mini_swe/prepare/docker_scripts/swe_gym.sh

# Convenience wrapper (subset)
bash examples/mini_swe/prepare/docker_scripts/swe_gym_subset.sh

# Manual invocation (equivalent)
bash examples/mini_swe/prepare/docker_scripts/prefetch_images.sh \
    --parquet examples/mini_swe/data/swe_gym_2438/train.parquet \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe_gym \
    --workers 4 \
    --retries 5 \
    --mirrors docker.xuanyuan.me,docker.1ms.run,docker.1panel.live,hub.rat.dev
```

Then fan out to all cluster nodes:

```bash
bash examples/mini_swe/prepare/docker_scripts/load_all_nodes.sh \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --image-dir ${PSRL_WORKSPACE}/docker_images/swe_gym
```

#### Pre-fetching SWE-Gym-293

One wrapper handles both splits and the fan-out:

```bash
bash examples/mini_swe/prepare/docker_scripts/swe_gym_293.sh
```

It prefetches `train.parquet` (293 × `xingyaoww/sweb.eval.x86_64.*`) **and**
`val.parquet` (23 × legacy `swebench/sweb.eval.x86_64.*` images), then runs
`docker load` on the nodes listed in `${PSRL_WORKSPACE}/hosts/16GPUs` when that
file exists (otherwise it stops after the tar cache and tells you to run
`load_all_nodes.sh` manually).

> **Note**: the validation split (23 instances: sqlfluff/marshmallow/pvlib/
> astroid/pyvista/pydicom) uses a *disjoint* set of `swebench/`-namespace images
> that never appear in `train.parquet`. Prefetching the training split alone
> makes validation fail at sandbox creation with
> `Docker Engine returned HTTP 404: No such image`.

Env overrides: `SWE_GYM_293_TRAIN` / `SWE_GYM_293_VAL` (defaults under
`data/swe_gym_293/`), `SWE_GYM_293_IMAGE_DIR` (shared tar cache) and
`SWE_PREFETCH_WORKERS` (skopeo parallelism, default 64).

### Step 3: Harness runtime trees + git-purged derivatives (harness mode only)

This step applies to **any** data path (A/B/C), not just SWE-Gym — skip it unless
you plan to run a harness. The native mini-SWE-agent loop runs on the host and
needs no image changes. The **harness** loops (Claude Code / Codex) run a
self-contained native CLI inside the sandbox, so prepare two things:

1. **Runtime trees (required, once on the shared filesystem).** Each sandbox
   mounts one tree read-only; there is no Node, no npm and no in-sandbox install.

   ```bash
   export PSRL_HARNESS_RUNTIME_ROOT=/shared/psrl/harness-runtimes
   bash examples/mini_swe/prepare/docker_scripts/build_harness_runtimes.sh
   ```

2. **Git-purged derivative (optional, per worker host).** Once the base images
   are present on a node, bake one derivative per unique image:

   ```bash
   bash examples/mini_swe/prepare/docker_scripts/bake_harness_image.sh \
       --parquet examples/mini_swe/data/swe_gym_293/train.parquet
   ```

The derivative only **purges leaked git metadata** (remotes, refs, reflog,
unreachable objects) so an agent cannot read a future fix commit from the image;
the harness executable is never baked. Run it on every worker host that creates
sandboxes (Docker images are node-local), or distribute the derivative with
`docker save`/`docker load`. Without a bake, training still works — a runtime git
probe purges only images that actually leak — but you pay that cost on every
rollout.

The tag is keyed on the base image alone, so an existing derivative is skipped.
After changing the bake steps, re-bake with `rebake_harness_image.sh` (same
arguments) to delete and re-create it:

```bash
bash examples/mini_swe/prepare/docker_scripts/rebake_harness_image.sh \
    --parquet examples/mini_swe/data/swe_gym_293/train.parquet
```

#### Launching the harness recipe

Once the parquets and images are ready, `examples/mini_swe/megatron_qwen_4b_swe_cc.sh`
is the ready-to-run Claude Code script (Megatron, Qwen3.5-4B, SWE-Gym-293). It
exports `PSRL_HARNESS_RUNTIME_ROOT` and sets
`psrl.rollout_gateway.trajectory_id_strategy=auto`, so the two host-side steps
above are all that remain:

```bash
bash examples/mini_swe/megatron_qwen_4b_swe_cc.sh
```

See the main README's
[Harness training: preprocessing & bake](../README.md#harness-training-preprocessing--bake)
for the full checklist, re-bake and runtime-fallback details, or the
condensed docs page
[docs → SWE Data Preparation](https://psrl.readthedocs.io/en/latest/examples/agentic_rl/swe/prepare.html).

> **Disk budget**: The full 2438-instance dataset uses ~200 unique images,
> totalling ~500–800 GB of `docker-archive` tars. The 100-instance subset
> uses ~80 unique images (~200 GB). Plan shared-FS and `/var/lib/docker`
> capacity accordingly.

---

## Parquet schema

Each output row produced by `prepare_swebench.py` or `prepare_swe_gym.py` contains:

| Field | Type | Description |
|-------|------|-------------|
| `prompt` | `list[dict]` | Single `[{"role": "user", "content": problem_statement}]`. Agent templates are applied at runtime. |
| `data_source` | `str` | `"swe_smith_py"`, `"swebench_verified"`, or `"swe_gym"`. Determines which reward branch fires in `reward.py`. |
| `ability` | `str` | Always `"software_engineering"`. |
| `reward_model.style` | `str` | `"swebench_test_exec"`. Signals test-execution reward path. |
| `reward_model.ground_truth.instance_id` | `str` | HuggingFace `instance_id` (e.g. `django__django-11039`). Consumed by PSRL's reward computation and the standalone evaluation CLI. |
| `reward_model.ground_truth.FAIL_TO_PASS` | `list[str]` | Tests that must go from failing to passing. |
| `reward_model.ground_truth.PASS_TO_PASS` | `list[str]` | Tests that must continue passing. |
| `reward_model.ground_truth.gold_patch` | `str` | Reference patch (for offline analysis only; not used in RL reward). |
| `extra_info.swe_problem_id` | `str` | The SWE problem's HuggingFace `instance_id`, used for logging and grader correlation. |
| `extra_info.swe_problem` | `dict` | Full HuggingFace dataset row for this SWE problem, including the frozen grading fields: `eval_script` (executed in the grader sandbox) and `log_parser` (a key into the vendored parser registry). Verified/Gym rows inherit both from the HF dataset; SWE-smith rows get them frozen at prepare time. |
| `extra_info.swe_problem_image` | `str` | Docker image name for this SWE problem. |
| `extra_info.swe_restore_tests` | `bool` | `True` for SWE-smith-py (must run `git checkout HEAD~1` to restore F2P test files). `False` for Verified and SWE-Gym. |
| `extra_info.swe_grader` | `str` | `"swebench_fresh_container"`. Activates post-rollout fresh-container grading in the agent loop. |
| `extra_info.sandbox_overrides.environment.image` | `str` | Per-SWE-problem image injected into `MiniEnvironmentConfig` at rollout time. |
| `extra_info.sandbox_overrides.environment.cwd` | `str` | Always `"/testbed"` for real SWE problems. |
| `agent_name` | `str` | `"mini_swe_agent"`. Selects `MiniSWEAgentLoopV1` in the agent loop registry. |

### SWE-Gym-specific fields

The following fields distinguish SWE-Gym rows from SWE-smith / Verified rows:

| Field | SWE-smith | SWE-Gym | Effect |
|-------|-----------|---------|--------|
| `data_source` | `"swe_smith_py"` | `"swe_gym"` | Routes to the same reward path (`_compute_swe_reward`) |
| `extra_info.swe_restore_tests` | `True` | `False` | SWE-Gym repos have standard layout; no HEAD~1 restore needed |
| `extra_info.swe_problem.eval_script` | absent | pre-computed bash script | SWE-Gym embeds the full eval script; SWE-smith generates it at grading time via `swesmith.profiles` |
| `extra_info.swe_problem_image` | `swebench/swesmith.x86_64.*` | `xingyaoww/sweb.eval.x86_64.*` | Different Docker image registries and naming conventions |

The `grader_kind` is automatically determined at runtime from these fields:
- `swe_restore_tests=True` → `"smith"` (uses `swesmith.harness.grading`)
- `swe_problem.eval_script` present → `"gym"` (uses `parse_log_pytest` directly)
- Neither → `"verified"` (uses `swebench.harness.grading`)
