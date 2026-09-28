# SciBuddy Rollout

Batch-rollout collection for SciBuddy-family Harbor research tasks. No training,
no reward, no evaluation: this records trajectories for later analysis.

Two task families are supported:

| Family | Description | Verifier contract |
|---|---|---|
| `oxford-eis` | Battery EIS analysis from Oxford schedule-history data | `results.json` schema and numeric check |
| `ukb-cystatin` | UKB renal endpoint prediction with cystatin-C | `manifest.json` plus `interpretation.json` |
| `synthetic-tempstats` | Synthetic series summary, included to smoke-test the wiring | recomputed against the bound input |

Both tasks are CPU-only and may run for hours. Set `max_turns` accordingly.

## Quick start

### 1. Build the dataset

Extract Harbor packages and build a parquet:

```bash
python examples/scibuddy_rollout/prepare/build_dataset.py \
    --packages /path/to/job-*-package.tar.gz \
    --output examples/scibuddy_rollout/data/scibuddy.parquet
```

This extracts each `.tar.gz` to `data/tasks/<hash>/` alongside the parquet.
The `task_path` in `extra_info` points to the extracted `package/` directory,
which is what Harbor reads at runtime.

### 2. Run

```bash
bash examples/scibuddy_rollout/batch_rollout_smg.sh
```

Or directly with overrides:

```bash
model_path=/path/to/model num_workers=2 \
    bash examples/scibuddy_rollout/batch_rollout_smg.sh
```

### 3. Inspect the trajectories

Results land under `outputs/scibuddy_rollout_<date>/`:

```
rollout.jsonl         # one record per episode
summary.json          # aggregate: record count, termination histogram
transcripts/          # full per-episode message history
logs/trajectories/    # human-readable turn-by-turn text dumps
```

Score the dump offline:

```python
import json
from examples.scibuddy_rollout.reward import compute_score

for line in open("outputs/.../rollout.jsonl"):
    rec = json.loads(line)
    info = {**(rec.get("extra_info") or {}), **(rec.get("extra_fields") or {})}
    score = compute_score("scibuddy_rollout", "", None, info)["score"]
    print(rec["uid"], rec["terminate_reason"], score)
```

## Requirements

- **Base images must be pre-pulled** on every Harbor node before running.
  Docker pulls at episode time will block the Harbor loop and time out.
  See [Data and images](#data-and-images) below.

- **`bindings.json` is metadata, not a Harbor input.** Harbor mounts data
  through Compose and never reads that file, so listing a source there does not
  make it appear at `/data`. The prepare step records it in
  `extra_info.bindings` for inspection only. Bake the input into the
  environment image, or pass Compose mounts explicitly.

## Data and images

The `bindings.json` in each package names the source paths the upstream pipeline
intended to mount at `/data/...`. Harbor will not mount them for you, but they
still tell you which files a task needs. Check what is present before running:

```bash
python - <<'PY'
import json, pathlib, sys
for line in open("examples/scibuddy_rollout/data/scibuddy.parquet"):
    pass  # handled below via pandas
import pandas as pd
df = pd.read_parquet("examples/scibuddy_rollout/data/scibuddy.parquet")
ok = True
for _, row in df.iterrows():
    for b in row["extra_info"].get("bindings", []):
        p = pathlib.Path(b["source"])
        if not p.exists():
            print(f"MISSING: {p}")
            ok = False
if ok:
    print("All binding sources present.")
PY
```

Pre-pull base images on all Harbor nodes before running. These images are not
on the public registry so they must be loaded from a private source:

```bash
# Load from a private archive or pull from your internal registry.
for node in $HARBOR_NODES; do
    ssh "$node" docker load -i /path/to/battery-discovery-cpu-0.6.0.tar
    ssh "$node" docker load -i /path/to/ehr-campaign-cpu-20260908.tar
done
```

If the images are not pre-loaded, Docker build returns exit 17 with a 403 from
the corp mirror. The episode fails with an exception but no rollout_error
propagates to the output writer.

## Results

A synthetic task in the same package format is included, so the integration can
be exercised without the real base images. Build and run it with:

```bash
python examples/scibuddy_rollout/prepare/build_dataset.py \
    --packages examples/scibuddy_rollout/synthetic_data/job-synthetic-tempstats-package.tar.gz \
    --output examples/scibuddy_rollout/data_synth/scibuddy_synth.parquet

bash examples/scibuddy_rollout/batch_rollout_smg.sh
export NOVITA_API_KEY=...
bash examples/scibuddy_rollout/batch_rollout_api.sh
```

| Backend | Model | Tasks | Score |
|---|---|---|---|
| `openai_api` | `openai/gpt-oss-120b` | 8 variants | **8/8, mean 1.000** |
| `smg_local` | Qwen3.5-4B | 1 | **1.000** |
| `openai_api` | `openai/gpt-oss-20b` | 1 | 0.000 |

The 8-variant run used 2 nodes, 4 workers, and 2 concurrent episodes each.
Each variant carries its own series, so the 8 references differ
in `n` (40 to 82), mean, and trend class. Passing all 8 therefore requires
computing each one rather than reusing a constant.

Generate the variants with:

```bash
python examples/scibuddy_rollout/synthetic_data/make_variants.py --count 8
python examples/scibuddy_rollout/prepare/build_dataset.py     --packages "examples/scibuddy_rollout/synthetic_data/variants/*.tar.gz"     --output examples/scibuddy_rollout/data_variants/scibuddy_variants.parquet
```

Both backends drive Harbor, grade through the task's own verifier, and write
`rollout.jsonl` plus `summary.json`.

`gpt-oss-20b`'s zero is a model-behavior result, not an integration failure. It
set `task_complete: true` in the same turn that emitted its heredoc, so
terminus-2 ended the episode before the script had run. Raising `max_turns` does
not help, because the model stops itself after 3 turns. `gpt-oss-120b` sequences
the same work correctly and is the cheapest reachable model that passes.

Model capability shows up in the reply fields too: 120b filled `content` on all
3 turns, 20b on 2 of 3.

## Differences from sciaccel_rl

| Aspect | sciaccel_rl | scibuddy_rollout |
|---|---|---|
| Task type | Code repair (inject + fix) | Research analysis (data + JSON output) |
| GPU | Optional (CPU tasks common) | CPU-only |
| Reward contract | Multi-key dict (`reward_repair`, `equivalence_pass`, ...) | Scalar `reward` (0 or 1) |
| Thinking template | `multi_traj` (TITO with forking) | `multi_traj`, and inert on openai_api |
| Max turns | 40 | 200 (analysis tasks run longer) |
| Task timeout | 2700 s | 28800 s (8 hours) |

## Notes

- **`prompt` column is informational only.** Harbor re-reads `instruction.md`
  from `task_path` on disk; editing the parquet prompt has no effect.
- **Task data must reach the Harbor node.** Paths resolve on whichever node runs
  the episode, so use shared-filesystem or baked-in data. `/tmp` is node-local.
- **Only `/logs/artifacts` reaches the verifier.** With
  `environment_mode = "separate"` the verifier runs in a fresh container after
  the agent's is gone. A submission left in `/workspace` is never collected, and
  the verifier rejects it while the agent looks like it succeeded.
- **Episodes are long.** A single analysis task may run for hours. Set
  `max_concurrent_episodes` conservatively to avoid overloading nodes.
