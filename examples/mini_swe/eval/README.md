# Standalone Evaluation and Model Serving

This directory packages every tool needed to evaluate SWE-bench / SWE-smith
outside the training loop, and to serve your own checkpoint as an
OpenAI-compatible endpoint for mini-swe-agent to drive.

| File | Purpose |
|------|---------|
| [`eval_swebench.py`](eval_swebench.py) | Single-host evaluation entry point (rollout + grading). Supports an HF dataset key *or* a prepared parquet file via `--dataset`. |
| [`eval_swebench_multinode.py`](eval_swebench_multinode.py) | Hash-shards a prepared parquet across hosts, fans `eval_swebench` out over ssh, merges per-shard artefacts into one output directory. Forwards `OPENAI_API_BASE` / `OPENAI_API_KEY` to every host. |

Model serving is **not** in this directory: it lives in
[`psrl/eval/`](../../../psrl/eval/) because `examples/sciaccel_rl/eval/` needs the
same thing. Use `python -m psrl.eval.serve` with `topology=single|fleet|multinode`.

The grader itself — [`../swebench_grader.py`](../swebench_grader.py) — stays at
the top level of `examples/mini_swe/` because it is shared between standalone
evaluation here and the post-rollout grading used inside the training loop.

---

## Standalone Evaluation

Evaluation runs independently of the training loop via `eval_swebench.py`.
This is used for:

- **Gold-patch sanity check** — verifies that Docker images and the grader are
  correctly configured before investing in a full training run.
- **Baseline measurement** — run on the initial checkpoint to establish a reference.
- **Checkpoint evaluation** — run on a saved checkpoint to produce leaderboard-ready
  `preds.json`.

### Gold-patch sanity check

Every SWE-bench SWE problem should resolve when the gold patch is applied. Use
this to confirm that images are present and the grading pipeline works end-to-end:

```bash
python -m examples.mini_swe.eval.eval_swebench \
    --gold-patches \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --subset-spec "^astropy" \
    --output-dir examples/mini_swe/output/eval/gold_sanity \
    --workers 4
```

Expect most of the problems are resolved, e.g, 68/80 (85.0%). SWE problems that fail consistently likely have fragile test environments; exclude them from future evaluation subsets if needed.

Heavy repos (`scikit-learn`, `psf/requests`, `matplotlib`) spend most of their
time on `pip install -e .` + the full PASS_TO_PASS suite and can legitimately
take 10–20 minutes each; keep `--grader-timeout` at the default `1800` (or
higher) and tune `--workers` upward until per-task `elapsed_s` starts to rise
(rough rule: `min(CPU_cores / 8, RAM_GiB / 12, 16)`).

### Multi-node evaluation

For larger subsets (e.g. the full 500-problem Verified split) fan the eval
out across every host listed in a hosts file. Each host is given a shard of
the parquet (bucketed by `hash(instance_id)`), runs `eval_swebench` locally
with its usual `--workers`, and the per-shard artefacts are merged into one
combined output directory:

```bash
python -m examples.mini_swe.eval.eval_swebench_multinode \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/gold_sanity_mn \
    --gold-patches \
    --workers-per-node 8 \
    --grader-timeout 1800 \
    --ssh-timeout 3600
```

Prerequisites are the same as `prepare/docker_scripts/load_all_nodes.sh`:
the repo, conda env, and output directory live on a shared FS; the target
hosts already have their shard's Docker images loaded (run
`load_all_nodes.sh` first); and passwordless ssh works from the launcher
to every host. Inspect `--dry-run` output first to review the exact ssh
commands that will be issued. Per-host stdout/stderr lands in
`<output-dir>/host_logs/`, raw per-shard artefacts in
`<output-dir>/host_output/<host>/`, and the merged `preds.json` /
`results.jsonl` / `summary.json` live at the top of `<output-dir>`
(per-instance directories are symlinked up from the host-specific output).

### Evaluating your own checkpoint

`eval_swebench.py` is **decoupled from the training stack**: it does not use
PSRL's agent loop at all. Internally it calls mini-swe-agent's `LitellmModel`,
which issues `litellm.completion(model=<prefixed_model_name>, ...)` for every
turn. The `openai/` prefix applied in [`eval_swebench.py`](eval_swebench.py)
means every request goes to whatever HTTP endpoint `OPENAI_API_BASE` points
at. So to evaluate your own checkpoint, serve it with **any OpenAI-compatible
server** (vLLM / sglang / TGI / llama.cpp server / litellm proxy) and point
the eval at it.

Serving lives in [`psrl/eval/`](../../../psrl/eval/), shared with
`examples/sciaccel_rl/eval/`:

#### Single-node

PSRL trains with the `mswea_bash_command` text-block format (not OpenAI
tool-calls), so vLLM should be started as a **plain text-completion server**
— no `--tool-call-parser`, no `--enable-auto-tool-choice`:

```bash
python -m psrl.eval.serve \
    topology=single \
    topology.tp=4 \
    server.checkpoint=${PSRL_WORKSPACE}/checkpoints/my-step-1000 \
    server.served_model_name=my-model \
    output_dir=output/serve/my-step-1000
# server.tool_call_parser is empty in every preset. The model was trained to
# emit ```mswea_bash_command blocks, not OpenAI tool-call JSON.
```

For several independent replicas on one host use `topology=fleet`
(`topology.replicas=4 topology.tp=2` fills 8 GPUs, one endpoint each). Prefer that
over `topology.dp` — vLLM's in-server data parallelism — which is broken in this
repo's patched build. Both write `<output_dir>/endpoints.json`, containing only
replicas that passed a health check.

`--tool-call-parser` is only needed when serving **external models** (GPT-4,
Claude, Llama3-Instruct, etc.) that natively output OpenAI tool-call JSON and
are evaluated with `--model-class litellm` (see "Evaluating external models"
below).

The script launches vLLM with `nohup setsid`, polls `/v1/models` until the
server reports ready, and writes `/tmp/vllm_<port>.{log,pid}` so you can
tail logs and kill cleanly. Add `--foreground` when debugging.

Then point the eval at it — the default `--model-class litellm_textbased`
matches the training-time action format:

```bash
export OPENAI_API_BASE=http://<serve_host>:8000/v1
export OPENAI_API_KEY=dummy   # vLLM ignores it, but litellm requires the field

python -m examples.mini_swe.eval.eval_swebench \
    --model my-model \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/my_step1000 \
    --workers 8 \
    --grader-timeout 1800
# --model-class litellm_textbased is the default; no need to specify it
# unless you want to override to 'litellm' for an external model.
```

#### Cross-node (one fleet per host)

For bigger throughput, run a fleet on every host in a hosts file:

```bash
python -m psrl.eval.serve \
    topology=multinode \
    topology.hosts_file=${PSRL_WORKSPACE}/hosts/32GPUs \
    topology.replicas=1 \
    topology.tp=4 \
    server.checkpoint=${PSRL_WORKSPACE}/checkpoints/my-step-1000 \
    server.served_model_name=my-model \
    output_dir=examples/mini_swe/output/serve/my_step1000
# server.tool_call_parser stays empty for PSRL-trained models.
```

Total endpoints are `hosts x topology.replicas`. Keep `replicas=1` when using the
direct-to-localhost pattern below, since each eval shard is given exactly one
`OPENAI_API_BASE`; put extra parallelism in `topology.dp` instead so all replicas
share one port. Scaling out is a different `hosts_file` and nothing else.

The launcher writes `<output_dir>/endpoints.json` listing every healthy endpoint
with its host, GPUs, and PID. Hosts that came up with nothing are logged and
excluded. It exits non-zero only when healthy endpoints fall below
`topology.min_healthy_frac` (default 0.5), so a partial cluster still runs.

**Recommended: direct-to-localhost (no central proxy)**

Each eval shard calls its own local vLLM.  
This avoids the `litellm` CLI, which hangs on startup in environments with a corporate HTTP proxy:

```bash
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export NO_PROXY="localhost,127.0.0.1"

python -m examples.mini_swe.eval.eval_swebench_multinode \
    --hosts ${PSRL_WORKSPACE}/hosts/32GPUs \
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/my_step1000_mn \
    --model my-model \
    --workers-per-node 8 \
    --grader-timeout 1800 \
    --ssh-timeout 3600
```

`eval_swebench_multinode.py` auto-forwards `OPENAI_API_BASE`, `NO_PROXY` and
`OPENAI_API_KEY` to every remote host, so no per-host bashrc editing is
needed.  
Use `--forward-env NAME` to add extra vars or `--set-env NAME=VALUE`
to override.

**Alternative: litellm proxy (open-network environments)**

If there is no corporate proxy, a central litellm router can load-balance across
all replicas. The launcher no longer generates a router config — build one from
`endpoints.json`, which is the authoritative list of healthy replicas:

```bash
python -c "
import json, sys
payload = json.load(open(sys.argv[1]))
name = payload['served_model_name']
print('model_list:')
for e in payload['endpoints']:
    print(f'  - model_name: {name}')
    print('    litellm_params:')
    print(f'      model: openai/{name}')
    print(f\"      api_base: {e['url']}\")
    print('      api_key: dummy')
print('router_settings:')
print('  routing_strategy: least-busy')
" <output_dir>/endpoints.json > litellm_proxy.yaml

litellm --config litellm_proxy.yaml --port 4000 &
export OPENAI_API_BASE=http://<launcher_ip>:4000/v1
export OPENAI_API_KEY=dummy
# then run eval_swebench_multinode as above
```

#### Evaluating external models (optional: OpenAI / Claude / third-party)

For models that natively support OpenAI tool-calling (GPT-4o, Claude,
Llama3-Instruct, etc.) you need the tool-call parser on the vLLM side:

```bash
python -m psrl.eval.serve \
    topology=single \
    topology.port=8001 \
    topology.tp=4 \
    server.checkpoint=${PSRL_WORKSPACE}/checkpoints/external-model \
    server.served_model_name=ext-model \
    server.tool_call_parser=hermes \
    output_dir=output/serve/ext-model
# llama3_json / mistral / deepseek_v3 as needed
```

Then eval with `--model-class litellm`:

```bash
export OPENAI_API_BASE=http://<serve_host>:8001/v1
export OPENAI_API_KEY=dummy
python -m examples.mini_swe.eval.eval_swebench \
    --model ext-model \
    --model-class litellm \        # switches to the OpenAI tool-call path
    --dataset examples/mini_swe/data/verified_subset_80/train.parquet \
    --output-dir examples/mini_swe/output/eval/ext_model
```

#### Cross-node tensor parallelism

`topology=multinode` does **not** do cross-node TP — every replica stays inside one
host, so a wedged host costs only its own capacity instead of hanging a replica.
If your model is too large for one host's GPUs, start a Ray cluster yourself
(`ray start --head` on the head node, `ray start --address=<head>:6379` on the
workers), then run a single server on the head node and pass the Ray backend
through:

```bash
python -m psrl.eval.serve \
    topology=single \
    topology.tp=<total_gpus> \
    server.checkpoint=<ckpt> \
    server.served_model_name=big-model \
    'server.extra=[--distributed-executor-backend,ray]' \
    output_dir=output/serve/big-model
```

#### Stopping servers

`endpoints.json` records each replica's PID, so teardown does not need a pid file:

```bash
# Every replica this launcher started on one host
python -c "
import json, os, signal, sys
for e in json.load(open(sys.argv[1]))['endpoints']:
    os.kill(e['pid'], signal.SIGTERM)
" <output_dir>/endpoints.json

# Every host in a hosts file
pssh -h ${PSRL_WORKSPACE}/hosts/32GPUs -i \
    "pkill -f 'vllm.entrypoints.openai.api_server'"
```

### Checkpoint evaluation (HF dataset mode)

When you just want to run on a slice of the raw HF Verified split instead
of a prepared parquet:

```bash
python -m examples.mini_swe.eval.eval_swebench \
    --model /path/to/checkpoint \
    --dataset verified \
    --split test \
    --subset-spec "0:100" \
    --output-dir output/eval/step200 \
    --workers 8 \
    --max-turns 30
```

For a full leaderboard submission run omit `--subset-spec` to evaluate all 500
Verified SWE problems.

### Output artefacts

```
<output-dir>/
  preds.json          # { instance_id: {model_patch, model_name_or_path, ...} }
  summary.json        # { resolved, total, resolve_rate, avg_turns, elapsed_s, ... }
  results.jsonl       # One JSON object per line, per-SWE-problem result
  <instance_id>/      # One directory per SWE problem (named after its HF instance_id)
    traj.json         # Full conversation + exit status
    patch.diff        # Submitted patch
    grading.json      # Raw output of grade_fresh_container
```

`preds.json` is compatible with the official `swebench.harness.run_evaluation`
grader for leaderboard submission.

### In-training validation

During training, PSRL runs validation rollouts on `test_files` every `test_freq`
steps using the same `MiniSWEAgentLoopV1` and `compute_score` as training. The
`train/acc` and `val/acc` wandb metrics track resolve rate throughout training
without needing to invoke `eval_swebench.py`.
