# `psrl/batch_rollout/`: offline agentic rollout collection

Runs PSRL's agent loops against a fixed model and writes the trajectories to disk.
No training, no reward, no evaluation. The third entry point alongside
`psrl.trainer.main_ppo`, in the spirit of veRL's `main_generation_server.py`.

Use it to sample a task bank with a checkpoint or an external API, then score,
filter, or inspect the dump offline. Because it reuses the same agent loops as
training, a loop that works under RL works here unmodified.

## Quickstart

```bash
# Against any OpenAI-compatible endpoint, needing no GPU.
python -m psrl.batch_rollout.main_batch_rollout \
    serving=openai_api \
    batch_rollout.serving.api_base_url=http://192.168.1.10:8000/v1 \
    batch_rollout.serving.model_name=my-model \
    batch_rollout.output_dir=outputs/rollout \
    data.train_files=${PSRL_WORKSPACE}/data/train.parquet \
    data.use_multi_dataset=False \
    gen_actor_rollout_ref.model.path=${PSRL_WORKSPACE}/models/Qwen3.5-4B \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${PSRL_WORKSPACE}/agent.yaml \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=my_loop
```

A worked recipe lives in `examples/sciaccel_rl/batch_rollout_qwen35_4b.sh`.

## Layout

This package holds what is specific to collection:

| File | Role |
|------|------|
| `main_batch_rollout.py` | Hydra entry point and `BatchRolloutTaskRunner`. |
| `output_writer.py` | `RolloutOutputWriter`: append-only `rollout.jsonl` plus resume. |
| `record.py` | Builds one JSON record per trajectory. |
| `serving/` | Serving backends. `base.py` holds the ABC and the factory. |
| `config/` | Hydra groups. `batch_rollout.yaml` is the thin root. |

The agent-loop and dataset subclasses live beside their RL siblings, because
that is where the base class and the role convention already are:

| File | Role |
|------|------|
| `workers/agent_loop/batch_rollout_worker.py` | `BatchRolloutAgentLoopWorker`: runs episodes, records results. |
| `workers/agent_loop/batch_rollout_manager.py` | `BatchRolloutAgentLoopManager`: round-robin dispatch, bounded queue. |
| `utils/dataset/batch_rollout_data_processor.py` | `BatchRolloutDataProcessor`: streams the dataset once. |

What is reused rather than reimplemented: `AgentLoopWorkerBase` and
`AgentLoopManagerBase` (shared with RL), `DataProcessorBase`, `TrajectoryWriter`,
`TurnOutputWriter`, `RolloutGateway`, `PSRL_vLLMReplica`, and the agent loops
themselves.

The parameter server is optional throughout, not bypassed. A replica built with
`GenInterface(ps_manager_handle=None)` loads its own weights and reports no status,
SMG accepts an empty `ps_manager_addr`, and the TMS / NIXL / LMCache config groups
are read optionally. So a rollout-only consumer declares none of them.

## Output

```
<output_dir>/
├── rollout.jsonl      # one JSON record per trajectory
├── summary.json       # counts, termination histogram, run metadata
├── transcripts/       # per-episode messages (openai_api backend)
└── logs/
    ├── trajectories/v0/{uid}.txt   # readable transcript per trajectory
    └── session_turns/{sid}.jsonl   # raw per-turn traffic
```

A record carries `uid`, `parent_id`, `terminate_reason`, `num_turns`,
`response_len`, and the agent loop's `extra_fields` **verbatim**. That last field
is the contract: whatever a loop attaches (a verifier reward dict, a patch, a task
name) reaches the dump untouched, so scoring is a pure offline transform.

For SciAccel, score the dump with the training reward function:

```python
import json
from examples.sciaccel_rl.reward import compute_score

for line in open("outputs/rollout/rollout.jsonl"):
    r = json.loads(line)
    extra = {**r.get("extra_info", {}), **r["extra_fields"]}
    print(r["uid"], compute_score("sciaccel_rl", "", None, extra)["score"])
```

## Serving backends

| Backend | Model source | Token capture |
|---|---|---|
| `openai_api` | Any OpenAI-compatible endpoint, or a fleet's `endpoints.json` | No |
| `smg_local` | Local SMG gateway over PSRL vLLM replicas, run without a parameter server | Yes |

`openai_api` covers a subscription proxy and a vLLM fleet started separately with
`python -m psrl.eval.serve`. Pass `batch_rollout.serving.endpoints_file` to read
that fleet's manifest instead of a URL.

`smg_local` launches everything itself: the SMG gateway, the SessionRouter, and
`n_instances` vLLM replicas that load the checkpoint directly. It needs GPUs, and
`tp * pp * dp` must equal `ngpus_per_node_per_instance * nnodes_per_instance`.

```bash
python -m psrl.batch_rollout.main_batch_rollout \
    serving=smg_local \
    batch_rollout.serving.n_instances=2 \
    batch_rollout.serving.ngpus_per_node_per_instance=2 \
    batch_rollout.dump_tokens=true \
    gen_actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    ...
```

**Session-scoped loops work on both.** A loop that hands an external agent a
session URL and reads the session back needs a session layer, which a plain
OpenAI endpoint lacks. `openai_api` starts a local session proxy to provide it.

**The proxy does not reconstruct tokens.** It records messages and reports turn
counts from the endpoint's own `usage` block, leaving `response_ids` empty. Local
re-tokenization would produce ids that never passed through the server, and
writing those into the field that carries real token output would give fabricated
tokens the provenance of measured ones. So `batch_rollout.dump_tokens` is
meaningful only under `smg_local`, and logs a warning otherwise.

## Flow control

Two knobs bound the work in flight, and nothing else does:

| Knob | Meaning |
|---|---|
| `batch_rollout.num_workers` | Worker actors. One per node is the useful default, because each worker's episodes share that node's Docker daemon. |
| `batch_rollout.max_in_flight_per_worker` | Concurrent episodes per worker. |

The prompt queue is sized at `num_workers * max_in_flight_per_worker`, and the
dataset feed blocks when it is full, so the dataset is never fully materialized.
Set `max_in_flight_per_worker` no higher than the agent loop's own
`max_concurrent_episodes`, which is what actually bounds containers.

Worker placement round-robins over `gen_actor_rollout_ref.rollout.agent.node_ips`
when that list is set, with hard node affinity.

## Resume

With `batch_rollout.resume=True` (the default), a rerun reads the existing
`rollout.jsonl` and skips the uids in it. Each record is flushed as it lands, so a
killed run loses only the episodes still in flight.

Resume keys on `uid`, which is allocated over dataloader order. It is therefore
only meaningful when the dataset files, `data.seed`, and `data.shuffle` are
unchanged. `summary.json` records all three so a later reader can check.

## Gotchas

- **`max_model_len` must be set explicitly.** veRL leaves it null so a vLLM engine
  can read it off the model. No engine is involved here and the agent harness uses
  it as a turn budget, so the run refuses to start without it.
- **Set the model id on the backend, not the loop.** Agent loops send
  `model_config.path`, which is right for a local vLLM but which a hosted API
  rejects. `batch_rollout.serving.model_name` is substituted by the proxy.
- **Pass the credential by env var.** `batch_rollout.serving.api_key_env` names the
  variable. A literal in a Hydra override lands in shell history, the Ray
  dashboard, and the run log.
- **`/tmp` is node-local.** Put the dataset, the agent loop config, and the output
  directory on the shared filesystem, or a remote worker reads a stale copy or
  fails outright.
- **Relative paths resolve against the actor's cwd, not yours.** Pass absolute
  paths for `data.train_files` and `agent_loop_config_path`.
- **A failed episode is recorded, not dropped.** Check the `terminate_reason`
  histogram in `summary.json`: a run dominated by `rollout_error` completed
  without producing usable data.
- **`thinking_template` cannot control a hosted endpoint.** Its knobs are SMG and
  vLLM conventions that a third-party gateway drops silently. A reasoning model
  there may answer in `reasoning_content` and leave `content` empty, which an
  agent harness reads as an empty reply. Nothing in the config fixes that, so
  pick a model that fills `content` and check a transcript on a new endpoint.
- **`dump_tokens` is inert on `openai_api`.** The warning in the log is the only
  signal, because no token field is written at all.
- **A PS-free replica must not be given a PSManager handle.** It would start on
  dummy weights and wait for a push that never arrives, serving noise that still
  grades. `BatchRolloutReplica` rejects one outright.
- **`num_turns: 1` on a long episode means turn chaining was lost.** SMG links a
  session's turns by prefix only under `trajectory_id_strategy=auto`. The episode
  still grades correctly, so the missing turns are easy to overlook.
- **An empty `extra_fields` means the loop attached nothing.** For a verifier-based
  task that is indistinguishable from a graded zero once defaulted, so treat it as
  missing rather than as a score.

## Testing

```bash
source ${PSRL_WORKSPACE}/env/psrl.sh
pytest tests/batch_rollout/ -v
```

`tests/batch_rollout/stub_server.py` and `stub_agent_loop.py` drive the whole
pipeline with no GPU, model, or Docker:

```bash
python -m tests.batch_rollout.stub_server --port 9999 &
python -m psrl.batch_rollout.main_batch_rollout \
    serving=openai_api \
    batch_rollout.serving.api_base_url=http://127.0.0.1:9999/v1 \
    batch_rollout.serving.model_name=stub \
    data.train_max_samples=2 batch_rollout.num_workers=1 \
    ...
```
