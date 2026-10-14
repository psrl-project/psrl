# `psrl/eval/` — model serving for offline evaluation

Shared by `examples/mini_swe/eval/` and `examples/sciaccel_rl/eval/`. Both used to
reach into `examples/mini_swe/eval/serve_vllm.sh`; that script and its multi-node
companion are gone.

## Layout

| File | Role |
|------|------|
| [`serve.py`](serve.py) | CLI entry point. **The only Hydra-aware module.** |
| [`vllm_server.py`](vllm_server.py) | One server process: `ServerSpec`, `build_command`, `launch`, `wait_ready`. |
| [`vllm_fleet.py`](vllm_fleet.py) | N replicas on one host: GPU partitioning, concurrent readiness, `endpoints.json`. |
| [`vllm_multinode.py`](vllm_multinode.py) | One fleet per host over a hosts file, via ssh. |
| [`config/`](config/) | Hydra groups: `server/` presets and `topology/` shapes. |

Hydra lives only at the entry point. `serve.py` turns composed config into plain
frozen dataclasses and hands off; nothing below it imports hydra or omegaconf. That
is what lets `build_command` and `partition_gpus` be unit-tested with no GPU and no
config (see `tests/eval/`), and what lets an eval harness call
`launch_fleet` directly.

## Usage

Overrides address the **group name**, not the file name: a value defined in
`topology/fleet.yaml` is set with `topology.replicas=`, never `fleet.replicas=`.

```bash
# 4 independent replicas x TP=2 across 8 GPUs
python -m psrl.eval.serve \
    server=qwen35_9b topology=fleet \
    topology.replicas=4 topology.tp=2 \
    output_dir=outputs/serve/qwen35

# one server, to check a checkpoint loads and read its KV cache size
python -m psrl.eval.serve server=qwen35_9b topology=single output_dir=/tmp/probe

# every host in a hosts file
python -m psrl.eval.serve \
    server=qwen35_9b topology=multinode \
    topology.hosts_file=${PSRL_WORKSPACE}/hosts/32GPUs \
    output_dir=/shared/serve   # must be readable from every host

# print the plan, launch nothing
python -m psrl.eval.serve server=qwen35_9b dry_run=true output_dir=/tmp/plan
```

Adding a checkpoint means adding one file to `config/server/`, which keeps
checkpoint, served name, and context window moving together instead of as three
flags to mistype.

## `endpoints.json`

Every topology writes `<output_dir>/endpoints.json`:

```json
{"served_model_name": "qwen35-9b",
 "n_endpoints": 4,
 "endpoints": [
   {"url": "http://127.0.0.1:8000/v1", "host": "0.0.0.0",
    "gpu_ids": [0, 1], "pid": 12345, "healthy": true}
 ]}
```

Only replicas that answered `/v1/models` are listed, so a downstream eval cannot
dispatch work to one that never loaded. It also records PIDs, so teardown needs no
pid file. Consumers should read this rather than reconstruct URLs from a topology
they were told about.

## Choosing a topology

**`fleet` (N ports) is the default.** Independent processes fail independently, and
vLLM's `--data-parallel-size` is broken in this repo's patched build — the DP
coordinator never reports its ZMQ addresses.

**`topology.dp` (one port) only when the consumer accepts a single URL.**
`eval_swebench_multinode` forwards one `OPENAI_API_BASE` per host, so
`examples/mini_swe/eval/example.sh` uses `replicas=1 dp=N`. `eval_sciaccel` takes a
comma-separated list and prefers a fleet.

**Multinode replicates per node**: total endpoints are `hosts x replicas`, and no
replica crosses a host boundary. Cross-node TP would need a managed Ray cluster and
would let one node failure kill a whole replica. If a model genuinely does not fit
one host, start Ray yourself and use `topology=single` with
`'server.extra=[--distributed-executor-backend,ray]'`.

## Failure policy

`topology.min_healthy_frac` gates success. Fleet defaults to `1.0`: on one host a
missing replica almost always means a real misconfiguration. Multinode defaults to
`0.5`: across many nodes a wedged host is routine, and a multi-hour eval should
lose capacity rather than abort. Failed hosts are logged and excluded, never
silently counted.

A launch that fails its quorum tears down the local fleet so GPUs are not left
occupied. Remote servers are left running — killing them over what may be a
transient ssh error risks destroying a usable fleet — and the error says how to
stop them.

## Notes

- **`exec` is load-bearing.** Servers launch via
  `bash -lc 'source <env> && exec python -m vllm...'` because a Python process
  cannot source bash, and `psrl.sh` sets the conda env plus NCCL / UCX / cudnn
  library paths. Without `exec`, the recorded PID would be the wrapping shell's, so
  SIGTERM would kill the shell and orphan the server.
- **Health checks bypass the proxy.** A corporate `http_proxy` otherwise swallows
  requests to localhost and every probe fails.
- **A crashed load is detected immediately** rather than waiting out
  `wait_ready_sec`, so the generous 1800s default (first-time `torch.compile` on a
  large model can exceed 15 minutes) costs nothing when a server dies.
- **`tool_call_parser` is empty in every preset.** terminus-2 and mini-swe-agent
  parse their own text protocol out of the assistant message; enabling vLLM's
  tool-call extraction would strip the text they need.
