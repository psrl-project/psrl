#!/usr/bin/env bash
# Evaluate a PSRL checkpoint on the 80-problem Verified subset.
# Each host serves its own model and evaluates one dataset shard.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${PSRL_WORKSPACE}/models/SWE-agent-LM-7B"
SERVED_MODEL_NAME="SWE-agent-LM-7B"
HOSTS_FILE="${PSRL_WORKSPACE}/hosts/32GPUs_another"
DATASET="examples/mini_swe/data/verified_subset_80/train.parquet"
OUTPUT_DIR="$SCRIPT_DIR/output"

SERVE_PORT=8000
TP=2
DP=4
WORKERS_PER_NODE=8
GRADER_TIMEOUT=1800
SSH_TIMEOUT=7200

# Shared-FS env script (conda + NCCL / UCX / vLLM / library paths).
ENV_SCRIPT="${PSRL_WORKSPACE}/env/env_311.sh"

# Temporarily disable nounset because environment setup accepts unset variables.
set +u
# shellcheck disable=SC1090
source "$ENV_SCRIPT"
set -u

SERVE_OUTDIR="$OUTPUT_DIR/serve"
EVAL_OUTDIR="$OUTPUT_DIR/eval"

mkdir -p "$SERVE_OUTDIR" "$EVAL_OUTDIR"

# --- Start vLLM on each host ---
echo "=== Step 1: starting vLLM on all hosts ==="
# Each host exposes one endpoint backed by data parallel replicas.
# Read `endpoints.json` because partially healthy fleets remain usable.
set +e
python -m psrl.eval.serve \
    topology=multinode \
    topology.hosts_file="$HOSTS_FILE" \
    topology.replicas=1 \
    topology.tp="$TP" \
    topology.dp="$DP" \
    topology.base_port="$SERVE_PORT" \
    topology.wait_ready_sec=1800 \
    server.checkpoint="$MODEL" \
    server.served_model_name="$SERVED_MODEL_NAME" \
    server.max_model_len=32768 \
    env_script="$ENV_SCRIPT" \
    output_dir="$SERVE_OUTDIR"
SERVE_RC=$?
set -e
# The agent parses bash blocks, so vLLM needs no tool call parser.

ENDPOINTS_JSON="$SERVE_OUTDIR/endpoints.json"
if [[ ! -s "$ENDPOINTS_JSON" ]]; then
    echo "ERROR: no healthy endpoints after serve step (rc=$SERVE_RC). Aborting." >&2
    echo "Check per-host logs under $SERVE_OUTDIR/hosts/ and remote /tmp/vllm_${SERVE_PORT}.log." >&2
    exit 1
fi

N_HEALTHY=$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_endpoints'])" "$ENDPOINTS_JSON")
N_HOSTS=$(grep -cEv '^[[:space:]]*(#|$)' "$HOSTS_FILE")
echo
echo "Healthy endpoints: $N_HEALTHY / $N_HOSTS"
python -c "
import json, sys
for e in json.load(open(sys.argv[1]))['endpoints']:
    print(f\"  {e['url']}\")
" "$ENDPOINTS_JSON"
if [[ "$N_HEALTHY" -lt "$N_HOSTS" ]]; then
    echo
    echo "NOTE: Continuing with $N_HEALTHY healthy host(s)."
    echo "Inspect failed-host vLLM logs and increase topology.wait_ready_sec if compilation is still running."
fi
echo

# --- Configure local model endpoints ---

# Each shard uses local vLLM, and `NO_PROXY` bypasses corporate proxies.
echo "=== Step 2: configuring per-host vLLM endpoints ==="
export OPENAI_API_BASE="http://localhost:${SERVE_PORT}/v1"
export OPENAI_API_KEY="dummy"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
echo "Each eval shard will call its own local vLLM at $OPENAI_API_BASE"
echo "NO_PROXY includes localhost so http_proxy is bypassed for LLM calls"

# --- Run distributed evaluation ---
echo "=== Step 3: running multi-node eval ==="
cd "$REPO_ROOT"
python -m examples.mini_swe.eval.eval_swebench_multinode \
    --hosts "$HOSTS_FILE" \
    --dataset "$DATASET" \
    --output-dir "$EVAL_OUTDIR" \
    --model "$SERVED_MODEL_NAME" \
    --model-class "examples.mini_swe.eval.xml_fc_model.XmlFcModel" \
    --config "examples/mini_swe/config/swebench_agent_config_full_sweagent.yaml" \
    --workers-per-node "$WORKERS_PER_NODE" \
    --grader-timeout "$GRADER_TIMEOUT" \
    --ssh-timeout "$SSH_TIMEOUT" \
    --repo-root "$REPO_ROOT" \
    --env-script "$ENV_SCRIPT"

# --- Print the summary ---
echo
echo "=== Results ==="
EVAL_OUTDIR="$EVAL_OUTDIR" python - <<'PY'
import json, pathlib, os
summary_path = pathlib.Path(os.environ["EVAL_OUTDIR"]) / "summary.json"
if summary_path.exists():
    s = json.loads(summary_path.read_text())
    print(f"Resolved  : {s['resolved']}/{s['total']} ({s['resolve_rate']:.1%})")
    print(f"Avg turns : {s['avg_turns']:.1f}")
    print(f"Wall clock: {s.get('elapsed_s', '?')}s")
    print(f"Output    : {summary_path.parent}")
else:
    print(f"WARNING: {summary_path} not found.")
PY

echo
echo "To stop vLLM replicas:"
echo "  pssh -h $HOSTS_FILE -i \"pkill -f 'vllm.entrypoints.openai.api_server.*--port $SERVE_PORT'\""
