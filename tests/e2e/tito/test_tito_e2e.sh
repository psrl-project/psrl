#!/usr/bin/env bash
# Exercise TITO training data construction across vLLM, SMG, and `SessionRouter`. Prerequisites are `env/psrl.sh`, the SMG bindings, and a downloaded model.
# Usage: bash tests/e2e/tito/test_tito_e2e.sh [model_path] or HOST_IP=127.0.0.1 bash tests/e2e/tito/test_tito_e2e.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSRL_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-${PSRL_WORKSPACE}/models/Qwen2.5-0.5B-Instruct}}"
HOST_IP="${HOST_IP:-127.0.0.1}"
VLLM_GRPC_PORT="${VLLM_GRPC_PORT:-50051}"
SMG_PORT="${SMG_PORT:-8150}"
SESSION_ROUTER_PORT="${SESSION_ROUTER_PORT:-8200}"
MAX_TURNS="${MAX_TURNS:-3}"
TRAJECTORY_ID_STRATEGY="${TRAJECTORY_ID_STRATEGY:-manual}"
LOG_DIR="${LOG_DIR:-/tmp/tito_e2e_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
echo "=== TITO E2E Test (Full SMG Pipeline) ==="
echo "  Model:   $MODEL_PATH"
echo "  Host:    $HOST_IP"
echo "  Ports:   vLLM(gRPC)=$VLLM_GRPC_PORT  SMG=$SMG_PORT  SessionRouter=$SESSION_ROUTER_PORT"
echo "  Trajectory ID strategy: $TRAJECTORY_ID_STRATEGY"
echo "  Logs:    $LOG_DIR"
echo ""

PIDS=()
cleanup() {
    echo ""
    echo ">>> Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${PIDS[@]}" 2>/dev/null || true
    echo ">>> Done. Logs in $LOG_DIR"
}
trap cleanup EXIT

# --- Step 1: Launch vLLM gRPC Server ---
echo ">>> [Step 1/4] Launching vLLM gRPC server on ${HOST_IP}:${VLLM_GRPC_PORT} ..."
python -m vllm.entrypoints.grpc_server \
    --model "$MODEL_PATH" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.5 \
    --host "$HOST_IP" \
    --port "$VLLM_GRPC_PORT" \
    --max-model-len 4096 \
    --trust-remote-code \
    > "$LOG_DIR/vllm_grpc.log" 2>&1 &
PIDS+=($!)

echo "    Waiting for vLLM gRPC to be ready..."
for i in $(seq 1 120); do
    if python -c "
import grpc
channel = grpc.insecure_channel('${HOST_IP}:${VLLM_GRPC_PORT}')
grpc.channel_ready_future(channel).result(timeout=1)
" 2>/dev/null; then
        echo "    ✓ vLLM gRPC ready (${i}s)"
        break
    fi
    if [ "$i" -eq 120 ]; then
        echo "    ✗ vLLM gRPC failed to start. See $LOG_DIR/vllm_grpc.log"
        tail -20 "$LOG_DIR/vllm_grpc.log"
        exit 1
    fi
    sleep 1
done

# --- Step 2: Launch SMG and Register vLLM ---
echo ">>> [Step 2/4] Launching SMG router on ${HOST_IP}:${SMG_PORT} ..."
python -c "
from smg.launch_router import launch_router
from smg.router_args import RouterArgs

router_args = RouterArgs(
    host='0.0.0.0',
    port=${SMG_PORT},
    connection_mode='grpc',
    policy='round_robin',
    disable_retries=True,
    enable_tito=True,
    tito_debug=False,
    trajectory_id_strategy='${TRAJECTORY_ID_STRATEGY}',
    request_timeout_secs=120,
    log_level='info',
    log_dir='${LOG_DIR}',
    model_path='${MODEL_PATH}',
)
launch_router(router_args)
" > "$LOG_DIR/smg.log" 2>&1 &
PIDS+=($!)

echo "    Waiting for SMG to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://${HOST_IP}:${SMG_PORT}/health" > /dev/null 2>&1; then
        echo "    ✓ SMG ready (${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "    ✗ SMG failed to start. See $LOG_DIR/smg.log"
        tail -20 "$LOG_DIR/smg.log"
        exit 1
    fi
    sleep 1
done

# Register the vLLM gRPC worker with SMG.
# NOTE(lhy): The worker URL must omit the `http://` prefix.
echo "    Registering vLLM worker with SMG..."
REGISTER_HTTP_CODE=$(curl -s -o "$LOG_DIR/register_resp.json" -w "%{http_code}" \
    -X POST "http://${HOST_IP}:${SMG_PORT}/workers" \
    -H "Content-Type: application/json" \
    -d "{
        \"url\": \"${HOST_IP}:${VLLM_GRPC_PORT}\",
        \"connection_mode\": \"grpc\",
        \"runtime_type\": \"vllm\"
    }")
if [ "$REGISTER_HTTP_CODE" -ge 400 ]; then
    echo "    ✗ Worker registration returned HTTP $REGISTER_HTTP_CODE:"
    cat "$LOG_DIR/register_resp.json"
    echo ""
    tail -10 "$LOG_DIR/smg.log"
    exit 1
fi
echo "    ✓ Worker registered (HTTP $REGISTER_HTTP_CODE)"

echo "    Waiting for SMG to serve chat completions..."
SMG_MODEL=""
for i in $(seq 1 60); do
    SMG_MODEL=$(curl -sf "http://${HOST_IP}:${SMG_PORT}/v1/models" 2>/dev/null \
        | python -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null || echo "")
    if [ -n "$SMG_MODEL" ]; then
        echo "    ✓ Model registered as: $SMG_MODEL (${i}s)"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "    ✗ No model registered after 60s."
        tail -30 "$LOG_DIR/smg.log"
        exit 1
    fi
    sleep 1
done

for i in $(seq 1 30); do
    RESP=$(curl -sf -X POST "http://${HOST_IP}:${SMG_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${SMG_MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}" 2>&1) || true
    if echo "$RESP" | python -c "import sys,json; d=json.load(sys.stdin); assert 'choices' in d" 2>/dev/null; then
        echo "    ✓ SMG end-to-end ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "    ✗ SMG+vLLM not serving after 30s. Last: $RESP"
        tail -30 "$LOG_DIR/smg.log"
        exit 1
    fi
    sleep 1
done

# --- Step 3: Launch SessionRouter ---
echo ">>> [Step 3/4] Launching SessionRouter on ${HOST_IP}:${SESSION_ROUTER_PORT} ..."
python -c "
import uvicorn
from psrl.workers.gen.session_router import SessionRouter
router = SessionRouter(
    smg_url='http://${HOST_IP}:${SMG_PORT}',
    trajectory_id_strategy='${TRAJECTORY_ID_STRATEGY}',
)
uvicorn.run(router.app, host='0.0.0.0', port=${SESSION_ROUTER_PORT}, log_level='warning')
" > "$LOG_DIR/session_router.log" 2>&1 &
PIDS+=($!)

echo "    Waiting for SessionRouter to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://${HOST_IP}:${SESSION_ROUTER_PORT}/openapi.json" > /dev/null 2>&1; then
        echo "    ✓ SessionRouter ready (${i}s)"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "    ✗ SessionRouter failed to start. See $LOG_DIR/session_router.log"
        exit 1
    fi
    sleep 1
done

# --- Step 4: Run Verification ---
echo ""
echo ">>> [Step 4/4] Running verification tests..."
echo ""

python "$SCRIPT_DIR/verify_tito_training_data.py" \
    --session-router-url "http://${HOST_IP}:${SESSION_ROUTER_PORT}" \
    --smg-url "http://${HOST_IP}:${SMG_PORT}" \
    --model-path "$MODEL_PATH" \
    --max-turns "$MAX_TURNS" \
    --timeout 120.0

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo ">>> E2E TEST PASSED ✓"
else
    echo ">>> E2E TEST FAILED ✗ (exit code: $EXIT_CODE)"
    echo ">>> Check logs in: $LOG_DIR"
fi

exit $EXIT_CODE
