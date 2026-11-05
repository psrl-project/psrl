#!/bin/bash
# Bring up a Ray cluster for the multi-node NIXL tests.
#
#   scripts/start_nixl_cluster.sh <head-ip> <worker-ip>...
#
# Every node is started through scripts/nixl_env.sh, which sources the cluster's shared
# environment. That is the point of this script rather than running `ray start` by hand:
#
#   * Ray refuses to join a cluster whose head runs a different interpreter
#     ("Version mismatch: ... Python 3.12.13 / 3.12.2"). An interactive shell and a
#     non-interactive `ssh host cmd` resolve different condas, so starting the head
#     locally and the workers over ssh is enough to produce that skew.
#   * Ray and gRPC must not go through the HTTP proxy the cluster exports, or the GCS
#     connection times out. The proxy is unset here for the same reason the cluster's
#     own env file flags it.
#   * A fresh `ray stop --force` everywhere first: node IDs are regenerated on restart,
#     and actors pinned with NodeAffinitySchedulingStrategy to a previous node ID sit in
#     PENDING_CREATION forever rather than failing. A partial restart is worse than none.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SH="${REPO}/scripts/nixl_env.sh"
RAY_BIN="${PSRL_RAY_BIN:-/apdcephfs_zwfy10/share_303541817/lhy/anaconda3/bin/ray}"
PORT="${PSRL_RAY_PORT:-6399}"
GPUS="${PSRL_RAY_GPUS:-8}"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <head-ip> [worker-ip ...]" >&2
    exit 2
fi
HEAD_IP="$1"; shift
WORKERS=("$@")

echo "=== stopping any previous cluster ==="
# `ray stop` only reaps processes it can find through its own session bookkeeping.
# A head start that died partway through (the session-name race in
# `Node._write_cluster_info_to_kv`, seen when a previous attempt's gcs_server/raylet
# were not fully torn down before the next one claimed the port) leaves orphans that
# bookkeeping never learns about, so `ray stop` silently does nothing to them and the
# next `ray start --head` reaches the same race again. Killing the known process
# names directly is what actually frees the port.
KILL_CMD='pkill -9 -f "ray/core/src/ray/gcs/gcs_server"; pkill -9 -f "ray/core/src/ray/raylet/raylet"; pkill -9 -f "ray/dashboard"; pkill -9 -f "runtime_env/agent"; true'
for ip in "$HEAD_IP" "${WORKERS[@]}"; do
    # `ray stop --force` itself can hang (observed waiting past 90s against a node
    # whose previous session died mid-start), so it gets its own short budget rather
    # than sharing the outer timeout -- otherwise a hung `ray stop` eats the whole
    # budget and the pkill fallback after it never runs. The pkill commands are
    # inlined rather than shipped as a `declare -f` function body: that round-trips
    # through this script's own quoting, the ssh argv, and the remote shell, and the
    # nesting is exactly the kind of thing that breaks silently rather than loudly.
    timeout 90 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$ip" \
        "timeout 15 ${RAY_BIN} stop --force >/dev/null 2>&1; ${KILL_CMD}" \
        </dev/null >/dev/null 2>&1 || true
done
timeout 15 "${RAY_BIN}" stop --force > /dev/null 2>&1 || true
eval "${KILL_CMD}" > /dev/null 2>&1
sleep 5

if timeout 10 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -n "$HEAD_IP" \
        "ss -ltn 2>/dev/null | grep -q ':${PORT} '" 2>/dev/null; then
    echo "ERROR: port ${PORT} on ${HEAD_IP} is still bound after cleanup" >&2
    exit 1
fi

echo "=== checking network profile ==="
# shellcheck disable=SC1090
source "${ENV_SH}"
unset http_proxy https_proxy

# UCX device selection is per cluster, not per node: a static list that spans several
# RoCE rails is only usable where the rails are mutually routable. Forwarding the
# caller's override makes `PSRL_UCX_NET_DEVICES=bond1 scripts/start_nixl_cluster.sh ...`
# apply to every node, including the workers that do not inherit this shell.
UCX_ENV=""
if [ -n "${PSRL_UCX_NET_DEVICES:-}" ]; then
    UCX_ENV="export PSRL_UCX_NET_DEVICES=${PSRL_UCX_NET_DEVICES};"
fi

# A split device list is worse than a slow one. NIXL resolves a peer's UCX endpoints
# during the notification handshake; an address present on only one side is tried and
# times out before the control message is delivered (`NIXL_ERR_NOT_FOUND` at the
# PSManager, thirty seconds per endpoint). The same profile must be sourced everywhere.
HEAD_UCX="${UCX_NET_DEVICES:-}"
for ip in "${WORKERS[@]}"; do
    worker_ucx=$(timeout 60 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ip" \
        "${UCX_ENV} source ${ENV_SH} >/dev/null 2>&1; printf '%s' \"\${UCX_NET_DEVICES:-}\"" 2>/dev/null)
    if [ "${worker_ucx}" != "${HEAD_UCX}" ]; then
        echo "ERROR: UCX_NET_DEVICES differs between nodes" >&2
        echo "  head ${HEAD_IP}: ${HEAD_UCX}" >&2
        echo "  node ${ip}: ${worker_ucx}" >&2
        echo "Source the same cluster profile on every node, or set PSRL_UCX_NET_DEVICES for all." >&2
        exit 1
    fi
done

echo "=== starting head on ${HEAD_IP} ==="
if ! timeout 240 "${RAY_BIN}" start --head \
        --node-ip-address="${HEAD_IP}" --port="${PORT}" --num-gpus="${GPUS}" \
        2>&1 | grep -q "Ray runtime started"; then
    echo "ERROR: head failed to start" >&2
    exit 1
fi

for ip in "${WORKERS[@]}"; do
    echo "=== joining ${ip} ==="
    joined=$(timeout 260 ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ip" \
        "${UCX_ENV} source ${ENV_SH} >/dev/null 2>&1; unset http_proxy https_proxy; \
         ${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR:+export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR};} \
         timeout 200 ${RAY_BIN} start --address='${HEAD_IP}:${PORT}' --num-gpus=${GPUS} 2>&1 \
         | grep -c 'Ray runtime started'" 2>/dev/null | tail -1)
    [ "${joined:-0}" = "1" ] || { echo "ERROR: ${ip} failed to join" >&2; exit 1; }
done

echo "=== verifying ==="
python - "$HEAD_IP" "$PORT" <<'PY'
import sys
import ray

ray.init(address=f"{sys.argv[1]}:{sys.argv[2]}", ignore_reinit_error=True, logging_level="ERROR")
alive = [n for n in ray.nodes() if n["Alive"]]
gpus = int(sum(n["Resources"].get("GPU", 0) for n in alive))
print(f"CLUSTER_READY nodes={len(alive)} gpus={gpus}")
PY
