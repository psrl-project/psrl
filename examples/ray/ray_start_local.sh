#!/bin/bash
# Exit on undefined variables / pipe errors to avoid silent failures.
set -uo pipefail

env_file="/data_storage/yl_test/lhy/psrl.sh"
source "${env_file}"

HOSTFILE=${1:-""}
PORT=8887                # Ray node communication port
SSH_PORT=${SSH_PORT:-2222}       # sshd listen port (local sshd is on 2222, not the default 22)
DASHBOARD_PORT=8265      # Ray Dashboard port

CONDA_SH=/data_storage/yl_test/lhy/anaconda3/etc/profile.d/conda.sh
ENV_NAME=psrl-new

# Determine whether a host is the local machine (run directly, skip ssh).
is_local_host() {
    local h="$1"
    case "$h" in
        127.0.0.1|localhost|0.0.0.0) return 0 ;;
    esac
    # If the address belongs to a local network interface, treat it as local too.
    if ip -4 addr show 2>/dev/null | grep -qw "$h"; then
        return 0
    fi
    if [ "$h" = "$(hostname)" ]; then
        return 0
    fi
    return 1
}

# Run one command on a given host: local runs directly, remote goes over ssh.
# Key point: no longer swallow stderr, no blind backgrounding, return the real exit code.
run_on_host() {
    local host="$1"
    local cmd="$2"
    local wrapped="source ${CONDA_SH} && conda activate ${ENV_NAME} && source ${env_file} && ${cmd}"

    if is_local_host "$host"; then
        echo "[$host] (local) $cmd"
        bash -c "$wrapped"
        return $?
    else
        echo "[$host] (ssh) $cmd"
        ssh -p "${SSH_PORT}" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$host" "$wrapped"
        return $?
    fi
}

# Read host list
if [ -n "${HOSTFILE}" ]; then
    # Filter blank/comment lines with grep, also handles a missing trailing newline.
    mapfile -t hosts < <(grep -vE '^\s*(#|$)' "${HOSTFILE}" | awk '{print $1}')
    if [ ${#hosts[@]} -eq 0 ]; then
        echo "Error: Empty hostfile: ${HOSTFILE}" >&2
        exit 1
    fi
else
    mapfile -t hosts < <(echo "${NODE_IP_LIST:-}" | sed "s/:.//g; s/,/\n/g" | head -n "${NODE_NUM:-1}")
    if [ ${#hosts[@]} -eq 0 ]; then
        echo "Error: NODE_IP_LIST is empty" >&2
        exit 1
    fi
fi

HEAD_IP=${hosts[0]}
workers=( "${hosts[@]:1}" )

echo "Head: ${HEAD_IP}   Workers: ${workers[*]:-<none>}"

# ---- 1. Stop any leftover Ray on all nodes ----
echo "Stopping any existing Ray processes on all nodes..."
for host in "${hosts[@]}"; do
    run_on_host "$host" "ray stop --force" || echo "[$host] ray stop returned non-zero (likely nothing was running, ignoring)"
done
# Remove the stale cluster address file, otherwise ray status would try to reach the previous dead address.
rm -f /tmp/ray/ray_current_cluster 2>/dev/null || true
echo "All nodes cleaned up."

# ---- 2. Start the head node ----
echo "Starting Head node at ${HEAD_IP}"
run_on_host "${HEAD_IP}" \
    "cd ${PSRL_WORKSPACE} && ray start --head --node-ip-address=${HEAD_IP} --port=${PORT} --dashboard-host=0.0.0.0 --dashboard-port=${DASHBOARD_PORT} --num-cpus=32"
head_rc=$?
if [ $head_rc -ne 0 ]; then
    echo "ERROR: Head node failed to start (exit code ${head_rc})" >&2
    exit $head_rc
fi

# ---- 3. Start the worker nodes ----
for host in "${workers[@]}"; do
    [ -z "$host" ] && continue
    echo "Starting Worker node at ${host}"
    run_on_host "$host" \
        "cd ${PSRL_WORKSPACE} && ray start --address=${HEAD_IP}:${PORT} --num-cpus=32"
    worker_rc=$?
    if [ $worker_rc -ne 0 ]; then
        echo "ERROR: Worker node ${host} failed to start (exit code ${worker_rc})" >&2
        exit $worker_rc
    fi
done

# ---- 4. Verify cluster status ----
echo "Verifying cluster status..."
export RAY_ADDRESS="${HEAD_IP}:${PORT}"
sleep 3
if ray status; then
    echo "Ray cluster started successfully. RAY_ADDRESS=${RAY_ADDRESS}"
else
    echo "ERROR: 'ray status' failed, cluster may not be healthy." >&2
    exit 1
fi
