#!/bin/bash
env_file="${PSRL_WORKSPACE}/env/env.sh"
source ${env_file}

HOSTFILE=${1:-""}
PORT=8887                # Ray node communication port
DASHBOARD_PORT=8265      # Ray Dashboard port

# Read host list
if [ -n "${HOSTFILE}" ]; then
    mapfile -t hosts < "${HOSTFILE}"
    if [ ${#hosts[@]} -eq 0 ]; then
        echo "Error: Empty hostfile"
        exit 1
    fi
else
    mapfile -t hosts < <(echo "$NODE_IP_LIST" | sed "s/:.//g; s/,/\\n/g" | head -n $NODE_NUM)
    if [ ${#hosts[@]} -eq 0 ]; then
        echo "Error: NODE_IP_LIST is empty"
        exit 1
    fi
fi

HEAD_IP=${hosts[0]}
workers=( "${hosts[@]:1}" )

# unset http_proxy && \
# unset https_proxy && \

echo "Stopping any existing Ray processes on all nodes..."
for host in "${hosts[@]}"; do
    pssh -H "${host}" -i "source ${env_file} && ray stop --force 2>/dev/null || true" &
done
wait
echo "All nodes cleaned up."

# Start head node
echo "Starting Head node at ${HEAD_IP}"
pssh -H "${HEAD_IP}" -i \
    "source ${env_file} && \
    cd ${PSRL_WORKSPACE} && \
    ray start --head \
    --port=${PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${DASHBOARD_PORT} \
    --num-cpus=32"

# Start worker nodes
if [ ${#workers[@]} -gt 0 ]; then
    echo "Starting ${#workers[@]} Worker nodes"
    pssh -H "${workers[*]}" -i \
        "source ${env_file} && \
        cd ${PSRL_WORKSPACE} && \
        ray start --address=${HEAD_IP}:${PORT} \
        --num-cpus=32"
fi
