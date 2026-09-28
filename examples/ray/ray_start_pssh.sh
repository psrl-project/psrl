#!/bin/bash
env_file="/data_storage/yl_test/lhy/psrl.sh"
source ${env_file}

HOSTFILE=${1:-""}
PORT=8887                # Ray node communication port
DASHBOARD_PORT=8265      # Ray Dashboard port

CONDA_SH=/data_storage/yl_test/lhy/anaconda3/etc/profile.d/conda.sh
ENV_NAME=psrl-new

pssh_run() {   
while read -r line; do
    h=$(echo "$line" | awk '{print $1}')
    [ -z "$h" ] && continue
    case "$h" in \#*) continue;; esac
    echo "[$h] $1"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$h" \
        "source $CONDA_SH && conda activate $ENV_NAME && $1" &
done < "$HOSTFILE"
wait
}

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
    pssh_run -H "${host}" -i "source ${env_file} && ray stop --force 2>/dev/null || true" &
done
wait
echo "All nodes cleaned up."

# Start head node
echo "Starting Head node at ${HEAD_IP}"
pssh_run -H "${HEAD_IP}" -i \
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
    pssh_run -H "${workers[*]}" -i \
        "source ${env_file} && \
        cd ${PSRL_WORKSPACE} && \
        ray start --address=${HEAD_IP}:${PORT} \
        --num-cpus=32"
fi
