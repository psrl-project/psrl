#!/usr/bin/env bash
# Shared model and runtime settings for Qwen2.5-7B rollout (TP1) with
# DeepSeek-R1-Distill-Qwen-32B reward model (TP4).

HETERO_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
# shellcheck disable=SC1091
source "${HETERO_SCRIPT_DIR}/../deployment_modes/_common_deployment.sh"

export PSRL_DEPLOY_MODEL_NAME=Qwen2.5-7B
export PSRL_DEPLOY_RM_MODEL_NAME=Qwen3-30B-A3B-Thinking-2507
export PSRL_DEPLOY_ROLLOUT_TP=1
export PSRL_DEPLOY_RM_TP=4

launch_qwen7b_rm32b_deployment_mode() {
    launch_deployment_mode \
        reward.managers.gen_qwen3_8b.rollout.gpu_memory_utilization=0.65 \
        reward.managers.gen_qwen3_8b.rollout.max_num_seqs=512 \
        "$@"
}
