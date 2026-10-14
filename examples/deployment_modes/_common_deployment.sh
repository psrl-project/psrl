#!/usr/bin/env bash
# Shared launch body for the five deployment-mode example scripts.
#
# Mode scripts set `PSRL_DEPLOY_*`, source this file, and call `launch_deployment_mode` with Hydra overrides.
#
# Set pool topology and mode with `PSRL_DEPLOY_*`. Model, dataset, validation, and smoke defaults are defined below.
#
# Optional positional args to launch_deployment_mode are forwarded to main_ppo.

_psrl_total_gpus_from_node_spec() {
    local ngpus_spec=$1
    local nnodes=$2
    local total=0
    local value

    if [[ "${ngpus_spec}" =~ ^[0-9]+$ ]]; then
        echo $(( ngpus_spec * nnodes ))
        return
    fi

    if [[ "${ngpus_spec}" =~ ^\[(.*)\]$ ]]; then
        local values=${BASH_REMATCH[1]}
        values=${values// /}
        IFS=',' read -ra _psrl_gpu_values <<< "${values}"
        for value in "${_psrl_gpu_values[@]}"; do
            if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
                echo "Invalid GPU count '${value}' in PSRL_DEPLOY_SHARED_NGPUS=${ngpus_spec}" >&2
                return 1
            fi
            total=$(( total + value ))
        done
        if [[ "${#_psrl_gpu_values[@]}" -ne "${nnodes}" ]]; then
            echo "PSRL_DEPLOY_SHARED_NGPUS=${ngpus_spec} has ${#_psrl_gpu_values[@]} entries, expected PSRL_DEPLOY_SHARED_NNODES=${nnodes}" >&2
            return 1
        fi
        echo "${total}"
        return
    fi

    echo "Invalid PSRL_DEPLOY_SHARED_NGPUS=${ngpus_spec}; expected integer or Hydra list like [4, 8]" >&2
    return 1
}

_psrl_require_choice() {
    local name=$1 value=$2
    shift 2
    local item
    for item in "$@"; do
        if [[ "${value}" == "${item}" ]]; then
            return 0
        fi
    done
    echo "Invalid ${name}=${value}; expected one of: $*" >&2
    return 1
}

_psrl_require_bool() {
    local name=$1 value=$2
    case "${value}" in
        True|False|true|false) return 0 ;;
        *)
            echo "Invalid ${name}=${value}; expected True or False" >&2
            return 1
            ;;
    esac
}

_psrl_is_true() {
    case "${1}" in
        True|true) return 0 ;;
        *) return 1 ;;
    esac
}

_psrl_require_nonneg_int() {
    local name=$1 value=$2
    if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
        echo "Invalid ${name}=${value}; expected a non-negative integer" >&2
        return 1
    fi
}

_psrl_require_pos_int() {
    local name=$1 value=$2
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid ${name}=${value}; expected a positive integer" >&2
        return 1
    fi
}

_psrl_require_unit_interval() {
    local name=$1 value=$2
    awk -v raw="${value}" -v name="${name}" 'BEGIN {
        if (raw !~ /^[0-9]+(\.[0-9]+)?$/) {
            printf "Invalid %s=%s; expected a number in (0, 1].\n", name, raw > "/dev/stderr"
            exit 1
        }
        if (!(raw > 0 && raw <= 1)) {
            printf "Invalid %s=%s; expected a number in (0, 1].\n", name, raw > "/dev/stderr"
            exit 1
        }
    }'
}

_psrl_require_nonneg_number() {
    local name=$1 value=$2
    awk -v raw="${value}" -v name="${name}" 'BEGIN {
        if (raw !~ /^[0-9]+(\.[0-9]+)?$/) {
            printf "Invalid %s=%s; expected a non-negative number.\n", name, raw > "/dev/stderr"
            exit 1
        }
        if (raw < 0) {
            printf "Invalid %s=%s; expected a non-negative number.\n", name, raw > "/dev/stderr"
            exit 1
        }
    }'
}

_psrl_gpus_divisible() {
    local total=$1 size=$2 field=$3
    if (( size <= 0 )); then
        echo "Invalid ${field}=${size}; expected a positive GPU group size" >&2
        return 1
    fi
    if (( total % size != 0 )); then
        echo "GPU total ${total} is not divisible by ${field}=${size}" >&2
        return 1
    fi
}

_psrl_vllm_ep_size_for_model() {
    local model_path=$1
    local tp_size=$2

    python - "${model_path}" "${tp_size}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]) / "config.json"
tp_size = int(sys.argv[2])

with config_path.open(encoding="utf-8") as config_file:
    config = json.load(config_file)

expert_count_keys = {
    "moe_num_experts",
    "n_experts",
    "n_routed_experts",
    "num_experts",
    "num_local_experts",
    "num_routed_experts",
}


def is_moe(value):
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = key.lower()
            if normalized_key in expert_count_keys and isinstance(child, (int, float)) and child > 1:
                return True
            if normalized_key == "moe_intermediate_size" and isinstance(child, (int, float)) and child > 0:
                return True
            if normalized_key in {"architectures", "model_type"} and "moe" in str(child).lower():
                return True
            if is_moe(child):
                return True
    elif isinstance(value, list):
        return any(is_moe(child) for child in value)
    return False


print(tp_size if is_moe(config) else 1)
PY
}

launch_deployment_mode() {
    set -xeuo pipefail

    : "${PSRL_DEPLOY_MODE:?PSRL_DEPLOY_MODE must be set by the mode script}"
    : "${PSRL_DEPLOY_EXPERIMENT:?PSRL_DEPLOY_EXPERIMENT must be set}"
    : "${PSRL_DEPLOY_STALENESS:?PSRL_DEPLOY_STALENESS must be set}"
    : "${PSRL_DEPLOY_RM_ASYNC:?PSRL_DEPLOY_RM_ASYNC must be set}"
    : "${PSRL_DEPLOY_NNODES:?PSRL_DEPLOY_NNODES must be set}"
    : "${PSRL_DEPLOY_TRAIN_NNODES:?PSRL_DEPLOY_TRAIN_NNODES must be set}"
    : "${PSRL_DEPLOY_TRAIN_NGPUS:?PSRL_DEPLOY_TRAIN_NGPUS must be set}"
    : "${PSRL_DEPLOY_SHARED_NNODES:?PSRL_DEPLOY_SHARED_NNODES must be set}"
    : "${PSRL_DEPLOY_SHARED_NGPUS:?PSRL_DEPLOY_SHARED_NGPUS must be set}"
    PSRL_DEPLOY_SMOKE=${PSRL_DEPLOY_SMOKE:-0}
    PSRL_NEED_VALIDATION=${PSRL_NEED_VALIDATION:-0}
    # PSRL_DEPLOY_DATASET=${PSRL_DEPLOY_DATASET:-gsm8k}
    # PSRL_DEPLOY_DATASET=${PSRL_DEPLOY_DATASET:-dapo}
    PSRL_DEPLOY_DATASET=${PSRL_DEPLOY_DATASET:-mixed}
    PSRL_DEPLOY_RM_NUM_REPLICAS=${PSRL_DEPLOY_RM_NUM_REPLICAS:-0}
    PSRL_DEPLOY_TRAINER_READY_GRACE_S=${PSRL_DEPLOY_TRAINER_READY_GRACE_S:-5}
    case "${PSRL_DEPLOY_MODE}" in
        elastic_rl|colocated|rollout_rm_colocated)
            PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN:-True}
            PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE=${PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE:-False}
            ;;
        *)
            PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN:-False}
            PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE=${PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE:-True}
            ;;
    esac
    PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION=${PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION:-0.6}
    PSRL_DEPLOY_VAL_BEFORE_TRAIN=${PSRL_DEPLOY_VAL_BEFORE_TRAIN:-False}
    PSRL_DEPLOY_ROLLOUT_TP=${PSRL_DEPLOY_ROLLOUT_TP:-1}
    PSRL_DEPLOY_RM_TP=${PSRL_DEPLOY_RM_TP:-1}
    PSRL_DEPLOY_EXTRA=${PSRL_DEPLOY_EXTRA:-}
    if [[ -z "${PSRL_DEPLOY_OPTIMIZER_OFFLOAD+x}" ]]; then
        case "${PSRL_DEPLOY_MODE}" in
            colocated|trainer_pool_only|elastic_rl)
                PSRL_DEPLOY_OPTIMIZER_OFFLOAD=True
                ;;
            *)
                PSRL_DEPLOY_OPTIMIZER_OFFLOAD=${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN}
                ;;
        esac
    fi

    _psrl_require_choice PSRL_DEPLOY_MODE "${PSRL_DEPLOY_MODE}" \
        disaggregated colocated rollout_rm_colocated trainer_pool_only elastic_rl
    _psrl_require_choice PSRL_DEPLOY_SMOKE "${PSRL_DEPLOY_SMOKE}" 0 1
    _psrl_require_choice PSRL_NEED_VALIDATION "${PSRL_NEED_VALIDATION}" 0 1
    _psrl_require_choice PSRL_DEPLOY_DATASET "${PSRL_DEPLOY_DATASET}" dapo gsm8k mixed
    _psrl_require_bool PSRL_DEPLOY_RM_ASYNC "${PSRL_DEPLOY_RM_ASYNC}"
    _psrl_require_bool PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN "${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN}"
    _psrl_require_bool PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE "${PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE}"
    _psrl_require_bool PSRL_DEPLOY_VAL_BEFORE_TRAIN "${PSRL_DEPLOY_VAL_BEFORE_TRAIN}"
    _psrl_require_bool PSRL_DEPLOY_OPTIMIZER_OFFLOAD "${PSRL_DEPLOY_OPTIMIZER_OFFLOAD}"
    _psrl_require_nonneg_int PSRL_DEPLOY_STALENESS "${PSRL_DEPLOY_STALENESS}"
    _psrl_require_pos_int PSRL_DEPLOY_NNODES "${PSRL_DEPLOY_NNODES}"
    _psrl_require_pos_int PSRL_DEPLOY_TRAIN_NNODES "${PSRL_DEPLOY_TRAIN_NNODES}"
    _psrl_require_pos_int PSRL_DEPLOY_TRAIN_NGPUS "${PSRL_DEPLOY_TRAIN_NGPUS}"
    _psrl_require_nonneg_int PSRL_DEPLOY_SHARED_NNODES "${PSRL_DEPLOY_SHARED_NNODES}"
    _psrl_require_nonneg_int PSRL_DEPLOY_RM_NUM_REPLICAS "${PSRL_DEPLOY_RM_NUM_REPLICAS}"
    _psrl_require_pos_int PSRL_DEPLOY_ROLLOUT_TP "${PSRL_DEPLOY_ROLLOUT_TP}"
    _psrl_require_pos_int PSRL_DEPLOY_RM_TP "${PSRL_DEPLOY_RM_TP}"
    _psrl_require_nonneg_number PSRL_DEPLOY_TRAINER_READY_GRACE_S "${PSRL_DEPLOY_TRAINER_READY_GRACE_S}"
    _psrl_require_unit_interval PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION "${PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION}"

    if (( PSRL_DEPLOY_TRAIN_NNODES > PSRL_DEPLOY_NNODES )); then
        echo "PSRL_DEPLOY_TRAIN_NNODES=${PSRL_DEPLOY_TRAIN_NNODES} exceeds PSRL_DEPLOY_NNODES=${PSRL_DEPLOY_NNODES}" >&2
        return 1
    fi
    if (( PSRL_DEPLOY_TRAIN_NNODES + PSRL_DEPLOY_SHARED_NNODES > PSRL_DEPLOY_NNODES )); then
        echo "PSRL_DEPLOY_TRAIN_NNODES=${PSRL_DEPLOY_TRAIN_NNODES} + PSRL_DEPLOY_SHARED_NNODES=${PSRL_DEPLOY_SHARED_NNODES} exceeds PSRL_DEPLOY_NNODES=${PSRL_DEPLOY_NNODES}" >&2
        return 1
    fi

    case "${PSRL_DEPLOY_MODE}" in
        colocated)
            if [[ "${PSRL_DEPLOY_STALENESS}" != "0" ]]; then
                echo "PSRL_DEPLOY_MODE=colocated requires PSRL_DEPLOY_STALENESS=0, got ${PSRL_DEPLOY_STALENESS}" >&2
                return 1
            fi
            if ! _psrl_is_true "${PSRL_DEPLOY_RM_ASYNC}"; then
                echo "PSRL_DEPLOY_MODE=colocated requires PSRL_DEPLOY_RM_ASYNC=True" >&2
                return 1
            fi
            if (( PSRL_DEPLOY_SHARED_NNODES != 0 )); then
                echo "PSRL_DEPLOY_MODE=colocated requires PSRL_DEPLOY_SHARED_NNODES=0, got ${PSRL_DEPLOY_SHARED_NNODES}" >&2
                return 1
            fi
            ;;
        rollout_rm_colocated)
            if ! _psrl_is_true "${PSRL_DEPLOY_RM_ASYNC}"; then
                echo "PSRL_DEPLOY_MODE=rollout_rm_colocated requires PSRL_DEPLOY_RM_ASYNC=True" >&2
                return 1
            fi
            if (( PSRL_DEPLOY_SHARED_NNODES < 1 )); then
                echo "PSRL_DEPLOY_MODE=rollout_rm_colocated requires PSRL_DEPLOY_SHARED_NNODES>=1" >&2
                return 1
            fi
            ;;
        elastic_rl)
            if (( PSRL_DEPLOY_SHARED_NNODES < 1 )); then
                echo "PSRL_DEPLOY_MODE=elastic_rl requires PSRL_DEPLOY_SHARED_NNODES>=1" >&2
                return 1
            fi
            ;;
        disaggregated|trainer_pool_only)
            if (( PSRL_DEPLOY_SHARED_NNODES < 1 )); then
                echo "PSRL_DEPLOY_MODE=${PSRL_DEPLOY_MODE} requires PSRL_DEPLOY_SHARED_NNODES>=1" >&2
                return 1
            fi
            ;;
    esac

    case "${PSRL_DEPLOY_MODE}" in
        elastic_rl|colocated|rollout_rm_colocated)
            if ! _psrl_is_true "${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN}" \
                || _psrl_is_true "${PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE}"; then
                echo "Elastic modes require colocated trainer validation and dedicated validation routing." >&2
                return 1
            fi
            ;;
    esac

    if ! _psrl_is_true "${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN}" \
        && ! _psrl_is_true "${PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE}"; then
        echo "PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE must be True when PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN is False" >&2
        return 1
    fi

    case "${PSRL_DEPLOY_MODE}" in
        colocated|trainer_pool_only|elastic_rl)
            if ! _psrl_is_true "${PSRL_DEPLOY_OPTIMIZER_OFFLOAD}"; then
                echo "PSRL_DEPLOY_MODE=${PSRL_DEPLOY_MODE} can sleep the trainer, so PSRL_DEPLOY_OPTIMIZER_OFFLOAD must be True" >&2
                return 1
            fi
            ;;
    esac
    if _psrl_is_true "${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN}" \
        && ! _psrl_is_true "${PSRL_DEPLOY_OPTIMIZER_OFFLOAD}"; then
        echo "PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN=True requires PSRL_DEPLOY_OPTIMIZER_OFFLOAD=True" >&2
        return 1
    fi

    PSRL_WORKSPACE=/apdcephfs_zwfy10/share_303541817/yfzhao/psrl

    if [[ "${PSRL_DEPLOY_SMOKE}" == "1" ]]; then
        experiment_suffix="_smoke"
        total_training_steps=10
        test_freq=-1
        save_freq=-1
        train_prompt_bsz=32
        train_prompt_mini_bsz=8
        n_resp_per_prompt=8
    else
        experiment_suffix=""
        total_training_steps=100
        test_freq=5
        save_freq=-1
        train_prompt_bsz=128
        train_prompt_mini_bsz=32
        n_resp_per_prompt=16
    fi

    if [[ "${PSRL_NEED_VALIDATION}" == "1" ]]; then
        experiment_suffix="${experiment_suffix}_val"
        test_freq=5
    fi

    project_name='psrl_example'
    MODEL_NAME=${PSRL_DEPLOY_MODEL_NAME:-Qwen2.5-1.5B}
    RM_MODEL_NAME=${PSRL_DEPLOY_RM_MODEL_NAME:-GLM-Z1-9B-0414}
    # MODEL_NAME=${PSRL_DEPLOY_MODEL_NAME:-DeepSeek-R1-Distill-Qwen-7B}
    # RM_MODEL_NAME=${PSRL_DEPLOY_RM_MODEL_NAME:-Qwen3-8B}
    experiment_name="${PSRL_DEPLOY_DATASET}_${PSRL_DEPLOY_EXPERIMENT}_${MODEL_NAME}_${RM_MODEL_NAME}${experiment_suffix}"

    source ${PSRL_WORKSPACE}/env/psrl.sh

    HOME=${PSRL_WORKSPACE}
    PSRL_PATH=$(python -c "import psrl; import os; print(os.path.dirname(os.path.dirname(psrl.__file__)))")
    if [[ -z "${LOCAL_IP:-}" ]]; then
        LOCAL_IP=$(hostname --ip-address | awk '{print $1}')
    fi
    if [[ -z "${LOCAL_IP}" ]]; then
        echo "LOCAL_IP is empty. Set LOCAL_IP to the PS manager address." >&2
        return 1
    fi

    HF_MODEL_PATH=${PSRL_DEPLOY_MODEL_PATH:-${PSRL_WORKSPACE}/models/${MODEL_NAME}}
    RM_MODEL_PATH=${PSRL_DEPLOY_RM_MODEL_PATH:-${PSRL_WORKSPACE}/models/${RM_MODEL_NAME}}
    if [[ ! -f "${HF_MODEL_PATH}/config.json" ]]; then
        echo "Rollout model config.json not found under ${HF_MODEL_PATH}" >&2
        return 1
    fi
    if [[ ! -f "${RM_MODEL_PATH}/config.json" ]]; then
        echo "Reward model config.json not found under ${RM_MODEL_PATH}" >&2
        return 1
    fi

    # Dataset specs follow `data.train_datas` / `data.val_datas`.
    GSM8K_TRAIN="${PSRL_WORKSPACE}/data/gsm8k/train.parquet"
    GSM8K_TEST="${PSRL_WORKSPACE}/data/gsm8k/test.parquet"
    DAPO_TRAIN="${PSRL_WORKSPACE}/data/dapo/dapo-math-17k.parquet"
    DAPO_VAL="${PSRL_WORKSPACE}/data/dapo/aime-2024.parquet"

    _DS_NAIVE="reward_fn_key:data_source,reward_loop_type:naive,reward_fn:default,reward_model_name:null,reward_coef:1.0"
    _DS_DAPO="reward_fn_key:data_source,reward_loop_type:dapo,reward_fn:default,reward_model_name:null,reward_coef:1.0"
    _DS_GEN="reward_fn_key:data_source,reward_loop_type:gen,reward_fn:default,reward_model_name:${RM_MODEL_NAME},reward_coef:1.0"

    _ROW_TRAIN_GSM8K="{file:${GSM8K_TRAIN},data_source_name:openai/gsm8k,prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}},{${_DS_GEN}}]}"
    _ROW_TRAIN_DAPO="{file:${DAPO_TRAIN},data_source_name:dapo/dapo-math-17k,prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}},{${_DS_GEN}}]}"

    _ROW_VAL_GSM8K="{file:${GSM8K_TEST},prompt_key:prompt,reward_model_dicts:[{${_DS_NAIVE}}]}"
    _ROW_VAL_DAPO="{file:${DAPO_VAL},prompt_key:prompt,reward_model_dicts:[{${_DS_DAPO}}]}"

    case "${PSRL_DEPLOY_DATASET}" in
        dapo)
            TRAIN_DATAS="[${_ROW_TRAIN_DAPO}]"
            VAL_DATAS="[${_ROW_VAL_DAPO}]"
            TRAIN_DATASETS_RATIOS='[1.0]'
            ;;
        gsm8k)
            TRAIN_DATAS="[${_ROW_TRAIN_GSM8K}]"
            VAL_DATAS="[${_ROW_VAL_GSM8K}]"
            TRAIN_DATASETS_RATIOS='[1.0]'
            ;;
        mixed)
            TRAIN_DATAS="[${_ROW_TRAIN_GSM8K},${_ROW_TRAIN_DAPO}]"
            VAL_DATAS="[${_ROW_VAL_GSM8K},${_ROW_VAL_DAPO}]"
            TRAIN_DATASETS_RATIOS='[0.5,0.5]'
            ;;
        *)
            echo "Invalid PSRL_DEPLOY_DATASET=${PSRL_DEPLOY_DATASET}; expected dapo, gsm8k, or mixed" >&2
            return 1
            ;;
    esac

    # rollout settings
    GEN_TP=${PSRL_DEPLOY_ROLLOUT_TP:-1}
    GEN_PP=1
    GEN_EP=$(_psrl_vllm_ep_size_for_model "${HF_MODEL_PATH}" "${GEN_TP}")
    GEN_NGPUS_PER_NODE_PER_INSTANCE=$(( ${GEN_TP} * ${GEN_PP} ))

    # reward-model rollout settings
    RM_TP=${PSRL_DEPLOY_RM_TP:-2}
    RM_PP=1
    RM_EP=$(_psrl_vllm_ep_size_for_model "${RM_MODEL_PATH}" "${RM_TP}")
    RM_NGPUS_PER_NODE_PER_INSTANCE=$(( ${RM_TP} * ${RM_PP} ))

    # validation settings (on train_pool. does not use elastic path)
    VAL_TP=1
    VAL_PP=1
    VAL_EP=$(_psrl_vllm_ep_size_for_model "${HF_MODEL_PATH}" "${VAL_TP}")
    VAL_NGPUS_PER_NODE_PER_INSTANCE=$(( ${VAL_TP} * ${VAL_PP} ))
    TRAIN_TOTAL_NGPUS=$(( PSRL_DEPLOY_TRAIN_NNODES * PSRL_DEPLOY_TRAIN_NGPUS ))
    SHARED_TOTAL_NGPUS=$(_psrl_total_gpus_from_node_spec "${PSRL_DEPLOY_SHARED_NGPUS}" "${PSRL_DEPLOY_SHARED_NNODES}")
    GEN_INSTANCE_GPUS=$(( GEN_TP * GEN_PP ))
    RM_INSTANCE_GPUS=$(( RM_TP * RM_PP ))
    VAL_INSTANCE_GPUS=$(( VAL_TP * VAL_PP ))
    _psrl_gpus_divisible "${TRAIN_TOTAL_NGPUS}" "${VAL_INSTANCE_GPUS}" VAL_TP
    VAL_INSTANCES=$(( TRAIN_TOTAL_NGPUS / VAL_INSTANCE_GPUS ))
    if (( VAL_INSTANCES < 1 )); then
        echo "Computed n_validate_instances=${VAL_INSTANCES} from train GPUs ${TRAIN_TOTAL_NGPUS} / ${VAL_INSTANCE_GPUS}" >&2
        return 1
    fi
    if [[ "${PSRL_DEPLOY_MODE}" != "colocated" ]]; then
        _psrl_gpus_divisible "${SHARED_TOTAL_NGPUS}" "${GEN_INSTANCE_GPUS}" GEN_TP
        _psrl_gpus_divisible "${SHARED_TOTAL_NGPUS}" "${RM_INSTANCE_GPUS}" RM_TP
        if (( SHARED_TOTAL_NGPUS < 1 )); then
            echo "PSRL_DEPLOY_MODE=${PSRL_DEPLOY_MODE} requires a non-empty shared pool, got ${SHARED_TOTAL_NGPUS} GPUs" >&2
            return 1
        fi
    fi

    sp_size=1
    fsdp_size=${PSRL_DEPLOY_TRAIN_NGPUS}
    use_dynamic_bsz=True

    adv_estimator=grpo
    use_kl_in_reward=False
    kl_coef=0.0
    use_kl_loss=False
    kl_loss_coef=0.0
    clip_ratio_low=0.2
    clip_ratio_high=0.28
    max_prompt_length=$((1024 * 1))
    max_response_length=$((1024 * 15))
    max_num_batched_tokens=$((1024 * 16))
    packing_length=$(((max_prompt_length + max_response_length) * 2))
    enable_overlong_buffer=True
    overlong_buffer_len=$((1024 * 5))
    overlong_penalty_factor=1.0
    loss_agg_mode="token-mean"


    temperature=1.0
    top_p=1.0
    top_k=-1
    val_top_p=0.7
    filter_groups_metric=acc

    rollout_is="token"
    rollout_is_threshold=2.0

    offload=${PSRL_DEPLOY_OPTIMIZER_OFFLOAD}

    REWARD_MODELS=(
        reward.active_managers='[naive,dapo,gen_qwen3_8b]'
        reward.reward_normalization=none
        reward.managers.naive.reward_loop_type=naive
        reward.managers.dapo.reward_loop_type=dapo
        reward.managers.dapo.reward_kwargs.max_resp_len=10240
        reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.enable=True
        reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.len=5120
        reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
        reward.managers.dapo.reward_kwargs.overlong_buffer_cfg.log=False
        reward.managers.gen_qwen3_8b.reward_loop_type=gen
        reward.managers.gen_qwen3_8b.reward_model_name=${RM_MODEL_NAME}
        reward.managers.gen_qwen3_8b.enable_resource_pool=True
        reward.managers.gen_qwen3_8b.rollout_ngpus_per_instance_per_node=${RM_NGPUS_PER_NODE_PER_INSTANCE}
        reward.managers.gen_qwen3_8b.rollout_nnodes_per_instance=1
        reward.managers.gen_qwen3_8b.max_concurrent_requests_per_instance=128
        reward.managers.gen_qwen3_8b.model.path=${RM_MODEL_PATH}
        reward.managers.gen_qwen3_8b.model.use_shm=False
        reward.managers.gen_qwen3_8b.model.trust_remote_code=False
        reward.managers.gen_qwen3_8b.rollout._target_=psrl.workers.config.RolloutConfig
        reward.managers.gen_qwen3_8b.rollout.name=vllm
        reward.managers.gen_qwen3_8b.rollout.mode=psrl_async
        reward.managers.gen_qwen3_8b.rollout.disable_attn=False
        reward.managers.gen_qwen3_8b.rollout.dtype=bfloat16
        reward.managers.gen_qwen3_8b.rollout.gpu_memory_utilization=0.8
        reward.managers.gen_qwen3_8b.rollout.enforce_eager=true
        reward.managers.gen_qwen3_8b.rollout.free_cache_engine=true
        reward.managers.gen_qwen3_8b.rollout.data_parallel_size=1
        reward.managers.gen_qwen3_8b.rollout.expert_parallel_size=${RM_EP}
        reward.managers.gen_qwen3_8b.rollout.tensor_model_parallel_size=${RM_TP}
        reward.managers.gen_qwen3_8b.rollout.pipeline_model_parallel_size=${RM_PP}
        reward.managers.gen_qwen3_8b.rollout.max_num_batched_tokens=$((1024 * 40))
        reward.managers.gen_qwen3_8b.rollout.max_num_seqs=1024
        reward.managers.gen_qwen3_8b.rollout.enable_chunked_prefill=false
        reward.managers.gen_qwen3_8b.rollout.enable_prefix_caching=false
        +reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.async_scheduling=false
        # GLM-Z1 native context is 32k. YaRN factor 1.25 extrapolates to 40k without editing config.json.
        # vLLM Transformers 5 reads rope_type. keep type for the official GLM docs field name.
        +reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.type=yarn
        +reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.rope_type=yarn
        +reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.factor=1.25
        +reward.managers.gen_qwen3_8b.rollout.engine_kwargs.vllm.hf_overrides.rope_scaling.original_max_position_embeddings=32768
        reward.managers.gen_qwen3_8b.rollout.disable_log_stats=false
        reward.managers.gen_qwen3_8b.rollout.skip_tokenizer_init=false
        reward.managers.gen_qwen3_8b.rollout.prompt_length=$((1024 * 16))
        reward.managers.gen_qwen3_8b.rollout.response_length=$((1024 * 20))
        reward.managers.gen_qwen3_8b.rollout.max_model_len=$((1024 * 40))
        reward.managers.gen_qwen3_8b.rollout.runner=generate
        reward.managers.gen_qwen3_8b.rollout.task=generate
        reward.managers.gen_qwen3_8b.sampling_config.temperature=1.0
        reward.managers.gen_qwen3_8b.sampling_config.top_p=1.0
        reward.managers.gen_qwen3_8b.sampling_config.top_k=-1
    )

    mkdir -p "${PSRL_WORKSPACE}/logs/${project_name}"

    # ---- Mode-derived deployment args ----
    DEPLOY_ARGS=(
        +psrl.deployment.mode=${PSRL_DEPLOY_MODE}
        psrl.staleness=${PSRL_DEPLOY_STALENESS}
        reward.launch_reward_fn_async=${PSRL_DEPLOY_RM_ASYNC}
        psrl.deployment.elastic.shared_nnodes=${PSRL_DEPLOY_SHARED_NNODES}
        "psrl.deployment.elastic.shared_ngpus_per_node=${PSRL_DEPLOY_SHARED_NGPUS}"
        psrl.deployment.elastic.trainer_ready_grace_s=${PSRL_DEPLOY_TRAINER_READY_GRACE_S}
        psrl.deployment.train_nnodes=${PSRL_DEPLOY_TRAIN_NNODES}
        psrl.deployment.train_ngpus_per_node=${PSRL_DEPLOY_TRAIN_NGPUS}
        psrl.deployment.total_nnodes=${PSRL_DEPLOY_NNODES}
        psrl.deployment.rollout_nnodes_per_instance=1
        psrl.deployment.rollout_ngpus_per_node_per_instance=${GEN_NGPUS_PER_NODE_PER_INSTANCE}
        psrl.deployment.n_validate_instances=${VAL_INSTANCES}
        psrl.deployment.validate_nnodes_per_instance=1
        psrl.deployment.validate_ngpus_per_node_per_instance=${VAL_NGPUS_PER_NODE_PER_INSTANCE}
    )

    # Independent pools require explicit rollout and RM replica counts. Elastic modes derive counts from pool sizes.
    case "${PSRL_DEPLOY_MODE}" in
        disaggregated)
            N_ROLLOUT_INSTANCES=$(( SHARED_TOTAL_NGPUS / GEN_INSTANCE_GPUS ))
            if (( N_ROLLOUT_INSTANCES < 1 )); then
                echo "Computed n_rollout_instances=${N_ROLLOUT_INSTANCES} from shared GPUs ${SHARED_TOTAL_NGPUS} / ${GEN_INSTANCE_GPUS}" >&2
                return 1
            fi
            DEPLOY_ARGS+=(
                psrl.deployment.n_rollout_instances=${N_ROLLOUT_INSTANCES}
                psrl.deployment.elastic.enable=False
                psrl.deployment.elastic.enable_trainer_pool=False
            )
            if [[ "${PSRL_DEPLOY_RM_NUM_REPLICAS}" != "0" ]]; then
                DEPLOY_ARGS+=( reward.managers.gen_qwen3_8b.num_replicas=${PSRL_DEPLOY_RM_NUM_REPLICAS} )
            fi
            ;;
        trainer_pool_only)
            N_ROLLOUT_INSTANCES=$(( SHARED_TOTAL_NGPUS / GEN_INSTANCE_GPUS ))
            if (( N_ROLLOUT_INSTANCES < 1 )); then
                echo "Computed n_rollout_instances=${N_ROLLOUT_INSTANCES} from shared GPUs ${SHARED_TOTAL_NGPUS} / ${GEN_INSTANCE_GPUS}" >&2
                return 1
            fi
            DEPLOY_ARGS+=(
                psrl.deployment.n_rollout_instances=${N_ROLLOUT_INSTANCES}
                psrl.deployment.elastic.enable=False
                psrl.deployment.elastic.enable_trainer_pool=False
                psrl.deployment.elastic.enable_policy=False
            )
            if [[ "${PSRL_DEPLOY_RM_NUM_REPLICAS}" != "0" ]]; then
                DEPLOY_ARGS+=( reward.managers.gen_qwen3_8b.num_replicas=${PSRL_DEPLOY_RM_NUM_REPLICAS} )
            fi
            ;;
        colocated)
            DEPLOY_ARGS+=(
                psrl.deployment.elastic.enable=True
                psrl.deployment.elastic.enable_trainer_pool=True
                psrl.deployment.elastic.enable_policy=False
            )
            ;;
        rollout_rm_colocated)
            DEPLOY_ARGS+=(
                psrl.deployment.elastic.enable=True
                psrl.deployment.elastic.enable_trainer_pool=False
                psrl.deployment.elastic.enable_policy=False
            )
            ;;
        elastic_rl)
            DEPLOY_ARGS+=( psrl.deployment.elastic.enable=True )
            ;;
        *)
            echo "Unknown PSRL_DEPLOY_MODE=${PSRL_DEPLOY_MODE}" >&2
            return 1
            ;;
    esac

    # Modes 1-4 use simple routing with a practically unbounded preemption
    # waiting cap. Mode 5 keeps the stricter cap required by elastic scaling.
    case "${PSRL_DEPLOY_MODE}" in
        elastic_rl)
            ROLLOUT_COST_MODEL=${PSRL_PATH}/psrl/trainer/config/cost_model/qwen2.5_1.5b.json
            if [[ ! -f "${ROLLOUT_COST_MODEL}" ]]; then
                echo "Rollout cost model not found: ${ROLLOUT_COST_MODEL}" >&2
                return 1
            fi
            ROUTING_ARGS=(
                psrl.rollout_coordination.routing_strategy.method=throughput_optimal
                psrl.rollout_coordination.routing_strategy.candidate_sort_indicator=reserve_capability
                psrl.rollout_coordination.routing_strategy.enable_multi_priority_queue=True
                psrl.rollout_coordination.routing_strategy.cost_model_path=${ROLLOUT_COST_MODEL}
                psrl.rollout_coordination.routing_strategy.delta_throughput_threshold=0.2
                psrl.rollout_coordination.routing_strategy.request_budget=1024
                psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=3
                psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance=512
            )
            ;;
        disaggregated|colocated|rollout_rm_colocated|trainer_pool_only)
            ROUTING_ARGS=(
                psrl.rollout_coordination.routing_strategy.method=round_robin
                psrl.rollout_coordination.routing_strategy.candidate_sort_indicator=version
                psrl.rollout_coordination.routing_strategy.enable_multi_priority_queue=False
                psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption=10000
                ++gen_actor_rollout_ref.rollout.use_psrl_scheduler=True
                ++reward.managers.gen_qwen3_8b.routing_strategy.method=round_robin
                ++reward.managers.gen_qwen3_8b.rollout.use_psrl_scheduler=True
            )
            ;;
    esac

    # Extra mode-specific overrides (space-separated string) split into args.
    if [[ -n "${PSRL_DEPLOY_EXTRA}" ]]; then
        # shellcheck disable=SC2206
        DEPLOY_ARGS+=( ${PSRL_DEPLOY_EXTRA} )
    fi

    PYTHONUNBUFFERED=1 python -m psrl.trainer.main_ppo \
        psrl.ps_manager_ip=${LOCAL_IP} \
        psrl.reward_service_ip=${LOCAL_IP} \
        psrl.rollout_n=${n_resp_per_prompt} \
        psrl.staleness_buffer_entries=${train_prompt_bsz} \
        psrl.ps_mode=nixl_cpu \
        psrl.logging_path=${PSRL_PATH}/logs/${project_name}/${experiment_name} \
        \
        "${DEPLOY_ARGS[@]}" \
        \
        psrl.log_prob.enable_rollout_engine_log_prob=True \
        psrl.colocate_validate_and_train=${PSRL_DEPLOY_COLOCATE_VALIDATE_AND_TRAIN} \
        psrl.fuse_rollout_with_validate=${PSRL_DEPLOY_FUSE_ROLLOUT_WITH_VALIDATE} \
        \
        psrl.nixl.server_port=23456 \
        psrl.group_post_process.enable=False \
        psrl.group_post_process.name=dynamic_sampling_filter \
        psrl.rollout_coordination.redundant_rollout.enable=False \
        psrl.rollout_coordination.partial_rollout.enable=True \
        \
        "${ROUTING_ARGS[@]}" \
        \
        gen_actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        gen_actor_rollout_ref.rollout.expert_parallel_size=${GEN_EP} \
        gen_actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP} \
        gen_actor_rollout_ref.rollout.pipeline_model_parallel_size=${GEN_PP} \
        gen_actor_rollout_ref.rollout.enable_chunked_prefill=True \
        gen_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
        gen_actor_rollout_ref.rollout.temperature=${temperature} \
        gen_actor_rollout_ref.rollout.top_p=${top_p} \
        gen_actor_rollout_ref.rollout.top_k=${top_k} \
        gen_actor_rollout_ref.rollout.disable_log_stats=false \
        \
        train_actor_rollout_ref.model.path="$HF_MODEL_PATH" \
        train_actor_rollout_ref.model.use_remove_padding=True \
        train_actor_rollout_ref.model.enable_gradient_checkpointing=True \
        train_actor_rollout_ref.rollout.max_num_batched_tokens=${packing_length} \
        train_actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
        train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        train_actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_num_batched_tokens} \
        train_actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
        train_actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        train_actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
        train_actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
        train_actor_rollout_ref.rollout.val_kwargs.n=1 \
        train_actor_rollout_ref.rollout.expert_parallel_size=${VAL_EP} \
        train_actor_rollout_ref.rollout.tensor_model_parallel_size=${VAL_TP} \
        train_actor_rollout_ref.rollout.gpu_memory_utilization=${PSRL_DEPLOY_VAL_GPU_MEMORY_UTILIZATION} \
        train_actor_rollout_ref.actor.strategy=fsdp2 \
        train_actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        train_actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        train_actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
        train_actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
        train_actor_rollout_ref.actor.clip_ratio_c=10.0 \
        train_actor_rollout_ref.actor.optim.lr=1e-6 \
        train_actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        train_actor_rollout_ref.actor.optim.weight_decay=0.1 \
        train_actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
        train_actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_num_batched_tokens} \
        train_actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
        train_actor_rollout_ref.actor.fsdp_config.param_offload=False \
        train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        train_actor_rollout_ref.actor.entropy_coeff=0 \
        train_actor_rollout_ref.actor.grad_clip=1.0 \
        train_actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
        \
        "${REWARD_MODELS[@]}" \
        \
        data.truncation='left' \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.return_raw_chat=True \
        data.train_batch_size=${train_prompt_bsz} \
        data.train_datas="${TRAIN_DATAS}" \
        data.val_datas="${VAL_DATAS}" \
        data.train_datasets_ratios="${TRAIN_DATASETS_RATIOS}" \
        \
        algorithm.adv_estimator=${adv_estimator} \
        algorithm.use_kl_in_reward=${use_kl_in_reward} \
        algorithm.kl_ctrl.kl_coef=${kl_coef} \
        algorithm.rollout_correction.rollout_is=${rollout_is} \
        algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
        psrl.rollout_coordination.proactive_filter_strategy.method="retry" \
        psrl.rollout_coordination.proactive_filter_strategy.threshold=0 \
        \
        trainer.logger='["console", "wandb"]' \
        +trainer.wandb_proxy=http://star-proxy.oa.com:3128 \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${experiment_name}" \
        trainer.val_before_train=${PSRL_DEPLOY_VAL_BEFORE_TRAIN} \
        trainer.test_freq=${test_freq} \
        trainer.save_freq=${save_freq} \
        trainer.total_epochs=1 \
        trainer.total_training_steps=${total_training_steps} \
        "$@" \
        2>&1 | tee ${PSRL_WORKSPACE}/logs/${experiment_name}_${LOCAL_IP}.log
}
