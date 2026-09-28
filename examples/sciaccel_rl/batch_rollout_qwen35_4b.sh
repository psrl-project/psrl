#!/usr/bin/env bash
# Collect SciAccel-RL rollouts from a served model, with no training.
#
# Reuses the RL recipe's dataset, agent loop, and Harbor wiring unchanged. Every
# training argument is dropped: no optimizer, no reward, no parameter server.
# Score the dump afterwards with `reward.py::compute_score`, see the README.

set -xeuo pipefail

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RAY_prestart_worker_first_driver=false
export RAY_memory_monitor_refresh_ms=0

# Keep Ray sockets within the Unix path limit and off the shared filesystem.
export TMPDIR=${SCIACCEL_TMPDIR:-/tmp}
mkdir -p "${TMPDIR}"

PSRL_PATH=${PSRL_PATH:-$(python3 -c "import os, psrl; print(os.path.dirname(os.path.dirname(psrl.__file__)))")}

# --- Model and data ---

# Name the agent sends, and what each record is stamped with. A `:free` OpenRouter
# id is rate limited hard, so keep MAX_IN_FLIGHT low when using one.
MODEL_NAME=${MODEL_NAME:-nvidia/nemotron-3-super-120b-a12b:free}
# Only read for the tokenizer and chat template. No weights are loaded here,
# because the model is served by SERVING_API_BASE.
HF_MODEL_PATH=${HF_MODEL_PATH:-${PSRL_WORKSPACE:-}/models/Qwen3.5-4B}
# `L1` adds file, line, and defect note, `L2` drops the line, `L3` is unhinted.
HINT_LEVEL=${HINT_LEVEL:-L1}
DATA_DIR=${DATA_DIR:-${PSRL_PATH}/examples/sciaccel_rl/data/pluto-cooling-chemistry/repair_easy}
train_files=${TRAIN_FILES:-${DATA_DIR}/train/${HINT_LEVEL}.parquet}

# An OpenAI-compatible endpoint. A hosted API, a subscription proxy, or a local
# vLLM fleet started first with `python -m psrl.eval.serve` (pass its
# endpoints.json as SERVING_ENDPOINTS_FILE instead).
SERVING_API_BASE=${SERVING_API_BASE:-https://openrouter.ai/api/v1}
SERVING_ENDPOINTS_FILE=${SERVING_ENDPOINTS_FILE:-}
# Name of the env var holding the bearer token, never the token itself: a Hydra
# override lands in shell history, the Ray dashboard, and the run log.
API_KEY_ENV=${API_KEY_ENV:-OPENROUTER_API_KEY}
if [[ -z "${SERVING_API_BASE}" && -z "${SERVING_ENDPOINTS_FILE}" ]]; then
    echo "ERROR: set SERVING_API_BASE (an OpenAI /v1 URL) or SERVING_ENDPOINTS_FILE." >&2
    exit 1
fi
if [[ -n "${API_KEY_ENV}" && -z "${!API_KEY_ENV:-}" ]]; then
    echo "ERROR: ${API_KEY_ENV} is unset. Export it, or set API_KEY_ENV='' for a" >&2
    echo "local server that ignores the token. Check it first with check_api.sh." >&2
    exit 1
fi
if [[ ! -d "${HF_MODEL_PATH}" ]]; then
    echo "ERROR: model directory not found: ${HF_MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -f "${train_files}" ]]; then
    echo "ERROR: parquet not found: ${train_files}" >&2
    echo "Build it: bash examples/sciaccel_rl/prepare/prepare_all.sh --repo <sciaccel-rl> --envs <env>" >&2
    exit 1
fi

# --- Run ---

# `<env>_<category>_<tier>` from the last two path segments, because every env's
# dataset dir ends in the same `repair_easy` and the basename alone would collide.
dataset_tag=$(basename "$(dirname "${DATA_DIR}")")_$(basename "${DATA_DIR}")
run_name=${RUN_NAME:-rollout-${MODEL_NAME}-${dataset_tag}-${HINT_LEVEL}}
OUTPUT_DIR=${OUTPUT_DIR:-${PSRL_PATH}/examples/sciaccel_rl/rollouts/${run_name}}
mkdir -p "${OUTPUT_DIR}"

agent_loop_config_path=${PSRL_PATH}/examples/sciaccel_rl/config/sciaccel_agent_config.yaml

# --- Sequence lengths and turns ---

max_prompt_length=2048
max_response_length=${MAX_RESPONSE_LENGTH:-65536}
# NOTE(lhy): This reaches terminus-2 as `max_input_tokens`, so any headroom is
# budget the agent spends before the verifier ever runs.
max_model_len=$(( max_prompt_length + max_response_length ))
# Bounded by the response budget: at ~1100 response tokens per turn, 50 turns
# already spends 55k of 65536.
max_turns=${MAX_TURNS:-50}

# Trajectories per task. >1 samples the same task repeatedly.
rollout_n=${ROLLOUT_N:-1}

# --- Concurrency ---

# Nodes allowed to host agent loop workers, and therefore Docker containers. A
# node with a degraded daemon accepts actors and then hangs. Empty means every node.
AGENT_NODE_IPS=${AGENT_NODE_IPS:-}
if [ -n "${AGENT_NODE_IPS}" ]; then
    NUM_WORKERS=${NUM_WORKERS:-$(awk -F, '{print NF}' <<< "${AGENT_NODE_IPS}")}
else
    NUM_WORKERS=${NUM_WORKERS:-2}
fi

# Concurrent episodes per worker. Must not exceed `max_concurrent_episodes` in
# the agent config, which is what actually bounds containers.
MAX_IN_FLIGHT=${MAX_IN_FLIGHT:-4}

# How chain-of-thought carries across turns. A hosted endpoint ignores these
# knobs, so on that path this only selects the retention policy.
thinking_template=${thinking_template:-multi_traj}
if [ "${thinking_template}" = "multi_thinking" ]; then
    chat_template_path=${PSRL_PATH}/examples/sciaccel_rl/config/qwen35_acc_thinking.jinja2
    chat_template_arg="+gen_actor_rollout_ref.rollout.chat_template=${chat_template_path}"
else
    chat_template_arg=""
fi

PYTHONUNBUFFERED=1 python3 -m psrl.batch_rollout.main_batch_rollout \
    serving=openai_api \
    ${SERVING_API_BASE:+batch_rollout.serving.api_base_url=${SERVING_API_BASE}} \
    ${SERVING_ENDPOINTS_FILE:+batch_rollout.serving.endpoints_file=${SERVING_ENDPOINTS_FILE}} \
    ${API_KEY_ENV:+batch_rollout.serving.api_key_env=${API_KEY_ENV}} \
    batch_rollout.serving.model_name=${MODEL_NAME} \
    batch_rollout.output_dir=${OUTPUT_DIR} \
    batch_rollout.num_workers=${NUM_WORKERS} \
    batch_rollout.max_in_flight_per_worker=${MAX_IN_FLIGHT} \
    `# Skips uids already in rollout.jsonl, so a killed run resumes where it stopped.` \
    batch_rollout.resume=${RESUME:-True} \
    \
    psrl.logging_path=${OUTPUT_DIR}/logs \
    psrl.rollout_n=${rollout_n} \
    psrl.agentic_rl.thinking_template=${thinking_template} \
    psrl.agentic_rl.trajectory_output.enable=True \
    psrl.agentic_rl.turn_output.enable=True \
    \
    gen_actor_rollout_ref.model.path=${HF_MODEL_PATH} \
    gen_actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    gen_actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    gen_actor_rollout_ref.rollout.response_length=${max_response_length} \
    gen_actor_rollout_ref.rollout.temperature=${TEMPERATURE:-1.0} \
    gen_actor_rollout_ref.rollout.top_p=${TOP_P:-1.0} \
    gen_actor_rollout_ref.rollout.top_k=-1 \
    gen_actor_rollout_ref.rollout.multi_turn.enable=True \
    gen_actor_rollout_ref.rollout.multi_turn.max_turns=${max_turns} \
    ${chat_template_arg} \
    gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=${agent_loop_config_path} \
    gen_actor_rollout_ref.rollout.agent.default_agent_loop=sciaccel \
    ${AGENT_NODE_IPS:+gen_actor_rollout_ref.rollout.agent.node_ips=[${AGENT_NODE_IPS}]} \
    \
    data.train_files=${train_files} \
    data.use_multi_dataset=False \
    data.train_batch_size=${BATCH_SIZE:-8} \
    data.train_max_samples=${MAX_SAMPLES:--1} \
    data.prompt_key=prompt \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    "$@" 2>&1 | tee -a "${OUTPUT_DIR}/${run_name}.log"
