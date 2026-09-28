#!/usr/bin/env bash
# Batch rollout for SciBuddy tasks through a hosted OpenAI-compatible API.
# Set the key in an env var and pass its NAME, never the literal value.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ── Endpoint ──────────────────────────────────────────────────────────────────
api_base_url=${api_base_url:-https://api.novita.ai/openai/v1}
api_key_env=${api_key_env:-NOVITA_API_KEY}
model_name=${model_name:-openai/gpt-oss-120b}

# Local tokenizer directory. The endpoint serves the model, but the agent loop
# still reads a tokenizer, so this has to resolve even on the API path.
model_path=${model_path:-${PSRL_WORKSPACE:-}/models/Qwen3.5-4B}
if [ ! -d "$model_path" ]; then
    echo "ERROR: tokenizer directory not found: $model_path" >&2
    echo "  Set PSRL_WORKSPACE, or pass model_path=/path/to/model." >&2
    exit 1
fi

if [ -z "${!api_key_env:-}" ]; then
    echo "ERROR: \$${api_key_env} is empty. Export the key first, for example:" >&2
    echo "  export ${api_key_env}=\"\$(cat ~/.novita_key)\"" >&2
    exit 1
fi

# ── Data ──────────────────────────────────────────────────────────────────────
data_parquet=${data_parquet:-examples/scibuddy_rollout/data_synth/scibuddy_synth.parquet}
train_max_samples=${train_max_samples:-1}

# Ray actors resolve relative paths against their own cwd, so absolutise here.
case "$data_parquet" in /*) ;; *) data_parquet="$ROOT/$data_parquet" ;; esac

# ── Workers ───────────────────────────────────────────────────────────────────
# Nodes with a healthy dockerd, like "[10.0.0.1,10.0.0.2]". Empty places anywhere.
node_ips=${node_ips:-""}
num_workers=${num_workers:-1}
max_in_flight_per_worker=${max_in_flight_per_worker:-1}

# ── Budget ────────────────────────────────────────────────────────────────────
# The agent loop reads max_model_len to size the budget it advertises.
max_model_len=${max_model_len:-32768}
max_turns=${max_turns:-30}

# Trajectory-retention policy. See psrl/utils/agent/thinking.py.
thinking_template=${thinking_template:-multi_traj}

output_dir=${output_dir:-outputs/scibuddy_api_$(date +%Y%m%d_%H%M%S)}
case "$output_dir" in /*) ;; *) output_dir="$ROOT/$output_dir" ;; esac

echo "=== SciBuddy batch rollout (openai_api) ==="
echo "  endpoint: $api_base_url"
echo "  model:    $model_name"
echo "  key from: \$$api_key_env"
echo "  output:   $output_dir"
echo ""

# A hosted endpoint ignores the CoT knobs, so this only selects the retention
# policy. Whether `content` arrives non-empty is the model's behavior, not ours.
PYTHONUNBUFFERED=1 python -m psrl.batch_rollout.main_batch_rollout \
  serving=openai_api \
  batch_rollout.serving.api_base_url="$api_base_url" \
  batch_rollout.serving.api_key_env="$api_key_env" \
  batch_rollout.serving.model_name="$model_name" \
  batch_rollout.output_dir="$output_dir" \
  batch_rollout.num_workers="$num_workers" \
  batch_rollout.max_in_flight_per_worker="$max_in_flight_per_worker" \
  batch_rollout.progress_interval_s=60 \
  data.train_files="$data_parquet" \
  data.use_multi_dataset=False \
  data.train_batch_size="$train_max_samples" \
  data.train_max_samples="$train_max_samples" \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  gen_actor_rollout_ref.model.path="$model_path" \
  gen_actor_rollout_ref.rollout.prompt_length=2048 \
  gen_actor_rollout_ref.rollout.response_length=$((max_model_len - 2048)) \
  gen_actor_rollout_ref.rollout.max_model_len="$max_model_len" \
  gen_actor_rollout_ref.rollout.multi_turn.enable=True \
  gen_actor_rollout_ref.rollout.multi_turn.max_turns="$max_turns" \
  gen_actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/examples/scibuddy_rollout/config/scibuddy_agent_config.yaml" \
  gen_actor_rollout_ref.rollout.agent.default_agent_loop=scibuddy \
  ${node_ips:+gen_actor_rollout_ref.rollout.agent.node_ips="$node_ips"} \
  psrl.agentic_rl.thinking_template="$thinking_template" \
  psrl.logging_path="$output_dir/logs"

echo ""
echo "=== Done. Output at $output_dir ==="
