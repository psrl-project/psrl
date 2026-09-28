#!/usr/bin/env bash
# Batch rollout for SciBuddy Harbor research tasks using local vLLM (smg_local).
# Override variables before invoking: data_parquet=... num_workers=2 bash ...

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ── Data ─────────────────────────────────────────────────────────────────────
data_parquet=${data_parquet:-examples/scibuddy_rollout/data/scibuddy.parquet}
train_max_samples=${train_max_samples:-4}

# Ray actors resolve relative paths against their own cwd, so absolutise here.
case "$data_parquet" in /*) ;; *) data_parquet="$ROOT/$data_parquet" ;; esac

# ── Model ────────────────────────────────────────────────────────────────────
model_path=${model_path:-${PSRL_WORKSPACE:-}/models/Qwen3.5-4B}
if [ ! -d "$model_path" ]; then
    echo "ERROR: model directory not found: $model_path" >&2
    echo "  Set PSRL_WORKSPACE, or pass model_path=/path/to/model." >&2
    exit 1
fi
tensor_parallel_size=${tensor_parallel_size:-2}
n_instances=${n_instances:-2}
prompt_length=${prompt_length:-2048}
response_length=${response_length:-126976}
max_model_len=${max_model_len:-129024}

# ── Workers ───────────────────────────────────────────────────────────────────
# Nodes with a healthy dockerd, like "[10.0.0.1,10.0.0.2]". A wedged one hangs.
node_ips=${node_ips:-""}
num_workers=${num_workers:-2}
max_in_flight_per_worker=${max_in_flight_per_worker:-2}

# ── Output ────────────────────────────────────────────────────────────────────
output_dir=${output_dir:-outputs/scibuddy_rollout_$(date +%Y%m%d_%H%M%S)}
case "$output_dir" in /*) ;; *) output_dir="$ROOT/$output_dir" ;; esac

# ── Thinking template ─────────────────────────────────────────────────────────
# multi_traj uses TITO forking, which is the right mode for smg_local.
thinking_template=${thinking_template:-multi_traj}

echo "=== SciBuddy batch rollout (smg_local) ==="
echo "  data:    $data_parquet"
echo "  model:   $model_path"
echo "  nodes:   $node_ips"
echo "  output:  $output_dir"
echo ""

PYTHONUNBUFFERED=1 python -m psrl.batch_rollout.main_batch_rollout \
  serving=smg_local \
  batch_rollout.output_dir="$output_dir" \
  batch_rollout.num_workers="$num_workers" \
  batch_rollout.max_in_flight_per_worker="$max_in_flight_per_worker" \
  batch_rollout.progress_interval_s=120 \
  data.train_files="$data_parquet" \
  data.use_multi_dataset=False \
  data.train_batch_size="$train_max_samples" \
  data.train_max_samples="$train_max_samples" \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=False \
  data.truncation=error \
  gen_actor_rollout_ref.model.path="$model_path" \
  gen_actor_rollout_ref.rollout.prompt_length="$prompt_length" \
  gen_actor_rollout_ref.rollout.response_length="$response_length" \
  gen_actor_rollout_ref.rollout.max_model_len="$max_model_len" \
  gen_actor_rollout_ref.rollout.tensor_model_parallel_size="$tensor_parallel_size" \
  gen_actor_rollout_ref.rollout.multi_turn.enable=True \
  gen_actor_rollout_ref.rollout.multi_turn.max_turns=200 \
  gen_actor_rollout_ref.rollout.agent.agent_loop_config_path="$ROOT/examples/scibuddy_rollout/config/scibuddy_agent_config.yaml" \
  gen_actor_rollout_ref.rollout.agent.default_agent_loop=scibuddy \
  ${node_ips:+gen_actor_rollout_ref.rollout.agent.node_ips="$node_ips"} \
  psrl.agentic_rl.thinking_template="$thinking_template" \
  psrl.logging_path="$output_dir/logs" \
  "batch_rollout.serving.n_instances=$n_instances" \
  "batch_rollout.serving.ngpus_per_node_per_instance=$tensor_parallel_size"

echo ""
echo "=== Done. Output at $output_dir ==="
