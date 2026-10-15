# MemAgent on PSRL

This example integrates the context-independent MemAgent workflow with PSRL's
session-scoped OpenAI Chat Completion API and TITO training data collection.

Each document chunk is processed as an independent conversation.  All
conversations from one episode share a SessionRouter session (and therefore a
pinned rollout worker/model version). With the default `manual` trajectory strategy,
MemAgent sends ID `0` and records them as one session-local TITO trajectory. With
`psrl.rollout_gateway.trajectory_id_strategy=auto`, the independent conversations
become separate TITO leaves and SessionAgentLoop returns all of them; the final-answer
reward is computed once and broadcast to those outputs.

## Files

- `runner.py`: MemAgent state machine using a normal OpenAI-compatible HTTP endpoint.
- `agent_loop.py`: thin `SessionAgentLoop` adapter that supplies the session API URL.
- `config/mem_agent_loop.yaml`: Hydra agent-loop registration.
- `dataset.py`: optional adapter for raw RULER-HQA JSON validation files.
- `reward.py`: boxed-answer rule reward.
- `eval_ruler_hqa.py`: standalone evaluator.
- `prepare-eval-data.sh`: RULER-HQA data validation/download helper.
- `run-eval.sh`: serve a PSRL HF checkpoint and run RULER-HQA evaluation.
- `run_qwen3-4b.sh`: complete Qwen3-4B Megatron/GRPO training launcher.

## Data

`hotpotqa_train_32k.parquet` already follows PSRL's native dataset contract. It
contains `prompt`, `context`, `reward_model`, `extra_info`, `data_source`, and
`ability`, so it should be used directly without rewriting the parquet file.

Select MemAgent for rows without an `agent_name` column with
`gen_actor_rollout_ref.rollout.agent.default_agent_loop=mem_agent`.

## Training configuration

The provided launcher maps VIME's Qwen3-4B MemAgent settings onto PSRL. It
expects two 8-GPU nodes by default because PSRL allocates rollout and training
workers separately. Model, dataset, and output paths can be overridden through
environment variables. Algorithm and deployment settings are grouped as normal
shell variables in the script, following the other PSRL launch examples;
additional Hydra overrides passed after the script take highest precedence:

```bash
HF_MODEL_PATH=/models/Qwen3-4B \
TRAIN_FILE=/data/hotpotqa_train_32k.parquet \
bash examples/mem_agent/run_qwen3-4b.sh trainer.logger='["console","wandb"]'
```

The MemAgent runtime settings are exposed as `MEM_CHUNK_TOKENS`,
`MEM_MAX_MEMORY`, `MEM_MAX_FINAL`, `MEM_MAX_CHUNKS`, and
`MEM_ALLOW_CONTEXT_TRUNCATION`.

Start from an existing PSRL GRPO/DAPO launch script and set at least:

```text
psrl.rollout_gateway.enable=True
psrl.log_prob.enable_rollout_engine_log_prob=True
gen_actor_rollout_ref.rollout.multi_turn.enable=True
gen_actor_rollout_ref.rollout.multi_turn.max_turns=65
gen_actor_rollout_ref.rollout.agent.agent_loop_config_path=examples/mem_agent/config/mem_agent_loop.yaml
gen_actor_rollout_ref.rollout.agent.default_agent_loop=mem_agent
gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj
gen_actor_rollout_ref.rollout.prompt_length=4096
gen_actor_rollout_ref.rollout.response_length=1024
train_actor_rollout_ref.actor.loss_agg_mode=token-mean
algorithm.adv_estimator=grpo
algorithm.norm_adv_by_std_in_grpo=False
reward.active_managers='[dapo]'
reward.managers.dapo.reward_fn.0.path=examples/mem_agent/reward.py
reward.managers.dapo.reward_fn.0.name=compute_score
data.reward_model_dicts.0.reward_loop_type=dapo
data.reward_model_dicts.0.reward_fn=compute_score
data.return_raw_chat=True
```

`prompt_length` must cover the rendered template, question, memory, and one
chunk.  `response_length` is per independent conversation and must be at least
`max_memory_tokens`; it is not the sum of every turn in an episode.

For the paper-style Multi-Conv objective, use token-mean loss and disable GRPO
standard-deviation normalization as above.  VIME's example converter additionally
divides each episode advantage by its number of conversations; that optional
compatibility weighting is intentionally not applied by this example.

## External agent interface

Every agent-loop constructor receives one immutable `AgentLoopContext` instead
of PSRL's individual worker dependencies. An external agent can subclass
`SessionAgentLoop` and only implement
`run_session(request, api_base_url)`. The base class owns session
creation, routing and version pinning, strategy-aware TITO collection,
reward computation, and cleanup. The external agent keeps its normal
OpenAI-compatible HTTP client and only replaces its API base URL; it never
handles session IDs or PSRL HTTP headers.

## Validation during training

Native HotpotQA parquet can be evaluated directly by setting `data.val_files`:

```text
+data.train_files=/data/hotpotqa_train_32k.parquet
+data.val_files=examples/mem_agent/hotpotqa_dev.parquet
trainer.val_before_train=True
trainer.test_freq=10
```

VIME's RULER-HQA benchmark uses `eval_{length}.json` from
`BytedTsinghua-SIA/hotpotqa`, where `length` is the number of distractor
documents. These raw files contain `input`, `answers`, `context`, and
`num_docs`. They can participate in the same PSRL validation loop without a
conversion step by enabling the dataset adapter:

```text
+data.val_files=/data/hotpotqa/eval_200.json
data.custom_cls.path=examples/mem_agent/dataset.py
data.custom_cls.name=MemAgentDataset
data.filter_overlong_prompts=False
```

For lengths that require more than 64 chunks, increase both
`runtime.max_chunks` in `mem_agent_loop.yaml` and
`rollout.multi_turn.max_turns` to at least `max_chunks + 1`.

The training reward reports `acc`, `f1`, `em`, and `sub_em`, so PSRL validation
tracks the RULER-HQA metrics together with the normal rollout metrics.

## Standalone evaluation

`run-eval.sh` intentionally has one checkpoint input: `MODEL_PATH` must point
directly to the complete HuggingFace export saved by PSRL, normally
`global_step_N/actor/model/huggingface`. It starts `vllm serve`, runs PSRL's
standalone evaluator using `runner.py`, and stops only the server process it
started.

Prepare the RULER-HQA files and evaluate:

```bash
LENGTHS="50 200 800" DATA_DIR=/data/hotpotqa \
  bash examples/mem_agent/prepare-eval-data.sh --download

MODEL_PATH=/outputs/mem_agent/ckpts/mem_agent/qwen3_4b_mem_agent_grpo/global_step_50/actor/model/huggingface \
DATA_ROOT=/data/hotpotqa \
LENGTH="50 200 800" \
  bash examples/mem_agent/run-eval.sh
```

The evaluator can also be called directly when a compatible vLLM server is
already running:

```bash
python -m examples.mem_agent.eval_ruler_hqa \
  --length 200 \
  --data-root /data/hotpotqa \
  --model /models/Qwen3-4B \
  --tokenizer /models/Qwen3-4B \
  --save-dir results/ruler_hqa_200 \
  --save-file Qwen3-4B
```
