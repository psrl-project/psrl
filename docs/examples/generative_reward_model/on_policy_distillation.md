# On-Policy Distillation

On-Policy Distillation (OPD) trains a student on teacher log probabilities for
responses sampled by the student. PSRL manages each teacher as a reward model,
sharing its resource pools, SMG gateway, concurrency limits, sleep and wake-up
lifecycle. Teachers return token targets and a scalar reward of zero.

PSRL's losses and per-trajectory teacher selection follow
[veRL's implementation](https://github.com/verl-project/verl/tree/e5ca4acb/verl/trainer/distillation).
The algorithm terminology follows [veRL's OPD guide](https://verl.readthedocs.io/en/latest/algo/opd.html),
by Jacob Helwig. For more details, please check
[veRL's documentation further](https://verl.readthedocs.io/en/latest/algo/opd.html).

## Requirements

- Install PSRL's pinned veRL patches and native SMG from `psrl-project/smg`,
  branch `psrl-dev`, using `scripts/install_basic.sh`. Rebuild an existing SMG
  checkout with `bash scripts/reinstall_smg.sh` after updating it.
- SMG must support `/generate` prompt logprobs, `prompt_token_ids`, and raw
  `input_top_logprobs`. Top-k requires the latter. Protocol fields are validated
  on every request, and unsupported servers fail explicitly.
- Use a vLLM teacher with the same token ID vocabulary as the student.
  Token IDs are passed directly, without decoding or retokenization.
- Reserve teacher context for the complete student prompt and response **plus
  one output token**. OPD rejects overlong inputs instead of truncating them.
- Use text trajectories and `gen_actor_rollout_ref.rollout.agent.traj_reward_mode=traj`.
  Single-turn and multi-turn rollouts are supported. Tool observations remain in
  teacher context and are excluded from training through `response_mask`.
  Multimodal teacher scoring is unsupported.

## Configure a Teacher Reward Model

The following overlay uses the existing `gen_qwen3_8b` preset as a configuration
label. Set `PSRL_WORKSPACE` to the directory containing your local `models/` and
`data/` directories. Merge the overlay into the `ppo_trainer` configuration.

```yaml
reward:
  active_managers: [naive, gen_qwen3_8b]
  managers:
    gen_qwen3_8b:
      reward_loop_type: opd
      reward_manager:
        name: opd
      reward_model_name: teacher
      enable_resource_pool: true
      nnodes: 1
      n_gpus_per_node: 1
      num_replicas: 1
      rollout_nnodes_per_instance: 1
      rollout_ngpus_per_instance_per_node: 1
      model:
        path: ${oc.env:PSRL_WORKSPACE}/models/teacher
      rollout:
        name: vllm
        runner: generate
        tensor_model_parallel_size: 1
        prompt_length: 8192
        response_length: 1
        max_model_len: 8193

data:
  train_datas:
    - file: ${oc.env:PSRL_WORKSPACE}/data/gsm8k/train.parquet
      prompt_key: prompt
      reward_model_dicts:
        - reward_loop_type: opd
          reward_fn: default
          reward_model_name: teacher
          teacher_weight: 1.0
        - reward_loop_type: naive
          reward_fn: default
          reward_coef: 1.0
```

When distillation is enabled, every training trajectory must select exactly one
active teacher through `(reward_loop_type, reward_fn, reward_model_name)`.
Missing, duplicate, and unmatched teacher selections are errors. Keep validation
reward specifications independent, for example using only `naive`.

`teacher_weight` defaults to `1.0` and must be finite and positive. It scales
this trajectory's distillation loss after token clamping. Weights are not
normalized within a batch. `reward_coef` scales only scalar task rewards.

The teacher scores every trajectory independently, including requests that emit
several training trajectories. Scalar rewards retain their request-level
aggregation and broadcast behavior. Teacher tensors are never broadcast between
trajectories.

### Multiple Teachers

Register additional teachers under `reward.managers`, include their names in
`reward.active_managers`, and give each a distinct `reward_model_name`. Each
dataset or dataset row selects one of those models through `reward_model_dicts`.
For example, one dataset can select `teacher_math` with `teacher_weight: 0.5`,
and another `teacher_code` with `teacher_weight: 1.5`.

A mixed batch can contain trajectories scored by different teachers. PSRL keeps
their targets and weights attached to each sample during batching and reordering.
This follows [veRL's teacher routing](https://github.com/verl-project/verl/blob/e5ca4acb/verl/experimental/teacher_loop/teacher_manager.py):
one teacher per trajectory, without combining several teacher distributions on
the same trajectory.

## Choose an Update

Configure losses under `distillation.distillation_loss`.

| Update | `loss_mode` | `use_policy_gradient` | Teacher targets |
|---|---|---|---|
| PG OPD | `k1` | `true` | Observed response token logprobs, K=1 |
| Supervised Top-k | `forward_kl_topk` | `false` | The K highest teacher logprobs and their raw token IDs |

### PG OPD

```yaml
distillation:
  enabled: true
  distillation_loss:
    loss_mode: k1
    use_policy_gradient: true
    use_task_rewards: false
    policy_loss_mode: vanilla
    clip_ratio_low: 0.2
    clip_ratio_high: 0.2
```

The sampled-token divergence is `student_logprob - teacher_logprob`. Its detached
negative becomes a token advantage for veRL's clipped policy-gradient objective.
Only `policy_loss_mode=vanilla` is supported. Direct supervised `k1` is rejected
because its gradient does not depend on the teacher.

The estimator aliases `kl`, `abs`, `mse`, `k2`, `low_var_kl`, and `k3` are also
available. The default loss mode is `k3` with policy gradient enabled.

### Supervised Top-k

```yaml
distillation:
  enabled: true
  distillation_loss:
    loss_mode: forward_kl_topk
    topk: 32
    use_policy_gradient: false
    use_task_rewards: false
train_actor_rollout_ref:
  model:
    use_fused_kernels: false
```

The logits processor computes the truncated forward KL on teacher-selected
vocabulary entries. Negative truncated KL values are clamped to zero. Gradients
flow through the student logits. Both FSDP2 and Megatron use PSRL's loss.
Teacher services automatically request sufficient vLLM logprob capacity for K.

### Task Rewards and Weighting

With `use_task_rewards=false`, PSRL skips the task policy loss entirely. With
`use_task_rewards=true`, the objective is:

$$
L = L_{\mathrm{task}} +
\mathrm{distillation\_loss\_coef}\,L_{\mathrm{distill}}.
$$

Both modes clamp each token divergence with `loss_max_clamp` before multiplying
by `teacher_weight`. Masking and aggregation follow veRL. Defaults are
`loss_max_clamp: 10.0`, `log_prob_min_clamp: -10.0`, and
`distillation_loss_coef: 1.0`. Set either clamp to `null` to disable it.

## Data Contract and Diagnostics

SMG scores raw prompt and response IDs with `temperature=1` and
`max_new_tokens=1`. It returns input token IDs and prompt logprobs, with no
probability for the first input token. PG selects the observed response token's
probability. Top-k selects alternatives by descending probability, preserving
raw vocabulary IDs.

Reward workers return teacher outputs keyed by `request_key`. TransferQueue
stores `teacher_logprobs` as float32 `[R,K]`, `teacher_ids` as int64 `[R,K]` in
Top-k mode, and a scalar `teacher_weight`. The actor adapter places targets at
full-sequence prediction positions `P-1` through `P+R-2`. Synthetic training
padding has weight zero. Missing or malformed targets fail before actor updates.

Monitor `distillation/loss`, token loss minima and maxima, and
`distillation/abs_loss` for sampled-token estimators. Top-k also reports teacher
and student probability mass on the selected tokens.

## Run Examples and Tests

`examples/opd_trainer/opd_qwen_gsm8k.sh` configures PG OPD. The larger launchers
show DAPO task rewards and teacher tensor parallelism. They source your local
`env/psrl.sh` and require `LOCAL_IP`, prepared parquet files, local student and
teacher checkpoints, and GPUs matching their configured resource pools. Set
`STUDENT_MODEL_PATH` and `TEACHER_MODEL_PATH` before launching them.

Run CPU protocol, loss, and HTTP pipeline tests in the PSRL environment:

```bash
python -m pytest tests/workers/reward/test_opd_pipeline.py tests/trainer/distillation/test_losses.py
```

The pipeline suite starts an HTTP teacher and includes real local Ray and
TransferQueue storage tests. It covers both update modes, synchronous and
asynchronous submission, multi-turn masks, teacher routing, and stale results.

Run native SMG protocol conversions from its checkout:

```bash
cd third_party/smg
cargo test -p openai-protocol --lib
cargo test -p smg-grpc-client native_generate_requests_prompt_logprobs
cargo test -p smg --lib prompt_top_logprobs_tests
cd ../..
```

GPU smoke tests are explicitly enabled. Set `MODEL_PATH` to a small local
HuggingFace causal LM checkpoint that fits beside a vLLM teacher. The test starts
real SMG and vLLM services and updates a CUDA student for PG and Top-k:

```bash
python -m pytest tests/trainer/distillation/test_opd_gpu.py --run-gpu \
  --gpu-model-path "$MODEL_PATH" --gpu-tp 1
```
