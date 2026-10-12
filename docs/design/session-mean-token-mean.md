# Session-mean-token-mean loss aggregation

Enable with `train_actor_rollout_ref.actor.loss_agg_mode=session-mean-token-mean`.
Keep `actor.policy_loss.loss_mode=vanilla` for standard PPO clipping. The former
`policy_loss.loss_mode=per_rollout_mean` has been removed, not aliased.

For each session `s`, let `T_s` be the number of unmasked response tokens across
all of its training rows, and let `S` count sessions with `T_s > 0`. The objective is

```text
L = (1 / S) * sum_s [sum_(row,token in s) mask * token_loss / T_s].
```

PSRL's `uid` identifies the session; `parent_id` groups independent rollouts of the
same prompt for advantage estimation and must not be used for session aggregation.
The old experimental `algorithm.rollout_loss_key` override is no longer used.

The trainer computes one FP32 `session_loss_weights` scalar per row after rollout
rejection masking, over the entire advantage/training window. Each nonempty row
receives `1 / (S * T_s)`; empty rows receive zero. TransferQueue carries this dense
row field alongside the nested response tensors. Row selection, balancing, and
micro-batch splitting preserve its association with the sample.

The engine passes the current micro-batch weights through `global_batch_info` to
verl's `agg_loss`. That function multiplies the response mask by the per-row FP32
session weights and calls `masked_sum`, then multiplies by `dp_size` to compensate
for DP gradient averaging. The weighted mask keeps low precision losses in FP32
during reduction, preserves FP64 inputs, and excludes masked nonfinite values.
The existing Megatron micro-batch/CP scaling remains unchanged. No extra collective
communication or token-sized advantage copy is needed. Metadata uses O(rows) space
instead of O(response tokens); computing session totals takes O(rows) work after
the existing mask reduction.

Normalization intentionally remains at the full training window, matching the old
implementation. Optimizer mini-batches contribute partial window losses; they do
not independently renormalize sessions. Changing this boundary changes the
objective scale and is a separate design decision. Splitting or reordering rows
within a window does not change their total contribution.

Policy, entropy regularization, and KL regularization now share this aggregation.
This intentionally changes entropy/KL weighting relative to the old implementation,
which normalized only policy advantages. Raw advantages and PPO clipping metrics
remain unchanged. Old-policy entropy diagnostics use session counts from their
own mask, before rollout rejection has run.

Empty windows produce zero aggregate loss and gradient. Missing/misaligned weights
fail explicitly. `pre_step` overlap remains unsupported because it computes
normalizers separately per chunk; use `recompute` or disable overlap. `gspo`, `sapo`,
and `geo_mean` fix their own aggregation and are rejected by the engine for this
mode. Bypass rejection sampling is also rejected: its mask is only known inside a
worker micro-batch, too late to compute global session totals. Decoupled rejection
sampling is supported. Legacy worker paths that do not supply session weights fail
rather than silently falling back to a different objective.

Regression tests: `pytest tests/trainer/test_session_loss.py`. Numerical and
worker adapter tests require PyTorch and TensorDict; they skip explicitly when
those dependencies are unavailable. Multi-GPU FSDP/Megatron execution still needs
to be validated in a training environment.
