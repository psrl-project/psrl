# LLM-as-a-Judge

## What Is LLM-as-a-Judge?

LLM-as-a-Judge uses an LLM or an LLM-based reward model to evaluate a policy's
response and assign a **scalar reward to its trajectory**. The judge can be a
general instruction-following model prompted to evaluate answers, or a model
fine-tuned for judging correctness, quality, or preferences. Its input combines
the task prompt and the policy response, optionally with a task-specific rubric
or reference supplied by the application.

The reward depends on how the judge's output is interpreted:

| Scoring approach | Judge output | Trajectory reward |
|------------------|--------------|-------------------|
| Direct scalar scoring | A learned reward or classification head returns a value | Use or transform that value |
| Token-probability scoring | Probabilities of judgment tokens such as `Yes` and `No` | Derive a score from the verdict probabilities |
| Generated-response scoring | A critique, rating, or final verdict in generated text | Parse the response into a scalar |

For example, a generative verifier can use the probability of its `Yes` token as
the correctness score, optionally after generating a critique. See
[Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240).
These are probabilities of the judge's verdict tokens, rather than the policy's
response tokens.

PSRL's `gen` reward loop implements generated-response and scalar-output paths.
It constructs the judge input, runs inference through the reward model's SMG
gateway, and calls `compute_score` to produce the trajectory reward. The built-in
generation path forwards decoded text. Token-probability scoring requires a
reward-loop extension that requests and forwards judge token probabilities to
the scorer.

## Built-in Judge Examples

### Default Generative Judge

For each policy response, `DefaultGenRewardFunction` constructs one user message
containing the math problem and the proposed solution. It asks the judge to
critique the solution and end with `\boxed{True}` or `\boxed{False}`.

The parser reads the **last boxed expression**. Exactly `True` gives `1.0`.
`False`, a missing box, another label, or a parsing failure gives `0.0`. The score
is determined by the judge's output, not by a ground-truth comparison inside this
function. A cutoff before the final verdict can therefore produce a zero score.

The `llm_as_a_judge` preset selects the `default` reward function,
`rollout.runner=generate`, and `rollout.task=generate`. Set its model name and
checkpoint path explicitly. The loop sends tokenized
judge prompts to `/v1/completions` and passes the generated text as `rm_output`
to the scoring function. The prompt and parser are in
`psrl/workers/reward/gen_reward_function/default_gen_rm.py`.

### Skywork Scalar Reward Model

The `gen_skywork` preset uses
[Skywork-Reward-V2-Qwen3-8B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-8B)
with the `skywork` reward function, `runner=pooling`, and `task=classify`.
`SkyworkGenRewardFunction` constructs user/assistant messages from the task prompt
and policy response, and the loop queries `/v1/classify`.

The returned scalar is passed as `rm_output_value`. The scorer converts it to a
float and returns it directly. This preset uses `pooling_config.use_activation=false`
and `normalize=false`, so its reward is not constrained to `[0, 1]`.
It produces no generated critique. The prompt and scorer are in
`psrl/workers/reward/gen_reward_function/skywork_rm.py`.

Both examples use `reward_loop_type=gen`. Choose the checkpoint, registered
scorer, runner, and task as a matching set:

| Loop preset | Reward function | Runner / task | Scorer input |
|----------------|-----------------|---------------|--------------|
| `llm_as_a_judge` | `default` | `generate` / `generate` | `rm_output` text |
| `gen_qwen3_8b` | `default` | `generate` / `generate` | `rm_output` text |
| `gen_skywork` | `skywork` | `pooling` / `classify` | `rm_output_value` scalar |

## Configure a GenRM

Model serving and sample-level reward selection are configured separately.
`reward.active_loops` enables named runtime configurations. Each dataset's
`reward_model_dicts` selects the reward loops to execute for its samples.

The following overlay configures the `llm_as_a_judge` preset. Merge it with a
complete training configuration and set the model path, model name, and resource
placement for your launcher:

```yaml
reward:
  active_loops: [naive, llm_as_a_judge]
  reward_normalization: none
  reward_loops:
    llm_as_a_judge:
      reward_model_name: judge
      reward_fn:
        - path: null
          name: default
      model:
        path: ${oc.env:PSRL_WORKSPACE}/models/judge
      rollout:
        runner: generate
        task: generate
        response_length: 16384
        max_model_len: 32768
      sampling_config:
        temperature: 1.0
        top_p: 1.0
        top_k: -1
data:
  train_datas:
    - file: ${oc.env:PSRL_WORKSPACE}/data/gsm8k/train.parquet
      data_source_name: openai/gsm8k
      prompt_key: prompt
      reward_model_dicts:
        - reward_fn_key: data_source
          reward_loop_type: gen
          reward_fn: default
          reward_model_name: judge
          reward_coef: 1.0
```

This training dataset selects **judge-only reward**. The active `naive` loop
remains available for validation specifications that request verifiable reward.
Configure `data.val_datas` independently if validation should also use the judge.

Keep the tuple `(reward_loop_type, reward_fn, reward_model_name)` consistent with
the enabled loop. In the dataset, `reward_fn` is the registered function name.
In the loop, `reward_fn` is a list of function configurations. Merely enabling
a loop does not attach it to every dataset. The loop key is a configuration
label. Set `reward_model_name` and `model.path` to your actual checkpoint when
adapting a preset.

For the supplied Skywork preset, activate `gen_skywork` in place of
`llm_as_a_judge`. In the dataset reward specification, select `reward_fn: skywork`
and `reward_model_name: Skywork-Reward-V2-Qwen3-8B`. Keep `naive` active if the
validation dataset still requests verifiable reward.

### Combine Judge and Verifiable Rewards

To combine verification and judging, add a second entry to that dataset's
`reward_model_dicts`:

```yaml
- reward_fn_key: data_source
  reward_loop_type: naive
  reward_fn: default
  reward_model_name: null
  reward_coef: 1.0
```

Use `reward_loop_type: dapo` for the DAPO verifier and activate the `dapo` manager
as well. `RewardLoopWorker` runs the selected loops and computes:

$$
R_{\mathrm{trajectory}} = \sum_j c_j R_j,
\qquad c_j = \texttt{reward\_coef}_j.
$$

The combined reward can exceed one even when each component is binary.
`reward.reward_normalization` controls optional post-aggregation batch or group
normalization. `none` leaves this combined reward unnormalized.

### Tune Generation and Context Budgets

Judge inference uses its **own tokenizer and chat template**. Policy prompts and
responses are decoded before the judge prompt is constructed and retokenized.
Account for the judging instructions as well as the policy response when sizing
the judge's context window.

| Setting under `reward.reward_loops.<name>` | Effect |
|---------------------------------------|--------|
| `model.path` | Judge checkpoint and default tokenizer location |
| `model.tokenizer_path` | Optional separate tokenizer location |
| `rollout.tensor_model_parallel_size` | Judge inference TP, matched to its placement |
| `rollout.response_length` | Default judge generation budget |
| `rollout.max_model_len` | Total context budget for judge prompt plus generated critique |
| `sampling_config.max_tokens` | Optional override of the output token budget |
| `sampling_config.temperature`, `top_p`, `top_k` | Judge sampling parameters, independent of policy sampling |
| `max_concurrent_requests_per_instance` | Optional concurrency cap per reward instance |

The request path reserves the configured output budget first. If the judge prompt
exceeds the remaining capacity, it keeps its tail, which may remove the original
question. Set a larger supported context or shorten the input and output budgets
to preserve the complete judgment prompt. An output budget that leaves no prompt
capacity is rejected.

## Customize Judge Prompts and Scores

Implement `GenRewardFunctionBase` and register it with `@gen_reward_func`. For
example, put this class in
`psrl/workers/reward/gen_reward_function/helpfulness_judge.py`:

```python
import re

from psrl.workers.reward.gen_reward_function.base import GenRewardFunctionBase
from psrl.workers.reward.gen_reward_function.registry import gen_reward_func


@gen_reward_func("helpfulness_judge")
class HelpfulnessJudge(GenRewardFunctionBase):
    def prompt_constructor(self, prompt_str: str, response_str: str) -> list[dict]:
        return [{
            "role": "user",
            "content": (
                f"Request:\n{prompt_str}\n\nResponse:\n{response_str}\n\n"
                "Evaluate whether the response directly answers the request. "
                "Explain your judgment, then end with <score>1</score> if it does, "
                "or <score>0</score> otherwise."
            ),
        }]

    def compute_score(self, data_source, solution_str, rm_output, **kwargs) -> float:
        verdict = re.search(r"<score>([01])</score>\s*$", rm_output)
        return float(verdict.group(1)) if verdict else 0.0
```

Import the class from `psrl/workers/reward/gen_reward_function/__init__.py` so
registration runs in the reward worker processes:

```python
from .helpfulness_judge import HelpfulnessJudge
```

Set the manager's `reward_fn` to `[{path: null, name: helpfulness_judge}]` and each
participating dataset reward spec's `reward_fn` to `helpfulness_judge`.
The registered `gen` loop looks up these names in its class registry. A
`reward_fn.path` override does not load an external generative reward class.

`prompt_constructor` receives only `prompt_str` and `response_str`.
`compute_score` additionally receives `data_source`, `rm_output`,
`rm_output_value`, `ground_truth`, and `extra_info`. Ground truth and extra
metadata are not automatically included in the judge prompt. A scorer may be
synchronous or asynchronous and return a float, or a dictionary containing
`score` plus additional metrics.

## Inspect Judge Outputs

Per-request `reward_extra_info` is grouped by a key such as
`gen/default/Qwen3-8B`. It retains `rm_output`, `agent_response`, `rm_input_len`,
and, for generated judgments, `rm_output_len`. Inspect these values when changing
the prompt or output budget. A scalar-returning scorer also records its score as
`acc` in this metadata.

Request metrics include `rm_prompt_truncated_tokens` and
`rm_request_interruptions`. The `gen_reward_loop.log` file records reward-model
requests and computed scores under the configured logging directory.

### Implementation Reference

| Responsibility | Source |
|----------------|--------|
| Math judge prompt and boxed-verdict parser | `psrl/workers/reward/gen_reward_function/default_gen_rm.py` |
| Skywork conversation input and scalar scorer | `psrl/workers/reward/gen_reward_function/skywork_rm.py` |
| Custom judge interface and registry | `psrl/workers/reward/gen_reward_function/base.py`, `registry.py` |
| Prompt tokenization, HTTP inference, and scoring | `psrl/workers/reward/reward_loop/gen.py` |
| Reward selection and weighted aggregation | `psrl/workers/reward/reward_worker.py` |
| Named manager presets | `psrl/trainer/config/reward/reward.yaml` |
