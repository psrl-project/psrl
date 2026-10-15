# Multi-Datasets & Multi-Reward Models Usages

## 1. Multi-Reward Models

Currently, `PSRL` supports both functional-based or generative-based reward model, as well as their combinations.

### Functional-based Reward Model

We provide some typical `Reward Loop Manager` (`naive` `dapo` etc. see in `psrl/workers/reward/reward_loop`) and a default `compute_score` function can compute the reward score for several commonly used datasets.

An example functional-based reward model should be configured as follows:
```yaml
reward_loop_type: naive    # Reward Loop Manager Type
reward_fn:                 # Function `compute_score` used in reward loop. 
  - default                # Allowing using several kinds of functions in one reward loop
```

Also, we support two levels of granularity in customizing your own functional-based reward model.
#### Reward Loop Level

You should define your own `RewardLoop` like following codes:
```python
from psrl.workers.reward.reward_loop import register
from psrl.workers.reward.reward_loop.base import RewardLoopBase

@register("custom_loop")   # use your own cutomized name to register
class CustomizedRewardLoop(RewardLoopBase):
    """The reward loop."""

    def __init__(
        self,
        config,
        tokenizer,
        **kwargs
    ):
        super().__init__(config, tokenizer)
        ### your code here

    async def run_single(self, data: DataProto) -> dict:
        """ The function to handle reward inputs. Like compute reward score etc."""
        ### your code here
```

Configuration can be set as follow:
```yaml
reward_loop_type: custom_loop    # the customized name you set
reward_fn:
  - ...
reward_loop_kwargs: ...          # kwargs for reward_loop_manager (e.g. overlong config for dapo)
```

#### Reward Function Level
You can use any pre-defined or customized `Reward Loop Manager`, but you should provide your own `compute_score` function, which would be used in `run_single` to compute the reward score based on prompt, response, ground truth etc.

Configuration should be set as follow:
```yaml
reward_loop_type: naive              # or others
reward_fn:
  - name: custom_compute_score     # shold be the actual function name
    path: path/to/file             # where you implement your function
    reward_fn_kwargs: ...          # kwargs for reward_fn (if needed)
```
---
### Generative-based Reward Model

When using a generative-based reward model, you can implement your own `Reward Loop Manager` as above. 

Besides, we provide a general loop for this kind of reward model. When using pre-defined generaive reward loop, you should implement your own `gen_reward_fn`, which is responsible for constructing prompt for reward model and post-processing after calling LLM and compute reward score. Also, we provide a `default` reward function for reward LLM (in `psrl/workers/reward/gen_reward_function/default_gen_rm.py`).

```python
from psrl.workers.reward.gen_reward_function.base import GenRewardFunctionBase
from psrl.workers.reward.gen_reward_function.registry import gen_reward_func


@gen_reward_func("custom_gen_reward_fn")
class CustomGenRewardFunction(GenRewardFunctionBase):
    def __init__(self):
        super().__init__(using_sys_prompt=False)  # if using LLM's pre-defined systematic prompt template, set it True, else False
        # your code here
        
    def prompt_constructor(self, prompt_str, response_str) -> list[dict]:
        """ Customized Prompt Construction. """
        # your code here
        
    def compute_score(
        self,
        data_source,
        solution_str,
        rm_output,
        rm_output_value = None,
        ground_truth = "",
        extra_info = None,
        **kwargs,
    ) -> float:
        """ Customized Function to Compute Reward Score. """
        # your code here
```

You should set configurations like following codes:
```yaml
# Gen-based Reward Model Definition
reward_loop_type: gen
reward_fn: 
    - skywork
model_name: Skywork-Reward-V2-Qwen3-8B

# Resource Pool for Reward Model
enable_resource_pool: True
nnodes: 1
n_gpus_per_node: 1

# Reward Model Replica Configs
num_replicas: 1
rollout_ngpus_per_instance_per_node: 1
rollout_nnodes_per_instance: 1

# Reward Model Config, like actor/ref/rollout
model:
    path: models/Skywork-Reward-V2-Qwen3-8B
    external_lib: ${oc.select:train_actor_rollout_ref.model.external_lib,null}
    trust_remote_code: False

# Reward Model Rollout Config, like rollout
rollout:
    _target_: psrl.workers.config.RolloutConfig
    # ...
    runner: pooling  # or `generate`
    task: classify   # or `score` `generate` etc.

# only used for runner==pooling, see https://docs.vllm.ai/en/latest/api/vllm/#vllm.PoolingParams
pooling_config:
    task: classify
    use_activation: false
    normalize: false

# only used for runner==generate, see https://docs.vllm.ai/en/latest/api/vllm/#vllm.SamplingParams
sampling_config:
    temperature: 0.0
    top_p: 1.0
    top_k: 0
```
---
### Using Multi-kinds of Reward Models

We support to use multiple kinds of reward models, and routing each request to the corresponding reward model to compute reward score.

An example is as follow.
```yaml
# Whether to launch custom reward function asynchronously during log_prob
launch_reward_fn_async: True

# How to normalize rewards after calling reward models
# none: Doing Nothing.
# group: normalize rewards within a sampling group (like in grpo).
# batch: normalize rewards within a group of data which is from the same dataset.
reward_normalization: group

# Reward Models Definitions & Configs
reward_models:
  - reward_loop_type: naive
    reward_fn: 
      - default

  - reward_loop_type: gen
    reward_fn: 
      - default
    reward_model_name: Qwen3-8B
    # ...
```

## 2. Multi-Datasets Config

A dataset's configuration should contain the following keys:

```yaml
file: path/to/dataset/parquet  # The parquet file that stores training/validation datasets.
prompt_key: prompt             # In which field that contains prompt. Currently, we don't support multimodal datasets.

# Reward Model Definition
reward_model_dicts:
  ...
```

We support you to use multiple reward models to evaluate trajectories.
You can configure reward models for each dataset in key `reward_model_dicts`, like this:

**IMPORTANCE** Currently, we only support one kind of reward model in Validation Dataset.

```yaml
# Reward Model Definition
reward_model_dicts:
  - reward_fn_key: data_source    # The field used to select the reward function (if using different ones per example).
    reward_loop_type: naive       # The reward loop type going to use. (e.g. naive/dapo/gen)
    reward_fn: default            # The reward function used in reward loop. (e.g. default/customized)
    reward_model_name: Qwen3-8B   # If using a generative reward model, LLM's name must be provided.
    reward_coef: 1.0              # The coefficient of current reward model, 
                                  # the return score will be `reward_coef * reward_score` as a part of total reward score.
  - ...
```

Training/Validation datasets should be construct as follows:
```yaml
train_datas: 
  - file: data/gsm8k_verl/train.parquet
    prompt_key: prompt
    reward_model_dicts:
      - reward_fn_key: data_source
        reward_loop_type: naive
        reward_fn: default
        reward_model_name: null
        reward_coef: 1.0
      - reward_fn_key: data_source
        reward_loop_type: gen
        reward_fn: default
        reward_model_name: Qwen3-8B
        reward_coef: 1.0
      - ...
  # ...

# split ratio, which instruct how to form a batch with datasets above.
train_datasets_ratios: [0.3, 0.3, 0.4]

# Validation parquet. Can be a list or a single file.
val_datas: 
  - file: data/gsm8k_verl/test.parquet
    prompt_key: prompt
    - reward_model_dicts:
      reward_fn_key: data_source
      reward_loop_type: naive
      reward_fn: default
      reward_model_name: null
      reward_coef: 1.0
  # ...
```

## 3. Example

See `examples/precision_test/multi_datasets/multi_datasets_stream_megatron_qwen_7b.sh`.
See `psrl/trainer/config/data/data.yaml` for the dataset spec schema.