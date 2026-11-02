# Generative Reward Model

Use a separate model to provide a learned reward or a teacher signal for policy
training. The judge path includes a built-in mathematical correctness prompt and
supports custom prompts for other tasks. Generative reward can also complement a
verifiable reward on the same sample.

PSRL groups two complementary recipes under this umbrella:

| Recipe | One-line summary | Status |
|---|---|---|
| [LLM-as-a-Judge](llm_as_a_judge) | A separate model judges policy responses, with configurable prompt construction and scoring | Available |
| [On-Policy Distillation](on_policy_distillation) | Teacher token-level targets on student rollouts, with veRL-style distillation losses | Available |

These recipes use PSRL's vLLM serving and `RewardManager` / `RewardLoop`
infrastructure. Dataset reward specifications determine which signals each
sample uses.

```{seealso}
For runnable policy/GenRM model-pair recipes with dynamic GPU allocation, see
{doc}`../elastic`. Configuration and runtime design are covered in
{doc}`../../design/resource_elasticity_configuration` and
{doc}`../../design/resource_elasticity`.
```

```{toctree}
:maxdepth: 1
:hidden:

llm_as_a_judge
on_policy_distillation
```
