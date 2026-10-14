# Custom Environments & AgentData in PSRL

This document explains **what an Environment and AgentData are in PSRL**, and how to **create your own** and plug them into `multi_turn_agent_loop` **without changing the agent loop code**.

> TL;DR: You implement and register a pair:
>
> - `Environment[ObsType, ActType]` (defines the interaction dynamics)
> - `AgentData[ObsType, ActType]` (adapts env observations to model tokens, and model tokens to env actions)
>
> Then select them in config: `rollout.agent.env.name` and `rollout.agent.data.name` or per-request in `DataProto`.

## Concept Overview

### Environment

An **Environment** is the “world” that the agent interacts with. It:

- Receives an **action** from the agent (`ActType`)
- Produces the next **observation** (`ObsType`), a numeric **reward**, a **done** flag, and optional **info**

In PSRL, `multi_turn_agent_loop` calls:

- `await env.reset(task=request, ...) -> (observation, info)` where `task` is a `dict`
- `await env.step(action) -> EnvStepOutput` (`observation`, `reward`, `done`, `info`)

Your `ObsType` and `ActType` can be **anything** (dicts, strings, custom dataclasses, etc.). For example, in multi-turn tool use cases, `ObsType` can be `ConversationType` (list of messages) and `ActType` can be a dict representing a tool call.

### AgentData

**AgentData** is the adapter between:

- **Environment space** (`ObsType`/`ActType`)
- **Model space** (token IDs and logprobs)

AgentData also builds `Trajectory` / `Step` records under `session_data` for training (prompt/response ids, masks, optional per-step reward, etc.). The tokenizer is taken from the paired `Environment` (`self.tokenizer = self.env.tokenizer`).

In PSRL, `multi_turn_agent_loop` calls on AgentData:

1. `init_trajectory(request)`: initialize a new trajectory for the incoming task (request).
2. `update_from_env(observation, reward, done, info)`: update internal state from env step.
3. `prepare_generation_request(request)`: prepare inputs for model generation in inference engines such as vLLM.
4. `update_from_model_token_ids(output)`: update internal state from model output tokens, return env action and done flag.
5. `finalize_output(request)`: finalize trajectory after episode ends.

### Multi-turn Agent Loop

The loop is intentionally generic: it only knows it must alternate:

1. env produces an observation
2. AgentData turns it into model input
3. model generates tokens
4. AgentData turns tokens into env action
5. env consumes action and returns the next observation

As long as your custom `Environment` and `AgentData` follow the required interfaces and are registered, **you do not need to modify the loop**.

## How to Integrate Custom Environment & AgentData

### Implement a custom Environment

Create a new file, e.g. `psrl/environments/my_env.py`:

```python
from __future__ import annotations

from typing import Any

import ray
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

from psrl.environments.base import Environment, EnvStepOutput


MyObs = dict[str, Any]
MyAct = dict[str, Any]


@Environment.register("my_env")
class MyEnvironment(Environment[MyObs, MyAct]):
    @classmethod
    def init_class(cls, config: DictConfig, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        # heavy init here (load resources, build tool set, etc.)

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor | None = None,
        max_turns: int = 10,
        **kwargs,
    ):
        super().__init__(config, reward_manager, tokenizer, processor=processor)
        self.max_turns = max_turns
        self.turn = 0

    async def reset(self, task: dict, **kwargs) -> tuple[MyObs, dict]:
        self.turn = 0
        # build initial observation from task dict
        obs: MyObs = {"raw_task": task.get("raw_prompt")}
        return obs, {}

    async def step(self, action: MyAct, **kwargs) -> EnvStepOutput:
        self.turn += 1
        next_obs: MyObs = {"turn": self.turn, "last_action": action}
        reward = [0.0]
        done = self.turn >= self.max_turns
        return EnvStepOutput(observation=next_obs, reward=reward, done=done, info={})

    async def close(self) -> None:
        return

    @property
    def state(self) -> Any:
        return {"turn": self.turn}
```

You need to implement:

- `init_class`: class-level heavy initialization (loading resources, etc.) which are shared across env instances.
- `__init__`: instance-level initialization (config, reward manager, tokenizer, etc.)
- `reset`: reset the env for a new episode/task (`task: dict`).
- `step`: apply an action and return next observation, reward, done, info.
- `close`: clean up resources.
- `state` (optional): property returning serializable env state for checkpointing.

Notes:

- Return `info` as a dict (use `{}`) for best compatibility.
- `reward` in `EnvStepOutput` is a **list** of floats (one entry per tool call when applicable).
- `max_turns` is passed by `multi_turn_agent_loop` when constructing the env.

### Implement a custom AgentData

Create a new file, e.g. `psrl/workers/agent_loop/agent_data/my_agent_data.py`:

```python
from __future__ import annotations

import json
from typing import Any

import numpy as np
import ray
from omegaconf import DictConfig
from PIL import Image
import torch

from psrl.environments.base import ConversationType, Environment
from psrl.workers.agent_loop.agent_data.base import AgentData, SessionData, Trajectory
from psrl.workers.gen.utils import TokenOutput

# should match the Env Obs/Act types
MyObs = dict[str, Any]
MyAct = dict[str, Any]


@AgentData.register("my_agent_data")
class MyAgentData(AgentData[MyObs, MyAct]):

    def __init__(
        self,
        config: DictConfig,
        reward_manager: ray.actor.ActorHandle,
        env: Environment,
        **kwargs,
    ):
        super().__init__(config, reward_manager, env, **kwargs)

    def reset(self) -> None:
        self.session_data = SessionData()

    def init_trajectory(self, request) -> None:
        self.create_trajectory()
        request_id = request.get("uid", 0) if isinstance(request, dict) else 0
        self.session_data.trajectories[-1].request_id = request_id

    # --- Required hooks (recommended minimal customization surface) ---
    def format_chat_completions(self, observation: MyObs, *, is_init: bool) -> ConversationType:
        # Always keep Step.chat_completions as ConversationType.
        # For non-chat envs, a simple strategy is to serialize observation to JSON.
        return [{"role": "user", "content": json.dumps(observation, ensure_ascii=False)}]

    async def encode_observation(
        self,
        observation: MyObs,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        is_init: bool = False,
    ) -> tuple[list[int], bool]:
        # Convert observation to tokens (offload blocking tokenizer work to an executor).
        messages = self.format_chat_completions(observation, is_init=is_init)
        token_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            ),
        )
        return token_ids, is_init

    def decode_action_from_token_ids(self, token_ids: list[int]) -> MyAct:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return {"raw_text": text}

    # --- Main loop-facing methods ---
    async def update_from_env(
        self,
        observation: MyObs,
        reward: float | list[float] | None,
        done: bool,
        info: dict,
        **kwargs,
    ) -> bool:
        trajectory = self.session_data.trajectories[-1]
        is_init = len(trajectory.steps) == 0

        step_reward = reward if isinstance(reward, list) or reward is None else [float(reward)]
        step = self.start_step(observation=observation, reward=step_reward, done=done, info=info)
        step.chat_completions = self.format_chat_completions(observation, is_init=is_init)

        token_ids, is_prompt = await self.encode_observation(observation, is_init=is_init)
        if token_ids:
            if is_prompt:
                self.append_prompt_ids(token_ids)
            else:
                self.append_user_tokens(token_ids)

        return trajectory.response_length >= self.config.gen_actor_rollout_ref.rollout.response_length

    async def update_from_model_token_ids(self, output: TokenOutput, **kwargs) -> tuple[MyAct, bool]:
        trajectory = self.session_data.trajectories[-1]
        self.update_trajectory_state_from_output(output)

        response_ids = list(output.response_ids)
        self.append_assistant_tokens(response_ids, logprobs=output.response_log_probs)

        action = self.decode_action_from_token_ids(response_ids)
        self.set_step_action(action)
        self.set_step_model_response(self.tokenizer.decode(response_ids, skip_special_tokens=True))

        overlong = trajectory.response_length >= self.config.gen_actor_rollout_ref.rollout.response_length
        return action, overlong
```

What you MUST implement in a custom AgentData:

- `reset()`: reset internal state for a new episode.
- `init_trajectory(request)`: initialize a new trajectory for the incoming task (request).
- `format_chat_completions(observation, is_init)`: convert env observation to `ConversationType` for logging.
- `encode_observation(...)`: async; convert env observation to model input token IDs (optional multimodal args).
- `decode_action_from_token_ids(token_ids)`: convert model output token IDs to env action.
- `update_from_env(observation, reward, done, info)`: update internal state from env step.
- `update_from_model_token_ids(output)`: update internal state from model output tokens, return env action and done flag.

It's recommended to use the provided `AgentData` helper methods to keep step logs consistent across different env types.

- `create_trajectory`, `start_step`, `append_prompt_ids`, `append_user_tokens`, `append_assistant_tokens` to build up the trajectory steps.
- `set_step_action`, `set_step_model_response` to set per-step action and model response text.
- `format_chat_completions` to keep step logs consistent across different env types.

Prefer subclassing `ConversationAgentData` when your observations are already chat messages — see `psrl/workers/agent_loop/agent_data/conversation_agent_data.py`.

### Register and configure

`multi_turn_agent_loop` will construct your env and agent data by **name** via the registries.

Make sure:

1. Your modules are imported at runtime (so the decorators run).
   - The simplest way is to add them to some package `__init__.py` import, or ensure your runner imports the module.
2. Set config values:

If all training data use the same environment and agent data, you can set in your config:

```yaml
gen_actor_rollout_ref.rollout.agent.env.name: "my_env"
gen_actor_rollout_ref.rollout.agent.data.name: "my_agent_data"
```

Otherwise, you can set them per-request in the `DataProto` non-tensor batch:

```python
request.non_tensor_batch["env_class"] = np.array(["my_env"])
request.non_tensor_batch["data_class"] = np.array(["my_agent_data"])
```
