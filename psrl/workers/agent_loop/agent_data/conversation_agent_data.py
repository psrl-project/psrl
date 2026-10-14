"""Common agent data implementation for chat-message observations."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from verl.utils import tensordict_utils as tu
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.tokenizer import normalize_token_ids

from psrl.environments.base import ConversationType, Environment
from psrl.utils.dataset.utils import as_object_sample
from psrl.workers.agent_loop.agent_data.base import AgentData, SessionData, Trajectory
from psrl.workers.gen.utils import TokenOutput
from psrl.workers.reward.client import RewardManagerClient


def normalize_openai_messages(openai_messages: list[dict]) -> ConversationType:
    """Normalize OpenAI-format messages for tokenizer chat templates."""
    messages: ConversationType = []
    for message in openai_messages:
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(str(part.get("text", part)))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        elif content is None:
            content = ""
        else:
            content = str(content)
        messages.append({"role": str(message.get("role", "")), "content": content})
    return messages


class ConversationAgentData(AgentData[ConversationType, object]):
    """Manage trajectory state shared by chat-message-based agents."""

    def __init__(
        self,
        config: DictConfig,
        reward_manager: RewardManagerClient,
        env: Environment,
        **kwargs,
    ):
        super().__init__(config, reward_manager, env, **kwargs)
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self._system_prompt_ids = initialize_system_prompt(
            self.tokenizer,
            **self.apply_chat_template_kwargs,
        )

    @property
    def max_turns(self) -> int:
        """Return the configured multi-turn limit."""
        multi_turn = getattr(self.config.gen_actor_rollout_ref.rollout, "multi_turn", None)
        return int(getattr(multi_turn, "max_turns", 10_000))

    def _get_chat_template_tools(self, is_init: bool) -> list[dict] | None:
        """Return tools to include when encoding an observation."""
        return None

    async def _apply_chat_template_ids(
        self,
        messages: ConversationType,
        *,
        add_generation_prompt: bool,
        tools: list[dict] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
    ) -> list[int]:
        """Apply the active chat template and return plain token IDs."""
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=add_generation_prompt,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            if videos is not None:
                video_values, video_metadatas = zip(*videos, strict=False)
                videos = list(video_values)
                video_metadatas = list(video_metadatas)
            else:
                video_metadatas = None
            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            return normalize_token_ids(model_inputs.pop("input_ids"))

        tokenized_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer,
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return normalize_token_ids(tokenized_prompt)

    async def encode_observation(
        self,
        observation: ConversationType,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        is_init: bool = False,
    ) -> tuple[list[int], bool]:
        """Encode an initial prompt or an incremental environment observation."""
        if not observation:
            return [], is_init

        prompt_ids = await self._apply_chat_template_ids(
            observation,
            add_generation_prompt=True,
            tools=self._get_chat_template_tools(is_init),
            images=images,
            videos=videos,
        )
        if not is_init:
            prompt_ids = prompt_ids[len(self._system_prompt_ids) :]
        return prompt_ids, is_init

    def format_chat_completions(
        self,
        observation: ConversationType,
        *,
        is_init: bool,
    ) -> ConversationType:
        """Return a copy of an already normalized conversation observation."""
        return list(observation)

    async def update_from_env(
        self,
        observation: ConversationType,
        reward: float | list[float] | None,
        done: bool,
        info: dict,
        **kwargs,
    ) -> bool:
        """Append an environment observation and its masked tokens."""
        trajectory = self.session_data.trajectories[-1]
        is_init = len(trajectory.steps) == 0
        info = info or {}
        multi_modal_data = info.get("multi_modal_data", {})
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        if images:
            if trajectory.image_data is None:
                trajectory.image_data = []
            trajectory.image_data.extend(images)
        if videos:
            if trajectory.video_data is None:
                trajectory.video_data = []
            trajectory.video_data.extend(videos)

        if len(trajectory.steps) < self.max_turns:
            step_reward = reward if isinstance(reward, list) or reward is None else [float(reward)]
            step = self.start_step(
                observation=observation,
                reward=step_reward,
                done=done,
                info=info,
            )
            step.chat_completions = self.format_chat_completions(observation, is_init=is_init)
            token_ids, is_prompt = await self.encode_observation(
                observation,
                images=images,
                videos=videos,
                is_init=is_init,
            )
            if is_prompt:
                self.append_prompt_ids(token_ids)
            else:
                self.append_user_tokens(token_ids)

        if reward is not None:
            trajectory.tool_rewards.append(float(np.sum(reward)))

        return self._is_overlong(trajectory)

    async def update_from_model_token_ids(
        self,
        output: TokenOutput,
        **kwargs,
    ) -> tuple[object | None, bool]:
        """Append model output, decode it, and parse an optional action."""
        trajectory = self.session_data.trajectories[-1]
        if len(trajectory.steps) == 0:
            self.start_step(observation=[], reward=None, done=False, info={})

        self.update_trajectory_state_from_output(output)
        response_ids = list(output.response_ids)
        routed_experts = output.routed_experts
        if routed_experts is not None and hasattr(routed_experts, "tolist"):
            routed_experts = routed_experts.tolist()
        self.append_assistant_tokens(
            response_ids,
            logprobs=output.response_log_probs,
            routed_experts=routed_experts,
        )

        assistant_content = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        self.add_step_chat_message({"role": "assistant", "content": assistant_content})
        self.set_step_model_response(assistant_content)

        try:
            action = self.decode_action_from_token_ids(response_ids)
        except NotImplementedError:
            action = None
        except Exception as exc:
            self.get_current_step().info["action_decode_error"] = repr(exc)
            action = []
        if action is not None:
            self.set_step_action(action)

        if self.config.gen_actor_rollout_ref.rollout.agent.traj_reward_mode == "step":
            output.num_turns = self.session_data.assistant_turns + self.session_data.user_turns + 1
            tensor_dict = {
                "prompts": torch.tensor(output.prompt_ids, dtype=torch.int64).unsqueeze(0),
                "responses": torch.tensor(output.response_ids, dtype=torch.int64).unsqueeze(0),
                "multi_modal_data": np.array([output.multi_modal_data], dtype=object),
                "num_turns": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                "uid": np.array([self.session_data.request_id]),
                "n_trajectory": np.array([len(self.session_data.trajectories)]),
                "data_source": np.array([self.session_data.data_source]),
                "reward_model": np.array([self.session_data.reward_model], dtype=object),
                "extra_info": np.array([self.session_data.extra_info], dtype=object),
                "agent_reward_info": np.array([self.session_data.agent_reward_info], dtype=object),
                "reward_model_dicts": as_object_sample(self.session_data.reward_model_dicts),
            }
            if self.session_data.parent_id is not None:
                tensor_dict["parent_id"] = np.array([self.session_data.parent_id])
            reward_data = tu.get_tensordict(
                tensor_dict=tensor_dict,
                non_tensor_dict={"validate": self.session_data.validate},
            )
            reward_result = await self.reward_manager.compute_score(reward_data)
            self.get_current_step().model_reward = reward_result["reward_score"]

        overlong = self._is_overlong(trajectory)
        return action, overlong

    def decode_action_from_token_ids(self, token_ids: list[int]) -> object:
        """Parse an action from model output in concrete subclasses."""
        raise NotImplementedError

    def _is_overlong(self, trajectory: Trajectory) -> bool:
        """
        Return whether the accumulated context reached the model token limit.

        Observation tokens are appended to `response_ids`, so the limit applies
        to the combined prompt and response tokens.

        Args:
            trajectory (Trajectory): The trajectory whose accumulated context to check.

        Returns:
            bool: True if `len(prompt_ids) + len(response_ids) >= max_model_len`.
        """
        rollout = self.config.gen_actor_rollout_ref.rollout
        max_model_len = rollout.prompt_length + rollout.response_length
        context_len = len(trajectory.prompt_ids) + len(trajectory.response_ids)
        return context_len >= max_model_len

    def reset(self) -> None:
        """Reset all session state."""
        self.session_data = SessionData()

    def init_trajectory(self, request: dict) -> None:
        """Initialize session metadata and the first trajectory."""
        self.session_data = SessionData(
            request_id=request.get("uid", 0),
            parent_id=request.get("parent_id"),
            validate=request.get("validate", False),
            data_source=request.get("data_source", "unknown"),
            reward_model=request.get("reward_model", {}),
            extra_info=request.get("extra_info", {}),
            agent_reward_info=request.get("agent_reward_info", {}),
            reward_model_dicts=request.get("reward_model_dicts", []),
        )
        self.session_data.trajectories.append(Trajectory())
