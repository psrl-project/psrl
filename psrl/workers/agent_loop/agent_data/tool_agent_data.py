import copy
import json
import logging
import os
import uuid

import torch
from omegaconf import DictConfig
from PIL import Image
from verl.utils.tokenizer import normalize_token_ids

from psrl.environments.base import ConversationType, Environment
from psrl.environments.tool_env import ToolAction
from psrl.tools.tool_parser.base import ToolParser
from psrl.workers.agent_loop.agent_data.base import AgentData
from psrl.workers.agent_loop.agent_data.conversation_agent_data import ConversationAgentData
from psrl.workers.reward.client import RewardSubmissionClient

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def sort_tool_schema_keys(value):
    """Return a copy of a tool schema value with all dict keys sorted recursively."""
    if isinstance(value, dict):
        return {key: sort_tool_schema_keys(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [sort_tool_schema_keys(item) for item in value]
    return copy.deepcopy(value)


@AgentData.register("tool_agent_data")
class ToolAgentData(ConversationAgentData):
    """
    Agent data implementation for tool-based interactions.

    This class manages trajectories where the agent can invoke tools during
    multi-turn conversations. It handles parsing tool calls from model outputs,
    updating conversation history with tool responses, and computing rewards.

    The class supports both synchronous and asynchronous tool execution and
    integrates with chat templates for proper formatting.
    """

    def __init__(
        self,
        config: DictConfig,
        reward_manager: RewardSubmissionClient,
        env: Environment,
        **kwargs,
    ):
        """Initialize ToolAgentData instance.

        Args:
            config: Configuration object containing training settings
            reward_manager: Shared client for submitting reward requests.
            env: Environment instance with ``get_tool_schemas()`` method
            tokenizer: Tokenizer for converting between text and tokens
            processor: Optional multimodal processor (e.g. Qwen2VLProcessor).
            **kwargs: Additional keyword arguments from configuration
        """
        assert hasattr(env, "get_tool_schemas"), "Environment must implement get_tool_schemas method."

        super().__init__(config, reward_manager, env)
        self.tool_schemas = sort_tool_schema_keys(self.env.get_tool_schemas())
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self.tool_parser_name = self.config.gen_actor_rollout_ref.rollout.multi_turn.format
        self.tool_parser = ToolParser.get_tool_parser(self.tool_parser_name, self.tokenizer)
        self.tool_call_names: list[str] = []

    def _get_chat_template_tools(self, is_init: bool) -> list[dict] | None:
        """Include tool schemas in the initial prompt only."""
        return self.tool_schemas if is_init else None

    async def encode_observation(
        self,
        observation: ConversationType,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        is_init: bool = False,
    ) -> tuple[list[int], bool]:
        """
        Encode tool environment messages into token IDs asynchronously.

        The initial observation includes tool schemas and yields prompt tokens.
        Later observations omit the system prefix and yield masked response tokens.
        Blocking processor work runs in the default thread pool.

        Returns:
            (token_ids, is_prompt) where is_prompt is True only for the initial
            observation (which becomes part of the padded prompt tensor).
        """
        if not is_init and self.tool_parser_name in {"gpt-oss", "gemma4"}:
            if images or videos:
                raise NotImplementedError(
                    f"Tool parser {self.tool_parser_name!r} does not support multimodal tool responses."
                )
            if self.tool_parser_name == "gpt-oss":
                response_text = self._format_gpt_oss_tool_response(observation)
            else:
                response_text = self._format_gemma4_tool_response(observation)
            token_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.encode(response_text, add_special_tokens=False),
            )
            return normalize_token_ids(token_ids), False
        return await super().encode_observation(
            observation,
            images=images,
            videos=videos,
            is_init=is_init,
        )

    def decode_action_from_token_ids(self, token_ids: list[int]) -> ToolAction:
        """Decode model generated token ids into ToolAction.

        Returns a list of OpenAI-function-call style dicts:
        [{"type": "function", "function": {"name": ..., "arguments": "..."}}]
        """
        _, tool_calls = self.tool_parser.extract_tool_calls_from_token_ids(token_ids, tools=self.tool_schemas)
        tool_calls_dict = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": tool_call.to_dict(),
            }
            for tool_call in tool_calls
        ]
        # Ensure tool call arguments are JSON strings (required by chat template)
        for i, call in enumerate(tool_calls_dict):
            if isinstance(call.get("function", {}).get("arguments"), dict):
                tool_calls_dict[i]["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        self.tool_call_names = [
            call["function"]["name"]
            for call in tool_calls_dict
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        ]
        return tool_calls_dict

    def decode_action_from_response_str(self, response_str: str) -> ToolAction:
        """Decode model generated response string into ToolAction.

        Returns a list of OpenAI-function-call style dicts:
        [{"type": "function", "function": {"name": ..., "arguments": "..."}}]
        """
        _, tool_calls = self.tool_parser.extract_tool_calls_from_str(response_str, tools=self.tool_schemas)
        tool_calls_dict = [
            {
                "id": f"call_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": tool_call.to_dict(),
            }
            for tool_call in tool_calls
        ]
        # Ensure tool call arguments are JSON strings (required by chat template)
        for i, call in enumerate(tool_calls_dict):
            if isinstance(call.get("function", {}).get("arguments"), dict):
                tool_calls_dict[i]["function"]["arguments"] = json.dumps(call["function"]["arguments"])
        self.tool_call_names = [
            call["function"]["name"]
            for call in tool_calls_dict
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        ]
        return tool_calls_dict

    def _tool_message_text(self, message: dict) -> str:
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
        return "" if content is None else str(content)

    def _format_gpt_oss_tool_response(self, observation: ConversationType) -> str:
        parts = []
        for message, tool_name in zip(observation, self.tool_call_names, strict=False):
            content = self._tool_message_text(message)
            parts.append(
                f"<|start|>functions.{tool_name} to=assistant<|channel|>commentary<|message|>{content}<|end|>"
            )
        return "".join(parts) + "<|start|>assistant"

    def _format_gemma4_tool_response(self, observation: ConversationType) -> str:
        parts = []
        for message, tool_name in zip(observation, self.tool_call_names, strict=False):
            content = self._tool_message_text(message)
            parts.append(f'<|tool_response>response:{tool_name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
        return "".join(parts)

    def prepare_generation_request(self, request: dict) -> dict:
        request = super().prepare_generation_request(request)
        if self.tool_parser.stop_token_ids:
            request["stop_token_ids"] = self.tool_parser.stop_token_ids
        return request

    def prepare_chat_completion_request(self) -> tuple[list[dict], list[dict] | None]:
        """Build messages and tools from current trajectory state."""
        messages = []
        for step in self.session_data.trajectories[-1].steps:
            messages.extend(step.chat_completions)
        tools = self.tool_schemas
        return messages, tools

    def _parse_tool_calls(self, tool_calls: list[dict]):
        """Extract tool call actions from OpenAI format tool_calls."""
        return tool_calls

    async def update_from_model_chat_completion(self, output: dict, **kwargs) -> tuple:
        """Parse chat completion response and update trajectory.

        Args:
            output: Raw chat completion response dict from SMG.

        Returns:
            Tuple of (action, overlong_terminate).
        """
        choice = output["choices"][0]
        assistant_msg = choice["message"]

        # Update step
        self.session_data.assistant_turns += 1
        self.session_data.trajectories[-1].assistant_turns += 1
        model_response = assistant_msg.get("content", "") or ""

        try:
            tool_calls_dict = self.decode_action_from_response_str(model_response)
        except Exception as e:
            psrl_logger.error("Failed to parse tool calls: %s", e)
            tool_calls_dict = []

        # Update current step with assistant response
        self.add_step_chat_message(assistant_msg)
        self.set_step_model_response(model_response)
        self.set_step_action(tool_calls_dict)

        # Check length limit
        usage = output.get("usage", {})
        total_tokens = usage.get("total_tokens", 0)
        overlong = total_tokens > (
            self.config.gen_actor_rollout_ref.rollout.prompt_length
            + self.config.gen_actor_rollout_ref.rollout.response_length
        )

        return tool_calls_dict, overlong
