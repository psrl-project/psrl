import asyncio
import json
import logging
import os
from typing import Any

from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

from psrl.environments.base import ConversationType, Environment, EnvStepOutput
from psrl.tools.base import ToolGroup, ToolResponse, load_all_tools
from psrl.utils.logger import FileOnlyHandler
from psrl.workers.reward.client import RewardManagerClient

ToolAction = list[dict] | dict

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@Environment.register("tool_env")
class ToolEnvironment(Environment[ConversationType, ToolAction]):
    """
    Environment for tool-based agent interactions.

    This environment manages interactions where an agent can invoke tools based
    on model-generated tool calls. It handles tool execution, response formatting,
    and turn management for multi-turn tool-using conversations.

    Type Parameters:
        ConversationType: List of conversation messages (observations)
        ToolAction: List of tool call dictionaries or single dict (actions)
    """

    @classmethod
    def init_class(cls, config: DictConfig, **kwargs) -> None:
        """Perform heavy initialization work shared across all instances.

        This method is called only once per class to avoid redundant initialization.

        Args:
            config: Configuration object containing training settings
            **kwargs: Additional keyword arguments from configuration
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.max_tool_response_length = config.gen_actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.gen_actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        cls.max_parallel_calls = config.gen_actor_rollout_ref.rollout.multi_turn.max_parallel_calls

        # Initialize native and function-based tools from configuration.
        tool_config_path = config.gen_actor_rollout_ref.rollout.multi_turn.tool_config_path
        function_tool_path = config.gen_actor_rollout_ref.rollout.multi_turn.function_tool_path
        tools = load_all_tools(tool_config_path=tool_config_path, function_tool_path=function_tool_path)

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection:
            selected = [tool for tool in tools if tool.name in tool_selection]
            tools = selected

        cls.tools = ToolGroup(tools=tools)

        # Build a class-level logger for tool calls.
        log_prefix = "ToolEnvironment"
        psrl_logger.propagate = False
        psrl_logger.addHandler(FileOnlyHandler(config.psrl.logging_path, log_prefix))
        psrl_logger.info("ToolEnvironment logger initialized for tool calls.")

    def __init__(
        self,
        config: DictConfig,
        reward_manager: RewardManagerClient,
        max_turns: int,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor | None = None,
        dataset_cls=None,
        **kwargs,
    ):
        """
        Initialize the ToolEnvironment.

        Args:
            config: Configuration object containing training settings
            reward_manager: Shared client for submitting reward requests.
            max_turns: Maximum number of turns allowed in an episode
            tokenizer: Tokenizer for converting between text and tokens
            processor: Optional multimodal processor (e.g. Qwen2VLProcessor).
            dataset_cls: Optional dataset class for loading data.
        """
        super().__init__(config, reward_manager, tokenizer, processor, dataset_cls, **kwargs)

        self.max_turns = max_turns
        self.num_turn = 0
        self.tools_kwargs = {}

    async def process_vision_info(
        self,
        messages: list[dict],
    ) -> tuple[list | None, list | None]:
        """Extract images and videos from messages.

        Delegates to ``dataset_cls.process_vision_info`` (the same path used by
        ``AgentLoopBase.process_vision_info``), mirroring verl's design where a
        single canonical extraction function is shared across the whole stack.

        When no processor is configured (text-only model) both return values are None.

        Args:
            messages: Chat messages that may contain image/video content parts.

        Returns:
            (images, videos):
                images: List of `PIL.Image.Image` objects, or None if none are found.
                videos: List of `(video_tensor, metadata)` tuples, or None.
        """
        if self.processor is None:
            return None, None

        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is None:
            psrl_logger.warning(
                "AgentData.process_vision_info: processor %s has no image_processor attribute. "
                "Skipping vision extraction.",
                type(self.processor).__name__,
            )
            return None, None

        if self.dataset_cls is None:
            raise RuntimeError(
                "AgentData.process_vision_info: dataset_cls is required when processor is set. "
                "Pass dataset_cls= when constructing AgentData."
            )

        images, videos = await self.dataset_cls.process_vision_info(
            messages,
            image_patch_size=image_processor.patch_size,
            config=self.config.data,
        )
        return (images if images else None), (videos if videos else None)

    async def reset(self, task: dict, **kwargs) -> tuple[ConversationType, dict]:
        """Reset the environment to an initial state.

        Extracts the initial task prompt from the input dict and prepares
        the environment for a new episode.

        Args:
            task: dict containing the initial task/prompt
            **kwargs: Additional keyword arguments

        Returns:
            tuple: (initial_observation, info_dict) where initial_observation is
                   the task prompt and info_dict contains additional metadata

        Raises:
            AssertionError: If task doesn't contain exactly one item or lacks
                          'raw_prompt' in non_tensor_batch
        """
        info = {}
        self.num_turn = 0

        self.task = task
        assert "raw_prompt" in task, "For ReTool recipe, task must contain 'raw_prompt' in non_tensor_batch"
        initial_observation = task["raw_prompt"]
        self.tools_kwargs = task.get("tools_kwargs", {})
        images, videos = await self.process_vision_info(initial_observation)
        if images is not None or videos is not None:
            info["multi_modal_data"] = {"images": images, "videos": videos}

        return initial_observation, info

    async def step(self, action: ToolAction) -> EnvStepOutput:
        """Execute a step in the environment by calling tools.

        Takes a tool action (one or more tool calls) from the agent, executes
        the requested tools asynchronously, and returns the results as observations.

        Each tool result is a `ToolResponse` with `text`, `image`, and `video`
        fields. Multimodal responses use structured content, while text-only
        responses use plain strings. Video responses are not supported.

        Args:
            action: A single tool call dict or a list of tool calls.

        Returns:
            EnvStepOutput: Dictionary containing:
                - observation: List of tool response messages
                - reward: Cumulative reward from all tool executions
                - done: Whether the episode should terminate (True if no tools called)
                - info: Additional metadata (empty dict)

        Note:
            - Empty action list terminates the episode (done=True)
            - Exceeding max_turns returns empty observation (done=False)
            - Tool calls are executed concurrently using asyncio.gather
        """
        if isinstance(action, dict):
            action = [action]

        tool_rewards = []

        if len(action) == 0:
            return EnvStepOutput(observation=[], reward=tool_rewards, done=True, info={})

        self.num_turn += 1
        if self.num_turn > self.max_turns:
            return EnvStepOutput(observation=[], reward=tool_rewards, done=False, info={})

        sem = asyncio.Semaphore(self.max_parallel_calls)

        async def _guarded_call(tc: dict):
            async with sem:
                return await self._call_tool(tc, self.tools_kwargs)

        results: list[tuple[ToolResponse, float | None]] = await asyncio.gather(*[_guarded_call(tc) for tc in action])

        next_observation: list[dict] = []
        new_images_this_turn: list[Any] = []

        for tool_response, tool_reward in results:
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "tool_call_id": tool_response.id, "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "tool_call_id": tool_response.id, "content": tool_response.text or ""}

            next_observation.append(message)

            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)

            if tool_response.video:
                raise NotImplementedError("Video responses are not yet supported in ToolEnvironment.")

            if tool_reward is not None:
                tool_rewards.append(tool_reward)

        return EnvStepOutput(
            observation=next_observation,
            reward=tool_rewards,
            done=False,
            info={
                "multi_modal_data": {
                    "images": new_images_this_turn if new_images_this_turn else None,
                    "videos": None,  # Video handling not implemented
                }
            },
        )

    async def _call_tool(self, tool_call: dict, tools_kwargs: dict | None = None) -> tuple[ToolResponse, float | None]:
        """Call a single tool and format the response.

        Executes the specified tool with the provided arguments, handles errors,
        and truncates responses if needed based on configuration.

        Args:
            tool_call: Dictionary containing tool call information with structure:
                      {"function": {"name": str, "arguments": str (JSON)}}
            tools_kwargs: Optional per-sample kwargs keyed by tool name. When
                present, the matching entry is merged into the keyword arguments
                passed to the tool.

        Returns:
            tuple: (tool_message, tool_reward) where:
                - tool_message is a dict with role "tool" and content string
                - tool_reward is an optional float reward from the tool

        Note:
            - Catches and logs all exceptions during tool execution
            - Truncates long responses based on max_tool_response_length
            - Truncation side controlled by tool_response_truncate_side config
        """
        try:
            tool_name = tool_call["function"]["name"]
            tool_args = json.loads(tool_call["function"]["arguments"])
            tools_kwargs = tools_kwargs or self.tools_kwargs
            if tools_kwargs:
                extra = tools_kwargs.get(tool_name, {})
                if extra:
                    tool_args = {**extra, **tool_args}

            tool = self.tools.get_tool(tool_name)
            assert tool is not None, f"Tool {tool_name} not found in ToolGroup."
            tool_response = await tool(**tool_args)
            tool_reward = tool_response.output.get("score", None)
        except Exception as e:
            psrl_logger.warning(f"Error when executing tool: {e}")
            return ToolResponse(id=tool_call["id"], text=f"Error when executing tool: {e}"), 0.0

        # Truncate text response when needed.
        tool_response_text = tool_response.output.get("text", None)
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            elif self.tool_response_truncate_side == "right":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            else:
                half = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:half] + "...(truncated)..." + tool_response_text[-half:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text, "id": tool_call["id"]}
        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if attr_name in tool_response.output:
                attr_value = tool_response.output.get(attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        # Log each tool invocation with its input arguments and (possibly truncated) output.
        psrl_logger.info(
            f"Tool call. Name: {tool_name}, id: {tool_call['id']}, args: {tool_args}, "
            f"response: {tool_response_text}, reward: {tool_reward}"
        )
        return ToolResponse(**tool_response_kwargs), tool_reward

    async def close(self) -> None:
        """Cleans up resources used by the environment.

        Currently a no-op as tools don't require explicit cleanup.
        """
        return

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get the JSON schemas for all available tools.

        Returns:
            list: List of tool schema dictionaries in OpenAI function calling format
        """
        return self.tools.json

    @property
    def state(self) -> dict:
        """Get the current state of the environment.

        Returns:
            dict: Dictionary containing:
                - num_turn: Current turn number
                - max_turns: Maximum allowed turns
                - task: Current task dict
        """
        return {
            "num_turn": self.num_turn,
            "max_turns": self.max_turns,
            "task": self.task,
        }
