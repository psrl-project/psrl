from abc import ABC, abstractmethod

from psrl.tools.base import ToolCall


class ToolParser(ABC):
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @property
    def stop_token_ids(self) -> list[int]:
        """
        Return token IDs that should stop generation before parsing.

        Models like Qwen3 naturally emit EOS after a tool call, so no extra stop
        tokens are needed. Models like Gemma4 emit <tool_call|> but continue
        generating without EOS and need the closing token as an explicit stop.

        Returns empty list by default (rely on model's EOS behavior).
        """
        return []

    @abstractmethod
    def extract_tool_calls_from_token_ids(
        self,
        responses_ids: list[int],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Extract tool calls from the responses.

        Args:
            responses_ids (List[int]): The ids of the responses.
            tools: OpenAI function tool schemas, when a parser needs schema-aware
                argument conversion.

        Returns:
            tuple[str, List[ToolCall]]: Extracted text and tool calls.
        """
        raise NotImplementedError

    @abstractmethod
    def extract_tool_calls_from_str(
        self,
        response_str: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        """Extract tool calls from the response string.

        Args:
            response_str (str): The response string.
            tools: OpenAI function tool schemas, when a parser needs schema-aware
                argument conversion.

        Returns:
            tuple[str, List[ToolCall]]: Extracted text and tool calls.
        """
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        def decorator(subclass: type[ToolParser]) -> type[ToolParser]:
            cls._registry[name] = subclass
            return subclass

        return decorator
