import logging
import os

import regex

from psrl.tools.base import ToolCall
from psrl.tools.tool_parser.base import ToolParser
from psrl.utils.rollout.rollout_trace import rollout_trace_op

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("gpt-oss")
class GptOssToolParser(ToolParser):
    """Tool parser for gpt-oss models.

    Adapted from verl.experimental.agent_loop.tool_parser.GptOssToolParser.
    The Harmony format keeps tool-call control tokens in the decoded text, so
    extraction must decode with ``skip_special_tokens=False``.
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.cot_pattern = regex.compile(
            r"<\|start\|>assistant<\|channel\|>analysis<\|message\|>.*?<\|end\|>",
            regex.DOTALL,
        )
        self.partial_cot_pattern = regex.compile(
            r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>",
            regex.DOTALL,
        )
        self.tool_call_pattern = regex.compile(
            r"<\|start\|>assistant<\|channel\|>[^<]* to=functions\.([^<]+) "
            r"<\|constrain\|>json<\|message\|>(.*?)<\|call\|>",
            regex.DOTALL,
        )

    @rollout_trace_op
    def extract_tool_calls_from_token_ids(
        self,
        responses_ids: list[int],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        response_str = self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        if self.tokenizer.pad_token:
            response_str = response_str.replace(self.tokenizer.pad_token, "")
        return self.extract_tool_calls_from_str(response_str, tools=tools)

    def extract_tool_calls_from_str(
        self,
        response_str: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        response_str = regex.sub(self.cot_pattern, "", response_str)
        response_str = regex.sub(self.partial_cot_pattern, "", response_str)

        matches = regex.findall(self.tool_call_pattern, response_str)
        if not matches:
            return response_str, []

        tool_calls = []
        for name, arguments in matches:
            try:
                tool_calls.append(ToolCall(name=name, arguments=arguments))
            except Exception as e:
                psrl_logger.warning(f"Failed to decode gpt-oss tool call: {e}")

        content = regex.sub(self.tool_call_pattern, "", response_str)
        return content, tool_calls
