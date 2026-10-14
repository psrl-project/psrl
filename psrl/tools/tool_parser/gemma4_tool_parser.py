import json
import logging
import os

import regex

from psrl.tools.base import ToolCall
from psrl.tools.tool_parser.base import ToolParser
from psrl.utils.rollout.rollout_trace import rollout_trace_op

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("gemma4")
class Gemma4ToolParser(ToolParser):
    """Tool parser for Google Gemma 4 models.

    Format:
        <|tool_call>call:func_name{key:<|"|>str_val<|"|>,key2:num_val}<tool_call|>

    Adapted from verl.experimental.agent_loop.tool_parser.Gemma4ToolParser.
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tool_call_start_token = "<|tool_call>"
        self.tool_call_end_token = "<tool_call|>"
        self._stop_token_id = tokenizer.convert_tokens_to_ids(self.tool_call_end_token)
        self.tool_call_regex = regex.compile(r"<\|tool_call>call:([\w.-]+)\{(.*?)\}<tool_call\|>", regex.DOTALL)
        self.arg_regex = regex.compile(r'([\w.-]+):(?:<\|"\|>(.*?)<\|"\|>|([^,}]*))')

    @property
    def stop_token_ids(self) -> list[int]:
        return [self._stop_token_id]

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
        if self.tool_call_start_token not in response_str:
            return response_str, []

        matches = self.tool_call_regex.findall(response_str)
        if not matches:
            return response_str, []

        tool_calls = []
        for name, args_str in matches:
            try:
                arguments = self._parse_arguments(args_str)
                tool_calls.append(ToolCall(name=name, arguments=json.dumps(arguments, ensure_ascii=False)))
            except Exception as e:
                psrl_logger.warning(f"Failed to parse Gemma4 tool call: {e}")

        content_index = response_str.find(self.tool_call_start_token)
        content = response_str[:content_index] if content_index >= 0 else response_str
        return content, tool_calls

    def _parse_arguments(self, args_str: str) -> dict:
        result = {}
        for match in self.arg_regex.finditer(args_str):
            key = match.group(1)
            str_val, bare_val = match.group(2), match.group(3)
            if str_val is not None:
                result[key] = str_val
                continue
            if bare_val is None:
                continue
            bare_val = bare_val.strip()
            if bare_val.lower() == "true":
                result[key] = True
            elif bare_val.lower() == "false":
                result[key] = False
            elif bare_val.lower() == "null":
                result[key] = None
            else:
                try:
                    result[key] = int(bare_val)
                except ValueError:
                    try:
                        result[key] = float(bare_val)
                    except ValueError:
                        result[key] = bare_val
        return result
