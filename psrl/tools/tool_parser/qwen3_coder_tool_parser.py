import ast
import json
import logging
import os
from typing import Any

import regex

from psrl.tools.base import ToolCall
from psrl.tools.tool_parser.base import ToolParser
from psrl.utils.rollout.rollout_trace import rollout_trace_op

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ToolParser.register("qwen3_coder")
class Qwen3XMLToolParser(ToolParser):
    """Tool parser for Qwen3-Coder/Qwen3.5 XML-style tool calls.

    Adapted from verl.experimental.agent_loop.tool_parser.Qwen3XMLToolParser.
    PSRL passes OpenAI tool schemas as plain dictionaries, so schema lookup is
    implemented against dicts instead of verl's Pydantic schema objects.
    """

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_prefix = "<function="

        self.tool_call_complete_regex = regex.compile(r"<tool_call>(.*?)</tool_call>", regex.DOTALL)
        self.tool_call_regex = regex.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", regex.DOTALL)
        self.tool_call_function_regex = regex.compile(r"<function=(.*?)</function>|<function=(.*)$", regex.DOTALL)
        self.tool_call_parameter_regex = regex.compile(r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", regex.DOTALL)

    @rollout_trace_op
    def extract_tool_calls_from_token_ids(
        self,
        responses_ids: list[int],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        response_str = self.tokenizer.decode(responses_ids)
        return self.extract_tool_calls_from_str(response_str, tools=tools)

    def extract_tool_calls_from_str(
        self,
        response_str: str,
        tools: list[dict] | None = None,
    ) -> tuple[str, list[ToolCall]]:
        if self.tool_call_start_token not in response_str and self.tool_call_prefix not in response_str:
            return response_str, []

        try:
            function_calls = self._get_function_calls(response_str)
            if not function_calls or len(function_calls) == 0:
                return response_str, []

            tool_calls = [
                self._parse_xml_function_call(function_call_str, tools) for function_call_str in function_calls
            ]

            content_index = response_str.find(self.tool_call_start_token)
            content_index = content_index if content_index >= 0 else response_str.find(self.tool_call_prefix)
            content = response_str[:content_index]

            return content, tool_calls
        except Exception as e:
            psrl_logger.exception(f"Error in extracting qwen3-coder tool call from response: {e}")
            return response_str, []

    def _get_function_calls(self, model_output: str) -> list[str]:
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [match[0] if match[0] else match[1] for match in matched_ranges]
        if not raw_tool_calls:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))
        return [match[0] if match[0] else match[1] for match in raw_function_calls]

    def _parse_xml_function_call(self, function_call_str: str, tools: list[dict] | None) -> ToolCall:
        end_index = function_call_str.index(">")
        function_name = function_call_str[:end_index]
        param_config = self._get_arguments_config(function_name, tools)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}

        for match in self.tool_call_parameter_regex.findall(parameters):
            match_text = match[0] if match[0] else match[1]
            idx = match_text.index(">")
            param_name = match_text[:idx]
            param_value = str(match_text[idx + 1 :])
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value,
                param_name,
                param_config,
                function_name,
            )

        return ToolCall(name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False))

    def _get_arguments_config(self, func_name: str, tools: list[dict] | None) -> dict[str, dict]:
        for config in tools or []:
            if not isinstance(config, dict) or config.get("type") != "function":
                continue
            function = config.get("function", {})
            if function.get("name") != func_name:
                continue
            parameters = function.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            return {str(k): v for k, v in properties.items() if isinstance(v, dict)}
        psrl_logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _convert_param_value(
        self,
        param_value: str,
        param_name: str,
        param_config: dict[str, dict],
        func_name: str,
    ) -> Any:
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config:
                psrl_logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in tool '{func_name}', "
                    "returning the string value."
                )
            return param_value

        param_type = str(param_config[param_name].get("type", "string")).strip().lower()
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        if (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except Exception:
                psrl_logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer "
                    f"in tool '{func_name}', degenerating to string."
                )
                return param_value
        if param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_value = float(param_value)
                return float_value if float_value - int(float_value) != 0 else int(float_value)
            except Exception:
                psrl_logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float "
                    f"in tool '{func_name}', degenerating to string."
                )
                return param_value
        if param_type in ["boolean", "bool", "binary"]:
            lowered = param_value.lower()
            if lowered not in ["true", "false"]:
                psrl_logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not boolean "
                    f"in tool '{func_name}', degenerating to false."
                )
            return lowered == "true"

        if param_type == "object" or param_type.startswith("dict"):
            try:
                return json.loads(param_value)
            except Exception:
                psrl_logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a valid JSON object "
                    f"in tool '{func_name}', trying literal parsing."
                )
        try:
            return ast.literal_eval(param_value)
        except Exception:
            psrl_logger.warning(
                f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted "
                f"in tool '{func_name}', degenerating to string."
            )
            return param_value
