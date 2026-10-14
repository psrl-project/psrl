# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Register lightweight function-based tools for PSRL.

Registration and schema inference follow `verl.tools.utils.function_tool`, while
runtime values use the PSRL `Tool` and `ToolOutput` interfaces.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import sys
from collections.abc import Callable
from typing import Any

from transformers.utils import get_json_schema
from verl.tools.schemas import OpenAIFunctionToolSchema

from psrl.tools.base import Tool, ToolOutput, ToolResponse

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

FUNCTION_TOOL_REGISTRY: dict[str, FunctionTool] = {}
_LOADED_FUNCTION_TOOL_PATHS: dict[str, list[FunctionTool]] = {}


class FunctionTool(Tool):
    """PSRL tool wrapper around a plain Python function."""

    def __init__(self, name: str, fn: Callable[..., Any], tool_schema: Any):
        self.fn = fn
        self.tool_schema = tool_schema
        self.is_async = inspect.iscoroutinefunction(fn)
        super().__init__(name=name, description=_schema_description(tool_schema))

    @property
    def json(self) -> dict[str, Any]:
        if hasattr(self.tool_schema, "model_dump"):
            return self.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
        return self.tool_schema

    async def async_forward(self, *args, **kwargs) -> ToolOutput:
        if args:
            raise TypeError("@function_tool only supports keyword arguments")
        if self.is_async:
            raw = await self.fn(**kwargs)
        else:
            raw = await asyncio.to_thread(self.fn, **kwargs)
        return normalize_function_tool_return(raw, self.name or self.fn.__name__)


def function_tool(
    name: str | Callable | None = None,
    *,
    schema: OpenAIFunctionToolSchema | dict | None = None,
):
    """Register a Python function as a PSRL tool.

    The schema inference behavior mirrors verl's ``@function_tool`` decorator:
    Google-style docstrings and type hints are parsed by
    ``transformers.utils.get_json_schema``.
    """

    def _make_decorator(tool_name_override: str | None):
        def decorator(fn: Callable):
            tool_name = tool_name_override or fn.__name__

            if isinstance(schema, OpenAIFunctionToolSchema):
                built_schema = schema
            elif isinstance(schema, dict):
                built_schema = OpenAIFunctionToolSchema.model_validate(schema)
            else:
                built_schema = _build_schema_from_fn(fn, tool_name)

            entry = FunctionTool(
                name=tool_name,
                fn=fn,
                tool_schema=built_schema,
            )
            existing = FUNCTION_TOOL_REGISTRY.get(tool_name)
            if existing is not None and existing.fn is not fn:
                raise ValueError(
                    f"Function tool '{tool_name}' is already registered to "
                    f"{existing.fn.__module__}.{existing.fn.__qualname__}; "
                    f"refusing to overwrite with {fn.__module__}.{fn.__qualname__}."
                )
            FUNCTION_TOOL_REGISTRY[tool_name] = entry
            psrl_logger.info("Registered function tool '%s' from %s.%s", tool_name, fn.__module__, fn.__qualname__)
            return fn

        return decorator

    if callable(name) and schema is None:
        return _make_decorator(None)(name)

    return _make_decorator(name)


def get_function_tool(name: str) -> FunctionTool:
    if name not in FUNCTION_TOOL_REGISTRY:
        raise KeyError(
            f"Function tool '{name}' not found in registry. Make sure its defining "
            "file is referenced via rollout.multi_turn.function_tool_path."
        )
    return FUNCTION_TOOL_REGISTRY[name]


def load_function_tools_from_path(path: str) -> list[FunctionTool]:
    """Execute ``path`` and return tools registered by ``@function_tool``."""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"function_tool_path does not exist: {path}")

    if abs_path in _LOADED_FUNCTION_TOOL_PATHS:
        return _LOADED_FUNCTION_TOOL_PATHS[abs_path]

    before = set(FUNCTION_TOOL_REGISTRY)
    module_name = "_psrl_function_tools_" + abs_path.replace(os.sep, "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for function_tool_path: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    new_names = sorted(set(FUNCTION_TOOL_REGISTRY) - before)
    if not new_names:
        psrl_logger.warning(
            "function_tool_path '%s' loaded without any @function_tool decorators. "
            "Did you forget to apply the decorator?",
            path,
        )
    else:
        psrl_logger.info("Loaded %d function tool(s) from %s: %s", len(new_names), path, new_names)

    tools = [FUNCTION_TOOL_REGISTRY[name] for name in new_names]
    _LOADED_FUNCTION_TOOL_PATHS[abs_path] = tools
    return tools


def _build_schema_from_fn(fn: Callable, tool_name: str) -> OpenAIFunctionToolSchema:
    sig = inspect.signature(fn)
    variadic = [
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if variadic:
        raise ValueError(
            f"@function_tool '{tool_name}' ({fn.__module__}.{fn.__qualname__}) "
            f"declares variadic parameter(s) {variadic}, which cannot be expressed "
            "in an OpenAI tool schema."
        )

    raw = get_json_schema(fn)
    raw["function"]["name"] = tool_name
    return OpenAIFunctionToolSchema.model_validate(raw)


def _schema_description(schema: Any) -> str:
    if hasattr(schema, "function"):
        return schema.function.description
    if isinstance(schema, dict):
        return schema.get("function", {}).get("description", "")
    return ""


def normalize_function_tool_return(ret: Any, tool_name: str) -> ToolOutput:
    """Coerce a function return into PSRL's ``ToolOutput`` contract."""
    response, reward, metrics = _normalize_response_reward_metrics(ret)
    output = {"text": response.text, "score": reward}
    if response.image is not None:
        output["image"] = response.image
    if response.video is not None:
        output["video"] = response.video
    if metrics:
        output["metadata"] = metrics
    return ToolOutput(name=tool_name, output=output)


def _normalize_response_reward_metrics(ret: Any) -> tuple[ToolResponse, float, dict]:
    if isinstance(ret, ToolOutput):
        return _tool_output_to_response(ret)
    if isinstance(ret, ToolResponse):
        return ret, 0.0, {}
    if isinstance(ret, str):
        return ToolResponse(text=ret), 0.0, {}
    if isinstance(ret, dict):
        return ToolResponse(text=json.dumps(ret, ensure_ascii=False)), 0.0, {}
    if isinstance(ret, tuple):
        if not 1 <= len(ret) <= 3:
            raise TypeError(
                f"@function_tool return tuple must have length 1, 2, or 3 (got length {len(ret)}: {ret!r})."
            )
        response = _coerce_response(ret[0])
        reward = 0.0 if len(ret) < 2 or ret[1] is None else float(ret[1])
        metrics = {} if len(ret) < 3 or ret[2] is None else dict(ret[2])
        return response, reward, metrics
    return ToolResponse(text=str(ret)), 0.0, {}


def _tool_output_to_response(ret: ToolOutput) -> tuple[ToolResponse, float, dict]:
    output = ret.output or {}
    response = ToolResponse(
        text=output.get("text") if isinstance(output, dict) else str(output),
        image=output.get("image") if isinstance(output, dict) else None,
        video=output.get("video") if isinstance(output, dict) else None,
    )
    reward = float(output.get("score", 0.0)) if isinstance(output, dict) else 0.0
    metrics = output.get("metadata", {}) if isinstance(output, dict) else {}
    return response, reward, metrics


def _coerce_response(value: Any) -> ToolResponse:
    if isinstance(value, ToolResponse):
        return value
    if isinstance(value, ToolOutput):
        return _tool_output_to_response(value)[0]
    if isinstance(value, str):
        return ToolResponse(text=value)
    if isinstance(value, dict):
        return ToolResponse(text=json.dumps(value, ensure_ascii=False))
    return ToolResponse(text=str(value))
