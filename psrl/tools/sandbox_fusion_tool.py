# Modified from verl/utils/reward_score/sandbox_fusion/__init__.py and utils.py
import asyncio
import json
import logging
import os
import traceback
import uuid
from typing import Any

import aiohttp

from psrl.tools.base import Tool, ToolOutput
from psrl.utils.rollout.rollout_trace import rollout_trace_op

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


@Tool.register("sandbox_fusion")
class SandboxFusionTool(Tool):
    """
    A tool for executing code using Sandbox Fusion service.

    This tool can be used in two modes:
    1. Tool call mode: Returns text output for multi-turn training
    2. Reward scoring mode: Returns numerical scores for reward computation

    The tool supports various programming languages and can execute code
    with test cases or standalone.
    """

    def __init__(
        self,
        sandbox_fusion_url: str,
        memory_limit_mb: int = 1024,
        default_timeout: int = DEFAULT_TIMEOUT,
        default_language: str = "python",
        name: str = "code_interpreter",
        description: str = "A tool for execute code",
        type: str = "native",
    ):
        """
        Initialize the SandboxFusionTool.

        Args:
            sandbox_fusion_url: The URL of the sandbox_fusion service
            memory_limit_mb: Memory limit in MB for code execution
            default_timeout: Timeout for each test case execution
            default_language: Programming language (default: "python")
            name: Tool name for registration
            description: Tool description
            type: Tool type
        """
        # Initialize the base Tool class
        super().__init__(name=name, description=description)

        self.sandbox_fusion_url = sandbox_fusion_url
        self.memory_limit_mb = memory_limit_mb
        self.timeout = default_timeout
        self.language = default_language
        self.type = type

        if not self.sandbox_fusion_url:
            raise ValueError("sandbox_fusion_url is required")

        psrl_logger.info(
            f"Initialized SandboxFusionTool with url={sandbox_fusion_url}, "
            f"memory_limit={memory_limit_mb}MB, timeout={default_timeout}s, language={default_language}"
        )

    @property
    def json(self) -> dict[str, Any]:
        """
        Return the tool schema in OpenAI function calling format.

        Returns:
            Dictionary containing the tool schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute.",
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    @rollout_trace_op
    async def async_forward(
        self,
        code: str,
        case_index: int = 0,
        stdin_data: dict | None = None,
        expected_output: Any | None = None,
        language: str | None = None,
        timeout: int | None = None,
        concurrent_semaphore: Any | None = None,
        fn_name: str | None = None,
        return_score: bool = False,
        **kwargs,
    ) -> ToolOutput:
        """
        Execute code using the Sandbox Fusion service.

        Args:
            code: The code string to execute.
            stdin_data: Optional stdin data for code execution.
            expected_output: Optional expected output for correctness check.
            language: Programming language, overriding the default when provided.
            timeout: Execution timeout, overriding the default when provided.
            return_score: Whether to return a numerical score instead of text output.
            **kwargs: Additional arguments.

        Returns:
            ToolOutput: Text output or a score with metadata.
        """
        lang = language or self.language
        exec_timeout = timeout or self.timeout

        if not isinstance(code, str):
            code = str(code)

        try:
            if stdin_data is None:
                result_status, metadata = await self._process_single_case(
                    case_index=case_index,
                    stdin_data=None,
                    expected_output=None,
                    sandbox_fusion_url=self.sandbox_fusion_url,
                    generation=code,
                    timeout=exec_timeout,
                    memory_limit_mb=self.memory_limit_mb,
                    language=lang,
                    concurrent_semaphore=concurrent_semaphore,
                    fn_name=fn_name,
                )

                if metadata["run_status"] == "Finished":
                    output_text = metadata["stdout"] + metadata["stderr"]
                else:
                    output_text = "no stdout here"
                # For tool call mode, return text output
                if not return_score:
                    return ToolOutput(
                        name=self.name,
                        output={
                            "text": output_text,
                            "result_status": result_status,
                        },
                        metadata=metadata,
                    )
                else:
                    # For reward mode without test cases, return based on execution status
                    score = 1.0 if result_status else 0.0
                    return ToolOutput(
                        name=self.name,
                        output={
                            "text": output_text,
                            "result_status": result_status,
                            "score": score,
                        },
                        metadata=metadata,
                    )
            else:
                try:
                    result_status, metadata = await self._process_single_case(
                        case_index=case_index,
                        stdin_data=stdin_data,
                        expected_output=expected_output,
                        sandbox_fusion_url=self.sandbox_fusion_url,
                        generation=code,
                        timeout=exec_timeout,
                        memory_limit_mb=self.memory_limit_mb,
                        language=lang,
                        concurrent_semaphore=concurrent_semaphore,
                        fn_name=fn_name,
                    )
                except Exception as exc:
                    psrl_logger.error(f"Test case={case_index} raised {exc}.")
                    traceback.print_exc()
                    result_status = -1  # Mark as API/internal error
                    metadata = {
                        "case_index": case_index,
                        "input": str(stdin_data),
                        "expected_output": str(expected_output) if expected_output else None,
                        "api_request_error": f"Internal execution error: {exc}",
                        "status": "internal_error",
                    }

                if return_score:
                    score = 1.0 if result_status else 0.0
                    return ToolOutput(
                        name=self.name,
                        output={
                            "result_status": result_status,
                            "score": score,
                        },
                        metadata=metadata,
                    )
                else:
                    return ToolOutput(
                        name=self.name,
                        output={
                            "result_status": result_status,
                        },
                        metadata=metadata,
                    )
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse test_cases JSON: {e}"
            psrl_logger.error(error_msg)
            return ToolOutput(name=self.name, error=error_msg)
        except Exception as e:
            error_msg = f"Error during code execution: {e}"
            psrl_logger.error(error_msg)
            return ToolOutput(name=self.name, error=error_msg)

    async def _process_single_case(
        self,
        case_index: int,
        stdin_data: Any,
        expected_output: Any,
        sandbox_fusion_url: str,
        generation: str,
        timeout: int,
        memory_limit_mb: int,
        language: str,
        concurrent_semaphore: asyncio.Semaphore | None = None,
        fn_name: str | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Helper function to process a single test case."""
        api_response = None
        error_msg = None
        psrl_logger.debug(f"Processing test case {case_index + 1}.")

        current_generation_code = generation

        if fn_name and language == "python":
            # Wrapper assumes stdin_data is a JSON string for function arguments.
            wrapper_code = f"""
    import traceback
    from string import *
    from re import *
    from datetime import *
    from collections import *
    from heapq import *
    from bisect import *
    from copy import *
    from math import *
    from random import *
    from statistics import *
    from itertools import *
    from functools import *
    from operator import *
    from io import *
    from sys import *
    from json import *
    from builtins import *
    from typing import *
    import string
    import re
    import datetime
    import collections
    import heapq
    import bisect
    import copy
    import math
    import random
    import statistics
    import itertools
    import functools
    import operator
    import io
    import sys
    import json

    # === User's Original Code START ===
    {generation}
    # === User's Original Code END ===

    _SANDBOX_FN_NAME = "{fn_name}"

    def _execute_user_function():
        # --- Input Parsing ---
        _raw_input_str = sys.stdin.read()
        _args = []
        if _raw_input_str.strip(): # If there's input
            try:
                _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
            except json.JSONDecodeError as _je:
                sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                                f"{{_raw_input_str[:200]}}\\n")
                return None, True # result, error_occurred

        # --- Function Location and Execution ---
        try:
            _target_callable = None
            # Try global scope first
            if _SANDBOX_FN_NAME in globals():
                _target_callable = globals()[_SANDBOX_FN_NAME]
            # Else, if 'Solution' class exists, try to get its method
            elif 'Solution' in globals():
                _Solution_class = globals()['Solution']
                # Attempt to instantiate and get method.
                # Errors (e.g., Solution not a class, instantiation fails, method missing)
                # will be caught by the broad except block below.
                _solution_instance = _Solution_class()
                _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

            if not _target_callable:
                sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
                return None, True # result, error_occurred

            _fn_result = _target_callable(*_args)
            return _fn_result, False # result, no_error
        except Exception: # Catches errors from Solution instantiation, getattr, or function call
            sys.stderr.write(
                f"Error during setup or execution of "
                f"'{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n"
            )
            return None, True # result, error_occurred

    if __name__ == '__main__':
        _result, _error_occurred = _execute_user_function()

        if not _error_occurred:
            # Serialize result to stdout
            if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
                print(json.dumps(_result))
            elif isinstance(_result, (int, float, str)):
                print(str(_result)) # Ensure string conversion for print
            else:
                # For other types, default to string representation.
                print(str(_result))
        # Optional: To explicitly exit with an error code if the sandbox relies on it
        # else:
        #    sys.exit(1)
    """
            current_generation_code = wrapper_code

        stdin = None if stdin_data is None else str(stdin_data)
        try:
            if concurrent_semaphore:
                async with concurrent_semaphore:
                    api_response, error_msg = await self.call_sandbox_api(
                        sandbox_fusion_url=sandbox_fusion_url,
                        code=current_generation_code,
                        stdin=stdin,
                        compile_timeout=timeout,
                        run_timeout=timeout,
                        memory_limit_mb=memory_limit_mb,
                        language=language,
                    )
            else:
                api_response, error_msg = await self.call_sandbox_api(
                    sandbox_fusion_url=sandbox_fusion_url,
                    code=current_generation_code,
                    stdin=stdin,
                    compile_timeout=timeout,
                    run_timeout=timeout,
                    memory_limit_mb=memory_limit_mb,
                    language=language,
                )
        except Exception as e:
            error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
            psrl_logger.error(f"Case {case_index + 1}: {error_msg}")
            traceback.print_exc()

        metadata = {
            "case_index": case_index,
            "input": stdin,
            "expected_output": str(expected_output) if expected_output else None,
            "api_request_error": error_msg,
            "api_response": None,
            "status": "unknown",
            "stdout": None,
            "stderr": None,
            "exit_code": None,
            "duration": None,
            "compile_duration": None,
            "compile_stderr": None,
            "api_status": None,
            "compile_status": None,
            "run_status": None,
        }
        result_status = -1  # Default error: API request error or unknown sandbox error

        if error_msg:
            metadata["status"] = "api_error"
            result_status = -1  # API request itself failed (includes timeout after retries)
            psrl_logger.error(f"Case {case_index}: API error occurred: {error_msg}")
            # Log code and input only on error for brevity
            generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
            psrl_logger.error(f"Case {case_index}: code: {generation_to_log}")
            psrl_logger.error(f"Case {case_index}: input: {stdin}")
        elif api_response:
            # --- Add debug logging ---
            psrl_logger.debug(f"Case {case_index}: API Response: {api_response}")
            metadata["api_response"] = api_response
            metadata["api_status"] = api_response.get("status")
            compile_result = api_response.get("compile_result")
            run_result = api_response.get("run_result")

            # Extract compile information
            if compile_result:
                metadata["compile_status"] = compile_result.get("status")
                metadata["compile_duration"] = compile_result.get("execution_time")
                metadata["compile_stderr"] = compile_result.get("stderr")

            # Extract run information
            if run_result:
                metadata["run_status"] = run_result.get("status")
                metadata["stdout"] = run_result.get("stdout")
                metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
                metadata["exit_code"] = run_result.get("return_code")
                metadata["duration"] = run_result.get("execution_time")

            # --- Determine status based on API response ---
            api_status = metadata["api_status"]

            if api_status == "SandboxError":
                metadata["status"] = "sandbox_error"
                result_status = -1  # Internal sandbox error
            elif api_status == "Failed":
                # --- Add debug logging ---
                psrl_logger.debug(f"API returned Failed status. Response: {api_response}")
                psrl_logger.debug(f"Compile Result: {compile_result}")
                psrl_logger.debug(f"Run Result: {run_result}")
                # --- Check the logic here ---
                # Compile failed or timed out
                is_compile_error = compile_result and (
                    metadata["compile_status"] in ["Error", "TimeLimitExceeded"]
                    or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
                )
                if is_compile_error:
                    # Differentiate between compile_error and compile_timeout based on specific status
                    if metadata["compile_status"] == "TimeLimitExceeded":
                        metadata["status"] = "compile_timeout"
                    else:  # Includes Error and Finished but return_code != 0 cases
                        metadata["status"] = "compile_error"
                    result_status = -4
                # Run failed or timed out
                elif run_result:
                    # Modified condition: Check for TimeLimitExceeded OR
                    # (Finished with non-zero exit code) OR Error status
                    is_runtime_error = (
                        metadata["run_status"] == "TimeLimitExceeded"
                        or metadata["run_status"] == "Error"
                        or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                    )
                    if is_runtime_error:
                        if metadata["run_status"] == "TimeLimitExceeded":
                            metadata["status"] = "timeout"  # Runtime timeout
                            result_status = -3
                        else:  # Includes Error and Finished with non-zero return_code
                            metadata["status"] = "runtime_error"
                            result_status = -2
                    else:
                        # Other Failed status with run_result, classify as unknown failure
                        psrl_logger.warning(
                            f"Unknown run_status '{metadata['run_status']}' or state within Failed API status."
                        )
                        metadata["status"] = "unknown_failure"
                        result_status = -1  # Default to -1
                else:
                    # Status is Failed but neither a clear compile error nor run_result exists
                    psrl_logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                    metadata["status"] = "unknown_failure_state"
                    result_status = -1  # Default to -1
            elif api_status == "Success":
                # Run completed successfully, now check the answer
                if run_result and metadata["run_status"] == "Finished":
                    actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                    # Note: Output might contain trailing newlines, need normalization
                    if expected_output is None or str(actual_output).rstrip("\n") == str(expected_output).rstrip("\n"):
                        result_status = True
                        metadata["status"] = "success"
                    else:
                        result_status = False
                        metadata["status"] = "wrong_answer"
                else:
                    # Status is Success but run_result status is not Finished, this is unexpected
                    metadata["status"] = "unexpected_success_state"
                    result_status = -1  # Classify as unknown error
            else:
                # API returned an unknown top-level status
                psrl_logger.warning(f"Unknown API status received: {api_status}")
                metadata["status"] = f"unknown_api_status_{api_status}"
                result_status = -1  # Default to -1
        else:  # api_response is None and no error_msg (Should not happen with current call_sandbox_api logic)
            metadata["status"] = "unknown_api_state"
            result_status = -1
            psrl_logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
        return result_status, metadata

    async def call_sandbox_api(
        self,
        sandbox_fusion_url: str,
        code: str,
        stdin: str | None,
        compile_timeout: int,
        run_timeout: int,
        memory_limit_mb: int,
        language: str = "python",
    ) -> tuple[dict[str, Any] | None, str | None]:
        """
        Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
        using increasing delay between retries. Logs internal calls with a unique ID.
        This is an async version using aiohttp for concurrent requests.

        Args:
            sandbox_fusion_url: The URL of the sandbox fusion API.
            code: The code string to execute.
            stdin: The standard input string.
            compile_timeout: Compile timeout in seconds.
            run_timeout: Run timeout in seconds.
            memory_limit_mb: Memory limit in MB.
            language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".

        Returns:
            A tuple (response_json, error_message).
            If successful, response_json is the API's returned JSON object, error_message is None.
            If failed after retries, response_json is None, error_message contains the error information.
        """
        request_id = str(uuid.uuid4())
        log_prefix = f"[Request ID: {request_id}]"

        if language not in SUPPORTED_LANGUAGES:
            error_msg = f"{log_prefix} Unsupported language: {language}"
            psrl_logger.error(error_msg)
            return None, error_msg

        payload = {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,
            "files": {},
            "fetch_files": [],
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
        request_timeout = compile_timeout + run_timeout + API_TIMEOUT

        last_error = None  # Store the last error encountered

        # Create a timeout for aiohttp
        timeout = aiohttp.ClientTimeout(total=request_timeout)

        for attempt in range(MAX_RETRIES):
            try:
                psrl_logger.debug(
                    f"{log_prefix} Attempt {attempt + 1}/{MAX_RETRIES}: Calling sandbox API at {sandbox_fusion_url}"
                )

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        sandbox_fusion_url,
                        headers=headers,
                        json=payload,  # aiohttp automatically serializes to JSON
                    ) as response:
                        # Check for Gateway Timeout (504) specifically for retrying
                        if response.status == 504:
                            last_error = (
                                f"{log_prefix} API Request Error: Gateway Timeout (504) on attempt "
                                f"{attempt + 1}/{MAX_RETRIES}"
                            )
                            psrl_logger.warning(last_error)
                            if attempt < MAX_RETRIES - 1:  # Don't sleep after the last attempt
                                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                                psrl_logger.info(f"{log_prefix}: Retrying after {delay}s...")
                                await asyncio.sleep(delay)
                            continue  # Go to the next retry attempt

                        # Check for other HTTP errors (e.g., 4xx, other 5xx)
                        response.raise_for_status()

                        # If successful (status code 2xx)
                        psrl_logger.debug(f"{log_prefix} Sandbox API call successful on attempt {attempt + 1}")
                        response_json = await response.json()
                        return response_json, None

            except aiohttp.ClientResponseError as e:
                last_error = f"{log_prefix} API Response Error: {e.status} - {e.message}"
                psrl_logger.error(last_error)
                break  # Exit retry loop on non-504 response errors
            except aiohttp.ClientError as e:
                last_error = f"{log_prefix} API Client Error: {e}"
                psrl_logger.error(last_error)
                break  # Exit retry loop on client errors
            except asyncio.TimeoutError:
                last_error = f"{log_prefix} API Request Timeout after {request_timeout}s"
                psrl_logger.error(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    psrl_logger.info(f"{log_prefix}: Retrying after timeout in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                break
            except Exception as e:
                last_error = f"{log_prefix} Unexpected Error: {e}"
                psrl_logger.error(last_error)
                break

        # If loop finishes without returning success, return the last recorded error
        psrl_logger.error(f"{log_prefix} Sandbox API call failed. Last error: {last_error}")
        # Return the error message without the prefix, as the caller doesn't need the internal ID
        # Ensure API call failure returns error message, leading to -1 in check_correctness
        return None, last_error.replace(
            log_prefix, "API Call Failed: "
        ) if last_error else "API Call Failed after retries"
