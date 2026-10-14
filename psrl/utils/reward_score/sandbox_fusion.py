import asyncio
import json
import logging
import os
import traceback
from typing import Any

from psrl.tools.base import Tool
from psrl.tools.sandbox_fusion_tool import DEFAULT_TIMEOUT

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


async def compute_score(
    completion,
    test_cases,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    continuous=False,
    timeout=10,
):
    """
    Compute the code score with the remote `SandboxFusionTool`.

    Args:
        completion: The completion string containing the code.
        test_cases: JSON string or dictionary containing "inputs" and "outputs".
        sandbox_fusion_url: The URL of the sandbox_fusion service, eg: "https://<your service endpoint>/run_code"
        concurrent_semaphore: Optional semaphore for rate limiting (currently unused in tool version).
        memory_limit_mb: Memory limit in MB for code execution.
        continuous: Whether to compute a continuous score (based on the first N test cases).
        timeout: Timeout for each test case.

    Returns:
        A tuple (score, metadata_list).
        score: Float score (0.0 to 1.0).
        metadata_list: List containing execution metadata for each test case.
    """
    solution = completion
    if "```python" in completion:
        solution = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        # Accept unlabeled fenced blocks such as ````\ncode\n````.
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():
                    solution = rest
    else:
        return 0.0, [{"error": "Invalid completion (missing code block)"}]

    try:
        if not isinstance(test_cases, dict):
            try:
                test_cases = json.loads(test_cases)
            except json.JSONDecodeError as e:
                psrl_logger.error(f"Failed to parse test_cases JSON: {e}")
                return 0.0, [{"error": "Invalid test_cases JSON format"}]

        if test_cases is not None and "assert_case" in test_cases and isinstance(test_cases.get("assert_case"), list):
            assert_cases = test_cases.get("assert_case")
            test_cases.setdefault("inputs", ["" for _ in assert_cases])
            test_cases.setdefault("outputs", [None for _ in assert_cases])
        elif not test_cases or "inputs" not in test_cases or "outputs" not in test_cases:
            psrl_logger.error("Invalid test_cases structure.")
            return 0.0, [{"error": "Invalid test_cases structure (missing inputs/outputs)"}]

        res_list, metadata_list = await check_correctness(
            sandbox_fusion_url=sandbox_fusion_url,
            in_outs=test_cases,
            generation=solution,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            memory_limit_mb=memory_limit_mb,
        )

        if not res_list:
            return 0.0, metadata_list

        if continuous:
            num_to_consider = min(len(res_list), 10)
            if num_to_consider == 0:
                score = 0.0
            else:
                passed_count = sum(1 for r in res_list[:num_to_consider] if r is True)
                score = passed_count / num_to_consider
            final_metadata = metadata_list
        else:
            passed_count = sum(1 for r in res_list if r is True)
            total_cases = len(res_list)
            score = passed_count / total_cases if total_cases > 0 else 0.0
            final_metadata = metadata_list

    except Exception as e:
        psrl_logger.error(f"Error during compute_score: {e}")
        traceback.print_exc()
        score = 0.0
        final_metadata = metadata_list if "metadata_list" in locals() else [{"error": f"Unhandled exception: {e}"}]

    return float(score), final_metadata if isinstance(final_metadata, list) else [final_metadata]


async def check_correctness(
    sandbox_fusion_url: str,
    in_outs: dict | None,
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: asyncio.Semaphore | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Checks the correctness of code generation using the remote sandbox API,
    processing test cases concurrently with async/await.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        in_outs: Dictionary containing "inputs" and "outputs" lists.
        generation: The generated code string.
        timeout: Timeout for each test case (compile and run share this timeout).
        memory_limit_mb: Memory limit in MB.
        language: The programming language of the code.
        concurrent_semaphore: Optional asyncio.Semaphore for limiting concurrency.

    Returns:
        A tuple (results, metadata_list).
        results: A list containing the test result for each input/output pair
                 (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                 Results are ordered corresponding to the inputs.
        metadata_list: A list containing metadata dictionaries for each test case,
                       ordered corresponding to the inputs.
    """
    psrl_logger.info("Starting correctness check for generation.")

    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        psrl_logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    assert_cases = in_outs.get("assert_case", [""] * num_cases)
    results = [None] * num_cases
    metadata_list = [None] * num_cases

    if num_cases == 0:
        psrl_logger.warning("Empty inputs provided.")
        return [], []

    if len(inputs) != len(expected_outputs):
        psrl_logger.warning(
            f"Mismatch between number of inputs ({len(inputs)}) and outputs ({len(expected_outputs)})."
        )
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    if len(assert_cases) != num_cases:
        psrl_logger.warning(
            f"Mismatch between number of assert cases ({len(assert_cases)}) and inputs/outputs ({num_cases})."
        )
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    first_compile_error_index = -1

    tool = Tool.get_tool(
        "sandbox_fusion",
        sandbox_fusion_url=sandbox_fusion_url,
        memory_limit_mb=memory_limit_mb,
        default_timeout=timeout,
    )
    assert tool.has_async_forward, "SandboxFusionTool must have async_forward implemented."

    tasks = [
        tool(
            code=generation + "\n\n" + assert_cases[i],
            case_index=i,
            stdin_data=stdin_data,
            expected_output=expected_outputs[i],
            language=language,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            fn_name=fn_name,
            return_score=False,
        )
        for i, stdin_data in enumerate(inputs)
    ]

    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, task_result in enumerate(task_results):
        if task_result.error:
            psrl_logger.error(f"Test case index={i} generated an error: {task_result.error}")
            results[i] = -1
            metadata_list[i] = {
                "case_index": i,
                "input": str(inputs[i]),
                "expected_output": str(expected_outputs[i]) if expected_outputs[i] else None,
                "api_request_error": f"Internal execution error: {task_result.error}",
                "status": "internal_error",
            }
        else:
            result_status = task_result.output["result_status"]
            metadata = task_result.metadata
            results[i] = result_status
            metadata_list[i] = metadata

            if result_status == -4:
                if first_compile_error_index == -1 or i < first_compile_error_index:
                    first_compile_error_index = i

    if first_compile_error_index != -1:
        psrl_logger.warning(
            f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors."
        )
        for i in range(first_compile_error_index + 1, num_cases):
            if results[i] != -4:
                results[i] = -4
                if metadata_list[i] is None:
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]) if expected_outputs[i] else None,
                        "api_request_error": None,
                        "status": "compile_error_skipped",
                    }
                else:
                    metadata_list[i]["status"] = "compile_error_skipped"

    psrl_logger.info(f"Correctness check finished. Results: {results}")
    return results, metadata_list
