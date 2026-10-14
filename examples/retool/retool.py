import hashlib
import logging
import os
import re
from typing import Any

import datasets
from psrl.tools.base import ToolOutput
from psrl.tools.sandbox_fusion_tool import DEFAULT_TIMEOUT, SandboxFusionTool
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo


class CustomSandboxFusionTool(SandboxFusionTool):
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
        super().__init__(
            sandbox_fusion_url=sandbox_fusion_url,
            memory_limit_mb=memory_limit_mb,
            default_timeout=default_timeout,
            default_language=default_language,
            name=name,
            description=description,
            type=type,
        )
        self.code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

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
        if not isinstance(code, str):
            code = str(code)

        matches = self.code_pattern.findall(code)
        if matches:
            code = matches[0].strip()

        # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
        lines = code.split("\n")
        for i, line in reversed(list(enumerate(lines))):
            if line == "":
                continue
            if not lines[i].startswith("print"):
                lines[i] = f"print({line})"
            break
        code = "\n".join(lines)

        return await super().async_forward(
            code=code,
            case_index=case_index,
            stdin_data=stdin_data,
            expected_output=expected_output,
            language=language,
            timeout=timeout,
            concurrent_semaphore=concurrent_semaphore,
            fn_name=fn_name,
            return_score=return_score,
            **kwargs,
        )


answer_format = """\nThe answer format must be: \\boxed{'The final answer goes here.'}"""

logger = logging.getLogger(__name__)


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process dapo/aime-2024, dapo/aime-2025 datasets."""

    AGENT_NAME = "multi_turn_completion_agent"

    def _get_cache_file_path(self, parquet_file: str) -> str:
        """Generate cache file path based on parquet file path and agent_name."""
        # Include agent_name in the hash so cache invalidates when it changes
        cache_key = f"{parquet_file}|agent_name={self.AGENT_NAME}"
        file_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_dir = os.path.join(self.cache_dir, "processed_datasets")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{file_hash}.parquet")
        return cache_file

    def _load_from_cache(self, cache_file: str) -> datasets.Dataset | None:
        """Load dataset from cache file if it exists."""
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading dataset from cache: {cache_file}")
                dataframe = datasets.load_dataset("parquet", data_files=cache_file)["train"]
                return dataframe
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
                return None
        return None

    def _save_to_cache(self, dataframe: datasets.Dataset, cache_file: str):
        """Save processed dataset to cache file."""
        try:
            logger.info(f"Saving dataset to cache: {cache_file}")
            dataframe.to_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Failed to save cache file {cache_file}: {e}")

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # Generate cache file path
            cache_file = self._get_cache_file_path(parquet_file)

            # Try to load from cache first
            dataframe = self._load_from_cache(cache_file)

            if dataframe is None:
                # Cache miss, need to process
                logger.info(f"Cache miss for {parquet_file}, processing...")
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
                data_source = "/".join(parquet_file.split("/")[-2:]).split(".")[0]  # e.g., dapo/aime-2024
                if data_source in ["dapo/aime-2024-raw", "dapo/aime-2024", "dapo/aime-2025"]:
                    dataframe = dataframe.map(
                        self.map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                    )
                else:
                    dataframe = dataframe.map(self.map_fn2, num_proc=16)

                # Save to cache for future use
                self._save_to_cache(dataframe, cache_file)
            else:
                logger.info(f"Cache hit for {parquet_file}, skipping map operation")

            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

    def map_fn(self, row: dict, *, data_source: str = None):
        if data_source == "dapo/aime-2024-raw":
            problem, answer = row["prompt"][0]["content"], row["reward_model"]["ground_truth"]
        elif data_source == "dapo/aime-2024":
            problem, answer = row["problem"], row["answer"]
        elif data_source == "dapo/aime-2025":
            problem, answer = row["problem"], row["answer"]

        prompt = problem + answer_format
        data = {
            "data_source": data_source.split("/")[1].lower(),  # aime_2024_raw, aime_2024, aime_2025
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "MATH",
            "reward_model": {"ground_truth": str(answer)},
            "agent_name": self.AGENT_NAME,
        }
        return data

    def map_fn2(self, row: dict):
        content = row["prompt"][0]["content"]
        row["prompt"][0]["content"] = content + answer_format
        row["agent_name"] = self.AGENT_NAME
        return row


def compute_score(data_source, solution_str, ground_truth, extra_info):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
