"""
Verify simple and benchmark datasets through mini-SWE environment setup and reward routing.
"""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from examples.mini_swe.config import build_runtime_config
from examples.mini_swe.prepare.prepare_simple_data import generate_simple_test_data
from examples.mini_swe.reward import compute_score
from examples.mini_swe.runner import _build_environment_config, _grade_patch
from omegaconf import OmegaConf
from psrl.environments.mini_swe_env import MiniSWEEnvironment

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def environment():
    path = Path(__file__).resolve().parents[2] / "examples/mini_swe/config/simple_agent_config.yaml"
    runtime_config = build_runtime_config(OmegaConf.load(path)[0])
    return MiniSWEEnvironment(OmegaConf.create({}), None, None, runtime_config=runtime_config)


@pytest.mark.asyncio
@pytest.mark.parametrize("split,repo_name", [("train", "train_0"), ("test", "val_0")])
@pytest.mark.parametrize("serialize_extra_info", [False, True])
async def test_simple_parquet_reaches_patch_reward_without_benchmark_grader(
    environment, tmp_path, split, repo_name, serialize_extra_info
):
    path = tmp_path / f"{split}.parquet"
    generate_simple_test_data(1, split).to_parquet(path)
    task = pd.read_parquet(path).iloc[0].to_dict()
    expected_patch = task["reward_model"]["ground_truth"]
    if serialize_extra_info:
        task["extra_info"] = json.dumps(task["extra_info"])

    observation, _ = await environment.reset(task)
    payload = {"runtime_config": asdict(observation["runtime_config"]), "observation": observation}
    sandbox = _build_environment_config(payload)

    assert observation["problem_statement"] == task["prompt"][0]["content"], "The task text must reach the runner."
    assert sandbox["cwd"] == f"/{repo_name}", "Simple episodes must use the prepared repository."
    assert sandbox["image"] == "python:3.11-slim", "Simple episodes must retain the configured image."
    assert _grade_patch(payload, expected_patch) is None, "Simple patches must be scored by the reward function."
    reward = compute_score(
        task["data_source"],
        solution_str="",
        ground_truth=expected_patch,
        extra_info={"patch": expected_patch, "num_turns": 1},
    )
    assert reward == 1.0, "The expected simple patch must receive full reward."


def _benchmark_task(data_source):
    return {
        "data_source": data_source,
        "extra_info": {
            "problem_statement": "Fix the failing test.",
            "swe_grader": "swebench_fresh_container",
            "swe_problem": {"instance_id": "example__repo-1", "FAIL_TO_PASS": ["test_fix"]},
            "swe_problem_image": "example/swe:latest",
            "swe_restore_tests": data_source == "swe_smith_py",
            "sandbox_overrides": {"environment": {"image": "example/swe:latest", "cwd": "/testbed"}},
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("data_source", ["swebench_verified", "swe_smith_py", "swe_gym"])
async def test_benchmark_keeps_fresh_container_grading(environment, monkeypatch, data_source):
    from examples.mini_swe import swebench_grader

    task = _benchmark_task(data_source)
    if data_source == "swe_gym":
        task["extra_info"]["swe_problem"]["eval_script"] = "pytest"
    grader_result = {"resolved": True}
    grade = Mock(return_value=grader_result)
    monkeypatch.setattr(swebench_grader, "grade_fresh_container", grade)

    observation, _ = await environment.reset(task)
    payload = {"runtime_config": asdict(observation["runtime_config"]), "observation": observation}
    sandbox = _build_environment_config(payload)

    assert sandbox["image"] == task["extra_info"]["swe_problem_image"], "Benchmark image overrides must survive."
    assert sandbox["cwd"] == "/testbed", "Benchmark episodes must use the image's testbed."
    assert _grade_patch(payload, "a patch") is grader_result, "Benchmark patches must reach the harness."
    expected_kind = {"swebench_verified": "verified", "swe_smith_py": "smith", "swe_gym": "gym"}[data_source]
    assert grade.call_args.args[:4] == (
        task["extra_info"]["swe_problem"],
        "a patch",
        expected_kind,
        task["extra_info"]["swe_problem_image"],
    ), "The runner must preserve the benchmark grading inputs."


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["swe_grader", "swe_problem", "swe_problem_image"])
async def test_incomplete_benchmark_is_rejected(environment, field):
    task = _benchmark_task("swebench_verified")
    del task["extra_info"][field]

    with pytest.raises(ValueError, match=field):
        await environment.reset(task)


@pytest.mark.asyncio
async def test_unknown_source_cannot_silently_skip_benchmark_grading(environment):
    with pytest.raises(ValueError, match="Unsupported mini-SWE data source"):
        await environment.reset({"data_source": "unknown", "extra_info": {}})
