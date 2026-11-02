"""
Verify rollout context limits are resolved by the native vLLM startup path.
"""

import argparse
import ast
import asyncio
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def server_startup():
    """
    Execute the production startup methods with GPU and RPC operations replaced by CPU stubs.
    """
    path = Path(__file__).resolve().parents[3] / "psrl/workers/gen/vllm_async_server.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PSRL_vLLMHttpServer")
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("_validate_configs", "run_server")
    ]
    native_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=40960),
        parallel_config=SimpleNamespace(),
        scheduler_config=SimpleNamespace(),
    )
    engine_args = SimpleNamespace(create_engine_config=Mock(return_value=native_config))
    client = SimpleNamespace(reset_mm_cache=AsyncMock(), collective_rpc=AsyncMock())
    factory = Mock(return_value=client)
    engine_args_cls = SimpleNamespace(from_cli_args=Mock(return_value=engine_args))
    namespace = {
        "argparse": argparse,
        "asyncio": asyncio,
        "inspect": inspect,
        "AsyncEngineArgs": engine_args_cls,
        "UsageContext": SimpleNamespace(OPENAI_API_SERVER="api_server"),
        "AsyncLLM": SimpleNamespace(from_vllm_config=factory),
        "PreemptionStatLogger": Mock(),
        "psrl_logger": Mock(),
    }
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    server = SimpleNamespace(
        config=OmegaConf.create({"max_model_len": 40960, "disable_log_stats": True, "data_parallel_size": 1}),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(max_position_embeddings=32768), tokenizer=range(8)),
        psrl_config=OmegaConf.create(
            {
                "rollout_coordination": {"routing_strategy": {"max_num_waiting_reqs_after_preemption": 3}},
                "profile": {"prefill_composition": {"enable": False}},
                "logging_path": "logs",
            }
        ),
        _dp_master_port=1234,
        get_replica_idx=lambda: 1,
        replica_rank=1,
        node_rank=0,
        kv_cache_manager=Mock(),
        _is_reward_model=lambda: False,
        is_pooling_model=False,
        _start_grpc_server=AsyncMock(return_value=9876),
    )
    return SimpleNamespace(
        server=server,
        validate=namespace["_validate_configs"],
        run=namespace["run_server"],
        engine_args_cls=engine_args_cls,
        engine_args=engine_args,
        native_config=native_config,
        factory=factory,
    )


@pytest.mark.parametrize("requested,resolved", [(None, 32768), (40960, 40960), (-1, 8192)])
async def test_native_context_limit_is_used_for_generation(server_startup, requested, resolved):
    startup = server_startup
    startup.server.config.max_model_len = requested
    arguments = argparse.Namespace(
        hf_overrides={
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": 1.25,
                "original_max_position_embeddings": 32768,
            }
        }
    )
    original = copy.deepcopy(arguments.hf_overrides)

    def create_engine(**kwargs):
        startup.native_config.model_config.max_model_len = resolved
        return startup.factory.return_value

    startup.factory.side_effect = create_engine
    startup.validate(startup.server)
    assert startup.server.config.max_model_len == requested, "veRL must not impose an unscaled context ceiling."
    await startup.run(startup.server, arguments)
    startup.engine_args_cls.from_cli_args.assert_called_once_with(arguments)
    startup.engine_args.create_engine_config.assert_called_once_with(usage_context="api_server")
    assert startup.server.config.max_model_len == resolved, "Generation must use vLLM's resolved engine limit."
    assert arguments.hf_overrides == original, "PSRL must forward HF overrides without rewriting RoPE parameters."
    assert startup.server.model_config.hf_config.max_position_embeddings == 32768, (
        "The original HF config must remain intact."
    )
    assert startup.server.engine is startup.factory.return_value, "The validated configuration must start the engine."


async def test_native_validation_error_prevents_engine_startup(server_startup):
    startup = server_startup
    startup.engine_args.create_engine_config.side_effect = ValueError("Context exceeds model capacity.")
    with pytest.raises(ValueError, match="Context exceeds model capacity"):
        await startup.run(startup.server, argparse.Namespace())
    startup.factory.assert_not_called()
    assert startup.server.config.max_model_len == 40960, "Failed validation must not substitute a context limit."
