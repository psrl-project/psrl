"""
Exercise native rollout and reward continuations against a CPU gRPC engine.
"""

import asyncio
import base64
import contextlib
import io
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from omegaconf import OmegaConf
from psrl.workers.gen.rollout_coordination.command_loop import CommandHandlerMixin
from psrl.workers.gen.smg_adapter import build_worker_registration_payload
from psrl_state_grpc_proto import psrl_manager_pb2 as state_pb
from psrl_state_grpc_proto import psrl_manager_pb2_grpc as state_rpc
from smg_grpc_proto import vllm_engine_pb2 as pb
from smg_grpc_proto import vllm_engine_pb2_grpc as rpc
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import grpc

pytestmark = [pytest.mark.cpu_test, pytest.mark.integration]

_ROOT = Path(__file__).resolve().parents[3]
_BINARY = Path(os.environ.get("PSRL_SMG_BINARY", _ROOT / "third_party/smg/target/debug/smg"))


class Engine(rpc.VllmEngineServicer):
    def __init__(self, tokenizer_path, segments, *, hold=False, dp_size=1, capture_re=False):
        self.tokenizer_path = str(tokenizer_path)
        self.segments = list(segments)
        self.hold = hold
        self.dp_size = dp_size
        self.capture_re = capture_re
        self.requests = []
        self.started = asyncio.Event()
        self.aborted = asyncio.Event()

    async def GetModelInfo(self, request, context):
        return pb.GetModelInfoResponse(
            model_path=self.tokenizer_path,
            tokenizer_path=self.tokenizer_path,
            served_model_name="test-model",
            is_generation=True,
            max_context_length=256,
            vocab_size=32,
        )

    async def GetServerInfo(self, request, context):
        return pb.GetServerInfoResponse(server_type="vllm", data_parallel_size=self.dp_size)

    async def HealthCheck(self, request, context):
        return pb.HealthCheckResponse(healthy=True)

    async def GetLoads(self, request, context):
        return pb.GetLoadsResponse(
            dp_rank_count=self.dp_size,
            loads=[pb.SchedulerLoad(dp_rank=i, max_running_requests=16) for i in range(self.dp_size)],
        )

    async def Abort(self, request, context):
        self.aborted.set()
        return pb.AbortResponse()

    async def Generate(self, request, context):
        index = len(self.requests)
        self.requests.append(request)
        self.started.set()
        if self.hold and index == 0:
            await self.aborted.wait()
        tokens, reason = self.segments[min(index, len(self.segments) - 1)]
        complete = pb.GenerateComplete(
            output_ids=tokens,
            finish_reason=reason,
            prompt_tokens=len(request.tokenized.input_ids),
            completion_tokens=len(tokens),
            cached_tokens=len(request.tokenized.input_ids),
            output_logprobs=pb.OutputLogProbs(token_ids=tokens, token_logprobs=[-0.25] * len(tokens)),
        )
        if self.capture_re and tokens:
            start = request.sampling_params.routed_experts_prompt_start
            positions = len(request.tokenized.input_ids) - start + len(tokens) - 1
            complete.routed_experts.CopyFrom(
                pb.RoutedExpertsTensor(data=bytes([index + 1]) * positions, num_layers=1, top_k=1, dtype="uint8")
            )
        yield pb.GenerateResponse(complete=complete)

    async def SubscribePreemptionEvents(self, request, context):
        await asyncio.Future()
        yield pb.PreemptionEvent()

    async def SubscribeKvEvents(self, request, context):
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "This CPU engine has no KV cache events.")
        yield


class ParameterServer(state_rpc.PSManagerStateServicer):
    def __init__(self):
        self.reservations = []
        self.updates = []

    async def CheckAbortedRequests(self, request, context):
        return state_pb.CheckAbortedRequestsResp(is_aborted=[False] * len(request.request_ids))

    async def UpdateRequestInstanceId(self, request, context):
        self.updates.append(request)
        return state_pb.UpdateRequestInstanceIdResp(success=True)

    async def ReserveRolloutInstanceRequests(self, request, context):
        self.reservations.append(request)
        return state_pb.ReserveRolloutInstanceRequestsResp(success=True, buffer_ids=[0], entry_ids=[0])


async def _until(probe, description):
    async with asyncio.timeout(30):
        while True:
            result = await probe()
            if result:
                return result
            await asyncio.sleep(0.02)
    raise AssertionError(f"Timed out waiting for {description}.")


@pytest.fixture
async def gateway(tmp_path, request):
    if not _BINARY.is_file():
        pytest.skip("Build the native SMG binary or set PSRL_SMG_BINARY to run this integration test.")
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **{f"t{i}": i for i in range(1, 32)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer"
    PreTrainedTokenizerFast(tokenizer_object=tokenizer).save_pretrained(tokenizer_path)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    servers = []
    selector = getattr(request, "param", "naive")
    parameter_server = None
    command = [str(_BINARY)]
    if selector == "psrl":
        parameter_server = ParameterServer()
        server = grpc.aio.server()
        state_rpc.add_PSManagerStateServicer_to_server(parameter_server, server)
        state_port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        servers.append(server)
        command = [
            sys.executable,
            "-m",
            "smg.launch_router",
            "--worker-selection-strategy",
            "psrl",
            "--psrl-ps-manager-addr",
            f"http://127.0.0.1:{state_port}",
        ]
    log = (tmp_path / "gateway.log").open("w")
    process = await asyncio.create_subprocess_exec(
        *command,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--connection-mode",
        "grpc",
        "--policy",
        "round_robin",
        "--dp-aware",
        "--enable-routing-loop",
        "--routing-loop-request-sort-key",
        "small_id",
        "--routing-loop-dispatch-batch-size",
        "1",
        "--routing-loop-max-running-dispatch-tasks",
        "1",
        "--disable-tokenizer-autoload",
        "--disable-health-check",
        stdout=log,
        stderr=log,
    )
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=30)

    async def ready():
        if process.returncode is not None:
            pytest.fail((tmp_path / "gateway.log").read_text())
        try:
            return (await client.get("/health")).status_code == 200
        except httpx.ConnectError:
            return False

    async def register(engine, worker_id="source"):
        server = grpc.aio.server()
        rpc.add_VllmEngineServicer_to_server(engine, server)
        worker_port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        servers.append(server)
        payload = build_worker_registration_payload(
            url=f"grpc://127.0.0.1:{worker_port}",
            model_id="test-model",
            role="rollout" if selector == "psrl" else "reward_model_test-model",
            fuse_rollout_with_validate=False,
            max_model_len=256,
            dp_size=engine.dp_size,
            tp_size=1,
            pp_size=1,
            worker_id=worker_id,
        )
        response = await client.post("/workers", json=payload)
        response.raise_for_status()

        async def registered():
            response = await client.get("/workers")
            return f"127.0.0.1:{worker_port}" in response.text

        await _until(registered, "worker registration")

    try:
        await _until(ready, "gateway startup")
        response = await client.post("/v1/tokenizers", json={"name": "test-model", "source": str(tokenizer_path)})
        response.raise_for_status()
        tokenizer_id = response.json()["id"]

        async def tokenizer_ready():
            result = (await client.get(f"/v1/tokenizers/{tokenizer_id}/status")).json()
            assert result["status"] != "failed", f"Tokenizer registration failed: {result!r}."
            return result["status"] == "completed"

        await _until(tokenizer_ready, "tokenizer registration")
        yield SimpleNamespace(
            client=client,
            register=register,
            tokenizer_path=tokenizer_path,
            parameter_server=parameter_server,
        )
    finally:
        await client.aclose()
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(process.wait(), 5)
            if process.returncode is None:
                process.kill()
                await process.wait()
        await asyncio.gather(*(server.stop(0) for server in servers))
        log.close()


def _request(endpoint, budget=8, *, routed_experts=False):
    if endpoint == "/generate":
        return {
            "model": "test-model",
            "input_ids": [1, 2, 3],
            "return_logprob": True,
            "return_routed_experts": routed_experts,
            "sampling_params": {"max_new_tokens": budget, "min_new_tokens": min(4, budget)},
        }
    return {"model": "test-model", "prompt": [1, 2, 3], "max_tokens": budget, "logprobs": 1}


def _headers(uid=7):
    return {"x-request-id": str(uid), "x-prompt-id": str(uid), "x-version-tag": "0"}


@pytest.mark.parametrize("endpoint", ["/generate", "/v1/completions"])
@pytest.mark.parametrize(
    "segments,budget,expected",
    [
        ([([4, 5], "abort"), ([6], "abort"), ([7], "stop")], 8, [4, 5, 6, 7]),
        ([([], "abort"), ([4], "stop")], 8, [4]),
        ([([4, 5], "abort")], 2, [4, 5]),
    ],
)
async def test_partial_rollout_preserves_prefix_and_budget(gateway, endpoint, segments, budget, expected):
    engine = Engine(gateway.tokenizer_path, segments)
    await gateway.register(engine)
    response = await gateway.client.post(endpoint, json=_request(endpoint, budget), headers=_headers())
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list):
        result = result[0]
    assert len(engine.requests) == len(segments), "Each interrupted segment must continue exactly once."
    prefix = []
    for request, (tokens, _) in zip(engine.requests, segments, strict=True):
        assert list(request.tokenized.input_ids) == [1, 2, 3] + prefix, "Continuation lost its generated prefix."
        assert request.sampling_params.max_tokens == budget - len(prefix), "Continuation reset its output budget."
        if endpoint == "/generate":
            assert request.sampling_params.min_tokens == max(0, min(4, budget) - len(prefix)), (
                "Minimum budget was reset."
            )
        prefix.extend(tokens)
    if endpoint == "/generate":
        assert result["output_ids"] == expected, "Rollout token sequence must include every segment."
        usage = result["meta_info"]
        assert len(usage["output_token_logprobs"]) == len(expected), "Rollout logprobs lost a segment."
    else:
        assert result["choices"][0]["text"] == " ".join(f"t{i}" for i in expected), "RM text lost a segment."
        usage = result["usage"]
    assert usage["prompt_tokens"] == 3, "Usage must retain the original prompt length."
    assert usage["completion_tokens"] == len(expected), "Usage must count the full completion."
    assert response.headers["x-smg-partial-rollout-count"] == str(len(segments) - 1), "Wrong interruption count."


async def test_rollout_continuation_preserves_routed_expert_alignment(gateway):
    engine = Engine(gateway.tokenizer_path, [([4, 5], "abort"), ([6], "abort"), ([7], "stop")], capture_re=True)
    await gateway.register(engine)
    response = await gateway.client.post(
        "/generate",
        json=_request("/generate", routed_experts=True),
        headers=_headers(),
    )
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list):
        result = result[0]
    assert result["output_ids"] == [4, 5, 6, 7], "Routed expert capture must preserve completion tokens."
    assert [request.sampling_params.routed_experts_prompt_start for request in engine.requests] == [0, 4, 5], (
        "Each continuation must capture only uncovered routed expert positions."
    )
    experts = np.load(io.BytesIO(base64.b64decode(result["meta_info"]["routed_experts"])))
    assert experts.shape == (6, 1, 1), "Routed expert data must align with all tokens."
    assert experts[:, 0, 0].tolist() == [1, 1, 1, 1, 2, 3], "Routed expert data lost a segment."


@pytest.mark.parametrize("gateway", ["psrl"], indirect=True)
async def test_rollout_continuation_uses_production_selector(gateway):
    engine = Engine(gateway.tokenizer_path, [([4, 5], "abort"), ([6], "stop")])
    await gateway.register(engine)
    response = await gateway.client.post("/generate", json=_request("/generate"), headers=_headers())
    response.raise_for_status()
    assert [list(request.tokenized.input_ids) for request in engine.requests] == [[1, 2, 3], [1, 2, 3, 4, 5]], (
        "PSRL selection must retain the completion prefix."
    )
    assert [request.sampling_params.max_tokens for request in engine.requests] == [8, 6], (
        "PSRL selection must retain the original output budget."
    )
    assert len(gateway.parameter_server.reservations) == 1, "A continuation must reuse its initial reservation."
    assert list(gateway.parameter_server.reservations[0].request_ids) == [7], "The initial request identity changed."
    assert len(gateway.parameter_server.updates) == 1, "The continuation must commit its selected instance."
    assert gateway.parameter_server.updates[0].request_id == 7, "The continuation must retain its PS request identity."


async def test_reward_queue_dispatches_small_ids_first(gateway):
    engine = Engine(gateway.tokenizer_path, [([4], "stop")])
    await gateway.register(engine)
    (await gateway.client.post("/routing_loop/pause")).raise_for_status()
    tasks = []
    try:
        for uid, prompt in [(9, [9]), (2, [2] * 5), (5, [5] * 3)]:
            payload = _request("/v1/completions")
            payload["prompt"] = prompt
            tasks.append(
                asyncio.create_task(
                    gateway.client.post(
                        "/v1/completions",
                        json=payload,
                        headers=_headers(uid),
                    )
                )
            )

        async def queued():
            return (await gateway.client.get("/routing_loop/status")).json()["queue_len"] == 3

        await _until(queued, "queued reward requests")
        (await gateway.client.post("/routing_loop/resume")).raise_for_status()
        for response in await asyncio.gather(*tasks):
            response.raise_for_status()
        assert [request.tokenized.input_ids[0] for request in engine.requests] == [2, 5, 9], (
            "Reward dispatch must prefer numeric request IDs over arrival order and prompt length."
        )
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("terminal", [False, True])
async def test_reward_migration_uses_destination_and_settles_completion_race(gateway, terminal):
    source = Engine(gateway.tokenizer_path, [([4, 5], "abort")], hold=True)
    await gateway.register(source)
    response_task = asyncio.create_task(
        gateway.client.post(
            "/v1/completions",
            json=_request("/v1/completions", 2 if terminal else 8),
            headers=_headers(),
        )
    )
    try:
        await asyncio.wait_for(source.started.wait(), 10)
        destination = Engine(gateway.tokenizer_path, [([6], "stop")], dp_size=2)
        await gateway.register(destination, "destination")

        @contextlib.asynccontextmanager
        async def acquire(_):
            yield

        async def post(path, payload):
            result = await gateway.client.post(path, json=payload)
            result.raise_for_status()
            return result.json()

        async def get(path, **kwargs):
            result = await gateway.client.get(path, **kwargs)
            result.raise_for_status()
            return result.json()

        async def abort(_):
            source.aborted.set()
            return [source.requests[0].request_id]

        coordinator = object.__new__(CommandHandlerMixin)
        coordinator.config = OmegaConf.create(
            {"psrl": {"deployment": {"elastic": {"request_migration_timeout_s": 5}}}}
        )
        coordinator._acquire_replica_operations = AsyncMock(return_value=acquire(None))
        coordinator.server_handles = {
            "source": SimpleNamespace(
                get_external_request_id=SimpleNamespace(remote=AsyncMock(return_value=source.requests[0].request_id)),
                abort_requests=SimpleNamespace(remote=abort),
            )
        }
        coordinator._gateway_post_json = post
        coordinator._gateway_get_json = get
        await coordinator.execute_request_migrations(
            [
                {
                    "request_id": "scheduler-id",
                    "source_instance_id": ("source", 0),
                    "destination_instance_id": ("destination", 1),
                }
            ]
        )
        response = await response_task
        response.raise_for_status()
        assert len(source.requests) == 1, "Migration must not return to the source."
        assert len(destination.requests) == (0 if terminal else 1), "Migration dispatched an invalid budget."
        if not terminal:
            request = destination.requests[0]
            assert list(request.tokenized.input_ids) == [1, 2, 3, 4, 5], "Migration lost its prefix."
            assert request.sampling_params.max_tokens == 6, "Migration reset its budget."
            assert request.data_parallel_rank == 1, "Migration ignored the destination DP rank."
        status = await get("/routing_loop/migration", params={"request_id": source.requests[0].request_id})
        assert status is None, "The coordinator must release its migration reservation."
    finally:
        response_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await response_task
