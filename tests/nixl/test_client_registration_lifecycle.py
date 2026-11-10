import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import nixl._bindings as nixl_bindings
import pytest
import torch
from omegaconf import OmegaConf
from psrl.utils.nixl import client as client_module
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.nixl_spec import NIXLClientType, NIXLSharding

pytestmark = pytest.mark.cpu_test


@pytest.fixture(params=[False, True], ids=["release_temp_buffers", "retain_tms_buffers"])
def registration_client(request, monkeypatch):
    agent = Mock()
    agent.nixl_mems = {"cpu": nixl_bindings.nixl_mem_t.DRAM_SEG}
    agent.register_memory.side_effect = lambda desc: desc
    agent.get_serialized_descs.side_effect = lambda desc: (
        f"descriptor-{agent.get_serialized_descs.call_count}".encode()
    )
    agent.get_xfer_descs.side_effect = lambda entries, mem_type: nixl_bindings.nixlXferDList(
        agent.nixl_mems[mem_type], entries
    )
    agent.get_agent_metadata.return_value = b"agent-metadata"
    monkeypatch.setattr(client_module, "get_local_ip", lambda: "127.0.0.1")
    monkeypatch.setattr(client_module, "get_local_gpu_id", lambda: -1)
    monkeypatch.setattr(client_module, "torch_memory_saver", SimpleNamespace(region=lambda **kwargs: nullcontext()))
    config = OmegaConf.create(
        {
            "server_ip": "127.0.0.1",
            "server_port": 20000,
            "max_pinned_temp_memory_slots": None,
            "enable_tms_for_temp_buffers": False,
        }
    )
    client = NIXLStorageClient(
        client_name="validation-client",
        server_name="metadata-server",
        use_gpu=False,
        client_type=NIXLClientType.PULL_SIDE,
        nixl_config=config,
        binded_agent=agent,
    )
    client.enable_tms_for_temp_buffers = request.param
    yield client
    client.shutdown()


def _register_weights(client, *, meta_only=False):
    state = {
        "contiguous": torch.arange(8, dtype=torch.float32),
        "strided": torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),
    }
    client.register_local_tensors(state, {key: NIXLSharding.default() for key in state}, meta_only=meta_only)
    return state


def test_deregister_before_protocol_is_idempotent(registration_client):
    client = registration_client
    client.deregister_local_tensors()
    client.deregister_local_tensors()

    assert client.local_client_info is None, "Initial sleep must not require or create registration metadata."
    assert client._registered_descs == {}, "Initial sleep must leave no registered memory."
    client.agent.deregister_memory.assert_not_called()


def test_register_after_initial_sleep_and_repeated_deregistration(registration_client):
    client = registration_client
    client.deregister_local_tensors()
    _register_weights(client, meta_only=True)
    assert not client.local_client_info.is_registered, "The handshake must support metadata without pinned memory."
    client.deregister_local_tensors()

    state = _register_weights(client)
    info = client.local_client_info
    temporary_buffers = client._temp_tensor_mapping
    assert info.is_registered, "Wake must register real memory after the metadata handshake."
    assert temporary_buffers, "Strided shards must allocate temporary transfer buffers."
    descriptors = list(client._registered_descs.values())
    assert descriptors, "Registered weights must have descriptors to release."

    client.deregister_local_tensors()
    client.deregister_local_tensors()
    assert client.agent.deregister_memory.call_count == len(descriptors), "Each descriptor must be released once."
    assert client._registered_descs == {}, "Sleeping clients must hold no pinned descriptors."
    if client.enable_tms_for_temp_buffers:
        assert client.local_client_info is info, "TMS wake must retain metadata for the existing virtual addresses."
        assert client._temp_tensor_mapping is temporary_buffers, "TMS must retain its managed transfer buffers."
    else:
        assert client.local_client_info is None, "Released buffers must discard local registration metadata."
        assert client._temp_tensor_mapping == {}, "Unmanaged temporary buffers must be released."
        assert client._original_tensor_mapping == {}, "Released tensor mappings must be rebuilt on wake."

    client.agent.get_agent_metadata.return_value = b"refreshed-agent-metadata"
    client.register_local_tensors(state, {key: NIXLSharding.default() for key in state})
    assert client.local_client_info.is_registered, "Repeated sleep must allow another complete registration."
    assert client.local_client_info.meta == b"refreshed-agent-metadata", (
        "Wake must publish fresh registration metadata."
    )
    assert client._registered_descs, "Wake must pin memory again."
    assert client.agent.register_memory.call_count == 2, "Wake must register physical memory for every cycle."


def test_deregister_cleans_partial_registration_without_client_info(registration_client):
    client = registration_client
    client.agent.get_xfer_descs.side_effect = RuntimeError("Descriptor construction failed.")
    with pytest.raises(RuntimeError, match="Descriptor construction failed"):
        _register_weights(client)
    assert client.local_client_info is None, "Failed descriptor construction must precede metadata publication."
    descriptors = list(client._registered_descs.values())
    assert descriptors, "Failure after memory registration must leave descriptors requiring cleanup."

    client.deregister_local_tensors()

    assert client.agent.deregister_memory.call_count == len(descriptors), "Partial registrations must be released."
    assert client._registered_descs == {}, "Missing client metadata must not bypass resource cleanup."


def test_deregister_propagates_memory_release_failure(registration_client):
    client = registration_client
    _register_weights(client)
    client.agent.deregister_memory.side_effect = RuntimeError("Memory deregistration failed.")

    with pytest.raises(RuntimeError, match="Memory deregistration failed"):
        client.deregister_local_tensors()


@pytest.mark.parametrize("registration", ["initial", "meta", "full"])
@pytest.mark.asyncio
async def test_server_sleep_releases_nixl_before_engine_memory(registration_client, registration):
    from psrl.workers.gen.vllm_async_server import PSRL_vLLMHttpServer
    from psrl.workers.gen.vllm_extension import vLLMWorkerExtension

    client = registration_client
    if registration != "initial":
        _register_weights(client, meta_only=registration == "meta")
    worker = object.__new__(vLLMWorkerExtension)
    worker.nixl_storage_client = client
    worker.get_instance_local_rank = lambda: 0
    worker.get_instance_local_tp_rank = lambda: 0
    server = object.__new__(PSRL_vLLMHttpServer)
    server._is_init_nixl_client = asyncio.Event()
    server._is_init_nixl_client.set()
    server._is_reward_model = lambda: False
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(range="none"))
    server.curr_rollout_instance_model_version = [0]
    server.get_instance_num = lambda: 1

    async def collective_rpc(*, method):
        return getattr(worker, method)()

    async def sleep(level, *, mode):
        assert level == 2 and mode == "abort", "Validation sleep must release all engine memory."
        assert client._registered_descs == {}, "NIXL must release pinned pages before engine memory is unmapped."

    server.collective_rpc = AsyncMock(side_effect=collective_rpc)
    server.engine = SimpleNamespace(sleep=AsyncMock(side_effect=sleep))

    await server.nixl_sleep(level=2)

    server.engine.sleep.assert_awaited_once_with(2, mode="abort")
    assert server.curr_rollout_instance_model_version == [-1], "Released weights must invalidate the model version."
