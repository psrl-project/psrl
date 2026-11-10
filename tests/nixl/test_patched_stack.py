"""
Verify real host and GPU memory transfers through the patched NIXL and UCX stack.
"""

import ctypes
import time
from uuid import uuid4

import pytest

nixl_api = pytest.importorskip("nixl._api")

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("num_workers", [1, 2, 4])
@pytest.mark.parametrize(
    "memory_type",
    [pytest.param("DRAM", marks=pytest.mark.cpu_test), pytest.param("VRAM", marks=pytest.mark.gpu)],
)
def test_ucx_transfer_with_shared_workers(num_workers, memory_type):
    payload = bytes(range(256)) * 1024
    if memory_type == "DRAM":
        source_buffer = ctypes.create_string_buffer(payload, len(payload))
        target_buffer = ctypes.create_string_buffer(len(payload))
        source_address = ctypes.addressof(source_buffer)
        target_address = ctypes.addressof(target_buffer)
        device_id = 0
    else:
        import torch

        source_buffer = torch.frombuffer(bytearray(payload), dtype=torch.uint8).cuda()
        target_buffer = torch.zeros_like(source_buffer)
        torch.cuda.synchronize()
        source_address = source_buffer.data_ptr()
        target_address = target_buffer.data_ptr()
        device_id = torch.cuda.current_device()
    config = nixl_api.nixl_agent_config(
        enable_prog_thread=True,
        enable_listen_thread=False,
        backends=["UCX"],
        num_workers=num_workers,
    )
    source = nixl_api.nixl_agent(f"source-{uuid4().hex}", config)
    target = nixl_api.nixl_agent(f"target-{uuid4().hex}", config)
    source_registration = source.register_memory([(source_address, len(payload), device_id, "")], memory_type)
    target_registration = target.register_memory([(target_address, len(payload), device_id, "")], memory_type)
    handle = None
    try:
        source.add_remote_agent(target.get_agent_metadata())
        target.add_remote_agent(source.get_agent_metadata())
        local_descs = source.get_xfer_descs([(source_address, len(payload), device_id)], memory_type)
        remote_descs = source.get_xfer_descs([(target_address, len(payload), device_id)], memory_type)
        handle = source.initialize_xfer("WRITE", local_descs, remote_descs, target.name)
        state = source.transfer(handle)
        deadline = time.monotonic() + 10
        while state == "PROC" and time.monotonic() < deadline:
            time.sleep(0.001)
            state = source.check_xfer_state(handle)
        assert state == "DONE", f"UCX transfer did not complete: {state!r}."
        assert source.query_xfer_backend(handle) == "UCX", "Transfer did not use UCX."
        received = target_buffer.raw if memory_type == "DRAM" else target_buffer.cpu().numpy().tobytes()
        assert received == payload, "Transferred payload differs from the source."
    finally:
        if handle is not None:
            source.release_xfer_handle(handle)
        source.remove_remote_agent(target.name)
        target.remove_remote_agent(source.name)
        source.deregister_memory(source_registration)
        target.deregister_memory(target_registration)
