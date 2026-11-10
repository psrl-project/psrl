import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from psrl.utils.transfer_queue import payload
from psrl.utils.transfer_queue.payload import PayloadState

pytestmark = pytest.mark.cpu_test


def test_clear_payload_filters_missing_keys(monkeypatch):
    kv_list = MagicMock(return_value={"train": {"34_0": {}, "34_1": {}}})
    kv_clear = MagicMock()
    monkeypatch.setattr(payload.tq, "kv_list", kv_list)
    monkeypatch.setattr(payload.tq, "kv_clear", kv_clear)

    payload.clear_payload(
        keys=["34", "34_0", "34_1"],
        partition_id="train",
        state=PayloadState.DROPPED,
    )

    kv_clear.assert_called_once_with(keys=["34_0", "34_1"], partition_id="train")


def test_async_clear_payload_skips_absent_keys(monkeypatch):
    async_kv_list = AsyncMock(return_value={"train": {"other": {}}})
    async_kv_clear = AsyncMock()
    monkeypatch.setattr(payload.tq, "async_kv_list", async_kv_list)
    monkeypatch.setattr(payload.tq, "async_kv_clear", async_kv_clear)

    asyncio.run(
        payload.async_clear_payload(
            keys=["34"],
            partition_id="train",
            state=PayloadState.DROPPED,
        )
    )

    async_kv_clear.assert_not_awaited()
