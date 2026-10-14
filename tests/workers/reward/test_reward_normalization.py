"""Verify reward normalization for padded terminal batches and partial trajectories."""

import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from psrl.workers.reward import reward_manager as manager_module
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("normalization", [None, "batch", "group"])
def test_padding_does_not_consume_request_metadata_or_change_normalization(monkeypatch, normalization):
    keys = ["11_0", "padding_0", "11_1", "12_0", "12_1"]
    batch = KVBatchMeta(
        keys=keys,
        tags=[{}, {"is_padding": True}, {}, {}, {}],
        partition_id="train",
    )
    data = tu.get_tensordict(
        tensor_dict={
            "uid": np.array([11, -7, 11, 12, 12]),
            "reward_score": np.array([1.0, 0.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "reward_extra_info": np.array([{} for _ in keys], dtype=object),
            "response_mask": torch.ones(len(keys), 3),
        },
    )
    writes = []

    async def get(keys, partition_id, select_fields):
        assert partition_id == "train", "Normalization must retain the batch partition."
        assert "padding_0" not in keys, "Padding must not enter reward normalization."
        indices = [batch.keys.index(key) for key in keys]
        return data[indices].select(*select_fields)

    async def put(keys, partition_id, fields):
        writes.append((keys, fields))

    monkeypatch.setattr(manager_module.tq, "async_kv_batch_get", get)
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_put", put)
    manager = object.__new__(manager_module.RewardLoopManager)
    manager.reward_normalization = normalization
    manager.request_id_to_data_source = {11: "math", 12: "math", 99: "unused"}
    manager.request_id_to_group = {11: "same-group", 12: "same-group"}

    assert asyncio.run(manager.normalize_reward(batch)) is batch, "Normalization must retain padding metadata."
    assert manager.request_id_to_data_source == {99: "unused"}, "Only real UIDs in this batch may be consumed."
    written_keys, written = writes[0]
    assert written_keys == ["11_0", "11_1", "12_0", "12_1"], "Writes must exclude padding and retain row order."
    extras = tu.get(written, "reward_extra_info")
    assert [extra["data_source"] for extra in extras] == ["math"] * 4, "Real rows need their original data source."
    assert [extra["original_reward_score"] for extra in extras] == [1, 2, 3, 4], "Original rewards must be retained."
    if normalization is not None:
        np.testing.assert_allclose(tu.get(written, "reward_score"), [-1, -1, 1, 1])
        torch.testing.assert_close(written["rm_scores"][:, -1], torch.tensor([-1.0, -1.0, 1.0, 1.0]))


def test_padding_only_batch_does_not_access_reward_storage(monkeypatch):
    get = AsyncMock()
    put = AsyncMock()
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_get", get)
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_put", put)
    manager = object.__new__(manager_module.RewardLoopManager)
    batch = KVBatchMeta(keys=["padding_0"], tags=[{"is_padding": True}], partition_id="train")

    assert asyncio.run(manager.normalize_reward(batch)) is batch, "An empty real batch must retain its metadata."
    get.assert_not_awaited()
    put.assert_not_awaited()


def test_missing_real_uid_fails_without_consuming_other_request_metadata(monkeypatch):
    data = tu.get_tensordict(
        tensor_dict={
            "uid": np.array([11, -7]),
            "reward_score": np.array([1.0, 2.0], dtype=np.float32),
            "reward_extra_info": np.array([{}, {}], dtype=object),
        },
    )
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_get", AsyncMock(return_value=data))
    put = AsyncMock()
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_put", put)
    manager = object.__new__(manager_module.RewardLoopManager)
    manager.request_id_to_data_source = {11: "math"}
    batch = KVBatchMeta(keys=["11", "missing"], tags=[{}, {}], partition_id="train")

    with pytest.raises(KeyError, match="-7"):
        asyncio.run(manager.normalize_reward(batch))
    assert manager.request_id_to_data_source == {11: "math"}, "Failed metadata validation must be atomic."
    put.assert_not_awaited()
