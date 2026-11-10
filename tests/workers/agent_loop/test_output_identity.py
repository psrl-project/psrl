"""
Verify committed rollout identities remain indexable during advantage computation.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf
from psrl.trainer.ppo import ray_trainer
from psrl.workers.agent_loop import worker as worker_module
from psrl.workers.gen.utils import TokenOutput
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_id", [None, 3])
@pytest.mark.parametrize("trajectory_count", [1, 2])
async def test_committed_output_reaches_advantage_with_prompt_identity(monkeypatch, parent_id, trajectory_count):
    cls = worker_module.PSRL_AgentLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.overlong_filtering = False
    worker._compute_multi_modal_inputs = lambda output, input_ids: {}
    worker._compute_position_ids = lambda input_ids, attention_mask, multimodal: torch.arange(
        input_ids.shape[-1]
    ).unsqueeze(0)
    worker.agent_loop_manager = SimpleNamespace(put_result=SimpleNamespace(remote=AsyncMock()))
    commit = AsyncMock()
    monkeypatch.setattr(worker_module.tq, "async_kv_batch_put", commit)
    request = {"uid": [7], "validate": [False], "version_tag": [0]}
    if parent_id is not None:
        request["parent_id"] = [parent_id]
    outputs = [
        TokenOutput(prompt_ids=[1, 2], response_ids=[3] * (i + 1), response_mask=[1] * (i + 1))
        for i in range(trajectory_count)
    ]

    await worker.postprocess_output(outputs, tu.get_tensordict(request))

    committed = commit.call_args.kwargs
    payload = committed["fields"]
    expected_parent = 7 if parent_id is None else parent_id
    assert tu.get(payload, "parent_id") == [expected_parent] * trajectory_count, (
        "Every committed trajectory must retain its prompt identity."
    )
    masks = [torch.ones(i + 1, dtype=torch.int64) for i in range(trajectory_count)]
    payload["response_mask"] = torch.nested.as_nested_tensor(masks, layout=torch.jagged)
    payload["rm_scores"] = torch.nested.as_nested_tensor([mask.float() for mask in masks], layout=torch.jagged)
    monkeypatch.setattr(ray_trainer.tq, "kv_batch_get", lambda **kwargs: payload)
    put = Mock()
    monkeypatch.setattr(ray_trainer.tq, "kv_batch_put", put)
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "gamma": 1.0, "lam": 1.0, "use_kl_in_reward": False},
            "gen_actor_rollout_ref": {"rollout": {"n": 1 if parent_id is None else 2}},
        }
    )
    batch = KVBatchMeta(
        keys=committed["keys"],
        tags=[{"uid": 7, "parent_id": expected_parent}] * trajectory_count,
        partition_id="train",
    )

    trainer._compute_advantage(batch, {})

    written = put.call_args.kwargs["fields"]
    for field in ("advantages", "returns"):
        assert [row.numel() for row in written[field].unbind()] == list(range(1, trajectory_count + 1)), (
            "Advantage output must preserve each trajectory's response length."
        )
        assert torch.isfinite(written[field].values()).all(), "Advantage output must remain finite."
