"""Exercise teacher HTTP scoring, reward routing, queue publication and student updates."""

import asyncio
import copy
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from aiohttp import web
from omegaconf import OmegaConf
from psrl.trainer.distillation.data import align_teacher_targets
from psrl.trainer.distillation.losses import distillation_ppo_loss
from psrl.utils.dataset.utils import as_object_sample, normalize_reward_model_dicts
from psrl.workers.config.distillation import DistillationConfig, DistillationLossConfig
from psrl.workers.reward import reward_manager as manager_module
from psrl.workers.reward.reward_loop.opd import OPDRewardLoop, parse_teacher_response
from psrl.workers.reward.reward_manager import RewardLoopManager
from psrl.workers.reward.reward_protocol import RewardModelRuntimeInfo
from psrl.workers.reward.reward_worker import RewardLoopWorker, RewardSpec
from psrl.workers.reward.utils import attach_reward_trajectories
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

pytestmark = pytest.mark.cpu_test


class Tokenizer:
    def get_vocab(self):
        return {str(token): token for token in range(8)}

    def __len__(self):
        return 8


def teacher_response(ids, bias=0.0):
    probabilities = torch.log_softmax(torch.arange(8, dtype=torch.float32) * (0.1 + bias), -1)
    return {
        "meta_info": {
            "prompt_token_ids": ids,
            "input_token_logprobs": [[None, ids[0]]] + [[float(probabilities[t]), t] for t in ids[1:]],
            "input_top_logprobs": [[]] + [[[float(p), t] for t, p in enumerate(probabilities)] for _ in ids[1:]],
        }
    }


@asynccontextmanager
async def teacher_service():
    requests = []

    async def generate(request):
        payload = await request.json()
        requests.append(payload)
        return web.json_response(teacher_response(payload["input_ids"], 0.05 if payload["model"] == "b" else 0.0))

    app = web.Application()
    app.router.add_post("/generate", generate)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}", requests
    finally:
        await runner.cleanup()


def config(tmp_path, topk=False, launch_async=True):
    pool = {"result_queue_size": 8, "result_drain_batch_size": 8}
    return OmegaConf.create(
        {
            "psrl": {"logging_path": str(tmp_path), "rollout_coordination": {"redundant_rollout": {"enable": False}}},
            "reward": {
                "launch_reward_fn_async": launch_async,
                "reward_normalization": None,
                "reward_loop_worker": pool,
                "validation_reward_loop_worker": pool,
            },
            "gen_actor_rollout_ref": {"rollout": {"n": 1}},
            "train_actor_rollout_ref": {"rollout": {"val_kwargs": {"n": 1}}},
            "distillation": {
                "enabled": True,
                "distillation_loss": {
                    "loss_mode": "forward_kl_topk" if topk else "k1",
                    "topk": 3,
                },
            },
        }
    )


def make_loop(cfg, url, name="a"):
    runtime = RewardModelRuntimeInfo(
        url,
        Tokenizer(),
        OmegaConf.create(
            {
                "reward_model_name": name,
                "rollout": {"runner": "generate", "max_model_len": 32},
            }
        ),
    )
    return OPDRewardLoop(cfg, Tokenizer(), runtime)


def request(uid="sample", teacher="a", trajectories=None, scalar=True):
    if trajectories is None:
        trajectories = [{"prompts": [1, 2], "responses": [3, 4, 5]}]
    specs = [{"reward_loop_type": "opd", "reward_fn": "default", "reward_model_name": teacher, "teacher_weight": 1.7}]
    if scalar:
        specs.append({"reward_loop_type": "naive", "reward_fn": "default", "reward_coef": 0.25})
    last = trajectories[-1]
    inputs = tu.get_tensordict(
        tensor_dict={
            "prompts": torch.tensor([last["prompts"]]),
            "responses": torch.tensor([last["responses"]]),
            "uid": np.array([uid]),
            "n_trajectory": np.array([len(trajectories)]),
            "data_source": np.array(["text"]),
            "reward_model_dicts": as_object_sample(specs),
        },
        non_tensor_dict={"validate": False},
    )
    attach_reward_trajectories(
        inputs,
        [
            SimpleNamespace(
                prompt_ids=row["prompts"],
                response_ids=row["responses"],
                multi_modal_data=None,
            )
            for row in trajectories
        ],
    )
    return inputs


def make_worker(cfg, loops):
    cls = RewardLoopWorker.__ray_metadata__.modified_class
    worker = object.__new__(cls)
    worker.config, worker.is_validate, worker.worker_id = cfg, False, 0
    worker.completed = worker.failed = 0
    scalar = SimpleNamespace(run_single=AsyncMock(return_value={"reward_score": 4.0, "reward_extra_info": {}}))
    worker.reward_spec_to_manager = {RewardSpec("opd", "default", name): loop for name, loop in loops.items()}
    worker.reward_spec_to_manager[RewardSpec("naive", "compute_score", None)] = scalar
    return worker


def student_update(data, topk):
    data = align_teacher_targets(data, 3 if topk else 1, topk)
    student = torch.nn.Embedding(8, 8)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    inputs = torch.cat([torch.cat((p, r)) for p, r in zip(data["prompts"], data["responses"])])
    logits = student(inputs).unsqueeze(0)
    labels = inputs.roll(-1)
    output = {"log_probs": logits.log_softmax(-1).gather(-1, labels[None, :, None]).squeeze(-1).squeeze(0)}
    cfg = SimpleNamespace(strategy="fsdp2", loss_agg_mode="token-mean", global_batch_info={})
    distill = DistillationConfig(
        enabled=True,
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk" if topk else "k1",
            topk=3,
            use_policy_gradient=not topk,
            use_task_rewards=False,
            log_prob_min_clamp=None,
        ),
    )
    if topk:
        output.update(
            {
                key: value.squeeze(0)
                for key, value in distillation_ppo_loss(
                    cfg,
                    distill,
                    data=data,
                    student_logits=logits,
                ).items()
            }
        )
    data["old_log_probs"] = torch.nested.as_nested_tensor(
        [torch.full_like(row, -2.0, dtype=torch.float32) for row in data["responses"]],
        layout=torch.jagged,
    )
    before = student.weight.detach().clone()
    loss, _ = distillation_ppo_loss(cfg, distill, model_output=output, data=data)
    assert torch.isfinite(loss), "Student loss must be finite."
    loss.backward()
    assert torch.isfinite(student.weight.grad).all(), "Student gradients must be finite."
    optimizer.step()
    assert not torch.equal(before, student.weight), "OPD must update student weights."


@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("launch_async", [False, True])
async def test_http_to_reward_queue_and_student(tmp_path, monkeypatch, topk, launch_async, real_queue=False):
    cfg = config(tmp_path, topk, launch_async)
    stored = {}
    queue_put = manager_module.tq.async_kv_batch_put

    async def put(keys, partition_id, fields):
        if real_queue:
            await queue_put(keys=keys, partition_id=partition_id, fields=fields)
            fields = await manager_module.tq.async_kv_batch_get(keys=keys, partition_id=partition_id)
        for i, key in enumerate(keys):
            stored[(partition_id, key)] = {field: fields[field][i] for field in fields.keys()}

    monkeypatch.setattr(manager_module.tq, "async_kv_batch_put", put)
    async with teacher_service() as (url, calls):
        worker = make_worker(cfg, {name: make_loop(cfg, url, name) for name in ("a", "b")})
        rpc = SimpleNamespace(
            start_busy_loop=SimpleNamespace(remote=Mock()), stop_busy_loop=SimpleNamespace(remote=AsyncMock())
        )
        ps = SimpleNamespace(update_request_status=SimpleNamespace(remote=AsyncMock(return_value=[True])))
        manager = RewardLoopManager(cfg, ps, [rpc], is_validate=False)
        worker.reward_manager_client = SimpleNamespace(put_reward_result=manager.put_reward_result)

        async def dispatch(items):
            for item in items:
                await worker._run_reward_loop(item)

        rpc.add_reward_requests = SimpleNamespace(remote=AsyncMock(side_effect=dispatch))
        manager.start_busy_loop()
        trajectories = [{"prompts": [1, 2], "responses": [3, 4, 5]}, {"prompts": [1, 2, 3, 4], "responses": [5, 6]}]
        try:
            await manager.compute_score(request(trajectories=trajectories))
            await manager.compute_score(request(uid="other", teacher="b", scalar=False))
            await manager.wait_for_reward_ready(["sample_0", "sample_1", "other"])
            keys = ["sample_1", "other", "sample_0"]
            rows = []
            for key in keys:
                reward = stored[("train", key)]
                trajectory = trajectories[1 if key == "sample_1" else 0]
                assert reward["teacher_logprobs"].shape[0] == len(trajectory["responses"]), (
                    "Targets must be per trajectory."
                )
                assert float(reward["teacher_weight"]) == pytest.approx(1.7), "Weights must not be normalized."
                assert float(reward["reward_score"]) == (0.0 if key == "other" else 1.0), (
                    "Only scalars use reward_coef."
                )
                rows.append(
                    {
                        **reward,
                        "prompts": torch.tensor(trajectory["prompts"]),
                        "responses": torch.tensor(trajectory["responses"]),
                        "response_mask": torch.tensor([1, 0, 1] if key != "sample_1" else [1, 1]),
                    }
                )
            student_update(list_of_dict_to_tensordict(rows), topk)
            assert len(calls) == 3, "Every trajectory must be scored once."
            assert all(call["top_logprobs_num"] == (3 if topk else 0) for call in calls), "K must match the loss mode."
            assert all(call["return_logprob"] and call["logprob_start_len"] == 0 for call in calls), (
                "All prompt positions must be scored."
            )
            assert calls[2]["model"] == "b", "Each request must select its configured teacher."
            assert not torch.equal(
                stored[("train", "sample_0")]["teacher_logprobs"], stored[("train", "other")]["teacher_logprobs"]
            ), "Mixed batches must preserve different teachers' distributions."
            assert [call["input_ids"] for call in calls[:2]] == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]], (
                "All tokens must be sent."
            )
            assert all(call["sampling_params"] == {"temperature": 1.0, "max_new_tokens": 1} for call in calls), (
                "Scoring must use fixed sampling."
            )
            assert all(call["return_prompt_token_ids"] and not call["return_text_in_logprobs"] for call in calls), (
                "Scoring must preserve IDs."
            )
        finally:
            await manager.stop_busy_loop()


@pytest.mark.parametrize("topk", [0, 3])
def test_protocol_response_slice_and_topk_sort(topk):
    result = parse_teacher_response(teacher_response([1, 2, 3, 4]), [1, 2, 3, 4], 2, topk, 8)
    assert result["teacher_logprobs"].shape == (2, topk or 1), "Only response targets must be returned."
    assert result["teacher_logprobs"].dtype == torch.float32, "Teacher probabilities must be float32."
    if topk:
        assert result["teacher_ids"].tolist() == [[7, 6, 5]] * 2, "Top-k must use the most probable raw IDs."
    else:
        assert (
            result["teacher_logprobs"][0, 0]
            == teacher_response([1, 2, 3, 4])["meta_info"]["input_token_logprobs"][2][0]
        ), "PG must use the observed token."


@pytest.mark.parametrize("damage", ["ids", "length", "first", "missing", "topk", "duplicate", "vocab", "nan"])
def test_protocol_rejects_invalid_targets(damage):
    result = copy.deepcopy(teacher_response([1, 2, 3]))
    meta = result["meta_info"]
    if damage == "ids":
        meta["prompt_token_ids"][-1] = 4
    elif damage == "length":
        meta["input_token_logprobs"].pop()
    elif damage == "first":
        meta["input_top_logprobs"][0] = [[-1.0, 1]]
    elif damage == "missing":
        del meta["input_top_logprobs"]
    elif damage == "topk":
        meta["input_top_logprobs"][1] = [[-1.0, 0]]
    elif damage == "duplicate":
        meta["input_top_logprobs"][1][1] = meta["input_top_logprobs"][1][0]
    elif damage == "vocab":
        meta["input_token_logprobs"][1][1] = 100
    else:
        meta["input_token_logprobs"][1][0] = float("nan")
    with pytest.raises(ValueError):
        parse_teacher_response(result, [1, 2, 3], 1, 3, 8)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), float("inf"), True])
def test_invalid_teacher_weight(weight):
    with pytest.raises(ValueError, match="teacher_weight"):
        normalize_reward_model_dicts([{"reward_loop_type": "opd", "teacher_weight": weight}])


async def test_missing_duplicate_and_unknown_teacher(tmp_path):
    cfg = config(tmp_path)
    worker = make_worker(cfg, {"a": make_loop(cfg, "http://127.0.0.1:1")})
    for specs in (
        [],
        [{"reward_loop_type": "opd", "reward_model_name": "missing"}],
        [{"reward_loop_type": "opd", "reward_model_name": "a"}] * 2,
    ):
        data = request()
        tu.assign_non_tensor_stack(data, "request_keys", [["sample"]])
        tu.assign_non_tensor_stack(data, "reward_model_dicts", [specs])
        with pytest.raises(ValueError):
            await worker._compute_score(worker.pre_process(data))


async def test_context_capacity_and_http_failure(tmp_path):
    cfg = config(tmp_path)
    loop = make_loop(cfg, "http://127.0.0.1:1")
    data = make_worker(cfg, {}).pre_process(request())
    loop.max_model_len = 5
    with pytest.raises(ValueError, match="capacity"):
        await loop.run_single(data)
    async with teacher_service() as (url, _):
        loop.gateway_url, loop.max_model_len = url + "/missing", 32
        with pytest.raises(RuntimeError, match="HTTP 404"):
            await loop.run_single(data)


async def test_stale_results_errors_and_validation_isolation(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    rpc = SimpleNamespace()
    ps = SimpleNamespace(update_request_status=SimpleNamespace(remote=AsyncMock(return_value=[True])))
    manager = RewardLoopManager(cfg, ps, [rpc], is_validate=False)
    manager.request_key_to_attempt["key"] = 2
    result = {
        "uid": "key",
        "request_keys": ["key"],
        "attempt_id": 1,
        "is_validate": False,
        "worker_id": 0,
        "result": None,
        "error": "teacher unavailable",
        "response_len": 2,
    }
    await manager._handle_reward_results([result])
    assert "key" not in manager.request_key_to_reward, "Stale results must not populate rewards."
    result["attempt_id"] = 2
    await manager._handle_reward_results([result])
    put = AsyncMock()
    monkeypatch.setattr(manager_module.tq, "async_kv_batch_put", put)
    with pytest.raises(RuntimeError, match="teacher unavailable"):
        await manager.wait_for_reward_ready(["key"])
    put.assert_not_awaited()
    result["is_validate"] = True
    with pytest.raises(ValueError, match="pool"):
        await manager._handle_reward_results([result])
    with pytest.raises(ValueError, match="pool"):
        await manager.wait_for_reward_of_requests(KVBatchMeta(keys=["key"], tags=[{}], partition_id="val"))


@pytest.mark.integration
@pytest.mark.parametrize("topk", [False, True])
async def test_real_transfer_queue_pipeline(tmp_path, monkeypatch, ray_cluster, topk):
    import transfer_queue as tq

    tq.init(
        OmegaConf.create(
            {
                "controller": {"polling_mode": True},
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {
                        "total_storage_size": 32,
                        "num_data_storage_units": 1,
                    },
                },
            }
        )
    )
    try:
        await test_http_to_reward_queue_and_student(tmp_path, monkeypatch, topk, True, real_queue=True)
    finally:
        tq.close()


async def test_retry_during_status_update_does_not_publish_stale_targets(tmp_path):
    entered, resume = asyncio.Event(), asyncio.Event()

    async def status(*args, **kwargs):
        entered.set()
        await resume.wait()
        return [True]

    ps = SimpleNamespace(update_request_status=SimpleNamespace(remote=status))
    manager = RewardLoopManager(config(tmp_path), ps, [SimpleNamespace()], is_validate=False)
    manager.request_key_to_attempt["sample"] = 1
    result = {
        "uid": "sample",
        "request_keys": ["sample"],
        "attempt_id": 1,
        "is_validate": False,
        "worker_id": 0,
        "result": {"reward_score": 0.0},
        "response_len": 2,
        "error": None,
    }
    collecting = asyncio.create_task(manager._handle_reward_results([result]))
    await entered.wait()
    manager.request_key_to_attempt["sample"] = 2
    resume.set()
    await collecting
    assert "sample" not in manager.request_key_to_reward, "An in-flight old attempt must not publish into a retry."
    assert manager.request_key_to_attempt["sample"] == 2, "The retry's attempt ID must remain active."


async def test_teacher_metadata_is_merged_with_scalar_metadata(tmp_path):
    cfg = config(tmp_path)
    teacher = SimpleNamespace(
        run_single=AsyncMock(
            return_value={
                "reward_score": 0.0,
                "teacher_logprobs": torch.full((3, 1), -1.0),
                "reward_extra_info": {"source": "teacher"},
                "reward_metrics": {"latency": 0.5},
            }
        )
    )
    worker = make_worker(cfg, {"a": teacher})
    data = request()
    tu.assign_non_tensor_stack(data, "request_keys", [["sample"]])
    result = await worker._compute_score(worker.pre_process(data))
    manager = RewardLoopManager(cfg, SimpleNamespace(), [SimpleNamespace()], is_validate=False)
    await manager._set_reward_for_request_keys(["sample"], result)
    reward = manager.request_key_to_reward["sample"]
    assert reward["reward_extra_info"]["opd/default/a"]["source"] == "teacher", "Teacher metadata must be retained."
    assert reward["reward_metrics"]["opd/default/a"]["latency"] == 0.5, "Teacher metrics must be retained."
    assert "naive/compute_score/None" in reward["reward_extra_info"], "Scalar metadata must be retained."
