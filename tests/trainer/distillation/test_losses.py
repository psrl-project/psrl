"""Compare OPD loss values and gradients with explicit PyTorch objectives."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from psrl.trainer.distillation import losses
from psrl.trainer.distillation.data import align_teacher_targets
from psrl.workers.config import DistillationConfig, DistillationLossConfig
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

pytestmark = pytest.mark.cpu_test


def batch(topk, padding=False, scale=1.0):
    rows = []
    for prompt, response, mask, weight in [([1, 2], [3, 4, 5], [1, 0, 1], 0.7), ([2], [4, 5], [1, 1], 1.8)]:
        probabilities = torch.log_softmax(torch.tensor([0.1, 1.0, -0.3]), -1)[:topk]
        row = {
            "prompts": torch.tensor(prompt),
            "responses": torch.tensor(response),
            "response_mask": torch.tensor(mask),
            "teacher_logprobs": probabilities.repeat(len(response), 1),
            "teacher_weight": torch.tensor(weight * scale),
        }
        if topk > 1:
            row["teacher_ids"] = torch.tensor([3, 5, 7]).repeat(len(response), 1)
        rows.append(row)
    if padding:
        row = {
            "prompts": torch.tensor([0]),
            "responses": torch.tensor([0]),
            "response_mask": torch.tensor([0]),
            "teacher_logprobs": torch.zeros(1, topk),
            "teacher_weight": torch.tensor(0.0),
        }
        if topk > 1:
            row["teacher_ids"] = torch.zeros(1, topk, dtype=torch.int64)
        rows.append(row)
    return list_of_dict_to_tensordict(rows)


@pytest.mark.parametrize("topk", [False, True])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_loss_and_gradient_match_reference(topk, padding, scale):
    data = batch(3 if topk else 1, padding, scale)
    targets = [row.clone() for row in data["teacher_logprobs"]]
    ids = [row.clone() for row in data["teacher_ids"]] if topk else None
    positions, start = [], 0
    for prompt, response in zip(data["prompts"], data["responses"]):
        positions.append(torch.arange(start + len(prompt) - 1, start + len(prompt) + len(response) - 1))
        start += len(prompt) + len(response)
    torch.manual_seed(4)
    logits = torch.randn(start, 8, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    cfg = SimpleNamespace(strategy="fsdp2", loss_agg_mode="token-mean", global_batch_info={})
    distill = DistillationConfig(
        enabled=True,
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk" if topk else "k1",
            topk=3,
            use_policy_gradient=not topk,
            use_task_rewards=False,
            loss_max_clamp=0.4,
            log_prob_min_clamp=None,
        ),
    )
    data = align_teacher_targets(data, 3 if topk else 1, topk)
    labels = torch.cat([torch.cat((p, r)).roll(-1) for p, r in zip(data["prompts"], data["responses"])])
    output = {"log_probs": logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)}
    data["old_log_probs"] = torch.nested.as_nested_tensor(
        [output["log_probs"][index].detach() for index in positions],
        layout=torch.jagged,
    )
    if topk:
        output.update(
            {
                key: value.squeeze(0)
                for key, value in losses.distillation_ppo_loss(
                    cfg,
                    distill,
                    data=data,
                    student_logits=logits.unsqueeze(0),
                ).items()
            }
        )
    actual, _ = losses.distillation_ppo_loss(cfg, distill, model_output=output, data=data)
    terms = []
    ref_logprobs = reference_logits.log_softmax(-1)
    for i, index in enumerate(positions):
        mask = data["response_mask"][i]
        if topk:
            selected = ref_logprobs[index].gather(-1, ids[i])
            divergence = (targets[i].exp() * (targets[i] - selected)).sum(-1).clamp_min(0).clamp(-0.4, 0.4)
        else:
            selected = ref_logprobs[index].gather(-1, data["responses"][i][:, None]).squeeze(-1)
            divergence = (selected - targets[i].squeeze(-1)).detach().clamp(-0.4, 0.4)
            divergence = divergence * (selected - data["old_log_probs"][i]).exp()
        terms.append((divergence * data["teacher_weight"][i] * mask).sum())
    expected = sum(terms) / sum(row.sum() for row in data["response_mask"])
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)
    if padding:
        assert torch.count_nonzero(logits.grad[-2:]) == 0, "Padding must not contribute gradients."


@pytest.mark.parametrize("use_task_rewards", [False, True])
def test_task_policy_loss_switch(monkeypatch, use_task_rewards):
    distill = DistillationConfig(
        enabled=True,
        distillation_loss=DistillationLossConfig(
            use_task_rewards=use_task_rewards,
            distillation_loss_coef=2.5,
        ),
    )
    task = Mock(return_value=(torch.tensor(3.0), {}))
    monkeypatch.setattr(losses, "ppo_loss", task)
    monkeypatch.setattr(losses, "distillation_loss", Mock(return_value=(torch.tensor(2.0), {})))
    value, _ = losses.distillation_ppo_loss(SimpleNamespace(), distill)
    assert value == (8.0 if use_task_rewards else 2.0), "Only mixed training must apply the configured coefficient."
    assert task.call_count == int(use_task_rewards), "Pure OPD must skip task policy loss."


@pytest.mark.parametrize("damage", ["missing", "shape", "dtype", "nan", "weight", "ids", "vocab", "sequence"])
def test_invalid_targets_fail_before_actor(damage):
    data = batch(3)
    if damage == "missing":
        del data["teacher_logprobs"]
    elif damage == "shape":
        data["teacher_logprobs"] = torch.zeros(2, 1, 3)
    elif damage == "dtype":
        data["teacher_logprobs"] = data["teacher_logprobs"].double()
    elif damage == "nan":
        data["teacher_logprobs"][0][0, 0] = float("nan")
    elif damage == "weight":
        data["teacher_weight"][0] = 0
    elif damage == "ids":
        del data["teacher_ids"]
    elif damage == "vocab":
        data["teacher_ids"][0][0, 0] = 100
    else:
        data["input_ids"] = torch.nested.as_nested_tensor([torch.zeros(5), torch.zeros(3)], layout=torch.jagged)
    with pytest.raises(ValueError):
        align_teacher_targets(data, 3, True, vocab_size=8)


def test_alignment_and_minimal_padding():
    from verl.trainer.ppo.padding_utils import construct_minimal_padding_template

    source = {field: value[0] for field, value in batch(3).items()}
    padding, tag = construct_minimal_padding_template(source, {}, 0, pad_uid=-1)
    assert tag["is_padding"], "Padding must be identified in scheduling metadata."
    assert padding["teacher_logprobs"].shape == (1, 3), "Padding must retain K and shrink R."
    assert padding["teacher_ids"].shape == (1, 3), "Top-k IDs must align with padding."
    assert padding["teacher_weight"] == 0, "Padding distillation weight must be zero."
    data = align_teacher_targets(list_of_dict_to_tensordict([source, padding]), 3, True)
    torch.testing.assert_close(data["teacher_logprobs"][0][1:4], source["teacher_logprobs"])
    assert data["teacher_logprobs"][0].shape == (5, 3), "Targets must span the complete actor sequence."


def test_megatron_dispatch(monkeypatch):
    import sys

    from psrl.trainer.distillation import megatron

    kernel = Mock(return_value={"distillation_losses": torch.zeros(1, 4)})
    module = SimpleNamespace(compute_forward_kl_topk=kernel)
    monkeypatch.setattr(megatron, "losses", module, raising=False)
    monkeypatch.setitem(sys.modules, "psrl.trainer.distillation.megatron.losses", module)
    data = {"teacher_logprobs": torch.zeros(1, 4, 2), "teacher_ids": torch.zeros(1, 4, 2, dtype=torch.int64)}
    losses.compute_topk_loss(SimpleNamespace(strategy="megatron"), None, data, torch.zeros(1, 4, 8), "thd")
    kernel.assert_called_once()


def test_global_aggregation_is_initialized_without_task_loss(monkeypatch):
    from tensordict import TensorDict

    config = SimpleNamespace(global_batch_info={}, loss_scale_factor=7.0)
    distill = DistillationConfig(enabled=True, distillation_loss=DistillationLossConfig(use_task_rewards=False))
    data = TensorDict({}, batch_size=[1])
    from verl.utils import tensordict_utils as tu

    tu.assign_non_tensor(data, dp_size=2, batch_num_tokens=13, global_batch_size=5)
    captured = {}

    def compute(config, *args):
        captured.update(config.global_batch_info)
        return torch.tensor(1.0), {}

    monkeypatch.setattr(losses, "distillation_loss", compute)
    losses.distillation_ppo_loss(config, distill, data=data)
    assert captured == {
        "dp_size": 2,
        "batch_num_tokens": 13,
        "global_batch_size": 5,
        "loss_scale_factor": 7.0,
    }, "Pure OPD must aggregate with the current global batch counts."


@pytest.mark.parametrize("topk", [False, True])
def test_padding_only_microbatch_is_finite_and_has_zero_gradient(topk):
    from verl.utils import tensordict_utils as tu

    data = tu.index_select_tensor_dict(batch(3 if topk else 1, padding=True), [2])
    data = align_teacher_targets(data, 3 if topk else 1, topk)
    logits = torch.randn(1, 2, 8, requires_grad=True)
    output = {"log_probs": logits.log_softmax(-1)[0, :, 0]}
    data["old_log_probs"] = torch.nested.as_nested_tensor([torch.zeros(1)], layout=torch.jagged)
    config = SimpleNamespace(strategy="fsdp2", loss_agg_mode="token-mean", global_batch_info={})
    distill = DistillationConfig(
        enabled=True,
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk" if topk else "k1",
            topk=3,
            use_policy_gradient=not topk,
            use_task_rewards=False,
        ),
    )
    if topk:
        output.update(
            {
                key: value.squeeze(0)
                for key, value in losses.distillation_ppo_loss(
                    config,
                    distill,
                    data=data,
                    student_logits=logits,
                ).items()
            }
        )
    value, metrics = losses.distillation_ppo_loss(config, distill, model_output=output, data=data)
    assert value == 0, "A padding-only microbatch must contribute zero loss."
    value.backward()
    assert torch.count_nonzero(logits.grad) == 0, "Padding must not update student parameters."
    for metric in metrics.values():
        if hasattr(metric, "aggregate"):
            metric = metric.aggregate()
        assert torch.isfinite(torch.as_tensor(metric)), "Padding metrics must remain finite."
