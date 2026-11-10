"""Check tail distillation against an explicit coarse vocabulary distribution."""

from types import SimpleNamespace

import pytest
import torch
from psrl.trainer.distillation.data import align_teacher_targets
from psrl.workers.config import DistillationConfig, DistillationLossConfig
from verl.trainer.distillation.fsdp.losses import compute_forward_kl_topk
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.trainer.distillation.tail_kl import compute_tail_aware_logit_gradient
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("topk", [1, 3, 8])
@pytest.mark.parametrize("chunked", [False, True])
def test_tail_loss_and_gradient_match_coarse_distribution(topk, chunked):
    torch.manual_seed(43)
    teacher = torch.randn(5, 8).softmax(-1)
    probabilities, ids = teacher.topk(topk, -1)
    targets = torch.nested.as_nested_tensor([probabilities[:2].log(), probabilities[2:].log()], layout=torch.jagged)
    token_ids = torch.nested.as_nested_tensor([ids[:2], ids[2:]], layout=torch.jagged)
    logits = torch.randn(1, 5, 8, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    config = DistillationConfig(
        enabled=True,
        distillation_loss=DistillationLossConfig(
            loss_mode="forward_kl_topk_tail", topk=topk, use_chunked_topk=chunked, chunked_topk_chunk_size=2
        ),
    )
    actual = compute_forward_kl_topk(logits, targets, token_ids, config, "thd", include_tail=True)
    student = reference_logits.softmax(-1).squeeze(0).gather(-1, ids)
    teacher_tail = (1 - probabilities.sum(-1, keepdim=True)).clamp_min(0)
    student_tail = (1 - student.sum(-1, keepdim=True)).clamp_min(config.distillation_loss.tail_mass_eps)
    teacher_coarse = torch.cat((probabilities, teacher_tail), -1)
    student_coarse = torch.cat((student, student_tail), -1)
    expected = (
        teacher_coarse * (teacher_coarse.clamp_min(torch.finfo(torch.float32).tiny).log() - student_coarse.log())
    ).sum(-1)
    torch.testing.assert_close(actual["distillation_losses"].squeeze(0), expected, atol=1e-6, rtol=1e-5)
    actual["distillation_losses"].sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, atol=1e-6, rtol=1e-5)
    full_kl = (teacher * (teacher.log() - reference_logits.detach().log_softmax(-1).squeeze(0))).sum(-1)
    assert (expected >= -1e-6).all(), "Coarse KL must be nonnegative."
    assert (expected <= full_kl + 1e-6).all(), "Coarse KL must not exceed the complete vocabulary KL."


def test_tail_preserves_mask_and_trajectory_weight():
    row = {
        "prompts": torch.tensor([0, 1]),
        "responses": torch.tensor([2, 3]),
        "attention_mask": torch.ones(4, dtype=torch.int64),
        "response_mask": torch.tensor([1, 0]),
        "teacher_logprobs": torch.tensor([[0.4, 0.35], [0.4, 0.35]]).log(),
        "teacher_ids": torch.tensor([[0, 1], [0, 1]]),
        "teacher_weight": torch.tensor(1.7),
    }
    data = align_teacher_targets(list_of_dict_to_tensordict([row]), 2, True)
    config = DistillationConfig(
        enabled=True,
        distillation_loss=DistillationLossConfig(loss_mode="forward_kl_topk_tail", topk=2, use_task_rewards=False),
    )
    actor = SimpleNamespace(strategy="fsdp2", loss_agg_mode="token-mean", global_batch_info={})
    logits = torch.tensor([[[0.6, 0.35, 0.05]]]).log().repeat(1, 4, 1).requires_grad_()
    output = distillation_ppo_loss(actor, config, data=data, student_logits=logits)
    output = {key: value.squeeze(0) for key, value in output.items()}
    loss, _ = distillation_ppo_loss(actor, config, model_output=output, data=data)
    expected = (0.4 * torch.log(torch.tensor(0.4 / 0.6)) + 0.25 * torch.log(torch.tensor(0.25 / 0.05))) * 1.7
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 1].abs().sum() > 0, "The first response predictor must receive the tail gradient."
    assert torch.count_nonzero(logits.grad[0, [0, 2, 3]]) == 0, (
        "Prompt and masked response tokens must have no gradient."
    )


@pytest.mark.parametrize("epsilon", [1e-6, 0.9])
def test_sharded_tail_gradient_matches_autograd(epsilon):
    teacher = torch.tensor([0.30, 0.15, 0.14, 0.13, 0.18, 0.10])
    logits = torch.tensor([0.8, 0.3, -0.2, 0.1, -0.4, 0.6], requires_grad=True)
    student = logits.softmax(-1)
    ids = torch.tensor([0, 4])
    topk = teacher[ids]
    teacher_tail = 1 - topk.sum()
    student_tail = (1 - student[ids].sum()).clamp_min(epsilon)
    expected = (topk * (topk.log() - student[ids].log())).sum()
    expected = expected + teacher_tail * (teacher_tail.log() - student_tail.log())
    expected.backward()
    gradients = []
    for start in (0, 3):
        mask = (ids >= start) & (ids < start + 3)
        gradients.append(
            compute_tail_aware_logit_gradient(
                student_probs=student.detach()[start : start + 3].unsqueeze(0),
                teacher_topk_probs=topk.masked_fill(~mask, 0).unsqueeze(0),
                teacher_topk_indices=(ids - start).masked_fill(~mask, 0).unsqueeze(0),
                teacher_topk_mask=mask.unsqueeze(0),
                student_topk_mass=student[ids].sum().detach().unsqueeze(0),
                teacher_topk_mass=topk.sum().unsqueeze(0),
                tail_mass_eps=epsilon,
            ).squeeze(0)
        )
    torch.testing.assert_close(torch.cat(gradients), logits.grad)
