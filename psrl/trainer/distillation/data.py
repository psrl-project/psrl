"""Validate response targets and align them with full sequence prediction positions."""

import torch
from tensordict import TensorDict


def validate_teacher_targets(
    targets: dict,
    response_length: int,
    topk: int,
    *,
    padding: bool = False,
    vocab_size: int | None = None,
) -> None:
    """Reject missing, malformed or nonfinite teacher targets before actor execution."""
    probs = targets.get("teacher_logprobs")
    if not isinstance(probs, torch.Tensor) or probs.dtype != torch.float32:
        raise ValueError("teacher_logprobs must be a float32 tensor.")
    if probs.shape != (response_length, topk) or not torch.isfinite(probs).all() or (probs > 0).any():
        raise ValueError(f"Invalid teacher_logprobs for response length {response_length!r} and K={topk!r}.")
    weight = torch.as_tensor(targets.get("teacher_weight", float("nan")))
    if weight.numel() != 1 or not torch.isfinite(weight).all() or weight.item() < 0:
        raise ValueError("teacher_weight must be a finite positive scalar, or zero for padding.")
    if (weight.item() == 0) != padding:
        raise ValueError("Only training padding may have zero teacher_weight.")
    ids = targets.get("teacher_ids")
    if topk > 1 or ids is not None:
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.shape != probs.shape:
            raise ValueError("teacher_ids must be an int64 tensor matching teacher_logprobs.")
        if not padding and topk > 1 and (ids.sort(dim=-1).values.diff(dim=-1) == 0).any():
            raise ValueError("Teacher Top-k IDs must be distinct at each response position.")
        if (ids < 0).any() or (vocab_size is not None and (ids >= vocab_size).any()):
            raise ValueError("Teacher IDs are outside the student vocabulary.")


def align_teacher_targets(
    data: TensorDict,
    topk: int,
    use_topk: bool,
    vocab_size: int | None = None,
) -> TensorDict:
    """
    Place response targets at predictor positions `P-1` through `P+R-2`.

    TransferQueue stores targets at response positions. This adapter runs exactly once per actor update.
    """
    required = ["teacher_logprobs", "teacher_weight"] + (["teacher_ids"] if use_topk else [])
    if any(key not in data for key in required):
        raise ValueError(f"Actor input requires teacher fields {required!r}.")
    aligned = {key: [] for key in required if key != "teacher_weight"}
    for index in range(len(data)):
        prompt, response = data["prompts"][index], data["responses"][index]
        if not len(prompt) or not len(response):
            raise ValueError("Actor OPD input requires a nonempty prompt and response.")
        targets = {key: data[key][index] for key in required}
        response_mask = data["response_mask"][index]
        if response_mask.shape != response.shape:
            raise ValueError("OPD response_mask must align with response tokens.")
        padding = float(targets["teacher_weight"]) == 0.0
        if padding and bool(response_mask.any()):
            raise ValueError("Zero teacher_weight requires a fully masked padding trajectory.")
        validate_teacher_targets(targets, len(response), topk, padding=padding, vocab_size=vocab_size)
        if "input_ids" in data and not torch.equal(data["input_ids"][index], torch.cat((prompt, response))):
            raise ValueError("OPD prompt and response do not match the actor input token sequence.")
        for key in aligned:
            values = targets[key]
            full = values.new_zeros((len(prompt) + len(response), topk))
            full[len(prompt) - 1 : len(prompt) + len(response) - 1] = values
            aligned[key].append(full)
    for key, rows in aligned.items():
        data[key] = torch.nested.as_nested_tensor(rows, layout=torch.jagged, device=rows[0].device)
    return data
