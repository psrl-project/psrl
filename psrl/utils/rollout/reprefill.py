"""Dependency-free helpers for partial-rollout reprefill handling."""

from collections.abc import Sequence
from typing import Any


def _as_list(value: Any) -> list:
    """Convert numpy-like/object-array values to a plain list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def reprefill_prefix_matches(
    previous_response_ids: Sequence[int] | Any,
    prompt_token_ids: Sequence[int] | Any,
    previous_response_len: int,
) -> bool:
    """Return whether a reprefill prompt contains the previous response verbatim."""
    previous = _as_list(previous_response_ids)
    prompt = _as_list(prompt_token_ids)
    length = int(previous_response_len)
    if length <= 0 or len(previous) != length or len(prompt) < length:
        return False
    return prompt[-length:] == previous


def should_update_reprefill_log_probs(
    configured: bool,
    *,
    is_reward_model: bool,
    is_teacher_model: bool,
) -> bool:
    """Restrict reprefill logprob updates to policy rollout workers."""
    return bool(configured) and not is_reward_model and not is_teacher_model


def collect_token_log_probs(token_ids: Sequence[int] | Any, token_log_probs: Sequence[Any] | Any) -> list[float]:
    """Collect the selected-token logprob from each vLLM logprob mapping."""
    tokens = _as_list(token_ids)
    log_probs = _as_list(token_log_probs)
    if len(tokens) != len(log_probs):
        raise ValueError(f"Expected one logprob mapping per token, got {len(log_probs)} for {len(tokens)} tokens")

    selected_log_probs = []
    for token_id, log_prob_by_token in zip(tokens, log_probs, strict=True):
        try:
            selected_log_probs.append(log_prob_by_token[token_id].logprob)
        except (KeyError, TypeError):
            # Some vLLM versions stringify prompt-logprob token ids.
            selected_log_probs.append(log_prob_by_token[str(token_id)].logprob)
    return selected_log_probs


def collect_reprefill_log_probs(
    previous_response_ids: Sequence[int] | Any,
    prompt_token_ids: Sequence[int] | Any,
    prompt_log_probs: Sequence[Any] | Any,
    current_response_ids: Sequence[int] | Any,
    current_log_probs: Sequence[Any] | Any,
    previous_response_len: int,
) -> tuple[list[float], bool]:
    """Collect a reprefill chunk and report whether it replaces the old prefix."""
    decode_log_probs = collect_token_log_probs(current_response_ids, current_log_probs)
    previous_response_len = int(previous_response_len)
    if previous_response_len <= 0 or prompt_log_probs is None:
        return decode_log_probs, False

    prompt_tokens = _as_list(prompt_token_ids)
    prefix_tokens = prompt_tokens[-previous_response_len:]
    prefix_log_prob_mappings = _as_list(prompt_log_probs)[-previous_response_len:]
    if len(prefix_tokens) != previous_response_len or len(prefix_log_prob_mappings) != previous_response_len:
        return decode_log_probs, False
    if any(log_prob_mapping is None for log_prob_mapping in prefix_log_prob_mappings):
        return decode_log_probs, False
    if not reprefill_prefix_matches(previous_response_ids, prompt_tokens, previous_response_len):
        return decode_log_probs, False

    prefix_log_probs = collect_token_log_probs(prefix_tokens, prefix_log_prob_mappings)
    return prefix_log_probs + decode_log_probs, True


def merge_reprefill_log_probs(
    previous_log_probs: Sequence[float] | Any,
    current_log_probs: Sequence[float] | Any,
    *,
    replace_previous: bool,
) -> list:
    """Merge one rollout chunk, optionally replacing stale prefix logprobs."""
    previous = _as_list(previous_log_probs)
    current = _as_list(current_log_probs)
    if replace_previous and current:
        return current
    return previous + current


def filter_rollout_request_ids(
    instance_to_request_ids: dict[int, Sequence[int]],
    request_is_validate: dict[int, bool],
) -> dict[int, list[int]]:
    """Filter validation requests from an instance-grouped in-flight snapshot."""
    filtered_by_instance = {}
    for instance_id, request_ids in instance_to_request_ids.items():
        filtered_request_ids = [
            int(request_id) for request_id in request_ids if not request_is_validate.get(int(request_id), False)
        ]
        if filtered_request_ids:
            filtered_by_instance[int(instance_id)] = filtered_request_ids
    return filtered_by_instance
