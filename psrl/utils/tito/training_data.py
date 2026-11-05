"""Build RL training data from a TITO trajectory.

Converts accumulated_token_ids + per-turn records (from SMG GET /tito/sessions)
into the canonical prompt, response, mask, and log-probability fields.
"""

from __future__ import annotations

import base64
import io
import logging
import os

import numpy as np
import torch

from psrl.utils.routed_experts import validate_routed_experts_array
from psrl.utils.sampler_distribution import SamplerDistribution

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_CAPTURE_FULL = "full"


def _assemble_routed_experts(
    records: list[dict],
    prompt_length: int,
    response_length: int,
) -> torch.Tensor | None:
    """Place per-turn routed-expert segments at their absolute prompt offsets."""
    sequence_length = prompt_length + response_length
    expected_shape: tuple[int, ...] | None = None
    expected_dtype: np.dtype | None = None
    segments: list[tuple[int, np.ndarray]] = []
    for turn_index, record in enumerate(records):
        routed_experts = record.get("routed_experts")
        if routed_experts is None:
            continue
        array = np.load(io.BytesIO(base64.b64decode(routed_experts["data"])))
        array = validate_routed_experts_array(array)

        metadata_shape = (int(routed_experts["num_layers"]), int(routed_experts["top_k"]))
        metadata_dtype = np.dtype(routed_experts["dtype"])
        if metadata_shape != array.shape[1:] or metadata_dtype != array.dtype:
            raise ValueError(
                f"routed_experts metadata mismatch at TITO turn {turn_index}, "
                f"metadata shape/dtype={metadata_shape!r}/{metadata_dtype!r}, "
                f"array shape/dtype={array.shape[1:]!r}/{array.dtype!r}."
            )
        if expected_shape is None:
            expected_shape = array.shape[1:]
            expected_dtype = array.dtype
        elif array.shape[1:] != expected_shape or array.dtype != expected_dtype:
            raise ValueError(
                f"routed_experts shape or dtype changed at TITO turn {turn_index}, "
                f"expected {expected_shape!r}/{expected_dtype!r}, "
                f"got {array.shape[1:]!r}/{array.dtype!r}."
            )
        start = int(routed_experts.get("prompt_start", 0))
        segments.append((start, array))

    if not segments or expected_shape is None:
        return None

    canonical = np.zeros((sequence_length, *expected_shape), dtype=np.int16)
    covered = np.zeros(sequence_length, dtype=bool)
    for start, array in segments:
        end = start + array.shape[0]
        if start < 0 or end > sequence_length:
            raise ValueError(
                "routed_experts segment is outside the canonical sequence: "
                f"start={start}, rows={array.shape[0]}, sequence_length={sequence_length}."
            )
        if covered[start:end].any():
            raise ValueError(
                "routed_experts segments overlap in the canonical sequence: "
                f"start={start}, rows={array.shape[0]}."
            )
        canonical[start:end] = array.astype(np.int16, copy=False)
        covered[start:end] = True
    return torch.from_numpy(canonical)


def _record_owner(record: dict, trainable_trajectory_id: int | None) -> int | None:
    owner = record.get("owner_trajectory_id")
    if owner is None:
        return trainable_trajectory_id
    return int(owner)


def _record_capture(record: dict) -> str:
    return str(record.get("capture") or _CAPTURE_FULL)


def _record_output_ids(record: dict) -> list[int]:
    output_ids = record.get("output_token_ids")
    if output_ids is not None:
        return [int(token_id) for token_id in output_ids]
    raw_lps = record.get("output_logprobs") or []
    return [int(pair[1]) for pair in raw_lps]


def _record_logprobs(record: dict) -> list[float]:
    raw_lps = record.get("output_logprobs") or []
    return [float(pair[0]) for pair in raw_lps]


def build_training_data(
    accumulated_token_ids: list[int],
    records: list[dict],
    max_trim_tokens: int = 0,
    prompt_ids_override: list[int] | None = None,
    strict_validation: bool = False,
    trainable_trajectory_id: int | None = None,
    shared_prefix: str = "owner",
) -> dict:
    """Convert one TITO trajectory into canonical training data.

    Args:
        accumulated_token_ids: Full token sequence from TITO store (prompt + all turns).
        records: Per-turn records, each with ``prompt_token_count``, ``output_token_ids``
            (or legacy ``output_logprobs``), and optional ``finish_reason``.
        max_trim_tokens: Maximum number of trailing boundary tokens tolerated on non-last turns.
        prompt_ids_override: Optional initial prompt token ids rendered by the Python/verl path.
        strict_validation: Enable strict sampler distribution validation.
        trainable_trajectory_id: The trajectory whose owned turns may train. None keeps the
            legacy behavior where every turn is trainable.
        shared_prefix: ``owner`` trains every owned node. ``context`` trains only owned
            nodes that are not shared with another trajectory.

    Returns:
        Dict with keys: prompt_ids, response_ids, response_mask, logprobs,
        routed_experts, sampler_distribution, num_turns.

    Raises:
        ValueError: If a turn contributed tokens to ``accumulated_token_ids`` but its
            record cannot describe them, or if the fields appended per turn drift apart.
    """
    if not records:
        return {
            "prompt_ids": [],
            "response_ids": [],
            "response_mask": [],
            "logprobs": [],
            "routed_experts": None,
            "sampler_distribution": None,
            "num_turns": 0,
            "trainable_turns": 0,
        }

    first_record = 0
    if trainable_trajectory_id is not None:
        owned_indices = [
            index
            for index, record in enumerate(records)
            if _record_owner(record, trainable_trajectory_id) == trainable_trajectory_id
        ]
        if not owned_indices:
            return {
                "prompt_ids": [],
                "response_ids": [],
                "response_mask": [],
                "logprobs": [],
                "routed_experts": None,
                "sampler_distribution": None,
                "num_turns": 0,
            }
        first_record = owned_indices[0]

    first_prompt_len = records[first_record]["prompt_token_count"]
    prompt_ids = (
        list(prompt_ids_override) if prompt_ids_override is not None else accumulated_token_ids[:first_prompt_len]
    )
    processed_records = records[first_record:]
    owned_turns = sum(
        1
        for record in processed_records
        if _record_owner(record, trainable_trajectory_id) == trainable_trajectory_id
    )
    trainable_turns = sum(
        1
        for record in processed_records
        if _record_owner(record, trainable_trajectory_id) == trainable_trajectory_id
        and record.get("contributes_to_session", True)
    )
    if trainable_trajectory_id is not None and trainable_turns == 0:
        return {
            "prompt_ids": [],
            "response_ids": [],
            "response_mask": [],
            "logprobs": [],
            "routed_experts": None,
            "sampler_distribution": None,
            "num_turns": 0,
            "trainable_turns": 0,
        }

    all_response_ids: list[int] = []
    all_response_mask: list[int] = []
    all_logprobs: list[float] = []
    capture_distribution = any(record.get("sampler_distribution") is not None for record in processed_records)
    distribution_segments: list[SamplerDistribution] = []

    cursor = first_prompt_len
    total_acc_len = len(accumulated_token_ids)
    accumulated_arr = np.asarray(accumulated_token_ids)
    for offset, record in enumerate(processed_records):
        turn_index = first_record + offset
        prompt_len = record["prompt_token_count"]
        owner = _record_owner(record, trainable_trajectory_id)
        capture = _record_capture(record)
        is_owned = trainable_trajectory_id is None or owner == trainable_trajectory_id
        shared = bool(record.get("shared", False))
        contributes = bool(record.get("contributes_to_session", True))
        train_here = (
            is_owned
            and contributes
            and capture == _CAPTURE_FULL
            and (shared_prefix == "owner" or not shared)
        )

        distribution = None
        if capture_distribution:
            payload = record.get("sampler_distribution")
            raw_lps = record.get("output_logprobs") or []
            if payload is None and train_here:
                raise ValueError(f"Missing sampler distribution at TITO turn {turn_index}.")
            if payload is not None and train_here:
                distribution = SamplerDistribution.from_wire(payload, raw_lps, strict=strict_validation)

        if cursor > prompt_len:
            raise ValueError(
                f"TITO turn {turn_index} prompt length {prompt_len} is behind cursor {cursor}."
            )
        if prompt_len > cursor:
            env_ids = accumulated_token_ids[cursor:prompt_len]
            all_response_ids.extend(env_ids)
            all_response_mask.extend([0] * len(env_ids))
            all_logprobs.extend([0.0] * len(env_ids))
            if capture_distribution:
                distribution_segments.append(SamplerDistribution.empty(len(env_ids)))

        output_ids = _record_output_ids(record)
        output_logprobs = _record_logprobs(record)
        if output_ids and output_logprobs and len(output_ids) != len(output_logprobs):
            raise ValueError(
                f"TITO turn {turn_index} token and logprob counts differ: "
                f"{len(output_ids)} tokens, {len(output_logprobs)} logprobs."
            )
        is_last = turn_index == len(records) - 1
        if not output_ids and prompt_len < total_acc_len:
            raise ValueError(f"Missing output tokens at TITO turn {turn_index}.")

        matched = 0
        compare = min(len(output_ids), total_acc_len - prompt_len)
        if compare > 0:
            equal = np.asarray(output_ids[:compare]) == accumulated_arr[prompt_len : prompt_len + compare]
            matched = compare if bool(equal.all()) else int(np.argmin(equal))
        trim_count = len(output_ids) - matched
        allowed = 0 if is_last else max_trim_tokens
        if trim_count > allowed:
            raise ValueError(
                f"TITO output tokens diverge at turn {turn_index}: trim_count={trim_count} exceeds "
                f"allowed={allowed} (is_last={is_last}, max_trim_tokens={max_trim_tokens}). "
                f"output_ids[-3:]={output_ids[-3:]}, "
                f"accumulated[{prompt_len + matched}:{prompt_len + matched + 3}]="
                f"{accumulated_token_ids[prompt_len + matched : prompt_len + matched + 3]}"
            )
        if trim_count > 0:
            output_ids = output_ids[:matched]
            output_logprobs = output_logprobs[:matched]
            psrl_logger.debug(
                "[TITO turn %d] trimmed %d trailing boundary tokens, remaining output_len=%d",
                turn_index,
                trim_count,
                len(output_ids),
            )

        all_response_ids.extend(output_ids)
        if train_here:
            all_response_mask.extend([1] * len(output_ids))
            if len(output_logprobs) != len(output_ids):
                raise ValueError(
                    f"Missing output_logprobs at TITO turn {turn_index}: "
                    f"{len(output_logprobs)} logprobs for {len(output_ids)} trainable tokens."
                )
            all_logprobs.extend(output_logprobs)
        else:
            all_response_mask.extend([0] * len(output_ids))
            all_logprobs.extend([0.0] * len(output_ids))
        if capture_distribution:
            segment = (
                distribution.prefix(len(output_ids))
                if distribution is not None
                else SamplerDistribution.empty(len(output_ids))
            )
            distribution_segments.append(segment)

        cursor = prompt_len + len(output_ids)

    if cursor != total_acc_len:
        raise ValueError(
            "TITO reconstruction did not consume the accumulated token stream: "
            f"cursor={cursor}, accumulated_length={total_acc_len}."
        )

    routed_experts = _assemble_routed_experts(processed_records, len(prompt_ids), len(all_response_ids))

    if not (len(all_response_ids) == len(all_response_mask) == len(all_logprobs)):
        raise AssertionError(
            f"[TITO build_training_data] length drift over {len(processed_records)} turns: "
            f"response_ids={len(all_response_ids)} response_mask={len(all_response_mask)} "
            f"logprobs={len(all_logprobs)}. These are appended in lockstep per turn, so a "
            "mismatch means one append site diverged from the others."
        )

    psrl_logger.debug(
        "[TITO build_training_data] prompt_len=%d tito_prompt_len=%d response_len=%d "
        "mask_sum=%d logprobs_len=%d num_turns=%d total_acc_len=%d re_tokens=%s",
        len(prompt_ids),
        first_prompt_len,
        len(all_response_ids),
        sum(all_response_mask),
        len(all_logprobs),
        owned_turns,
        total_acc_len,
        None if routed_experts is None else routed_experts.shape[0],
    )

    sampler_distribution = SamplerDistribution.concatenate(distribution_segments) if capture_distribution else None
    if sampler_distribution is not None and len(sampler_distribution) != len(all_response_ids):
        raise ValueError(
            f"sampler distribution has {len(sampler_distribution)} rows but the multi-turn response "
            f"has {len(all_response_ids)} tokens; the two must stay in lockstep across turns."
        )

    return {
        "prompt_ids": prompt_ids,
        "response_ids": all_response_ids,
        "response_mask": all_response_mask,
        "logprobs": all_logprobs,
        "routed_experts": routed_experts,
        "sampler_distribution": sampler_distribution,
        "num_turns": owned_turns if trainable_trajectory_id is not None else len(records),
        "trainable_turns": trainable_turns if trainable_trajectory_id is not None else len(records),
    }
