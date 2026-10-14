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

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _assemble_routed_experts(records: list[dict], total_len: int) -> np.ndarray | None:
    """Assemble per-turn routed-experts blobs into one position-indexed tensor."""
    blobs = []
    for record in records:
        re = record.get("routed_experts")
        if not re:
            continue
        arr = np.load(io.BytesIO(base64.b64decode(re["data"])))
        blobs.append((int(re["prompt_start"]), arr))

    if not blobs:
        return None

    _, sample = blobs[0]
    routed_experts = np.zeros((max(total_len, 0), *sample.shape[1:]), dtype=sample.dtype)
    for prompt_start, arr in blobs:
        end = min(prompt_start + arr.shape[0], total_len)
        if end > prompt_start:
            routed_experts[prompt_start:end] = arr[: end - prompt_start]
    return routed_experts


def build_training_data(
    accumulated_token_ids: list[int],
    records: list[dict],
    max_trim_tokens: int = 0,
    prompt_ids_override: list[int] | None = None,
) -> dict:
    """Convert one TITO trajectory into canonical training data.

    Args:
        accumulated_token_ids: Full token sequence from TITO store (prompt + all turns).
        records: Per-turn records, each with:
            - prompt_token_count: int
            - output_logprobs: list of [logprob, token_id] pairs (or None)
            - finish_reason: str
        max_trim_tokens: Maximum number of trailing boundary tokens allowed to be trimmed
            on non-last turns. Final-turn boundary tokens are retained. Sourced from
            the SMG GET endpoint ``max_trim_tokens`` field. A ``ValueError`` is raised
            if the actual trim count exceeds this limit, which indicates a TITO merge
            bug rather than a normal boundary-token situation.
        prompt_ids_override: Optional initial prompt token ids rendered by the Python/verl
            path. When provided, these ids are used as ``prompt_ids`` instead of slicing
            TITO ``accumulated_token_ids`` by the first record's ``prompt_token_count``.

    Returns:
        Dict with keys: prompt_ids, response_ids, response_mask, logprobs, num_turns.
    """
    if not records:
        return {
            "prompt_ids": [],
            "response_ids": [],
            "response_mask": [],
            "logprobs": [],
            "routed_experts": None,
            "num_turns": 0,
        }

    all_response_ids: list[int] = []
    all_response_mask: list[int] = []
    all_logprobs: list[float] = []

    cursor = 0
    total_acc_len = len(accumulated_token_ids)
    for i, record in enumerate(records):
        prompt_len = record["prompt_token_count"]
        raw_lps = record.get("output_logprobs") or []

        # Environment/user tokens between previous cursor and this turn's prompt end
        if cursor > 0 and prompt_len > cursor:
            env_ids = accumulated_token_ids[cursor:prompt_len]
            all_response_ids.extend(env_ids)
            all_response_mask.extend([0] * len(env_ids))
            all_logprobs.extend([0.0] * len(env_ids))

            psrl_logger.debug(
                "[TITO turn %d] env_ids: cursor=%d, prompt_len=%d, env_count=%d, env_ids[:5]=%s",
                i,
                cursor,
                prompt_len,
                len(env_ids),
                str(env_ids[:5]),
            )

        # Assistant output tokens
        output_ids = [int(pair[1]) for pair in raw_lps]
        output_logprobs = [float(pair[0]) for pair in raw_lps]

        # Recover token IDs from TITO when log probabilities are unavailable.
        # Zero log probabilities preserve positional alignment.
        if not output_ids and prompt_len < total_acc_len:
            is_last = i == len(records) - 1
            if is_last:
                end = total_acc_len
            else:
                end = records[i + 1]["prompt_token_count"]
            if end > prompt_len:
                output_ids = list(accumulated_token_ids[prompt_len:end])
                output_logprobs = [0.0] * len(output_ids)
                psrl_logger.warning(
                    "[TITO turn %d] output_logprobs missing, recovered %d tokens from accumulated_token_ids",
                    i,
                    len(output_ids),
                )

        # Trailing trim for non-last turns: greedy match against accumulated.
        # The last turn never trims -- boundary tokens are part of the final output.
        is_last = i == len(records) - 1
        if not is_last and output_ids:
            matched = 0
            for j, tid in enumerate(output_ids):
                pos = prompt_len + j
                if pos < len(accumulated_token_ids) and tid == accumulated_token_ids[pos]:
                    matched += 1
                else:
                    break
            trim_count = len(output_ids) - matched

            psrl_logger.debug(
                "[TITO turn %d] trim analysis: is_last=%s, matched=%d, "
                "trim_count=%d, allowed=%d, prompt_len=%d, "
                "output_len=%d, total_acc_len=%d",
                i,
                is_last,
                matched,
                trim_count,
                max_trim_tokens,
                prompt_len,
                len(output_ids),
                total_acc_len,
            )

            # Non-final turns may trim only the model-specific boundary-token allowance.
            allowed = max_trim_tokens  # is_last already guarded above
            if trim_count > allowed:
                raise ValueError(
                    f"TITO trailing trim overflow at turn {i}: "
                    f"trim_count={trim_count} exceeds allowed={allowed} "
                    f"(max_trim_tokens={max_trim_tokens}). "
                    f"output_ids[-3:]={output_ids[-3:]}, "
                    f"accumulated[{prompt_len + matched}:{prompt_len + matched + 3}]="
                    f"{accumulated_token_ids[prompt_len + matched : prompt_len + matched + 3]}"
                )

            if trim_count > 0:
                output_ids = output_ids[:matched]
                output_logprobs = output_logprobs[:matched]
                psrl_logger.debug(
                    "[TITO turn %d] trimmed %d tokens, remaining output_len=%d",
                    i,
                    trim_count,
                    len(output_ids),
                )
        all_response_ids.extend(output_ids)
        all_response_mask.extend([1] * len(output_ids))
        all_logprobs.extend(output_logprobs)

        cursor = prompt_len + len(output_ids)

    first_prompt_len = records[0]["prompt_token_count"]
    prompt_ids = (
        list(prompt_ids_override) if prompt_ids_override is not None else accumulated_token_ids[:first_prompt_len]
    )
    routed_experts = _assemble_routed_experts(records, len(prompt_ids) + len(all_response_ids) - 1)

    # These fields are extended together and indexed interchangeably downstream.
    if not (len(all_response_ids) == len(all_response_mask) == len(all_logprobs)):
        raise AssertionError(
            f"[TITO build_training_data] length drift over {len(records)} turns: "
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
        len(records),
        total_acc_len,
        None if routed_experts is None else routed_experts.shape[0],
    )

    if not all_response_ids and records:
        psrl_logger.error(
            "[TITO] build_training_data: response_ids empty but num_turns=%d! "
            "records=%s, accumulated_len=%d, prompt_len=%d",
            len(records),
            [
                {
                    "prompt_token_count": r.get("prompt_token_count"),
                    "output_logprobs_len": len(r.get("output_logprobs") or []),
                    "output_logprobs_type": type(r.get("output_logprobs")).__name__,
                    "finish_reason": r.get("finish_reason"),
                }
                for r in records
            ],
            total_acc_len,
            len(prompt_ids),
        )

    return {
        "prompt_ids": prompt_ids,
        "response_ids": all_response_ids,
        "response_mask": all_response_mask,
        "logprobs": all_logprobs,
        "routed_experts": routed_experts,
        "num_turns": len(records),
    }
