"""Convert vLLM EngineCoreEvent streams into prefill and decode records."""

import logging
import os

from psrl.utils.profiling.records import DecodeRecord, PrefillRecord, PrefillTrigger

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Event type constants matching EngineCoreEventType values.
# NOTE(claude): Duplicated here to avoid importing vLLM in pure-function code
_QUEUED = 1
_SCHEDULED = 2
_PREEMPTED = 3
_FIRST_TOKEN = 4
_LAST_TOKEN = 5


def _split_into_segments(events: list) -> list[dict]:
    """
    Split a flat event stream into segments, one per QUEUED event.

    Each segment is a dict with keys: queued_ts, scheduled_ts,
    first_token_ts, preempted_ts, last_token_ts (all float | None).

    Args:
        events (list): List of EngineCoreEvent objects with .type and
            .timestamp attributes.

    Returns:
        list[dict]: One segment dict per QUEUED→(SCHEDULED→FIRST_TOKEN|PREEMPTED) cycle.
    """
    segments: list[dict] = []
    current: dict | None = None

    for event in events:
        etype = int(event.type)
        ts = float(event.timestamp)

        if etype == _QUEUED:
            if current is not None:
                segments.append(current)
            current = {
                "queued_ts": ts,
                "scheduled_ts": None,
                "first_token_ts": None,
                "preempted_ts": None,
                "last_token_ts": None,
            }
        elif current is not None:
            if etype == _SCHEDULED:
                current["scheduled_ts"] = ts
            elif etype == _FIRST_TOKEN:
                current["first_token_ts"] = ts
            elif etype == _PREEMPTED:
                current["preempted_ts"] = ts
            elif etype == _LAST_TOKEN:
                current["last_token_ts"] = ts

    if current is not None:
        segments.append(current)

    return segments


def events_to_profiling_records(
    events: list,
    num_cached_tokens: int,
    total_seq_len: int,
    num_output_tokens: int,
) -> tuple[list[dict], list[dict]]:
    """
    Convert an accumulated event stream into PrefillRecord and DecodeRecord dicts.

    Args:
        events (list): Accumulated EngineCoreEvent objects from one engine call.
        num_cached_tokens (int): Prefix cache hit count (from EngineCoreOutput).
        total_seq_len (int): Total sequence length (prompt + output tokens).
        num_output_tokens (int): Number of output tokens generated in this call.

    Returns:
        tuple[list[dict], list[dict]]: (prefill_records, decode_records) as
            serialized dicts ready for non_tensor_batch.
    """
    if not events:
        return [], []

    segments = _split_into_segments(events)
    if not segments:
        return [], []

    prefill_records: list[dict] = []
    decode_records: list[dict] = []

    for seg_idx, seg in enumerate(segments):
        # The router overwrites the first resume. Later segments resume after preemption.
        if seg_idx == 0:
            trigger = PrefillTrigger.RESUME
        else:
            trigger = PrefillTrigger.INTERNAL_PREEMPT_RESUME

        scheduler_wait_s = 0.0
        if seg["queued_ts"] is not None and seg["scheduled_ts"] is not None:
            scheduler_wait_s = max(seg["scheduled_ts"] - seg["queued_ts"], 0.0)

        prefill_duration_s = 0.0
        if seg["scheduled_ts"] is not None and seg["first_token_ts"] is not None:
            prefill_duration_s = max(seg["first_token_ts"] - seg["scheduled_ts"], 0.0)

        # Cache hit info is meaningful only for the first segment.
        seg_num_computed = num_cached_tokens if seg_idx == 0 else 0
        seg_total_seq_len = total_seq_len
        cache_hit_rate = seg_num_computed / seg_total_seq_len if seg_total_seq_len > 0 else 0.0

        pr = PrefillRecord(
            trigger=trigger,
            total_seq_len=seg_total_seq_len,
            num_computed_tokens=seg_num_computed,
            num_prefill_tokens=max(seg_total_seq_len - seg_num_computed, 0),
            cache_hit_rate=cache_hit_rate,
            router_wait_s=0.0,  # Set by router later
            scheduler_wait_s=scheduler_wait_s,
            prefill_duration_s=prefill_duration_s,
        )
        prefill_records.append(pr.to_dict())

        # Record decode only after the segment has produced its first token.
        if seg["first_token_ts"] is not None:
            if seg["preempted_ts"] is not None:
                # Preempted after producing some tokens.
                decode_end_ts = seg["preempted_ts"]
            elif seg["last_token_ts"] is not None:
                # Normal completion: use LAST_TOKEN for accurate decode duration.
                decode_end_ts = seg["last_token_ts"]
            else:
                # A missing `LAST_TOKEN` has no reliable end timestamp. Record zero duration.
                decode_end_ts = seg["first_token_ts"]

            decode_duration_s = max(decode_end_ts - seg["first_token_ts"], 0.0)

            # For multi-segment cases, we can't easily split output tokens per segment.
            # Assign all output tokens to the last decode segment.
            seg_decode_tokens = num_output_tokens if seg_idx == len(segments) - 1 else 0

            dr = DecodeRecord(
                num_decode_tokens=seg_decode_tokens,
                decode_duration_s=decode_duration_s,
            )
            decode_records.append(dr.to_dict())

    return prefill_records, decode_records
