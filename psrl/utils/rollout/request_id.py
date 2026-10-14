"""Helpers for PSRL uid vs vLLM V1 internal request IDs.

vLLM V1 ``InputProcessor.assign_request_id`` rewrites externally supplied IDs to
``{external_req_id}-{random_uuid()[:8]}`` (8 hex chars) for uniqueness. PSRL
tracks active generation tasks by integer ``uid`` from the data pipeline. scheduler
stats and ABORT commands use the internal vLLM id string.
"""

from __future__ import annotations

_VLLM_INTERNAL_SUFFIX_LEN = 8


def is_vllm_internal_request_id(request_id: str) -> bool:
    """Return True if *request_id* matches vLLM's ``external-xxxxxxxx`` pattern."""
    if "-" not in request_id:
        return False
    external_part, suffix = request_id.rsplit("-", 1)
    if not external_part or len(suffix) != _VLLM_INTERNAL_SUFFIX_LEN:
        return False
    return suffix.isalnum() and all(c in "0123456789abcdef" for c in suffix.lower())


def parse_psrl_uid_from_request_id(request_id: int | str) -> int | str:
    """Map a request id (PSRL uid or vLLM internal id) to the PSRL active-task key.

    Examples:
        973 -> 973
        "973" -> 973
        "973-8d745705" -> 973 (vLLM internal id. external part is PSRL uid)
    """
    if isinstance(request_id, int):
        return request_id

    req_id_str = str(request_id)
    if is_vllm_internal_request_id(req_id_str):
        external_part = req_id_str.rsplit("-", 1)[0]
        try:
            return int(external_part)
        except ValueError:
            return external_part

    try:
        return int(req_id_str)
    except ValueError:
        return req_id_str


def canonical_psrl_request_id(request_id: int | str) -> str:
    """Return the router key for a logical or vLLM-internal request ID."""
    return str(parse_psrl_uid_from_request_id(request_id))


def normalize_request_ids_for_vllm_abort(request_ids) -> list[str]:
    """Normalize ids for ``vLLM.abort`` (must be the internal engine request id string)."""
    if request_ids is None:
        return []
    if not isinstance(request_ids, (list, set, tuple)):
        request_ids = [request_ids]
    return [str(rid) for rid in request_ids if rid is not None]
