"""Build the JSON record for one collected trajectory.

Everything here is assembled from what the agent loops already produce. Nothing is
recomputed and nothing task-specific is interpreted: a verifier reward, a patch, or
a task name reaches the record only because the loop placed it in
`TokenOutput.extra_fields`, and is written through verbatim. That is what keeps
this module free of any knowledge of Harbor, SWE-bench, or reward shaping, and
what lets a scorer run over the dump afterwards.
"""

from typing import Any

from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import TokenOutput

# Request fields worth carrying into the record. `extra_info` is what identifies the
# task to a downstream scorer, so it travels whole.
_REQUEST_FIELDS = ("data_source", "extra_info", "reward_model", "agent_name")

# `TokenOutput` fields that describe the episode rather than its token payload.
_OUTPUT_FIELDS = ("stop_reason", "num_turns", "rollout_instance_id", "reward_score")


def build_records(
    output: TokenOutput | list[TokenOutput],
    batch: TensorDict,
    terminate_reason: TerminateReason,
    *,
    model_name: str,
    dump_tokens: bool = False,
) -> list[dict[str, Any]]:
    """Convert one episode's trajectories into JSON records.

    Args:
        output (TokenOutput | list[TokenOutput]): Trajectories the loop produced. A
            list means the harness forked one context per turn.
        batch (TensorDict): The originating single-row prompt batch.
        terminate_reason (TerminateReason): Why the episode stopped.
        model_name (str): Model that served the episode.
        dump_tokens (bool): Whether to include the token-level payload. Only
            meaningful when the serving backend captured token ids.

    Returns:
        list[dict]: One record per trajectory.
    """
    outputs = output if isinstance(output, list) else [output]
    uid = tu.get(batch, "uid")[0]
    parent_id = tu.get(batch, "parent_id")[0] if "parent_id" in batch else uid

    request_meta = {field: tu.get(batch, field, [None])[0] for field in _REQUEST_FIELDS if field in batch}

    records = []
    for index, out in enumerate(outputs):
        record: dict[str, Any] = {
            "uid": uid,
            "parent_id": parent_id,
            "trajectory_index": index,
            "trajectory_num": len(outputs),
            "terminate_reason": terminate_reason.value,
            "successful": terminate_reason.is_successful,
            "model": model_name,
            "version_tag": tu.get(batch, "version_tag", [0])[0],
            **request_meta,
        }
        record.update({field: getattr(out, field, None) for field in _OUTPUT_FIELDS})

        response_ids = out.response_ids or []
        record["prompt_len"] = len(out.prompt_ids or [])
        record["response_len"] = len(response_ids)

        # Written through verbatim. The loop owns what goes in here, including any
        # verifier output a downstream scorer needs.
        record["extra_fields"] = dict(out.extra_fields or {})
        if out.agent_reward_info:
            record["agent_reward_info"] = dict(out.agent_reward_info)

        if dump_tokens:
            record["tokens"] = {
                "prompt_ids": out.prompt_ids,
                "response_ids": response_ids,
                "response_mask": out.response_mask,
                "logprobs": out.response_log_probs,
            }

        records.append(record)
    return records


def build_failure_record(
    batch: TensorDict,
    terminate_reason: TerminateReason,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Record an episode that produced no trajectory.

    A failure is written rather than dropped, so the dump distinguishes "this task
    was never attempted" from "this task was attempted and produced nothing". Only
    the latter is visible in the termination histogram, and only the former should
    be retried by a rerun.

    Args:
        batch (TensorDict): The originating single-row prompt batch.
        terminate_reason (TerminateReason): Why the episode produced nothing.
        model_name (str): Model that served the episode.

    Returns:
        list[dict]: A single-element list holding the failure record.
    """
    uid = tu.get(batch, "uid")[0]
    parent_id = tu.get(batch, "parent_id")[0] if "parent_id" in batch else uid
    record: dict[str, Any] = {
        "uid": uid,
        "parent_id": parent_id,
        "trajectory_index": 0,
        "trajectory_num": 0,
        "terminate_reason": terminate_reason.value,
        "successful": False,
        "model": model_name,
        "version_tag": tu.get(batch, "version_tag", [0])[0],
        **{field: tu.get(batch, field, [None])[0] for field in _REQUEST_FIELDS if field in batch},
    }
    return [record]
