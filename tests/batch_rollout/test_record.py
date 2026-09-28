"""Tests for the rollout record builder.

The record is the deliverable, so these pin what a downstream scorer can rely on:
verifier output survives untouched, token fields are absent rather than invented
when the backend could not measure them, and a failed episode is still recorded.
"""

import numpy as np
import pytest
from psrl.batch_rollout.record import build_failure_record, build_records
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import TokenOutput
from verl.utils import tensordict_utils as tu


def _batch(uid: int = 5, parent_id: int | None = 2, **extra):
    fields = {"uid": np.asarray([uid])}
    if parent_id is not None:
        fields["parent_id"] = np.asarray([parent_id])
    for name, value in extra.items():
        fields[name] = np.asarray([value], dtype=object)
    return tu.get_tensordict(fields)


def _output(**overrides) -> TokenOutput:
    defaults = {
        "prompt_ids": [1, 2, 3],
        "response_ids": [4, 5],
        "response_mask": [1, 1],
        "num_turns": 2,
        "stop_reason": "stop",
    }
    defaults.update(overrides)
    return TokenOutput(**defaults)


@pytest.mark.cpu_test
def test_verifier_output_survives_verbatim():
    """Scoring happens offline, so whatever the loop captured must reach the dump."""
    harbor_rewards = {"reward": 0.9, "reward_repair": 0.42}
    output = _output(
        extra_fields={
            "harbor_rewards": harbor_rewards,
            "reward_key": "reward_repair",
            "task_name": "laps/gr_repair_easy",
        }
    )

    record = build_records(output, _batch(), TerminateReason.FINISHED, model_name="qwen")[0]

    assert record["extra_fields"]["harbor_rewards"] == harbor_rewards
    assert record["extra_fields"]["reward_key"] == "reward_repair"


@pytest.mark.cpu_test
def test_tokens_are_omitted_unless_explicitly_dumped():
    """A backend that cannot measure tokens must not have them reconstructed."""
    record = build_records(_output(), _batch(), TerminateReason.FINISHED, model_name="qwen")[0]

    assert "tokens" not in record
    # Lengths are still reported, because they come from what the loop returned.
    assert record["response_len"] == 2
    assert record["prompt_len"] == 3


@pytest.mark.cpu_test
def test_tokens_are_included_when_requested():
    """With a token-capturing backend, the payload is carried through."""
    output = _output(response_log_probs=[-0.1, -0.2])

    record = build_records(output, _batch(), TerminateReason.FINISHED, model_name="qwen", dump_tokens=True)[0]

    assert record["tokens"]["response_ids"] == [4, 5]
    assert record["tokens"]["logprobs"] == [-0.1, -0.2]
    assert len(record["tokens"]["response_ids"]) == record["response_len"]


@pytest.mark.cpu_test
def test_forked_trajectories_each_get_a_record():
    """A forking harness yields several trajectories that must stay distinguishable."""
    outputs = [_output(), _output(), _output()]

    records = build_records(outputs, _batch(), TerminateReason.FINISHED, model_name="qwen")

    assert [r["trajectory_index"] for r in records] == [0, 1, 2]
    assert all(r["trajectory_num"] == 3 for r in records)
    assert all(r["uid"] == 5 and r["parent_id"] == 2 for r in records)


@pytest.mark.cpu_test
def test_request_metadata_reaches_the_record():
    """`extra_info` is what identifies the task to an offline scorer."""
    extra_info = {"task_path": "/tasks/gr_repair_easy", "reward_key": "reward_repair"}
    batch = _batch(data_source="sciaccel_rl", extra_info=extra_info)

    record = build_records(_output(), batch, TerminateReason.FINISHED, model_name="qwen")[0]

    assert record["data_source"] == "sciaccel_rl"
    assert record["extra_info"] == extra_info


@pytest.mark.cpu_test
def test_budget_truncated_episodes_are_still_marked_successful():
    """Truncated trajectories carry real work, which the dump must not discard."""
    record = build_records(_output(), _batch(), TerminateReason.MAX_TURNS_EXCEEDED, model_name="qwen")[0]

    assert record["terminate_reason"] == "max_turns_exceeded"
    assert record["successful"] is True


@pytest.mark.cpu_test
def test_failure_is_recorded_rather_than_dropped():
    """Silence would leave a rerun unable to tell "never ran" from "produced nothing"."""
    records = build_failure_record(_batch(), TerminateReason.ROLLOUT_ERROR, model_name="qwen")

    assert len(records) == 1
    assert records[0]["terminate_reason"] == "rollout_error"
    assert records[0]["successful"] is False
    assert records[0]["trajectory_num"] == 0
    assert records[0]["uid"] == 5


@pytest.mark.cpu_test
def test_uid_serves_as_parent_when_there_is_no_group():
    """With rollout_n=1 the batch carries no parent_id, and joins still need one."""
    record = build_records(_output(), _batch(parent_id=None), TerminateReason.FINISHED, model_name="qwen")[0]

    assert record["parent_id"] == record["uid"] == 5


@pytest.mark.cpu_test
def test_text_native_trajectory_reports_zero_response_tokens():
    """A session proxy backend returns no token ids, which must read as zero, not crash."""
    output = _output(response_ids=[], response_mask=[], num_turns=4)

    record = build_records(output, _batch(), TerminateReason.FINISHED, model_name="qwen")[0]

    assert record["response_len"] == 0
    assert record["num_turns"] == 4
