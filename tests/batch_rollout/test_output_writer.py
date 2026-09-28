"""Tests for `RolloutOutputWriter`.

The writer is what makes a long collection restartable, so the properties under
test are all about surviving a kill: every completed episode is on disk, a
half-written record is not mistaken for a completed one, and a rerun skips exactly
what is already there.
"""

import json
import os

import numpy as np
import pytest

# The writer is a Ray actor. Its behavior lives in the undecorated class, which is
# what these tests drive, so no Ray cluster is needed.
from psrl.batch_rollout.output_writer import ROLLOUT_FILE, SUMMARY_FILE, RolloutOutputWriter

_Writer = RolloutOutputWriter.__ray_metadata__.modified_class


def _writer(tmp_path, resume: bool = True):
    return _Writer(output_dir=str(tmp_path), resume=resume)


def _record(uid: int, reason: str = "finished", **extra):
    return {"uid": uid, "terminate_reason": reason, **extra}


@pytest.mark.cpu_test
def test_records_are_flushed_as_they_land(tmp_path):
    """A kill mid-run must not lose episodes that already finished."""
    writer = _writer(tmp_path)
    writer.append([_record(1)])
    writer.append([_record(2)])

    # Read without closing, as a post-mortem inspection would.
    with open(tmp_path / ROLLOUT_FILE, encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]

    assert [r["uid"] for r in lines] == [1, 2]


@pytest.mark.cpu_test
def test_resume_skips_exactly_the_recorded_uids(tmp_path):
    """Skipping too little repeats an hour of container time, too much loses data."""
    first = _writer(tmp_path)
    first.append([_record(1), _record(2)])
    first.finalize()

    second = _writer(tmp_path)

    assert second.completed_uids() == {1, 2}
    # The prior records survive rather than being truncated.
    assert second.record_count == 2


@pytest.mark.cpu_test
def test_resume_discards_a_truncated_final_line(tmp_path):
    """A half-written record names an episode whose result was never captured."""
    path = tmp_path / ROLLOUT_FILE
    path.write_text(
        json.dumps({"uid": 1, "terminate_reason": "finished"}) + "\n" + '{"uid": 2, "terminate_rea',
        encoding="utf-8",
    )

    writer = _writer(tmp_path)

    assert writer.completed_uids() == {1}, "uid 2 was never fully written and must be collected again."
    assert writer.record_count == 1


@pytest.mark.cpu_test
def test_resume_false_truncates(tmp_path):
    """An explicit fresh run must not silently inherit a prior run's records."""
    first = _writer(tmp_path)
    first.append([_record(1)])
    first.finalize()

    second = _writer(tmp_path, resume=False)

    assert second.completed_uids() == set()
    assert second.record_count == 0


@pytest.mark.cpu_test
def test_appending_after_resume_preserves_prior_records(tmp_path):
    """Resume opens for append, so a second pass must not overwrite the first."""
    first = _writer(tmp_path)
    first.append([_record(1)])
    first.finalize()

    second = _writer(tmp_path)
    second.append([_record(2)])
    second.finalize()

    with open(tmp_path / ROLLOUT_FILE, encoding="utf-8") as fh:
        uids = [json.loads(line)["uid"] for line in fh if line.strip()]
    assert uids == [1, 2]


@pytest.mark.cpu_test
def test_termination_histogram_counts_every_record(tmp_path):
    """The histogram is how a run is judged, so it must include failures."""
    writer = _writer(tmp_path)
    writer.append([_record(1, "finished"), _record(2, "finished")])
    writer.append([_record(3, "rollout_error")])

    summary = writer.finalize()

    assert summary["terminate_reasons"] == {"finished": 2, "rollout_error": 1}
    assert summary["records"] == 3


@pytest.mark.cpu_test
def test_histogram_survives_resume(tmp_path):
    """A resumed run reports the whole collection, not just its own share."""
    first = _writer(tmp_path)
    first.append([_record(1, "finished")])
    first.finalize()

    second = _writer(tmp_path)
    second.append([_record(2, "rollout_error")])
    summary = second.finalize()

    assert summary["terminate_reasons"] == {"finished": 1, "rollout_error": 1}


@pytest.mark.cpu_test
def test_numpy_scalars_from_the_dataloader_encode(tmp_path):
    """uids and extra_info arrive as numpy types, which plain json rejects."""
    writer = _writer(tmp_path)

    writer.append(
        [
            {
                "uid": np.int64(7),
                "terminate_reason": "finished",
                "score": np.float32(0.5),
                "flag": np.bool_(True),
                "ids": np.arange(3),
            }
        ]
    )
    writer.finalize()

    with open(tmp_path / ROLLOUT_FILE, encoding="utf-8") as fh:
        record = json.loads(fh.readline())
    assert record == {
        "uid": 7,
        "terminate_reason": "finished",
        "score": 0.5,
        "flag": True,
        "ids": [0, 1, 2],
    }


@pytest.mark.cpu_test
def test_summary_carries_the_run_metadata(tmp_path):
    """Resume keys on dataloader order, so the summary must record what fixed it."""
    writer = _writer(tmp_path)
    writer.append([_record(1)])

    summary = writer.finalize({"data_seed": 1, "data_shuffle": True, "model": "qwen"})

    on_disk = json.loads((tmp_path / SUMMARY_FILE).read_text(encoding="utf-8"))
    assert on_disk == summary
    assert on_disk["data_seed"] == 1
    assert on_disk["model"] == "qwen"


@pytest.mark.cpu_test
def test_empty_append_is_a_noop(tmp_path):
    """A loop that produced nothing must not create a blank line."""
    writer = _writer(tmp_path)

    assert writer.append([]) == 0
    assert writer.record_count == 0
    assert os.path.getsize(tmp_path / ROLLOUT_FILE) == 0
