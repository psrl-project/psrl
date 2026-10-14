import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def _load_tracker_class():
    module_path = Path(__file__).resolve().parents[3] / "psrl" / "workers" / "agent_loop" / "transition_tracker.py"
    spec = importlib.util.spec_from_file_location("transition_tracker_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InstanceTransitionTracker


InstanceTransitionTracker = _load_tracker_class()


def test_transition_tracks_snapshot_and_dispatches_crossing_pause_boundary():
    tracker = InstanceTransitionTracker()

    transition_id = tracker.begin({1, 2}, {0: [10], 1: [11], 2: [12, 13]})
    tracker.request_started(2, 14)
    tracker.request_started(0, 15)

    assert tracker.pending_request_ids(transition_id) == {11, 12, 13, 14}

    for request_id in (11, 12, 13, 14):
        tracker.request_resolved(request_id)

    assert tracker.pending_request_ids(transition_id) == set()
    assert tracker.finish(transition_id) == {1, 2}
    assert tracker.pending_request_ids(transition_id) is None
