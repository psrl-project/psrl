from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_test


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "psrl" / "utils" / "elastic" / "overhead.py"
    spec = importlib.util.spec_from_file_location("elastic_overhead_for_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_with_metrics(*, scheduled_ts: float, first_token_ts: float):
    metrics = SimpleNamespace(scheduled_ts=scheduled_ts, first_token_ts=first_token_ts)
    return SimpleNamespace(meta_info={"vllm_metrics": [metrics]})


def test_extract_vllm_reprefill_uses_scheduled_to_first_token():
    mod = _load_module()
    result = _result_with_metrics(scheduled_ts=10.25, first_token_ts=10.75)

    assert mod.extract_vllm_reprefill_s(result) == pytest.approx(0.5)


def test_extract_transformers_prefill_metric():
    mod = _load_module()
    result = SimpleNamespace(meta_info={"vllm_metrics": [{"prefill_time": 0.75}]})

    assert mod.extract_vllm_reprefill_s(result) == pytest.approx(0.75)


def test_migration_tracker_breakdown_sums_interrupt_network_and_reprefill(monkeypatch):
    mod = _load_module()
    monotonic_values = iter([100.0, 100.2, 100.3])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(monotonic_values))
    tracker = mod.RequestMigrationOverheadTracker()
    tracker.mark_batch(
        {2: ["request-1", "request-2"]},
        {
            "migration_id": "7:1:Rollout:model",
            "decision_id": 7,
            "planner_s": 0.2,
            "selected_count": 2,
            "planner_breakdown": {
                "candidate_set_construction_s": 0.04,
                "rebalance_simulation_s": 0.06,
                "router_simulation_s": 0.08,
                "best_candidate_selection_s": 0.01,
                "other_s": 0.01,
            },
        },
    )

    interrupted = tracker.mark_requeued("request-1")
    tracker.mark_dispatched("request-1", 5)
    result = tracker.complete(
        "request-1",
        _result_with_metrics(scheduled_ts=20.0, first_token_ts=20.4),
    )

    assert result is not None
    assert interrupted is not None
    assert interrupted["interrupt_s"] == pytest.approx(0.2)
    assert "planner_share_s" not in result
    assert result["interrupt_s"] == pytest.approx(0.2)
    assert result["network_s"] == pytest.approx(0.1)
    assert result["abort_to_redispatch_s"] == pytest.approx(0.3)
    assert result["reprefill_s"] == pytest.approx(0.4)
    assert result["migration_s"] == pytest.approx(0.7)
    assert result["source_instance_id"] == 2
    assert result["destination_instance_id"] == 5


def test_migration_tracker_records_only_the_first_requeue_as_interrupt_completion():
    mod = _load_module()
    tracker = mod.RequestMigrationOverheadTracker()
    tracker.mark_batch(
        {0: ["request-1"]},
        {"selected_count": 1},
    )

    first = tracker.mark_requeued("request-1")
    second = tracker.mark_requeued("request-1")

    assert first is not None
    assert second is None


def test_migration_tracker_waits_for_requeue_and_valid_prefill():
    mod = _load_module()
    tracker = mod.RequestMigrationOverheadTracker()
    tracker.mark_batch({0: ["request-1"]}, {"planner_s": 0.1, "selected_count": 1})

    tracker.mark_dispatched("request-1", 1)
    assert tracker.complete("request-1", _result_with_metrics(scheduled_ts=1.0, first_token_ts=1.2)) is None

    tracker.mark_requeued("request-1")
    tracker.mark_dispatched("request-1", 1)
    assert tracker.complete("request-1", _result_with_metrics(scheduled_ts=0.0, first_token_ts=0.0)) is None


def test_migration_tracker_resolves_router_logical_id_to_scheduler_id(monkeypatch):
    mod = _load_module()
    monotonic_values = iter([10.0, 10.15, 10.4])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(monotonic_values))
    tracker = mod.RequestMigrationOverheadTracker()
    tracker.mark_batch({0: ["12345-abc"]}, {"planner_s": 0.1, "selected_count": 1})

    tracker.mark_requeued("12345")
    tracker.mark_dispatched("12345", 3)
    result = tracker.complete("12345", _result_with_metrics(scheduled_ts=2.0, first_token_ts=2.2))

    assert result is not None
    assert result["interrupt_s"] == pytest.approx(0.15)
    assert result["network_s"] == pytest.approx(0.25)
    assert result["abort_to_redispatch_s"] == pytest.approx(0.4)
    assert result["destination_instance_id"] == 3


def test_mark_dispatched_returns_network_sample_before_completion(monkeypatch):
    mod = _load_module()
    monotonic_values = iter([10.0, 10.1, 10.25])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(monotonic_values))
    tracker = mod.RequestMigrationOverheadTracker()
    tracker.mark_batch({1: ["42-deadbeef"]}, {"planner_s": 0.2, "selected_count": 2})

    tracker.mark_requeued("42")
    dispatch = tracker.mark_dispatched("42", 3)

    assert dispatch is not None
    assert dispatch["interrupt_s"] == pytest.approx(0.1)
    assert dispatch["network_s"] == pytest.approx(0.15)
    assert dispatch["abort_to_redispatch_s"] == pytest.approx(0.25)
    assert dispatch["source_instance_id"] == 1
    assert dispatch["destination_instance_id"] == 3


def test_completion_status_reports_missing_prefill_and_can_be_discarded(monkeypatch):
    mod = _load_module()
    monotonic_values = iter([20.0, 20.2, 20.5])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(monotonic_values))
    tracker = mod.RequestMigrationOverheadTracker()
    tracker.mark_batch({0: ["7-deadbeef"]}, {"planner_s": 0.0, "selected_count": 1})
    tracker.mark_requeued("7")
    tracker.mark_dispatched("7", 2)

    context, status = tracker.complete_with_status("7", SimpleNamespace(meta_info={}))

    assert status == "missing_vllm_prefill"
    assert context is not None
    assert context["interrupt_s"] == pytest.approx(0.2)
    assert context["network_s"] == pytest.approx(0.3)
    tracker.discard("7")
    assert tracker.complete_with_status("7", SimpleNamespace(meta_info={}))[1] == "not_tracked"


def test_tracker_prefers_latest_internal_id_for_replanned_logical_request(monkeypatch):
    mod = _load_module()
    monotonic_values = iter([30.0, 31.0, 31.2, 31.4])
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(monotonic_values))
    tracker = mod.RequestMigrationOverheadTracker()
    tracker.mark_batch({0: ["9-aaaaaaaa"]}, {"planner_s": 0.0, "selected_count": 1})
    tracker.mark_batch({1: ["9-bbbbbbbb"]}, {"planner_s": 0.0, "selected_count": 1})

    tracker.mark_requeued("9")
    dispatch = tracker.mark_dispatched("9", 4)

    assert dispatch is not None
    assert dispatch["source_instance_id"] == 1
    assert dispatch["interrupt_s"] == pytest.approx(0.2)
    assert dispatch["network_s"] == pytest.approx(0.2)
    assert dispatch["abort_to_redispatch_s"] == pytest.approx(0.4)
    tracker.discard("9")
    assert tracker._requests == {}
