import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_weight_fingerprints.py"
    spec = importlib.util.spec_from_file_location("analyze_weight_fingerprints", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _record(stage, role, digests, *, version=1, instance_id=None, rank=0):
    return {
        "flow": "transfer_chain",
        "stage": stage,
        "role": role,
        "model_version": version,
        "rank": rank,
        "instance_id": instance_id,
        "tensor_digests": digests,
    }


def test_analyzer_compares_transfer_chain_and_sleep_wake(tmp_path):
    records = [
        _record("trainer_before_push", "trainer", {"a|(0,)": "x"}),
        _record("ps_after_receive", "ps_train", {"a|(0,)": "x"}),
        _record("ps_after_layout_copy", "ps_rollout", {"a|(0,)": "x"}),
        _record("rollout_after_raw_pull", "rollout", {"a|(0,)": "x"}, instance_id=3),
        _record("rollout_after_param_sync", "rollout", {"a|(0,)": "x"}, instance_id=3),
    ]
    lines = [f"prefix [WEIGHT_FINGERPRINT] {json.dumps(record)}\n" for record in records]
    lines.append('prefix [TRAINER_SLEEP_WAKE_WEIGHT_CHECK] {"expected_model_version": 1, "status": "match"}\n')
    (tmp_path / "worker.log").write_text("".join(lines), encoding="utf-8")

    fingerprints, sleep_checks = MODULE.parse_weight_verification_logs(tmp_path)
    report = MODULE.analyze_weight_fingerprints(fingerprints, sleep_checks)

    assert report["summary"]["mismatched_checks"] == 0
    assert report["summary"]["trainer_sleep_wake_checks"] == 1
    assert report["summary"]["trainer_sleep_wake_mismatches"] == 0
    assert {comparison["check"] for comparison in report["comparisons"]} == {
        "trainer_to_ps_train",
        "ps_train_to_ps_rollout",
        "ps_rollout_to_rollout_raw",
        "rollout_raw_to_param_sync",
    }


def test_analyzer_reports_changed_rollout_tensor():
    fingerprints = [
        _record("ps_after_layout_copy", "ps_rollout", {"a|(0,)": "x"}),
        _record("rollout_after_raw_pull", "rollout", {"a|(0,)": "bad"}, instance_id=2),
        _record("rollout_after_param_sync", "rollout", {"a|(0,)": "bad"}, instance_id=2),
    ]

    report = MODULE.analyze_weight_fingerprints(fingerprints, [])

    assert report["summary"]["mismatched_checks"] == 1
    mismatch = next(comparison for comparison in report["comparisons"] if not comparison["match"])
    assert mismatch["check"] == "ps_rollout_to_rollout_raw"
    assert mismatch["differing_tensors"] == ["a|(0,)"]
