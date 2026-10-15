import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def _load_module():
    module_path = Path(__file__).resolve().parents[2] / "scripts" / "analyze_partial_rollout_trace.py"
    spec = importlib.util.spec_from_file_location("analyze_partial_rollout_trace", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_reports_cross_version_and_boundary_violations(tmp_path):
    (tmp_path / "GenWorker_I0_R0.log").write_text(
        "2026-08-29 10:00:00,000 x [PARTIAL_ROLLOUT_TRACE] stage=generate_start "
        "uid=7 chunk=0 partial=0 request_version=0 loaded_version=0 instance=0 "
        "current_ps_version=0 request_staleness=0 loaded_staleness=0 previous_instance=none "
        "prefix_tokens=0 prefix_sha256=empty prior_logprob_tokens=0\n"
        "2026-08-29 10:00:01,000 x [PARTIAL_ROLLOUT_TRACE] stage=generate_result "
        "uid=7 chunk=0 partial=0 request_version=0 loaded_version=0 instance=0 prefix_tokens=0 "
        "response_tokens=3 generated_tokens=3 response_sha256=abc rollout_logprob_tokens=3 "
        "interrupted=1 interrupted_by_scheduler=0\n"
        "2026-08-29 10:00:02,000 x [PARTIAL_ROLLOUT_TRACE] stage=generate_start "
        "uid=7 chunk=1 partial=1 request_version=0 loaded_version=1 instance=2 "
        "current_ps_version=3 request_staleness=3 loaded_staleness=2 previous_instance=0 "
        "prefix_tokens=2 prefix_sha256=wrong prior_logprob_tokens=3\n",
        encoding="utf-8",
    )
    (tmp_path / "RolloutRouter.log").write_text(
        "2026-08-29 10:00:01,500 x [PARTIAL_ROLLOUT_TRACE] stage=route uid=7 partial=1 "
        "previous_instance=0 requested_version=0 selected_instance=2 selected_version=1 "
        "route_reason=primary candidates=2:1\n",
        encoding="utf-8",
    )

    events = MODULE.parse_trace_events(tmp_path)
    report, violations = MODULE.analyze_events(
        events,
        require_logprob_alignment=True,
        staleness_limit=2,
    )

    assert report["summary"]["cross_version_uids"] == 1
    assert report["summary"]["router_higher_version_routes"] == 1
    assert report["summary"]["router_cross_instance_chunks"] == 1
    assert report["summary"]["router_cross_instance_uids"] == 1
    assert report["summary"]["max_request_staleness"] == 3
    assert {violation["kind"] for violation in violations} == {
        "prefix_continuity",
        "prefix_logprob_alignment",
        "request_staleness",
    }
