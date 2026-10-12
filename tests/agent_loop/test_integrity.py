"""Unit tests for format-dispatched harness integrity scanning and patch re-check."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples.mini_swe.utils.integrity import (
    TRAJECTORY_FORMAT_CLAUDE_CODE,
    TRAJECTORY_FORMAT_CODEX,
    scan_claude_code_integrity,
    scan_trajectory_integrity,
)
from omegaconf import OmegaConf
from psrl.workers.agent_loop.harness import HarnessConfig
from psrl.workers.agent_loop.harness.codex import CodexHarness

_REPO = "foo/bar"
_PROTECTED_PATCH = (
    "diff --git a/tests/test_x.py b/tests/test_x.py\n"
    "--- a/tests/test_x.py\n"
    "+++ b/tests/test_x.py\n"
    "@@ -1 +1 @@\n"
    "-old\n"
    "+new\n"
)
_SWE_PROBLEM = {"FAIL_TO_PASS": ["tests/test_x.py::test_a"], "PASS_TO_PASS": []}


def _lines(*events: dict) -> bytes:
    return "\n".join(json.dumps(event) for event in events).encode()


# --- Format dispatch: Claude Code stream-json ---


def test_claude_code_wrapper_matches_format_dispatch() -> None:
    log = _lines({"type": "tool_use", "name": "Bash", "input": {"command": "git clone https://github.com/foo/bar"}})

    assert scan_claude_code_integrity(log, _REPO) == scan_trajectory_integrity(
        log, TRAJECTORY_FORMAT_CLAUDE_CODE, _REPO
    )
    assert scan_claude_code_integrity(log, _REPO)["violated"] is True


# --- Format dispatch: Codex JSONL ---


def test_codex_command_execution_repo_clone_is_violation() -> None:
    log = _lines(
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "git clone https://github.com/foo/bar /tmp/x"},
        }
    )

    result = scan_trajectory_integrity(log, TRAJECTORY_FORMAT_CODEX, _REPO)

    assert result["violated"] is True
    assert result["reasons"] == ["blocked_repo_web_access"]


def test_codex_file_change_uses_changes_paths() -> None:
    log = _lines(
        {
            "type": "item.completed",
            "item": {"type": "file_change", "changes": [{"path": "tests/test_x.py", "kind": "update"}]},
        }
    )

    result = scan_trajectory_integrity(log, TRAJECTORY_FORMAT_CODEX, _REPO)

    assert result["violated"] is True
    assert result["reasons"] == ["write_to_test_or_harness_path"]


def test_codex_web_search_query_mapped() -> None:
    log = _lines({"type": "item.completed", "item": {"type": "web_search", "query": "https://github.com/foo/bar"}})

    assert scan_trajectory_integrity(log, TRAJECTORY_FORMAT_CODEX, _REPO)["violated"] is True


def test_codex_mcp_tool_call_mapped() -> None:
    log = _lines(
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "tool": "bash",
                "arguments": {"command": "curl https://github.com/foo/bar"},
            },
        }
    )

    assert scan_trajectory_integrity(log, TRAJECTORY_FORMAT_CODEX, _REPO)["violated"] is True


def test_codex_clean_trajectory_is_not_violation() -> None:
    log = _lines(
        {"type": "item.completed", "item": {"type": "command_execution", "command": "pytest -q"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
    )

    result = scan_trajectory_integrity(log, TRAJECTORY_FORMAT_CODEX, _REPO)

    assert result["violated"] is False
    assert result["scannable"] is True


# --- Unscannable handling: warn, never a violation ---


def test_declared_format_zero_parsed_lines_not_violation() -> None:
    result = scan_trajectory_integrity(b"plain text, not json", TRAJECTORY_FORMAT_CODEX, _REPO)

    assert result["violated"] is False
    assert result["scannable"] is False
    assert result["scan_note"]


def test_unknown_format_not_violation() -> None:
    result = scan_trajectory_integrity(b"{}", "not_a_real_format", _REPO)

    assert result["violated"] is False
    assert result["scannable"] is False


def test_empty_stdout_not_violation() -> None:
    result = scan_trajectory_integrity(b"", TRAJECTORY_FORMAT_CLAUDE_CODE, _REPO)

    assert result["violated"] is False
    assert result["scannable"] is False


# --- HarnessConfig trajectory format ---


def test_trajectory_format_auto_resolves_by_kind() -> None:
    assert HarnessConfig(kind="claude_code", executable="claude").resolved_trajectory_format() == (
        TRAJECTORY_FORMAT_CLAUDE_CODE
    )
    assert HarnessConfig(kind="codex", executable="codex").resolved_trajectory_format() == (TRAJECTORY_FORMAT_CODEX)
    assert HarnessConfig(kind="other", executable="tool").resolved_trajectory_format() == "plain_text"


def test_explicit_trajectory_format_overrides_kind() -> None:
    config = HarnessConfig(kind="codex", executable="codex", trajectory_format="plain_text")

    assert config.resolved_trajectory_format() == "plain_text"


def test_config_rejects_unknown_trajectory_format() -> None:
    with pytest.raises(ValueError, match="trajectory_format"):
        HarnessConfig(kind="codex", executable="codex", trajectory_format="bogus")


def test_example_config_declares_trajectory_formats() -> None:
    path = Path(__file__).parents[2] / "examples/mini_swe/config/swebench_harness_config.yaml"
    configs = OmegaConf.load(path)

    resolved = [HarnessConfig.from_value(item.harness).resolved_trajectory_format() for item in configs]
    assert resolved == [TRAJECTORY_FORMAT_CLAUDE_CODE, TRAJECTORY_FORMAT_CODEX]


def test_codex_build_command_includes_json_without_shifting_prefix() -> None:
    harness = CodexHarness(HarnessConfig(kind="codex", executable="codex"), sandbox=None)  # type: ignore[arg-type]
    runtime = SimpleNamespace(session_id="s", session_root_url="http://router/s/s", model="m")

    command = harness.build_command("fix", runtime)  # type: ignore[arg-type]

    assert command[0] == "/opt/harness/bin/codex"
    assert command[1:3] == ("exec", "--skip-git-repo-check")
    assert "--json" in command


# --- Two-layer enforcement ---
#
# Import the loop package lazily and skip cleanly when the environment cannot
# provide its heavier dependencies.


def _import_loop_module():
    from psrl.workers.agent_loop.loops.mini_swe_harness_agent_loop import (
        MiniSWEHarnessAgentLoop,
        MiniSWEHarnessArtifact,
    )

    return MiniSWEHarnessAgentLoop, MiniSWEHarnessArtifact


try:
    _LOOP, _ARTIFACT = _import_loop_module()
    _LOOP_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover (environment dependent)
    _LOOP, _ARTIFACT, _LOOP_IMPORT_ERROR = None, None, exc

requires_loop = pytest.mark.skipif(_LOOP is None, reason=f"mini-SWE loop import unavailable: {_LOOP_IMPORT_ERROR}")


@requires_loop
def test_integrity_failure_merges_trajectory_and_patch_reasons() -> None:
    result = _LOOP._integrity_failure_result(
        _SWE_PROBLEM,
        {"violated": True, "reasons": ["blocked_repo_web_access"]},
        {"violated": True, "reasons": ["eval_test_file_modified"]},
    )

    assert result["policy_violated"] is True
    assert result["resolved"] is False
    assert result["policy_reasons"] == [
        "trajectory:blocked_repo_web_access",
        "patch:eval_test_file_modified",
    ]


def _task(observation: dict) -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(payload={"observation": observation}))


def _bare_loop(monkeypatch: pytest.MonkeyPatch):
    loop = _LOOP.__new__(_LOOP)
    calls: dict = {"graded": False}

    async def fake_grade(task, patch, clean_snapshot):  # noqa: ANN001
        calls["graded"] = True
        return {"resolved": True, "resolved_by": "harness"}

    monkeypatch.setattr(loop, "_grade_patch", fake_grade)
    return loop, calls


@requires_loop
@pytest.mark.asyncio
async def test_finalize_runs_patch_check_even_when_trajectory_violated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWE_STRICT_NO_TEST_PATCH", "1")
    loop, calls = _bare_loop(monkeypatch)
    observation = {"swe_grader": "swebench_fresh_container", "swe_problem": dict(_SWE_PROBLEM)}
    artifact = _ARTIFACT(
        patch=_PROTECTED_PATCH,
        integrity={"violated": True, "reasons": ["blocked_repo_web_access"]},
    )

    out = await loop.finalize_harness_task(_task(observation), artifact, None, {})

    assert calls["graded"] is False
    assert out["grader_result"]["resolved_by"] == "integrity_blocked"
    assert "trajectory:blocked_repo_web_access" in out["grader_result"]["policy_reasons"]
    assert "patch:eval_test_file_modified" in out["grader_result"]["policy_reasons"]


@requires_loop
@pytest.mark.asyncio
async def test_finalize_skips_patch_check_for_non_fresh_container(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWE_STRICT_NO_TEST_PATCH", "1")
    loop, calls = _bare_loop(monkeypatch)
    observation = {"swe_grader": "", "swe_problem": dict(_SWE_PROBLEM)}
    artifact = _ARTIFACT(patch=_PROTECTED_PATCH, integrity={"violated": False, "reasons": []})

    out = await loop.finalize_harness_task(_task(observation), artifact, None, {})

    assert calls["graded"] is True
    assert out["patch_policy"] == {}
