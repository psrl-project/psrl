"""Tests for the host-independent grading payload.

Two guarantees are covered:

1. The packaged driver (``grader.zip``) parses marker-delimited and raw logs the
   way the official harness does, and reports infrastructure failures without
   ever consulting a process exit code.
2. No runtime module imports the upstream ``swebench``/``swesmith`` packages.
   Those are confined to prepare-time tooling.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_MODULES = (
    "examples/mini_swe/swebench_grader.py",
    "examples/mini_swe/runner.py",
    "examples/mini_swe/grading/__init__.py",
    "examples/mini_swe/grading/driver.py",
    "examples/mini_swe/grading/payload.py",
    "examples/mini_swe/grading/runtime.py",
    "examples/mini_swe/grading/schema.py",
    "psrl/environments/mini_swe_env.py",
)


def _run_driver(tmp_path: Path, log_text: str, **payload) -> dict:
    """Execute the packaged driver against ``log_text`` and return the scorecard."""
    from examples.mini_swe.grading.payload import grader_zip_bytes

    grade_dir = tmp_path / "grade"
    grade_dir.mkdir(parents=True, exist_ok=True)
    zip_path = grade_dir / "grader.zip"
    zip_path.write_bytes(grader_zip_bytes())
    (grade_dir / "psrl-eval.log").write_text(log_text)
    (grade_dir / "psrl-grade-input.json").write_text(
        json.dumps(
            {
                "instance_id": "inst",
                "repo": payload.get("repo", "django/django"),
                "log_parser": payload.get("log_parser", "parse_log_pytest"),
                "markers": payload.get("markers", True),
                "f2p": payload.get("f2p", ["tests/a.py::test_a"]),
                "p2p": payload.get("p2p", []),
                "eval_type": payload.get("eval_type", "pass_and_fail"),
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(zip_path),
            "--input",
            str(grade_dir / "psrl-grade-input.json"),
            "--log",
            str(grade_dir / "psrl-eval.log"),
            "--output",
            str(grade_dir / "psrl-scorecard.json"),
        ],
        env={"PSRL_GRADE_DIR": str(grade_dir), "PATH": "/usr/bin:/bin"},
        check=True,
    )
    return json.loads((grade_dir / "psrl-scorecard.json").read_text())


def test_driver_grades_marker_delimited_log(tmp_path: Path) -> None:
    scorecard = _run_driver(
        tmp_path,
        "noise\n>>>>> Start Test Output\nPASSED tests/a.py::test_a\n>>>>> End Test Output\n",
        f2p=["tests/a.py::test_a"],
    )
    assert scorecard["resolved"] is True
    assert (scorecard["f2p_pass"], scorecard["f2p_total"]) == (1, 1)


def test_driver_grades_raw_log_without_markers(tmp_path: Path) -> None:
    scorecard = _run_driver(
        tmp_path,
        "PASSED tests/a.py::test_a\nPASSED tests/c.py::test_c\n",
        markers=False,
        f2p=["tests/a.py::test_a"],
        p2p=["tests/c.py::test_c"],
    )
    assert scorecard["resolved"] is True
    assert (scorecard["p2p_pass"], scorecard["p2p_total"]) == (1, 1)


def test_driver_reports_partial_and_unparseable(tmp_path: Path) -> None:
    partial = _run_driver(
        tmp_path / "partial",
        ">>>>> Start Test Output\nPASSED tests/a.py::test_a\nFAILED tests/b.py::test_b\n>>>>> End Test Output\n",
        f2p=["tests/a.py::test_a", "tests/b.py::test_b"],
    )
    assert partial["resolved"] is False
    assert partial["failure_reason"] is None  # model outcome, not infra
    assert (partial["f2p_pass"], partial["f2p_total"]) == (1, 2)

    unparseable = _run_driver(tmp_path / "bad", ">>>>> Tests Errored\n")
    assert unparseable["resolved"] is False
    assert unparseable["failure_reason"] == "log_unparseable"


def test_driver_failure_paths_do_not_use_exit_codes() -> None:
    source = (REPO_ROOT / "examples/mini_swe/grading/driver.py").read_text()
    assert "returncode" not in source
    assert "exit_code" not in source


def test_plan_defaults_to_pytest_for_gym_rows() -> None:
    from examples.mini_swe.grading.schema import GradingPlan

    plan = GradingPlan.from_swe_problem(
        {
            "instance_id": "x",
            "repo": "django/django",
            "eval_script": "#!/bin/bash\nset -xo pipefail\npytest -q\n",
            "FAIL_TO_PASS": ["t.py::a"],
            "PASS_TO_PASS": [],
        }
    )
    assert plan is not None
    # SWE-Gym rows carry no log_parser. Falling back to the repo-keyed map
    # would wrongly pick parse_log_django, so the plan pins the pytest parser.
    assert plan.parser_name == "parse_log_pytest"
    assert plan.markers is False
    assert GradingPlan.from_swe_problem({"instance_id": "x"}) is None


@pytest.mark.parametrize("relative", RUNTIME_MODULES)
def test_runtime_modules_do_not_import_upstream_graders(relative: str) -> None:
    tree = ast.parse((REPO_ROOT / relative).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            roots = [(node.module or "").split(".")[0]]
        else:
            continue
        assert "swebench" not in roots and "swesmith" not in roots, f"{relative} imports an upstream grader package"


def test_swesmith_parsers_accept_the_two_argument_call_convention() -> None:
    # Flattened SWE-smith parsers are single-argument. The registry adapts them
    # to (log, test_spec) so the driver can call every parser uniformly.
    from examples.mini_swe.grading.parsers import parser_registry

    PARSER_BY_NAME = parser_registry()

    smith_parsers = [p for name, p in PARSER_BY_NAME.items() if name.startswith("swesmith_log_parser_")]
    assert smith_parsers
    for parser in smith_parsers:
        parser("PASSED tests/a.py::test_a\n", None)


def test_vendored_package_has_no_upstream_imports() -> None:
    vendor = REPO_ROOT / "examples/mini_swe/grading/_vendor"
    for path in vendor.rglob("*.py"):
        if path.name == "build_registry.py":
            continue  # prepare/build-time only
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import inside the vendored package
                    continue
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            assert "swebench" not in roots and "swesmith" not in roots, path.name


def test_container_command_selects_version_checked_interpreter() -> None:
    # The sandbox login shell auto-activates the task image's ``testbed`` env.
    # Its interpreter may be too old, so run a version-checked one, not ``python3``.
    from examples.mini_swe.grading.payload import build_container_command

    command = build_container_command()
    assert "sys.version_info <" in command
    assert "/opt/miniconda3/bin/python3" in command
    assert command.index("for _c in") < command.index("psrl-grader.zip")


def _plan(eval_script: str):
    from examples.mini_swe.grading.schema import GradingPlan

    return GradingPlan(
        instance_id="x",
        repo="r/r",
        eval_script=eval_script,
        parser_name="parse_log_pytest",
        markers=False,
        f2p=("tests/a.py::test_a",),
        p2p=(),
    )


@pytest.mark.parametrize(
    "line,expected",
    [
        ("python -m pip install -e .", "python"),
        ("python3 -m pip install -e .[all]", None),
        ("python -m pip install -e '.[dev]'", None),
        ("python -m pip install --no-deps -e .", "python"),
        ("python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true", None),
        # Commands that also install something else must keep running verbatim.
        ("python -m pip install -r test-requirements.txt; python -m pip install -e .", None),
        ("python -m pip install -r requirements.txt", None),
        ("python -m pip install -e ./sub", None),
        ("pip install -e .", None),
        ("python -m pip install -e . && pytest", None),
        ("# python -m pip install -e .", None),
    ],
)
def test_standalone_editable_install_detection(line: str, expected: str | None) -> None:
    from examples.mini_swe.grading.editable import standalone_editable_install

    assert standalone_editable_install(line) == expected


def test_render_eval_script_only_rewrites_standalone_editable_install() -> None:
    script = (
        "#!/bin/bash\n"
        "cd /testbed\n"
        "python -m pip install -e .\n"
        "python -m pip install -r test-requirements.txt; python -m pip install -e .\n"
        "pytest -rA tests/a.py\n"
    )
    plan = _plan(script)

    assert plan.render_eval_script(skip_editable_install=False, probe_path="/tmp/psrl-editable-probe.py") == script

    rendered = plan.render_eval_script(skip_editable_install=True, probe_path="/tmp/psrl-editable-probe.py")
    lines = rendered.split("\n")
    assert lines[2].startswith("if python /tmp/psrl-editable-probe.py; then")
    assert lines[2].endswith("else python -m pip install -e .; fi")
    # The chained install line is untouched: skipping it could drop dependencies.
    assert lines[3] == "python -m pip install -r test-requirements.txt; python -m pip install -e ."
    assert lines[4] == "pytest -rA tests/a.py"


@pytest.mark.parametrize(
    "patch,needs_install",
    [
        ("diff --git a/src/foo.py b/src/foo.py\n", False),
        ("diff --git a/pyproject.toml b/pyproject.toml\n", True),
        ("diff --git a/requirements-dev.txt b/requirements-dev.txt\n", True),
        (
            "diff --git a/newpkg/__init__.py b/newpkg/__init__.py\nnew file mode 100644\n",
            True,
        ),
        (
            "diff --git a/src/pkg/__init__.py b/src/pkg/__init__.py\n--- a/src/pkg/__init__.py\n",
            False,
        ),
    ],
)
def test_editable_install_is_kept_only_when_the_patch_needs_it(patch: str, needs_install: bool) -> None:
    """The redundant install is skipped only when the patch leaves the package map intact."""
    from examples.mini_swe.swebench_grader import _patch_needs_editable_install

    assert _patch_needs_editable_install(patch) is needs_install


def test_editable_probe_reports_checkout_state(tmp_path: Path) -> None:
    """The probe rejects empty directories and ambiguous legacy metadata."""
    from examples.mini_swe.grading.payload import editable_probe_source

    probe = tmp_path / "probe.py"
    probe.write_text(editable_probe_source())

    # No distribution is installed from an empty directory.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert (
        subprocess.run(
            [sys.executable, str(probe)],
            cwd=empty,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ""},
        ).returncode
        == 1
    )

    # A legacy (``setup.py develop``) install keeps its egg-info in the tree.
    checkout = tmp_path / "checkout"
    egg_info = checkout / "pkg.egg-info"
    egg_info.mkdir(parents=True)
    (egg_info / "PKG-INFO").write_text("Metadata-Version: 2.1\nName: pkg\nVersion: 0\n")
    result = subprocess.run(
        [sys.executable, str(probe)],
        cwd=checkout,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(checkout)},
    )
    assert result.returncode == 1


@pytest.mark.parametrize("status", ["FAILED", "ERROR", "SKIPPED", "UNKNOWN"])
def test_expected_nonpassing_test_never_disappears(status: str) -> None:
    from examples.mini_swe.grading.scoring import get_eval_tests_report

    report = get_eval_tests_report({"test": status}, {"FAIL_TO_PASS": ["test"]}, eval_type="pass_and_fail")
    assert report["FAIL_TO_PASS"] == {"success": [], "failure": ["test"]}, "Expected test must remain failed."


def test_missing_expected_test_fails_after_json_roundtrip(tmp_path: Path) -> None:
    result = _run_driver(tmp_path, "PASSED unrelated_test\n", markers=False, f2p=["missing_test"])
    assert not result["resolved"], "Missing expected test must not resolve the task."
    assert result["f2p_total"] == 1 and result["f2p_pass"] == 0, "Missing test must count as a failure."


@pytest.mark.parametrize("log", ["", "unparseable output", ">>>>> Start Test Output\n>>>>> End Test Output\n"])
def test_empty_parser_output_is_infrastructure_failure(tmp_path: Path, log: str) -> None:
    result = _run_driver(tmp_path, log, markers=False, eval_type="fail_only")
    assert result["failure_reason"] == "log_unparseable", "Empty parser output must not produce reward."
    assert result["f2p_total"] == 1, "Infrastructure failure must preserve expected counts."


@pytest.mark.parametrize(
    "fields",
    [
        {"f2p": [], "p2p": []},
        {"f2p": "bad json"},
        {"f2p": {"a": 1}},
        {"f2p": [1]},
        {"log_parser": "typo"},
        {"eval_type": "typo"},
        {"markers": "false"},
        {"f2p": ["a"], "p2p": ["a"]},
    ],
)
def test_driver_rejects_invalid_contract(tmp_path: Path, fields: dict) -> None:
    result = _run_driver(tmp_path, "PASSED test\n", **fields)
    assert result["failure_reason"] == "invalid_plan", "Invalid task metadata must fail explicitly."


def test_fail_only_is_explicit_and_distinct() -> None:
    from examples.mini_swe.grading.scoring import get_eval_tests_report

    expected = {"FAIL_TO_PASS": ["missing"]}
    assert get_eval_tests_report({}, expected, eval_type="pass_and_fail")["FAIL_TO_PASS"]["failure"] == ["missing"], (
        "Default scoring must fail missing tests."
    )
    assert get_eval_tests_report({}, expected, eval_type="fail_only")["FAIL_TO_PASS"]["success"] == ["missing"], (
        "Explicit fail_only scoring retains its upstream semantics."
    )
    with pytest.raises(ValueError):
        get_eval_tests_report({}, expected, eval_type="invalid")


@pytest.mark.parametrize("value", ["no JSON", '"single"', "{}", "null", "[1]", 10])
def test_plan_rejects_malformed_test_lists(value) -> None:
    from examples.mini_swe.grading.schema import GradingPlan

    with pytest.raises(ValueError):
        GradingPlan.from_swe_problem({"eval_script": "pytest", "FAIL_TO_PASS": value})


def test_preparation_pins_dataset_specific_parser() -> None:
    from examples.mini_swe.grading.freeze import validate_prepared_problem
    from examples.mini_swe.grading.parsers import DEFAULT_PARSER

    verified = {"repo": "django/django", "eval_script": "pytest", "FAIL_TO_PASS": ["test"]}
    gym = dict(verified)
    validate_prepared_problem(verified)
    validate_prepared_problem(gym, default_parser=DEFAULT_PARSER)
    assert verified["log_parser"] == "parse_log_django", "Verified must use its repository parser."
    assert gym["log_parser"] == DEFAULT_PARSER, "Gym must explicitly select pytest."


@pytest.mark.parametrize(
    "line",
    [
        "python3.14 -m pip install -e .",
        "python -m pip install -ve . --no-deps",
    ],
)
def test_editable_recognizer_has_no_python_minor_version_allowlist(line: str) -> None:
    from examples.mini_swe.grading.editable import standalone_editable_install

    assert standalone_editable_install(line), "Ordinary Python versions must not require a code update."


def test_probe_accepts_only_exact_pep660_checkout(tmp_path: Path) -> None:
    from examples.mini_swe.grading import editable_probe

    class Distribution:
        def __init__(self, target):
            self.target = target

        def read_text(self, name):
            return json.dumps({"url": self.target.as_uri(), "dir_info": {"editable": True}})

    assert editable_probe._pep660_target(Distribution(tmp_path)) == str(tmp_path.resolve()), "Must decode file URLs."


def test_payload_rebuild_is_deterministic_and_self_contained() -> None:
    import io
    import zipfile

    from examples.mini_swe.grading.payload import grader_zip_bytes

    first = grader_zip_bytes()
    grader_zip_bytes.cache_clear()
    assert grader_zip_bytes() == first, "Rebuilding identical sources must produce identical bytes."
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        names = archive.namelist()
        assert "psrl_grading/_vendor/log_parsers/python.py" in names, "Payload must include the parsers."
        assert not any("vendor_parsers" in name or "freeze" in name for name in names), "Build tools must stay out."
        for name in names:
            if name.endswith(".py"):
                ast.parse(archive.read(name), feature_version=(3, 9))


@pytest.mark.parametrize(
    "scorecard",
    [
        [],
        {},
        {"resolved": "false"},
        {"resolved": True, "f2p_pass": 1, "f2p_total": 1, "p2p_pass": 0, "p2p_total": 0, "timeout": True},
        {"resolved": False, "f2p_pass": -1, "f2p_total": 1, "p2p_pass": 0, "p2p_total": 0},
        {"resolved": True, "f2p_pass": 1, "f2p_total": 2, "p2p_pass": 0, "p2p_total": 0},
    ],
)
def test_runtime_rejects_malformed_scorecards(scorecard) -> None:
    from examples.mini_swe.grading.runtime import run_grading

    class Session:
        def exec(self, *args, **kwargs):
            return None

        def write_bytes(self, *args):
            pass

        def read_bytes(self, path):
            return json.dumps(scorecard).encode()

    result = run_grading(Session(), _plan("pytest"), workdir="/testbed", timeout_s=1)
    assert result["failure_reason"] == "grading_incomplete", "Malformed scorecard must be rejected."
    assert not result["resolved"] and result["f2p_total"] == 1, "Failure must preserve task expectations."


def test_runtime_timeout_preserves_counts() -> None:
    from examples.mini_swe.grading.runtime import run_grading

    class Session:
        def write_bytes(self, *args):
            pass

        def exec(self, *args, **kwargs):
            raise TimeoutError("deadline")

    result = run_grading(Session(), _plan("pytest"), workdir="/testbed", timeout_s=1)
    assert result["timeout"] and result["failure_reason"] == "eval_timeout", "Timeout must remain distinguishable."
    assert result["f2p_total"] == 1, "Timeout must preserve expected counts."


@pytest.mark.parametrize("scenario", ["pass", "missing", "no_python"])
def test_runtime_command_executes_real_zipapp(tmp_path: Path, monkeypatch, scenario: str) -> None:
    from types import SimpleNamespace

    from examples.mini_swe.grading import payload, runtime

    for name in (
        "DRIVER_ZIP_PATH",
        "INPUT_PATH",
        "EVAL_SCRIPT_PATH",
        "LOG_PATH",
        "SCORECARD_PATH",
        "EDITABLE_PROBE_PATH",
    ):
        value = str(tmp_path / Path(getattr(payload, name)).name)
        monkeypatch.setattr(payload, name, value)
        if hasattr(runtime, name):
            monkeypatch.setattr(runtime, name, value)
    # A stale success file must never survive into another grading run.
    Path(payload.SCORECARD_PATH).write_text('{"resolved":true}')

    class LocalSession:
        def write_bytes(self, path, data):
            Path(path).write_bytes(data)

        def read_bytes(self, path):
            return Path(path).read_bytes()

        def exec(self, command, *, cwd, timeout_s):
            completed = subprocess.run(
                ["bash", "-c", command], cwd=cwd, timeout=timeout_s, capture_output=True, text=True
            )
            return SimpleNamespace(exit_code=completed.returncode, stderr=completed.stderr)

    test_name = "tests/a.py::test_a" if scenario == "pass" else "unrelated"
    script = f"printf 'PASSED {test_name}\\n'\nexit 1\n"
    candidates = (sys.executable,) if scenario != "no_python" else ("/nonexistent/driver-python",)
    result = runtime.run_grading(
        LocalSession(), _plan(script), workdir=str(tmp_path), timeout_s=10, python_candidates=candidates
    )
    assert result["resolved"] == (scenario == "pass"), "Reward must come from the new log, not a stale scorecard."
    if scenario == "no_python":
        assert result["failure_reason"] == "no_python", "Missing driver interpreter must be reported."
        assert not Path(payload.LOG_PATH).exists(), "Do not run expensive tests without a driver interpreter."


@pytest.mark.parametrize("failure_stage", ["baseline", "reset", "restore_tests", "timeout"])
def test_fresh_grader_stops_on_invalid_baseline(monkeypatch, failure_stage: str) -> None:
    from types import SimpleNamespace

    from examples.mini_swe import swebench_grader
    from psrl.sandbox import ExecResult, SandboxSource, SandboxSpec

    calls = []

    class Session:
        ref = SimpleNamespace(sandbox_id="fake")
        closed = False

        def exec(self, command, **kwargs):
            calls.append(command)
            if failure_stage == "timeout":
                raise TimeoutError("deadline")
            fails = (
                failure_stage == "baseline"
                and command == "git checkout HEAD~1"
                or failure_stage == "reset"
                and command.startswith("git reset")
                or failure_stage == "restore_tests"
                and command.startswith("git checkout --")
            )
            return ExecResult(1 if fails else 0, "failure" if fails else "", "")

        def write_bytes(self, *args):
            pass

        def close(self):
            self.closed = True

    session = Session()
    sandbox = SimpleNamespace(create=lambda spec: session)
    monkeypatch.setattr(swebench_grader, "_extract_eval_test_files", lambda problem: ["tests/a.py"])
    monkeypatch.setattr(
        swebench_grader, "run_grading", lambda *a, **kw: pytest.fail("Invalid baseline reached grading.")
    )
    problem = {"instance_id": "task", "repo": "django/django", "eval_script": "pytest", "FAIL_TO_PASS": ["test"]}
    result = swebench_grader.grade_fresh_container(
        problem,
        "diff --git a/a.py b/a.py\n",
        "smith",
        "image",
        sandbox=sandbox,
        sandbox_spec=SandboxSpec(source=SandboxSource.image("image")),
    )
    assert not result["resolved"] and session.closed, "Baseline failure must fail grading and release its sandbox."
    assert result["timeout"] == (failure_stage == "timeout"), "Timeout classification must survive the adapter."


def test_invalid_plan_does_not_create_a_grader() -> None:
    from types import SimpleNamespace

    from examples.mini_swe.swebench_grader import grade_fresh_container

    sandbox = SimpleNamespace(create=lambda spec: pytest.fail("Invalid task created a sandbox."))
    result = grade_fresh_container(
        {"instance_id": "task", "eval_script": "pytest", "FAIL_TO_PASS": []},
        "diff --git a/a.py b/a.py\n",
        "verified",
        "image",
        sandbox=sandbox,
    )
    assert result["failure_reason"] == "invalid_plan", "Invalid task metadata must stop before sandbox creation."


def test_vendoring_hash_matches_preparation() -> None:
    from examples.mini_swe.grading.parsers import smith_parser_name
    from examples.mini_swe.grading.vendor_parsers import _flatten_method

    source = "def log_parser(self, log):\n    return {}\n"
    assert f"def {smith_parser_name(source)}(" in _flatten_method(None, source), "Parser identity must be shared."
    with pytest.raises(ValueError):
        _flatten_method(None, "def log_parser(self, log):\n    return self.helper(log)\n")


@pytest.mark.parametrize("p2p,expected_f2p_only", [([], True), (["test"], False), ('["test"]', False)])
def test_smith_preparation_runs_every_expected_test(monkeypatch, p2p, expected_f2p_only: bool) -> None:
    from examples.mini_swe.grading import freeze

    calls = []
    monkeypatch.setattr(freeze, "smith_log_parser_name", lambda problem: "parse_log_pytest")

    def generate(problem, *, f2p_only):
        calls.append(f2p_only)
        return "pytest"

    monkeypatch.setattr(freeze, "generate_smith_eval_script", generate)
    freeze.freeze_smith_grading({"PASS_TO_PASS": p2p})
    assert calls == [expected_f2p_only], "Prepared eval script must include all expected P2P tests."
