"""
Grade SWE-bench patches in fresh Docker containers.

The grader enforces patch policy, runs tests, and parses official harness results.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from psrl.utils.common.docker_utils import force_remove_containers_by_label

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# --- Constants ---

_DEFAULT_CONTAINER_TIMEOUT = "30m"
_DEFAULT_EVAL_TIMEOUT = 900  # seconds
_OUTPUT_TAIL_BYTES = 4096

# Docker run args that match the agent loop's defaults minus the volume/label
# args (those are added per-call).
_BASE_RUN_ARGS: list[str] = [
    "--rm",
    "--memory=30g",  # 10g was too small: heavy repos (scikit-learn, xarray)
    # run `pip install -e .` inside the container and can
    # temporarily exceed 10g, triggering cgroup OOM kills.
    "--network",
    "host",
    "--add-host",
    "host.docker.internal:host-gateway",
]

# --- Patch policy ---


def _changed_files_from_patch(patch_text: str) -> list[str]:
    """
    Extract the list of changed file paths from a unified git diff.

    Args:
        patch_text (str): Unified diff produced by ``git diff``.

    Returns:
        list[str]: Sorted list of changed file paths.
    """
    files: set[str] = set()
    for line in patch_text.splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            files.add(m.group(2))
    return sorted(files)


def _is_test_like_path(path: str) -> bool:
    """
    Return True if a path looks like a test file or test directory.

    Args:
        path (str): File path relative to the repo root.

    Returns:
        bool: True when the path appears to be a test artifact.
    """
    lower = path.lower()
    parts = lower.split("/")
    return (
        any(part in {"tests", "test", "specs", "spec"} for part in parts)
        or Path(lower).stem.startswith("test_")
        or Path(lower).stem.endswith("_test")
        or Path(lower).stem.startswith("spec_")
        or Path(lower).stem.endswith("_spec")
    )


def _is_config_like_path(path: str) -> bool:
    """
    Return True if a path looks like a project configuration file.

    Args:
        path (str): File path relative to the repo root.

    Returns:
        bool: True when the path appears to be a configuration artifact.
    """
    lower = path.lower()
    name = Path(lower).name
    return name in {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",
        "noxfile.py",
        "makefile",
        "dockerfile",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-test.txt",
        ".pre-commit-config.yaml",
        "conftest.py",
    }


def _extract_eval_test_files(swe_problem: dict[str, Any]) -> list[str]:
    """
    Extract F2P + P2P test file paths from a SWE problem dict.

    Args:
        swe_problem (dict[str, Any]): Dataset row with FAIL_TO_PASS / PASS_TO_PASS.

    Returns:
        list[str]: Sorted list of unique test-file paths for the eval tests.
    """
    files: set[str] = set()
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        for test_id in swe_problem.get(key, []):
            # test_id format: "path/to/test_file.py::ClassName::method"
            file_part = test_id.split("::")[0]
            if file_part:
                files.add(file_part)
    return sorted(files)


def analyze_patch_policy(
    patch_text: str,
    swe_problem: dict[str, Any],
) -> dict[str, Any]:
    """
    Check whether a model patch violates submission policy.

    Policy rules (configurable via env vars, all enabled by default):

    - ``SWE_STRICT_NO_TEST_PATCH=1``: disallow changes to test files.
    - ``SWE_STRICT_NO_CONFIG_PATCH=1``: disallow changes to config files.
    - ``SWE_TEST_PATCH_POLICY_SCOPE=eval_tests_only``: when enforcing the
      test-file rule, only flag files that appear in the FAIL_TO_PASS /
      PASS_TO_PASS lists. Set to ``all_tests`` to flag any test-like path.

    Args:
        patch_text (str): Unified diff produced by the agent.
        swe_problem (dict[str, Any]): Dataset row for the SWE problem
            (needs FAIL_TO_PASS, PASS_TO_PASS).

    Returns:
        dict[str, Any]: Policy analysis result containing at least
            ``violated`` (bool) and ``reasons`` (list[str]).
    """
    strict_no_test = os.getenv("SWE_STRICT_NO_TEST_PATCH", "1").strip() != "0"
    strict_no_config = os.getenv("SWE_STRICT_NO_CONFIG_PATCH", "1").strip() != "0"
    scope = os.getenv("SWE_TEST_PATCH_POLICY_SCOPE", "eval_tests_only").strip().lower()
    if scope not in {"all_tests", "eval_tests_only"}:
        scope = "eval_tests_only"

    changed_files = _changed_files_from_patch(patch_text)
    test_files = [f for f in changed_files if _is_test_like_path(f)]
    config_files = [f for f in changed_files if _is_config_like_path(f)]
    eval_test_files = _extract_eval_test_files(swe_problem)
    eval_test_file_set = set(eval_test_files)
    matched_eval_test_files = [f for f in test_files if f in eval_test_file_set]

    reasons: list[str] = []
    if strict_no_test:
        if scope == "all_tests" and test_files:
            reasons.append("test_file_modified")
        elif scope == "eval_tests_only" and matched_eval_test_files:
            reasons.append("eval_test_file_modified")
    if strict_no_config and config_files:
        reasons.append("config_file_modified")

    return {
        "violated": len(reasons) > 0,
        "reasons": reasons,
        "changed_files": changed_files,
        "test_files": test_files,
        "config_files": config_files,
        "eval_test_files": eval_test_files,
        "matched_eval_test_files": matched_eval_test_files,
        "test_policy_scope": scope,
    }


# --- Evaluation script resolution ---


@lru_cache(maxsize=2048)
def _get_verified_eval_script(swe_problem_id: str, swe_problem_json: str) -> str:
    """
    Build and cache the eval script for a SWE-bench Verified SWE problem.

    Args:
        swe_problem_id (str): SWE problem ID (the HF ``instance_id`` field),
            used as the cache key.
        swe_problem_json (str): JSON-serialised SWE problem dict (full row).

    Returns:
        str: Bash eval script from ``make_test_spec``.
    """
    import json

    from swebench.harness.test_spec.test_spec import make_test_spec

    swe_problem = json.loads(swe_problem_json)
    ts = make_test_spec(swe_problem)
    return ts.eval_script


def _get_smith_eval_script(swe_problem: dict[str, Any]) -> str:
    """
    Build the eval script for a SWE-smith SWE problem using the profiles registry.

    Args:
        swe_problem (dict[str, Any]): Full SWE-smith dataset row.

    Returns:
        str: Bash eval script string.
    """
    from swesmith.profiles import registry

    rp = registry.get_from_inst(swe_problem)
    cmd, _ = rp.get_test_cmd(swe_problem, f2p_only=False)
    # Wrap into a minimal bash script consistent with SWE-smith eval.sh format.
    return "\n".join(
        [
            "#!/bin/bash",
            "set -uxo pipefail",
            "cd /testbed",
            cmd,
        ]
    )


# --- Grading helpers ---


def _grade_verified(
    swe_problem: dict[str, Any],
    model_patch: str,
    eval_output: str,
    apply_ok: bool,
) -> dict[str, Any]:
    """
    Grade a Verified SWE problem rollout with the swebench harness.

    Args:
        swe_problem (dict[str, Any]): Full SWE-bench Verified row.
        model_patch (str): Patch submitted by the model.
        eval_output (str): Raw stdout/stderr from the eval script.
        apply_ok (bool): Whether git apply succeeded.

    Returns:
        dict[str, Any]: Grading result with at least ``resolved``,
            ``f2p_pass``, ``f2p_total``, ``p2p_pass``, ``p2p_total``,
            and ``resolved_by`` fields.
    """
    import json

    from swebench.harness.grading import get_eval_report
    from swebench.harness.test_spec.test_spec import make_test_spec

    f2p = swe_problem.get("FAIL_TO_PASS", [])
    p2p = swe_problem.get("PASS_TO_PASS", [])
    if isinstance(f2p, str):
        f2p = json.loads(f2p)
    if isinstance(p2p, str):
        p2p = json.loads(p2p)

    if not apply_ok:
        return {
            "resolved": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "resolved_by": "apply_failed",
        }

    try:
        ts = make_test_spec(swe_problem)
        prediction = {
            "instance_id": swe_problem["instance_id"],
            "model_name_or_path": "psrl/rollout",
            "model_patch": model_patch,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tf:
            tf.write(eval_output)
            log_path = tf.name
        # Unwrap the instance entry returned by swebench 4 and newer.
        report_map = get_eval_report(ts, prediction, log_path, include_tests_status=True)
        report = report_map.get(swe_problem["instance_id"], {})
        resolved = bool(report.get("resolved", False))
        tests_status = report.get("tests_status", {})
        f2p_status = tests_status.get("FAIL_TO_PASS", {})
        p2p_status = tests_status.get("PASS_TO_PASS", {})
        return {
            "resolved": resolved,
            "f2p_pass": len(f2p_status.get("success", [])),
            "f2p_total": len(f2p),
            "p2p_pass": len(p2p_status.get("success", [])),
            "p2p_total": len(p2p),
            "resolved_by": "harness",
        }
    except Exception as exc:
        psrl_logger.warning(f"[swebench_grader] Verified harness grading failed, falling back to returncode: {exc}.")
        return {
            "resolved": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "resolved_by": "harness_error",
        }


def _grade_smith(
    swe_problem: dict[str, Any],
    model_patch: str,
    eval_output: str,
    apply_ok: bool,
    returncode: int,
) -> dict[str, Any]:
    """
    Grade a SWE-smith-py SWE problem rollout with the swesmith harness.

    Falls back to returncode if the harness grading raises.

    Args:
        swe_problem (dict[str, Any]): Full SWE-smith-py row.
        model_patch (str): Patch submitted by the model.
        eval_output (str): Raw stdout/stderr from the eval script.
        apply_ok (bool): Whether git apply succeeded.
        returncode (int): Exit code of the eval script.

    Returns:
        dict[str, Any]: Grading result.
    """

    from swesmith.harness.grading import get_eval_report as smith_get_eval_report

    f2p = swe_problem.get("FAIL_TO_PASS", [])
    p2p = swe_problem.get("PASS_TO_PASS", [])

    if not apply_ok:
        return {
            "resolved": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "resolved_by": "apply_failed",
        }

    try:
        prediction = {
            "instance_id": swe_problem["instance_id"],
            "model_name_or_path": "psrl/rollout",
            "model_patch": model_patch,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tf:
            tf.write(eval_output)
            log_path = tf.name
        report = smith_get_eval_report(prediction, swe_problem, log_path)
        resolved = bool(report.get("resolved", False))
        tests_status = report.get("tests_status", {})
        f2p_status = tests_status.get("FAIL_TO_PASS", {})
        p2p_status = tests_status.get("PASS_TO_PASS", {})
        return {
            "resolved": resolved,
            "f2p_pass": len(f2p_status.get("success", [])),
            "f2p_total": len(f2p),
            "p2p_pass": len(p2p_status.get("success", [])),
            "p2p_total": len(p2p),
            "resolved_by": "harness",
        }
    except Exception as exc:
        psrl_logger.warning(f"[swebench_grader] SWE-smith harness grading failed, falling back to returncode: {exc}.")
        resolved_fallback = returncode == 0
        return {
            "resolved": resolved_fallback,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "resolved_by": "returncode_fallback",
        }


# --- SWE-Gym grading ---


def _get_gym_eval_script(swe_problem: dict[str, Any]) -> str:
    """Retrieve the pre-computed eval_script from the swe_problem dict.

    SWE-Gym instances store the eval_script directly in the parquet
    (generated by prepare_swe_gym.py) because the swegym fork of swebench
    cannot coexist with swebench 4.x at runtime.

    Args:
        swe_problem (dict[str, Any]): Full SWE-Gym dataset row.

    Returns:
        str: Bash eval script string.

    Raises:
        ValueError: If eval_script is missing from swe_problem.
    """
    eval_script = swe_problem.get("eval_script", "")
    if not eval_script:
        raise ValueError(
            f"SWE-Gym instance {swe_problem.get('instance_id', '?')} "
            f"missing eval_script in swe_problem. "
            f"Re-run prepare_swe_gym.py to regenerate the parquet."
        )
    return eval_script


def _grade_gym(
    swe_problem: dict[str, Any],
    model_patch: str,
    eval_output: str,
    apply_ok: bool,
) -> dict[str, Any]:
    """Grade a SWE-Gym instance using swebench's pytest parser directly.

    Bypasses ``get_eval_report`` (which requires the repo to be in
    ``MAP_REPO_TO_PARSER``, and SWE-Gym repos are not in swebench 4.x)
    by directly calling ``parse_log_pytest`` + ``get_eval_tests_report``.

    Args:
        swe_problem (dict[str, Any]): Full SWE-Gym dataset row.
        model_patch (str): Patch submitted by the model.
        eval_output (str): Raw stdout/stderr from the eval script.
        apply_ok (bool): Whether git apply succeeded.

    Returns:
        dict[str, Any]: Grading result with ``resolved``, ``f2p_pass``,
            ``f2p_total``, ``p2p_pass``, ``p2p_total``, ``resolved_by``.
    """
    import json as _json

    f2p = swe_problem.get("FAIL_TO_PASS", [])
    p2p = swe_problem.get("PASS_TO_PASS", [])
    if isinstance(f2p, str):
        f2p = _json.loads(f2p)
    if isinstance(p2p, str):
        p2p = _json.loads(p2p)

    if not apply_ok:
        return {
            "resolved": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "resolved_by": "apply_failed",
        }

    try:
        from swebench.harness.constants import KEY_INSTANCE_ID
        from swebench.harness.grading import (
            FAIL_TO_PASS as _FAIL_TO_PASS,
        )
        from swebench.harness.grading import (
            PASS_TO_PASS as _PASS_TO_PASS,
        )
        from swebench.harness.grading import (
            EvalType,
            ResolvedStatus,
            get_eval_tests_report,
            get_resolution_status,
        )
        from swebench.harness.log_parsers.python import parse_log_pytest

        # The pytest parser does not access its type-only `test_spec` argument.
        status_map = parse_log_pytest(eval_output, None)  # type: ignore[arg-type]

        # Build eval_ref for get_eval_tests_report.
        eval_ref = {
            KEY_INSTANCE_ID: swe_problem["instance_id"],
            _FAIL_TO_PASS: f2p,
            _PASS_TO_PASS: p2p,
        }

        report = get_eval_tests_report(status_map, eval_ref, eval_type=EvalType.PASS_AND_FAIL)
        resolved = get_resolution_status(report) == ResolvedStatus.FULL.value

        f2p_status = report.get("FAIL_TO_PASS", {})
        p2p_status = report.get("PASS_TO_PASS", {})
        return {
            "resolved": resolved,
            "f2p_pass": len(f2p_status.get("success", [])),
            "f2p_total": len(f2p),
            "p2p_pass": len(p2p_status.get("success", [])),
            "p2p_total": len(p2p),
            "resolved_by": "harness",
        }
    except Exception as exc:
        psrl_logger.warning(f"[swebench_grader] SWE-Gym harness grading failed, falling back to returncode: {exc}.")
        return {
            "resolved": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "resolved_by": "harness_error",
        }


# --- Main grader entry point ---


def grade_fresh_container(
    swe_problem: dict[str, Any],
    model_patch: str,
    grader_kind: str,
    image_name: str,
    timeout: int = _DEFAULT_EVAL_TIMEOUT,
    swe_task_id: str = "",
    memory: str = "",
) -> dict[str, Any]:
    """
    Grade a model patch in a fresh Docker container.

    Patch policy is checked before the isolated test harness runs.

    Args:
        swe_problem (dict[str, Any]): Full HF dataset row for one SWE problem.
        model_patch (str): Patch submitted by the agent (unified diff).
        grader_kind (str): ``"verified"`` or ``"smith"``.
        image_name (str): Docker image used for the rollout.
        timeout (int): Eval script execution timeout in seconds.
        swe_task_id (str): PSRL rollout episode ID for container labelling.
        memory (str): ``--memory`` limit for the grading container.

    Returns:
        dict[str, Any]: Test counts, policy state, and execution diagnostics.
    """
    from minisweagent.environments.docker import DockerEnvironment

    swe_problem_id: str = swe_problem.get("instance_id", "unknown")
    log_prefix = f"[swebench_grader, task_id={swe_task_id or swe_problem_id}]"
    t0 = time.monotonic()

    f2p = swe_problem.get("FAIL_TO_PASS", [])
    p2p = swe_problem.get("PASS_TO_PASS", [])

    # --- 0. Patch policy guard ---
    if not model_patch:
        psrl_logger.info(f"{log_prefix} No patch submitted, skipping grading.")
        return {
            "policy_violated": False,
            "policy_reasons": [],
            "resolved": False,
            "apply_ok": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "timeout": False,
            "error": "no_patch",
            "elapsed_s": 0.0,
            "output_tail": "",
            "resolved_by": "no_patch",
        }

    policy = analyze_patch_policy(model_patch, swe_problem)
    if policy["violated"]:
        psrl_logger.info(f"{log_prefix} Patch policy violated: {policy['reasons']!r}, skipping eval container.")
        return {
            "policy_violated": True,
            "policy_reasons": policy["reasons"],
            "resolved": False,
            "apply_ok": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "timeout": False,
            "error": None,
            "elapsed_s": time.monotonic() - t0,
            "output_tail": "",
            "resolved_by": "policy_blocked",
        }

    # --- 1. Build eval script before spawning container (may raise) ---
    import json

    eval_script: str = ""
    try:
        if grader_kind == "gym":
            eval_script = _get_gym_eval_script(swe_problem)
        elif grader_kind == "smith":
            eval_script = _get_smith_eval_script(swe_problem)
        else:
            swe_problem_json = json.dumps(swe_problem, default=str)
            eval_script = _get_verified_eval_script(swe_problem_id, swe_problem_json)
    except Exception as exc:
        psrl_logger.error(f"{log_prefix} Failed to build eval script: {exc}.")
        return {
            "policy_violated": False,
            "policy_reasons": [],
            "resolved": False,
            "apply_ok": False,
            "f2p_pass": 0,
            "f2p_total": len(f2p),
            "p2p_pass": 0,
            "p2p_total": len(p2p),
            "timeout": False,
            "error": f"eval_script_build_error: {exc}",
            "elapsed_s": time.monotonic() - t0,
            "output_tail": "",
            "resolved_by": "eval_script_error",
        }

    # --- 2. Spawn fresh eval container ---
    grader_label = (
        f"psrl.grader_task_id={swe_task_id}__eval" if swe_task_id else f"psrl.grader_task_id={swe_problem_id}__eval"
    )
    run_args = list(_BASE_RUN_ARGS)
    if memory:
        # Override the --memory=Xg entry in _BASE_RUN_ARGS with the caller's value.
        run_args = [a for a in run_args if not a.startswith("--memory=")]
        run_args.append(f"--memory={memory}")
    run_args += ["--label", grader_label]
    # The reaper uses the shared actor ID to find grader and rollout containers.
    _actor_id = os.environ.get("PSRL_ACTOR_ID", "")
    if _actor_id:
        run_args += ["--label", f"psrl.actor_id={_actor_id}"]

    # Forward corporate proxy environment variables to the grading container
    # so that pip/apt inside the eval script can reach external package indexes.
    _PROXY_ENV_KEYS = [
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "no_proxy",
        "NO_PROXY",
    ]

    docker_env: DockerEnvironment | None = None
    apply_ok = False
    eval_output = ""
    eval_returncode = -1
    timed_out = False
    error_msg: str | None = None

    try:
        psrl_logger.info(f"{log_prefix} Spawning eval container: image={image_name!r}, grader_kind={grader_kind!r}.")
        docker_env = DockerEnvironment(
            image=image_name,
            cwd="/testbed",
            run_args=run_args,
            forward_env=_PROXY_ENV_KEYS,
            container_timeout=_DEFAULT_CONTAINER_TIMEOUT,
        )
        psrl_logger.info(f"{log_prefix} Eval container started: id={docker_env.container_id!r}.")

        # --- 3. SWE-smith: restore F2P test files via HEAD~1 ---
        if grader_kind == "smith":
            out = docker_env.execute(
                {"command": "git checkout HEAD~1"},
                cwd="/testbed",
            )
            if out["returncode"] != 0:
                psrl_logger.warning(
                    f"{log_prefix} git checkout HEAD~1 failed (rc={out['returncode']}): {out['output'][:200]}."
                )

        # --- 4. Reset tree and apply model patch ---
        apply_cmd = "git reset --hard HEAD && git clean -fd"
        out = docker_env.execute({"command": apply_cmd}, cwd="/testbed")
        if out["returncode"] != 0:
            psrl_logger.warning(f"{log_prefix} git reset failed (rc={out['returncode']}).")

        # Write patch to a tmp file inside the container via heredoc.
        delimiter = "PSRL_PATCH_EOF"
        apply_cmd2 = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"
        out2 = docker_env.execute({"command": apply_cmd2}, cwd="/testbed")
        apply_ok = out2["returncode"] == 0
        if not apply_ok:
            psrl_logger.info(f"{log_prefix} git apply failed (rc={out2['returncode']}): {out2['output'][:300]}.")

        # --- 5. SWE-smith: revert test-file changes from the patch ---
        if grader_kind == "smith" and apply_ok:
            eval_test_files = _extract_eval_test_files(swe_problem)
            if eval_test_files:
                files_str = " ".join(eval_test_files)
                out3 = docker_env.execute(
                    {"command": f"git checkout -- {files_str}"},
                    cwd="/testbed",
                )
                if out3["returncode"] != 0:
                    psrl_logger.warning(f"{log_prefix} Reverting test files failed (rc={out3['returncode']}).")

        # --- 6. Run eval script ---
        eval_delim = "PSRL_EVAL_EOF"
        eval_cmd = f"bash <<'{eval_delim}'\n{eval_script}\n{eval_delim}"
        psrl_logger.info(f"{log_prefix} Running eval script (timeout={timeout}s)...")

        # Pass a per-command override because the environment timeout covers the container.
        try:
            out_eval = docker_env.execute(
                {"command": eval_cmd},
                cwd="/testbed",
                timeout=timeout,
            )
            eval_output = out_eval.get("output", "")
            eval_returncode = out_eval.get("returncode", -1)
            if out_eval.get("exception_info"):
                timed_out = "timeout" in out_eval.get("exception_info", "").lower()
        except Exception as exc:
            timed_out = "timeout" in str(exc).lower() or "TimeoutExpired" in type(exc).__name__
            error_msg = str(exc)
            psrl_logger.warning(f"{log_prefix} Eval script raised: {exc}.")

    except Exception as exc:
        error_msg = str(exc)
        psrl_logger.error(f"{log_prefix} Container error: {exc}.")
    finally:
        if docker_env is not None:
            try:
                docker_env.cleanup()
            except Exception as cleanup_exc:
                psrl_logger.warning(f"{log_prefix} Container cleanup failed: {cleanup_exc}.")
        # Sweep by label because asynchronous cleanup may return before removal.
        try:
            force_remove_containers_by_label("psrl.grader_task_id", grader_label.split("=", 1)[1])
        except Exception as sweep_exc:
            psrl_logger.warning(f"{log_prefix} Label sweep failed: {sweep_exc}.")

    elapsed = time.monotonic() - t0
    output_tail = eval_output[-_OUTPUT_TAIL_BYTES:] if eval_output else ""

    # --- 7. Grade the eval output ---
    if grader_kind == "smith":
        grade = _grade_smith(swe_problem, model_patch, eval_output, apply_ok, eval_returncode)
    elif grader_kind == "gym":
        grade = _grade_gym(swe_problem, model_patch, eval_output, apply_ok)
    else:
        grade = _grade_verified(swe_problem, model_patch, eval_output, apply_ok)

    # Merge grading result with meta fields.
    result: dict[str, Any] = {
        "policy_violated": False,
        "policy_reasons": [],
        "apply_ok": apply_ok,
        "timeout": timed_out,
        "error": error_msg,
        "elapsed_s": round(elapsed, 2),
        "output_tail": output_tail,
        **grade,
    }

    psrl_logger.info(
        f"{log_prefix} Grading complete: resolved={result['resolved']}, "
        f"apply_ok={apply_ok}, f2p={result['f2p_pass']}/{result['f2p_total']}, "
        f"p2p={result['p2p_pass']}/{result['p2p_total']}, "
        f"elapsed={elapsed:.1f}s, resolved_by={result['resolved_by']!r}."
    )
    return result
