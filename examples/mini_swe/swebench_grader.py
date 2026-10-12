"""
SWE-bench / SWE-smith-py Grader for PSRL RL Training.

Grades a model patch in a fresh sandbox (isolated from the rollout sandbox),
applies the patch, runs the per-SWE-problem eval script, and grades the result.

The evaluation itself is host-independent: the eval script is taken straight
from the prepared parquet (``swe_problem.eval_script``, populated for Verified /
SWE-Gym / SWE-smith) and parsed *inside the grader sandbox* by a stdlib-only
driver built from ``examples/mini_swe/grading``. No ``swebench`` / ``swesmith``
import happens on the training host at rollout time.

Patch policy enforcement (disallow test / config file changes) mirrors
OpenClaw-RL's ``_analyze_patch_policy`` with the same env-var configuration
interface.

Public API
----------
analyze_patch_policy(patch_text, swe_problem) -> dict
grade_fresh_container(swe_problem, model_patch, grader_kind, image_name, ...) -> dict
"""

import base64
import logging
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from examples.mini_swe.grading.runtime import run_grading
from examples.mini_swe.grading.schema import GradingPlan, GradingResult
from psrl.sandbox import ExecResult, SandboxSpec, SnapshotRef, SyncSandboxManager, SyncSandboxSession

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
    # Heavy repositories can exceed 10 GiB while installing build dependencies.
    "--memory=30g",
]

# Forward corporate proxy environment variables to the grading container so that
# pip/apt inside the eval script can reach external package indexes.
_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
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


# Files whose modification can change the installed dependency set or build backend. A patch
# touching one must keep the eval script's editable re-install, which propagates new metadata.
_PACKAGING_FILE_NAMES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "manifest.in",
        "pipfile",
        "pipfile.lock",
        "poetry.lock",
        "environment.yml",
        "environment.yaml",
        "tox.ini",
        "constraints.txt",
    }
)


def _added_files_from_patch(patch: str) -> list[str]:
    """
    Return the paths a unified diff creates.
    """
    added: list[str] = []
    current: str | None = None
    for line in patch.splitlines():
        match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if match:
            current = match.group(2)
        elif line.startswith("new file mode") and current:
            added.append(current)
            current = None
    return added


def _patch_needs_editable_install(patch: str) -> bool:
    """
    Whether re-running the eval script's editable install is required.

    The image already has the checkout installed editable, so that install is
    normally a redundant rebuild. Two cases still need it: a patch that changes
    packaging metadata/dependency pins, and a patch that adds a Python package
    directory. Setuptools editable installs use a static package map, so a
    brand-new package would not be importable until the map is regenerated.
    """
    for path in _changed_files_from_patch(patch):
        name = os.path.basename(path).lower()
        if name in _PACKAGING_FILE_NAMES or name.startswith("requirements"):
            return True
    return any(os.path.basename(path) == "__init__.py" for path in _added_files_from_patch(patch))


def _extract_eval_test_files(swe_problem: dict[str, Any]) -> list[str]:
    """
    Extract F2P + P2P test file paths from a SWE problem dict.
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
        swe_problem (dict[str, Any]): Dataset row for the SWE problem.

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


# --- Grader failure shaping ---


def _grader_failure(swe_problem: dict[str, Any], error: str) -> dict[str, Any]:
    """Build a grader result for a host-side failure (no container involved)."""
    return {
        "policy_violated": False,
        "policy_reasons": [],
        "resolved": False,
        "apply_ok": False,
        "f2p_pass": 0,
        "f2p_total": len(swe_problem.get("FAIL_TO_PASS", [])),
        "p2p_pass": 0,
        "p2p_total": len(swe_problem.get("PASS_TO_PASS", [])),
        "timeout": False,
        "error": error,
        "parser_error": None,
        "failure_reason": None,
        "elapsed_s": 0.0,
        "output_tail": "",
        "resolved_by": "grader_error",
    }


# --- Standalone adapter ---


class _DockerGradingSession:
    """Adapt ``minisweagent``'s DockerEnvironment to the grading session API.

    Only the standalone evaluation CLI takes this path, and training always supplies
    a PSRL ``SyncSandboxManager``. File transfers use base64 so the tiny payload
    survives shell quoting.
    """

    def __init__(self, environment: Any) -> None:
        self._environment = environment

    def exec(self, command: str, *, cwd: str | None = None, timeout_s: float | None = None, **_: Any) -> ExecResult:
        """Run ``command`` and translate minisweagent's dict result."""
        kwargs = {} if timeout_s is None else {"timeout": timeout_s}
        result = self._environment.execute({"command": command}, cwd=cwd, **kwargs)
        exception = result.get("exception_info") or ""
        if exception:
            if "timeout" in exception.lower():
                raise TimeoutError(exception)
            raise RuntimeError(exception)
        return ExecResult(int(result.get("returncode", -1)), result.get("output", ""), "")

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write bytes by decoding a base64 literal through the shell."""
        encoded = base64.b64encode(data).decode()
        command = f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}"
        result = self._environment.execute({"command": command}, cwd="/")
        if result.get("returncode", -1) != 0:
            raise RuntimeError(f"Could not write {path!r}: {result.get('output', '')}")

    def read_bytes(self, path: str) -> bytes:
        """Read bytes by base64-encoding the file through the shell."""
        result = self._environment.execute({"command": f"base64 -w0 {shlex.quote(path)}"}, cwd="/")
        if result.get("returncode", -1) != 0:
            raise RuntimeError(f"Could not read {path!r}: {result.get('output', '')}")
        return base64.b64decode(result.get("output", ""))

    def close(self) -> None:
        """Tear down the underlying Docker environment."""
        self._environment.cleanup()


# --- Main grader entry point ---


def grade_fresh_container(
    swe_problem: dict[str, Any],
    model_patch: str,
    grader_kind: str,
    image_name: str,
    timeout: int = _DEFAULT_EVAL_TIMEOUT,
    swe_task_id: str = "",
    memory: str | int | None = "",
    sandbox: SyncSandboxManager | None = None,
    sandbox_spec: SandboxSpec | None = None,
    sandbox_snapshot: SnapshotRef | None = None,
    grading_plan: GradingPlan | None = None,
) -> dict[str, Any]:
    """
    Grade a model patch in a fresh sandbox.

    1. Runs ``analyze_patch_policy`` and returns immediately if it is violated.
    2. Builds the :class:`GradingPlan` from the prepared row (host-independent).
    3. Spawns a fresh sandbox from the per-problem image.
    4. For SWE-smith, runs ``git checkout HEAD~1`` to restore F2P test files.
    5. Resets the tree and applies the model patch via ``git apply``.
    6. Runs the eval script and grades the log inside the sandbox.

    Args:
        swe_problem (dict[str, Any]): Full dataset row for one SWE problem.
        model_patch (str): Patch submitted by the agent (unified diff).
        grader_kind (str): ``"verified"``, ``"gym"`` or ``"smith"``.
        image_name (str): Docker image used for the rollout.
        timeout (int): Eval script execution timeout in seconds.
        swe_task_id (str): PSRL rollout episode ID for container labelling.
        memory (str): ``--memory`` limit for the grading container.
        sandbox: Generic synchronous PSRL sandbox manager used by training.
        sandbox_spec: Backend-neutral grading sandbox request used by training.
        sandbox_snapshot: Compatible clean baseline restored by stateful backends.
        grading_plan: Pre-built plan. When omitted it is derived from
            ``swe_problem`` (``eval_script`` + ``log_parser``).

    Returns:
        dict[str, Any]: Grading result. Infrastructure failures add
        ``failure_reason`` / ``parser_error`` diagnostics, and reward semantics
        are unchanged (callers keep their existing ``reward_mode`` handling).
    """
    swe_problem_id: str = swe_problem.get("instance_id", "unknown")
    log_prefix = f"[swebench_grader, task_id={swe_task_id or swe_problem_id}]"
    t0 = time.monotonic()

    # --- 0. Patch policy guard ---
    if not model_patch:
        psrl_logger.info(f"{log_prefix} No patch submitted, skipping grading.")
        return {
            "policy_violated": False,
            "policy_reasons": [],
            "resolved": False,
            "apply_ok": False,
            "f2p_pass": 0,
            "f2p_total": len(swe_problem.get("FAIL_TO_PASS", [])),
            "p2p_pass": 0,
            "p2p_total": len(swe_problem.get("PASS_TO_PASS", [])),
            "timeout": False,
            "error": "no_patch",
            "parser_error": None,
            "failure_reason": "no_patch",
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
            "f2p_total": len(swe_problem.get("FAIL_TO_PASS", [])),
            "p2p_pass": 0,
            "p2p_total": len(swe_problem.get("PASS_TO_PASS", [])),
            "timeout": False,
            "error": None,
            "parser_error": None,
            "failure_reason": "policy_blocked",
            "elapsed_s": time.monotonic() - t0,
            "output_tail": "",
            "resolved_by": "policy_blocked",
        }

    # --- 1. Resolve the frozen grading plan (host-independent) ---
    try:
        plan = grading_plan or GradingPlan.from_swe_problem(swe_problem)
    except (ValueError, TypeError) as exc:
        result = _grader_failure(swe_problem, str(exc))
        result.update(failure_reason="invalid_plan", resolved_by="invalid_plan", elapsed_s=time.monotonic() - t0)
        return result
    if plan is None:
        psrl_logger.error(
            f"{log_prefix} No eval_script on the prepared row. Re-run the dataset preparation step for this split."
        )
        result = _grader_failure(swe_problem, "missing_eval_script")
        result["failure_reason"] = "missing_eval_script"
        result["resolved_by"] = "missing_eval_script"
        result["elapsed_s"] = time.monotonic() - t0
        return result

    # --- 2. Spawn fresh eval container ---
    grader_label = (
        f"psrl.grader_task_id={swe_task_id}__eval" if swe_task_id else f"psrl.grader_task_id={swe_problem_id}__eval"
    )
    run_args = list(_BASE_RUN_ARGS)
    if memory:
        run_args = [a for a in run_args if not a.startswith("--memory=")]
        run_args.append(f"--memory={memory}")
    run_args += ["--label", grader_label]
    # Per-actor label consumed by the reaper sidecar in
    # psrl.sandbox.utils.docker_utils.
    _actor_id = os.environ.get("PSRL_ACTOR_ID", "")
    if _actor_id:
        run_args += ["--label", f"psrl.actor_id={_actor_id}"]

    sandbox_session: SyncSandboxSession | _DockerGradingSession | None = None
    docker_environment: Any | None = None
    uses_psrl_sandbox = sandbox is not None
    apply_ok = False
    verdict: dict[str, Any] = {}

    def execute(command: str, *, cwd: str, command_timeout: int | None = None) -> ExecResult:
        """Execute a shell command through whichever session is active."""
        if sandbox_session is None:
            raise RuntimeError("Grading sandbox is not initialized.")
        return sandbox_session.exec(command, cwd=cwd, timeout_s=command_timeout)

    try:
        psrl_logger.info(f"{log_prefix} Spawning eval container: image={image_name!r}, grader_kind={grader_kind!r}.")
        if sandbox is not None:
            if sandbox_spec is None:
                raise ValueError("sandbox_spec is required when sandbox is provided.")
            if sandbox_snapshot is not None:
                sandbox_session = sandbox.restore(
                    sandbox_snapshot, sandbox_spec, state_policy=sandbox_spec.state_policy
                )
            else:
                sandbox_session = sandbox.create(sandbox_spec)
            sandbox_id = sandbox_session.ref.sandbox_id
        else:
            from minisweagent.environments.docker import DockerEnvironment

            docker_environment = DockerEnvironment(
                image=image_name,
                cwd="/testbed",
                run_args=run_args,
                forward_env=_PROXY_ENV_KEYS,
                container_timeout=_DEFAULT_CONTAINER_TIMEOUT,
            )
            sandbox_session = _DockerGradingSession(docker_environment)
            sandbox_id = docker_environment.container_id
        psrl_logger.info(f"{log_prefix} Eval container started: id={sandbox_id!r}.")

        # --- 3. SWE-smith: restore F2P test files via HEAD~1 ---
        if grader_kind == "smith":
            out = execute("git checkout HEAD~1", cwd="/testbed")
            if out.exit_code != 0:
                raise RuntimeError(f"Could not restore SWE-smith baseline: {out.stdout[:200]!r}.")

        # --- 4. Reset tree and apply model patch ---
        # Reset to the dataset baseline so agent-created commits stay part of the patch.
        base_commit = str(swe_problem.get("base_commit") or "")
        reset_target = shlex.quote(base_commit) if base_commit else "HEAD"
        out = execute(f"git reset --hard {reset_target} && git clean -fd", cwd="/testbed")
        if out.exit_code != 0:
            raise RuntimeError(f"Could not reset grading baseline: {out.stdout[:200]!r}.")

        # PSRL's file API avoids heredoc delimiter collisions and shell expansion.
        write_bytes = sandbox_session.write_bytes
        write_bytes("/tmp/psrl-model.patch", model_patch.encode())
        out2 = execute("git apply --binary /tmp/psrl-model.patch", cwd="/testbed")
        apply_ok = out2.exit_code == 0
        if not apply_ok:
            psrl_logger.info(f"{log_prefix} git apply failed: {out2.stdout[:300]}.")

        if apply_ok:
            # --- 5. SWE-smith: revert test-file changes from the patch ---
            if grader_kind == "smith":
                eval_test_files = _extract_eval_test_files(swe_problem)
                if eval_test_files:
                    out3 = execute(f"git checkout -- {shlex.join(eval_test_files)}", cwd="/testbed")
                    if out3.exit_code != 0:
                        raise RuntimeError(f"Could not restore grading tests: {out3.stdout[:200]!r}.")

            # --- 6. Run + grade inside the sandbox ---
            # Skip the eval script's editable re-install unless the patch changed that wheel.
            skip_editable_install = not _patch_needs_editable_install(model_patch)
            psrl_logger.info(f"{log_prefix} Running eval script (timeout={timeout}s)...")
            verdict = run_grading(
                sandbox_session,
                plan,
                workdir="/testbed",
                timeout_s=timeout,
                skip_editable_install=skip_editable_install,
            )
        else:
            verdict = GradingResult(
                f2p_total=len(plan.f2p),
                p2p_total=len(plan.p2p),
                failure_reason="apply_failed",
                output_tail=out2.stdout[-_OUTPUT_TAIL_BYTES:],
            ).to_dict()

    except Exception as exc:
        psrl_logger.error(f"{log_prefix} Container error: {exc}.")
        verdict = GradingResult(
            f2p_total=len(plan.f2p),
            p2p_total=len(plan.p2p),
            failure_reason="eval_timeout" if isinstance(exc, TimeoutError) else "container_error",
            timeout=isinstance(exc, TimeoutError),
            error=str(exc),
        ).to_dict()

    finally:
        if sandbox_session is not None:
            try:
                sandbox_session.close()
            except Exception as cleanup_exc:
                psrl_logger.warning(f"{log_prefix} Sandbox cleanup failed: {cleanup_exc}.")
        # Synchronous belt-and-suspenders sweep by label for the standalone path, since
        # ``docker_env.cleanup`` has been observed to silently succeed without killing anything.
        if not uses_psrl_sandbox:
            try:
                from psrl.sandbox.utils.docker_utils import force_remove_containers_by_label

                force_remove_containers_by_label("psrl.grader_task_id", grader_label.split("=", 1)[1])
            except Exception as sweep_exc:
                psrl_logger.warning(f"{log_prefix} Label sweep failed: {sweep_exc}.")

    result: dict[str, Any] = {
        "policy_violated": False,
        "policy_reasons": [],
        "apply_ok": apply_ok,
        "resolved_by": "harness",
        **verdict,
    }
    result["resolved_by"] = verdict.get("failure_reason") or "harness"
    result["elapsed_s"] = round(time.monotonic() - t0, 2)

    psrl_logger.info(
        f"{log_prefix} Grading complete: resolved={result['resolved']}, "
        f"apply_ok={apply_ok}, f2p={result['f2p_pass']}/{result['f2p_total']}, "
        f"p2p={result['p2p_pass']}/{result['p2p_total']}, "
        f"elapsed={result['elapsed_s']:.1f}s, resolved_by={result['resolved_by']!r}."
    )
    return result
