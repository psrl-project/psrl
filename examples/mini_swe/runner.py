"""Run one mini-SWE-agent task through its standard Python bindings."""

import functools
import hashlib
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any

from examples.mini_swe.grading.schema import GradingPlan
from examples.mini_swe.harness_adapter import MiniSWEAgentAdapter, MiniSWEAgentConfig, RunnerCancelled
from psrl.sandbox import (
    MountSpec,
    ResourceSpec,
    SandboxFeature,
    SandboxSource,
    SandboxSpec,
    SandboxStatePolicy,
    SnapshotKind,
    SnapshotRef,
    SyncSandboxManager,
)
from psrl.sandbox.capacity import parse_memory_mb

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

psrl_logger = logging.getLogger(__name__)
logging.getLogger("minisweagent.environment").setLevel(logging.WARNING)


def _silence_litellm() -> None:
    """Suppress LiteLLM's BadRequestError banner (print, not logging).

    LiteLLM's exception mapper always ``print``s
    ``Give Feedback / Get Help`` + ``LiteLLM.Info: ...`` unless
    ``litellm.suppress_debug_info`` is True. Logger level tweaks do nothing.
    """
    try:
        import litellm

        litellm.suppress_debug_info = True
    except Exception:
        pass
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("litellm").setLevel(logging.ERROR)


_PROXY_ENV_KEYS = [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
]


def parse_duration_seconds(value: str | int | float | None) -> float | None:
    """Parse a compact MiniSWE duration into seconds."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([hms]?)", value.strip().lower())
    if match is None:
        raise ValueError(f"Invalid sandbox duration: {value!r}.")
    return float(match.group(1)) * {"h": 3600, "m": 60, "s": 1, "": 1}[match.group(2)]


def _resolve_harness_sandbox_image(configured_image: str) -> str:
    """Prefer the pre-baked per-image derivative when present.

    Each SWE task uses its own per-problem base image (from the parquet's
    ``sandbox_overrides.environment.image``), so a single global baked image is
    meaningless here. ``bake_harness_image.sh`` derives one image per base:
    ``psrl/swebench-harness:<sha12(base)>``. The derivative purges leaked git
    metadata, so a sandbox that starts from it already passes the runtime
    git-leak probe. When the derivative is absent we fall back to the original
    image plus the runtime git sanitization, which is correct but slower. A missing bake
    must never block the run.

    The tag keys on the base image alone, so changing the bake steps does not
    invalidate an existing derivative. Re-run ``rebake_harness_image.sh`` after
    such a change to replace the stale image.

    The harness executable itself is never baked into the image: it comes from
    the read-only runtime tree mounted by the harness loop.
    """
    baked = f"psrl/swebench-harness:{_image_digest(configured_image)}"
    if _docker_image_exists(baked):
        return baked
    return configured_image


def _image_digest(reference: str) -> str:
    """Short deterministic digest of an image reference (matches the baker)."""
    return hashlib.sha256(reference.encode()).hexdigest()[:12]


@functools.lru_cache(maxsize=512)
def _docker_image_exists(reference: str) -> bool:
    """Whether ``reference`` is present in the local Docker daemon."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", reference],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        return False
    return result.returncode == 0


def resolve_container_config(payload: dict[str, Any], *, grading: bool = False) -> dict[str, Any]:
    """Merge the shared container config with rollout- or grader-specific overrides."""
    sandbox_config = payload["runtime_config"]["sandbox_config"]
    settings = dict(sandbox_config["environment"])
    overrides = dict(sandbox_config.get("grader_environment" if grading else "rollout_environment", {}))
    if "env" in overrides:
        settings["env"] = {**dict(settings.get("env", {})), **dict(overrides.pop("env"))}
    settings.update(overrides)

    if not grading:
        observation = payload["observation"]
        if observation.get("use_preexisting_repo", True) and observation.get("preexisting_repo_name"):
            settings["cwd"] = f"/{observation['preexisting_repo_name']}"

    configured_forward_env = settings.get("forward_env", [])
    settings["forward_env"] = list(dict.fromkeys([*configured_forward_env, *_PROXY_ENV_KEYS]))
    return settings


def build_sandbox_spec(
    payload: dict[str, Any],
    grading: bool = False,
    image: str | None = None,
    template: str | None = None,
) -> SandboxSpec:
    """Map MiniSWE task data onto the backend-neutral creation contract."""
    container_config = resolve_container_config(payload, grading=grading)
    observation = payload["observation"]
    cwd = str(container_config.get("cwd", "/testbed"))
    mounts: list[MountSpec] = []
    if not grading and not observation.get("use_preexisting_repo", True) and observation.get("repo_path"):
        mounts.append(MountSpec(str(observation["repo_path"]), "/testbed"))
        cwd = "/testbed"

    task_id = str(observation.get("swe_task_id", ""))
    metadata = {"psrl.swe_task_id": task_id}
    if grading:
        metadata["psrl.grader_task_id"] = f"{task_id}__eval"
    forward_env = container_config.get("forward_env", _PROXY_ENV_KEYS)
    sandbox_env = dict(container_config.get("env", {}))
    sandbox_env.update({key: os.environ[key] for key in forward_env if key in os.environ})

    sandbox_config = payload["runtime_config"]["sandbox_config"]
    selected_template = template or (container_config.get("template") if image is None else None)
    # Prefer the per-image baked derivative (git-purged, see bake_harness_image.sh),
    # falling back to the configured image. The grader keeps its explicit problem image.
    resolved_image = image or _resolve_harness_sandbox_image(str(container_config["image"]))
    source = (
        SandboxSource.template(str(selected_template)) if selected_template else SandboxSource.image(resolved_image)
    )
    sandbox_prefix = str(payload.get("sandbox_prefix", task_id))
    sandbox_cpu = sandbox_config.get("sandbox_cpu_count") or 0
    return SandboxSpec(
        source=source,
        resources=ResourceSpec(
            cpu_count=float(sandbox_cpu) if sandbox_cpu else None,
            memory_mb=parse_memory_mb(container_config.get("memory")),
        ),
        workdir=cwd,
        metadata=metadata,
        env=sandbox_env,
        mounts=tuple(mounts),
        policy_profile=sandbox_config.get("policy_profile"),
        idle_timeout_s=parse_duration_seconds(container_config.get("container_timeout")),
        idempotency_key=f"{sandbox_prefix}:{'grader' if grading else 'rollout'}",
        state_policy=SandboxStatePolicy(enabled=bool(sandbox_config.get("snapshot_verifier", True))),
    )


def create_harness(
    payload: dict[str, Any],
    sandbox: SyncSandboxManager,
    spec: SandboxSpec | None = None,
    cancel_event: threading.Event | None = None,
) -> MiniSWEAgentAdapter:
    """Create the rollout sandbox and MiniSWE's third-party protocol adapter."""
    container_config = resolve_container_config(payload)
    spec = spec or build_sandbox_spec(payload)
    session = sandbox.create(spec)
    config = MiniSWEAgentConfig(
        image=str(container_config["image"]),
        cwd=spec.workdir or "/",
        env=dict(container_config.get("env", {})),
        forward_env=list(container_config.get("forward_env", _PROXY_ENV_KEYS)),
        timeout=int(container_config.get("timeout", 30)),
    )
    return MiniSWEAgentAdapter(session, config, cancel_event=cancel_event)


def _build_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(payload["runtime_config"]["model"])
    model_kwargs = dict(payload["sampling_params"])
    top_k = model_kwargs.pop("top_k", None)
    if top_k is not None and int(top_k) >= 0:
        model_kwargs["extra_body"] = {"top_k": int(top_k)}
    extra_headers = dict(model_kwargs.get("extra_headers") or {})
    if payload.get("trajectory_id_strategy", "manual") == "manual":
        extra_headers["x-smg-tito-trajectory-id"] = "0"
    else:
        extra_headers.pop("x-smg-tito-trajectory-id", None)
    if extra_headers:
        model_kwargs["extra_headers"] = extra_headers
    else:
        model_kwargs.pop("extra_headers", None)
    model_kwargs.update(
        {
            "api_base": payload["base_url"],
            "api_key": "EMPTY",
            "timeout": payload["runtime_config"]["sandbox_config"]["rollout_turn_timeout"],
        }
    )
    model_config.update(
        {
            "model_name": payload["model"],
            "model_kwargs": model_kwargs,
            "cost_tracking": "ignore_errors",
        }
    )
    return model_config


def _grader_failure(swe_problem: dict[str, Any], error: str) -> dict[str, Any]:
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
        "elapsed_s": 0.0,
        "output_tail": "",
        "resolved_by": "grader_error",
    }


def _snapshot_matches_spec(snapshot: SnapshotRef, spec: SandboxSpec) -> bool:
    """Return whether ``snapshot`` can seed the grading sandbox ``spec``.

    A ``FILESYSTEM`` snapshot (docker commit) is self-contained: ``restore``
    recreates the container from the committed image using the grader's own
    source reference and resource flags, so neither has to match. A
    ``FULL_STATE`` snapshot must be rebuilt with the exact same source and
    resources, so those are compared.
    """
    if snapshot.kind == SnapshotKind.FILESYSTEM:
        return True
    metadata = snapshot.metadata
    return (
        metadata.get("psrl.source_kind") == spec.source.kind.value
        and metadata.get("psrl.source_reference") == spec.source.reference
        and metadata.get("psrl.cpu_count") == spec.resources.cpu_count
        and metadata.get("psrl.memory_mb") == spec.resources.memory_mb
        and metadata.get("psrl.disk_mb") == spec.resources.disk_mb
    )


def _snapshot_compatible(left: SandboxSpec, right: SandboxSpec) -> bool:
    """Return whether one full-state snapshot can satisfy both specs."""
    return left.source == right.source and left.resources == right.resources


def build_grader_spec(payload: dict[str, Any]) -> SandboxSpec | None:
    """Build the verifier spec when this task uses the fresh-container grader."""
    observation = payload["observation"]
    swe_image = str(observation.get("swe_problem_image", "") or "")
    if observation.get("swe_grader") != "swebench_fresh_container" or not swe_image:
        return None
    grader_template = observation.get("swe_problem_template") or resolve_container_config(
        payload,
        grading=True,
    ).get("template")
    return build_sandbox_spec(
        payload,
        grading=True,
        image=None if grader_template else swe_image,
        template=str(grader_template) if grader_template else None,
    )


def _capture_resource_metrics(
    harness: MiniSWEAgentAdapter,
    timing: dict[str, float],
) -> bool:
    """Sample optional resource metrics without changing the rollout result."""
    try:
        usage = harness.session.stats()
    except Exception:
        psrl_logger.warning("Could not collect MiniSWE sandbox resource metrics.", exc_info=True)
        return False
    timing["sandbox_memory_mib"] = usage.memory_bytes / (1024 * 1024)
    timing["sandbox_peak_memory_mib"] = usage.peak_memory_bytes / (1024 * 1024)
    timing["sandbox_cpu_total_s"] = usage.cpu_total_ns / 1_000_000_000
    return True


def grade_patch(
    payload: dict[str, Any],
    patch: str,
    sandbox: SyncSandboxManager,
    baseline_snapshot: SnapshotRef | None = None,
    grader_spec: SandboxSpec | None = None,
) -> dict[str, Any] | None:
    observation = payload["observation"]
    if not patch or observation.get("swe_grader") != "swebench_fresh_container":
        return None

    swe_problem = observation.get("swe_problem", {})
    swe_image = str(observation.get("swe_problem_image", "") or "")
    if not swe_problem or not swe_image:
        return _grader_failure(swe_problem, "missing_grader_input")

    try:
        from examples.mini_swe.swebench_grader import grade_fresh_container

        # ``grader_kind`` only selects the SWE-smith pre/post steps. Grading itself is
        # driven entirely by the frozen eval script and the vendored parser.
        grader_kind = "smith" if observation.get("swe_restore_tests", False) else "verified"
        grader_config = resolve_container_config(payload, grading=True)
        grader_spec = grader_spec or build_grader_spec(payload)
        if grader_spec is None:
            return _grader_failure(swe_problem, "missing_grader_spec")
        verifier_snapshot = (
            baseline_snapshot
            if baseline_snapshot is not None and _snapshot_matches_spec(baseline_snapshot, grader_spec)
            else None
        )
        return grade_fresh_container(
            swe_problem,
            patch,
            grader_kind,
            swe_image,
            int(grader_config.get("timeout", 900)),
            observation.get("swe_task_id", ""),
            grader_config.get("memory", ""),
            sandbox=sandbox,
            sandbox_spec=grader_spec,
            sandbox_snapshot=verifier_snapshot,
            grading_plan=GradingPlan.from_swe_problem(swe_problem),
        )
    except Exception as exc:
        return _grader_failure(swe_problem, str(exc))


def run_agent(
    payload: dict[str, Any],
    sandbox: SyncSandboxManager,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Run one task through mini-SWE-agent's standard Python bindings."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from psrl.utils.agent.overflow import PromptOverflowError, ensure_overflow_handling

    _silence_litellm()

    class _TimedAgent(DefaultAgent):
        """DefaultAgent that accumulates wall-clock model vs environment time.

        ``step()`` decomposes into ``query()`` (model/assistant turn) and
        ``execute_actions()`` (environment/tool execution), so timing each gives a
        clean assistant-vs-env split for the trajectory summary.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.assistant_s = 0.0
            self.env_s = 0.0

        def query(self, *args: Any, **kwargs: Any) -> Any:
            t = time.perf_counter()
            try:
                return super().query(*args, **kwargs)
            finally:
                self.assistant_s += time.perf_counter() - t

        def execute_actions(self, *args: Any, **kwargs: Any) -> Any:
            t = time.perf_counter()
            try:
                return super().execute_actions(*args, **kwargs)
            finally:
                self.env_s += time.perf_counter() - t

    harness = None
    baseline_snapshot = None
    grader_spec = None
    agent = None
    run_start = time.perf_counter()
    # `timing` is mutated by the finally block and shared by every return dict, so
    # all paths capture elapsed values, including mid-run exceptions.
    timing: dict[str, float] = {"prep_s": 0.0, "assistant_s": 0.0, "env_s": 0.0, "grading_s": 0.0, "elapsed_s": 0.0}
    resource_metrics_captured = False
    try:
        sandbox_create_start = time.perf_counter()
        rollout_spec = build_sandbox_spec(payload)
        harness = create_harness(payload, sandbox, rollout_spec, cancel_event)
        timing["sandbox_create_s"] = time.perf_counter() - sandbox_create_start
        sandbox_config = payload["runtime_config"]["sandbox_config"]
        grader_spec = build_grader_spec(payload)
        if (
            sandbox_config.get("snapshot_verifier", True)
            and grader_spec is not None
            and _snapshot_compatible(rollout_spec, grader_spec)
            and harness.session.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
            and harness.session.capabilities.supports(SandboxFeature.RESTORE)
        ):
            try:
                baseline_snapshot = harness.session.snapshot(state_policy=rollout_spec.state_policy)
            except Exception:
                psrl_logger.warning(
                    "Could not create a safe MiniSWE verifier snapshot. Recreating the rollout sandbox.",
                    exc_info=True,
                )
                harness.cleanup()
                recreate_started = time.perf_counter()
                harness = create_harness(payload, sandbox, rollout_spec, cancel_event)
                timing["sandbox_create_s"] += time.perf_counter() - recreate_started
        model = get_model(config=_build_model_config(payload))
        ensure_overflow_handling(model)
        agent_config = dict(payload["runtime_config"]["agent"])
        agent_config["instance_template"] = agent_config.pop("problem_template")
        agent_config["step_limit"] = payload["max_turns"]
        agent_config["output_path"] = None
        timing["prep_s"] = time.perf_counter() - run_start
        agent = _TimedAgent(model, harness, **agent_config)
        result = agent.run(payload["task"])
        patch = result.get("submission", "") or ""
        if sandbox_config.get("collect_resource_metrics", False):
            resource_metrics_captured = _capture_resource_metrics(harness, timing)
        try:
            harness.cleanup()
            harness = None
        except Exception:
            psrl_logger.warning("MiniSWE harness cleanup before grading failed.", exc_info=True)
        grade_start = time.perf_counter()
        grader_result = grade_patch(payload, patch, sandbox, baseline_snapshot, grader_spec)
        timing["grading_s"] = time.perf_counter() - grade_start
        return {
            "exit_status": result.get("exit_status", ""),
            "submission": patch,
            "grader_result": grader_result,
            "timing": timing,
        }
    except RunnerCancelled:
        return {
            "exit_status": "cancelled",
            "submission": "",
            "grader_result": None,
            "timing": timing,
        }
    except PromptOverflowError as exc:
        # Preserve turns generated before context overflow and terminate normally.
        psrl_logger.warning("mini-SWE-agent stopped on context overflow: %s.", exc)
        return {
            "exit_status": "context_exceeded",
            "submission": "",
            "grader_result": None,
            "timing": timing,
        }
    except Exception as exc:
        psrl_logger.warning("mini-SWE-agent task failed: %s.", exc, exc_info=True)
        return {
            "exit_status": "error",
            "submission": "",
            "grader_result": None,
            "error": str(exc),
            "timing": timing,
        }
    finally:
        # Shared return dictionaries receive final timing before control leaves.
        if agent is not None:
            timing["assistant_s"] = agent.assistant_s
            timing["env_s"] = agent.env_s
        timing["elapsed_s"] = time.perf_counter() - run_start
        if harness is not None:
            if not resource_metrics_captured and payload["runtime_config"]["sandbox_config"].get(
                "collect_resource_metrics", False
            ):
                _capture_resource_metrics(harness, timing)
            try:
                harness.cleanup()
            except Exception:
                psrl_logger.warning("MiniSWE harness cleanup failed.", exc_info=True)
        if baseline_snapshot is not None:
            try:
                sandbox.delete_snapshot(baseline_snapshot)
            except Exception:
                psrl_logger.warning("MiniSWE verifier snapshot cleanup failed.", exc_info=True)
