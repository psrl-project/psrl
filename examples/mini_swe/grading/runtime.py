"""
Run a prepared grading plan through a synchronous sandbox session.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Protocol

from .payload import (
    DEFAULT_DRIVER_PYTHONS,
    DRIVER_ZIP_PATH,
    EDITABLE_PROBE_PATH,
    EVAL_SCRIPT_PATH,
    INPUT_PATH,
    SCORECARD_PATH,
    build_container_command,
    editable_probe_source,
    grader_zip_bytes,
)
from .schema import GradingPlan, GradingResult


class GradingSession(Protocol):
    """
    The small file and execution interface shared by PSRL and standalone Docker.
    """

    def exec(self, command: str, *, cwd: str, timeout_s: float) -> object: ...
    def write_bytes(self, path: str, data: bytes) -> None: ...
    def read_bytes(self, path: str) -> bytes: ...


def run_grading(
    session: GradingSession,
    plan: GradingPlan,
    *,
    workdir: str,
    timeout_s: int,
    skip_editable_install: bool = False,
    python_candidates: tuple[str, ...] = DEFAULT_DRIVER_PYTHONS,
) -> dict:
    """
    Execute tests and return the scorecard using the existing reward interface.

    Infrastructure failures preserve expected test counts and cannot resolve a
    task. The caller owns sandbox cleanup, including cancellation after timeout.
    """
    if timeout_s <= 0:
        raise ValueError("Grading timeout_s must be greater than zero.")
    started = time.perf_counter()
    expected = {"f2p_total": len(plan.f2p), "p2p_total": len(plan.p2p)}

    def failure(reason: str, exc: Exception) -> dict:
        return GradingResult(
            **expected,
            failure_reason=reason,
            error=str(exc),
            timeout=reason == "eval_timeout",
            elapsed_s=time.perf_counter() - started,
        ).to_dict()

    try:
        command = build_container_command(python_candidates=python_candidates)
        session.write_bytes(DRIVER_ZIP_PATH, grader_zip_bytes())
        session.write_bytes(INPUT_PATH, json.dumps(plan.driver_input()).encode())
        if skip_editable_install:
            session.write_bytes(EDITABLE_PROBE_PATH, editable_probe_source().encode())
        script = plan.render_eval_script(skip_editable_install=skip_editable_install, probe_path=EDITABLE_PROBE_PATH)
        session.write_bytes(EVAL_SCRIPT_PATH, script.encode())
        result = session.exec(command, cwd=workdir, timeout_s=timeout_s)
        if getattr(result, "exit_code", 0) != 0:
            raise RuntimeError(f"Grading driver failed: {getattr(result, 'stderr', '')!r}.")
    except TimeoutError as exc:
        return failure("eval_timeout", exc)
    except Exception as exc:
        return failure("container_error", exc)

    try:
        verdict = GradingResult.from_dict(json.loads(session.read_bytes(SCORECARD_PATH)))
        if verdict.failure_reason:
            verdict = replace(verdict, **expected)
        elif (verdict.f2p_total, verdict.p2p_total) != (len(plan.f2p), len(plan.p2p)):
            raise ValueError("Scorecard test counts do not match the grading plan.")
    except Exception as exc:
        return failure("grading_incomplete", exc)
    return replace(verdict, elapsed_s=time.perf_counter() - started).to_dict()
