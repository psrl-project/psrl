"""
The per-task grading plan consumed by the in-sandbox driver.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ._vendor._constants import END_TEST_OUTPUT, START_TEST_OUTPUT, EvalType


def normalize_test_ids(value: Any) -> tuple[str, ...]:
    """
    Normalize a FAIL_TO_PASS / PASS_TO_PASS field to a tuple of test ids.

    Handles the several shapes the field takes in prepared parquet: a JSON
    string (SWE-bench rows sometimes encode it that way), a Python list, or a
    numpy array materialized by pandas/pyarrow.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Test expectations must contain a JSON array of test identifiers.") from exc
    if isinstance(value, (str, bytes, dict)):
        raise ValueError("Test expectations must be a sequence of test identifiers.")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError("Test expectations must be a sequence of test identifiers.") from exc
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError("Test identifiers must be nonempty strings.")
    return tuple(dict.fromkeys(items))


@dataclass(frozen=True)
class GradingPlan:
    """
    Everything the driver needs to grade one submitted patch.
    """

    instance_id: str
    repo: str
    eval_script: str
    parser_name: str
    markers: bool
    f2p: tuple[str, ...]
    p2p: tuple[str, ...]
    eval_type: str = EvalType.PASS_AND_FAIL.value

    def __post_init__(self) -> None:
        """
        Validate the scoring contract before creating a grading sandbox.
        """
        from .parsers import resolve_parser

        object.__setattr__(self, "eval_type", EvalType(self.eval_type).value)
        object.__setattr__(self, "f2p", normalize_test_ids(self.f2p))
        object.__setattr__(self, "p2p", normalize_test_ids(self.p2p))
        if type(self.markers) is not bool:
            raise ValueError("Grading markers must be a boolean.")
        resolve_parser(self.parser_name)
        if not self.f2p and not self.p2p:
            raise ValueError("Grading requires at least one expected test.")
        if set(self.f2p) & set(self.p2p):
            raise ValueError("FAIL_TO_PASS and PASS_TO_PASS must not overlap.")

    @classmethod
    def from_swe_problem(cls, swe_problem: dict[str, Any] | None) -> GradingPlan | None:
        """
        Build a plan from a dataset row, or `None` when no eval script exists.

        A missing `eval_script` is a hard stop: without it the task was never
        prepared for host-independent grading and must not fall back to the
        training host's interpreter.
        """
        problem = swe_problem or {}
        eval_script = problem.get("eval_script") or ""
        if not isinstance(eval_script, str) or not eval_script.strip():
            return None
        eval_type = problem.get("eval_type") or EvalType.PASS_AND_FAIL.value
        # Verified rows ship the repo-specific parser name, SWE-smith rows get a
        # flattened name at prepare time.
        #
        # SWE-Gym rows carry neither and are all pytest-based, so default to the
        # pytest parser rather than the repo-keyed map, which would pick an unrelated parser.
        from .parsers import DEFAULT_PARSER

        parser_name = str(problem.get("log_parser") or "").strip() or DEFAULT_PARSER
        return cls(
            instance_id=str(problem.get("instance_id") or ""),
            repo=str(problem.get("repo") or ""),
            eval_script=eval_script,
            parser_name=parser_name,
            markers=START_TEST_OUTPUT in eval_script and END_TEST_OUTPUT in eval_script,
            f2p=normalize_test_ids(problem.get("FAIL_TO_PASS")),
            p2p=normalize_test_ids(problem.get("PASS_TO_PASS")),
            eval_type=EvalType(eval_type).value,
        )

    def render_eval_script(self, *, skip_editable_install: bool, probe_path: str) -> str:
        """
        Return the eval script, optionally skipping a redundant editable install.

        The frozen eval scripts re-run `python -m pip install -e .` before the
        tests. When the image already has the checkout installed editable (the
        normal SWE-bench/SWE-Gym case), that command only rebuilds the same
        editable wheel (measured ~20s on a small repo, minutes on large ones),
        so it is replaced by a probe-guarded no-op. The caller keeps the install
        for patches that touch packaging metadata by passing
        `skip_editable_install=False`.

        Only whole-line `python -m pip install -e .` commands are rewritten.
        Anything that also installs dependencies is left verbatim.
        """
        from .editable import render_eval_script

        return render_eval_script(
            self.eval_script,
            skip_editable_install=skip_editable_install,
            probe_path=probe_path,
        )

    def driver_input(self) -> dict[str, Any]:
        """
        Serialize the subset of the plan the driver reads from disk.
        """
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "log_parser": self.parser_name,
            "markers": self.markers,
            "f2p": list(self.f2p),
            "p2p": list(self.p2p),
            "eval_type": self.eval_type,
        }


@dataclass(frozen=True)
class GradingResult:
    """
    The scorecard shared by the sandbox driver and the reward adapter.
    """

    resolved: bool = False
    f2p_pass: int = 0
    f2p_total: int = 0
    p2p_pass: int = 0
    p2p_total: int = 0
    parser_error: str | None = None
    failure_reason: str | None = None
    timeout: bool = False
    error: str | None = None
    output_tail: str = ""
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        """
        Reject malformed counts and contradictory success signals.
        """
        if type(self.resolved) is not bool or type(self.timeout) is not bool:
            raise ValueError("Scorecard status fields must be booleans.")
        for passed, total in ((self.f2p_pass, self.f2p_total), (self.p2p_pass, self.p2p_total)):
            if type(passed) is not int or type(total) is not int or not 0 <= passed <= total:
                raise ValueError("Scorecard test counts must satisfy 0 <= passed <= total.")
        for value in (self.parser_error, self.failure_reason, self.error):
            if value is not None and not isinstance(value, str):
                raise ValueError("Scorecard errors must be strings or null.")
        if not isinstance(self.output_tail, str):
            raise ValueError("Scorecard output_tail must be a string.")
        if self.resolved and (
            self.timeout
            or self.failure_reason
            or self.parser_error
            or self.error
            or self.f2p_pass != self.f2p_total
            or self.p2p_pass != self.p2p_total
            or self.f2p_total + self.p2p_total == 0
        ):
            raise ValueError("Successful scorecard must contain only passing tests and no errors.")

    def to_dict(self) -> dict[str, Any]:
        """
        Preserve the reward adapter's existing dictionary interface.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Any) -> GradingResult:
        """
        Validate an untrusted JSON scorecard before consuming its reward.
        """
        if not isinstance(value, dict):
            raise ValueError("Scorecard must be a JSON object.")
        required = {"resolved", "f2p_pass", "f2p_total", "p2p_pass", "p2p_total"}
        if not required <= value.keys():
            raise ValueError("Scorecard is missing required result fields.")
        return cls(**value)
