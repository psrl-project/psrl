"""
Prepare-time generation of SWE-smith grading fields.

Only SWE-smith needs dataset-side work: unlike Verified and SWE-Gym, its rows
carry neither an `eval_script` nor a parser name, because the official harness
builds `/eval.sh` from the repo profile at evaluation time. We freeze that
script (with the official sentinels) and the flattened parser name once here, so
the training host never needs `swesmith` at rollout time.

This module is the only place (besides `vendor_parsers.py`) allowed to
import the upstream grading packages, and it runs offline.
"""

from __future__ import annotations

import inspect
import shlex

from .parsers import parser_name_for_repo, resolve_parser, smith_parser_name
from .schema import GradingPlan, normalize_test_ids


def smith_log_parser_name(swe_problem: dict) -> str:
    """
    Return the registry name of the SWE-smith profile parser for this row.
    """
    from swesmith.profiles import registry

    profile = registry.get_from_inst(swe_problem)
    return smith_parser_name(inspect.getsource(type(profile).log_parser))


def generate_smith_eval_script(swe_problem: dict, *, f2p_only: bool = True, workdir: str = "/testbed") -> str:
    """
    Build the official-shaped SWE-smith eval script (with test sentinels).
    """
    from swesmith.constants import TEST_OUTPUT_END, TEST_OUTPUT_START
    from swesmith.profiles import registry

    profile = registry.get_from_inst(swe_problem)
    test_command, _ = profile.get_test_cmd(swe_problem, f2p_only=f2p_only)
    return (
        "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                f"cd {shlex.quote(workdir)} || exit 1",
                f"printf '%s\\n' {shlex.quote(TEST_OUTPUT_START)}",
                test_command,
                f"printf '%s\\n' {shlex.quote(TEST_OUTPUT_END)}",
            ]
        )
        + "\n"
    )


def freeze_smith_grading(swe_problem: dict) -> dict[str, str]:
    """
    Return the `eval_script` / `log_parser` fields for one SWE-smith row.
    """
    log_parser = smith_log_parser_name(swe_problem)
    resolve_parser(log_parser)
    return {
        "eval_script": generate_smith_eval_script(
            swe_problem,
            f2p_only=not bool(normalize_test_ids(swe_problem.get("PASS_TO_PASS"))),
        ),
        "log_parser": log_parser,
    }


def validate_prepared_problem(problem: dict, *, default_parser: str | None = None) -> None:
    """
    Pin the dataset's parser and reject malformed scoring metadata in preparation.
    """
    if not problem.get("log_parser"):
        problem["log_parser"] = default_parser or parser_name_for_repo(problem.get("repo", ""))
    plan = GradingPlan.from_swe_problem(problem)
    if plan is None:
        raise ValueError(f"Task {problem.get('instance_id')!r} is missing eval_script. Reprepare the dataset.")
    problem["FAIL_TO_PASS"] = list(plan.f2p)
    problem["PASS_TO_PASS"] = list(plan.p2p)
    problem["eval_type"] = plan.eval_type
