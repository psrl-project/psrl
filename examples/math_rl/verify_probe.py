"""Preflight for the external verifier.

Checks the deployment the recipe will use: the served version, the requested
model, the network policy, and two graded round trips through the real client.
Run it before a training run. A non-zero exit means do not start.

The graded cases are synthetic, so no dataset problem leaves the machine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import aiohttp
from examples.math_rl.verifier import (
    MAX_REQUEST_BYTES,
    VerifierConfig,
    VerifierError,
    evaluate,
    reset_verifier_clients,
)

_CORRECT = (
    "Claim: for every integer n, n + 0 = n. Proof: by the definition of addition on the "
    "integers, adding the additive identity zero leaves the value unchanged. This is the "
    "identity axiom of the additive group, so the claim holds."
)
_BROKEN = (
    "Claim: for every integer n, n + 0 = n + 1. Proof: subtract n from both sides, which "
    "leaves 0 = 1, and the conclusion follows."
)


def _default_proxy() -> str:
    """The proxy the operator's shell already uses for outbound HTTP."""
    for name in ("MATH_RL_VERIFIER_PROXY", "https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _health_url(endpoint: str) -> str:
    trimmed = endpoint.rstrip("/")
    for suffix in ("/verify/stream", "/verify"):
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)] + "/health"
    return trimmed + "/health"


async def _check_health(
    session: aiohttp.ClientSession, endpoint: str, version: str, model: str, proxy: str
) -> dict:
    url = _health_url(endpoint)
    async with session.get(
        url, timeout=aiohttp.ClientTimeout(total=20), proxy=proxy or None
    ) as response:
        if response.status != 200:
            raise SystemExit(f"The verifier health check failed at {url} with HTTP {response.status}.")
        health = await response.json()
    served = health.get("verifier_harness_version") or health.get("version") or ""
    if served != version:
        raise SystemExit(f"The verifier serves {served!r}, but the recipe asks for {version!r}.")
    models = health.get("models") or {}
    if models and model not in models:
        raise SystemExit(f"The verifier does not serve model {model!r}. Configured: {sorted(models)}.")
    if models and not models[model].get("configured", False):
        raise SystemExit(f"The verifier reports model {model!r} as not configured.")
    print(
        f"health: version {served}, model {model}, "
        f"web_search {health.get('web_search')}, network_access {health.get('network_access')}, "
        f"deployment {health.get('deployment_revision')}"
    )
    return health


async def _grade(config: VerifierConfig, label: str, problem: str, solution: str) -> float:
    try:
        verdict = await evaluate(config, problem, solution)
    except VerifierError as error:
        raise SystemExit(f"The {label} case produced no verdict: {error!r}") from error
    print(
        f"{label}: verdict {verdict.verdict}, score {verdict.score(config):.2f}, "
        f"run_id {verdict.run_id or 'unknown'}, duration {verdict.duration_s}, "
        f"usage {json.dumps(dict(verdict.usage), sort_keys=True)}"
    )
    if not verdict.run_id:
        print(f"warning: the {label} case returned no run id", file=sys.stderr)
    return verdict.score(config)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Check a verifier deployment before a training run.")
    parser.add_argument("--endpoint", default=None, help="Defaults to MATH_RL_VERIFIER_URL.")
    parser.add_argument("--version", default=None, help="Defaults to MATH_RL_VERIFIER_VERSION.")
    parser.add_argument("--model", default=None, help="Defaults to MATH_RL_VERIFIER_MODEL.")
    parser.add_argument("--effort", default=None, help="Defaults to MATH_RL_VERIFIER_EFFORT.")
    parser.add_argument("--stage-timeout", type=float, default=None)
    parser.add_argument("--transport", default=None, choices=("stream", "blocking"))
    parser.add_argument("--proxy", default=None, help="Defaults to the shell HTTP proxy.")
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--skip-cases", action="store_true")
    args = parser.parse_args()

    config = VerifierConfig(
        endpoint=args.endpoint or os.environ.get("MATH_RL_VERIFIER_URL", ""),
        verifier_harness_version=args.version or os.environ.get("MATH_RL_VERIFIER_VERSION", ""),
        model=args.model or os.environ.get("MATH_RL_VERIFIER_MODEL", ""),
        profile=os.environ.get("MATH_RL_VERIFIER_PROFILE", ""),
        reasoning_effort=args.effort or os.environ.get("MATH_RL_VERIFIER_EFFORT", ""),
        proxy=args.proxy or _default_proxy(),
        transport=args.transport or os.environ.get("MATH_RL_VERIFIER_TRANSPORT", "stream"),
        stage_timeout_s=args.stage_timeout or float(os.environ.get("MATH_RL_VERIFIER_STAGE_TIMEOUT_S", "900")),
        max_retries=int(os.environ.get("MATH_RL_VERIFIER_MAX_RETRIES", "1")),
        max_concurrency=int(os.environ.get("MATH_RL_VERIFIER_MAX_CONCURRENCY", "16")),
    )
    print(
        f"endpoint {config.endpoint}, version {config.verifier_harness_version}, "
        f"model {config.model or config.profile}, transport {config.transport}, "
        f"stage timeout {config.stage_timeout_s:.0f}s, body limit {MAX_REQUEST_BYTES} bytes"
    )

    try:
        if not args.skip_health:
            async with aiohttp.ClientSession() as session:
                await _check_health(
                    session, config.endpoint, config.verifier_harness_version, config.model, config.proxy
                )
        if not args.skip_cases:
            good = await _grade(config, "correct", "Prove that n + 0 = n for every integer n.", _CORRECT)
            bad = await _grade(config, "broken", "Prove that n + 0 = n for every integer n.", _BROKEN)
            if bad > good:
                raise SystemExit(f"The broken case scored {bad:.2f}, above the correct case at {good:.2f}.")
    finally:
        await reset_verifier_clients()
    print("preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
