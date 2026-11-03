"""A local stand-in for the external verifier, used by tests and dry runs.

It implements the blocking `/verify` contract: exactly four request fields, a
verdict in {0, 1, 6, 7}, and the receipt in response headers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Any

from aiohttp import web

VERSION_KEY = web.AppKey("version", str)
POLICY_KEY = web.AppKey("policy", str)

FIELDS = {"problem", "solution_to_verify", "model_config", "verifier_harness_version"}
MAX_BODY_BYTES = 16 * 1024 * 1024
VERDICTS = ("0", "1", "6", "7")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _verdict(policy: str, problem: str, solution: str) -> str:
    if policy == "always-correct":
        return "7"
    if policy == "always-wrong":
        return "0"
    digest = hashlib.sha256((problem + "\x00" + solution).encode("utf-8")).digest()
    return VERDICTS[digest[0] % len(VERDICTS)]


async def verify(request: web.Request) -> web.Response:
    """Grade one proof with the configured mock policy."""
    started = time.perf_counter()
    raw = await request.read()
    if len(raw) > MAX_BODY_BYTES:
        return web.json_response({"detail": "request too large"}, status=413)
    try:
        body: Any = json.loads(raw)
    except json.JSONDecodeError:
        return web.json_response({"detail": "invalid JSON"}, status=422)
    if not isinstance(body, dict) or set(body) != FIELDS:
        return web.json_response({"detail": "the request must carry exactly the four fields"}, status=422)
    problem = body["problem"]
    solution = body["solution_to_verify"]
    if not isinstance(problem, str) or not problem.strip():
        return web.json_response({"detail": "problem must be a non-empty string"}, status=422)
    if not isinstance(solution, str):
        return web.json_response({"detail": "solution_to_verify must be a string"}, status=422)
    config = body["model_config"]
    if not isinstance(config, dict):
        return web.json_response({"detail": "model_config must be an object"}, status=422)
    verdict = _verdict(request.app[POLICY_KEY], problem, solution)
    run_id = "mock_" + _sha256(problem + "\x00" + solution)[:12]
    usage = {"input_tokens": max(1, len(problem) // 4), "output_tokens": max(1, len(solution) // 4)}
    headers = {
        "x-verifier-receipt": run_id,
        "x-verifier-harness-sha256": _sha256(str(body["verifier_harness_version"])),
        "x-verifier-prompt-sha256": _sha256(problem),
        "x-verifier-skill-sha256": _sha256(json.dumps(config, sort_keys=True)),
        "x-verifier-duration-seconds": f"{time.perf_counter() - started:.3f}",
        "x-verifier-usage": json.dumps(usage, separators=(",", ":")),
    }
    return web.json_response(
        {"verdict": verdict, "reason": f"mock verdict {verdict} from policy {request.app[POLICY_KEY]}"},
        headers=headers,
    )


async def health(request: web.Request) -> web.Response:
    """Report the mock's identity and the version it serves."""
    version = request.app[VERSION_KEY]
    return web.json_response(
        {
            "status": "ok",
            "default_profile": "mock",
            "model": "mock-verifier",
            "web_search": "disabled",
            "network_access": False,
            "verifier_harness_version": version,
            "deployment_revision": "mock",
            "available_versions": [version],
        }
    )


def build_app(version: str, policy: str) -> web.Application:
    """Build the mock application."""
    app = web.Application(client_max_size=MAX_BODY_BYTES + 1024)
    app[VERSION_KEY] = version
    app[POLICY_KEY] = policy
    app.router.add_post("/verify", verify)
    app.router.add_get("/health", health)
    app.router.add_get("/healthz", health)
    return app


def main() -> None:
    """Serve the mock verifier."""
    parser = argparse.ArgumentParser(description="Serve a local stand-in for the external verifier.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--policy", choices=("hashed", "always-correct", "always-wrong"), default="hashed")
    parser.add_argument("--version", default="mock-v1")
    args = parser.parse_args()
    web.run_app(build_app(args.version, args.policy), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
