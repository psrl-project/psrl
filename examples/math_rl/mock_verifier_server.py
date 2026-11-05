"""A local stand-in for the external verifier, used by tests and dry runs.

It implements the blocking `/verify` contract and the `/verify/stream` stage
events: exactly four request fields, a verdict in {0, 1, 6, 7}, and the
receipt in response headers or in the terminal stream event.
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


def _read(body: Any) -> tuple[str, str, dict[str, Any]] | None:
    """Validate one request body and return the problem, solution and config."""
    if not isinstance(body, dict) or set(body) != FIELDS:
        return None
    problem = body["problem"]
    solution = body["solution_to_verify"]
    config = body["model_config"]
    if not isinstance(problem, str) or not problem.strip():
        return None
    if not isinstance(solution, str) or not isinstance(config, dict):
        return None
    return problem, solution, config


def _answer(
    app: web.Application, problem: str, solution: str, config: dict[str, Any], started: float
) -> dict[str, Any]:
    """Build the verdict, run id, usage and receipt headers for one request."""
    verdict = _verdict(app[POLICY_KEY], problem, solution)
    run_id = "mock_" + _sha256(problem + "\x00" + solution)[:12]
    usage = {"input_tokens": max(1, len(problem) // 4), "output_tokens": max(1, len(solution) // 4)}
    receipt = {
        "run_id": run_id,
        "harness_sha256": _sha256(str(config)),
        "prompt_sha256": _sha256(problem),
        "skill_sha256": _sha256(json.dumps(config, sort_keys=True)),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "usage": usage,
    }
    return {
        "verdict": verdict,
        "reason": f"mock verdict {verdict} from policy {app[POLICY_KEY]}",
        "receipt": receipt,
    }


async def _body(request: web.Request) -> Any:
    raw = await request.read()
    if len(raw) > MAX_BODY_BYTES:
        raise web.HTTPRequestEntityTooLarge(max_size=MAX_BODY_BYTES, actual_size=len(raw))
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise web.HTTPUnprocessableEntity(text="invalid JSON") from error


async def verify(request: web.Request) -> web.Response:
    """Grade one proof with the configured mock policy."""
    started = time.perf_counter()
    try:
        parsed = _read(await _body(request))
    except web.HTTPException as error:
        return error
    if parsed is None:
        return web.json_response({"detail": "the request must carry exactly the four fields"}, status=422)
    problem, solution, config = parsed
    answer = _answer(request.app, problem, solution, config, started)
    headers = {
        "x-verifier-receipt": answer["receipt"]["run_id"],
        "x-verifier-harness-sha256": answer["receipt"]["harness_sha256"],
        "x-verifier-prompt-sha256": answer["receipt"]["prompt_sha256"],
        "x-verifier-skill-sha256": answer["receipt"]["skill_sha256"],
        "x-verifier-duration-seconds": f"{answer['receipt']['duration_seconds']:.3f}",
        "x-verifier-usage": json.dumps(answer["receipt"]["usage"], separators=(",", ":")),
    }
    return web.json_response({"verdict": answer["verdict"], "reason": answer["reason"]}, headers=headers)


async def verify_stream(request: web.Request) -> web.StreamResponse:
    """Emit the stage events and the terminal result like the served verifier."""
    started = time.perf_counter()
    try:
        parsed = _read(await _body(request))
    except web.HTTPException as error:
        return error
    if parsed is None:
        return web.json_response({"detail": "the request must carry exactly the four fields"}, status=422)
    problem, solution, config = parsed
    answer = _answer(request.app, problem, solution, config, started)
    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "x-verifier-receipt": answer["receipt"]["run_id"],
        }
    )
    await response.prepare(request)

    async def emit(event: str, payload: dict[str, Any]) -> None:
        data = {"run_id": answer["receipt"]["run_id"], "timestamp": time.time(), **payload}
        await response.write(f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n".encode())

    for stage in ("accepted", "contract", "audit", "grading"):
        await emit("progress", {"stage": stage, "message": stage, "elapsed_seconds": 0.0})
    await emit(
        "result",
        {
            "stage": "completed",
            "result": {"verdict": answer["verdict"], "reason": answer["reason"]},
            "receipt": answer["receipt"],
        },
    )
    await response.write_eof()
    return response


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
    app.router.add_post("/verify/stream", verify_stream)
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
