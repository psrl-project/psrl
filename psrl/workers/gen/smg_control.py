"""
Validate native SMG control APIs before gateways accept workloads.
"""

import multiprocessing
import time

import httpx


def wait_for_router_ready(
    gateway_url: str,
    process: multiprocessing.Process,
    *,
    require_elastic_runtime: bool,
    timeout_s: float = 120.0,
) -> None:
    """
    Wait for HTTP readiness and verify the required routing control APIs.
    """
    deadline = time.monotonic() + timeout_s
    with httpx.Client(trust_env=False, timeout=5.0) as client:
        while True:
            if not process.is_alive():
                raise RuntimeError("SMG router exited during initialization.")
            if time.monotonic() >= deadline:
                raise TimeoutError("SMG router initialization timed out.")
            try:
                response = client.get(f"{gateway_url}/health")
            except httpx.TransportError:
                time.sleep(0.1)
                continue
            response.raise_for_status()
            break

        routes = [("/routing_loop/status", None, (dict,))]
        if require_elastic_runtime:
            routes.extend(
                [
                    ("/routing_loop/snapshot", None, (list,)),
                    ("/routing_loop/migration", {"request_id": "psrl-startup-check"}, (dict, type(None))),
                ]
            )
        for path, params, response_types in routes:
            response = client.get(f"{gateway_url}{path}", params=params)
            if response.status_code == 404:
                raise RuntimeError(
                    f"SMG control API {path!r} is unavailable: {response.text!r}. "
                    "Ensure routing_loop is enabled and rebuild the native SMG library with "
                    "bash scripts/reinstall_smg.sh, then restart the Ray job."
                )
            response.raise_for_status()
            if not isinstance(response.json(), response_types):
                expected = " or ".join(response_type.__name__ for response_type in response_types)
                raise TypeError(f"SMG control API {path!r} must return {expected}.")
