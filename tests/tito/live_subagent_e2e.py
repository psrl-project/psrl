"""Live TITO/subagent E2E over a real vLLM + SMG + SessionRouter stack.

The vLLM gRPC worker is external. Point the script at it with
`PSRL_TITO_E2E_MODEL` and, when it should launch the worker itself,
`PSRL_TITO_E2E_START_VLLM=1`:

    PSRL_TITO_E2E_MODEL=/models/Qwen3-8B \
    PSRL_TITO_E2E_MAX_MODEL_LEN=32768 \
    CUDA_VISIBLE_DEVICES=0 python tests/tito/live_subagent_e2e.py

It starts the SMG router and the SessionRouter in-process, then drives two
sessions: `main_only` and `all`. It asserts the snapshot roles, owners, capture
modes, owner-aware extraction rows, and prompt boundaries.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import uvicorn
from psrl.utils.subagent_training import SubagentTrainingPolicy
from psrl.utils.tito.snapshot import normalize_session_snapshot
from psrl.utils.tito.training_data import build_training_data
from psrl.utils.tito.trajectory_selection import select_trainable_trajectories
from psrl.workers.gen.session_router import SessionRouter

MODEL = os.environ.get("PSRL_TITO_E2E_MODEL", "")
MAX_MODEL_LEN = os.environ.get("PSRL_TITO_E2E_MAX_MODEL_LEN", "4096")
START_VLLM = os.environ.get("PSRL_TITO_E2E_START_VLLM") == "1"
VLLM_PORT = 51051
SMG_PORT = 51052
ROUTER_PORT = 51053
ROUTER_URL = f"http://127.0.0.1:{ROUTER_PORT}"

procs: list[subprocess.Popen] = []
LOG_DIR = tempfile.mkdtemp(prefix="tito-e2e-")


def wait_port(port: int, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.5)
    raise TimeoutError(f"port {port} did not open")


def wait_http(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2)
            if response.status_code < 500:
                return
            last = response
        except Exception as error:  # noqa: BLE001 - readiness probe
            last = error
        time.sleep(1)
    raise TimeoutError(f"{url} did not become ready: {last}")


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_stack() -> None:
    if not MODEL:
        raise RuntimeError("Set PSRL_TITO_E2E_MODEL to a local model path.")
    env = os.environ.copy()
    if START_VLLM and not port_open(VLLM_PORT):
        vllm = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.grpc_server",
                "--model",
                MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(VLLM_PORT),
                "--enforce-eager",
                "--max-model-len",
                MAX_MODEL_LEN,
                "--max-num-seqs",
                "8",
                "--gpu-memory-utilization",
                "0.35",
            ],
            env=env,
            start_new_session=True,
            stdout=open(os.path.join(LOG_DIR, "vllm.log"), "wb"),
            stderr=subprocess.STDOUT,
        )
        procs.append(vllm)
        wait_port(VLLM_PORT, 300)
        print("vLLM gRPC ready", flush=True)

    smg = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "smg.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(SMG_PORT),
            "--model-path",
            MODEL,
            "--policy",
            "round_robin",
            "--connection-mode",
            "grpc",
            "--enable-tito",
            "--trajectory-id-strategy",
            "auto",
            "--disable-retries",
            "--disable-health-check",
            "--tool-call-parser",
            os.environ.get("E2E_TOOL_PARSER", "qwen"),
            "--reasoning-parser",
            os.environ.get("E2E_REASONING_PARSER", "qwen3"),
            "--log-level",
            "info",
        ],
        env=env,
        start_new_session=True,
        stdout=open(os.path.join(LOG_DIR, "smg.log"), "wb"),
        stderr=subprocess.STDOUT,
    )
    procs.append(smg)
    wait_http(f"http://127.0.0.1:{SMG_PORT}/health", 120)
    registration = {
        "url": f"grpc://127.0.0.1:{VLLM_PORT}",
        "worker_type": "regular",
        "connection_mode": "grpc",
        "runtime_type": "vllm",
        "models": [{"id": MODEL}],
        "labels": {"max_model_len": MAX_MODEL_LEN, "dp_size": "1", "tp_size": "1", "pp_size": "1"},
    }
    response = httpx.post(f"http://127.0.0.1:{SMG_PORT}/workers", json=registration, timeout=30)
    response.raise_for_status()
    deadline = time.time() + 120
    while time.time() < deadline:
        response = httpx.get(f"http://127.0.0.1:{SMG_PORT}/workers", timeout=5)
        if response.status_code == 200 and MODEL in response.text:
            break
        time.sleep(1)
    else:
        raise TimeoutError("worker did not become routable")
    print("SMG ready", flush=True)


def create_session(client: httpx.Client, policy: SubagentTrainingPolicy) -> str:
    response = client.post("/sessions", json={"subagent_training": policy.to_dict()})
    response.raise_for_status()
    return response.json()["session_id"]


def turn(
    client: httpx.Client,
    sid: str,
    messages: list[dict],
    metadata: dict,
) -> dict:
    response = client.post(
        f"/sessions/{sid}/v1/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": 24, "temperature": 0.8, "top_p": 1.0},
        headers={"x-codex-turn-metadata": json.dumps(metadata)},
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"turn failed: {response.status_code} {response.text}")
    return response.json()


def drive(client: httpx.Client, sid: str) -> None:
    base = [{"role": "user", "content": "Think of a short plan. Reply with one sentence."}]
    first = turn(client, sid, base, {"thread_source": "user", "thread_id": "main-1"})
    assistant = first["choices"][0]["message"]

    # A subagent branches after the first main turn. Its two turns complete before
    # the main agent closes, so they remain causally eligible.
    sub_base = base + [assistant, {"role": "user", "content": "Act as a reviewer and answer briefly."}]
    sub_first = turn(
        client,
        sid,
        sub_base,
        {"thread_source": "subagent", "thread_id": "sub-1", "parent_thread_id": "main-1"},
    )
    sub_assistant = sub_first["choices"][0]["message"]
    sub_followup = sub_base + [sub_assistant, {"role": "user", "content": "Make it one word shorter."}]
    turn(
        client,
        sid,
        sub_followup,
        {"thread_source": "subagent", "thread_id": "sub-1", "parent_thread_id": "main-1"},
    )

    followup = base + [assistant, {"role": "user", "content": "Add one more detail."}]
    turn(client, sid, followup, {"thread_source": "user", "thread_id": "main-1"})


def assert_snapshot(snapshot: dict, *, mode: str) -> None:
    trajectories = snapshot["trajectories"]
    by_role = {item["thread_source"]: item for item in trajectories}
    assert set(by_role) == {"main", "subagent"}, by_role
    main = by_role["main"]
    sub = by_role["subagent"]
    assert main["capture"] == "full", main
    assert sub["capture"] == ("full" if mode == "all" else "tokens"), sub
    assert main.get("parent_trajectory_id") is None, main
    assert sub.get("parent_trajectory_id") == main["trajectory_id"], sub
    assert sub.get("agent_id") == "sub-1", sub
    assert any(record["owner_trajectory_id"] == sub["trajectory_id"] for record in sub["records"])
    assert snapshot["main_agent_id"] == "main-1", snapshot.get("main_agent_id")
    assert snapshot["subagent_training"]["mode"] == mode
    assert all(
        "sequence" in record
        for trajectory in trajectories
        for record in trajectory.get("records", [])
    ), trajectories

    policy = SubagentTrainingPolicy.from_mapping(snapshot["subagent_training"])
    selections = select_trainable_trajectories(snapshot, policy)
    if mode == "main_only":
        assert [item.trajectory["trajectory_id"] for item in selections] == [main["trajectory_id"]]
    else:
        assert {item.trajectory["trajectory_id"] for item in selections} == {
            main["trajectory_id"],
            sub["trajectory_id"],
        }

    for selection in selections:
        built = build_training_data(
            selection.trajectory["accumulated_token_ids"],
            selection.trajectory["records"],
            max_trim_tokens=snapshot.get("max_trim_tokens", 0),
            trainable_trajectory_id=selection.trainable_trajectory_id,
            shared_prefix=policy.shared_prefix,
        )
        assert built["num_turns"] > 0, (selection, built)
        assert built["response_ids"], (selection, built)
        assert sum(built["response_mask"]) > 0, (selection, built)
        first_owned = next(
            record
            for record in selection.trajectory["records"]
            if record["owner_trajectory_id"] == selection.trainable_trajectory_id
        )
        assert built["prompt_ids"] == selection.trajectory["accumulated_token_ids"][
            : first_owned["prompt_token_count"]
        ], (selection, built)
        print(
            f"row role={selection.thread_source} prompt={len(built['prompt_ids'])} "
            f"response={len(built['response_ids'])} mask={sum(built['response_mask'])} "
            f"turns={built['num_turns']} "
            f"re={None if built['routed_experts'] is None else tuple(built['routed_experts'].shape)}",
            flush=True,
        )


def main() -> None:
    start_stack()
    router = SessionRouter(smg_url=f"http://127.0.0.1:{SMG_PORT}", trajectory_id_strategy="auto")
    config = uvicorn.Config(router.app, host="127.0.0.1", port=ROUTER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    wait_http(f"{ROUTER_URL}/control/sessions", 30)
    print("SessionRouter ready", flush=True)

    with httpx.Client(base_url=ROUTER_URL, timeout=120) as client:
        for mode in ("main_only", "all"):
            policy = SubagentTrainingPolicy.from_mapping(
                {"mode": mode, "require_primary_capture": True}
            )
            sid = create_session(client, policy)
            drive(client, sid)
            expanded_raw = client.get(f"/sessions/{sid}").json()
            compact_raw = client.get(f"/sessions/{sid}", params={"format": "compact"}).json()
            snapshot = normalize_session_snapshot(expanded_raw)
            compact = normalize_session_snapshot(compact_raw)
            print(
                f"mode={mode} snapshot bytes expanded={len(json.dumps(expanded_raw))} "
                f"compact={len(json.dumps(compact_raw))}",
                flush=True,
            )
            assert_snapshot(snapshot, mode=mode)
            assert_snapshot(compact, mode=mode)
            expanded_rows = [
                (item["trajectory_id"], item["accumulated_token_ids"], item["records"])
                for item in snapshot["trajectories"]
            ]
            compact_rows = [
                (item["trajectory_id"], item["accumulated_token_ids"], item["records"])
                for item in compact["trajectories"]
            ]
            assert expanded_rows == compact_rows
            print(f"mode={mode} expanded/compact snapshot OK", flush=True)

        # A subagent that runs after the main agent closes is post-main. It must
        # remain in the snapshot but never produce a trainable row.
        post_policy = SubagentTrainingPolicy.from_mapping({"mode": "all"})
        post_sid = create_session(client, post_policy)
        post_base = [{"role": "user", "content": "Answer with one word."}]
        post_main = turn(client, post_sid, post_base, {"thread_source": "user", "thread_id": "main-9"})
        post_assistant = post_main["choices"][0]["message"]
        turn(
            client,
            post_sid,
            post_base + [post_assistant, {"role": "user", "content": "Now delegate."}],
            {"thread_source": "subagent", "thread_id": "sub-9", "parent_thread_id": "main-9"},
        )
        post_snapshot = normalize_session_snapshot(client.get(f"/sessions/{post_sid}").json())
        post_rows = select_trainable_trajectories(post_snapshot, post_policy)
        assert [row.thread_source for row in post_rows] == ["main"]
        sub_trajectory = next(
            trajectory for trajectory in post_snapshot["trajectories"]
            if trajectory["thread_source"] == "subagent"
        )
        assert all(
            record.get("contributes_to_session") is False
            for record in sub_trajectory["records"]
            if record.get("owner_trajectory_id") == sub_trajectory["trajectory_id"]
        )
        print("mode=all post-main subagent excluded OK", flush=True)

    server.should_exit = True
    thread.join(timeout=10)


if __name__ == "__main__":
    try:
        main()
    finally:
        for proc in reversed(procs):
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
