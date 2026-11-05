"""Live parallel, nested, and compaction episode over a real SMG/SessionRouter.

The main agent has one turn, then starts two parallel subagents. One of them
starts a nested subagent. The main context is then compacted and resumed. The
snapshot must show distinct parallel leaves, the nested parent, and a live
compaction main leaf that replaces the pre-compaction one in selection.
"""

from __future__ import annotations

import os
import threading

import httpx
import uvicorn

os.environ.setdefault(
    "PSRL_TITO_E2E_MODEL",
    "/apdcephfs_zwfy10_303541817/share_303541817/ls/psrl-wip/models/Qwen2.5-0.5B-Instruct",
)
os.environ.setdefault("PSRL_TITO_E2E_MAX_MODEL_LEN", "8192")
os.environ.setdefault("PSRL_TITO_E2E_START_VLLM", "1")

from psrl.utils.subagent_training import SubagentTrainingPolicy  # noqa: E402
from psrl.utils.tito.training_data import build_training_data  # noqa: E402
from psrl.utils.tito.trajectory_selection import select_trainable_trajectories  # noqa: E402

from tests.tito import live_subagent_e2e as e  # noqa: E402


def main() -> None:
    e.START_VLLM = True
    e.start_stack()
    router = e.SessionRouter(smg_url=f"http://127.0.0.1:{e.SMG_PORT}", trajectory_id_strategy="auto")
    config = uvicorn.Config(router.app, host="127.0.0.1", port=e.ROUTER_PORT, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    e.wait_http(f"{e.ROUTER_URL}/control/sessions", 30)
    print("SessionRouter ready", flush=True)
    try:
        with httpx.Client(base_url=e.ROUTER_URL, timeout=120) as client:
            policy = SubagentTrainingPolicy.from_mapping({"mode": "all", "require_primary_capture": True})
            sid = e.create_session(client, policy)
            base = [{"role": "user", "content": "Name one color."}]
            main_meta = {"thread_source": "user", "thread_id": "main-1"}
            main = e.turn(client, sid, base, main_meta)
            main_assistant = main["choices"][0]["message"]

            prefix = base + [main_assistant]
            a_meta = {"thread_source": "subagent", "thread_id": "sub-a", "parent_thread_id": "main-1"}
            b_meta = {"thread_source": "subagent", "thread_id": "sub-b", "parent_thread_id": "main-1"}
            a = e.turn(client, sid, prefix + [{"role": "user", "content": "Answer as A."}], a_meta)
            e.turn(client, sid, prefix + [{"role": "user", "content": "Answer as B."}], b_meta)
            a_assistant = a["choices"][0]["message"]
            c_meta = {"thread_source": "subagent", "thread_id": "sub-c", "parent_thread_id": "sub-a"}
            e.turn(
                client,
                sid,
                prefix + [a_assistant, {"role": "user", "content": "Answer as A's nested helper."}],
                c_meta,
            )

            compacted = [
                {"role": "system", "content": "Summary: the user asked for colors."},
                {"role": "user", "content": "Continue from the summary."},
            ]
            # Compaction is a Claude Code request class. It must not carry Codex
            # turn metadata, whose thread_source would win in classification.
            compact = client.post(
                f"/sessions/{sid}/v1/chat/completions",
                json={
                    "model": e.MODEL,
                    "messages": compacted,
                    "max_tokens": 24,
                    "temperature": 0.8,
                    "top_p": 1.0,
                },
                headers={
                    "x-claude-code-request-class": "compaction",
                    "x-claude-code-agent-id": "main-1",
                },
            )
            compact.raise_for_status()
            compact = compact.json()
            compact_assistant = compact["choices"][0]["message"]
            e.turn(
                client,
                sid,
                compacted + [compact_assistant, {"role": "user", "content": "Add one more."}],
                main_meta,
            )

            snapshot = client.get(f"/sessions/{sid}").json()
            roles = {}
            for trajectory in snapshot["trajectories"]:
                roles[trajectory["trajectory_id"]] = (
                    trajectory.get("thread_source"),
                    trajectory.get("agent_id"),
                    trajectory.get("parent_agent_id"),
                )
            print("trajectories", roles, flush=True)
            assert any(role == "subagent" and agent == "sub-a" for role, agent, _ in roles.values())
            assert any(role == "subagent" and agent == "sub-b" for role, agent, _ in roles.values())
            assert any(role == "subagent" and agent == "sub-c" for role, agent, _ in roles.values())
            assert any(role == "compaction" and agent == "main-1" for role, agent, _ in roles.values())
            assert any(parent == "main-1" for _, agent, parent in roles.values() if agent == "sub-a")
            assert any(parent == "sub-a" for _, agent, parent in roles.values() if agent == "sub-c")

            for mode in ("main_only", "all", "selected"):
                current = policy.overrides({"mode": mode, "train_agents": ["sub-a", "sub-c"]})
                rows = select_trainable_trajectories(snapshot, current)
                print(f"mode={mode} rows={[(r.thread_source, r.agent_id, r.is_primary) for r in rows]}", flush=True)
                for selection in rows:
                    built = build_training_data(
                        selection.trajectory["accumulated_token_ids"],
                        selection.trajectory["records"],
                        max_trim_tokens=snapshot.get("max_trim_tokens", 0),
                        trainable_trajectory_id=selection.trainable_trajectory_id,
                        shared_prefix=current.shared_prefix,
                    )
                    assert built["response_ids"], selection
                    assert sum(built["response_mask"]) > 0, selection
                if mode == "main_only":
                    assert len(rows) == 1 and rows[0].is_primary
                elif mode == "selected":
                    assert {r.agent_id for r in rows} == {"main-1", "sub-a", "sub-c"}
                else:
                    assert {r.agent_id for r in rows} == {"main-1", "sub-a", "sub-b", "sub-c"}
            print("parallel/nested/compaction snapshot OK", flush=True)
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        import signal

        for proc in reversed(e.procs):
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except Exception:  # noqa: BLE001 - teardown
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


if __name__ == "__main__":
    main()
