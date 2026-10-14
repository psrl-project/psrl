#!/usr/bin/env python3
"""E2E verification of TITO session training data pipeline.

Validates the full data path through SMG's TITO pipeline:
1. Session lifecycle (create/get/delete)
2. Chat completions with forced logprobs through TITO
3. accumulated_token_ids and per-turn records from SMG
4. Training-data construction (trailing trim, loss mask, logprobs alignment)
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx
from psrl.utils.tito.training_data import build_training_data
from transformers import AutoTokenizer

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Math expression"}},
            "required": ["expression"],
        },
    },
}


# --- Helpers ---
def create_session(base_url: str, client: httpx.Client) -> str:
    resp = client.post(f"{base_url}/sessions")
    resp.raise_for_status()
    sid = resp.json()["session_id"]
    print(f"  ✓ Created session {sid}")
    return sid


def delete_session(base_url: str, client: httpx.Client, sid: str):
    try:
        client.delete(f"{base_url}/sessions/{sid}")
        print(f"  ✓ Deleted session {sid}")
    except Exception as e:
        print(f"  ⚠ Delete failed: {e}")


def chat(base_url, client, sid, messages, model, tools=None, trajectory_id=None):
    body = {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 256}
    if tools:
        body["tools"] = tools
    headers = None
    if trajectory_id is not None:
        headers = {"x-smg-tito-trajectory-id": str(trajectory_id)}
    resp = client.post(f"{base_url}/sessions/{sid}/v1/chat/completions", json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()


def get_session_data(base_url: str, client: httpx.Client, sid: str) -> dict:
    resp = client.get(f"{base_url}/sessions/{sid}")
    resp.raise_for_status()
    return resp.json()


def run_tool(tool_calls):
    results = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        args = (
            json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
        )
        expr = args.get("expression", "0")
        try:
            val = str(eval(expr))  # noqa: S307
        except Exception:
            val = "error"
        results.append({"role": "tool", "tool_call_id": tc.get("id", "call_0"), "content": val})
    return results


# --- Verification Logic ---
def verify_training_data(accumulated, records, label, max_trim_tokens: int = 0):
    """Verify build_training_data produces correct output."""
    try:
        training_data = build_training_data(accumulated, records, max_trim_tokens=max_trim_tokens)
    except ValueError as e:
        print(f"  ✗ build_training_data raised ValueError: {e}")
        return False
    p = training_data["prompt_ids"]
    r = training_data["response_ids"]
    m = training_data["response_mask"]
    lp = training_data["logprobs"]
    nt = training_data["num_turns"]
    errors = []

    # 1. Array length consistency
    if len(r) != len(m):
        errors.append(f"len(response_ids)={len(r)} != len(response_mask)={len(m)}")
    if len(r) != len(lp):
        errors.append(f"len(response_ids)={len(r)} != len(logprobs)={len(lp)}")

    # 2. prompt + response must reconstruct accumulated
    recon = p + r
    if len(recon) != len(accumulated):
        errors.append(f"prompt({len(p)})+response({len(r)})={len(recon)} != accumulated({len(accumulated)})")
    elif recon != accumulated:
        diffs = sum(1 for a, b in zip(recon, accumulated) if a != b)
        errors.append(f"{diffs} token mismatches in reconstruction")

    # 3. Mask values ∈ {0, 1}
    bad = [v for v in m if v not in (0, 1)]
    if bad:
        errors.append(f"{len(bad)} invalid mask values")

    # 4. logprobs = 0.0 where mask = 0
    for i, (mi, li) in enumerate(zip(m, lp)):
        if mi == 0 and li != 0.0:
            errors.append(f"pos {i}: mask=0 but logprob={li}")
            break

    # 5. num_turns matches records
    if nt != len(records):
        errors.append(f"num_turns={nt} != len(records)={len(records)}")

    # 6. At least some model-generated tokens
    model_tok = sum(m)
    env_tok = len(m) - model_tok
    if model_tok == 0 and records:
        errors.append("no model tokens (all mask=0)")

    # 7. The last-turn output must fully align with `accumulated`.
    if records:
        last_lps = records[-1].get("output_logprobs") or []
        last_output_ids = [int(pair[1]) for pair in last_lps]
        if last_output_ids and accumulated:
            cursor = 0
            for rec_i, rec_r in enumerate(records[:-1]):
                cursor = rec_r["prompt_token_count"] + len(rec_r.get("output_logprobs") or [])
            last_prompt_len = records[-1]["prompt_token_count"]
            last_start = max(last_prompt_len, cursor)
            tail_matches = all(
                last_start + j < len(accumulated) and accumulated[last_start + j] == tid
                for j, tid in enumerate(last_output_ids)
            )
            if not tail_matches:
                errors.append(
                    f"last turn output_ids do not align with accumulated at offset {last_start} "
                    f"(unexpected trim on last turn)"
                )

    print(f"\n  [{label}] prompt={len(p)} response={len(r)} model_tok={model_tok} env_tok={env_tok} turns={nt}")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        return False
    print("  ✓ All 7 checks passed")
    return True


def verify_logprobs_in_response(resp, label):
    """Verify chat completion response contains logprobs."""
    choice = resp.get("choices", [{}])[0]
    logprobs = choice.get("logprobs")

    if logprobs is None:
        print(f"  [{label}] ✗ logprobs is None")
        return False

    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if content and len(content) > 0:
        first = content[0]
        print(
            f"  [{label}] ✓ logprobs: {len(content)} entries, first=({first.get('token')!r}, {first.get('logprob')})"
        )
        return True

    print(f"  [{label}] ⚠ logprobs structure: {str(logprobs)[:100]}")
    return True  # Non-fatal


# --- Tests ---
def test_lifecycle(base_url, client):
    print("\n" + "=" * 60)
    print("Test 1: Session lifecycle (create → get → delete → get-404)")
    print("=" * 60)
    sid = create_session(base_url, client)

    resp = client.get(f"{base_url}/sessions/{sid}")
    if resp.status_code != 200:
        print(f"  ✗ GET returned {resp.status_code}: {resp.text[:100]}")
        return False
    print("  ✓ GET returned 200")

    resp = client.delete(f"{base_url}/sessions/{sid}")
    if resp.status_code not in (200, 204):
        print(f"  ✗ DELETE returned {resp.status_code}")
        return False
    print(f"  ✓ DELETE returned {resp.status_code}")

    resp = client.get(f"{base_url}/sessions/{sid}")
    if resp.status_code != 404:
        print(f"  ✗ GET after DELETE returned {resp.status_code} (expected 404)")
        return False
    print("  ✓ GET after DELETE returned 404")
    return True


def test_single_turn_with_tito(base_url, client, model, tokenizer):
    print("\n" + "=" * 60)
    print("Test 2: Single-turn chat completion → TITO training data")
    print("=" * 60)
    sid = create_session(base_url, client)
    try:
        resp = chat(base_url, client, sid, [{"role": "user", "content": "What is the capital of France?"}], model)
        asst = resp["choices"][0]["message"]
        print(f"  Assistant: {str(asst.get('content', ''))[:100]}")

        ok = verify_logprobs_in_response(resp, "single-turn")

        data = get_session_data(base_url, client, sid)
        max_trim_tokens = data.get("max_trim_tokens", 0)

        trajectories = data.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != 1:
            count = len(trajectories) if isinstance(trajectories, list) else "invalid"
            print(f"  ✗ Expected one trajectory, got {count}")
            return False
        trajectory = trajectories[0]
        acc = trajectory.get("accumulated_token_ids", [])
        recs = trajectory.get("records", [])

        if not acc:
            print("  ✗ accumulated_token_ids is empty. TITO did not capture data.")
            return False
        if not recs:
            print("  ✗ records is empty. TurnRecord was not stored.")
            return False

        print(f"  TITO data: {len(acc)} accumulated tokens, {len(recs)} records, max_trim_tokens={max_trim_tokens}")
        for i, rec in enumerate(recs):
            n_lps = len(rec.get("output_logprobs") or [])
            print(
                f"    record[{i}]: prompt={rec['prompt_token_count']} "
                f"output_lps={n_lps} finish={rec.get('finish_reason', '?')}"
            )
            for m_entry in rec.get("mismatch_report", []):
                mtype = m_entry.get("mismatch_type", "?")
                if mtype != "assistant_text":
                    print(
                        f"  ✗ Turn {i} TITO mismatch [{mtype}] "
                        f"pos={m_entry.get('position')} "
                        f"{str(m_entry.get('detail', ''))[:80]}"
                    )
                    ok = False
                else:
                    print(f"  ~ Turn {i} TITO mismatch [assistant_text] (expected, non-fatal)")

        ok = verify_training_data(acc, recs, "single-turn-tito", max_trim_tokens=max_trim_tokens) and ok
        return ok
    finally:
        delete_session(base_url, client, sid)


def test_multi_turn_with_tito(base_url, client, model, max_turns, tokenizer):
    print("\n" + "=" * 60)
    print("Test 3: Multi-turn conversation → TITO training data")
    print("=" * 60)
    sid = create_session(base_url, client)
    messages = [{"role": "user", "content": "What is 2+3? Use the calculator tool."}]
    ok = True
    use_tools = True

    try:
        for turn in range(max_turns):
            try:
                tools = [CALCULATOR_TOOL] if use_tools else None
                resp = chat(base_url, client, sid, messages, model, tools=tools)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 400 and turn == 0 and use_tools:
                    print("  ⚠ Tool call returned 400, model may not support tools. Retrying without.")
                    use_tools = False
                    messages = [{"role": "user", "content": "What is 2+3? Think step by step."}]
                    resp = chat(base_url, client, sid, messages, model)
                else:
                    raise

            choice = resp["choices"][0]
            asst = choice["message"]
            print(f"  Turn {turn + 1}: {str(asst.get('content', ''))[:80]}")
            ok = verify_logprobs_in_response(resp, f"turn-{turn + 1}") and ok

            tool_calls = asst.get("tool_calls")
            if not tool_calls:
                messages.append(asst)
                break
            messages.append(asst)
            for tr in run_tool(tool_calls):
                print(f"  → Tool: {tr['content'][:60]}")
                messages.append(tr)

        data = get_session_data(base_url, client, sid)
        max_trim_tokens = data.get("max_trim_tokens", 0)

        trajectories = data.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != 1:
            count = len(trajectories) if isinstance(trajectories, list) else "invalid"
            print(f"  ✗ Expected one trajectory, got {count}")
            return False
        trajectory = trajectories[0]
        acc = trajectory.get("accumulated_token_ids", [])
        recs = trajectory.get("records", [])

        if not acc:
            print("  ✗ accumulated_token_ids is empty. TITO is not working.")
            return False
        if not recs:
            print("  ✗ records is empty. TurnRecords were not stored.")
            return False

        print(f"\n  TITO data: {len(acc)} accumulated tokens, {len(recs)} records, max_trim_tokens={max_trim_tokens}")
        for i, rec in enumerate(recs):
            n_lps = len(rec.get("output_logprobs") or [])
            print(
                f"    record[{i}]: prompt={rec['prompt_token_count']} "
                f"output_lps={n_lps} finish={rec.get('finish_reason', '?')}"
            )
            for m_entry in rec.get("mismatch_report", []):
                mtype = m_entry.get("mismatch_type", "?")
                if mtype != "assistant_text":
                    print(
                        f"  ✗ Turn {i} TITO mismatch [{mtype}] "
                        f"pos={m_entry.get('position')} "
                        f"{str(m_entry.get('detail', ''))[:80]}"
                    )
                    ok = False
                else:
                    print(f"  ~ Turn {i} TITO mismatch [assistant_text] (expected, non-fatal)")

        ok = verify_training_data(acc, recs, "multi-turn-tito", max_trim_tokens=max_trim_tokens) and ok
        return ok
    finally:
        delete_session(base_url, client, sid)


def test_independent_trajectories_with_tito(base_url, client, model, tokenizer):
    print("\n" + "=" * 60)
    print("Test 4: Independent conversations in one TITO session")
    print("=" * 60)
    sid = create_session(base_url, client)
    try:
        first = chat(
            base_url,
            client,
            sid,
            [{"role": "user", "content": "Return the single word alpha."}],
            model,
            trajectory_id=0,
        )
        second = chat(
            base_url,
            client,
            sid,
            [{"role": "user", "content": "Return the single word beta."}],
            model,
            trajectory_id=1,
        )
        ok = verify_logprobs_in_response(first, "independent-0")
        ok = verify_logprobs_in_response(second, "independent-1") and ok

        data = get_session_data(base_url, client, sid)
        trajectories = data.get("trajectories", [])
        by_id = {str(item.get("trajectory_id", item.get("id", 0))): item for item in trajectories}
        if set(by_id) != {"0", "1"}:
            print(f"  ✗ Expected trajectory IDs 0 and 1, got {sorted(by_id)}")
            return False

        training_data = []
        for trajectory_id in ("0", "1"):
            trajectory = by_id[trajectory_id]
            records = trajectory.get("records", [])
            if len(records) != 1:
                print(f"  ✗ Trajectory {trajectory_id} has {len(records)} records. Expected one.")
                return False
            if not verify_training_data(
                trajectory.get("accumulated_token_ids", []),
                records,
                f"independent-{trajectory_id}",
                max_trim_tokens=data.get("max_trim_tokens", 0),
            ):
                ok = False
            training_data.append(build_training_data(trajectory.get("accumulated_token_ids", []), records))

        if training_data[0]["prompt_ids"] == training_data[1]["prompt_ids"]:
            print("  ✗ Independent conversations unexpectedly have identical prompts")
            ok = False
        else:
            print("  ✓ Independent prompts were preserved as separate training samples")
        return ok
    finally:
        delete_session(base_url, client, sid)


# --- Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="E2E TITO training data verification")
    parser.add_argument("--session-router-url", required=True)
    parser.add_argument("--smg-url", default=None, help="SMG URL for model name auto-detection")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    base = args.session_router_url.rstrip("/")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    client = httpx.Client(timeout=httpx.Timeout(args.timeout))

    model = args.model_path
    if args.smg_url:
        try:
            r = client.get(f"{args.smg_url}/v1/models")
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    model = data[0]["id"]
        except Exception:
            pass
    print(f"Using model: {model}")

    results = [
        ("lifecycle", test_lifecycle(base, client)),
        ("single-turn", test_single_turn_with_tito(base, client, model, tokenizer)),
        ("multi-turn", test_multi_turn_with_tito(base, client, model, args.max_turns, tokenizer)),
        ("independent", test_independent_trajectories_with_tito(base, client, model, tokenizer)),
    ]

    print("\n" + "=" * 60)
    all_ok = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'✓' if ok else '✗'} {name}")
    print("=" * 60)
    print("ALL PASSED" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
