"""Tests for `OpenAISessionAdapter`.

The proxy stands in for SMG's session layer over an endpoint that has none, so the
contract it must honor is `SessionAgentLoop`'s: the snapshot shape that
`get_training_data` parses, and the turn counts a loop compares against its budget.

The load-bearing property is a negative one. The proxy must NOT invent token ids.
Re-encoding assistant text locally would yield ids that never passed through the
server, and writing them into `response_ids` would give fabricated tokens the same
provenance as measured ones.
"""

import asyncio
import json

import pytest
from psrl.batch_rollout.serving.openai_api.openai_session_adapter import OpenAISessionAdapter
from psrl.utils.tito.training_data import build_training_data


class _Request:
    """Minimal stand-in for a Starlette request."""

    def __init__(self, body: dict | None = None, headers: dict | None = None):
        self._body = json.dumps(body or {}).encode()
        self.headers = headers or {}

    async def body(self) -> bytes:
        return self._body


def _proxy(tmp_path) -> OpenAISessionAdapter:
    return OpenAISessionAdapter(upstream_url="http://upstream/v1", transcript_dir=str(tmp_path))


def _completion(content: str = "hi", prompt_tokens: int = 10, completion_tokens: int = 4, finish: str = "stop"):
    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def _body(response) -> dict:
    return json.loads(bytes(response.body).decode())


async def _open_session(proxy, uid: str = "42") -> str:
    created = await proxy.create_session(_Request(headers={"x-request-id": uid}))
    return _body(created)["session_id"]


@pytest.mark.cpu_test
def test_snapshot_matches_the_shape_get_training_data_parses(tmp_path):
    """`SessionAgentLoop` indexes these keys directly, so a missing one is a crash."""

    async def scenario():
        proxy = _proxy(tmp_path)
        sid = await _open_session(proxy)
        await proxy._record_turn(proxy.sessions[sid], {"messages": []}, _completion(), 200)
        return _body(await proxy.get_session(sid))

    snapshot = asyncio.run(scenario())

    assert isinstance(snapshot["trajectories"], list)
    trajectory = snapshot["trajectories"][0]
    assert trajectory["trajectory_id"] == 0
    assert "accumulated_token_ids" in trajectory
    assert "records" in trajectory
    assert snapshot["max_trim_tokens"] == 0
    assert isinstance(snapshot["header_info"], dict)


@pytest.mark.cpu_test
def test_snapshot_feeds_build_training_data_with_honest_turn_counts(tmp_path):
    """num_turns drives the loop's budget checks, and must count real turns."""

    async def scenario():
        proxy = _proxy(tmp_path)
        sid = await _open_session(proxy)
        for _ in range(3):
            await proxy._record_turn(proxy.sessions[sid], {"messages": []}, _completion(), 200)
        return _body(await proxy.get_session(sid))

    snapshot = asyncio.run(scenario())
    trajectory = snapshot["trajectories"][0]

    training_data = build_training_data(
        trajectory["accumulated_token_ids"],
        trajectory["records"],
        max_trim_tokens=snapshot["max_trim_tokens"],
    )

    assert training_data["num_turns"] == 3
    # The decisive assertion: no token stream was invented for an endpoint that
    # never returned one.
    assert training_data["response_ids"] == []
    assert training_data["prompt_ids"] == []


@pytest.mark.cpu_test
def test_usage_is_reported_from_upstream_not_recomputed(tmp_path):
    """Token counts come from the server that did the work, not a local estimate."""

    async def scenario():
        proxy = _proxy(tmp_path)
        sid = await _open_session(proxy)
        await proxy._record_turn(
            proxy.sessions[sid], {"messages": []}, _completion(prompt_tokens=100, completion_tokens=7), 200
        )
        await proxy._record_turn(
            proxy.sessions[sid], {"messages": []}, _completion(prompt_tokens=140, completion_tokens=9), 200
        )
        return _body(await proxy.get_session(sid))

    snapshot = asyncio.run(scenario())

    assert snapshot["usage"] == {"prompt_tokens": 240, "completion_tokens": 16, "turns": 2}


@pytest.mark.cpu_test
def test_finish_reason_is_carried_per_turn(tmp_path):
    """The loop reads the last turn's finish_reason to classify the episode."""

    async def scenario():
        proxy = _proxy(tmp_path)
        sid = await _open_session(proxy)
        await proxy._record_turn(proxy.sessions[sid], {"messages": []}, _completion(finish="stop"), 200)
        await proxy._record_turn(proxy.sessions[sid], {"messages": []}, _completion(finish="length"), 200)
        return _body(await proxy.get_session(sid))

    snapshot = asyncio.run(scenario())
    records = snapshot["trajectories"][0]["records"]

    assert [r["finish_reason"] for r in records] == ["stop", "length"]


@pytest.mark.cpu_test
def test_transcript_is_named_by_uid_so_it_joins_to_the_dump(tmp_path):
    """The record carries a uid, so the transcript must be findable from it."""

    async def scenario():
        proxy = _proxy(tmp_path)
        sid = await _open_session(proxy, uid="777")
        await proxy._record_turn(
            proxy.sessions[sid], {"messages": [{"role": "user", "content": "go"}]}, _completion("done"), 200
        )
        await proxy.delete_session(sid)

    asyncio.run(scenario())

    transcript = json.loads((tmp_path / "777.json").read_text(encoding="utf-8"))
    assert transcript["uid"] == "777"
    assert transcript["turns"][0]["messages"] == [{"role": "user", "content": "go"}]
    assert transcript["turns"][0]["response"]["choices"][0]["message"]["content"] == "done"


@pytest.mark.cpu_test
def test_unknown_session_is_rejected_rather_than_silently_empty(tmp_path):
    """An empty snapshot for a bad id would look like a legitimate zero-turn run."""

    async def scenario():
        proxy = _proxy(tmp_path)
        return await proxy.get_session("nope")

    response = asyncio.run(scenario())

    assert response.status_code == 404


@pytest.mark.cpu_test
def test_failed_turns_are_still_recorded(tmp_path):
    """A context-overflow 400 is diagnostic, so it must not vanish from the turn log."""

    async def scenario():
        proxy = _proxy(tmp_path)
        sid = await _open_session(proxy)
        await proxy._record_turn(proxy.sessions[sid], {"messages": []}, {"error": "context overflow"}, 400)
        snapshot = _body(await proxy.get_session(sid))
        await proxy.delete_session(sid)
        return snapshot

    snapshot = asyncio.run(scenario())

    assert len(snapshot["trajectories"][0]["records"]) == 1
    transcript = json.loads((tmp_path / "42.json").read_text(encoding="utf-8"))
    assert transcript["turns"][0]["status"] == 400


@pytest.mark.cpu_test
def test_delete_is_idempotent(tmp_path):
    """Agent loops delete in a finally block, which can run twice on an error path."""

    async def scenario():
        proxy = _proxy(tmp_path)
        sid = await _open_session(proxy)
        first = _body(await proxy.delete_session(sid))
        second = _body(await proxy.delete_session(sid))
        return first, second

    first, second = asyncio.run(scenario())

    assert first["deleted"] is True
    assert second["deleted"] is False


@pytest.mark.cpu_test
def test_upstream_client_honors_the_environment_proxy(tmp_path):
    """A cluster behind a corporate proxy has no direct route to a hosted API.

    aiohttp ignores `HTTP(S)_PROXY` unless `trust_env` is set, so without this the
    proxy reaches localhost fine and every external endpoint times out. That gap is
    invisible to a stub-server test, which is exactly why it is pinned here.
    """

    async def scenario():
        proxy = _proxy(tmp_path)
        client = await proxy._ensure_client()
        try:
            return client.trust_env
        finally:
            await proxy.aclose()

    assert asyncio.run(scenario()) is True


@pytest.mark.cpu_test
def test_model_id_is_rewritten_for_a_hosted_api(tmp_path):
    """Agent loops send `model_config.path`, which a hosted API rejects outright.

    That path is the right id for a local vLLM, which serves under it, so the
    substitution belongs in the proxy rather than in every loop.
    """

    async def scenario():
        proxy = OpenAISessionAdapter(
            upstream_url="http://upstream/v1",
            transcript_dir=str(tmp_path),
            model_name="vendor/real-model:free",
        )
        captured = {}

        class _Resp:
            status = 200

            async def text(self):
                return json.dumps(_completion())

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Client:
            def post(self, url, json=None, headers=None):
                captured["payload"] = json
                return _Resp()

        proxy.client = _Client()
        proxy.client.closed = False
        sid = await _open_session(proxy)
        await proxy.session_chat_completions(sid, _Request({"model": "/models/Qwen3.5-4B", "messages": []}))
        return captured["payload"]

    payload = asyncio.run(scenario())

    assert payload["model"] == "vendor/real-model:free"


@pytest.mark.cpu_test
def test_model_id_is_left_alone_when_unset(tmp_path):
    """A local server already accepts the loop's id, so do not rewrite it."""

    async def scenario():
        proxy = _proxy(tmp_path)
        captured = {}

        class _Resp:
            status = 200

            async def text(self):
                return json.dumps(_completion())

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        class _Client:
            closed = False

            def post(self, url, json=None, headers=None):
                captured["payload"] = json
                return _Resp()

        proxy.client = _Client()
        sid = await _open_session(proxy)
        await proxy.session_chat_completions(sid, _Request({"model": "/models/local", "messages": []}))
        return captured["payload"]

    assert asyncio.run(scenario())["model"] == "/models/local"


@pytest.mark.cpu_test
def test_empty_assistant_messages_are_dropped_before_forwarding():
    """Strict providers reject the whole request over one empty assistant turn.

    An agent harness emits one whenever the model returns an empty completion, so
    without this a single bad turn kills the rest of the episode.
    """
    from psrl.batch_rollout.serving.openai_api.openai_session_adapter import _drop_empty_messages

    kept = _drop_empty_messages(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "   "},
            {"role": "assistant", "content": None},
            {"role": "assistant", "content": "real work"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
            {"role": "user", "content": ""},
        ]
    )

    assert [m["role"] for m in kept] == ["system", "user", "assistant", "assistant", "user"]
    # A tool call with no text still carries an action, so it survives.
    assert kept[-2]["tool_calls"] == [{"id": "1"}]
    # A non-list payload passes through rather than raising.
    assert _drop_empty_messages(None) is None
