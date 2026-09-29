from __future__ import annotations

import pytest
from examples.mem_agent.config import MemAgentRuntimeConfig
from examples.mem_agent.runner import MemAgent


class WordTokenizer:
    def __init__(self):
        self._tokens: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        result = []
        for word in text.split():
            if word not in self._tokens:
                token_id = len(self._tokens) + 1
                self._tokens[word] = token_id
                self._words[token_id] = word
            result.append(self._tokens[word])
        return result

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self._words[token_id] for token_id in token_ids)


@pytest.mark.asyncio
async def test_mem_agent_uses_independent_conversations_and_overwrites_memory():
    calls = []

    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

        async def json(self):
            return {"choices": [{"message": {"content": self.text}, "finish_reason": "stop"}]}

    class RequestContext:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc, tb):
            return None

    class HttpSession:
        def post(self, url, *, json, headers):
            calls.append((url, json, headers))
            text = ("memory-one", "memory-two", "\\boxed{answer}")[len(calls) - 1]
            return RequestContext(Response(text))

    result = await MemAgent(
        tokenizer=WordTokenizer(),
        base_url="http://session-router/sessions/sid/v1",
        model="model",
        config=MemAgentRuntimeConfig(chunk_tokens=2, max_chunks=2),
        api_key="EMPTY",
        http_session=HttpSession(),
    ).run(
        "Which answer?",
        "alpha beta gamma delta",
        sampling_params={"temperature": 1.0, "max_tokens": 999},
    )

    assert len(result.turns) == 3
    assert result.chunks_processed == 2
    assert result.final_response == "\\boxed{answer}"
    assert calls[0][0] == "http://session-router/sessions/sid/v1/chat/completions"
    assert calls[0][1]["max_tokens"] == 1024
    assert calls[-1][1]["max_tokens"] == 256
    assert len(calls[0][1]["messages"]) == len(calls[1][1]["messages"]) == 1
    assert "alpha beta" in calls[0][1]["messages"][0]["content"]
    assert "memory-one" in calls[1][1]["messages"][0]["content"]
    assert "alpha beta" not in calls[1][1]["messages"][0]["content"]
    assert "memory-two" in calls[2][1]["messages"][0]["content"]
    assert "gamma delta" not in calls[2][1]["messages"][0]["content"]
    assert calls[0][2]["Authorization"] == "Bearer EMPTY"


@pytest.mark.asyncio
async def test_mem_agent_rejects_silent_context_truncation():
    class HttpSession:
        def post(self, *args, **kwargs):
            raise AssertionError("Chat Completion should not be called.")

    with pytest.raises(ValueError, match="exceeding the configured capacity"):
        await MemAgent(
            tokenizer=WordTokenizer(),
            base_url="http://session-router/sessions/sid/v1",
            model="model",
            config=MemAgentRuntimeConfig(chunk_tokens=1, max_chunks=2),
            http_session=HttpSession(),
        ).run("question", "one two three")
