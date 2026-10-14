import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
from examples.mem_agent.config import MemAgentRuntimeConfig


@dataclass(frozen=True)
class MemAgentTurn:
    """One context-independent conversation produced by MemAgent."""

    kind: str
    response: str
    finish_reason: str
    chunk_index: int | None = None


@dataclass(frozen=True)
class MemAgentRunResult:
    """Result and diagnostics for one complete MemAgent episode."""

    turns: list[MemAgentTurn]
    final_response: str
    final_memory: str
    context_tokens: int
    chunks_processed: int
    context_truncated: bool


class MemAgent:
    """Run memory updates through an OpenAI-compatible HTTP endpoint."""

    def __init__(
        self,
        tokenizer: Any,
        base_url: str,
        model: str,
        config: MemAgentRuntimeConfig,
        *,
        api_key: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_completions_url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.config = config
        self.api_key = api_key
        self.http_session = http_session

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[aiohttp.ClientSession]:
        if self.http_session is not None:
            yield self.http_session
            return
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            yield session

    async def _chat_completion(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict],
        sampling_params: dict[str, Any],
    ) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            **sampling_params,
        }
        headers = {}
        if self.api_key is not None:
            headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "api-key": self.api_key,
                }
            )
        async with session.post(self.chat_completions_url, json=payload, headers=headers or None) as response:
            response.raise_for_status()
            return await response.json()

    async def run(
        self,
        question: str,
        context: str,
        sampling_params: dict[str, Any] | None = None,
    ) -> MemAgentRunResult:
        """Run chunk-wise memory updates followed by one final answer turn."""
        async with self._session() as session:
            return await self._run(session, question, context, sampling_params)

    async def _run(
        self,
        session: aiohttp.ClientSession,
        question: str,
        context: str,
        sampling_params: dict[str, Any] | None,
    ) -> MemAgentRunResult:
        question = question.strip()
        context = context.strip()

        context_ids = await asyncio.to_thread(
            self.tokenizer.encode,
            context,
            add_special_tokens=False,
        )
        context_token_count = len(context_ids)
        max_context_tokens = self.config.chunk_tokens * self.config.max_chunks
        context_truncated = len(context_ids) > max_context_tokens
        if context_truncated and not self.config.allow_context_truncation:
            raise ValueError(
                f"MemAgent context has {len(context_ids)} tokens, exceeding the configured "
                f"capacity {max_context_tokens} ({self.config.max_chunks} chunks x "
                f"{self.config.chunk_tokens})."
            )
        context_ids = context_ids[:max_context_tokens]

        memory = self.config.no_memory
        turns: list[MemAgentTurn] = []
        base_sampling_params = dict(sampling_params or {})
        for chunk_index, start in enumerate(range(0, len(context_ids), self.config.chunk_tokens)):
            chunk = await asyncio.to_thread(
                self.tokenizer.decode,
                context_ids[start : start + self.config.chunk_tokens],
                skip_special_tokens=True,
            )
            response = await self._chat_completion(
                session,
                [
                    {
                        "role": "user",
                        "content": self.config.memory_template.format(
                            prompt=question,
                            memory=memory,
                            chunk=chunk,
                        ),
                    }
                ],
                {**base_sampling_params, "max_tokens": self.config.max_memory_tokens},
            )
            choice = response["choices"][0]
            response_text = str((choice.get("message") or {}).get("content") or "")
            for stop_token in self.config.stop_token_strings:
                response_text = response_text.replace(stop_token, "")
            response_text = response_text.strip()
            if response_text:
                memory = response_text
            finish_reason = choice.get("finish_reason") or "stop"
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("type", "stop")
            turns.append(
                MemAgentTurn(
                    kind="memory",
                    response=response_text,
                    finish_reason=str(finish_reason),
                    chunk_index=chunk_index,
                )
            )

        response = await self._chat_completion(
            session,
            [
                {
                    "role": "user",
                    "content": self.config.final_template.format(
                        prompt=question,
                        memory=memory,
                    ),
                }
            ],
            {**base_sampling_params, "max_tokens": self.config.max_final_tokens},
        )
        choice = response["choices"][0]
        final_response = str((choice.get("message") or {}).get("content") or "")
        for stop_token in self.config.stop_token_strings:
            final_response = final_response.replace(stop_token, "")
        final_response = final_response.strip()
        finish_reason = choice.get("finish_reason") or "stop"
        if isinstance(finish_reason, dict):
            finish_reason = finish_reason.get("type", "stop")
        turns.append(
            MemAgentTurn(
                kind="final",
                response=final_response,
                finish_reason=str(finish_reason),
            )
        )

        return MemAgentRunResult(
            turns=turns,
            final_response=final_response,
            final_memory=memory,
            context_tokens=context_token_count,
            chunks_processed=len(turns) - 1,
            context_truncated=context_truncated,
        )
