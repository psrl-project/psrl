"""HTTP wire-protocol renderers for harness responses.

SMG owns protocol conversion and TITO capture. SessionRouter only buffers one
completed turn and renders that native response using the streaming wire shape
expected by the CLI.
"""

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from fastapi.responses import JSONResponse, Response, StreamingResponse

from psrl.utils.common.http_utils import HttpResponse

SSE_HEADERS = {
    "cache-control": "no-cache",
    "connection": "keep-alive",
    "x-accel-buffering": "no",
}
ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "invalid_request_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
}


def anthropic_error_body(status_code: int, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": ANTHROPIC_ERROR_TYPES.get(status_code, "api_error"),
            "message": message,
        },
    }


def anthropic_error_response(result: HttpResponse) -> Response:
    """Convert an upstream failure into an Anthropic error response."""
    message = f"backend request failed with status {result.status}"
    try:
        body = result.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            message = str(error.get("message") or message)
        elif error:
            message = str(error)
    except (json.JSONDecodeError, UnicodeDecodeError):
        if result.body:
            message = result.body.decode(errors="replace")

    return JSONResponse(
        status_code=result.status,
        content=anthropic_error_body(result.status, message),
        headers={
            key: value
            for key, value in result.headers.items()
            if key.lower() not in {"content-type", "content-length", "content-encoding"}
        },
    )


def encode_sse_event(name: str, data: Mapping[str, Any]) -> bytes:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {name}\ndata: {payload}\n\n".encode()


def build_streaming_response(
    body: AsyncIterator[bytes],
    upstream_headers: Mapping[str, str],
) -> StreamingResponse:
    headers = {
        key: value
        for key, value in upstream_headers.items()
        if key.lower()
        not in {
            "content-type",
            "content-length",
            "content-encoding",
            "transfer-encoding",
            "connection",
            "cache-control",
        }
    }
    headers.update(SSE_HEADERS)
    return StreamingResponse(body, media_type="text/event-stream", headers=headers)


def anthropic_stream_response(
    message: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
) -> StreamingResponse:
    """Render a completed native Anthropic Message as protocol-valid SSE."""

    async def events() -> AsyncIterator[bytes]:
        usage = message["usage"]
        yield encode_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    **message,
                    "content": [],
                    "stop_reason": None,
                    "usage": {
                        "input_tokens": usage["input_tokens"],
                        "output_tokens": 0,
                    },
                },
            },
        )
        for index, block in enumerate(message["content"]):
            block_type = block["type"]
            if block_type == "thinking":
                start = {"type": "thinking", "thinking": "", "signature": ""}
                delta = {"type": "thinking_delta", "thinking": block["thinking"]}
            elif block_type == "text":
                start = {"type": "text", "text": ""}
                delta = {"type": "text_delta", "text": block["text"]}
            elif block_type == "tool_use":
                start = {
                    "type": "tool_use",
                    "id": block["id"],
                    "name": block["name"],
                    "input": {},
                }
                delta = {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(
                        block["input"],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            else:
                raise ValueError(f"unsupported Anthropic content block: {block_type}")

            yield encode_sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": start,
                },
            )
            yield encode_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": index,
                    "delta": delta,
                },
            )
            if block_type == "thinking":
                yield encode_sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "signature_delta",
                            "signature": block["signature"],
                        },
                    },
                )
            yield encode_sse_event(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )

        yield encode_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": message["stop_reason"],
                    "stop_sequence": message["stop_sequence"],
                },
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
        yield encode_sse_event("message_stop", {"type": "message_stop"})

    return build_streaming_response(events(), headers)


def openai_stream_response(
    response: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
) -> StreamingResponse:
    """Render a completed Chat Completion as Chat Completions SSE."""
    choice = response["choices"][0]
    message = choice["message"]

    def chunk(
        delta: dict[str, Any],
        finish_reason: Any = None,
        usage: Any = None,
    ) -> bytes:
        payload: dict[str, Any] = {
            "id": response["id"],
            "object": "chat.completion.chunk",
            "created": response["created"],
            "model": response["model"],
            "choices": [
                {
                    "index": choice["index"],
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n".encode()

    async def events() -> AsyncIterator[bytes]:
        yield chunk({"role": "assistant"})
        reasoning = message.get("reasoning_content", message.get("reasoning"))
        if reasoning:
            yield chunk({"reasoning_content": reasoning})
        if message.get("content"):
            yield chunk({"content": message["content"]})
        if message.get("tool_calls"):
            yield chunk(
                {
                    "tool_calls": [
                        {**tool_call, "index": index} for index, tool_call in enumerate(message["tool_calls"])
                    ]
                }
            )
        yield chunk({}, choice["finish_reason"], response.get("usage"))
        yield b"data: [DONE]\n\n"

    return build_streaming_response(events(), headers)


def responses_stream_response(
    completed: Mapping[str, Any],
    *,
    headers: Mapping[str, str],
) -> StreamingResponse:
    """Render a completed native Responses object as protocol-valid SSE."""

    async def events() -> AsyncIterator[bytes]:
        sequence = 0

        def response_event(event_type: str, **fields: Any) -> bytes:
            nonlocal sequence
            sequence += 1
            return encode_sse_event(
                event_type,
                {
                    "type": event_type,
                    **fields,
                    "sequence_number": sequence,
                },
            )

        in_progress = {
            **completed,
            "status": "in_progress",
            "completed_at": None,
            "output": [],
        }
        yield response_event("response.created", response=in_progress)
        yield response_event("response.in_progress", response=in_progress)

        for output_index, final_item in enumerate(completed["output"]):
            item_type = final_item["type"]
            if item_type == "message":
                item = {**final_item, "status": "in_progress", "content": []}
                yield response_event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=item,
                )
                for content_index, final_part in enumerate(final_item["content"]):
                    part = {**final_part, "text": ""}
                    yield response_event(
                        "response.content_part.added",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=content_index,
                        part=part,
                    )
                    text = final_part["text"]
                    yield response_event(
                        "response.output_text.delta",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=content_index,
                        delta=text,
                        logprobs=[],
                    )
                    yield response_event(
                        "response.output_text.done",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=content_index,
                        text=text,
                        logprobs=[],
                    )
                    yield response_event(
                        "response.content_part.done",
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=content_index,
                        part=final_part,
                    )
                yield response_event(
                    "response.output_item.done",
                    output_index=output_index,
                    item=final_item,
                )
            elif item_type == "function_call":
                item = {**final_item, "status": "in_progress", "arguments": ""}
                yield response_event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=item,
                )
                yield response_event(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=output_index,
                    delta=final_item["arguments"],
                )
                yield response_event(
                    "response.function_call_arguments.done",
                    item_id=item["id"],
                    output_index=output_index,
                    arguments=final_item["arguments"],
                    name=final_item["name"],
                )
                yield response_event(
                    "response.output_item.done",
                    output_index=output_index,
                    item=final_item,
                )
            else:
                yield response_event(
                    "response.output_item.added",
                    output_index=output_index,
                    item=final_item,
                )
                yield response_event(
                    "response.output_item.done",
                    output_index=output_index,
                    item=final_item,
                )

        yield response_event("response.completed", response=completed)

    return build_streaming_response(events(), headers)
