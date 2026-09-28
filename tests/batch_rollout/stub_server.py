"""A minimal OpenAI-compatible server for batch rollout smoke tests.

Answers `/v1/chat/completions` with a canned reply and a plausible usage block,
so the collection pipeline can be exercised end to end with no GPU and no model.

    python -m tests.batch_rollout.stub_server --port 9999
"""

import argparse
import json

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    payload = json.loads(await request.body())
    messages = payload.get("messages") or []
    turn = sum(1 for m in messages if m.get("role") == "assistant")
    return JSONResponse(
        content={
            "id": f"stub-{turn}",
            "object": "chat.completion",
            "model": payload.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": f"stub reply {turn}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 16 * (turn + 1),
                "completion_tokens": 8,
                "total_tokens": 16 * (turn + 1) + 8,
            },
        }
    )


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse(content={"data": [{"id": "stub"}]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
