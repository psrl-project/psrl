"""Training smoke loop that produces a main and one subagent trajectory.

This is a verification harness, not a product loop. It drives the real SMG
SessionRouter with two main turns and two subagent turns, then returns the
selected rows as normal TokenOutputs so the trainer exercises the same
TransferQueue, advantage, and loss path as a harness episode.
"""

from __future__ import annotations

import json

from psrl.utils.common.http_utils import post
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason, register


@register("subagent_smoke")
class SubagentSmokeLoop(SessionAgentLoop):
    """Produce one main and one subagent row for a trainer smoke test."""

    def __init__(self, context, **kwargs):
        del kwargs  # Hydra injects the registry name; the loop has no use for it.
        super().__init__(context=context)

    async def run(self, request: dict):
        session_id = await self.create_session(request)
        try:
            params = self.get_session_sampling_params(request)
            base = list(request.get("raw_prompt") or [{"role": "user", "content": "Name one color."}])

            async def turn(messages: list[dict], metadata: dict) -> dict:
                payload = {"model": self.model_config.path, "messages": messages, **params}
                return await post(
                    f"{self.session_api_url(session_id)}/chat/completions",
                    payload=payload,
                    headers={"x-codex-turn-metadata": json.dumps(metadata)},
                )

            main_meta = {"thread_source": "user", "thread_id": "main-1"}
            first = await turn(base, main_meta)
            assistant = first["choices"][0]["message"]
            await turn(
                base + [assistant, {"role": "user", "content": "Add one more detail."}],
                main_meta,
            )

            sub_meta = {
                "thread_source": "subagent",
                "thread_id": "sub-1",
                "parent_thread_id": "main-1",
            }
            sub_messages = base + [assistant, {"role": "user", "content": "Answer as a reviewer."}]
            sub_first = await turn(sub_messages, sub_meta)
            sub_assistant = sub_first["choices"][0]["message"]
            await turn(
                sub_messages + [sub_assistant, {"role": "user", "content": "Make it shorter."}],
                sub_meta,
            )

            training_data = await self.get_training_data(session_id)
            if not training_data:
                raise RuntimeError("subagent smoke loop produced no training rows")
            outputs = [self.build_token_output(item) for item in training_data]
            self.attach_tito_tree_metadata(training_data, outputs)
            reward = float(int(request.get("uid", 0)) % 2)
            for output in outputs:
                output.reward_score = reward
            outputs.sort(key=lambda output: bool(output.extra_fields.get("session_primary")))
            return (outputs[0] if len(outputs) == 1 else outputs), TerminateReason.FINISHED
        finally:
            await self.delete_session(session_id)
