"""A session-scoped agent loop with no environment, for smoke tests.

Exercises the same path `SciAccelAgentLoop` takes (open a session, drive turns
through it, read the trajectory back, build outputs) without Docker, Harbor, or a
task image. Registered as `stub_session` so a smoke run selects it with
`gen_actor_rollout_ref.rollout.agent.default_agent_loop=stub_session`.
"""

import logging
import os

from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger("psrl.tests.stub_agent_loop")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("stub_session")
class StubSessionAgentLoop(SessionAgentLoop):
    """Drive a fixed number of turns through a session and return the trajectory."""

    def __init__(self, context: AgentLoopContext, turns: int = 2, **kwargs):
        super().__init__(context=context)
        self.turns = int(turns)

    async def run(self, request: dict) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Take `turns` conversational turns, then report what the session captured."""
        uid = request.get("uid", "?")
        session_id = await self.create_session(request)
        try:
            messages = [{"role": "user", "content": "start"}]
            sampling_params = self.get_session_sampling_params(request)
            for turn in range(self.turns):
                response = await self.chat_completion(session_id, messages, sampling_params, trajectory_id=0)
                content = response["choices"][0]["message"]["content"]
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"continue {turn}"})

            training_data = await self.get_training_data(session_id)
            num_turns = sum(item["num_turns"] for item in training_data)
            psrl_logger.info("[uid=%s] stub episode captured %d turn(s).", uid, num_turns)
            if num_turns == 0:
                return None, TerminateReason.ROLLOUT_ERROR

            outputs = [
                self.build_token_output(item, extra_fields={"stub": True, "final_message": messages[-2]["content"]})
                for item in training_data
            ]
            return (outputs[0] if len(outputs) == 1 else outputs), TerminateReason.FINISHED
        finally:
            await self.delete_session(session_id)
