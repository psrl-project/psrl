"""Multi-turn completion agent loop backed by SMG SessionRouter and TITO."""

from __future__ import annotations

import asyncio

from psrl.environments.base import Environment
from psrl.utils.common.http_utils import PromptOverflowError
from psrl.utils.rollout.rollout_trace import rollout_trace_op
from psrl.workers.agent_loop.agent_data import AgentData
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop
from psrl.workers.agent_loop.loops.utils import TerminateReason, register
from psrl.workers.gen.utils import TokenOutput


@register("multi_turn_completion_agent")
class MultiTurnCompletionAgentLoop(SessionAgentLoop):
    """Drive an environment through session-scoped chat completions."""

    @rollout_trace_op
    async def run(self, request: dict) -> tuple[TokenOutput | None, TerminateReason]:
        env_class = request.get("env_class", self.config.gen_actor_rollout_ref.rollout.agent.env.name)
        data_class = request.get("data_class", self.config.gen_actor_rollout_ref.rollout.agent.data.name)
        env = Environment.get_environment(
            env_class,
            self.config,
            self.reward_manager,
            self.max_turns,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
        )
        agent_data = AgentData.get_agent_data(data_class, self.config, self.reward_manager, env)
        agent_data.reset()
        observation, info = await env.reset(task=request, seed=request.get("seed"))
        agent_data.init_trajectory(request)

        session_id: str | None = None
        try:
            session_id = await self.create_session(request)
            sampling_params = self.get_session_sampling_params(request)
            if await agent_data.update_from_env(observation, 0, False, info):
                return await agent_data.finalize_output(), TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED

            terminate_reason = TerminateReason.FINISHED
            done = False
            for _ in range(self.max_turns):
                messages, tools = agent_data.prepare_chat_completion_request()
                try:
                    response = await self.chat_completion(
                        session_id,
                        messages,
                        sampling_params,
                        tools=tools,
                        chat_template_kwargs=agent_data.apply_chat_template_kwargs,
                        multi_modal_data=agent_data.prepare_chat_completion_multimodal_data(),
                    )
                except PromptOverflowError:
                    terminate_reason = TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
                    break
                action, overlong = await agent_data.update_from_model_chat_completion(response)
                if overlong:
                    terminate_reason = TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
                    break

                try:
                    with self.timer.env():
                        env_step_output = await asyncio.wait_for(
                            env.step(action),
                            timeout=self.config.gen_actor_rollout_ref.rollout.agent.env.step_timeout,
                        )
                except asyncio.TimeoutError:
                    terminate_reason = TerminateReason.ENV_TIMEOUT
                    break

                observation = env_step_output["observation"]
                done = env_step_output["done"]
                if await agent_data.update_from_env(
                    observation,
                    env_step_output["reward"],
                    done,
                    env_step_output["info"],
                ):
                    terminate_reason = TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
                    break
                if done:
                    break

            if terminate_reason == TerminateReason.FINISHED and not done:
                terminate_reason = TerminateReason.MAX_TURNS_EXCEEDED

            training_data = await self.get_primary_training_data(session_id)
            self.attach_training_data(agent_data, training_data)
            return await agent_data.finalize_output(), terminate_reason
        finally:
            try:
                await env.close()
            finally:
                if session_id is not None:
                    await self.delete_session(session_id)
