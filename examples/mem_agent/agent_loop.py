from examples.mem_agent.config import MemAgentRuntimeConfig, build_runtime_config
from examples.mem_agent.runner import MemAgent
from omegaconf import DictConfig
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.session_agent_loop import SessionAgentLoop, SessionAgentResult
from psrl.workers.agent_loop.loops.utils import TerminateReason, register


@register("mem_agent")
class MemAgentAgentLoop(SessionAgentLoop):
    """Run MemAgent through a session-scoped OpenAI-compatible API URL."""

    def __init__(
        self,
        context: AgentLoopContext,
        runtime: DictConfig | dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(context=context)
        self.runtime_config: MemAgentRuntimeConfig = build_runtime_config(runtime)
        if self.max_turns < self.runtime_config.max_chunks + 1:
            raise ValueError(
                "rollout.multi_turn.max_turns must cover max_chunks plus the final answer turn "
                f"({self.max_turns} < {self.runtime_config.max_chunks + 1})."
            )
        if int(self.rollout_config.response_length) < self.runtime_config.max_memory_tokens:
            raise ValueError(
                "rollout.response_length must be at least runtime.max_memory_tokens "
                f"({self.rollout_config.response_length} < {self.runtime_config.max_memory_tokens})."
            )

    def get_generate_fields(self) -> list[str]:
        """Include the long document consumed by MemAgent."""
        return [*super().get_generate_fields(), "context"]

    async def run_session(
        self,
        request: dict,
        api_base_url: str,
    ) -> SessionAgentResult:
        """Run one MemAgent episode from the native HotpotQA row contract."""
        messages = request["raw_prompt"]
        question = messages[-1]["content"]
        context = request["context"]

        result = await MemAgent(
            tokenizer=self.tokenizer,
            base_url=api_base_url,
            model=self.model_config.path,
            config=self.runtime_config,
            api_key="EMPTY",
        ).run(
            question,
            context,
            sampling_params=self.get_session_sampling_params(request),
        )

        turn_metadata = [
            {
                "mem_agent_turn_index": index,
                "mem_agent_turn_kind": turn.kind,
                "mem_agent_chunk_index": turn.chunk_index,
                "mem_agent_finish_reason": turn.finish_reason,
                "mem_agent_context_tokens": result.context_tokens,
                "mem_agent_chunks_processed": result.chunks_processed,
                "mem_agent_context_truncated": result.context_truncated,
            }
            for index, turn in enumerate(result.turns)
        ]
        return SessionAgentResult(
            extra_fields={
                "mem_agent_final_output": result.final_response,
                "mem_agent_final_memory": result.final_memory,
                "mem_agent_turns": turn_metadata,
                "num_turns": len(result.turns),
            },
            terminate_reason=(
                TerminateReason.MAX_RESPONSE_LENGTH_EXCEEDED
                if result.turns[-1].finish_reason == "length"
                else TerminateReason.FINISHED
            ),
        )
