from importlib import import_module

__all__ = [
    "GenerateAgentLoop",
    "MLGymAgentLoop",
    "MultiTurnAgentLoop",
    "MultiTurnCompletionAgentLoop",
    "MiniSWEAgentLoopV1",
    "SessionAgentLoop",
    "SessionAgentResult",
]

_LAZY_EXPORTS = {
    "GenerateAgentLoop": (".generate_agent_loop", "GenerateAgentLoop"),
    "MLGymAgentLoop": (".mlgym_agent_loop", "MLGymAgentLoop"),
    "MultiTurnAgentLoop": (".multi_turn_agent_loop", "MultiTurnAgentLoop"),
    "MultiTurnCompletionAgentLoop": (".multi_turn_completion_agent_loop", "MultiTurnCompletionAgentLoop"),
    "MiniSWEAgentLoopV1": (".mini_swe_agent_loop_v1", "MiniSWEAgentLoopV1"),
    "SessionAgentLoop": (".session_agent_loop", "SessionAgentLoop"),
    "SessionAgentResult": (".session_agent_loop", "SessionAgentResult"),
}


def __getattr__(name: str):
    """Load individual loop implementations on demand."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
