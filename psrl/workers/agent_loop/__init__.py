from importlib import import_module

__all__ = [
    "PSRL_AgentLoopManager",
    "PSRL_AgentLoopWorker",
]


def __getattr__(name: str):
    """Load Ray-backed agent-loop components only when requested."""
    module_name = {"PSRL_AgentLoopManager": ".manager", "PSRL_AgentLoopWorker": ".worker"}.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
