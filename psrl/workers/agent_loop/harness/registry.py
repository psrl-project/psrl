"""Harness adapter registry."""

from collections.abc import Callable

from psrl.sandbox import SandboxSession
from psrl.workers.agent_loop.harness.base import Harness, HarnessConfig
from psrl.workers.agent_loop.harness.claude_code import ClaudeCodeHarness
from psrl.workers.agent_loop.harness.codex import CodexHarness

HarnessFactory = Callable[[HarnessConfig, SandboxSession], Harness]

_HARNESS_FACTORIES: dict[str, HarnessFactory] = {
    "claude_code": ClaudeCodeHarness,
    "codex": CodexHarness,
}


def register_harness(kind: str) -> Callable[[HarnessFactory], HarnessFactory]:
    """Register an out-of-tree harness adapter without changing the agent loop."""
    if not kind:
        raise ValueError("Harness kind cannot be empty.")

    def decorator(factory: HarnessFactory) -> HarnessFactory:
        if kind in _HARNESS_FACTORIES:
            raise ValueError(f"Harness kind {kind!r} is already registered.")
        _HARNESS_FACTORIES[kind] = factory
        return factory

    return decorator


def create_harness(config: HarnessConfig, sandbox: SandboxSession) -> Harness:
    """Create a fresh task-owned harness adapter."""
    try:
        factory = _HARNESS_FACTORIES[config.kind]
    except KeyError as exc:
        choices = ", ".join(sorted(_HARNESS_FACTORIES))
        raise ValueError(f"Unknown harness kind {config.kind!r}; expected one of: {choices}.") from exc
    return factory(config, sandbox)
