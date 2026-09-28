"""Serve batch rollout from an existing OpenAI-compatible endpoint.

`backend` owns the lifecycle the `ServingBackend` contract expects.
`openai_session_adapter` supplies the session-scoped surface that
`SessionAgentLoop` requires but a plain chat-completions endpoint does not have.
"""

from psrl.batch_rollout.serving.openai_api.backend import OpenAIServingBackend

__all__ = ["OpenAIServingBackend"]
