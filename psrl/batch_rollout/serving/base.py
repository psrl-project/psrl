"""Serving backends for batch rollout.

A backend is whatever answers the agent loops' model calls. Two exist: an external
OpenAI-compatible endpoint, and the full local SMG plus SessionRouter stack over
PSRL vLLM replicas. Both hand back a `BackendHandle`, which is all the workers
need to reach the model, so the rest of batch rollout is indifferent to which is
running.

Follows the `StepStrategy` / `build_step_strategy` convention in
`psrl/trainer/ppo/strategies/`.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from omegaconf import DictConfig

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass(frozen=True)
class BackendHandle:
    """How the agent loop workers reach the served model."""

    api_base_url: str
    """OpenAI-compatible base URL, ending in `/v1`."""

    model_name: str
    """Model name the agent sends, and that the record is stamped with."""

    session_router_url: str | None = None
    """SessionRouter base URL. `None` when the backend has no session layer, which
    leaves session-scoped loops unavailable."""

    rollout_gateway_url: str = ""
    """SMG gateway base URL, for loops that call `/generate` directly."""

    supports_token_capture: bool = False
    """Whether the backend returns token ids and masks. Only a TITO-backed local
    stack does, so `dump_tokens` is inert without it."""


class ServingBackend(ABC):
    """Bring a model endpoint up and down for one batch rollout run."""

    def __init__(self, config: DictConfig) -> None:
        """
        Args:
            config (DictConfig): The composed batch rollout configuration.
        """
        self.config = config

    @abstractmethod
    def start(self) -> BackendHandle:
        """Bring the endpoint up and return how to reach it."""

    def stop(self) -> None:  # noqa: B027
        """Tear the endpoint down.

        Not abstract: a backend that points at an endpoint PSRL did not start must
        not stop it, so doing nothing is the correct implementation there.
        """


def build_serving_backend(config: DictConfig) -> ServingBackend:
    """Construct the serving backend named by `batch_rollout.serving.name`.

    Args:
        config (DictConfig): The composed batch rollout configuration.

    Returns:
        ServingBackend: The selected backend, not yet started.

    Raises:
        ValueError: If the configured name is not a known backend.
    """
    name = str(config.batch_rollout.serving.name).lower()

    if name == "openai_api":
        from psrl.batch_rollout.serving.openai_api import OpenAIServingBackend

        return OpenAIServingBackend(config)
    if name == "smg_local":
        from psrl.batch_rollout.serving.smg_local import SMGLocalServingBackend

        return SMGLocalServingBackend(config)

    raise ValueError(f"Unknown serving backend {name!r}. Expected one of: openai_api, smg_local.")
