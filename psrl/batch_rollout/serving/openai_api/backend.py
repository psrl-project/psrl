"""Serve batch rollout from an OpenAI-compatible endpoint PSRL does not own.

Two things this covers, with no code difference between them:

- a hosted API (Novita, OpenRouter, or a subscription-to-API proxy), needing no
  GPU at all, and
- a local vLLM fleet started separately by `python -m psrl.eval.serve`, read back
  from its `endpoints.json`.

Session-scoped agent loops hand an external agent a per-episode URL and read the
session back afterwards, which a plain OpenAI endpoint cannot provide. A local
`OpenAISessionAdapter` supplies that surface. The endpoint itself is never
launched or torn down here: PSRL did not start it and must not stop it.
"""

import logging
import os

import ray
from omegaconf import DictConfig

from psrl.batch_rollout.serving.base import BackendHandle, ServingBackend
from psrl.batch_rollout.serving.openai_api.openai_session_adapter import launch_session_adapter
from psrl.eval.vllm_fleet import read_endpoints

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class OpenAIServingBackend(ServingBackend):
    """Point the agent loops at an already-running OpenAI-compatible endpoint."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self._proxy_thread = None

    @staticmethod
    def _resolve_api_key(serving) -> str:
        """Resolve the upstream bearer token, preferring an env var over config.

        `api_key_env` names a variable to read. That indirection exists because a
        Hydra override lands in the shell history, the Ray dashboard, and the run
        log, none of which should hold a credential. A literal `api_key` still
        works for a local server that ignores the token.

        Args:
            serving: The `batch_rollout.serving` config node.

        Returns:
            str: The bearer token to forward upstream.

        Raises:
            ValueError: If the named variable is unset or empty.
        """
        env_name = str(serving.get("api_key_env", "") or "")
        if env_name:
            value = os.environ.get(env_name, "")
            if not value:
                raise ValueError(
                    f"batch_rollout.serving.api_key_env={env_name!r} but that variable is unset. "
                    f"Export it before launching, for example `export {env_name}=...`."
                )
            return value
        return str(serving.get("api_key", "EMPTY"))

    def start(self) -> BackendHandle:
        """Resolve the endpoint, then front it with a session proxy.

        Returns:
            BackendHandle: Where to reach the endpoint and its session surface.

        Raises:
            ValueError: If neither an explicit URL nor an `endpoints.json` is given.
        """
        serving = self.config.batch_rollout.serving
        api_base_url = str(serving.get("api_base_url", "") or "")
        model_name = str(serving.get("model_name", "") or "")

        endpoints_file = str(serving.get("endpoints_file", "") or "")
        if endpoints_file:
            fleet_model_name, urls = read_endpoints(endpoints_file)
            if not urls:
                raise ValueError(
                    f"{endpoints_file} lists no healthy endpoint, so there is nothing to roll out against."
                )
            # One URL only. Balancing across a fleet is the gateway's job, and the
            # `smg_local` backend is what provides it.
            api_base_url = urls[0]
            model_name = model_name or fleet_model_name
            if len(urls) > 1:
                psrl_logger.warning(
                    "%s lists %d endpoints but this backend dispatches to one (%s). "
                    "Use serving=smg_local to spread load across replicas.",
                    endpoints_file,
                    len(urls),
                    api_base_url,
                )

        if not api_base_url:
            raise ValueError(
                "Set batch_rollout.serving.api_base_url (an OpenAI-compatible /v1 URL) "
                "or batch_rollout.serving.endpoints_file (an endpoints.json written by psrl.eval.serve)."
            )
        if not model_name:
            # The agent sends this as the model field, and the record is stamped with
            # it. Falling back to the checkpoint path keeps the dump attributable.
            model_name = str(self.config.gen_actor_rollout_ref.model.path)

        api_base_url = api_base_url.rstrip("/")

        # The agent containers reach the proxy over the network, so it must bind an
        # address they can route to rather than loopback.
        host = ray.util.get_node_ip_address().strip("[]")
        transcript_dir = os.path.join(
            os.path.abspath(os.path.expanduser(str(self.config.batch_rollout.output_dir))),
            "transcripts",
        )
        session_url, self._proxy_thread = launch_session_adapter(
            upstream_url=api_base_url,
            transcript_dir=transcript_dir,
            host=host,
            api_key=self._resolve_api_key(serving),
            model_name=model_name,
            client_concurrency=int(serving.get("client_concurrency", 256)),
            request_timeout_s=float(serving.get("request_timeout_s", 1800.0)),
        )

        psrl_logger.info(
            "Batch rollout will call %s as model %r, through session proxy %s.",
            api_base_url,
            model_name,
            session_url,
        )
        return BackendHandle(
            api_base_url=api_base_url,
            model_name=model_name,
            session_router_url=session_url,
            # The upstream returns text, so any token field would be reconstructed
            # rather than measured.
            supports_token_capture=False,
        )
