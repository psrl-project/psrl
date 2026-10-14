import logging
import multiprocessing
import os
import time

import httpx
import ray
from omegaconf import DictConfig

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.logger import DualOutputHandler
from psrl.workers.gen.rollout_gateway import _run_smg
from psrl.workers.gen.smg_adapter import build_reward_router_args, cfg_get
from psrl.workers.gen.smg_control import wait_for_router_ready

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_ROUTER_STARTUP_TIMEOUT_S = 120.0
_STARTUP_POLL_INTERVAL_S = 0.1


@ray.remote
class RewardModelGateway:
    """Run a dedicated SMG routing queue for one reward model."""

    def __init__(self, config: DictConfig, rm_config: DictConfig) -> None:
        self.config = config
        self.rm_config = rm_config
        self.model_name = rm_config.reward_model_name
        self.smg_ip: str | None = None
        self.smg_port: int | None = None
        self.smg_url: str | None = None
        self.router_process: multiprocessing.Process | None = None

        self.log_prefix = f"RewardModelGateway-{self.model_name}"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(f"Initialized RewardModelGateway for model_name={self.model_name!r}.")

    def _init_router_args(self):
        """Build the CLI args namespace passed to smg RouterArgs."""
        return build_reward_router_args(
            self.config,
            self.smg_ip,
            self.smg_port,
            prometheus_port=find_available_port(base_port=4100),
            rm_config=self.rm_config,
        )

    def launch_router(self) -> str:
        """Start the smg subprocess and return the gateway HTTP URL."""
        if self.smg_url is not None:
            return self.smg_url

        self.smg_ip = ray.util.get_node_ip_address().strip("[]")
        self.smg_port = find_available_port(base_port=8300)  # avoids rollout gateway ports 8100 and 8200

        router_args = self._init_router_args()

        self.router_process = multiprocessing.Process(
            target=_run_smg,
            args=(router_args,),
            daemon=True,
        )
        self.router_process.start()
        gateway_url = f"http://{self.smg_ip}:{self.smg_port}"
        try:
            wait_for_router_ready(
                gateway_url,
                self.router_process,
                require_elastic_runtime=bool(
                    cfg_get(
                        self.config,
                        "psrl.deployment.elastic.throughput_optimal.enable_request_level_candidate_evaluation",
                        False,
                    )
                ),
            )
            self._initialize_tokenizer(gateway_url, router_args)
        except Exception:
            self.shutdown_router()
            raise

        self.smg_url = gateway_url
        psrl_logger.info(f"RewardModelGateway for {self.model_name!r} launched at {self.smg_url!r}.")
        return self.smg_url

    def _initialize_tokenizer(self, gateway_url: str, router_args) -> None:
        """
        Register the tokenizer by model name and wait for its native loading job.
        """
        deadline = time.monotonic() + _ROUTER_STARTUP_TIMEOUT_S
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            response = client.post(
                f"{gateway_url}/v1/tokenizers",
                json={
                    "name": self.model_name,
                    "source": router_args.tokenizer_path,
                    "chat_template_path": router_args.chat_template,
                },
            )
            response.raise_for_status()
            tokenizer_id = response.json()["id"]

            while True:
                self._check_startup_deadline(deadline)
                response = client.get(f"{gateway_url}/v1/tokenizers/{tokenizer_id}/status")
                response.raise_for_status()
                status = response.json()
                if status["status"] == "failed":
                    raise RuntimeError(
                        f"Reward tokenizer {self.model_name!r} failed to load from "
                        f"{router_args.tokenizer_path!r}: {status['message']}"
                    )
                if status["status"] == "completed":
                    response = client.get(f"{gateway_url}/v1/tokenizers/{tokenizer_id}")
                    response.raise_for_status()
                    tokenizer = response.json()
                    if (
                        tokenizer["name"] != self.model_name
                        or tokenizer["source"] != router_args.tokenizer_path
                        or tokenizer["vocab_size"] <= 0
                    ):
                        raise RuntimeError(f"Reward gateway registered an unexpected tokenizer: {tokenizer!r}.")
                    psrl_logger.info(
                        f"Reward tokenizer {self.model_name!r} loaded from {tokenizer['source']!r} "
                        f"with vocab_size={tokenizer['vocab_size']}."
                    )
                    return
                time.sleep(_STARTUP_POLL_INTERVAL_S)

    def _check_startup_deadline(self, deadline: float) -> None:
        """
        Fail initialization if the router exits or its tokenizer loading times out.
        """
        if not self.router_process.is_alive():
            raise RuntimeError(f"Reward gateway for {self.model_name!r} exited during initialization.")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Reward gateway tokenizer initialization timed out for {self.model_name!r}.")

    def shutdown_router(self) -> None:
        """Terminate the smg subprocess if running."""
        if self.router_process is not None and self.router_process.is_alive():
            self.router_process.terminate()
            self.router_process.join()
            psrl_logger.info("RewardModelGateway for '%s' shut down.", self.model_name)
