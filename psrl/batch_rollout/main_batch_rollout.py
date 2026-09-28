"""Entry point for offline agentic rollout collection.

The third PSRL entry point, alongside `psrl.trainer.main_ppo` for training. It
drives the same agent loops against a fixed model and writes the resulting
trajectories to disk, with no training, no reward, and no evaluation.

    python -m psrl.batch_rollout.main_batch_rollout \\
        serving=openai_api \\
        batch_rollout.serving.api_base_url=http://127.0.0.1:8000/v1 \\
        data.train_files=/path/to/train.parquet \\
        batch_rollout.output_dir=outputs/rollout
"""

import logging
import os
import socket
import time

import hydra
import ray
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from psrl.batch_rollout.output_writer import RolloutOutputWriter
from psrl.batch_rollout.serving.base import build_serving_backend
from psrl.trainer.constants_ppo import get_ppo_ray_runtime_env
from psrl.utils.dataset.batch_rollout_data_processor import BatchRolloutDataProcessor
from psrl.workers.agent_loop.batch_rollout_manager import BatchRolloutAgentLoopManager
from psrl.workers.agent_loop.batch_rollout_worker import BatchRolloutAgentLoopWorker

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _validate_rollout_budget(config) -> None:
    """Reject a null `max_model_len` before any episode pays for it.

    veRL leaves it null so a vLLM engine can read the window off the model. No
    engine is involved when serving from an external API, and agent harnesses pass
    it straight to `int()`, so a null surfaces an hour into a run as an opaque
    `TypeError` from inside the loop rather than as a config error here.

    Args:
        config: The composed batch rollout configuration.

    Raises:
        ValueError: If `max_model_len` is unset.
    """
    rollout = config.gen_actor_rollout_ref.rollout
    if rollout.get("max_model_len") is not None:
        return
    prompt_length = rollout.get("prompt_length")
    response_length = rollout.get("response_length")
    raise ValueError(
        "gen_actor_rollout_ref.rollout.max_model_len is null. Batch rollout serves a "
        "model it does not own, so the context window cannot be inferred and the agent "
        "harness needs it as a turn budget. Set it explicitly, for example "
        f"`gen_actor_rollout_ref.rollout.max_model_len={(prompt_length or 0) + (response_length or 0)}` "
        f"(prompt_length={prompt_length} + response_length={response_length})."
    )


def _as_plain(value):
    """Convert an OmegaConf node to plain Python, passing scalars through."""
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


@hydra.main(config_path="config", config_name="batch_rollout", version_base=None)
def main(config):
    """Compose the configuration and run one collection pass."""
    run_batch_rollout(config)


def run_batch_rollout(config) -> dict:
    """Initialize Ray and run the collection to completion.

    Args:
        config: The composed batch rollout configuration.

    Returns:
        dict: The run summary, as written to `summary.json`.
    """
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env = OmegaConf.merge(default_runtime_env, ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray_init_kwargs = OmegaConf.to_container(ray_init_kwargs)

        # The task runner starts the session proxy in its own process, so the
        # variable naming the upstream credential has to travel with it.
        api_key_env = str(config.batch_rollout.serving.get("api_key_env", "") or "")
        if api_key_env:
            if not os.environ.get(api_key_env):
                raise ValueError(
                    f"batch_rollout.serving.api_key_env={api_key_env!r} but that variable is unset. "
                    f"Export it before launching, for example `export {api_key_env}=...`."
                )
            env_vars = ray_init_kwargs["runtime_env"].setdefault("env_vars", {})
            env_vars[api_key_env] = os.environ[api_key_env]

        # PATCH(lhy): Harbor runs inside Ray worker actors, which inherit
        # the env from `ray start`, not this driver process. The no-network
        # gate (HARBOR_DOCKER_STATIC_NO_NETWORK=1) that swaps the NET_RAW
        # egress sidecar for capability-free `network_mode: none` must
        # therefore travel into the workers explicitly, or rootless daemons
        # reject the sidecar.
        _fwd = ray_init_kwargs["runtime_env"].setdefault("env_vars", {})
        for _k in ("HARBOR_DOCKER_STATIC_NO_NETWORK", "OPENAI_API_KEY", "NO_PROXY", "no_proxy"):
            _v = os.environ.get(_k)
            if _v is not None and _k not in _fwd:
                _fwd[_k] = _v

        ray.init(**ray_init_kwargs)

    runner = BatchRolloutTaskRunner.remote()
    summary = ray.get(runner.run.remote(config))
    ray.shutdown()
    return summary


@ray.remote(num_cpus=1)
class BatchRolloutTaskRunner:
    """Ray driver for one collection pass.

    Owns serving bringup, the worker fleet, the prompt feed, and teardown. The
    equivalent of `TaskRunner` in `main_ppo.py`, minus every training role.
    """

    def run(self, config) -> dict:
        """Collect rollouts for the configured dataset.

        Args:
            config: The composed batch rollout configuration.

        Returns:
            dict: The run summary.
        """
        from pprint import pprint

        print(f"BatchRolloutTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        # NOTE(claude): Printed resolved, never resolved in place. veRL's rollout
        # config interpolates the profiler group with literal-list fallbacks that
        # in-place resolution rejects. Lazy access resolves each correctly.
        pprint(OmegaConf.to_container(config, resolve=True))

        os.makedirs(config.psrl.logging_path, exist_ok=True)
        _validate_rollout_budget(config)

        backend = build_serving_backend(config)
        output_writer = None
        manager = None
        started_at = time.time()

        try:
            backend_handle = backend.start()

            output_writer = RolloutOutputWriter.remote(
                output_dir=config.batch_rollout.output_dir,
                resume=config.batch_rollout.resume,
                logging_path=config.psrl.logging_path,
            )
            completed_uids = ray.get(output_writer.completed_uids.remote())
            if completed_uids:
                psrl_logger.info("Resuming: %d request(s) already collected.", len(completed_uids))

            workers = self._build_workers(config, backend_handle, output_writer)

            # One in-flight prompt per concurrent episode slot. Harbor containers are
            # invisible to Ray, so this queue is what bounds them.
            in_flight_per_worker = int(config.batch_rollout.max_in_flight_per_worker)
            data_queue_size = max(1, len(workers) * in_flight_per_worker)
            # An actor, because the feed reaches it over `put_data.remote`.
            # Concurrency must exceed the dispatch loop plus the feed's puts.
            manager = (
                ray.remote(BatchRolloutAgentLoopManager)
                .options(max_concurrency=max(4, data_queue_size))
                .remote(config, data_queue_size, workers)
            )

            data_processor = self._build_data_processor(config, completed_uids)
            ray.get(data_processor.set_agent_loop_manager.remote(manager))

            summary = self._collect(config, manager, data_processor, output_writer, backend_handle, started_at)
            return summary
        finally:
            if manager is not None:
                self._stop(manager)
            backend.stop()

    def _build_workers(self, config, backend_handle, output_writer) -> list:
        """Place one worker per allowed node, round-robin.

        Lifted from the RL trainer. `agent.node_ips` restricts placement to an
        allow list, because a node with a degraded Docker daemon accepts actors and
        then hangs. Affinity is hard whenever an allow list is given, since a soft
        placement lets Ray fall back to exactly the node being excluded.
        """
        num_workers = int(config.batch_rollout.num_workers)
        allowed_ips = list(config.gen_actor_rollout_ref.rollout.agent.get("node_ips") or [])
        alive_nodes = [n for n in ray.nodes() if n["Alive"]]
        if allowed_ips:
            selected = [n for n in alive_nodes if n["NodeManagerAddress"] in allowed_ips]
            if not selected:
                raise ValueError(
                    f"agent.node_ips={allowed_ips} matched no alive node. "
                    f"Alive: {sorted(n['NodeManagerAddress'] for n in alive_nodes)}."
                )
            psrl_logger.info(
                "Agent loop workers restricted to %d of %d nodes: %s.",
                len(selected),
                len(alive_nodes),
                sorted(n["NodeManagerAddress"] for n in selected),
            )
            alive_nodes = selected
        alive_node_ids = [n["NodeID"] for n in alive_nodes]

        max_concurrency = max(1, int(config.batch_rollout.max_in_flight_per_worker) * 4)
        workers = []
        for i in range(num_workers):
            node_id = alive_node_ids[i % len(alive_node_ids)]
            workers.append(
                BatchRolloutAgentLoopWorker.options(
                    name=f"batch_rollout_worker_{i}",
                    max_concurrency=max_concurrency,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=not allowed_ips),
                ).remote(
                    config,
                    backend_handle,
                    output_writer,
                    worker_id=i,
                    worker_num=num_workers,
                )
            )
            psrl_logger.info("Batch rollout worker %d scheduled on node %s.", i, node_id)
        self._workers = workers
        return workers

    def _build_data_processor(self, config, completed_uids: set):
        """Build the prompt feed, sharing the agent loops' tokenizer."""
        from verl.utils.config import omega_conf_to_dataclass
        from verl.workers.config import HFModelConfig

        model_config: HFModelConfig = omega_conf_to_dataclass(config.gen_actor_rollout_ref.model)
        return BatchRolloutDataProcessor.remote(
            config=config,
            tokenizer=model_config.tokenizer,
            processor=model_config.processor,
            completed_uids=completed_uids,
        )

    def _collect(self, config, manager, data_processor, output_writer, backend_handle, started_at: float) -> dict:
        """Run the feed to exhaustion, then wait for in-flight episodes to land."""
        ray.get(manager.start_busy_loop.remote())
        ray.get(data_processor.start_busy_loop.remote())

        poll_interval = float(config.batch_rollout.progress_interval_s)
        drain_timeout_s = float(config.batch_rollout.drain_timeout_s)
        last_records = -1
        idle_since = None

        while True:
            time.sleep(poll_interval)
            progress = ray.get(output_writer.progress.remote())
            in_flight = sum(ray.get([worker.in_flight.remote() for worker in self._workers]))
            psrl_logger.info(
                "Collected %d record(s) across %d uid(s), %d episode(s) in flight. Terminations: %s.",
                progress["records"],
                progress["uids"],
                in_flight,
                progress["terminate_reasons"],
            )

            feed_done = ray.get(data_processor.is_feed_finished.remote())
            if feed_done and in_flight == 0:
                psrl_logger.info("Feed exhausted and no episode in flight, collection is complete.")
                break

            # An episode can legitimately run for an hour, so progress is judged by
            # whether anything at all is still moving rather than by wall clock.
            if progress["records"] == last_records and in_flight > 0:
                idle_since = idle_since or time.time()
                if time.time() - idle_since > drain_timeout_s:
                    psrl_logger.warning(
                        "No record written in %.0fs with %d episode(s) still in flight. "
                        "Finalizing anyway, so the run is not lost. Check for wedged containers.",
                        drain_timeout_s,
                        in_flight,
                    )
                    break
            else:
                idle_since = None
            last_records = progress["records"]

        ray.get(data_processor.stop_busy_loop.remote())

        summary = ray.get(
            output_writer.finalize.remote(
                {
                    "model": backend_handle.model_name,
                    "serving": str(config.batch_rollout.serving.name),
                    "rollout_n": int(config.gen_actor_rollout_ref.rollout.n),
                    # `train_files` is a bare string for one file and a list for many.
                    "data_files": _as_plain(config.data.train_files),
                    "data_seed": config.data.get("seed", None),
                    "data_shuffle": config.data.get("shuffle", None),
                    "elapsed_s": round(time.time() - started_at, 1),
                }
            )
        )
        return summary

    def _stop(self, manager) -> None:
        """Stop the dispatcher and drain the workers."""
        try:
            ray.get(manager.stop_busy_loop.remote(), timeout=120)
        except Exception as exc:
            psrl_logger.warning("Failed to stop the agent loop manager cleanly: %s.", exc)


if __name__ == "__main__":
    main()
