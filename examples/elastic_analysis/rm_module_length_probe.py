from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from ray.util.queue import Queue as RayQueue
from tensordict import TensorDict
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup

from psrl.trainer.constants_ppo import get_ppo_ray_runtime_env
from psrl.trainer.ppo.utils import PSRL_Role, ResourcePoolManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


def _select_gen_reward_model(config) -> Any:
    probe_name = OmegaConf.select(config, "rm_probe.reward_model_name")
    probe_index = OmegaConf.select(config, "rm_probe.reward_model_index")
    candidates = [
        (idx, rm)
        for idx, rm in enumerate(config.reward_models_config.reward_models)
        if rm.reward_loop_type == "gen"
    ]
    if not candidates:
        raise ValueError("No reward_models_config.reward_models entry with reward_loop_type=gen.")

    if probe_name is not None:
        for _, rm in candidates:
            rm_name = rm.get("reward_model_name", rm.model.path.split("/")[-1])
            if rm_name == probe_name:
                return rm
        raise ValueError(f"Cannot find gen reward model named {probe_name!r}.")

    if probe_index is not None:
        idx = int(probe_index)
        try:
            return config.reward_models_config.reward_models[idx]
        except IndexError as exc:
            raise ValueError(f"reward_model_index={idx} is out of range.") from exc

    return candidates[0][1]


def _ray_init(config) -> None:
    if ray.is_initialized():
        return
    default_runtime_env = get_ppo_ray_runtime_env()
    default_runtime_env["env_vars"]["PSRL_LOGGING_PATH"] = config.psrl.logging_path
    ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
    runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
    runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
    ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
    print(f"ray init kwargs: {ray_init_kwargs}")
    ray.init(**OmegaConf.to_container(ray_init_kwargs, resolve=True))


def _force_probe_replica_count(config, reward_model) -> int:
    num_replicas = int(OmegaConf.select(config, "rm_probe.num_replicas", default=1))
    if num_replicas <= 0:
        raise ValueError("rm_probe.num_replicas must be positive.")
    with open_dict(reward_model):
        reward_model.num_replicas = num_replicas
    return num_replicas


def _create_rm_manager(config, reward_model):
    from psrl.workers.gen.gen_worker import GenInterface, PSRL_GenWorker
    from psrl.workers.reward.reward_model.manager import PSRL_RewardModelManager

    num_replicas = _force_probe_replica_count(config, reward_model)
    rm_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
    world_size = int(
        reward_model.rollout.tensor_model_parallel_size
        * reward_model.rollout.pipeline_model_parallel_size
        * reward_model.rollout.get("data_parallel_size", 1)
    )
    if world_size <= 0:
        raise ValueError(f"Invalid RM world_size={world_size}.")

    resource_pool_spec = {
        f"rm_probe_pool_{i}": [world_size] for i in range(num_replicas)
    }
    mapping = {
        PSRL_Role.RewardModel: [f"rm_probe_pool_{i}" for i in range(num_replicas)]
    }
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping,
    )
    resource_pool_manager.create_resource_pool()

    status_queues = [RayQueue() for _ in range(num_replicas)]
    reward_worker_cls = ray.remote(max_concurrency=10000)(PSRL_GenWorker)
    worker_groups = []
    for instance_id in range(num_replicas):
        gen_interface = GenInterface(
            rollout_instance_id=instance_id,
            status_queue=status_queues[instance_id],
        )
        ray_cls = RayClassWithInitArgs(
            cls=reward_worker_cls,
            config=reward_model,
            role="reward",
            psrl_config=config.psrl,
            instance_id=instance_id,
            gen_interface=gen_interface,
            reward_model_name=rm_name,
            rm_config=reward_model,
            is_teacher_model=False,
        )
        wg_kwargs = {"device_name": config.trainer.device}
        if OmegaConf.select(config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = config.trainer.ray_wait_register_center_timeout
        worker_groups.append(
            RayWorkerGroup(
                resource_pool=resource_pool_manager.resource_pool_dict[f"rm_probe_pool_{instance_id}"],
                ray_cls_with_init=ray_cls,
                **wg_kwargs,
            )
        )

    return PSRL_RewardModelManager(
        reward_model_name=rm_name,
        config=config,
        reward_model_config=reward_model,
        reward_model_wg_list=worker_groups,
        status_queues=status_queues,
        max_concurrency=int(reward_model.get("max_concurrent_requests_per_instance", 1) or 1),
    )


def _encode(tokenizer, text: str) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        token_ids = [tokenizer.eos_token_id or 0]
    return torch.tensor(token_ids, dtype=torch.long)


def _build_rollout_result(tokenizer, problem: str, solution: str) -> DataProto:
    prompt_ids = _encode(tokenizer, problem)
    response_ids = _encode(tokenizer, solution)
    attention_mask = torch.ones(prompt_ids.numel() + response_ids.numel(), dtype=torch.long)
    batch = TensorDict(
        {
            "prompts": prompt_ids.unsqueeze(0),
            "responses": response_ids.unsqueeze(0),
            "attention_mask": attention_mask.unsqueeze(0),
        },
        batch_size=[1],
    )
    non_tensor_batch = {
        "uid": np.array(["rm_probe_0"], dtype=object),
        "data_source": np.array(["rm_probe"], dtype=object),
        "reward_model": np.array([{"ground_truth": ""}], dtype=object),
        "extra_info": np.array([{"question": problem}], dtype=object),
    }
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


def _metric_subset(extra_info: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "score",
        "acc",
        "rm_input_len",
        "rm_output_len",
        "rm_input_len_local_plain",
        "rm_input_len_local_chat",
        "rm_output_len_local_plain",
        "rm_input_len_plain",
        "rm_output_len_retokenized",
    ]
    return {key: extra_info[key] for key in keys if key in extra_info}


async def _run_probe(config, reward_model, rm_manager) -> dict[str, Any]:
    from psrl.workers.reward.reward_loop.registry import load_reward_loop_manager

    tokenizer = rm_manager.get_reward_model_tokenizer()
    problem = OmegaConf.select(
        config,
        "rm_probe.problem",
        default="What is 1+1? Please answer with the final result.",
    )
    solution = OmegaConf.select(
        config,
        "rm_probe.solution",
        default="1+1=2, so the answer is \\boxed{2}.",
    )
    data = _build_rollout_result(tokenizer, str(problem), str(solution))
    reward_fn = reward_model.reward_fn
    if not isinstance(reward_fn, str):
        reward_fn = list(reward_fn)[0]
    reward_loop = load_reward_loop_manager(
        reward_model_config=config,
        input_tokenizer=tokenizer,
        reward_loop_type=reward_model.reward_loop_type,
        reward_fn=reward_fn,
        reward_model_manager=rm_manager,
        **reward_model.get("reward_loop_kwargs", {}),
    )
    result = await reward_loop.run_single(data)
    extra_info = result["reward_extra_info"]
    return {
        "reward_score": result["reward_score"],
        "metrics": _metric_subset(extra_info),
        "rm_output_preview": str(extra_info.get("rm_output", ""))[:1000],
    }


@hydra.main(config_path="../psrl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    OmegaConf.resolve(config)
    log_dir = Path(OmegaConf.select(config, "rm_probe.log_dir", default="/tmp/psrl_rm_probe"))
    log_dir.mkdir(parents=True, exist_ok=True)
    with open_dict(config):
        config.psrl.logging_path = str(log_dir)

    reward_model = _select_gen_reward_model(config)
    with open_dict(reward_model):
        # Keep the probe resource-bounded unless explicitly overridden.
        reward_model.num_replicas = int(OmegaConf.select(config, "rm_probe.num_replicas", default=1))

    _ray_init(config)
    rm_manager = _create_rm_manager(config, reward_model)
    try:
        result = asyncio.run(_run_probe(config, reward_model, rm_manager))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        if bool(OmegaConf.select(config, "rm_probe.ray_shutdown", default=True)):
            ray.shutdown()


if __name__ == "__main__":
    main()
