import enum
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayResourcePool, SubRayResourcePool
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator

from psrl.utils.reward_token_metrics import build_rm_length_metrics
from psrl.workers.elastic.pool_layout import ReplicaPlacement

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# from verl.trainer.ppo.ray_trainer import compute_response_mask


def _compute_termination_metrics(
    terminate_reasons: list[str],
    parent_ids: list[str],
    scores: list[float],
    trained_tokens: list[float],
) -> dict[str, float]:
    """Break the batch down by why each episode stopped, and by group degeneracy.

    Two failure modes are invisible in the aggregate metrics and were only found by
    post-hoc log parsing. Splitting the reward by termination separates "the policy
    solved it" from "the harness cut it off", which move in opposite directions while
    `critic/score/mean` reports a single blended number. The zero-variance group
    fraction tracks how much of the batch produces no learning signal at all, since a
    GRPO group whose rollouts all score alike yields an advantage of exactly zero.

    Args:
        terminate_reasons (list[str]): Per-sample `TerminateReason` values.
        parent_ids (list[str]): Per-sample GRPO group key.
        scores (list[float]): Per-sample sequence-level score.
        trained_tokens (list[float]): Per-sample count of unmasked tokens, which is
            zero for a trajectory dropped by overlong filtering.

    Returns:
        dict[str, float]: Metrics under the `termination/` and `group/` prefixes.
    """
    total = len(terminate_reasons)
    if total == 0:
        return {}

    metrics: dict[str, float] = {}
    by_reason: dict[str, list[float]] = defaultdict(list)
    for reason, score in zip(terminate_reasons, scores, strict=True):
        by_reason[reason].append(score)
    for reason, reason_scores in by_reason.items():
        metrics[f"termination/{reason}/fraction"] = len(reason_scores) / total
        metrics[f"termination/{reason}/score_mean"] = float(np.mean(reason_scores))

    # Track surviving response tokens because `token-mean` normalization uses this count after overlong filtering.
    metrics["termination/trained_tokens"] = float(np.sum(trained_tokens))
    metrics["termination/masked_sample_fraction"] = sum(1 for t in trained_tokens if t == 0) / total

    groups: dict[str, list[float]] = defaultdict(list)
    for parent_id, score in zip(parent_ids, scores, strict=True):
        groups[parent_id].append(score)
    degenerate = sum(1 for group in groups.values() if len(group) > 1 and float(np.std(group)) < 1e-6)
    metrics["group/zero_variance_fraction"] = degenerate / len(groups) if groups else 0.0
    metrics["group/count"] = float(len(groups))
    return metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


class PSRL_Role(Enum):
    Actor = enum.auto()
    Rollout = enum.auto()
    ActorRollout = enum.auto()
    Critic = enum.auto()
    RefPolicy = enum.auto()
    RewardModel = enum.auto()
    ActorRolloutRef = enum.auto()
    Validate = enum.auto()
    TeacherModel = enum.auto()
    DummyPolicy = enum.auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[PSRL_Role, list[str]]
    resource_num_per_bundle: dict[str, int] = field(default_factory=dict)
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    elastic_replica_placements: dict[str, ReplicaPlacement] = field(default_factory=dict)

    def create_resource_pool(self, device_name: str = "cuda"):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=4 so three colocated roles
        plus one spare CPU for WorkerGroup init helpers can share a bundle.
        For Megatron backend, uses max_colocate_count>1 for different models.
        Elastic replica tags in `mapping` are sliced from parent pools after
        the physical placement groups exist.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # PSRL assigns one worker group to each resource pool.
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=4,
                name_prefix=resource_pool_name,
                resource_num_per_bundle=self.resource_num_per_bundle.get(resource_pool_name, 1),
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()
        self._create_elastic_replica_pools(device_name=device_name)

    def _create_elastic_replica_pools(self, device_name: str) -> None:
        """Slice parent elastic pools into per-replica `SubRayResourcePool` objects."""
        if not self.elastic_replica_placements:
            return

        parent_ids = {placement.pool_id for placement in self.elastic_replica_placements.values()}
        for pool_id in parent_ids:
            parent = self.resource_pool_dict[pool_id]
            parent.get_placement_groups(strategy="STRICT_PACK", device_name=device_name)

        for tag, placement in self.elastic_replica_placements.items():
            if tag in self.resource_pool_spec:
                raise ValueError(f"Elastic replica tag {tag!r} collides with a physical resource pool name.")
            parent = self.resource_pool_dict[placement.pool_id]
            self.resource_pool_dict[tag] = SubRayResourcePool(
                process_on_nodes=parent.store,
                use_gpu=parent.use_gpu,
                name_prefix=f"{parent.name_prefix}_{tag}",
                max_colocate_count=parent.max_colocate_count,
                detached=parent.detached,
                accelerator_type=parent.accelerator_type,
                resource_num_per_bundle=parent.resource_num_per_bundle,
                placement_groups=parent.pgs,
                start_bundle_index=placement.start_bundle_index,
                subgroup_world_size=placement.world_size,
            )
            psrl_logger.info(
                "Elastic replica placement: tag=%r, pool=%r, start_bundle_index=%s, world_size=%s.",
                tag,
                placement.pool_id,
                placement.start_bundle_index,
                placement.world_size,
            )

    def get_resource_pool(self, role: PSRL_Role, replica_idx: int = 0) -> RayResourcePool:
        """Get the resource pool of the worker_cls for the given replica_idx."""
        return self.resource_pool_dict[self.mapping[role][replica_idx]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # Tolerate floating-point noise in fractional bundle resources without masking real shortages.
        _GPU_EPS = 1e-9
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [
                n_gpus * self.resource_num_per_bundle.get(resource_pool_name, 1)
                for resource_pool_name, process_on_nodes in self.resource_pool_spec.items()
                for n_gpus in process_on_nodes
            ]
        )
        if total_available_gpus < total_required_gpus - _GPU_EPS:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


class PSRL_DummyWorker(Worker):
    def __init__(self, config: DictConfig, **kwargs):
        Worker.__init__(self)

        self.config = config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def init_model(self):
        return


def PSRL_compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: AlgoConfig | None = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # AGENT(VERL): PSRL uses `parent_id` instead of `uid` to index each GRPO response group and prompt.
    # Remain compatible with trainers that do not compute a response mask in fit.
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        if "parent_id" in data.non_tensor_batch:
            index = data.non_tensor_batch["parent_id"]
        else:
            assert "uid" in data.non_tensor_batch, "Either parent_id or uid is required for GRPO"
            index = data.non_tensor_batch["uid"]
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "parent_id" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["parent_id"]
        elif "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # `old_log_probs` provides the path variance proxy for the baseline.
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


def compute_advantage_for_multi_trajectories(
    data: DataProto,
    batch_keys: list[str],
    adv_estimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Any = None,
) -> DataProto:
    """Compute GRPO advantages from each session's final output. For non-GRPO
    estimators, such as GAE, are delegated to the original compute_advantage() unchanged.

    For GRPO, only the final output in each ``{uid}_{session_id}`` group participates
    in advantage computation, and the result is broadcast to the other outputs in
    the same session. Sessions whose AgentLoop returns ``None`` simply do not appear
    in ``batch_keys``. Non-GRPO estimators, such as GAE, are delegated to the
    original ``compute_advantage()`` unchanged.
    """
    if adv_estimator != core_algos.AdvantageEstimator.GRPO:
        return PSRL_compute_advantage(
            data,
            adv_estimator=adv_estimator,
            gamma=gamma,
            lam=lam,
            num_repeat=num_repeat,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

    # final session of each agent loop: uid => (index, row_index)
    final_sessions: dict[str, tuple[int, int]] = {}
    row_session_keys = []
    for i, key in enumerate(batch_keys):
        # A padding key ends in a UUID, not a trajectory index.
        # Its unique `parent_id` and zero mask keep the standalone sample at zero advantage.
        fields = [key] if key.startswith("pad_") else key.rsplit("_", 1)
        if len(fields) == 2:
            uid, index = fields[0], int(fields[1])
            session_key = uid
            if session_key not in final_sessions or final_sessions[session_key][0] < index:
                final_sessions[session_key] = (index, i)
        else:
            session_key = key
            final_sessions[session_key] = (0, i)
        row_session_keys.append(session_key)

    # final session indices in batch data
    final_indices = []
    session_key_to_local_index = {}
    for session_key, (_, row_index) in final_sessions.items():
        final_indices.append(row_index)
        session_key_to_local_index[session_key] = len(final_indices) - 1
    row_to_local_index = [session_key_to_local_index[session_key] for session_key in row_session_keys]

    # select final sessions from batch data for group relative advantage computation
    final_data = PSRL_compute_advantage(
        data.select_idxs(final_indices),
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    first_nnz_indices = final_data.batch["response_mask"].argmax(dim=1)
    final_scores = final_data.batch["advantages"][torch.arange(len(final_data)), first_nnz_indices]

    # scatter final scores to all rows in batch data
    scores = final_scores[row_to_local_index]
    scores = scores.unsqueeze(-1) * data.batch["response_mask"]

    data.batch["advantages"] = scores
    data.batch["returns"] = scores
    return data


def _stats_to_timestamps(stats) -> dict | None:
    if stats is None:
        return None

    if hasattr(stats, "__len__") and len(stats) > 0 and not hasattr(stats, "arrival_time"):
        stats = stats[0]

    arrival = getattr(stats, "arrival_time", None)
    ft_latency = getattr(stats, "first_token_latency", None)
    ft_ts_mono = getattr(stats, "first_token_ts", None)
    last_ts_mono = getattr(stats, "last_token_ts", None)

    if arrival is None:
        return None

    arrival_ts = float(arrival)
    ttft_ts = None
    finish_ts = None

    if ft_latency is not None:
        ttft_ts = arrival_ts + float(ft_latency)

    if ft_latency is not None and ft_ts_mono is not None and last_ts_mono is not None:
        decode_dur = float(last_ts_mono) - float(ft_ts_mono)
        finish_ts = arrival_ts + float(ft_latency) + decode_dur

    if ttft_ts is None and finish_ts is None:
        return None

    return {
        "arrival_ts": arrival_ts,
        "ttft_ts": ttft_ts,
        "finish_ts": finish_ts,
    }


def compute_rm_token_metrics(
    reward_extra_infos,
    global_token_num,
    timing_raw: dict[str, float],
    n_gpus: int,
) -> dict[str, float]:
    """
    Combine policy tokens with reward-model tokens for one training batch.

    `perf/rm_input_length` and `perf/rm_output_length` sum every nested
    `rm_input_len` and `rm_output_len` in per-sample `reward_extra_info`.
    `perf/overall_token_num` is `sum(global_token_num)` plus those two totals.
    `perf/overall_throughput` uses the same GPU-normalized formula as
    `perf/throughput`. `rm_seq_length/*` reports min, mean, and max over each
    RM call, plus per-model totals.

    Args:
        reward_extra_infos (list): Per-sample `reward_extra_info` payloads.
        global_token_num (list): Per-sample policy prompt plus response token counts.
        timing_raw (dict[str, float]): Stage timings. Must contain `step`.
        n_gpus (int): GPU count used to normalize throughput.

    Returns:
        dict[str, float]: `perf/rm_input_length`, `perf/rm_output_length`,
            `perf/overall_token_num`, `perf/overall_throughput`, and
            `rm_seq_length/*` sequence-length stats.
    """
    extra_infos = list(reward_extra_infos)
    policy_tokens = [int(token_num) for token_num in global_token_num]
    assert len(extra_infos) == len(policy_tokens), (
        f"Reward extra info count {len(extra_infos)!r} does not match global_token_num count {len(policy_tokens)!r}."
    )
    length_metrics = build_rm_length_metrics(extra_infos)
    overall_token_num = float(
        sum(policy_tokens) + length_metrics["perf/rm_input_length"] + length_metrics["perf/rm_output_length"]
    )
    step_time = timing_raw["step"]
    return {
        **length_metrics,
        "perf/overall_token_num": overall_token_num,
        "perf/overall_throughput": overall_token_num / (step_time * n_gpus),
    }


def record_rollout_rm_metrics(data: DataProto, output_path: str | None = None) -> list[dict]:
    """Extract rollout/reward timestamps from DataProto and optionally write to jsonl.

    Output format (one line per .jsonl):
        {
            "uid": 123,
            "rollout_metrics": {"arrival_ts": xxx, "ttft_ts": xxx, "finish_ts": xxx},
            "reward_metrics": {"gen/default/Qwen3-8B": {"arrival_ts": xxx, "ttft_ts": xxx, "finish_ts": xxx}}
        }

    Args:
        data: DataProto containing meta_info['rollout_metrics'],
            meta_info['reward_metrics'] and non_tensor_batch['uid'].
        output_path: If provided, append each record of this batch to the jsonl file.

    Returns:
        List of records (dict) for each sample in this batch.
    """
    meta = getattr(data, "meta_info", None) or {}
    rollout_metrics_arr = meta.get("rollout_metrics")
    reward_metrics_arr = meta.get("reward_metrics")
    uids = data.non_tensor_batch.get("uid", None)
    if uids is not None and hasattr(uids, "tolist"):
        uids = uids.tolist()
    batch_size = data.batch.batch_size[0]
    if uids is None or len(uids) != batch_size:
        uids = list(range(batch_size))

    records = []
    for i in range(batch_size):
        rec = {"uid": int(uids[i]) if np.issubdtype(type(uids[i]), np.integer) else uids[i]}

        if rollout_metrics_arr is not None and i < len(rollout_metrics_arr):
            rec["rollout_metrics"] = _stats_to_timestamps(rollout_metrics_arr[i])
        else:
            rec["rollout_metrics"] = None

        if reward_metrics_arr is not None and i < len(reward_metrics_arr):
            rm_dict = reward_metrics_arr[i]
            if isinstance(rm_dict, dict):
                rec["reward_metrics"] = {}
                for key, val in rm_dict.items():
                    if val is None or (hasattr(val, "__len__") and len(val) == 0):
                        continue
                    ts = _stats_to_timestamps(val)
                    if ts is not None:
                        rec["reward_metrics"][key] = ts
            else:
                rec["reward_metrics"] = None
        else:
            rec["reward_metrics"] = None
        records.append(rec)

    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return records
