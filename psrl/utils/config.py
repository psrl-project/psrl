from omegaconf import DictConfig
from verl.utils.config import omega_conf_to_dataclass

from psrl.trainer.ppo.batch_schedule import (
    REQUEST_AGG_MODE,
    SUPPORTED_BATCH_AGG_MODES,
    TRAJECTORY_AGG_MODE,
)
from psrl.utils.agent.thinking import (
    MULTI_TRAJ,
    requires_accumulating_template,
    template_prefills_assistant_think,
    validate_thinking_template,
    wants_thinking_disabled,
)
from psrl.utils.common.chat_template import resolve_chat_template_value


def resolve_fine_grain_chunk_size(config, dp_size: int) -> tuple[str, int]:
    """
    Resolve the effective granularity and chunk size (in prompt-groups) for `fine_grain_overlap`.

    A "prompt-group" is one prompt with all its `rollout_n` trajectories.
    `chunk_groups` = number of prompt-groups per chunk.

    Clamping rules:
    - `micro_batch * multiplier > mini_batch` => clamp to `mini_batch`.
    - `mini_batch * multiplier > full_batch` => clamp to `none` (full batch).

    Args:
        config: OmegaConf DictConfig with `psrl`, `data`, `gen_actor_rollout_ref`,
            and `train_actor_rollout_ref` keys.
        dp_size (int): Data-parallel size for the training actor.

    Returns:
        tuple[str, int]: Effective granularity string and `chunk_groups` count.

    Raises:
        ValueError: If `multiplier < 1`, granularity is unknown, required fields are
            missing, or the resolved chunk is not divisible by `rollout_n` or `dp_size`.
    """
    fgo = config.psrl.get("fine_grain_overlap", None)
    if fgo is None:
        granularity = "none"
    else:
        granularity = str(fgo.get("granularity", "none"))
    multiplier = int(fgo.get("multiplier", 1)) if fgo is not None else 1

    if multiplier < 1:
        raise ValueError(f"psrl.fine_grain_overlap.multiplier must be >= 1, got {multiplier}")

    rollout_n = config.gen_actor_rollout_ref.rollout.n
    full_batch_samples = config.data.train_batch_size * rollout_n
    full_batch_groups = config.data.train_batch_size  # train_batch_size is already in prompts

    if granularity == "none":
        return "none", full_batch_groups

    mini_samples = config.train_actor_rollout_ref.actor.ppo_mini_batch_size * rollout_n

    if granularity == "micro_batch":
        if config.train_actor_rollout_ref.actor.get("use_dynamic_bsz", False):
            raise ValueError(
                "psrl.fine_grain_overlap.granularity=micro_batch is not compatible with "
                "use_dynamic_bsz=True (micro_batch granularity requires a static "
                "ppo_micro_batch_size_per_gpu)."
            )
        micro_per_gpu = config.train_actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu")
        if micro_per_gpu is None:
            raise ValueError(
                "psrl.fine_grain_overlap.granularity=micro_batch requires "
                "train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu to be set"
            )
        base_samples = micro_per_gpu * dp_size
        chunk_samples = base_samples * multiplier
        if chunk_samples > mini_samples:
            granularity = "mini_batch"
            chunk_samples = mini_samples
    elif granularity == "mini_batch":
        chunk_samples = mini_samples * multiplier
    else:
        raise ValueError(
            f"psrl.fine_grain_overlap.granularity must be one of "
            f"'none', 'mini_batch', 'micro_batch'; got '{granularity}'"
        )

    if chunk_samples > full_batch_samples:
        granularity = "none"
        chunk_samples = full_batch_samples

    if chunk_samples % rollout_n != 0:
        raise ValueError(
            f"Resolved chunk_samples ({chunk_samples}) is not divisible by rollout_n ({rollout_n}). "
            "Chunks must be prompt-group-aligned. Adjust ppo_mini_batch_size or multiplier."
        )

    chunk_groups = chunk_samples // rollout_n

    if chunk_groups % dp_size != 0:
        raise ValueError(
            f"chunk_groups ({chunk_groups}) must be divisible by dp_size ({dp_size}) "
            "for balance packing. Adjust ppo_mini_batch_size, multiplier, or dp_size."
        )

    return granularity, chunk_groups


def validate_config(
    config: DictConfig,
    use_reference_policy: bool,
    use_critic: bool,
) -> None:
    """Validate an OmegaConf DictConfig.

    Args:
        config (DictConfig): The OmegaConf DictConfig to validate.
        use_reference_policy (bool): is ref policy needed
        use_critic (bool): is critic needed
    """

    train_n_gpus = config.psrl.deployment.train_ngpus_per_node * config.psrl.deployment.train_nnodes
    if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
        if config.train_actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                train_n_gpus
                % (model_parallel_size * config.train_actor_rollout_ref.actor.megatron.context_parallel_size)
                == 0
            ), (
                f"n_gpus ({train_n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.train_actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = train_n_gpus // (
                model_parallel_size * config.train_actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.train_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = train_n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.train_actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

    def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options.

        Ensures that users don't set both deprecated micro_batch_size and
        the new micro_batch_size_per_gpu parameters simultaneously.

        Args:
            mbs: Deprecated micro batch size parameter value.
            mbs_per_gpu: New micro batch size per GPU parameter value.
            name (str): Configuration section name for error messages.

        Raises:
            ValueError: If both parameters are set or neither is set.
        """
        settings = {
            "train_actor_rollout_ref.ref": "log_prob_micro_batch_size",
            "train_actor_rollout_ref.rollout": "log_prob_micro_batch_size",
        }

        if name in settings:
            param = settings[name]
            param_per_gpu = f"{param}_per_gpu"

            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(
                    f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                    f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                )

    actor_config = omega_conf_to_dataclass(config.train_actor_rollout_ref.actor)
    actor_config.validate(train_n_gpus, config.data.train_batch_size, config.train_actor_rollout_ref.model)

    if not config.train_actor_rollout_ref.actor.use_dynamic_bsz:
        if use_reference_policy:
            check_mutually_exclusive(
                config.train_actor_rollout_ref.ref.log_prob_micro_batch_size,
                config.train_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                "train_actor_rollout_ref.ref",
            )

        check_mutually_exclusive(
            config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size,
            config.train_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
            "train_actor_rollout_ref.rollout",
        )

    if config.algorithm.use_kl_in_reward and config.train_actor_rollout_ref.actor.use_kl_loss:
        print("NOTICE: You have both enabled in-reward kl and kl loss.")

    # critic
    if use_critic:
        critic_config = omega_conf_to_dataclass(config.critic)
        critic_config.validate(train_n_gpus, config.data.train_batch_size)

    if config.data.get("val_batch_size", None) is not None:
        print(
            "WARNING: val_batch_size is deprecated."
            + " Validation datasets are sent to inference engines as a whole batch,"
            + " which will schedule the memory themselves."
        )

    # check eval config
    if config.train_actor_rollout_ref.rollout.val_kwargs.do_sample:
        assert config.train_actor_rollout_ref.rollout.temperature > 0, (
            "validation gen temperature should be greater than 0 when enabling do_sample"
        )

    # check LoRA rank in vLLM
    lora_config = config.train_actor_rollout_ref.model.get("lora", {})
    lora_rank = lora_config.get("rank", 0)
    if lora_rank <= 0:
        lora_rank = config.train_actor_rollout_ref.model.get("lora_rank", 0)
    if lora_config.get("merge", False):
        lora_rank = 0
    if lora_rank > 0 and config.train_actor_rollout_ref.rollout.name == "vllm":
        from verl.workers.rollout.vllm_rollout.utils import get_vllm_max_lora_rank

        get_vllm_max_lora_rank(lora_rank)

    # --- PSRL Validation ---

    thinking_template = config.psrl.agentic_rl.get("thinking_template", MULTI_TRAJ)
    validate_thinking_template(thinking_template)
    chat_template = config.gen_actor_rollout_ref.rollout.get("chat_template", None)
    if requires_accumulating_template(thinking_template) and not chat_template:
        raise ValueError(
            "psrl.agentic_rl.thinking_template='multi_thinking' requires "
            "gen_actor_rollout_ref.rollout.chat_template to point at an accumulating "
            "template that replays prior-turn <think> blocks (for example "
            "examples/sciaccel_rl/config/qwen35_acc_thinking.jinja2 for Qwen3.5). "
            "The stock Qwen3/Qwen3.5 templates strip them and would fork the session "
            "into one trajectory per turn."
        )
    if wants_thinking_disabled(thinking_template) and chat_template:
        # Reject only templates that prefill an unclosed `<think>` on history, since a balanced
        # accumulating template stays valid with thinking off.
        template_source = resolve_chat_template_value(chat_template)
        if template_source and template_prefills_assistant_think(template_source):
            raise ValueError(
                f"psrl.agentic_rl.thinking_template={thinking_template!r} turns thinking off, but "
                f"gen_actor_rollout_ref.rollout.chat_template={chat_template!r} opens a <think> on "
                "historical assistant turns and relies on the generated text to close it. With "
                "thinking off there is no closing tag, so every historical turn renders an "
                "unterminated <think>. Use a template whose assistant branch emits a bare content "
                "(examples/sciaccel_rl/config/qwen3_acc_thinking.jinja2), or unset the override to "
                "use the model's own template, which emits a balanced empty <think> block."
            )

    batch_agg_mode = config.psrl.agentic_rl.get("batch_agg_mode", TRAJECTORY_AGG_MODE)
    assert batch_agg_mode in SUPPORTED_BATCH_AGG_MODES, (
        f"psrl.agentic_rl.batch_agg_mode must be one of {sorted(SUPPORTED_BATCH_AGG_MODES)}, got {batch_agg_mode!r}."
    )
    if batch_agg_mode == REQUEST_AGG_MODE and not config.psrl.rollout_coordination.redundant_rollout.enable:
        rollout_n = config.gen_actor_rollout_ref.rollout.n
        requests_per_buffer = config.psrl.staleness_buffer_entries * rollout_n
        actor_entries_per_update = config.train_actor_rollout_ref.actor.ppo_mini_batch_size * rollout_n
        assert requests_per_buffer % actor_entries_per_update == 0, (
            "With psrl.agentic_rl.batch_agg_mode='request', "
            f"requests_per_buffer={requests_per_buffer} must be divisible by "
            f"actor_entries_per_update={actor_entries_per_update}. "
            "Adjust psrl.staleness_buffer_entries or "
            "train_actor_rollout_ref.actor.ppo_mini_batch_size."
        )
        if use_critic:
            critic_entries_per_update = config.critic.ppo_mini_batch_size * rollout_n
            assert requests_per_buffer % critic_entries_per_update == 0, (
                "With psrl.agentic_rl.batch_agg_mode='request', "
                f"requests_per_buffer={requests_per_buffer} must be divisible by "
                f"critic_entries_per_update={critic_entries_per_update}. "
                "Adjust psrl.staleness_buffer_entries or critic.ppo_mini_batch_size."
            )

    # Check NIXL compatibility
    if config.psrl.ps_mode == "nixl_cpu" or config.psrl.ps_mode == "nixl_gpu":
        assert config.psrl.nixl.server_ip == config.psrl.ps_manager_ip, (
            "PSManager IP and NIXL server IP must be the same"
        )
        assert config.train_actor_rollout_ref.actor.strategy != "fsdp", (
            "FSDP1 is not supported for NIXL because it uses flat_param"
        )
        print(f"NOTICE: NIXL is enabled. Actor strategy used is {config.train_actor_rollout_ref.actor.strategy}")

    # Check validate mode
    if not config.psrl.colocate_validate_and_train:
        assert config.psrl.fuse_rollout_with_validate, (
            "fuse_rollout_with_validate must be enabled when not colocate_validate_and_train"
        )

    # Check routing strategy
    routing_method = config.psrl.rollout_coordination.routing_strategy.method
    allowed_routing_methods = {
        "random",
        "round_robin",
        "request_num_balance",
        "throughput_optimal",
        "throughput_optimal_with_budget",
        "throughput_balance",
        "cache_aware",
        "cache_aware_v1",
    }
    assert routing_method in allowed_routing_methods, (
        f"psrl.rollout_coordination.routing_strategy.method must be one of {sorted(allowed_routing_methods)}, "
        f"got '{routing_method}'. Use 'cache_aware' instead of deprecated 'kv_cache_aware'."
    )

    if routing_method in ("request_num_balance", "throughput_balance"):
        assert config.psrl.status_collection.enable, (
            "status collection must be enabled when using request num balance or throughput balance routing strategy"
        )

    # Check TMS configuration
    if config.psrl.tms.enable_cuda_graph:
        assert config.psrl.tms.range == "all", "TMS CUDA graph can only be enabled when TMS range is 'all'"

    # Check LMCache and KV transfer configuration
    lmcache_cfg = config.psrl.get("lmcache", {})
    kv_transfer_cfg = config.psrl.rollout_coordination.routing_strategy.get("kv_transfer", {})

    if kv_transfer_cfg.get("enable", False):
        kv_transfer_required = (
            "must be True when psrl.rollout_coordination.routing_strategy.kv_transfer.enable is True."
        )
        assert lmcache_cfg.get("enable", False), f"psrl.lmcache.enable {kv_transfer_required}"
        assert lmcache_cfg.get("enable_p2p", False), f"psrl.lmcache.enable_p2p {kv_transfer_required}"
        assert config.psrl.rollout_coordination.partial_rollout.enable, (
            f"psrl.rollout_coordination.partial_rollout.enable {kv_transfer_required}"
        )

    if lmcache_cfg.get("enable_p2p", False):
        assert lmcache_cfg.get("enable", False), (
            "psrl.lmcache.enable must be True when psrl.lmcache.enable_p2p is True."
        )
        assert config.psrl.ps_mode in ("nixl_cpu", "nixl_gpu"), (
            "psrl.ps_mode must be nixl_cpu or nixl_gpu when psrl.lmcache.enable_p2p is True "
            "(NIXL infrastructure is required for P2P transfer)."
        )
        assert not lmcache_cfg.get("clear_on_weight_update", True), (
            "psrl.lmcache.clear_on_weight_update must be False when psrl.lmcache.enable_p2p is True "
            "(LMCache P2PBackend does not support clear. Stale entries cannot be flushed on weight update)."
        )
        assert lmcache_cfg.get("multi_version_kv", False), (
            "psrl.lmcache.multi_version_kv must be True when psrl.lmcache.enable_p2p is True "
            "(version tagging is required because P2P KV cannot be cleared on weight sync)."
        )

    if lmcache_cfg.get("enable", False):
        assert config.gen_actor_rollout_ref.rollout.enable_prefix_caching, (
            "gen_actor_rollout_ref.rollout.enable_prefix_caching must be True "
            "when psrl.lmcache.enable is True (LMCache requires prefix caching)."
        )

    # Session hang/continue (ThunderAgent) switches.
    thunder_agent_cfg = config.psrl.rollout_coordination.session_strategy.get("thunder_agent", {})
    if thunder_agent_cfg.get("enable", False):
        continue_scope = str(thunder_agent_cfg.get("continue_scope", "bucketed"))
        assert continue_scope in ("bucketed", "global"), (
            f"psrl.rollout_coordination.session_strategy.thunder_agent.continue_scope must be "
            f"'bucketed' or 'global', got {continue_scope!r}."
        )
        # Trajectory-sticky routing pins sessions to their current instance.
        sticky_enabled = bool(config.psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky)
        assert not (sticky_enabled and continue_scope == "global"), (
            "psrl.rollout_coordination.session_strategy.thunder_agent.continue_scope must be "
            "'bucketed' when psrl.rollout_coordination.routing_strategy.enable_trajectory_sticky "
            "is True (sticky sessions re-route to their current instance. Global continue would "
            "mismatch)."
        )

    # ---- fine_grain_overlap validation ----
    fgo = config.psrl.get("fine_grain_overlap", None)
    if fgo is not None and str(fgo.get("granularity", "none")) != "none":
        train_n_gpus_fgo = config.psrl.deployment.train_ngpus_per_node * config.psrl.deployment.train_nnodes
        actor_strategy = config.train_actor_rollout_ref.actor.strategy
        dp_size_fgo = train_n_gpus_fgo  # FSDP dp_size == train_n_gpus (no model parallelism for FSDP)
        if actor_strategy == "megatron":
            mp = (
                config.train_actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.train_actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            dp_size_fgo = train_n_gpus_fgo // (
                mp * config.train_actor_rollout_ref.actor.megatron.context_parallel_size
            )

        overlap_scope = str(fgo.get("overlap_scope", "recompute"))
        granularity = str(fgo.get("granularity", "none"))
        multiplier = int(fgo.get("multiplier", 1))

        if multiplier < 1:
            raise ValueError("psrl.fine_grain_overlap.multiplier must be >= 1.")

        effective_gran, chunk_groups = resolve_fine_grain_chunk_size(config, dp_size_fgo)

        if batch_agg_mode == REQUEST_AGG_MODE and overlap_scope == "pre_step":
            if config.psrl.rollout_coordination.redundant_rollout.enable:
                selected_requests_per_group = config.psrl.rollout_coordination.redundant_rollout.alg_rollout_n
            else:
                selected_requests_per_group = config.gen_actor_rollout_ref.rollout.n
            requests_per_window = chunk_groups * selected_requests_per_group
            actor_requests_per_update = (
                config.train_actor_rollout_ref.actor.ppo_mini_batch_size * selected_requests_per_group
            )
            assert requests_per_window % actor_requests_per_update == 0, (
                "Request-aggregated pre_step windows must contain a whole number of actor updates. "
                f"requests_per_window={requests_per_window}, "
                f"actor_requests_per_update={actor_requests_per_update}."
            )
            if use_critic:
                critic_requests_per_update = config.critic.ppo_mini_batch_size * selected_requests_per_group
                assert requests_per_window % critic_requests_per_update == 0, (
                    "Request-aggregated pre_step windows must contain a whole number of critic updates. "
                    f"requests_per_window={requests_per_window}, "
                    f"critic_requests_per_update={critic_requests_per_update}."
                )

        ppo_epochs = config.train_actor_rollout_ref.actor.get("ppo_epochs", 1)
        if overlap_scope == "pre_step" and ppo_epochs > 1:
            raise ValueError(
                "psrl.fine_grain_overlap.overlap_scope=pre_step is not compatible with "
                f"ppo_epochs={ppo_epochs} > 1 (streaming accumulation cannot revisit chunks). "
                "Set ppo_epochs=1 or use overlap_scope=recompute."
            )

        loss_agg_mode = config.train_actor_rollout_ref.actor.get("loss_agg_mode", "token-mean")
        if overlap_scope == "pre_step" and loss_agg_mode == "session-mean-token-mean":
            raise ValueError(
                "psrl.fine_grain_overlap.overlap_scope=pre_step is not compatible with "
                "train_actor_rollout_ref.actor.loss_agg_mode=session-mean-token-mean: pre_step "
                "computes session weights per chunk (chunk-local session counts), which breaks the "
                "window-level session equality required by session-mean-token-mean loss. "
                "Set overlap_scope=recompute (advantage+update run once on the concatenated "
                "full batch; per-sample GPU stages still overlap) or disable fine_grain_overlap "
                "(granularity=none) for the full-batch path."
            )

        if granularity == "micro_batch" and overlap_scope == "pre_step":
            if actor_strategy == "megatron":
                raise ValueError(
                    "micro_batch + pre_step is not supported for actor.strategy=megatron. "
                    "Use fsdp2 or set overlap_scope=recompute."
                )

        adv_estimator = str(config.algorithm.get("adv_estimator", "gae"))
        if overlap_scope == "pre_step" and adv_estimator not in ("grpo", "grpo_vectorized", "grpo_passk"):
            print(
                f"WARNING: psrl.fine_grain_overlap.overlap_scope=pre_step with "
                f"adv_estimator={adv_estimator}: per-chunk advantage normalization "
                "differs from full-batch normalization (masked_whiten scope changes). "
                "Use adv_estimator=grpo for exact equivalence."
            )

        reward_norm_mode = config.psrl.get("reward_normalization", "group")
        if str(reward_norm_mode) == "batch" and str(fgo.get("granularity", "none")) != "none":
            print(
                "WARNING: psrl.fine_grain_overlap with reward_normalization='batch': "
                "per-chunk reward normalization uses chunk-level statistics, not full-batch statistics. "
                "Use reward_normalization='group' for exact equivalence with the full-batch path."
            )

    # env worker checks
    env_worker_config = config.psrl.env_worker
    if env_worker_config.enable:
        if env_worker_config.placement not in ("colocated", "dedicated"):
            raise ValueError(
                f"psrl.env_worker.placement must be colocated or dedicated, got {env_worker_config.placement!r}."
            )
        if env_worker_config.placement == "dedicated" and not env_worker_config.dedicated_node_ips:
            raise ValueError(
                "psrl.env_worker.placement=dedicated requires a non-empty psrl.env_worker.dedicated_node_ips list."
            )
        if env_worker_config.routing.method not in ("least_loaded", "round_robin", "random"):
            raise ValueError(
                f"psrl.env_worker.routing.method must be least_loaded, round_robin, or random, "
                f"got {env_worker_config.routing.method!r}."
            )
        if int(env_worker_config.cpu_slots_per_worker) <= 0:
            raise ValueError(
                f"psrl.env_worker.cpu_slots_per_worker must be positive, "
                f"got {env_worker_config.cpu_slots_per_worker!r}."
            )

    print("[validate_config] All configuration checks passed successfully!")
