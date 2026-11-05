from omegaconf import DictConfig, OmegaConf, open_dict
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
from psrl.utils.quantization import QuantizationPlan, resolve_quantization_plan


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


def resolve_train_model_dtype(config: DictConfig) -> str:
    """The dtype the trainer's parameters are actually stored in.

    Megatron's DistributedOptimizer keeps bf16 model params alongside fp32 master
    params. FSDP2 has no master params: `model_dtype` (default "fp32", but set to
    "bf16" by many production recipes) *is* the storage dtype, and that is the dtype
    NIXL registers.

    Reading `model_dtype` rather than branching on the backend matters because the
    buffer the weight transport prepares must match the real bytes -- assuming fp32
    under `model_dtype=bf16` sizes every buffer 2x too large and silently
    misinterprets the wire format.
    """
    strategy = config.train_actor_rollout_ref.actor.strategy
    if strategy == "megatron":
        return "bf16"
    engine_cfg = config.train_actor_rollout_ref.actor.get("engine", {}) or {}
    return engine_cfg.get("model_dtype", "fp32")


def _reject_qat_for_direct_supply(config: DictConfig) -> None:
    """Direct supply publishes its own wire format. Upstream QAT is a different mechanism.

    veRL's ``rollout.quantization`` presets make the *trainer* quantize every refit with
    compressed-tensors and make vLLM load that format. PSRL's publisher already casts and
    quantizes the shards it publishes, so combining the two would advertise a format the
    trainer never produced.
    """
    for name, rollout in (
        ("gen_actor_rollout_ref", config.gen_actor_rollout_ref.rollout),
        ("train_actor_rollout_ref", config.train_actor_rollout_ref.rollout),
    ):
        quantization = rollout.get("quantization") or {}
        if quantization.get("linear") or quantization.get("moe") or quantization.get("config_file"):
            raise ValueError(
                f"{name}.rollout.quantization is the trainer-side QAT path. Direct supply "
                "publishes its own blocks, so set psrl.nixl.weight_quantization.scheme=fp8 "
                "for an FP8 rollout instead."
            )


def _resolve_publish_format(config: DictConfig) -> None:
    """
    Record the trainer and rollout dtypes and resolve the published weight format.

    Written into the config so the format a run used is visible in its dump.
    """
    nixl = config.psrl.nixl
    rollout = config.gen_actor_rollout_ref.rollout
    nixl.publish_train_dtype = resolve_train_model_dtype(config)
    nixl.publish_rollout_dtype = str(rollout.dtype)
    _reject_qat_for_direct_supply(config)
    # One resolved plan feeds the publisher, both rollout roles and the converter.
    quantization = resolve_quantization_plan(nixl.get("weight_quantization"))
    nixl.publish_quantization = quantization.to_config()

    reader_quantization = bool(nixl.get("reader_quantization", False))
    if reader_quantization:
        if quantization.scheme != "fp8":
            raise ValueError(
                "psrl.nixl.reader_quantization=true requires "
                "psrl.nixl.weight_quantization.scheme=fp8."
            )
        if str(config.train_actor_rollout_ref.actor.strategy) != "megatron":
            raise ValueError(
                "psrl.nixl.reader_quantization=true is implemented for Megatron only; "
                f"got actor.strategy={config.train_actor_rollout_ref.actor.strategy!r}."
            )
        from psrl.utils.ps_mode import has_ps_data_plane

        if has_ps_data_plane(config.psrl.ps_mode):
            raise ValueError(
                "psrl.nixl.reader_quantization=true requires direct supply "
                "(ps_mode=nixl_direct); got ps_mode="
                f"{config.psrl.ps_mode!r}."
            )
    nixl.reader_quantization = reader_quantization


def _validate_nixl_offload_flags(config: DictConfig) -> None:
    """Reject both FSDP2 parameter-offload switches (design doc 3.9.3c).

    `offload_fsdp2_model_to_cpu` reallocates every DTensor's local storage between
    steps, which dangles the GPU pointers NIXL registered. `CPUOffloadPolicy` keeps
    parameters on the host entirely. Both break trainer-direct supply.

    Both flags must be checked separately: verl forces `_is_offload_param` to False
    whenever `offload_policy` is set (`transformer_impl.py:460`), so inspecting only
    `param_offload` silently misses the `offload_policy` case.

    `optimizer_offload` is deliberately *not* rejected -- it is the only existing way
    to free optimizer state across a colocate switch (design doc 3.7.3.1).
    """
    actor_cfg = config.train_actor_rollout_ref.actor
    engine_cfg = actor_cfg.get("engine", {}) or {}

    if actor_cfg.get("param_offload", False):
        raise ValueError(
            "actor.param_offload=True is incompatible with the NIXL weight transport: "
            "offloading FSDP2 parameters to CPU reallocates each DTensor's local "
            "storage, dangling the GPU pointers registered with NIXL. Disable "
            "param_offload (optimizer_offload may stay enabled)."
        )
    if engine_cfg.get("offload_policy", False):
        raise ValueError(
            "actor.engine.offload_policy=True is incompatible with the NIXL weight "
            "transport: CPUOffloadPolicy keeps parameters on the host, so there is no "
            "stable GPU address to register. Disable offload_policy "
            "(optimizer_offload may stay enabled)."
        )


def _resolve_version_barrier(config: DictConfig) -> None:
    """Decide whether publishing must wait for readers, and record the decision (E1).

    Resolved here because it is the one place that can see all three inputs: the training
    backend, and both dtypes. `PSManager` receives only `config.psrl`, so it cannot work
    this out for itself and must be told.

    The answer, from `requires_version_barrier`:

    * **FSDP2 / HSDP at matched dtype: required.** No master parameters, so the whole
      optimizer step writes the memory a rollout may be reading.
    * **Megatron at matched dtype: wanted.** Its conflict window is only the write-back
      instant, but narrow is not zero and the cost when nobody is reading is a counter
      check.
    * **A cast path, or colocate: never.** A cast path reads a separate quantised buffer,
      and colocate pauses rollout for the duration, so neither shares memory with the
      writer.

    Written into the config rather than returned, so the value a run actually used is
    visible in a dumped config -- and an explicit `psrl.nixl.await_readers_before_publish`
    still wins, which is what the "turn it off and watch the weights break" criterion
    needs.
    """
    from psrl.workers.ps.version_barrier import requires_version_barrier

    nixl = config.psrl.nixl
    decided = nixl.get("await_readers_before_publish", "auto")
    # An explicit true/false is the operator's call and is left alone. Testing truthiness
    # would not do: `auto` is a non-empty string and so is true, while an explicit `False`
    # -- the setting the E1 criterion needs, to observe the corruption it prevents -- is
    # false and would be silently re-derived.
    if isinstance(decided, bool):
        return

    actor = config.actor_rollout_ref.actor
    megatron = actor.get("megatron") or {}
    # The training dtype lives under the backend's own block, not on the model, and the
    # rollout side has no dtype at all -- it is quantised or it is not. So "matched
    # precision" is: the trainer is not quantising and the rollout is not either. A
    # rollout that casts reads a separate buffer, which is exactly the case with no shared
    # memory to protect.
    train_dtype = str(megatron.get("dtype", "bfloat16"))
    quantized = bool(config.actor_rollout_ref.rollout.get("quantization"))

    nixl.await_readers_before_publish = requires_version_barrier(
        backend=str(config.train_actor_rollout_ref.actor.strategy),
        train_dtype="cast" if quantized else train_dtype,
        rollout_dtype=train_dtype,
        colocate=bool(config.psrl.colocate_validate_and_train),
    )


def _resolve_write_gate(config: DictConfig) -> None:
    """Decide whether optimizer writes must preempt live readers, and record the decision.

    The gate is needed exactly when rollouts read the trainer's **live parameter bytes**
    and the optimizer overwrites them in place:

    * direct supply (the train ranks are the source, no PS copy);
    * matched precision (a cast or FP8 rollout reads a separate published copy, refreshed
      under the existing publish lock);
    * any rollout placement. `colocate_validate_and_train` only says validation shares
      GPUs with training; a rollout fleet on other nodes still reads live parameters, so
      the earlier "colocate means no shared memory" shortcut was wrong for that topology.

    Written into the config so the value a run used is visible in its dump, and an
    explicit boolean still wins (the failure-injection tests need to turn it off).
    """
    from psrl.utils.ps_mode import has_ps_data_plane

    nixl = config.psrl.nixl
    decided = nixl.get("await_write_gate", "auto")
    if isinstance(decided, bool):
        return

    def _normalise(dtype: str) -> str:
        return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(str(dtype), str(dtype))

    direct = not has_ps_data_plane(config.psrl.ps_mode)
    matched = _normalise(nixl.publish_train_dtype) == _normalise(nixl.publish_rollout_dtype)
    quantization = QuantizationPlan.from_config(nixl.get("publish_quantization"))
    reader_quantization = bool(nixl.get("reader_quantization", False))
    # Reader-side quantization also reads the trainer's live BF16 bytes, so it needs
    # the same epoch gate as a matched-precision read. The old publisher-side path
    # reads a separate publish copy and does not.
    nixl.await_write_gate = bool(
        direct
        and (
            (matched and not quantization.is_quantized)
            or (reader_quantization and quantization.is_quantized)
        )
    )
    if nixl.await_write_gate and isinstance(nixl.get("await_readers_before_publish"), bool):
        # The gate supersedes the publish-time drain for this path. Leaving both on would
        # make publish invalidate versions that the gate already proved single-version and
        # that readers are repairing.
        nixl.await_readers_before_publish = False


def _validate_nixl_lora(config: DictConfig) -> None:
    """Reject un-merged LoRA, which the weight transport cannot express (3.11).

    The converters walk `state_dict()` / `named_parameters()`, which under PEFT yield
    the frozen base weights plus `lora_A`/`lora_B`. vLLM keeps its adapters in
    `lora_a_stacked` / `lora_b_stacked`, which are plain tensors -- neither
    parameters nor registered buffers -- so NIXL has no landing site for them.

    Without this check the run starts and silently synchronises the *base* model:
    reward never moves and nothing errors. `merge=true` is expressible because the
    merged weights have the same shape as full fine-tuning.
    """
    model_cfg = config.train_actor_rollout_ref.model
    lora_cfg = model_cfg.get("lora", {}) or {}
    lora_rank = lora_cfg.get("rank", 0) or model_cfg.get("lora_rank", 0)
    if lora_rank <= 0 and model_cfg.get("lora_adapter_path") is None:
        return
    if not lora_cfg.get("merge", False):
        raise ValueError(
            "LoRA adapter synchronisation is not supported by the NIXL weight "
            "transport: vLLM stores adapters in lora_a_stacked/lora_b_stacked, which "
            "are neither parameters nor buffers, so they cannot be registered. "
            "Set model.lora.merge=true to synchronise merged weights, or disable LoRA. "
            "Leaving it unmerged would silently sync the base model only."
        )


def _validate_megatron_fp8_param(config: DictConfig) -> None:
    """Reject persistent FP8 parameters, which this engine cannot build.

    `fp8_param` asks Transformer Engine to keep linear weights as `QuantizedTensor`s.
    That only happens when the model is constructed inside `fp8_model_init`; the
    engine builds it directly, so the flag leaves bf16 parameters while changing the
    FP8 backward's weight-gradient accounting. A run with it set looks healthy for
    one step and then degrades sharply. FP8 hybrid autocast (`fp8=hybrid` with
    `fp8_param=False`) is the supported FP8 training path.
    """
    if config.train_actor_rollout_ref.actor.strategy != "megatron":
        return
    overrides = config.train_actor_rollout_ref.actor.megatron.get("override_transformer_config") or {}
    if overrides.get("fp8_param", False):
        raise ValueError(
            "train_actor_rollout_ref.actor.megatron.override_transformer_config.fp8_param "
            "is not supported: the engine does not build the model inside TE's fp8_model_init, "
            "so the parameters stay bf16 and the first optimizer update corrupts training. "
            "Use FP8 hybrid autocast (fp8=hybrid, fp8_param=False)."
        )
def mirror_bypass_config_to_actor(config: DictConfig) -> None:
    """Mirror ``algorithm.rollout_correction`` into the actor's static ``policy_loss``.

    The actor worker picks its loss function once, in ``init_model``, from its own config snapshot
    (a separate process). A runtime mutation on the driver never reaches it. Upstream therefore
    requires the actor-side mirror in the config (``docs/algo/rollout_corr.md``). Doing it here, in
    ``validate_config`` before ``init_workers``, is what makes bypass mode and the score-centering
    loss actually take effect. A no-op unless ``bypass_mode`` is set.
    """
    rollout_correction = config.algorithm.get("rollout_correction", None) or {}
    if not rollout_correction.get("bypass_mode", False):
        return
    actor_policy_loss = config.train_actor_rollout_ref.actor.policy_loss
    # The default (yaml) is "vanilla". Anything else means the user set a different loss explicitly,
    # which bypass would silently override. Fail loudly instead of discarding their choice.
    configured_loss_mode = actor_policy_loss.get("loss_mode", "vanilla")
    if configured_loss_mode not in ("vanilla", "bypass_mode"):
        raise ValueError(
            "algorithm.rollout_correction.bypass_mode=True requires "
            "train_actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode, but it is set to "
            f"{configured_loss_mode!r}. Remove that override or drop bypass mode."
        )
    with open_dict(actor_policy_loss):
        # A deep copy, so later mutations of the algorithm node cannot alias into the actor node.
        actor_policy_loss["rollout_correction"] = OmegaConf.create(OmegaConf.to_container(rollout_correction))
        actor_policy_loss["loss_mode"] = "bypass_mode"


def validate_subagent_training_config(config: DictConfig, batch_agg_mode: str):
    """
    Validate the subagent training policy against the batch aggregation mode.

    A mode that can put more than one trajectory row in one session is only meaningful
    when those rows stay one sample, which is what request aggregation provides.

    Args:
        config (DictConfig): The resolved training config.
        batch_agg_mode (str): The resolved `psrl.agentic_rl.batch_agg_mode`.

    Returns:
        SubagentTrainingPolicy: The resolved session policy.
    """
    from psrl.utils.subagent_training import SubagentTrainingPolicy

    policy = SubagentTrainingPolicy.from_mapping(config.psrl.agentic_rl.get("subagent_training", {}))
    if {policy.mode, policy.val_mode} & {"all", "selected"}:
        assert batch_agg_mode == REQUEST_AGG_MODE, (
            "psrl.agentic_rl.subagent_training mode/val_mode 'all' or 'selected' can put "
            "multiple rows in one session; those rows are one sample, so "
            "psrl.agentic_rl.batch_agg_mode must be 'request'."
        )
    return policy


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

    # Router replay lives on the training engine config, which Hydra mounts under a strategy-specific
    # key. Only the Megatron and VeOmni engine templates expose it, so a missing key means disabled.
    actor_config = config.train_actor_rollout_ref.actor
    rollout_config = config.gen_actor_rollout_ref.rollout
    capture_mode = rollout_config.get("sampler_distribution", "disabled")
    if capture_mode == "auto":
        from psrl.utils.sampler_distribution import resolve_capture_mode

        resolved = resolve_capture_mode("auto", top_p=rollout_config.top_p, top_k=rollout_config.top_k)
        # Write the resolved mode back so every consumer (rollout server, agent loops, trainer)
        # sees a concrete mode instead of ``auto``.
        with open_dict(rollout_config):
            rollout_config.sampler_distribution = resolved
        capture_mode = resolved
    if capture_mode not in ("disabled", "topk", "support"):
        raise ValueError(f"Unknown sampler_distribution mode {capture_mode!r}.")
    if capture_mode in ("topk", "support"):
        from psrl.utils.sampler_distribution import SamplerCapture, configure_capture

        capture = SamplerCapture.from_rollout_config(rollout_config)
        configure_capture(
            {
                "temperature": rollout_config.temperature,
                "top_p": rollout_config.top_p,
                "top_k": rollout_config.top_k,
                "min_p": rollout_config.get("min_p", 0.0),
            },
            capture,
            protocol="chat",
        )
        # Upstream's rollout validation derives its own support checks from the sampler head and
        # receipt, so no additional sampler_support flag is mirrored into veRL.
        if rollout_config.logprobs_mode != "processed_logprobs":
            raise ValueError("Sampler capture requires rollout.logprobs_mode=processed_logprobs.")
        if OmegaConf.select(rollout_config, "mtp.enable", default=False) or OmegaConf.select(
            rollout_config, "engine_kwargs.vllm.speculative_config", default=None
        ):
            raise ValueError("Sampler capture requires a non-speculative rollout sampler.")
    # Score centering must be enabled in the actor's *static* policy_loss: the worker selects its
    # loss_fn once in init_model, before this runtime mutation exists on the driver.
    mirror_bypass_config_to_actor(config)
    rollout_correction = config.algorithm.get("rollout_correction", None) or {}
    if rollout_correction.get("score_centering", False):
        from dataclasses import fields as dataclass_fields

        from verl.trainer.config.algorithm import RolloutCorrectionConfig

        # Validate the score-centering contract (bypass_mode=True, loss_type="reinforce",
        # rollout_is None/"token", no batch normalize) over the dataclass fields only.
        known_fields = {field.name for field in dataclass_fields(RolloutCorrectionConfig)}
        RolloutCorrectionConfig(**{key: value for key, value in rollout_correction.items() if key in known_fields})
        if capture_mode == "disabled":
            raise ValueError(
                "algorithm.rollout_correction.score_centering requires "
                "gen_actor_rollout_ref.rollout.sampler_distribution=topk or support (or auto with a "
                "truncated/untruncated sampler) so the sampler head is captured."
            )
        if (config.get("distillation") or {}).get("enabled", False):
            raise ValueError("score centering cannot be combined with distillation.")
    engine_mount = "megatron" if actor_config.strategy == "megatron" else "fsdp_config"
    actor_router_replay = OmegaConf.select(actor_config, f"{engine_mount}.router_replay.mode", default="disabled")
    rollout_routing_replay = config.gen_actor_rollout_ref.rollout.enable_rollout_routing_replay
    if actor_router_replay == "R3" and not rollout_routing_replay:
        raise ValueError(
            "Router replay mode R3 requires gen_actor_rollout_ref.rollout.enable_rollout_routing_replay=True."
        )
    if rollout_routing_replay and actor_router_replay != "R3":
        raise ValueError("Rollout routing replay is only valid with the actor engine's router_replay.mode='R3'.")
    _validate_megatron_fp8_param(config)

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
    validate_subagent_training_config(config, batch_agg_mode)
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
    #
    # `nixl_direct` is trainer-direct supply: rollouts read train ranks and no PS data
    # plane exists (design 3.3). It validates like the other NIXL modes but must not be
    # paired with anything that assumes a PS holds the weights, so those pairings are
    # refused here rather than failing mid-transfer.
    if config.psrl.ps_mode == "nixl_direct":
        assert config.psrl.nixl.server_ip == config.psrl.ps_manager_ip, (
            "PSManager IP and NIXL server IP must be the same"
        )
        assert config.train_actor_rollout_ref.actor.strategy != "fsdp", (
            "FSDP1 is not supported for NIXL because it uses flat_param"
        )
        _validate_nixl_offload_flags(config)
        _validate_nixl_lora(config)
        _resolve_version_barrier(config)
        _resolve_publish_format(config)
        _resolve_write_gate(config)
        assert not config.psrl.get("lmcache", {}).get("enable_p2p", False), (
            "psrl.lmcache.enable_p2p requires a PS to route through; it cannot be used "
            "with ps_mode=nixl_direct, where rollouts read train ranks directly"
        )
        # The train ranks are the weight source here, so rollouts RDMA-read them while
        # they are computing. That waits for DONE, which needs the NIXL progress thread.
        # DCP with the nvrx async strategy (persistent=True, spawn-based worker) is safe
        # alongside the progress thread because the worker process does not inherit the
        # parent's UCX / NIXL context.  The old mcore fork-based async path is not safe
        # and is no longer used (see verl/utils/megatron/dist_checkpointing.py).

    if config.psrl.ps_mode == "nixl_cpu" or config.psrl.ps_mode == "nixl_gpu":
        assert config.psrl.nixl.server_ip == config.psrl.ps_manager_ip, (
            "PSManager IP and NIXL server IP must be the same"
        )
        assert config.train_actor_rollout_ref.actor.strategy != "fsdp", (
            "FSDP1 is not supported for NIXL because it uses flat_param"
        )
        _validate_nixl_offload_flags(config)
        _validate_nixl_lora(config)
        _resolve_version_barrier(config)
        quantization = resolve_quantization_plan(config.psrl.nixl.get("weight_quantization"))
        assert not quantization.is_quantized, (
            "psrl.nixl.weight_quantization is a direct-supply format. The PS data plane "
            "stores one wire format that every rollout role loads, so it currently supports "
            "scheme=none only."
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
            "(the LMCache MP P2P tier cannot be flushed on weight update)."
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

    print("[validate_config] All configuration checks passed successfully!")
