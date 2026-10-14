"""Standalone deployment-mode resolution (no heavy imports).

`resolve_deployment_mode` maps the unified `psrl.deployment.mode` config entry
to the concrete `elastic` flags that the rest of the codebase reads. It must
run BEFORE any code that branches on `elastic.enable` / `enable_trainer_pool`
/ `enable_policy`. In particular it must run before
`main_ppo.TaskRunner.init_resource_pool_mgr` (which lays out resource pools)
and before `PSRL_RayPPOTrainer.__init__` derives its mode flags.

It is intentionally free of ray / torch / nixl imports so it can be called from
`main_ppo` at the very top of `TaskRunner.run` without triggering implicit
`ray.init()`.
"""

from omegaconf import ListConfig, open_dict

VALID_DEPLOYMENT_MODES = (
    "disaggregated",
    "colocated",
    "rollout_rm_colocated",
    "trainer_pool_only",
    "elastic_rl",
)


def expand_ngpus_per_node(ngpus_per_node, nnodes, field_name: str = "ngpus_per_node") -> list[int]:
    """Normalize scalar or per-node GPU specs to one integer per node.

    `ngpus_per_node` may be a scalar, a Python sequence, or an OmegaConf
    ListConfig. Scalars are repeated `nnodes` times; sequences must already
    match `nnodes`.
    """
    nnodes = int(nnodes)
    if nnodes < 0:
        raise ValueError(f"nnodes must be >= 0, got {nnodes}.")

    if isinstance(ngpus_per_node, (list, tuple, ListConfig)):
        spec = [int(v) for v in ngpus_per_node]
        if len(spec) != nnodes:
            raise ValueError(
                f"{field_name} has {len(spec)} entries but shared_nnodes={nnodes}; "
                "provide one GPU count per shared node."
            )
    else:
        spec = [int(ngpus_per_node)] * nnodes

    if any(v < 0 for v in spec):
        raise ValueError(f"{field_name} values must be >= 0, got {spec}.")
    return spec


def resolve_deployment_mode(config) -> str:
    """Resolve `psrl.deployment.mode` into concrete `elastic` flags.

    Mutates `config` in place so downstream code sees a consistent flag set.
    When `mode` is null, infer from existing flags for backward compatibility.
    Returns the resolved mode string. Idempotent: safe to call multiple times.
    """
    deployment = config.psrl.deployment
    mode = deployment.get("mode", None)
    if mode is None:
        # When `mode` is unset, prefer existing `elastic.enable`, then colocate.
        if deployment.elastic.get("enable", False):
            mode = "elastic_rl"
        elif config.psrl.get("colocate", False):
            # Legacy sync colocate path; keep disaggregated semantics for pool
            # layout (the sync path does not use elastic pools).
            mode = "disaggregated"
        else:
            mode = "disaggregated"
    mode = str(mode)
    if mode not in VALID_DEPLOYMENT_MODES:
        raise ValueError(f"Invalid psrl.deployment.mode={mode!r}; expected one of {VALID_DEPLOYMENT_MODES}.")

    with open_dict(config):
        config.psrl.deployment.mode = mode
        elastic = config.psrl.deployment.elastic
        if mode == "disaggregated":
            elastic.enable = False
            elastic.enable_trainer_pool = False
        elif mode == "elastic_rl":
            elastic.enable = True
            # enable_trainer_pool and enable_policy stay as user-configured.
        elif mode == "trainer_pool_only":
            # Disaggregated baseline (rollout/rm on independent pools, always
            # awake) + a fixed number of extra rollout/rm replicas on train_pool
            # that are woken while the trainer is idle. No elastic auto-scaling,
            # no shared_rollout_pool. NIXL is required for actor sleep/wake.
            elastic.enable = False
            elastic.enable_trainer_pool = False
            elastic.enable_policy = False
        elif mode == "colocated":
            # All three roles share train_pool (colocated), time-multiplexed per
            # training step. Reuse elastic SubRayResourcePool / bundle-mapping /
            # coordinator sleep/wake + trainer-pool actor NIXL sleep/wake infra,
            # but disable auto-scaling. SleepWakeOrchestrator drives phases.
            elastic.enable = True
            elastic.enable_trainer_pool = True
            elastic.enable_policy = False
        elif mode == "rollout_rm_colocated":
            # rollout+rm share shared_rollout_pool (time-multiplexed per buffer).
            # Trainer stays on a separate train_pool, pipelined via staleness.
            elastic.enable = True
            elastic.enable_trainer_pool = False
            elastic.enable_policy = False
    validate_elastic_validation(config)
    return mode


def validate_elastic_validation(config) -> None:
    """
    Require dedicated validation replicas on trainer GPUs for elastic deployments.
    """
    if not config.psrl.deployment.elastic.enable:
        return
    if not config.psrl.colocate_validate_and_train:
        raise ValueError("Elastic deployments require psrl.colocate_validate_and_train=True.")
    if config.psrl.fuse_rollout_with_validate:
        raise ValueError("Elastic deployments require psrl.fuse_rollout_with_validate=False.")
    if int(config.psrl.deployment.n_validate_instances) <= 0:
        raise ValueError("Elastic deployments require psrl.deployment.n_validate_instances > 0.")
    if config.psrl.ps_mode not in ("nixl_cpu", "nixl_gpu"):
        raise ValueError(
            "Elastic trainer and validation storage transitions require psrl.ps_mode=nixl_cpu or nixl_gpu."
        )


def validate_trainer_sleep_optimizer_offload(config, mode: str | None = None) -> None:
    """Require optimizer offload whenever a deployment can sleep the trainer.

    Trainer sleep releases process-wide TMS allocations. Model weights are
    restored from the parameter server after wake-up, but optimizer state is
    not. Keeping optimizer tensors on CPU while the trainer is idle is therefore
    a correctness requirement, not only a memory optimization.
    """
    deployment = config.psrl.deployment
    resolved_mode = str(mode if mode is not None else deployment.get("mode", "disaggregated"))
    sleep_reasons = []
    if bool(deployment.elastic.get("enable_trainer_pool", False)):
        sleep_reasons.append("psrl.deployment.elastic.enable_trainer_pool=True")
    if resolved_mode == "trainer_pool_only":
        sleep_reasons.append("psrl.deployment.mode=trainer_pool_only")
    if bool(config.psrl.get("colocate_validate_and_train", False)):
        sleep_reasons.append("psrl.colocate_validate_and_train=True")
    if not sleep_reasons:
        return

    actor = config.train_actor_rollout_ref.actor
    strategy = str(actor.strategy)
    if strategy in ("fsdp", "fsdp2"):
        config_path = "train_actor_rollout_ref.actor.fsdp_config.optimizer_offload"
        optimizer_offload = bool(actor.fsdp_config.get("optimizer_offload", False))
    elif strategy == "megatron":
        config_path = "train_actor_rollout_ref.actor.megatron.optimizer_offload"
        optimizer_offload = bool(actor.megatron.get("optimizer_offload", False))
    else:
        raise ValueError(
            f"Trainer sleep is enabled by {', '.join(sleep_reasons)}, but actor strategy {strategy!r} "
            "does not define a supported optimizer-offload path."
        )

    if not optimizer_offload:
        raise ValueError(
            f"Trainer sleep is enabled by {', '.join(sleep_reasons)}; set {config_path}=True. "
            "Without optimizer offload, TMS sleep can discard optimizer state that is not restored by pull_model()."
        )
