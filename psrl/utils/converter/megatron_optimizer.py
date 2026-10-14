def sync_master_params_from_model(engine) -> None:
    """Resync the DistributedOptimizer's fp32 master params from model (bf16) params.

    Call once after the initial PS NIXL pull when the actor is empty-initialized.
    Without this, the first optimizer.step() writes the garbage fp32 master params
    (built during empty init) back over the correctly pulled bf16 weights.

    Handles optimizer offload: loads optimizer to GPU before sync and offloads again
    after, so the fp32 master copy and GPU-resident model params are co-located.
    """
    from verl.utils.megatron_utils import load_megatron_optimizer, offload_megatron_optimizer

    assert engine.optimizer is not None, (
        "sync_master_params_from_model requires a built optimizer. "
        "engine.optimizer is None (forward_only or not yet initialized?)"
    )
    if engine._is_offload_optimizer:
        load_megatron_optimizer(engine.optimizer)
    opts = getattr(engine.optimizer, "chained_optimizers", [engine.optimizer])
    for opt in opts:
        assert hasattr(opt, "_copy_model_params_to_main_params"), (
            f"{type(opt).__name__} must implement _copy_model_params_to_main_params for fp32 master param resync"
        )
        opt._copy_model_params_to_main_params()
    if engine._is_offload_optimizer:
        offload_megatron_optimizer(engine.optimizer)
