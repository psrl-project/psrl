import os

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env as get_verl_ppo_ray_runtime_env

# Defaults for the SMG and NIXL data path. veRL owns platform and engine sensitive policy, so
# PSRL only overlays its own values here.
PSRL_RAY_ENV_DEFAULTS = {
    "TOKENIZERS_PARALLELISM": "false",
    "NCCL_DEBUG": "VERSION",
    "VLLM_LOGGING_LEVEL": "WARN",
    "VLLM_SKIP_P2P_CHECK": "1",  # Avoid the startup cost of the vLLM P2P probe.
    "VERL_DATAPROTO_SERIALIZATION_METHOD": "numpy",
    "PSRL_LOGGING_LEVEL": "INFO",
    "NCCL_CUMEM_ENABLE": "0",
}

_HOST_RUNTIME_ENV_KEYS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
    "no_proxy",
    "NO_PROXY",
    "PSRL_HARNESS_RUNTIME_ROOT",
    "PSRL_LOGGING_LEVEL",
    "WANDB_MODE",
    "NIXL_PLUGIN_DIR",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)


def get_ppo_ray_runtime_env(config=None):
    """
    Build the veRL runtime environment with PSRL-specific defaults.

    An environment value already present in the launcher's environment always wins, so a
    resolved config can still be overridden from the shell.

    Args:
        config: Optional resolved or unresolved training configuration. Forwarded to veRL so
            it can apply engine and platform sensitive settings.

    Returns:
        dict: Ray runtime environment suitable for `ray.init`.
    """
    runtime_env = get_verl_ppo_ray_runtime_env(config)
    runtime_env.setdefault("env_vars", {})

    for key, value in PSRL_RAY_ENV_DEFAULTS.items():
        if os.environ.get(key) is None:
            runtime_env["env_vars"][key] = value

    for key in ("PYTHONPATH", "PYTHONPYCACHEPREFIX"):
        val = os.environ.get(key)
        if val:
            runtime_env["env_vars"][key] = val

    for key in _HOST_RUNTIME_ENV_KEYS:
        val = os.environ.get(key)
        if val:
            runtime_env["env_vars"][key] = val

    return runtime_env
