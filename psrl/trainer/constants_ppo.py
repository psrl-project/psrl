import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": "VERSION",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "VLLM_SKIP_P2P_CHECK": "1",  # Avoid the startup cost of the vLLM P2P probe.
        "VERL_DATAPROTO_SERIALIZATION_METHOD": "numpy",
        "PSRL_LOGGING_LEVEL": "INFO",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_CUMEM_ENABLE": "0",
    },
}

_RAY_ENVIRONMENT_KEYS = (
    "PYTHONPATH",
    "PYTHONPYCACHEPREFIX",
    "LD_LIBRARY_PATH",
    "NIXL_PLUGIN_DIR",
)


def get_ppo_ray_runtime_env():
    """
    Return worker settings with explicit driver overrides.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": {key: os.environ.get(key, value) for key, value in PPO_RAY_RUNTIME_ENV["env_vars"].items()},
        **({"working_dir": None} if working_dir is None else {}),
    }

    for key, val in os.environ.items():
        if val and (key in _RAY_ENVIRONMENT_KEYS or key.startswith("UCX_")):
            runtime_env["env_vars"][key] = val

    return runtime_env
