"""
Validate storage ownership for FSDP actors using NIXL parameter synchronization.
"""

import pytest
from omegaconf import OmegaConf
from psrl.utils.config.validation import validate_nixl_actor_offload

pytestmark = pytest.mark.cpu_test


def _config(ps_mode, strategy, **offload):
    return OmegaConf.create(
        {
            "psrl": {"ps_mode": ps_mode},
            "train_actor_rollout_ref": {"actor": {"strategy": strategy, "fsdp_config": offload}},
        }
    )


@pytest.mark.parametrize("ps_mode", ["nixl_cpu", "nixl_gpu"])
@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
@pytest.mark.parametrize("option", ["param_offload", "offload_policy"])
def test_nixl_rejects_fsdp_parameter_offloading(ps_mode, strategy, option):
    with pytest.raises(ValueError, match=rf"fsdp_config\.{option}=False"):
        validate_nixl_actor_offload(_config(ps_mode, strategy, **{option: True}))


@pytest.mark.parametrize("ps_mode", ["nixl_cpu", "nixl_gpu"])
def test_nixl_allows_optimizer_offloading(ps_mode):
    validate_nixl_actor_offload(_config(ps_mode, "fsdp2", optimizer_offload=True))


@pytest.mark.parametrize("ps_mode,strategy", [("cpu", "fsdp2"), ("cpu_ref", "fsdp2"), ("nixl_cpu", "megatron")])
def test_other_synchronization_backends_keep_their_offload_settings(ps_mode, strategy):
    validate_nixl_actor_offload(_config(ps_mode, strategy, param_offload=True, offload_policy=True))
