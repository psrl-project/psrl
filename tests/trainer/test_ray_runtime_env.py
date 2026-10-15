import pytest
from psrl.trainer.constants_ppo import get_ppo_ray_runtime_env

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("UCX_TLS", "tcp,cuda_copy,cuda_ipc"),
        ("UCX_NET_DEVICES", "eth0"),
        ("UCX_MAX_RMA_RAILS", "1"),
        ("UCX_LOG_LEVEL", "debug"),
        ("NIXL_PLUGIN_DIR", "/test/nixl/plugins"),
        ("LD_LIBRARY_PATH", "/test/nixl/lib:/test/ucx/lib"),
        ("PYTHONPATH", "/test/nixl/python"),
    ],
)
def test_driver_communication_environment_is_explicit_in_ray_runtime_env(monkeypatch, key, value):
    monkeypatch.setenv(key, value)

    env_vars = get_ppo_ray_runtime_env()["env_vars"]

    assert env_vars[key] == value, f"Driver communication setting {key!r} must reach Ray workers."


def test_unset_ucx_transports_are_not_overridden(monkeypatch):
    monkeypatch.delenv("UCX_TLS", raising=False)

    assert "UCX_TLS" not in get_ppo_ray_runtime_env()["env_vars"], "Ray must preserve UCX transport auto-selection."


def test_ucx_overrides_do_not_change_defaults_for_subsequent_jobs(monkeypatch):
    monkeypatch.setenv("UCX_TLS", "tcp")
    get_ppo_ray_runtime_env()
    monkeypatch.delenv("UCX_TLS")

    assert "UCX_TLS" not in get_ppo_ray_runtime_env()["env_vars"], (
        "A job transport override must not alter subsequent job defaults."
    )


def test_unrelated_driver_environment_is_not_copied(monkeypatch):
    monkeypatch.setenv("PSRL_UNRELATED_TEST_SETTING", "private")

    assert "PSRL_UNRELATED_TEST_SETTING" not in get_ppo_ray_runtime_env()["env_vars"], (
        "Runtime environment propagation must remain limited to communication settings."
    )
