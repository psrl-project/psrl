"""Tests for serving backend selection and resolution."""

import json

import pytest
from omegaconf import OmegaConf
from psrl.batch_rollout.serving.base import BackendHandle, build_serving_backend

_NODE_IP = "psrl.batch_rollout.serving.openai_api.backend.ray.util.get_node_ip_address"


def _config(**serving):
    return OmegaConf.create(
        {
            "batch_rollout": {
                "output_dir": "/tmp/batch_rollout_test",
                "serving": {"name": "openai_api", **serving},
            },
            "gen_actor_rollout_ref": {"model": {"path": "/models/qwen"}},
        }
    )


@pytest.mark.cpu_test
def test_unknown_backend_names_the_valid_choices():
    """A typo must not fail later as a confusing attribute error."""
    config = _config()
    config.batch_rollout.serving.name = "nope"

    with pytest.raises(ValueError, match="openai_api, smg_local"):
        build_serving_backend(config)


@pytest.mark.cpu_test
def test_openai_backend_requires_an_endpoint():
    """Neither a URL nor an endpoints file means there is nothing to call."""
    backend = build_serving_backend(_config(api_base_url="", endpoints_file=""))

    with pytest.raises(ValueError, match="api_base_url"):
        backend.start()


@pytest.mark.cpu_test
def test_openai_backend_reads_a_fleet_endpoints_file(tmp_path, monkeypatch):
    """A fleet launched by psrl.eval.serve is consumed through its own manifest."""
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_text(
        json.dumps(
            {
                "served_model_name": "qwen35-4b",
                "n_endpoints": 1,
                "endpoints": [{"url": "http://127.0.0.1:8000/v1", "healthy": True}],
            }
        ),
        encoding="utf-8",
    )

    captured = {}

    def fake_launch(upstream_url, transcript_dir, **kwargs):
        captured["upstream_url"] = upstream_url
        return "http://10.0.0.1:8400", None

    monkeypatch.setattr("psrl.batch_rollout.serving.openai_api.backend.launch_session_adapter", fake_launch)
    monkeypatch.setattr(_NODE_IP, lambda: "10.0.0.1")

    config = _config(endpoints_file=str(endpoints), api_base_url="", model_name="")
    handle = build_serving_backend(config).start()

    assert captured["upstream_url"] == "http://127.0.0.1:8000/v1"
    assert handle.model_name == "qwen35-4b"
    assert handle.session_router_url == "http://10.0.0.1:8400"


@pytest.mark.cpu_test
def test_openai_backend_never_claims_token_capture(monkeypatch):
    """An endpoint returning text cannot support a token dump, and must say so."""
    monkeypatch.setattr(
        "psrl.batch_rollout.serving.openai_api.backend.launch_session_adapter",
        lambda upstream_url, transcript_dir, **kwargs: ("http://10.0.0.1:8400", None),
    )
    monkeypatch.setattr(_NODE_IP, lambda: "10.0.0.1")

    handle = build_serving_backend(_config(api_base_url="http://api/v1")).start()

    assert handle.supports_token_capture is False
    # A session surface is still provided, or session-scoped loops could not run.
    assert handle.session_router_url


@pytest.mark.cpu_test
def test_model_name_falls_back_to_the_checkpoint_path(monkeypatch):
    """The record is stamped with this, so it must never be blank."""
    monkeypatch.setattr(
        "psrl.batch_rollout.serving.openai_api.backend.launch_session_adapter",
        lambda upstream_url, transcript_dir, **kwargs: ("http://10.0.0.1:8400", None),
    )
    monkeypatch.setattr(_NODE_IP, lambda: "10.0.0.1")

    handle = build_serving_backend(_config(api_base_url="http://api/v1", model_name="")).start()

    assert handle.model_name == "/models/qwen"


@pytest.mark.cpu_test
def test_backend_handle_defaults_are_conservative():
    """Defaults must not imply a capability a backend has not declared."""
    handle = BackendHandle(api_base_url="http://api/v1", model_name="m")

    assert handle.session_router_url is None
    assert handle.supports_token_capture is False
    assert handle.rollout_gateway_url == ""
