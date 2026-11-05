"""Tests for the verifier preflight helpers."""

from __future__ import annotations

from examples.math_rl.verify_probe import _health_url


def test_the_health_url_is_derived_from_both_verify_endpoints():
    assert _health_url("http://host/verify") == "http://host/health"
    assert _health_url("http://host/verify/stream") == "http://host/health"
    assert _health_url("http://host/verify/") == "http://host/health"
