"""Tests for `HOSTED_INLINE`, the chain-of-thought policy for a hosted endpoint.

A reasoning model behind a third-party API answers in `reasoning_content` and
leaves `content` empty. Agent harnesses read `content`, so they see nothing, take
no action, and the episode is wasted while the API reports 200 with hundreds of
completion tokens.

The fix belongs in the request, not in the response. Rewriting the reply would
have to guess whether a given `reasoning_content` is deliberation or the answer,
and a provider uses the field for both: measured on `openai/gpt-oss-20b`, some
turns carried a bare JSON command there and others carried "We need to output a
JSON with ...". Asking the endpoint not to split the fields removes the guess.
"""

import pytest
from psrl.utils.agent.thinking import (
    DISABLE_THINKING,
    HOSTED_INLINE,
    MULTI_THINKING,
    MULTI_TRAJ,
    SUPPORTED_THINKING_TEMPLATES,
    harness_extra_body,
    requires_accumulating_template,
    validate_thinking_template,
    wants_inline_reasoning,
    wants_thinking_disabled,
)


@pytest.mark.cpu_test
def test_hosted_inline_asks_the_endpoint_to_stop_splitting_the_reply():
    """Both knobs are needed: inline the reasoning, and stop generating it."""
    extra_body = harness_extra_body(HOSTED_INLINE)

    assert extra_body["separate_reasoning"] is False
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}
    assert wants_inline_reasoning(HOSTED_INLINE)
    assert wants_thinking_disabled(HOSTED_INLINE)


@pytest.mark.cpu_test
def test_hosted_inline_needs_no_local_chat_template():
    """A hosted endpoint renders with its own template, so there is none to set.

    `multi_thinking` is the mode that requires a local accumulating template, and
    `validate_config` rejects it without one. Reusing that mode against a hosted
    API would demand a file that cannot affect the provider's rendering.
    """
    assert requires_accumulating_template(MULTI_THINKING) is True
    assert requires_accumulating_template(HOSTED_INLINE) is False


@pytest.mark.cpu_test
def test_tito_modes_send_nothing_which_is_why_they_fail_on_a_hosted_api():
    """`multi_traj` relies on TITO to reconcile forks, so it shapes no request.

    That is correct when PSRL serves the model and wrong against a hosted
    endpoint, where nothing else asks the provider to inline its reasoning. This
    is the actual cause of the empty-content run, so it is pinned here.
    """
    assert harness_extra_body(MULTI_TRAJ) == {}


@pytest.mark.cpu_test
def test_every_mode_stays_validatable():
    """A new mode must be accepted by the validator and be one of the known set."""
    for mode in SUPPORTED_THINKING_TEMPLATES:
        assert validate_thinking_template(mode) == mode

    assert HOSTED_INLINE in SUPPORTED_THINKING_TEMPLATES
    with pytest.raises(ValueError, match="thinking_template"):
        validate_thinking_template("nonsense")


@pytest.mark.cpu_test
def test_hosted_inline_differs_from_plain_disable_thinking():
    """Turning thinking off is not enough by itself.

    `disable_thinking` leaves `separate_reasoning` at the SMG default of true. A
    provider that still emits a reasoning field then keeps `content` empty, so the
    hosted mode has to carry both knobs.
    """
    assert "separate_reasoning" not in harness_extra_body(DISABLE_THINKING)
    assert harness_extra_body(HOSTED_INLINE)["separate_reasoning"] is False
