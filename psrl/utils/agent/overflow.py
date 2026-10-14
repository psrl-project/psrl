"""Convert vLLM prompt-overflow HTTP 400 errors into terminal exceptions.

Litellm does not recognize vLLM's wording, so unclassified errors would retry
until the episode times out.
"""

from __future__ import annotations

import functools
from typing import TypeVar

from psrl.utils.common.http_utils import PromptOverflowError  # re-export

_T = TypeVar("_T")

_VLLM_OVERFLOW_MARKERS = (
    "maximum model length",
    "decoder prompt",
)


def is_prompt_overflow(exc: Exception) -> bool:
    """Return whether *exc* is a vLLM context-window overflow (HTTP 400).

    Harbor's ``LiteLLMModel._is_context_length_error`` and litellm's own
    ``ExceptionCheckers`` both miss vLLM's wording, so agent loops that drive
    Harbor must classify the raw ``BadRequestError`` themselves.
    """
    message = str(exc).lower()
    return any(marker in message for marker in _VLLM_OVERFLOW_MARKERS)


def ensure_overflow_handling(model: _T) -> _T:
    """Apply vLLM prompt-overflow detection to a model instance (idempotent).

    Wraps ``model._query`` so that a vLLM 400 overflow is converted to
    ``PromptOverflowError`` and registered in ``model.abort_exceptions`` to
    short-circuit the tenacity retry loop.
    """
    if PromptOverflowError in model.abort_exceptions:
        return model

    original_query = model._query

    @functools.wraps(original_query)
    def _wrapped_query(messages, **kwargs):
        try:
            return original_query(messages, **kwargs)
        except PromptOverflowError:
            raise
        except Exception as exc:
            if is_prompt_overflow(exc):
                raise PromptOverflowError(str(exc)) from exc
            raise

    model._query = _wrapped_query
    model.abort_exceptions = [*model.abort_exceptions, PromptOverflowError]
    return model


def handle_prompt_overflow(cls):
    """Class decorator: adds vLLM overflow detection to a LitellmModel subclass.

    Equivalent to calling ``ensure_overflow_handling(instance)`` after every
    ``__init__``, but applied statically at class-definition time.
    """
    original_init = cls.__init__
    original_query = cls._query

    @functools.wraps(original_init)
    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if PromptOverflowError not in self.abort_exceptions:
            self.abort_exceptions = [*self.abort_exceptions, PromptOverflowError]

    @functools.wraps(original_query)
    def _patched_query(self, messages, **kwargs):
        try:
            return original_query(self, messages, **kwargs)
        except PromptOverflowError:
            raise
        except Exception as exc:
            if is_prompt_overflow(exc):
                raise PromptOverflowError(str(exc)) from exc
            raise

    cls.__init__ = _patched_init
    cls._query = _patched_query
    return cls
