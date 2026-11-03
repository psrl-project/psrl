"""Tests for the dataset prompt builder."""

from __future__ import annotations

import pytest
from examples.math_rl.prompt import SOLUTION_PATH, build_messages, build_task_prompt


def test_prompt_carries_problem_and_constraints():
    prompt = build_task_prompt("Prove that 2 is prime.")
    assert "Prove that 2 is prime." in prompt
    assert SOLUTION_PATH in prompt
    assert "Do not use web search or subagents." in prompt
    assert "/workspace/scratch" in prompt


def test_messages_use_the_openai_role_shape():
    messages = build_messages("Prove that 2 is prime.")
    assert len(messages) == 1
    assert set(messages[0]) == {"role", "content"}
    assert messages[0]["role"] == "user"
    assert "Prove that 2 is prime." in messages[0]["content"]


def test_empty_problem_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        build_task_prompt("   ")
