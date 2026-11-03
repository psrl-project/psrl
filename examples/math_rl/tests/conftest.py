"""Shared fixtures for the math RL tests."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def source_record() -> dict:
    return {
        "uid": "tbs_probs_example_01",
        "problem": "Prove that the sum of the first n odd integers is n squared.",
        "solution": "",
        "msc_tag": None,
        "references": ["https://example.invalid/paper"],
        "extra": {"original_field": "Algebra", "difficulty": "Medium"},
    }


@pytest.fixture
def combined_record() -> dict:
    return {
        "uid": "danus_open_problems_m_001",
        "problem": "Determine whether every connected vertex-transitive graph of order 2pq has a Hamilton cycle.",
        "solution": "",
        "category": None,
        "references": ["A paper (https://example.invalid/paper)"],
        "extra": {"direction": "Graph Theory"},
        "difficulty": "1\n",
        "publicity": "2\n",
        "status": "resolved\n",
        "difficulty_reason": "The statement is refuted by one explicit graph.",
        "publicity_reason": "Narrow but connected to a classical conjecture.",
        "status_reason": "The Coxeter graph is a known counterexample.",
        "literature_reviewed_at": "2026-08-03T00:00:00+00:00",
    }


@pytest.fixture
def word_counter():
    return lambda text: len(text.split())
