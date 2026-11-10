import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test

MODULE_PATH = Path(__file__).parents[2] / "scripts" / "audit_prose_style.py"
SPEC = importlib.util.spec_from_file_location("audit_prose_style", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None, "Could not load the prose audit module."
AUDIT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT_MODULE
SPEC.loader.exec_module(AUDIT_MODULE)

PATH = Path("sample.py")
audit_non_python = AUDIT_MODULE.audit_non_python
audit_python = AUDIT_MODULE.audit_python


def _codes(findings) -> list[str]:
    return [finding.code for finding in findings]


def test_audits_python_prose_surfaces() -> None:
    source = '''"""Module text — detail."""

import logging

logger = logging.getLogger(__name__)

# First clause; second clause.
def run(value: str) -> None:
    assert value, "Value is required; provide one."
    logger.info("Running — please wait.")
    print("Output; details.")
'''

    codes = _codes(audit_python(PATH, source))

    assert codes.count("PS001") == 5


def test_preserves_f_string_boundaries() -> None:
    source = """
import logging

logger = logging.getLogger(__name__)
value = "ready"
logger.info(f"State {value} is valid.")
"""

    assert "PS003" not in _codes(audit_python(PATH, source))


def test_ignores_punctuation_inside_inline_code() -> None:
    source = '''"""Compute `end - start` for the interval."""\n'''

    assert "PS002" not in _codes(audit_python(PATH, source))


def test_requires_messages_for_production_assertions() -> None:
    source = "assert value\n"

    assert _codes(audit_python(Path("psrl/module.py"), source)) == ["PS007"]
    assert _codes(audit_python(Path("tests/test_module.py"), source)) == []


def test_rejects_long_comment_blocks_and_bad_markers() -> None:
    source = """
# first line

# second line

# third line
value = 1

# TODO(owner): lowercase message
other = 2
"""

    codes = _codes(audit_python(PATH, source))

    assert codes.count("PS006") == 1
    assert codes.count("PS005") == 2


def test_audits_shell_comments_without_touching_quoted_hashes() -> None:
    source = """
value="# not a comment; still data"
# Human prose; another clause.
"""

    findings = list(audit_non_python(Path("sample.sh"), source))

    assert _codes(findings) == ["PS001"]
    assert findings[0].line == 3
