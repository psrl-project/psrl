"""
Resolve explicit parser names against the bundled upstream implementations.

Dataset preparation chooses the parser. Unknown names are errors, so a typo
cannot silently select a different scoring algorithm during training.
"""

from __future__ import annotations

import hashlib
import inspect
import textwrap
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from types import ModuleType

DEFAULT_PARSER = "parse_log_pytest"
SMITH_PARSER_PREFIX = "swesmith_log_parser_"
Parser = Callable[..., dict[str, str]]

_VENDOR_DIR = Path(__file__).resolve().parent / "_vendor"
# Enough to let `schema` import and the parser aggregator resolve. Individual
# language modules are covered by the import guards in `parser_registry`.
_VENDOR_REQUIRED_FILES = (
    "_constants.py",
    "_utils.py",
    "log_parsers/__init__.py",
    "swesmith_parsers.py",
)
_VENDOR_MISSING_HINT = (
    "The vendored grading parser bundle is missing or incomplete. Restore it with "
    "`git checkout -- examples/mini_swe/grading/_vendor`, or regenerate it with "
    "`python -m examples.mini_swe.grading.vendor_parsers` in an environment that has the "
    "upstream versions pinned in `_vendor/PROVENANCE.md`."
)


def require_vendor_bundle() -> None:
    """
    Fail fast with guidance when the committed vendored parser bundle is absent.

    ``schema`` imports ``_vendor/_constants`` at module scope, so a pruned checkout
    would otherwise surface as a bare ModuleNotFoundError deep in the package rather
    than naming the fix. Also catches deleting ``log_parsers/__init__.py``, which
    would otherwise import as an empty namespace package.
    """
    missing = [name for name in _VENDOR_REQUIRED_FILES if not (_VENDOR_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"{_VENDOR_MISSING_HINT} Missing: {', '.join(missing)}.")


def _load_vendor_log_parsers() -> ModuleType:
    """
    Import the vendored upstream parser aggregator, failing fast with guidance.

    The bundle is committed so it can be packaged into the sandbox payload
    offline. An ImportError here means the checkout is incomplete or pruned.
    """
    try:
        from ._vendor import log_parsers
    except ImportError as exc:
        raise RuntimeError(_VENDOR_MISSING_HINT) from exc
    return log_parsers


def smith_parser_name(source: str) -> str:
    """
    Return the shared identifier used by preparation and parser vendoring.
    """
    digest = hashlib.sha256(textwrap.dedent(source).encode()).hexdigest()[:12]
    return f"{SMITH_PARSER_PREFIX}{digest}"


def _adapt(parser: Parser) -> Parser:
    """
    Adapt a SWE-smith parser to the common two argument call convention.
    """

    def parse(log: str, test_spec: object = None) -> dict[str, str]:
        return parser(log)

    return parse


@lru_cache(maxsize=1)
def parser_registry() -> dict[str, Parser]:
    """
    Index the modules exported by the upstream aggregator once per process.
    """
    log_parsers = _load_vendor_log_parsers()
    try:
        from ._vendor.swesmith_parsers import SWESMITH_PARSER_BY_DIGEST
    except ImportError as exc:
        raise RuntimeError(_VENDOR_MISSING_HINT) from exc

    parsers = {}
    for module in vars(log_parsers).values():
        if not inspect.ismodule(module) or not module.__name__.startswith(log_parsers.__name__ + "."):
            continue
        for name, parser in vars(module).items():
            if name.startswith("parse_log") and callable(parser):
                if name in parsers and parsers[name] is not parser:
                    raise ValueError(f"Duplicate grading parser name: {name!r}.")
                parsers[name] = parser
    parsers.update({f"{SMITH_PARSER_PREFIX}{digest}": _adapt(p) for digest, p in SWESMITH_PARSER_BY_DIGEST.items()})
    return parsers


def resolve_parser(name: str) -> Parser:
    """
    Resolve an explicit parser name, rejecting unsupported prepared data.
    """
    try:
        return parser_registry()[name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Unknown grading parser: {name!r}. Rebuild the parser bundle or reprepare the data."
        ) from exc


def parser_name_for_repo(repo: str) -> str:
    """
    Resolve a Verified dataset repository during preparation.
    """
    log_parsers = _load_vendor_log_parsers()

    try:
        return log_parsers.MAP_REPO_TO_PARSER[repo].__name__
    except KeyError as exc:
        raise ValueError(f"No grading parser registered for repository {repo!r}.") from exc
