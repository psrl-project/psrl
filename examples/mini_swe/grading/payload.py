"""
Build a reproducible stdlib only grading zipapp and its sandbox command.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shlex
import zipfile
from functools import lru_cache
from pathlib import Path

from .schema import GradingResult

_HERE = Path(__file__).resolve().parent
_PACKAGE = "psrl_grading"
_RUNTIME_MODULES = ("driver.py", "schema.py", "scoring.py", "parsers.py")
DRIVER_ZIP_PATH = "/tmp/psrl-grader.zip"
INPUT_PATH = "/tmp/psrl-grade-input.json"
EVAL_SCRIPT_PATH = "/tmp/psrl-eval.sh"
LOG_PATH = "/tmp/psrl-eval.log"
SCORECARD_PATH = "/tmp/psrl-scorecard.json"
EDITABLE_PROBE_PATH = "/tmp/psrl-editable-probe.py"

# NOTE(codex): Task images may activate Python 3.6 in their shell profile.
# Prefer an independent interpreter for the driver, leaving the tests' PATH intact.
DEFAULT_DRIVER_PYTHONS = (
    "/opt/miniconda3/bin/python3",
    "/usr/local/bin/python3",
    "/usr/bin/python3",
    "python3",
    "python",
)
MIN_DRIVER_PYTHON = (3, 9)

# One row of the ``PROVENANCE.md`` digest table, e.g. ``| `_utils.py` | `1926...78` |``.
_DIGEST_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*$")
_VENDOR_STALE_HINT = (
    "Regenerate the bundle with `python -m examples.mini_swe.grading.vendor_parsers`, or "
    "restore it with `git checkout -- examples/mini_swe/grading/_vendor`."
)


def _assert_vendor_bundle_intact() -> None:
    """
    Fail fast when the committed vendored bundle is missing or stale.

    The bundle ships verbatim into every grading sandbox, so a missing file or a
    hand-edited parser would silently change rewards. ``PROVENANCE.md`` records
    the expected sha256 of each file. Compare against it before the payload is
    built, so a bad checkout stops training instead of skewing it.
    """
    provenance = _HERE / "_vendor" / "PROVENANCE.md"
    try:
        text = provenance.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Missing _vendor/PROVENANCE.md. {_VENDOR_STALE_HINT}") from exc

    expected: dict[str, str] = {}
    for line in text.splitlines():
        match = _DIGEST_ROW.match(line)
        if match:
            expected[match.group(1)] = match.group(2)
    if not expected:
        raise RuntimeError(f"Could not read parser digests from _vendor/PROVENANCE.md. {_VENDOR_STALE_HINT}")

    problems: list[str] = []
    for relative, digest in sorted(expected.items()):
        path = _HERE / "_vendor" / relative
        if not path.is_file():
            problems.append(f"missing {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"modified {relative}")
    if problems:
        raise RuntimeError(
            f"Vendored grading parser bundle is out of date: {', '.join(problems)}. {_VENDOR_STALE_HINT}"
        )


@lru_cache(maxsize=1)
def editable_probe_source() -> str:
    """
    Read the standalone probe instead of embedding Python inside a string.
    """
    return (_HERE / "editable_probe.py").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def grader_zip_bytes() -> bytes:
    """
    Package only runtime modules, retaining ordinary relative package imports.
    """
    from .parsers import parser_registry

    parser_registry()
    _assert_vendor_bundle_intact()
    files = {"__main__.py": f"from {_PACKAGE}.driver import main\nraise SystemExit(main())\n"}
    files[f"{_PACKAGE}/__init__.py"] = ""
    for name in _RUNTIME_MODULES:
        files[f"{_PACKAGE}/{name}"] = (_HERE / name).read_text(encoding="utf-8")
    for path in sorted((_HERE / "_vendor").rglob("*.py")):
        files[f"{_PACKAGE}/{path.relative_to(_HERE).as_posix()}"] = path.read_text(encoding="utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, source in sorted(files.items()):
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, source.encode("utf-8"))
    return buffer.getvalue()


def build_container_command(
    *,
    python_candidates: tuple[str, ...] = DEFAULT_DRIVER_PYTHONS,
) -> str:
    """
    Select the driver interpreter, execute tests, then parse their log.

    The test interpreter comes from the original eval script. Nonzero test
    status alone does not determine the reward. The parsed test report does.
    """
    if not python_candidates or any(not p for p in python_candidates):
        raise ValueError("At least one nonempty driver Python candidate is required.")
    version_probe = shlex.quote(f"import sys; raise SystemExit(sys.version_info < {MIN_DRIVER_PYTHON!r})")
    driver_args = shlex.join(
        (
            DRIVER_ZIP_PATH,
            "--input",
            INPUT_PATH,
            "--log",
            LOG_PATH,
            "--output",
            SCORECARD_PATH,
        )
    )
    no_python = shlex.quote(json.dumps(GradingResult(failure_reason="no_python").to_dict()))
    return (
        f"rm -f {shlex.quote(SCORECARD_PATH)} {shlex.quote(LOG_PATH)}\n"
        f"PY=; for _c in {shlex.join(python_candidates)}; do "
        f'if command -v "$_c" >/dev/null 2>&1 && "$_c" -I -c {version_probe} >/dev/null 2>&1; '
        f'then PY="$_c"; break; fi; done\n'
        'if [ -n "$PY" ]; then\n'
        f"  bash {shlex.quote(EVAL_SCRIPT_PATH)} > {shlex.quote(LOG_PATH)} 2>&1 || :\n"
        f'  "$PY" -I {driver_args}\n'
        f"else printf '%s' {no_python} > {shlex.quote(SCORECARD_PATH)}; fi"
    )
