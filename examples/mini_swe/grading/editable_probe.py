"""
Report whether the checkout in argv[1] is already installed editable.
"""

try:
    import importlib.metadata as distributions
except ImportError:
    raise SystemExit(1) from None
import json
import os
import sys
from urllib.parse import unquote, urlsplit


def _pep660_target(distribution):
    """
    Return the source directory a PEP 660 editable install was made from.
    """
    try:
        raw = distribution.read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    info = data.get("dir_info")
    if not isinstance(info, dict) or info.get("editable") is not True:
        return None
    parsed = urlsplit(str(data.get("url") or ""))
    if parsed.scheme != "file":
        return None
    return os.path.realpath(unquote(parsed.path))


def main():
    workdir = os.path.realpath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    for distribution in distributions.distributions():
        if _pep660_target(distribution) == workdir:
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
