"""
Self-contained SWE grading for PSRL.

The grader derives everything it needs from fields already carried by the
prepared parquet (`swe_problem.eval_script` / `log_parser`) and a vendored,
stdlib-only parser registry, so no `swebench`/`swesmith` import happens on the
training host at rollout time. See `_vendor/PROVENANCE.md`.
"""

from __future__ import annotations

from .parsers import require_vendor_bundle

# `.schema` imports `_vendor._constants` at module scope, so this must run first.
require_vendor_bundle()

from .schema import GradingPlan  # noqa: E402

__all__ = ["GradingPlan"]
