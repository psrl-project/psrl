"""Vendored, dependency-free grading payload.

Nothing in this package may import ``swebench`` or ``swesmith`` at runtime; those
packages are used only by ``build_registry.py`` (a build-time tool) and by
``freeze.py`` (prepare-time). See ``PROVENANCE.md``.
"""

from __future__ import annotations
