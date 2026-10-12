# tests/fsdp/conftest.py
"""Conftest for the FSDP examples.

test_fsdp1_load_model.py / test_fsdp2_load_model.py are GPU/torchrun example
scripts (see scripts/), not pytest tests, so they are excluded from collection.
"""

collect_ignore = ["test_fsdp1_load_model.py", "test_fsdp2_load_model.py"]
