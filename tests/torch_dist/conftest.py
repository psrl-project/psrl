# tests/torch_dist/conftest.py
"""Conftest for the torch.distributed examples.

test_broadcast.py is a torchrun example (see scripts/), not a pytest test, so it
is excluded from collection.
"""

collect_ignore = ["test_broadcast.py"]
