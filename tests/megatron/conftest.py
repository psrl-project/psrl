# tests/megatron/conftest.py
"""Conftest for megatron tests.

test_megatron_model_init.py is an 8-GPU Megatron example run as a script
(see scripts/), so it is excluded from pytest collection.
"""

collect_ignore = ["test_megatron_model_init.py"]
