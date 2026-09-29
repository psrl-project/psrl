# tests/state_dict/conftest.py
"""Conftest for state_dict tests.

test_vllm_converter.py, test_fsdp1_converter.py and test_fsdp2_converter.py are
example/script drivers (see scripts/) that require GPU + torchrun + a local
checkpoint, so they are excluded from pytest.
"""

collect_ignore = ["test_vllm_converter.py", "test_fsdp1_converter.py", "test_fsdp2_converter.py"]
