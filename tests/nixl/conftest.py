# tests/nixl/conftest.py
"""
Conftest for nixl tests.

test_nixl_sharding.py and test_comm_planner.py are CPU-safe (pytestmark = cpu_test).

test_nixl_e2e.py, test_nixl_meta_server_comm.py, test_send_recv.py,
test_send_recv_model.py require a full nixl + GPU + multi-node environment
and are designed to be run as scripts (not via pytest in CI).
"""
