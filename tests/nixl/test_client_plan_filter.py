from unittest.mock import Mock

import pytest
from psrl.utils.nixl.client import NIXLStorageClient
from psrl.utils.nixl.comm_plan import NIXLCommPlan
from psrl.utils.nixl.nixl_spec import NIXLClientType

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    ("client_type", "operation", "plan"),
    [
        (
            NIXLClientType.PULL_SIDE,
            "client_read",
            NIXLCommPlan({}, {"client": {"weight": {"assigned_ps": [(0,)]}}}, {}),
        ),
        (
            NIXLClientType.PUSH_SIDE,
            "client_read",
            NIXLCommPlan({}, {}, {"client": {"weight": {"assigned_ps": [(0,)]}}}),
        ),
        (
            NIXLClientType.PUSH_SIDE,
            "client_write",
            NIXLCommPlan({"client": {"weight": {"assigned_ps": [(0,)]}}}, {}, {}),
        ),
    ],
)
def test_unplanned_target_is_skipped_before_client_info_lookup(client_type, operation, plan):
    client = NIXLStorageClient.__new__(NIXLStorageClient)
    client.client_name = "client"
    client.client_type = client_type
    client._comm_plan = plan
    client._ensure_client_info_fetched = Mock(side_effect=AssertionError("unexpected client-info lookup"))

    result = getattr(client, operation)("unused_agent", "unused_ps", "weight", "test")

    assert result == []
    client._ensure_client_info_fetched.assert_not_called()
