import pytest
from psrl.utils.rollout.request_id import canonical_psrl_request_id

pytestmark = pytest.mark.cpu_test


def test_canonical_psrl_request_id_strips_only_vllm_internal_suffix():
    assert canonical_psrl_request_id("389-b51b174a") == "389"
    assert canonical_psrl_request_id(389) == "389"
    assert canonical_psrl_request_id("request-with-hyphens") == "request-with-hyphens"
