import pytest
from psrl.trainer.ppo.utils import PSRL_Role

pytestmark = pytest.mark.cpu_test


class TestPSRLRole:
    def test_all_expected_roles_exist(self):
        expected = {
            "Actor",
            "Rollout",
            "ActorRollout",
            "Critic",
            "RefPolicy",
            "RewardModel",
            "ActorRolloutRef",
            "Validate",
            "DummyPolicy",
        }
        actual = {r.name for r in PSRL_Role}
        assert expected == actual

    def test_role_values_are_unique(self):
        values = [r.value for r in PSRL_Role]
        assert len(values) == len(set(values))

    def test_roles_are_enum_members(self):
        for role in PSRL_Role:
            assert PSRL_Role[role.name] == role

    def test_role_is_usable_as_dict_key(self):
        role_map = {role: role.name for role in PSRL_Role}
        assert role_map[PSRL_Role.Actor] == "Actor"
        assert role_map[PSRL_Role.Validate] == "Validate"
