# tests/environments/test_environment_registry.py
import pytest

pytestmark = pytest.mark.cpu_test


class TestEnvironmentRegistry:
    def test_environment_base_importable(self):
        from psrl.environments.base import Environment

        assert Environment is not None

    def test_environment_has_registry(self):
        from psrl.environments.base import Environment

        assert hasattr(Environment, "_registry")
        assert isinstance(Environment._registry, dict)

    def test_environment_has_register_decorator(self):
        from psrl.environments.base import Environment

        assert hasattr(Environment, "register")

    def test_registered_environment_retrievable(self):
        """A class decorated with @Environment.register('name') is retrievable by name."""
        from psrl.environments.base import Environment

        @Environment.register("test_env_for_unit_test")
        class _TestEnv(Environment):
            async def reset(self, task, **kwargs):
                return None, {}

            async def step(self, action, **kwargs):
                return None, 0.0, True, {}

            async def close(self):
                pass

            @property
            def state(self):
                return {}

        assert "test_env_for_unit_test" in Environment._registry
