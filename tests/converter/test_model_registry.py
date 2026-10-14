# tests/converter/test_model_registry.py
"""Tests for the ParameterMapping model registry (create_parameter_mapping, register_model)."""

import pytest
from psrl.utils.converter.model_mappings import ModelRegistry, ParameterMapping

pytestmark = pytest.mark.cpu_test


class TestModelRegistry:
    def test_registry_starts_empty(self):
        """A fresh ModelRegistry has no registered mappings."""
        registry = ModelRegistry()
        assert len(registry._mappings) == 0

    def test_register_string_key(self):
        """Registering by string class name allows lookup by the same string."""
        registry = ModelRegistry()

        class _DummyMapping(ParameterMapping):
            def get_mappings(self):
                return []

            def get_model_info(self):
                return {}

        registry.register("DummyModel", _DummyMapping)
        assert "DummyModel" in registry._mappings

    def test_register_class_key(self):
        """Registering by class object stores the class itself as key (not __name__)."""
        registry = ModelRegistry()

        class _TargetModel:
            pass

        class _DummyMapping(ParameterMapping):
            def get_mappings(self):
                return []

            def get_model_info(self):
                return {}

        registry.register(_TargetModel, _DummyMapping)
        # ModelRegistry stores the class object itself as key, not __name__
        assert _TargetModel in registry._mappings

    def test_register_list_of_keys(self):
        """Registering with a list registers all keys."""
        registry = ModelRegistry()

        class _DummyMapping(ParameterMapping):
            def get_mappings(self):
                return []

            def get_model_info(self):
                return {}

        registry.register(["ModelA", "ModelB"], _DummyMapping)
        assert "ModelA" in registry._mappings
        assert "ModelB" in registry._mappings

    def test_create_mapping_raises_for_unknown_class(self):
        """create_mapping raises for a class not in the registry."""
        registry = ModelRegistry()
        with pytest.raises((KeyError, ValueError, RuntimeError)):
            registry.create_mapping("UnknownModel", "/fake/path")


class TestGlobalModelRegistry:
    def test_megatron_model_registered(self):
        """The Megatron model mapping is registered by the megatron_modeling import."""
        from psrl.utils.converter import model_registry

        # megatron_modeling.py registers "Megatron" via @register_model
        assert "Megatron" in model_registry._mappings

    def test_model_registry_importable(self):
        from psrl.utils.converter import create_parameter_mapping, model_registry, register_model

        assert model_registry is not None
        assert callable(create_parameter_mapping)
        assert callable(register_model)
