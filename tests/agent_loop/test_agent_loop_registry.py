import pytest
from psrl.workers.agent_loop.loops.utils import get_agent_loop_config

pytestmark = pytest.mark.cpu_test


def test_default_generate_only_agent_is_importable_on_lookup():
    config = get_agent_loop_config("generate_only_agent")
    assert "GenerateAgentLoop" in config["_target_"]


def test_unknown_agent_loop_raises_value_error():
    with pytest.raises(ValueError, match="Unregistered agent loop"):
        get_agent_loop_config("not_a_registered_agent")


def test_lookup_imports_builtin_module_when_registry_is_empty(monkeypatch):
    from psrl.workers.agent_loop.loops import utils as loop_utils

    fake_registry: dict[str, dict] = {}
    imported: list[str] = []

    def fake_import(name: str):
        imported.append(name)
        fake_registry["generate_only_agent"] = {
            "_target_": "psrl.workers.agent_loop.loops.generate_agent_loop.GenerateAgentLoop",
        }

    monkeypatch.setattr(loop_utils, "AGENT_LOOP_REGISTRY", fake_registry)
    monkeypatch.setattr(loop_utils, "import_module", fake_import)

    config = loop_utils.get_agent_loop_config("generate_only_agent")
    assert imported == ["psrl.workers.agent_loop.loops.generate_agent_loop"]
    assert "GenerateAgentLoop" in config["_target_"]
