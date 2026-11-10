"""
Keep installation validation strict around the NCCL dependency override.
"""

import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def checker(monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts/check_environment.py"
    spec = importlib.util.spec_from_file_location("check_environment", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    versions = {"torch": "2.11.0+cu128", "nvidia-nccl-cu12": "2.29.7"}
    monkeypatch.setattr(module, "version", versions.__getitem__)
    monkeypatch.setattr(module, "distribution", lambda name: SimpleNamespace(locate_file=lambda path: Path(path)))

    def get_version(pointer):
        pointer._obj.value = 22907
        return 0

    library = SimpleNamespace(ncclCommSuspend=object(), ncclCommResume=object(), ncclGetVersion=get_version)
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: library)
    return module, versions


def test_accepts_only_the_configured_metadata_override(checker, monkeypatch):
    module, _ = checker
    output = (
        "torch 2.11.0+cu128 has requirement nvidia-nccl-cu12==2.28.9"
        '; platform_system == "Linux", but you have nvidia-nccl-cu12 2.29.7.\n'
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, output, "")
    )
    module.main()


def test_rejects_additional_dependency_conflicts(checker, monkeypatch):
    module, _ = checker
    output = (
        "torch 2.11.0+cu128 has requirement nvidia-nccl-cu12==2.28.9"
        '; platform_system == "Linux", but you have nvidia-nccl-cu12 2.29.7.\n'
        "example 1.0 has requirement numpy<2, but you have numpy 2.2.6.\n"
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, output, "")
    )
    with pytest.raises(SystemExit):
        module.main()


def test_rejects_nccl_downgrade_even_when_pip_check_passes(checker, monkeypatch):
    module, versions = checker
    versions["nvidia-nccl-cu12"] = "2.28.9"
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    with pytest.raises(RuntimeError, match="Expected PyTorch"):
        module.main()


def test_requires_native_suspend_symbols(checker, monkeypatch):
    module, _ = checker
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "", ""))
    monkeypatch.setattr(module.ctypes, "CDLL", lambda path: SimpleNamespace())
    with pytest.raises(AttributeError, match="ncclCommSuspend"):
        module.main()
