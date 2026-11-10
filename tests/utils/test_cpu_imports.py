"""
Keep CPU policies, shared fixtures, and logging independent of GPU runtimes.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


def test_cpu_modules_do_not_import_gpu_runtimes():
    script = """
import importlib.abc
import sys

class RejectGpuRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "ray", "verl", "vllm"}:
            raise AssertionError(f"CPU modules must not import {fullname!r}.")

sys.meta_path.insert(0, RejectGpuRuntime())
import tests.conftest
from psrl.utils.logger import FileOnlyHandler
from psrl.workers.elastic.scaling_policy.base import ScalingPolicy
from psrl.workers.gen.utils import TokenOutput
from psrl.workers.ps.staleness_controller import StalenessInventory
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CPU imports failed: {result.stderr!r}."
