import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_test


_ROOT = Path(__file__).resolve().parents[2]


def _trainer_method_names() -> set[str]:
    tree = ast.parse((_ROOT / "psrl/trainer/ppo/ray_trainer.py").read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PSRL_RayPPOTrainer")
    return {item.name for item in trainer.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _trainer_method(name: str) -> ast.FunctionDef:
    tree = ast.parse((_ROOT / "psrl/trainer/ppo/ray_trainer.py").read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PSRL_RayPPOTrainer")
    return next(item for item in trainer.body if isinstance(item, ast.FunctionDef) and item.name == name)


def _train_worker_method_source(name: str) -> str:
    path = _ROOT / "psrl/workers/train/engine_train_worker.py"
    source = path.read_text()
    tree = ast.parse(source)
    worker = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PSRL_EngineTrainWorker"
    )
    method = next(item for item in worker.body if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, method)


def test_fit_tags_elastic_actions_with_training_step():
    method = _trainer_method("fit")
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "set_current_training_step" for node in ast.walk(method)
    ), "Each training step must tag subsequently accepted elastic actions."


def test_trainer_does_not_own_elastic_pool_transitions():
    assert "_sleep_coordinator_instances" not in _trainer_method_names(), "The executor must own inference sleep."
    assert "release_trainer_pool_for_wait" not in _trainer_method_names(), "The executor must own trainer release."


def test_init_reward_model_servers_sleeps_each_elastic_batch():
    method = _trainer_method("init_reward_model_servers")
    reward_loops = [node for node in method.body if isinstance(node, ast.For)]
    assert reward_loops, "init_reward_model_servers must iterate reward models."
    assert any(
        isinstance(node, ast.Attribute) and node.attr == "sleep_coordinator_instances"
        for node in ast.walk(reward_loops[0])
    ), "Elastic reward initialization must delegate sleep to the executor."


def test_elastic_executor_defines_set_current_training_step():
    tree = ast.parse((_ROOT / "psrl/workers/elastic/executor.py").read_text())
    executor = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ElasticExecutor")
    names = {item.name for item in executor.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert "set_current_training_step" in names


def test_train_worker_deregisters_before_sleep_and_registers_after_wake():
    sleep_source = _train_worker_method_source("nixl_sleep")
    wake_source = _train_worker_method_source("nixl_wake_up")

    assert sleep_source.index("deregister_local_tensors") < sleep_source.index("self.sleep_model()")
    assert wake_source.index("self.wake_up_model()") < wake_source.index("register_local_tensors")
    assert wake_source.index("assert_fsdp_bindings_unchanged") < wake_source.index("register_local_tensors")
