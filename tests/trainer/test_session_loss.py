"""Regression tests for session aggregation without importing GPU training stacks."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "third_party/verl/verl/trainer/ppo/core_algos.py"
LOSSES = ROOT / "third_party/verl/verl/workers/utils/losses.py"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names), "All requested production functions must exist."
    for node in functions:
        node.decorator_list = []
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_session_mode_replaces_policy_loss():
    tree = ast.parse(CORE.read_text())
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "compute_policy_loss_per_rollout_mean" for node in tree.body
    ), "Session aggregation must not duplicate the PPO algorithm."
    for path in [CORE, LOSSES, ROOT / "psrl/trainer/ppo/ray_trainer.py"]:
        assert "advantages_for_loss" not in path.read_text(), "Advantages must remain unscaled."


@pytest.fixture
def ops():
    torch = pytest.importorskip("torch")
    functional_namespace = load_functions(
        ROOT / "third_party/verl/verl/utils/torch_functional.py",
        {"masked_sum", "masked_mean"},
        {"torch": torch},
    )
    functional = SimpleNamespace(**{name: functional_namespace[name] for name in ("masked_sum", "masked_mean")})
    namespace = dict(
        torch=torch,
        verl_F=functional,
        Optional=Optional,
        Any=Any,
        ActorConfig=SimpleNamespace,
        AlgoConfig=type("AlgoConfig", (), {}),
    )
    load_functions(CORE, {"agg_loss", "compute_policy_loss_vanilla"}, namespace)
    spec = importlib.util.spec_from_file_location("session_loss_under_test", ROOT / "psrl/trainer/ppo/session_loss.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SimpleNamespace(
        torch=torch,
        weights=module.compute_session_loss_weights,
        **{name: namespace[name] for name in ("agg_loss", "compute_policy_loss_vanilla")},
    )


def test_session_reference_and_distributed_gradients(ops):
    t = ops.torch
    ids = ["a", "b", "a", "empty", "b"]
    mask = t.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]])
    values = t.arange(15, dtype=t.float64).reshape(5, 3).requires_grad_()
    weights = ops.weights(mask, ids)
    reference = ((values[[0, 2]] * mask[[0, 2]]).sum() / 5 + (values[[1, 4]] * mask[[1, 4]]).sum() / 3) / 2
    expected_grad = t.autograd.grad(reference, values, retain_graph=True)[0]
    # Unequal micro batches and DP ranks, with one session split across ranks.
    distributed = (
        sum(
            ops.agg_loss(
                values[rows], mask[rows], "session-mean-token-mean", dp_size=2, session_loss_weights=weights[rows]
            )
            for rows in ([0, 1], [3], [4, 2])
        )
        / 2
    )
    t.testing.assert_close(distributed, reference)
    t.testing.assert_close(t.autograd.grad(distributed, values)[0], expected_grad)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_empty_and_long_sessions(ops, dtype_name):
    t = ops.torch
    mask = t.zeros((2, 8), dtype=t.bool)
    values = t.ones((2, 8), dtype=getattr(t, dtype_name), requires_grad=True)
    loss = ops.agg_loss(values, mask, "session-mean-token-mean", session_loss_weights=ops.weights(mask, ["a", "b"]))
    loss.backward()
    assert loss.item() == 0, "Empty windows must have zero loss."
    assert values.grad.count_nonzero().item() == 0, "Empty windows must have zero gradient."
    mask = t.ones((2, 70000), dtype=t.bool)
    values = t.ones(mask.shape, dtype=getattr(t, dtype_name))
    loss = ops.agg_loss(values, mask, "session-mean-token-mean", session_loss_weights=ops.weights(mask, ["a", "a"]))
    t.testing.assert_close(loss, t.tensor(1.0))


@pytest.mark.parametrize("masked_value", [float("nan"), float("inf"), -float("inf")])
def test_masked_nonfinite_values_do_not_affect_loss_or_gradients(ops, masked_value):
    t = ops.torch
    mask = t.tensor([[1, 0], [0, 0]], dtype=t.bool)
    values = t.tensor([[3.0, masked_value], [masked_value, masked_value]], requires_grad=True)
    loss = ops.agg_loss(
        values, mask, "session-mean-token-mean", session_loss_weights=ops.weights(mask, ["a", "empty"])
    )
    t.testing.assert_close(loss, t.tensor(3.0))
    loss.backward()
    t.testing.assert_close(values.grad, t.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_vanilla_matches_old_scaled_advantages_and_gradients(ops):
    t = ops.torch
    mask = t.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    advantages = t.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0], [4.0, 0.0, 0.0]])
    log_prob = t.tensor([[0.5, 1.5, 0.0], [-0.5, 0.1, 2.0], [0.2, 0.0, 0.0]], requires_grad=True)
    old_log_prob = t.zeros_like(log_prob)
    weights = ops.weights(mask, ["a", "a", "b"])

    class Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    config = Config(
        clip_ratio=0.2,
        clip_ratio_low=0.1,
        clip_ratio_high=0.3,
        clip_ratio_c=3.0,
        global_batch_info={"session_loss_weights": weights},
    )
    # The original implementation is vanilla clipping on positively scaled advantages plus token sum.
    config.global_batch_info = {}
    old, _ = ops.compute_policy_loss_vanilla(
        old_log_prob,
        log_prob,
        advantages * weights[:, None],
        mask,
        loss_agg_mode="token-sum",
        config=config,
        rollout_is_weights=t.full_like(log_prob, 1.2),
    )
    old_grad = t.autograd.grad(old, log_prob, retain_graph=True)[0]
    config.global_batch_info = {"session_loss_weights": weights}
    new, _ = ops.compute_policy_loss_vanilla(
        old_log_prob,
        log_prob,
        advantages,
        mask,
        loss_agg_mode="session-mean-token-mean",
        config=config,
        rollout_is_weights=t.full_like(log_prob, 1.2),
    )
    t.testing.assert_close(new, old)
    t.testing.assert_close(t.autograd.grad(new, log_prob)[0], old_grad)


def test_missing_and_misaligned_weights_fail(ops):
    t = ops.torch
    values = t.ones((2, 3))
    with pytest.raises(ValueError, match="requires session_loss_weights"):
        ops.agg_loss(values, values, "session-mean-token-mean")
    with pytest.raises(ValueError, match="one weight per loss row"):
        ops.agg_loss(values, values, "session-mean-token-mean", session_loss_weights=t.ones(2, 1))
    with pytest.raises(ValueError, match="one session ID per row"):
        ops.weights(values, ["a"])


def test_worker_applies_session_weights_to_policy_entropy_and_kl(ops):
    tensordict = pytest.importorskip("tensordict")
    t = ops.torch
    mask = t.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]])
    ids = ["a", "a", "b"]
    weights = ops.weights(mask, ids)
    log_prob = t.tensor([[0.1, -0.2, 0.0], [0.3, -0.1, 0.2], [-0.4, 0.0, 0.0]], requires_grad=True)
    entropy = t.ones_like(log_prob) * 2
    advantages = t.ones_like(log_prob)
    data = tensordict.TensorDict(
        {
            "response_mask": mask,
            "advantages": advantages,
            "old_log_probs": t.zeros_like(log_prob),
            "ref_log_prob": t.zeros_like(log_prob),
            "session_loss_weights": weights,
        },
        batch_size=[3],
    )
    for key, value in {"dp_size": 1, "batch_num_tokens": 6, "global_batch_size": 3}.items():
        data[key] = tensordict.NonTensorData(value)

    class Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    class Metric(SimpleNamespace):
        @staticmethod
        def from_dict(values, aggregation):
            return {key: Metric(value=value, aggregation=aggregation) for key, value in values.items()}

    config = Config(
        policy_loss={"loss_mode": "vanilla"},
        global_batch_info={},
        loss_scale_factor=None,
        loss_agg_mode="session-mean-token-mean",
        clip_ratio=0.2,
        clip_ratio_low=None,
        clip_ratio_high=None,
        entropy_coeff=0.1,
        use_kl_loss=True,
        kl_loss_coef=0.3,
        kl_loss_type="kl",
    )
    namespace = dict(
        torch=t,
        ActorConfig=SimpleNamespace,
        TensorDict=tensordict.TensorDict,
        agg_loss=ops.agg_loss,
        no_padding_2_padding=lambda value, data: value,
        get_policy_loss_fn=lambda mode: ops.compute_policy_loss_vanilla,
        kl_penalty=lambda logprob, ref_logprob, kl_penalty: logprob - ref_logprob,
        Metric=Metric,
        AggregationType=SimpleNamespace(SUM="sum", MEAN="mean"),
    )
    worker_loss = load_functions(LOSSES, {"ppo_loss"}, namespace)["ppo_loss"]
    loss, metrics = worker_loss(config, {"log_probs": log_prob, "entropy": entropy}, data)
    expected_pg, _ = ops.compute_policy_loss_vanilla(
        t.zeros_like(log_prob), log_prob, advantages, mask, loss_agg_mode=config.loss_agg_mode, config=config
    )
    expected_kl = ((log_prob * mask).sum(dim=-1) * weights).sum()
    t.testing.assert_close(loss, expected_pg - 0.2 + 0.3 * expected_kl)
    assert metrics["actor/pg_loss"].aggregation == "sum", "Window-normalized micro losses must be summed."
    t.testing.assert_close(metrics["actor/entropy_loss"].value, t.tensor(2.0))
    t.testing.assert_close(data["advantages"], advantages)
    with pytest.raises(ValueError, match="requires session_loss_weights"):
        worker_loss(config, {"log_probs": log_prob}, data.exclude("session_loss_weights"))
