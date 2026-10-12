"""
Evaluate SWE-bench problems with a local model.

Run agent rollouts, grade them in fresh containers, and write per-problem artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))
if not psrl_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    psrl_logger.addHandler(_h)
    psrl_logger.propagate = False

# Anchor relative configuration paths before libraries can change the working directory.
_MODULE_LOAD_CWD = os.getcwd()

# --- Rollout helpers ---


def _run_agent_on_swe_problem(
    swe_problem: dict[str, Any],
    model_path: str,
    config_path: str,
    max_turns: int,
    temperature: float,
    model_class: str = "litellm_textbased",
) -> dict[str, Any]:
    """
    Run mini-SWE-agent on a single SWE problem using a locally-served vLLM model.

    This invokes mini-swe-agent directly (not through the PSRL training loop)
    so that the standalone eval is independent of training state.

    Args:
        swe_problem (dict[str, Any]): Full dataset row for one SWE problem.
        model_path (str): Path to the model checkpoint or HF model ID.
        config_path (str): Path to the agent config YAML.
        max_turns (int): Maximum agent turns per episode.
        temperature (float): Sampling temperature.
        model_class (str): mini-swe-agent model class. ``'litellm_textbased'``
            (default) matches the ``mswea_bash_command`` format used during
            PSRL training and requires a plain text-completion endpoint
            (no ``--tool-call-parser`` on vLLM). Use ``'litellm'`` only for
            external models (GPT-4, Claude, etc.) that natively support
            OpenAI tool-calling.

    Returns:
        dict[str, Any]: Result containing ``patch``, ``messages``, ``n_turns``,
            ``exit_status``, and ``error``.
    """
    from examples.mini_swe.prepare.swebench_subsets import get_swebench_image_name
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.utils.serialize import recursive_merge
    from psrl.utils.agent.overflow import PromptOverflowError, ensure_overflow_handling

    image_name = get_swebench_image_name(swe_problem)
    problem = swe_problem.get("problem_statement", "")

    # Resolve relative config_path using the module-load cwd snapshot (worker
    # threads may inherit a different cwd after library import chains chdir).
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.normpath(os.path.join(_MODULE_LOAD_CWD, config_path))

    # Adapt PSRL training YAML schema (Hydra-style list, sandbox_config nesting,
    # problem_template) to mini-swe-agent's flat dict expectations.
    yaml_cfg = get_config_from_spec(config_path)
    if isinstance(yaml_cfg, list) and len(yaml_cfg) == 1 and isinstance(yaml_cfg[0], dict):
        yaml_cfg = yaml_cfg[0]
    if not isinstance(yaml_cfg, dict):
        raise TypeError(
            f"Unexpected config shape from {config_path!r}: "
            f"got {type(yaml_cfg).__name__}, expected dict (or 1-element list of dict)."
        )
    yaml_cfg = dict(yaml_cfg)
    for k in ("name", "_target_"):
        yaml_cfg.pop(k, None)
    sb = yaml_cfg.pop("sandbox_config", None)
    if isinstance(sb, dict) and "environment" in sb and "environment" not in yaml_cfg:
        yaml_cfg["environment"] = recursive_merge(
            sb["environment"],
            sb.get("rollout_environment", {}),
        )
    # Standalone evaluation calls mini-swe-agent directly, so it needs the upstream
    # `get_environment` dispatcher with an explicit class. Training uses SandboxManager.
    env_block = yaml_cfg.setdefault("environment", {})
    if isinstance(env_block, dict):
        env_block.setdefault("environment_class", "docker")
    if "agent" in yaml_cfg and isinstance(yaml_cfg["agent"], dict):
        agent_cfg = dict(yaml_cfg["agent"])
        if "problem_template" in agent_cfg and "instance_template" not in agent_cfg:
            agent_cfg["instance_template"] = agent_cfg.pop("problem_template")
        yaml_cfg["agent"] = agent_cfg

    # Build config from the (adapted) swebench YAML merged with per-run overrides.
    cfg = recursive_merge(
        yaml_cfg,
        {
            "environment": {"image": image_name, "cwd": "/testbed"},
            "agent": {"step_limit": max_turns},
            "model": {
                "model_name": f"openai/{model_path}",
                "model_class": model_class,
                "model_kwargs": {"temperature": temperature},
                # Local checkpoints may be absent from LiteLLM's pricing table.
                # Ignore pricing lookup errors so they cannot abort rollout.
                "cost_tracking": "ignore_errors",
            },
        },
    )

    from minisweagent.environments import get_environment

    env = get_environment(cfg.get("environment", {}))
    # Pass the model dictionary through `config` so it is not treated as a model name.
    model = get_model(config=cfg.get("model", {}))
    ensure_overflow_handling(model)
    agent = DefaultAgent(model, env, **cfg.get("agent", {}))

    try:
        info = agent.run(problem)
        return {
            "patch": info.get("submission", ""),
            "messages": getattr(agent, "messages", []),
            "n_turns": len([m for m in getattr(agent, "messages", []) if m.get("role") == "assistant"]),
            "exit_status": info.get("exit_status", ""),
            "error": None,
        }
    except PromptOverflowError as exc:
        return {
            "patch": "",
            "messages": getattr(agent, "messages", []),
            "n_turns": len([m for m in getattr(agent, "messages", []) if m.get("role") == "assistant"]),
            "exit_status": "context_exceeded",
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "patch": "",
            "messages": [],
            "n_turns": 0,
            "exit_status": "error",
            "error": str(exc),
        }
    finally:
        try:
            env.cleanup()
        except Exception:
            pass


# --- Per-problem evaluation ---


def _evaluate_swe_problem(
    swe_problem: dict[str, Any],
    *,
    model_path: str,
    config_path: str,
    output_dir: Path,
    max_turns: int,
    temperature: float,
    gold_patches: bool,
    grader_timeout: int,
    model_class: str = "litellm_textbased",
    grader_memory: str = "",
) -> dict[str, Any]:
    """
    Run rollout + grading for a single SWE problem and write per-problem artefacts.

    Args:
        swe_problem (dict[str, Any]): Full dataset row for one SWE problem.
        model_path (str): Model checkpoint path or ID.
        config_path (str): Agent config YAML path.
        output_dir (Path): Root output directory.
        max_turns (int): Max agent turns per episode.
        temperature (float): Sampling temperature.
        gold_patches (bool): Use the gold patch from the dataset instead of
            running the agent (useful for sanity checks).
        grader_timeout (int): Grading eval script timeout in seconds.
        model_class (str): mini-swe-agent model class passed to
            ``_run_agent_on_swe_problem``.
        grader_memory (str): ``--memory`` limit for the fresh grading
            container (e.g. ``"30g"``). Empty string uses the
            ``swebench_grader`` module default.

    Returns:
        dict[str, Any]: Per-SWE-problem result row.
    """
    from examples.mini_swe.grading.schema import GradingPlan
    from examples.mini_swe.prepare.swebench_subsets import get_swebench_image_name
    from examples.mini_swe.swebench_grader import grade_fresh_container

    swe_problem_id: str = swe_problem["instance_id"]
    problem_dir = output_dir / swe_problem_id
    problem_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()

    # --- Run rollout (or use gold patch) ---
    if gold_patches:
        patch = swe_problem.get("patch", "") or ""
        n_turns = 0
        exit_status = "gold"
        rollout_error = None
    else:
        rollout = _run_agent_on_swe_problem(
            swe_problem,
            model_path=model_path,
            config_path=config_path,
            max_turns=max_turns,
            temperature=temperature,
            model_class=model_class,
        )
        patch = rollout["patch"]
        n_turns = rollout["n_turns"]
        exit_status = rollout["exit_status"]
        rollout_error = rollout["error"]

        # Write trajectory.
        traj_path = problem_dir / "traj.json"
        with open(traj_path, "w") as f:
            json.dump(
                {
                    "instance_id": swe_problem_id,
                    "messages": rollout["messages"],
                    "exit_status": exit_status,
                    "n_turns": n_turns,
                    "error": rollout_error,
                },
                f,
                indent=2,
                default=str,
            )

    # Write patch.
    if patch:
        (problem_dir / "patch.diff").write_text(patch)

    # --- Grade ---
    image_name = get_swebench_image_name(swe_problem)
    needs_h1 = "swesmith" in image_name.lower()
    grader_kind = "smith" if needs_h1 else "verified"

    # Grading is host-independent: Verified rows already carry ``eval_script``,
    # while SWE-smith rows need it frozen from the profile first.
    eval_problem = dict(swe_problem)
    if needs_h1:
        from examples.mini_swe.grading.freeze import freeze_smith_grading

        eval_problem.update(freeze_smith_grading(eval_problem))

    grade = grade_fresh_container(
        eval_problem,
        patch,
        grader_kind=grader_kind,
        image_name=image_name,
        timeout=grader_timeout,
        swe_task_id=f"{swe_problem_id}__eval",
        memory=grader_memory,
        grading_plan=GradingPlan.from_swe_problem(eval_problem),
    )
    (problem_dir / "grading.json").write_text(json.dumps(grade, indent=2, default=str))

    elapsed = time.monotonic() - t0
    result = {
        "instance_id": swe_problem_id,
        "patch": patch,
        "n_turns": n_turns,
        "exit_status": exit_status,
        "rollout_error": rollout_error,
        "resolved": grade.get("resolved", False),
        "apply_ok": grade.get("apply_ok", False),
        "f2p_pass": grade.get("f2p_pass", 0),
        "f2p_total": grade.get("f2p_total", 0),
        "p2p_pass": grade.get("p2p_pass", 0),
        "p2p_total": grade.get("p2p_total", 0),
        "policy_violated": grade.get("policy_violated", False),
        "resolved_by": grade.get("resolved_by", ""),
        "grading_timeout": grade.get("timeout", False),
        "grading_error": grade.get("error"),
        "elapsed_s": round(elapsed, 2),
    }

    psrl_logger.info(
        f"[eval] {swe_problem_id}: resolved={result['resolved']}, turns={n_turns}, elapsed={elapsed:.0f}s."
    )
    return result


# --- Main evaluation loop ---


def _looks_like_path(dataset: str) -> bool:
    """
    Heuristic: treat ``dataset`` as a filesystem path rather than a dataset
    key when it looks like one.

    A value is treated as a path when it either exists on disk or when it
    contains a separator / has a supported extension (``.parquet``, ``.json``,
    ``.jsonl``).

    Args:
        dataset (str): Value of the ``--dataset`` argument.

    Returns:
        bool: True when ``dataset`` should be loaded from disk.
    """
    if not dataset:
        return False
    if os.path.exists(dataset):
        return True
    lower = dataset.lower()
    if any(lower.endswith(ext) for ext in (".parquet", ".json", ".jsonl")):
        return True
    return False


def _to_plain_python(obj: Any) -> Any:
    """
    Recursively convert numpy / pandas containers into plain Python types.

    Parquet-backed rows (read via ``pd.read_parquet`` + ``to_dict``) contain
    ``numpy.ndarray`` fields for list-typed columns such as ``FAIL_TO_PASS``.
    Those don't survive ``json.dumps`` unless coerced, which the downstream
    grader relies on.

    Args:
        obj (Any): Value to sanitize.

    Returns:
        Any: ``obj`` with numpy arrays converted to lists, numpy scalars
            converted to Python scalars, and nested dicts / lists recursed.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return [_to_plain_python(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_plain_python(x) for x in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _load_swe_problems_from_path(path: str) -> list[dict[str, Any]]:
    """
    Load HF-shaped SWE problems from parquet, JSONL, or JSON.

    Prepared rows are unwrapped from `extra_info`. Raw rows are retained.

    Args:
        path (str): Filesystem path to the prepared dataset.

    Returns:
        list[dict[str, Any]]: List of HF-shaped SWE problem dicts, ready for
            the downstream grader / rollout loop.
    """
    path_l = path.lower()
    if path_l.endswith(".parquet"):
        df = pd.read_parquet(path)
        rows = df.to_dict(orient="records")
    elif path_l.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif path_l.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else [data]
    else:
        raise ValueError(f"Unsupported dataset file extension: {path!r}. Expected .parquet, .json, or .jsonl.")

    # Prepared rows may nest the HF row under either supported key.
    _NESTED_HF_ROW_KEYS = ("swe_problem", "instance")
    swe_problems: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Row in {path!r} is not a dict (type={type(row).__name__}).")
        extra = row.get("extra_info")
        nested: dict[str, Any] | None = None
        if isinstance(extra, dict):
            for k in _NESTED_HF_ROW_KEYS:
                cand = extra.get(k)
                if isinstance(cand, dict) and cand.get("instance_id"):
                    nested = cand
                    break
        if nested is not None:
            swe_problems.append(_to_plain_python(nested))
        elif row.get("instance_id"):
            swe_problems.append(_to_plain_python(row))
        else:
            raise ValueError(
                f"Row in {path!r} has no instance_id and no extra_info."
                f"{{swe_problem,instance}}; keys={list(row.keys())}."
            )
    return swe_problems


def run_eval(
    *,
    dataset: str,
    split: str,
    subset_spec: str,
    model_path: str,
    config_path: str,
    output_dir: Path,
    workers: int,
    max_turns: int,
    temperature: float,
    gold_patches: bool,
    grader_timeout: int,
    model_class: str = "litellm_textbased",
    grader_memory: str = "",
) -> None:
    """
    Orchestrate the full evaluation run.

    Args:
        dataset (str): Dataset key (``verified``, ``lite``, ``smith``, etc.)
            or a path to a prepared parquet, JSONL, or JSON file. When a path is given, ``split`` is
            ignored.
        split (str): HF split name (unused when ``dataset`` is a path).
        subset_spec (str): Slice or regex filter on the SWE problem's
            ``instance_id`` field.
        model_path (str): Model checkpoint path or HF model ID.
        config_path (str): Agent config YAML path.
        output_dir (Path): Root output directory.
        workers (int): Maximum concurrent evaluation threads.
        max_turns (int): Maximum agent turns per SWE problem.
        temperature (float): Sampling temperature.
        gold_patches (bool): Use gold patches instead of running the agent.
        grader_timeout (int): Grading eval script timeout in seconds.
        model_class (str): mini-swe-agent model class. ``'litellm_textbased'``
            (default) matches the ``mswea_bash_command`` format used during
            PSRL training. Use ``'litellm'`` for external models that
            natively support OpenAI tool-calling.
    """
    from examples.mini_swe.prepare.swebench_subsets import filter_by_spec

    if _looks_like_path(dataset):
        psrl_logger.info(f"Loading prepared SWE problems from path {dataset!r}...")
        swe_problems: list[dict[str, Any]] = _load_swe_problems_from_path(dataset)
    else:
        from datasets import load_dataset
        from examples.mini_swe.prepare.prepare_swebench import _DATASET_HF_MAP

        hf_path = _DATASET_HF_MAP.get(dataset, dataset)
        psrl_logger.info(f"Loading HF dataset={hf_path!r}, split={split!r}...")
        swe_problems = list(load_dataset(hf_path, split=split))

    if subset_spec:
        swe_problems = filter_by_spec(swe_problems, subset_spec)
    psrl_logger.info(f"Evaluation problem count: {len(swe_problems)}. Worker count: {workers}.")
    # One-time diagnostics so config / dataset path issues are visible up-front.
    psrl_logger.info(
        f"Eval context: cwd={os.getcwd()!r}, config_path={config_path!r} "
        f"(abs={os.path.isabs(config_path)}, exists={os.path.exists(config_path)})."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_swe_problem_id = {
            pool.submit(
                _evaluate_swe_problem,
                prob,
                model_path=model_path,
                config_path=config_path,
                output_dir=output_dir,
                max_turns=max_turns,
                temperature=temperature,
                gold_patches=gold_patches,
                grader_timeout=grader_timeout,
                model_class=model_class,
                grader_memory=grader_memory,
            ): prob["instance_id"]
            for prob in swe_problems
        }
        for future in as_completed(future_to_swe_problem_id):
            swe_problem_id = future_to_swe_problem_id[future]
            try:
                results.append(future.result())
            except Exception as exc:
                # Preserve the traceback for per-problem infrastructure failures.
                psrl_logger.exception(f"[eval] Problem raised an exception. Instance: {swe_problem_id!r}.")
                results.append(
                    {
                        "instance_id": swe_problem_id,
                        "resolved": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    # --- Write preds.json (leaderboard-compatible) ---
    preds: dict[str, Any] = {}
    for r in results:
        preds[r["instance_id"]] = {
            "instance_id": r["instance_id"],
            "model_name_or_path": model_path,
            "model_patch": r.get("patch", ""),
        }
    (output_dir / "preds.json").write_text(json.dumps(preds, indent=2))

    # --- Write results.jsonl ---
    with open(output_dir / "results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # --- Write summary.json ---
    total = len(results)
    resolved = sum(1 for r in results if r.get("resolved", False))
    resolve_rate = resolved / total if total > 0 else 0.0
    avg_turns = sum(r.get("n_turns", 0) for r in results) / total if total > 0 else 0.0
    elapsed = time.monotonic() - t_start

    summary = {
        "resolved": resolved,
        "total": total,
        "resolve_rate": round(resolve_rate, 4),
        "avg_turns": round(avg_turns, 2),
        "elapsed_s": round(elapsed, 1),
        "model": model_path,
        "model_class": model_class,
        "dataset": dataset,
        "split": split,
        "subset_spec": subset_spec,
        "gold_patches": gold_patches,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Evaluation complete ===")
    print(f"Resolved : {resolved}/{total} ({resolve_rate:.1%})")
    print(f"Avg turns: {avg_turns:.1f}")
    print(f"Elapsed  : {elapsed:.0f}s")
    print(f"Output   : {output_dir}")


# --- CLI ---


def main() -> None:
    """
    CLI entry point for standalone SWE-bench evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Standalone SWE-bench / SWE-smith-py evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="", help="Model path or HF ID.")
    parser.add_argument(
        "--config",
        default="examples/mini_swe/config/swebench_agent_config.yaml",
        help="Agent config YAML path.",
    )
    parser.add_argument(
        "--dataset",
        default="verified",
        help=(
            "Dataset to evaluate on.  Either an HF dataset key "
            "('verified' / 'lite' / 'full' / 'smith') or a filesystem path to a "
            "prepared parquet / jsonl / json file produced by prepare_swebench.py. "
            "Using a path is recommended when only a curated set of Docker images "
            "has been pre-fetched locally."
        ),
    )
    parser.add_argument(
        "--split",
        default="",
        help="HF split (default: test for Verified, train for smith). Ignored when --dataset is a path.",
    )
    parser.add_argument("--subset-spec", default="", help="Slice or regex filter.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Concurrent eval threads (= concurrent Docker containers). Each worker "
            "spawns one container with --memory=10g and one pytest run, so the "
            "practical ceiling on a single host is roughly "
            "min(CPU_cores / 8, RAM_GiB / 12, 16).  Use eval_swebench_multinode.py "
            "to scale across hosts."
        ),
    )
    parser.add_argument("--max-turns", type=int, default=30, help="Max agent turns per SWE problem.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--gold-patches",
        action="store_true",
        help="Use gold patches instead of running the agent (sanity check).",
    )
    parser.add_argument(
        "--grader-timeout",
        type=int,
        default=1800,
        help=(
            "Per-problem grading eval script timeout in seconds.  Heavy repos "
            "(scikit-learn, requests) need more than the old 900s default "
            "because `pip install -e .` + full P2P suite exceeds 15min."
        ),
    )
    parser.add_argument(
        "--model-class",
        default="litellm_textbased",
        help=(
            "mini-swe-agent model class used for the rollout.  "
            "'litellm_textbased' (default) matches the mswea_bash_command "
            "code-block format used during PSRL training and requires a plain "
            "text-completion endpoint (no --tool-call-parser on vLLM).  "
            "Use 'litellm' only for external models (GPT-4, Claude, etc.) "
            "that natively support OpenAI tool-calling."
        ),
    )
    parser.add_argument(
        "--grader-memory",
        default="",
        help=(
            "Docker --memory limit for the fresh grading container "
            "(e.g. '30g'). Heavy repos (scikit-learn, xarray, matplotlib) "
            "run `pip install -e .` inside the grader container and can "
            "temporarily need 15 to 25 GB. Empty string uses the "
            "swebench_grader module default (currently 30g)."
        ),
    )
    args = parser.parse_args()

    if not args.gold_patches and not args.model:
        parser.error("--model is required unless --gold-patches is set.")

    # Resolve configuration paths before worker threads can observe a changed
    # working directory.
    if args.config and not os.path.isabs(args.config):
        candidate = os.path.normpath(os.path.join(_MODULE_LOAD_CWD, args.config))
        if os.path.isfile(candidate):
            args.config = candidate

    split = args.split
    if not split and not _looks_like_path(args.dataset):
        split = "train" if args.dataset == "smith" else "test"

    run_eval(
        dataset=args.dataset,
        split=split,
        subset_spec=args.subset_spec,
        model_path=args.model,
        config_path=args.config,
        output_dir=Path(args.output_dir),
        workers=args.workers,
        max_turns=args.max_turns,
        temperature=args.temperature,
        gold_patches=args.gold_patches,
        grader_timeout=args.grader_timeout,
        model_class=args.model_class,
        grader_memory=args.grader_memory,
    )


if __name__ == "__main__":
    main()
