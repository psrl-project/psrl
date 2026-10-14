"""
Launch SWE-bench evaluation across SSH hosts.

Shard a prepared dataset, run one evaluator per host, and merge artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

# Env vars forwarded to every remote host. NO_PROXY / no_proxy are included
# to prevent corporate HTTP proxies from intercepting LLM requests.
_DEFAULT_FORWARD_ENV: tuple[str, ...] = (
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "NO_PROXY",
    "no_proxy",
    "PSRL_LOGGING_LEVEL",
    "LITELLM_MODEL_REGISTRY_PATH",
    "MSWEA_COST_TRACKING",
    "MSWEA_SILENT_STARTUP",
)


psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))
if not psrl_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    psrl_logger.addHandler(_h)

# --- Host list parsing ---


def _read_hosts(path: str) -> list[str]:
    """
    Read a hosts file (same format as ``load_all_nodes.sh``): one entry per
    line, blank lines and ``#``-prefixed comments ignored.

    Args:
        path (str): Filesystem path to the hosts file.

    Returns:
        list[str]: List of hostnames / IPs in file order (duplicates preserved).
    """
    hosts: list[str] = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            hosts.append(line)
    if not hosts:
        raise ValueError(f"No hosts in {path!r}.")
    return hosts


# --- Parquet sharding ---


def _extract_instance_id(row: dict[str, Any]) -> str:
    """
    Pull the SWE problem's ``instance_id`` out of a parquet row, supporting
    both the PSRL-prepared schemas (``extra_info.swe_problem`` /
    ``extra_info.instance``) and raw HF-shaped rows.

    Args:
        row (dict[str, Any]): One parquet row as a plain dict.

    Returns:
        str: The ``instance_id`` string.
    """
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in ("swe_problem", "instance"):
            nested = extra.get(key)
            if isinstance(nested, dict) and nested.get("instance_id"):
                return str(nested["instance_id"])
    iid = row.get("instance_id")
    if iid:
        return str(iid)
    raise ValueError(f"Row has no instance_id (keys={list(row.keys())!r}); cannot shard.")


def _hash_bucket(instance_id: str, n_shards: int) -> int:
    """
    Deterministic bucket assignment using MD5 over ``instance_id``.

    Args:
        instance_id (str): SWE problem identifier.
        n_shards (int): Number of shards.

    Returns:
        int: Integer in ``[0, n_shards)``.
    """
    h = hashlib.md5(instance_id.encode("utf-8")).hexdigest()
    return int(h, 16) % n_shards


def _shard_parquet(src: Path, out_dir: Path, n_shards: int) -> list[Path]:
    """
    Split ``src`` parquet into ``n_shards`` shard parquet files by
    ``hash(instance_id) % n_shards``.

    Hash-based sharding gives roughly uniform rows-per-shard *and* spreads
    each repository's problems across all hosts, which helps load-balance
    the heavy repos (scikit-learn, matplotlib, ...) against the cheap ones
    (sympy, requests).

    Args:
        src (Path): Source parquet (prepared by ``prepare_swebench.py`` or raw).
        out_dir (Path): Directory to write shard parquet files into.
        n_shards (int): Number of shards to produce (typically = number of hosts).

    Returns:
        list[Path]: Paths to the shard parquets, in shard-index order.
    """
    df = pd.read_parquet(src)
    if len(df) == 0:
        raise ValueError(f"{src!r} contains zero rows.")
    records = df.to_dict(orient="records")
    iids = [_extract_instance_id(r) for r in records]
    buckets = [_hash_bucket(iid, n_shards) for iid in iids]

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    for i in range(n_shards):
        mask = [b == i for b in buckets]
        shard_df = df[mask].reset_index(drop=True)
        p = out_dir / f"shard_{i:03d}.parquet"
        shard_df.to_parquet(p)
        shard_paths.append(p)
    return shard_paths


# --- Remote command assembly ---


def _build_remote_command(
    *,
    repo_root: str,
    env_script: str,
    dataset_path: str,
    output_dir: str,
    workers_per_node: int,
    max_turns: int,
    temperature: float,
    grader_timeout: int,
    gold_patches: bool,
    model: str,
    config: str,
    subset_spec: str,
    model_class: str = "litellm_textbased",
    extra_env: dict[str, str] | None = None,
) -> str:
    """
    Build the single-line bash command executed on every remote host.

    Args:
        repo_root (str): Absolute path to the psrl_agent repo on the shared FS.
        env_script (str): Path to a shared-FS env script that performs
            ``conda activate`` and exports the NCCL / UCX / vLLM /
            LD_LIBRARY_PATH knobs (typical: ``${PSRL_WORKSPACE}/env/env_311.sh``).
            Pass ``""`` to skip sourcing entirely.
        dataset_path (str): Absolute path to the per-host shard parquet.
        output_dir (str): Absolute path to the per-host output directory.
        workers_per_node (int): ``--workers`` passed to the inner eval.
        max_turns (int): ``--max-turns`` passed through.
        temperature (float): ``--temperature`` passed through.
        grader_timeout (int): ``--grader-timeout`` passed through.
        gold_patches (bool): ``--gold-patches`` toggle passed through.
        model (str): ``--model`` value (empty string when unused).
        config (str): ``--config`` value.
        subset_spec (str): Optional ``--subset-spec`` value passed through.
        model_class (str): ``--model-class`` passed through (default:
            ``'litellm_textbased'``).
        extra_env (dict[str, str] | None): Extra env vars to export before the
            inner eval (e.g. ``PSRL_LOGGING_LEVEL``).

    Returns:
        str: The full command, safe to execute as ``bash -lc '<cmd>'``.
    """
    env_prefix = ""
    if extra_env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in extra_env.items())
        env_prefix += " "

    eval_args = [
        f"{env_prefix}python -m examples.mini_swe.eval.eval_swebench",
        f"--dataset {shlex.quote(dataset_path)}",
        f"--output-dir {shlex.quote(output_dir)}",
        f"--workers {workers_per_node}",
        f"--max-turns {max_turns}",
        f"--temperature {temperature}",
        f"--grader-timeout {grader_timeout}",
        f"--config {shlex.quote(config)}",
        f"--model-class {shlex.quote(model_class)}",
    ]
    if gold_patches:
        eval_args.append("--gold-patches")
    if model:
        eval_args.append(f"--model {shlex.quote(model)}")
    if subset_spec:
        eval_args.append(f"--subset-spec {shlex.quote(subset_spec)}")

    parts = [f"cd {shlex.quote(repo_root)}"]
    if env_script:
        # set +u: env script may reference unset vars (e.g. no_proxy).
        parts.append(f"{{ set +u; source {shlex.quote(env_script)}; set -u; }}")
    parts.append(" ".join(eval_args))
    return " && ".join(parts)


_SSH_DEFAULT_OPTS: list[str] = [
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "LogLevel=ERROR",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=240",
    "-o",
    "BatchMode=yes",
]


_LIVE_LOG_LOCK = threading.Lock()


def _stream_to_file_and_terminal(
    stream,
    file_handle,
    prefix: str,
    live: bool,
) -> None:
    """
    Pump ``stream`` line-by-line into ``file_handle`` (always) and into
    the launcher's stderr (when ``live`` is True), prefixing every line
    with ``prefix`` so output from concurrent hosts stays distinguishable.

    Lines from different hosts can interleave. The lock keeps individual
    line writes atomic on the launcher side.
    """
    try:
        for raw in iter(stream.readline, ""):
            if not raw:
                break
            file_handle.write(raw)
            file_handle.flush()
            if live:
                with _LIVE_LOG_LOCK:
                    sys.stderr.write(prefix + raw if not raw.startswith(prefix) else raw)
                    sys.stderr.flush()
    except Exception as e:
        with _LIVE_LOG_LOCK:
            sys.stderr.write(f"{prefix}[stream-pump error: {e}]\n")
            sys.stderr.flush()
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _run_on_host(
    host: str,
    remote_cmd: str,
    *,
    stdout_path: Path,
    stderr_path: Path,
    ssh_user: str = "",
    ssh_opts: list[str] | None = None,
    timeout_s: int | None = None,
    live_logs: bool = True,
) -> int:
    """
    Execute ``remote_cmd`` on ``host`` via ssh, streaming stdout/stderr to
    per-host log files and (when ``live_logs`` is True) live to the launcher's
    own stderr with a ``[host]`` prefix so progress / errors are visible
    without manually tailing files.

    Args:
        host (str): Hostname / IP, optionally ``IP:port`` (port is stripped and
            passed via ``-p``).
        remote_cmd (str): Bash command wrapped in ``bash -lc``.
        stdout_path (Path): File to write remote stdout into.
        stderr_path (Path): File to write remote stderr into.
        ssh_user (str): Optional ssh username (``-l``).
        ssh_opts (list[str] | None): Additional ssh options (default:
            :data:`_SSH_DEFAULT_OPTS`).
        timeout_s (int | None): Local subprocess timeout in seconds. ``None``
            waits indefinitely.
        live_logs (bool): When True, also tee remote stdout/stderr to the
            launcher's stderr with ``[host]`` / ``[host !]`` prefixes.

    Returns:
        int: Exit code of the remote command (or ``124`` on local timeout).
    """
    target = host
    port_arg: list[str] = []
    if ":" in host and host.rsplit(":", 1)[-1].isdigit():
        target, port = host.rsplit(":", 1)
        port_arg = ["-p", port]

    ssh_cmd = ["ssh"] + list(ssh_opts or _SSH_DEFAULT_OPTS) + port_arg
    if ssh_user:
        ssh_cmd += ["-l", ssh_user]
    ssh_cmd += [target, "bash", "-lc", remote_cmd]

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ssh_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    out_prefix = f"[{host}] "
    err_prefix = f"[{host} !] "

    so = open(stdout_path, "w")
    se = open(stderr_path, "w")
    try:
        out_t = threading.Thread(
            target=_stream_to_file_and_terminal,
            args=(proc.stdout, so, out_prefix, live_logs),
            daemon=True,
        )
        err_t = threading.Thread(
            target=_stream_to_file_and_terminal,
            args=(proc.stderr, se, err_prefix, live_logs),
            daemon=True,
        )
        out_t.start()
        err_t.start()
        try:
            rc = proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            with _LIVE_LOG_LOCK:
                sys.stderr.write(f"{err_prefix}[multinode] local ssh timeout after {timeout_s}s.\n")
                sys.stderr.flush()
            se.write(f"\n[multinode] local ssh timeout after {timeout_s}s.\n")
            rc = 124
        # Wait briefly for pumps to drain anything still buffered.
        out_t.join(timeout=10)
        err_t.join(timeout=10)
        return rc
    finally:
        so.close()
        se.close()


# --- Output merge ---


def _merge_outputs(
    host_dirs: list[Path],
    final_dir: Path,
    *,
    link_instance_dirs: bool = True,
) -> dict[str, Any]:
    """
    Merge per-host eval_swebench output into ``final_dir``.

    Concatenates ``results.jsonl`` across shards, merges ``preds.json``, and
    either symlinks or copies each per-instance subdirectory into the
    top-level ``final_dir`` so downstream tooling can look up per-problem
    artefacts in one place.

    Args:
        host_dirs (list[Path]): List of per-host output directories.
        final_dir (Path): Top-level output directory (created if missing).
        link_instance_dirs (bool): When True, symlink per-instance subdirectories.
            When False, copy them.

    Returns:
        dict[str, Any]: Merged summary dict (also written to
            ``final_dir/summary.json``).
    """
    final_dir.mkdir(parents=True, exist_ok=True)

    combined_results: list[dict[str, Any]] = []
    combined_preds: dict[str, Any] = {}
    seen_instance_ids: set[str] = set()

    for d in host_dirs:
        rj = d / "results.jsonl"
        if rj.is_file():
            with open(rj) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    iid = row.get("instance_id")
                    if iid and iid in seen_instance_ids:
                        continue
                    if iid:
                        seen_instance_ids.add(iid)
                    combined_results.append(row)

        pj = d / "preds.json"
        if pj.is_file():
            try:
                combined_preds.update(json.loads(pj.read_text()))
            except json.JSONDecodeError:
                psrl_logger.warning(f"Skipping invalid preds.json in directory={d!r}.")

        for sub in d.iterdir():
            if not sub.is_dir():
                continue
            target = final_dir / sub.name
            if target.exists() or target.is_symlink():
                continue
            if link_instance_dirs:
                try:
                    target.symlink_to(sub.resolve())
                    continue
                except OSError:
                    pass
            shutil.copytree(sub, target)

    with open(final_dir / "results.jsonl", "w") as f:
        for r in combined_results:
            f.write(json.dumps(r, default=str) + "\n")
    (final_dir / "preds.json").write_text(json.dumps(combined_preds, indent=2))

    total = len(combined_results)
    resolved = sum(1 for r in combined_results if r.get("resolved"))
    rate = resolved / total if total else 0.0
    avg_turns = sum(r.get("n_turns", 0) for r in combined_results) / total if total else 0.0
    summary: dict[str, Any] = {
        "resolved": resolved,
        "total": total,
        "resolve_rate": round(rate, 4),
        "avg_turns": round(avg_turns, 2),
        "num_shards": len(host_dirs),
    }
    (final_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --- Orchestrator ---


def _collect_forwarded_env(
    names: list[str] | tuple[str, ...],
    extras: list[str] | None = None,
) -> dict[str, str]:
    """
    Snapshot environment variables from the launching process so they can be
    re-exported on every remote host.

    Names that are unset in the launcher's environment are silently dropped.
    ``extras`` is a list of ``NAME=VALUE`` literals that override / add to the
    ``names``-based snapshot.

    Args:
        names (list[str] | tuple[str, ...]): Env-var names to snapshot.
        extras (list[str] | None): Explicit ``NAME=VALUE`` overrides.

    Returns:
        dict[str, str]: Name→value map of env vars to forward.
    """
    out: dict[str, str] = {}
    for name in names:
        val = os.getenv(name)
        if val is not None:
            out[name] = val
    for spec in extras or []:
        if "=" not in spec:
            raise ValueError(f"--forward-env / --set-env value must be NAME=VALUE, got {spec!r}.")
        k, v = spec.split("=", 1)
        out[k] = v
    return out


def run_multinode(
    *,
    hosts_file: str,
    dataset_path: str,
    output_dir: Path,
    workers_per_node: int,
    gold_patches: bool,
    model: str,
    config: str,
    max_turns: int,
    temperature: float,
    grader_timeout: int,
    subset_spec: str,
    repo_root: str,
    env_script: str,
    ssh_user: str,
    ssh_timeout_s: int | None,
    dry_run: bool,
    forward_env: list[str] | None = None,
    set_env: list[str] | None = None,
    model_class: str = "litellm_textbased",
) -> int:
    """
    Orchestrate the full multi-node evaluation run. Returns the number of
    hosts that exited non-zero (0 means every host succeeded).

    See module docstring for argument semantics.
    """
    hosts = _read_hosts(hosts_file)
    n = len(hosts)
    psrl_logger.info(f"[multinode] Host count: {n}. Hosts: {hosts!r}.")

    forward_names = list(forward_env) if forward_env else list(_DEFAULT_FORWARD_ENV)
    # Guarantee at least a logging knob is forwarded so remote stdout is legible.
    if "PSRL_LOGGING_LEVEL" not in forward_names:
        forward_names.append("PSRL_LOGGING_LEVEL")
    forwarded_env = _collect_forwarded_env(forward_names, set_env or [])
    if "PSRL_LOGGING_LEVEL" not in forwarded_env:
        forwarded_env["PSRL_LOGGING_LEVEL"] = "INFO"

    # Auto-prepend the API host to NO_PROXY so corp proxies don't intercept LLM requests.
    api_base = forwarded_env.get("OPENAI_API_BASE") or forwarded_env.get("OPENAI_BASE_URL")
    if api_base:
        try:
            from urllib.parse import urlparse

            api_host = urlparse(api_base).hostname
        except Exception:
            api_host = None
        if api_host:
            for k in ("NO_PROXY", "no_proxy"):
                existing = forwarded_env.get(k, "")
                parts = [p.strip() for p in existing.split(",") if p.strip()]
                for needed in (api_host, "localhost", "127.0.0.1"):
                    if needed not in parts:
                        parts.insert(0, needed)
                forwarded_env[k] = ",".join(parts)

    if forwarded_env:
        forwarded_keys_preview = {
            k: (v if k != "OPENAI_API_KEY" else "***redacted***") for k, v in forwarded_env.items()
        }
        psrl_logger.info(f"[multinode] Forwarding env to every host: {forwarded_keys_preview}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_shards_dir = output_dir / "input_shards"
    host_output_dir = output_dir / "host_output"
    host_logs_dir = output_dir / "host_logs"
    host_logs_dir.mkdir(parents=True, exist_ok=True)
    host_output_dir.mkdir(parents=True, exist_ok=True)

    # Sharding requires a readable dataset file.
    if not os.path.isfile(dataset_path):
        raise ValueError(
            f"--dataset must be a filesystem path for multi-node mode "
            f"(got {dataset_path!r}).  Run prepare_swebench.py first."
        )
    dataset_abs = os.path.abspath(dataset_path)
    repo_root_abs = os.path.abspath(repo_root)

    psrl_logger.info(f"[multinode] Sharding dataset={dataset_abs!r} into parts={n}...")
    shard_paths = _shard_parquet(Path(dataset_abs), input_shards_dir, n)
    shard_sizes = [len(pd.read_parquet(p)) for p in shard_paths]
    psrl_logger.info(
        f"[multinode] Shard sizes: min={min(shard_sizes)} max={max(shard_sizes)} "
        f"total={sum(shard_sizes)} (per-shard: {shard_sizes})"
    )

    host_dirs: list[Path] = []
    host_cmds: list[tuple[str, str, Path, Path, Path]] = []
    for i, (host, shard) in enumerate(zip(hosts, shard_paths)):
        safe = host.replace(":", "_").replace("/", "_")
        host_out = (host_output_dir / safe).resolve()
        host_dirs.append(host_out)
        stdout_path = host_logs_dir / f"{safe}.stdout"
        stderr_path = host_logs_dir / f"{safe}.stderr"
        remote_cmd = _build_remote_command(
            repo_root=repo_root_abs,
            env_script=env_script,
            dataset_path=str(shard.resolve()),
            output_dir=str(host_out),
            workers_per_node=workers_per_node,
            max_turns=max_turns,
            temperature=temperature,
            grader_timeout=grader_timeout,
            gold_patches=gold_patches,
            model=model,
            config=config,
            subset_spec=subset_spec,
            model_class=model_class,
            extra_env=forwarded_env,
        )
        host_cmds.append((host, remote_cmd, stdout_path, stderr_path, host_out))

    if dry_run:
        print("=== DRY RUN: commands that would execute ===")
        for host, cmd, so, se, ho in host_cmds:
            print(f"\n[{host}]  (stdout→{so}, out→{ho})")
            print(f"  ssh {host} bash -lc {shlex.quote(cmd)}")
        return 0

    t0 = time.monotonic()
    psrl_logger.info(f"[multinode] Launching SSH worker count: {n}...")
    failures: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = {
            pool.submit(
                _run_on_host,
                host,
                cmd,
                stdout_path=so,
                stderr_path=se,
                ssh_user=ssh_user,
                timeout_s=ssh_timeout_s,
            ): host
            for host, cmd, so, se, _ in host_cmds
        }
        for fut in as_completed(futs):
            host = futs[fut]
            rc = fut.result()
            tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
            psrl_logger.info(f"[multinode] [{host}] -> {tag}")
            if rc != 0:
                failures[host] = rc

    elapsed = time.monotonic() - t0
    psrl_logger.info(f"[multinode] All hosts finished in {elapsed:.0f}s. Merging...")

    summary = _merge_outputs(host_dirs, output_dir)
    summary["elapsed_s"] = round(elapsed, 1)
    summary["hosts"] = hosts
    summary["host_failures"] = failures
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Multi-node evaluation complete ===")
    print(f"Hosts     : {n}  (failures: {len(failures)})")
    print(f"Resolved  : {summary['resolved']}/{summary['total']} ({summary['resolve_rate']:.1%})")
    print(f"Wall clock: {elapsed:.0f}s")
    print(f"Output    : {output_dir}")
    if failures:
        print("\nHost failures (see host_logs/<host>.stderr for details):")
        for h, rc in failures.items():
            print(f"  {h}: rc={rc}")
    return len(failures)


# --- CLI ---


def main() -> None:
    """
    CLI entry point for multi-node SWE-bench / SWE-smith-py evaluation.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Shard a prepared parquet across hosts and fan eval_swebench out "
            "in parallel via ssh. Merges results into one combined output dir."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hosts",
        required=True,
        help="Hosts file, one IP (or IP:port) per line. Comments and blanks ignored.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a prepared parquet (produced by prepare_swebench.py).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory (must live on the shared FS).",
    )
    parser.add_argument(
        "--workers-per-node",
        type=int,
        default=8,
        help="--workers passed to eval_swebench on each host.",
    )
    parser.add_argument(
        "--grader-timeout",
        type=int,
        default=1800,
        help="--grader-timeout passed to eval_swebench on each host.",
    )
    parser.add_argument(
        "--gold-patches",
        action="store_true",
        help="Use gold patches (sanity check); passed through to eval_swebench.",
    )
    parser.add_argument("--model", default="", help="Model path or HF ID.")
    parser.add_argument(
        "--config",
        default="examples/mini_swe/config/swebench_agent_config.yaml",
        help="Agent config YAML path (relative to --repo-root).",
    )
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--subset-spec",
        default="",
        help=("Optional slice / regex filter applied on each host *after* sharding.  Useful for quick smoke tests."),
    )
    parser.add_argument(
        "--repo-root",
        default="${PSRL_WORKSPACE}/psrl_agent",
        help="Absolute path to the psrl_agent repo on the shared FS.",
    )
    parser.add_argument(
        "--env-script",
        default="${PSRL_WORKSPACE}/env/env_311.sh",
        help=(
            "Path on the shared FS to an env script sourced on every remote "
            "host before invoking eval_swebench.  It must perform `conda "
            "activate` and export the NCCL / UCX / vLLM / cudnn / torch "
            "LD_LIBRARY_PATH knobs (the same file PSRL training sources). "
            "Pass '' to disable sourcing entirely."
        ),
    )
    parser.add_argument(
        "--ssh-user",
        default="",
        help="Optional ssh username (-l); defaults to ssh config / $USER.",
    )
    parser.add_argument(
        "--ssh-timeout",
        type=int,
        default=0,
        help=(
            "Local per-host ssh subprocess timeout in seconds (0 = wait forever). "
            "Typically set to a bit more than the longest host's expected runtime."
        ),
    )
    parser.add_argument(
        "--forward-env",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Name of an environment variable to forward from the launcher to every "
            "remote host. Can be given multiple times. When omitted, the default set "
            "is used: " + ", ".join(_DEFAULT_FORWARD_ENV) + ". "
            "Typical use: point the eval at a vLLM / litellm proxy by exporting "
            "OPENAI_API_BASE (+ OPENAI_API_KEY=dummy) locally before launching."
        ),
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help=(
            "Explicit NAME=VALUE env override, applied on every remote host. "
            "Can be given multiple times and takes precedence over --forward-env "
            "values pulled from the launcher's own environment."
        ),
    )
    parser.add_argument(
        "--model-class",
        default="litellm_textbased",
        help=(
            "mini-swe-agent model class forwarded to every remote eval_swebench call. "
            "'litellm_textbased' (default) matches the mswea_bash_command format used "
            "during PSRL training.  Use 'litellm' for external models that natively "
            "support OpenAI tool-calling."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ssh commands that would be issued and exit.",
    )
    args = parser.parse_args()

    if not args.gold_patches and not args.model:
        parser.error("--model is required unless --gold-patches is set.")

    # Resolve --config to absolute path before forwarding to remote hosts.
    if args.config and os.path.isfile(args.config) and not os.path.isabs(args.config):
        args.config = os.path.abspath(args.config)

    ssh_timeout_s: int | None = args.ssh_timeout if args.ssh_timeout > 0 else None

    n_fail = run_multinode(
        hosts_file=args.hosts,
        dataset_path=args.dataset,
        output_dir=Path(args.output_dir).resolve(),
        workers_per_node=args.workers_per_node,
        gold_patches=args.gold_patches,
        model=args.model,
        config=args.config,
        max_turns=args.max_turns,
        temperature=args.temperature,
        grader_timeout=args.grader_timeout,
        subset_spec=args.subset_spec,
        repo_root=args.repo_root,
        env_script=args.env_script,
        ssh_user=args.ssh_user,
        ssh_timeout_s=ssh_timeout_s,
        dry_run=args.dry_run,
        forward_env=args.forward_env,
        set_env=args.set_env,
        model_class=args.model_class,
    )
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
