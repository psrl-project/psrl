#!/usr/bin/env python3
"""Measure TMS/NIXL wake-up scaling with only FSDP actor and CPU PS workers."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from ray.util import placement_group, remove_placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import RayResourcePool

from psrl.trainer.constants_ppo import get_ppo_ray_runtime_env
from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import ps_agent_name, train_client_name

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "psrl" / "trainer" / "config"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, required=True, choices=(8, 16, 32))
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operation-timeout-s", type=int, default=1800)
    parser.add_argument("--ray-address", default=os.environ.get("RAY_ADDRESS", "auto"))
    parser.add_argument(
        "--local-node-only",
        action="store_true",
        help="Hard-pin every actor bundle to the Ray node running this probe.",
    )
    parser.add_argument(
        "--verify-exact-weights",
        action="store_true",
        help="SHA-256 every transferred logical tensor after the initial pull and each wake cycle.",
    )
    parser.add_argument("--fingerprint-chunk-bytes", type=int, default=64 * 1024**2)
    parser.add_argument(
        "--weight-arena",
        action="store_true",
        help="Pack actor FSDP2 weight storages into arenas before NIXL registration.",
    )
    parser.add_argument(
        "--weight-arena-max-chunk-gb",
        type=float,
        default=4,
        help="Maximum arena allocation size in GB; oversized individual storages remain standalone.",
    )
    parser.add_argument(
        "--reuse-existing-registration",
        action="store_true",
        help=(
            "Experimental: keep the initial NIXL registrations across TMS pause/resume and "
            "perform a real PS pull without re-registering. This probes descriptor validity "
            "after physical memory remapping; it does not change the production wake path."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides applied after the probe defaults.",
    )
    args = parser.parse_args()
    if args.world_size % args.gpus_per_node != 0:
        parser.error("--world-size must be divisible by --gpus-per-node")
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if args.weight_arena_max_chunk_gb <= 0:
        parser.error("--weight-arena-max-chunk-gb must be positive")
    if args.fingerprint_chunk_bytes <= 0:
        parser.error("--fingerprint-chunk-bytes must be positive")
    if args.local_node_only and args.world_size > args.gpus_per_node:
        parser.error("--local-node-only requires --world-size <= --gpus-per-node")
    if not (args.model_path / "config.json").is_file():
        parser.error(f"model config not found: {args.model_path / 'config.json'}")
    return args


def _compose_config(args: argparse.Namespace, server_ip: str, server_port: int):
    logging_path = args.output.parent / f"fsdp{args.world_size}_worker_logs"
    logging_path.mkdir(parents=True, exist_ok=True)
    defaults = [
        f"train_actor_rollout_ref.model.path={args.model_path}",
        "train_actor_rollout_ref.model.use_shm=False",
        "train_actor_rollout_ref.model.enable_gradient_checkpointing=False",
        "train_actor_rollout_ref.actor.strategy=fsdp2",
        "train_actor_rollout_ref.actor.use_dynamic_bsz=True",
        f"train_actor_rollout_ref.actor.fsdp_config.fsdp_size={args.world_size}",
        "train_actor_rollout_ref.actor.fsdp_config.param_offload=False",
        # Full TMS sleep requires optimizer state to be offloaded before pausing.
        "train_actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "train_actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "train_actor_rollout_ref.actor.optim.total_training_steps=1",
        "trainer.device=cuda",
        "trainer.logger=[console]",
        "psrl.ps_mode=nixl_cpu",
        "++psrl.val_rollout_n=1",
        "psrl.tms.range=train",
        "psrl.tms.enable_nixl=True",
        "psrl.memory_logger.enable=False",
        f"psrl.ps_manager_ip={server_ip}",
        f"psrl.nixl.server_ip={server_ip}",
        f"psrl.nixl.server_port={server_port}",
        f"psrl.logging_path={logging_path}",
        f"psrl.nixl.weight_arena.actor_enabled={args.weight_arena}",
    ]
    if args.weight_arena:
        defaults.append(f"psrl.nixl.weight_arena.max_chunk_gb={args.weight_arena_max_chunk_gb}")
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="ppo_trainer", overrides=[*defaults, *args.overrides])
    OmegaConf.resolve(config)
    return config


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def _summarize_worker_wake(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in ("resume_s", "reregister_s", "total_s"):
        summary[key] = _summary([float(result[key]) for result in results])

    client_keys = (
        "register_memory",
        "build_shard_index",
        "group_contig_descs",
        "contig_get_xfer_descs",
        "contig_serialize_descs",
        "group_temp_descs",
        "temp_get_xfer_descs",
        "temp_serialize_descs",
        "total",
    )
    client_results = [result["nixl_client"] for result in results if result.get("nixl_client")]
    summary["nixl_client"] = {
        key: _summary([float(result[key]) for result in client_results])
        for key in client_keys
        if client_results and key in client_results[0]
    }
    summary["counts_per_rank"] = {
        key: sorted({int(result[key]) for result in client_results})
        for key in ("registered_regions", "registered_bytes", "logical_slices", "contig_descs", "temp_descs")
        if client_results and key in client_results[0]
    }
    arena_counts = [
        len(result["arena_virtual_addresses"])
        for result in results
        if result.get("arena_virtual_addresses") is not None
    ]
    if arena_counts:
        summary["arena_counts_per_rank"] = sorted(set(arena_counts))
    node_results: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        node_results.setdefault(str(result["node_id"]), []).append(result)
    summary["nodes"] = {
        node_id: {
            "ranks": sorted(int(result["rank"]) for result in entries),
            "max_reregister_s": max(float(result["reregister_s"]) for result in entries),
        }
        for node_id, entries in node_results.items()
    }
    return summary


def _wait(refs: list[ray.ObjectRef], timeout_s: int):
    return ray.get(refs, timeout=timeout_s)


def _assert_fingerprints_match(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    expected_by_rank = {int(item["rank"]): item for item in expected}
    actual_by_rank = {int(item["rank"]): item for item in actual}
    if expected_by_rank.keys() != actual_by_rank.keys():
        raise RuntimeError(
            f"{label}: fingerprint ranks changed from {sorted(expected_by_rank)} to {sorted(actual_by_rank)}."
        )
    for rank, expected_item in expected_by_rank.items():
        actual_item = actual_by_rank[rank]
        if expected_item["digest"] == actual_item["digest"]:
            continue
        expected_tensors = expected_item["tensor_digests"]
        actual_tensors = actual_item["tensor_digests"]
        differing = sorted(
            key
            for key in expected_tensors.keys() | actual_tensors.keys()
            if expected_tensors.get(key) != actual_tensors.get(key)
        )
        raise RuntimeError(
            f"{label}: exact weight fingerprint mismatch on rank {rank}; "
            f"differing logical tensors={differing[:20]} (total={len(differing)})."
        )


def _compact_fingerprints(fingerprints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in fingerprint.items() if key != "tensor_digests"} for fingerprint in fingerprints
    ]


def _run_wake_cycle(
    actor_wg: RayWorkerGroup,
    ps_wg: Any,
    ps_manager,
    timeout_s: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    worker_results = _wait(actor_wg.execute_all_async("nixl_wake_up"), timeout_s)
    result["wake_rpc_s"] = time.perf_counter() - stage_start
    result["worker_results"] = worker_results
    result["worker_summary"] = _summarize_worker_wake(worker_results)

    updated_client_names = [train_client_name(rank) for rank in range(actor_wg.world_size)]
    stage_start = time.perf_counter()
    gather_refs = actor_wg.execute_all_async("nixl_send_local_info_to", NIXL_META_SERVER_NAME)
    gather_refs.append(ps_manager.nixl_wait_for_update_infos.remote(actor_wg.world_size))
    _wait(gather_refs, timeout_s)
    result["gather_infos_s"] = time.perf_counter() - stage_start

    destination_agents = [ps_agent_name(rank) for rank in range(ps_wg.world_size)]
    destination_agents.extend(updated_client_names)
    wait_refs = ps_wg.execute_all_async("nixl_wait_for_update_infos", 1)
    wait_refs.extend(actor_wg.execute_all_async("nixl_wait_for_update_infos", 1))
    stage_start = time.perf_counter()
    broadcast_ref = ps_manager.nixl_broadcast_update_client_infos.remote(
        destination_agents,
        updated_client_names,
    )
    ray.get(broadcast_ref, timeout=timeout_s)
    result["broadcast_send_s"] = time.perf_counter() - stage_start
    _wait(wait_refs, timeout_s)
    result["broadcast_all_receivers_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _wait(actor_wg.execute_all_async("pull_model"), timeout_s)
    result["pull_model_s"] = time.perf_counter() - stage_start
    result["total_s"] = time.perf_counter() - total_start
    return result


def _run_reused_registration_cycle(actor_wg: RayWorkerGroup, timeout_s: int) -> dict[str, Any]:
    """Resume TMS memory and pull using the descriptors created before sleep.

    This is deliberately probe-only. A successful transfer establishes whether the
    installed UCX/NIXL stack accepts descriptors after TMS remaps physical memory
    at the same virtual address.
    """
    result: dict[str, Any] = {"registration_reused": True}
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    _wait(actor_wg.execute_all_async("wake_up_fsdp_model"), timeout_s)
    result["resume_rpc_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _wait(actor_wg.execute_all_async("pull_model"), timeout_s)
    result["pull_model_s"] = time.perf_counter() - stage_start
    result["total_s"] = time.perf_counter() - total_start
    return result


def _sleep_actor(actor_wg: RayWorkerGroup, reuse_existing_registration: bool, timeout_s: int) -> None:
    method_name = "sleep_fsdp_model" if reuse_existing_registration else "nixl_sleep"
    method_args = () if reuse_existing_registration else ("full",)
    _wait(actor_wg.execute_all_async(method_name, *method_args), timeout_s)


def _print_cycle(world_size: int, cycle: int, result: dict[str, Any]) -> None:
    if result.get("registration_reused"):
        print(
            "[TRAINER_WAKE_PROBE] "
            f"world_size={world_size} cycle={cycle} registration_reused=True "
            f"total_s={result['total_s']:.6f} resume_rpc_s={result['resume_rpc_s']:.6f} "
            f"pull_model_s={result['pull_model_s']:.6f}",
            flush=True,
        )
        return

    worker = result["worker_summary"]
    client = worker["nixl_client"]
    counts = worker["counts_per_rank"]
    print(
        "[TRAINER_WAKE_PROBE] "
        f"world_size={world_size} cycle={cycle} total_s={result['total_s']:.6f} "
        f"wake_rpc_s={result['wake_rpc_s']:.6f} "
        f"resume_max_s={worker['resume_s']['max']:.6f} "
        f"reregister_max_s={worker['reregister_s']['max']:.6f} "
        f"register_memory_max_s={client['register_memory']['max']:.6f} "
        f"get_xfer_descs_max_s="
        f"{client['contig_get_xfer_descs']['max'] + client['temp_get_xfer_descs']['max']:.6f} "
        f"serialize_descs_max_s="
        f"{client['contig_serialize_descs']['max'] + client['temp_serialize_descs']['max']:.6f} "
        f"registered_regions={counts.get('registered_regions')} "
        f"registered_bytes={counts.get('registered_bytes')} "
        f"logical_slices={counts.get('logical_slices')} "
        f"arena_counts={worker.get('arena_counts_per_rank')} "
        f"gather_infos_s={result['gather_infos_s']:.6f} "
        f"broadcast_send_s={result['broadcast_send_s']:.6f} "
        f"broadcast_all_receivers_s={result['broadcast_all_receivers_s']:.6f} "
        f"pull_model_s={result['pull_model_s']:.6f}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "world_size": args.world_size,
        "gpus_per_node": args.gpus_per_node,
        "cycles": args.cycles,
        "reuse_existing_registration": args.reuse_existing_registration,
        "weight_arena": args.weight_arena,
        "weight_arena_max_chunk_gb": args.weight_arena_max_chunk_gb,
        "local_node_only": args.local_node_only,
        "verify_exact_weights": args.verify_exact_weights,
        "model_path": str(args.model_path),
        "setup_timings_s": {},
        "wake_cycles": [],
        "error": None,
    }

    runtime_env = get_ppo_ray_runtime_env()
    runtime_env["env_vars"]["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    ray.init(address=args.ray_address, runtime_env=runtime_env)

    # These imports create the global port-scanner actor, so load them only
    # after connecting to the intended Ray cluster.
    from psrl.utils.nixl import GLOBAL_PORT_SCANNER, NIXLInterface
    from psrl.workers.ps import PSManager, PSStoragePlan, PSStorageWorker
    from psrl.workers.ps.ps_worker_group import (
        PSClassWithInitArgs,
        PSResourcePool,
        PSResourceSpec,
        PSWorkerGroup,
    )
    from psrl.workers.train.base_train_worker import TrainInterface
    from psrl.workers.train.fsdp_train_worker import PSRL_FSDPTrainWorker

    actor_wg = None
    actor_resource_pool = None
    ps_wg = None
    ps_manager = None
    try:
        alive_nodes = [node for node in ray.nodes() if node["Alive"]]
        server_node = next(
            (node for node in alive_nodes if node["NodeManagerAddress"] == ray.util.get_node_ip_address()),
            alive_nodes[0],
        )
        server_ip = server_node["NodeManagerAddress"]
        server_port = ray.get(GLOBAL_PORT_SCANNER.find_free_port.remote(host=server_ip), timeout=60)
        config = _compose_config(args, server_ip, int(server_port))

        ps_manager = (
            ray.remote(PSManager)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=server_node["NodeID"],
                    soft=False,
                )
            )
            .remote(config.psrl)
        )
        ray.get(ps_manager.get_ps_model_version.remote(), timeout=60)
        nixl_interface = NIXLInterface(port_scanner=GLOBAL_PORT_SCANNER)

        actor_resource_pool = RayResourcePool(
            process_on_nodes=[args.gpus_per_node] * (args.world_size // args.gpus_per_node),
            use_gpu=True,
            name_prefix=f"trainer_wake_probe_fsdp{args.world_size}_",
            max_colocate_count=1,
        )
        if args.local_node_only:
            local_node_resource = f"node:{server_ip}"
            if local_node_resource not in server_node["Resources"]:
                raise RuntimeError(f"Local Ray node does not advertise {local_node_resource}.")
            local_pg = placement_group(
                bundles=[{"CPU": 1, "GPU": 1, local_node_resource: 0.001} for _ in range(args.world_size)],
                strategy="STRICT_PACK",
                name=f"trainer_wake_probe_local_fsdp{args.world_size}_{os.getpid()}",
            )
            ray.get(local_pg.ready(), timeout=args.operation_timeout_s)
            actor_resource_pool.pgs = [local_pg]
        actor_cls = RayClassWithInitArgs(
            cls=ray.remote(PSRL_FSDPTrainWorker),
            config=config.train_actor_rollout_ref,
            role="actor",
            psrl_config=config.psrl,
            train_interface=TrainInterface(ps_manager_handle=ps_manager),
            nixl_interface=nixl_interface,
            distillation_config=config.get("distillation", None),
        )

        import torch_memory_saver

        preload_path = Path(torch_memory_saver.__file__).resolve().parents[1] / (
            "torch_memory_saver_hook_mode_preload.abi3.so"
        )
        if not preload_path.is_file():
            raise FileNotFoundError(f"TMS preload library not found: {preload_path}")
        worker_env = {
            "LD_PRELOAD": str(preload_path),
            "TMS_INIT_ENABLE": "1",
            "TMS_INIT_ENABLE_CPU_BACKUP": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            "PSRL_TMS_ENABLE": "1",
            "PSRL_EMPTY_INIT_NO_RANDOM_WEIGHTS": "1",
        }
        setup_start = time.perf_counter()
        actor_wg = RayWorkerGroup(
            resource_pool=actor_resource_pool,
            ray_cls_with_init=actor_cls,
            device_name="cuda",
            worker_env=worker_env,
        )
        report["setup_timings_s"]["create_actor_group"] = time.perf_counter() - setup_start

        actor_node_ids = actor_wg.execute_all_sync("get_node_id")
        if args.local_node_only and set(actor_node_ids) != {server_node["NodeID"]}:
            raise RuntimeError(
                f"Actor placement escaped the local node: expected {server_node['NodeID']}, got {actor_node_ids}."
            )
        unique_actor_node_ids = list(dict.fromkeys(actor_node_ids))
        node_by_id = {node["NodeID"]: node for node in alive_nodes}
        report["actor_nodes"] = [
            {
                "node_id": node_id,
                "node_ip": node_by_id[node_id]["NodeManagerAddress"],
                "ranks": [rank for rank, rank_node_id in enumerate(actor_node_ids) if rank_node_id == node_id],
            }
            for node_id in unique_actor_node_ids
        ]

        ps_resource_pool = PSResourcePool(
            ps_spec_list=[PSResourceSpec(node_id=node_id, attached_gpu_id=None) for node_id in unique_actor_node_ids]
        )
        ps_wg = PSWorkerGroup(
            resource_pool=ps_resource_pool,
            ps_cls_with_init=PSClassWithInitArgs(
                cls=ray.remote(PSStorageWorker),
                storage_plan=PSStoragePlan(
                    train_model_dtype=torch.float32,
                    gen_model_dtype=torch.bfloat16,
                ),
                model_config=config.train_actor_rollout_ref.model,
                psrl_config=config.psrl,
                nixl_interface=nixl_interface,
            ),
        )

        setup_start = time.perf_counter()
        ps_init_refs = ps_wg.execute_all_async("init_nixl_client")
        ps_model_refs = ps_wg.execute_all_async("init_model")
        ps_preload_refs = ps_wg.execute_all_async("preload_checkpoint_to_cpu")
        actor_wg.init_model("empty")
        _wait(actor_wg.execute_all_async("init_nixl_client"), args.operation_timeout_s)
        _wait(actor_wg.execute_all_async("nixl_convert_params"), args.operation_timeout_s)
        _wait([*ps_init_refs, *ps_model_refs], args.operation_timeout_s)
        report["setup_timings_s"]["initialize_actor_and_ps"] = time.perf_counter() - setup_start

        ray.get(
            ps_manager.init_nixl_server.remote(ps_wg.world_size + actor_wg.world_size),
            timeout=args.operation_timeout_s,
        )
        setup_start = time.perf_counter()
        protocol_refs = [ps_manager.nixl_protocol.remote()]
        protocol_refs.extend(ps_wg.execute_all_async("nixl_protocol"))
        protocol_refs.extend(actor_wg.execute_all_async("nixl_protocol", "full"))
        _wait(protocol_refs, args.operation_timeout_s)
        report["setup_timings_s"]["nixl_protocol"] = time.perf_counter() - setup_start

        setup_start = time.perf_counter()
        _wait(ps_preload_refs, args.operation_timeout_s)
        _wait(ps_wg.execute_all_async("write_checkpoint_to_registered_tensors"), args.operation_timeout_s)
        report["setup_timings_s"]["checkpoint_preload_and_write_wait"] = time.perf_counter() - setup_start
        ray.get(ps_manager.bind_ps_worker_group.remote(ps_wg), timeout=args.operation_timeout_s)

        setup_start = time.perf_counter()
        _wait(actor_wg.execute_all_async("pull_model"), args.operation_timeout_s)
        report["setup_timings_s"]["initial_pull"] = time.perf_counter() - setup_start
        baseline_fingerprints = None
        if args.verify_exact_weights:
            precision_start = time.perf_counter()
            baseline_fingerprints = _wait(
                actor_wg.execute_all_async("get_nixl_weight_fingerprint", args.fingerprint_chunk_bytes),
                args.operation_timeout_s,
            )
            report["initial_weight_fingerprints"] = baseline_fingerprints
            report["setup_timings_s"]["initial_weight_fingerprint"] = time.perf_counter() - precision_start
        setup_start = time.perf_counter()
        _sleep_actor(actor_wg, args.reuse_existing_registration, args.operation_timeout_s)
        report["setup_timings_s"]["initial_sleep"] = time.perf_counter() - setup_start

        for cycle in range(1, args.cycles + 1):
            if args.reuse_existing_registration:
                cycle_result = _run_reused_registration_cycle(actor_wg, args.operation_timeout_s)
            else:
                cycle_result = _run_wake_cycle(actor_wg, ps_wg, ps_manager, args.operation_timeout_s)
            cycle_result["cycle"] = cycle
            if baseline_fingerprints is not None:
                precision_start = time.perf_counter()
                cycle_fingerprints = _wait(
                    actor_wg.execute_all_async("get_nixl_weight_fingerprint", args.fingerprint_chunk_bytes),
                    args.operation_timeout_s,
                )
                _assert_fingerprints_match(
                    baseline_fingerprints,
                    cycle_fingerprints,
                    label=f"cycle {cycle}",
                )
                cycle_result["exact_weight_match"] = True
                cycle_result["weight_fingerprints"] = _compact_fingerprints(cycle_fingerprints)
                cycle_result["weight_fingerprint_s"] = time.perf_counter() - precision_start
            report["wake_cycles"].append(cycle_result)
            _print_cycle(args.world_size, cycle, cycle_result)
            sleep_start = time.perf_counter()
            _sleep_actor(actor_wg, args.reuse_existing_registration, args.operation_timeout_s)
            cycle_result["sleep_after_s"] = time.perf_counter() - sleep_start
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if actor_wg is not None:
            for worker in actor_wg._workers:
                ray.kill(worker, no_restart=True)
        if ps_wg is not None:
            try:
                ray.get(ps_wg.execute_all_async("shutdown"), timeout=60)
            except Exception:
                pass
            for worker in ps_wg._workers:
                ray.kill(worker, no_restart=True)
        if ps_manager is not None:
            ray.kill(ps_manager, no_restart=True)
        if actor_resource_pool is not None and actor_resource_pool.pgs is not None:
            for actor_pg in actor_resource_pool.pgs:
                remove_placement_group(actor_pg)
        ray.shutdown()
    print(f"Trainer wake timing report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
