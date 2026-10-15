#!/usr/bin/env python3
"""Measure NIXL CUDA registration latency as the region count changes.

The launcher starts a fresh source/initiator process pair for every region
count. The source owns stable tensors and the initiator owns the TMS-managed
CUDA tensors under test. Every initiator registration is followed by a real
NIXL READ and a full-byte correctness check.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

SOURCE_AGENT = "nixl_registration_probe_source"
INITIATOR_AGENT = "nixl_registration_probe_initiator"
TMS_TAG = "nixl_registration_probe"
DEFAULT_REGION_COUNTS = (1, 4, 16, 64, 199, 339, 771)
VERIFY_CHUNK_BYTES = 64 * 1024**2


def _parse_byte_size(value: str) -> int:
    normalized = value.strip().lower()
    suffixes = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
    }
    digits = normalized.rstrip("abcdefghijklmnopqrstuvwxyz")
    suffix = normalized[len(digits) :]
    if not digits.isdigit() or suffix not in suffixes:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}; examples: 1073741824, 512MiB, 1GiB")
    result = int(digits) * suffixes[suffix]
    if result <= 0:
        raise argparse.ArgumentTypeError("byte size must be positive")
    return result


def _parse_region_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("region counts must be comma-separated positive integers") from exc
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("region counts must be comma-separated positive integers")
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("region counts must not contain duplicates")
    return counts


def _split_total_bytes(total_bytes: int, region_count: int) -> list[int]:
    if region_count <= 0:
        raise ValueError("region_count must be positive")
    if total_bytes < region_count:
        raise ValueError(f"total_bytes={total_bytes} is too small for region_count={region_count}")
    quotient, remainder = divmod(total_bytes, region_count)
    return [quotient + (index < remainder) for index in range(region_count)]


def _pattern_byte(region_index: int) -> int:
    return region_index % 251 + 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region-counts",
        type=_parse_region_counts,
        default=DEFAULT_REGION_COUNTS,
        help="Comma-separated registration region counts (default: 1,4,16,64,199,339,771).",
    )
    parser.add_argument(
        "--total-bytes",
        type=_parse_byte_size,
        default=1024**3,
        help="Fixed bytes allocated at each endpoint for every case (default: 1GiB).",
    )
    parser.add_argument(
        "--transfer-slices",
        type=int,
        default=771,
        help="Logical tensor slices transferred in every case, independent of registration regions (default: 771).",
    )
    parser.add_argument("--cycles", type=int, default=2, help="TMS pause/resume cycles after the initial READ.")
    parser.add_argument("--source-device", default="cpu", help="Stable READ source device (default: cpu).")
    parser.add_argument("--initiator-device", default="cuda:0", help="TMS-managed READ destination (default: cuda:0).")
    parser.add_argument("--host", default="127.0.0.1", help="Source NIXL metadata endpoint address.")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Source metadata port. Zero chooses a free port for each case (recommended).",
    )
    parser.add_argument("--operation-timeout-s", type=float, default=300.0)
    parser.add_argument("--case-timeout-s", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, help="Combined JSON report path.")

    # Internal worker options used by the launcher. They are intentionally not
    # exposed in --help because workers require the launcher's TMS environment.
    parser.add_argument("--worker-role", choices=("source", "initiator"), help=argparse.SUPPRESS)
    parser.add_argument("--region-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    if args.transfer_slices < 1:
        parser.error("--transfer-slices must be positive")
    if args.operation_timeout_s <= 0 or args.case_timeout_s <= 0:
        parser.error("timeouts must be positive")
    if args.worker_role is None:
        if args.output is None:
            parser.error("--output is required")
        if not args.initiator_device.startswith("cuda"):
            parser.error("--initiator-device must be a CUDA device because this probe exercises TMS")
        too_large = [count for count in args.region_counts if count > args.total_bytes]
        if too_large:
            parser.error(f"region counts exceed --total-bytes: {too_large}")
        if any(count > args.transfer_slices for count in args.region_counts):
            parser.error("every region count must be <= --transfer-slices")
        if args.transfer_slices > args.total_bytes:
            parser.error("--transfer-slices must be <= --total-bytes")
    else:
        if args.region_count is None or args.worker_output is None:
            parser.error("internal workers require --region-count and --worker-output")
        if args.worker_role == "source" and args.ready_file is None:
            parser.error("the source worker requires --ready-file")


def _sync_device(torch_module, device) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def _allocate_tensors(torch_module, device, sizes: list[int]) -> list[Any]:
    tensors = [torch_module.empty(size, dtype=torch_module.uint8, device=device) for size in sizes]
    _sync_device(torch_module, device)
    storage_addresses = {tensor.untyped_storage().data_ptr() for tensor in tensors}
    if len(storage_addresses) != len(tensors):
        raise RuntimeError(f"expected {len(tensors)} distinct tensor storages, found {len(storage_addresses)}")
    return tensors


def _make_transfer_slices(regions: list[Any], transfer_slice_count: int) -> list[Any]:
    slices_per_region = _split_total_bytes(transfer_slice_count, len(regions))
    slices = []
    for region, region_slice_count in zip(regions, slices_per_region, strict=True):
        slice_sizes = _split_total_bytes(region.numel(), region_slice_count)
        offset = 0
        for slice_size in slice_sizes:
            slices.append(region.narrow(0, offset, slice_size))
            offset += slice_size
    if len(slices) != transfer_slice_count:
        raise RuntimeError(f"expected {transfer_slice_count} transfer slices, built {len(slices)}")
    return slices


def _fill_source(torch_module, tensors: list[Any], device) -> float:
    start = time.perf_counter()
    for index, tensor in enumerate(tensors):
        tensor.fill_(_pattern_byte(index))
    _sync_device(torch_module, device)
    return time.perf_counter() - start


def _zero_destination(torch_module, tensors: list[Any], device) -> float:
    start = time.perf_counter()
    for tensor in tensors:
        tensor.zero_()
    _sync_device(torch_module, device)
    return time.perf_counter() - start


def _verify_destination(torch_module, tensors: list[Any], device) -> tuple[float, bool]:
    start = time.perf_counter()
    checks = []
    for index, tensor in enumerate(tensors):
        expected = _pattern_byte(index)
        checks.extend(torch_module.all(chunk == expected) for chunk in tensor.split(VERIFY_CHUNK_BYTES))
    verified = bool(torch_module.all(torch_module.stack(checks)).item())
    _sync_device(torch_module, device)
    return time.perf_counter() - start, verified


def _register_tensors(agent, tensors: list[Any], device, torch_module) -> tuple[Any, dict[str, Any]]:
    _sync_device(torch_module, device)
    build_start = time.perf_counter()
    reg_descs = agent.get_reg_descs(tensors)
    build_s = time.perf_counter() - build_start
    if reg_descs is None:
        raise RuntimeError("NIXL failed to build registration descriptors")

    register_start = time.perf_counter()
    registered_descs = agent.register_memory(reg_descs)
    register_s = time.perf_counter() - register_start
    if registered_descs is None:
        raise RuntimeError("NIXL memory registration failed")
    desc_count = int(registered_descs.descCount())
    if desc_count != len(tensors):
        raise RuntimeError(f"expected {len(tensors)} registered descriptors, got {desc_count}")
    return registered_descs, {
        "descriptor_build_s": build_s,
        "register_memory_s": register_s,
        "registered_regions": desc_count,
    }


def _wait_until(predicate: Callable[[], bool], timeout_s: float, description: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        time.sleep(0.001)


def _wait_for_notification(agent, remote_agent: str, timeout_s: float) -> bytes:
    result: list[bytes] = []

    def poll() -> bool:
        notifications = agent.get_new_notifs()
        result.extend(notifications.get(remote_agent, []))
        return bool(result)

    _wait_until(poll, timeout_s, f"a notification from {remote_agent}")
    if len(result) != 1:
        raise RuntimeError(f"expected one descriptor notification from {remote_agent}, got {len(result)}")
    return result[0]


def _completion_tag(cycle: int) -> bytes:
    return f"nixl-registration-probe-cycle-{cycle}".encode()


def _set_device(torch_module, device_text: str):
    device = torch_module.device(device_text)
    if device.type == "cuda":
        torch_module.cuda.set_device(device)
    return device


def _source_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nixl._api import nixl_agent, nixl_agent_config

    device = _set_device(torch, args.source_device)
    sizes = _split_total_bytes(args.total_bytes, args.region_count)
    agent = nixl_agent(SOURCE_AGENT, nixl_agent_config(True, True, args.port))

    allocate_start = time.perf_counter()
    tensors = _allocate_tensors(torch, device, sizes)
    transfer_slices = _make_transfer_slices(tensors, args.transfer_slices)
    allocate_s = time.perf_counter() - allocate_start
    fill_s = _fill_source(torch, transfer_slices, device)
    reg_descs, registration = _register_tensors(agent, tensors, device, torch)
    xfer_desc_start = time.perf_counter()
    xfer_descs = agent.get_xfer_descs(transfer_slices)
    xfer_desc_build_s = time.perf_counter() - xfer_desc_start
    if xfer_descs is None or int(xfer_descs.descCount()) != args.transfer_slices:
        raise RuntimeError(f"failed to build {args.transfer_slices} source transfer descriptors")
    serialized_descs = agent.get_serialized_descs(xfer_descs)

    args.ready_file.write_text("ready\n", encoding="utf-8")
    _wait_until(
        lambda: agent.check_remote_metadata(INITIATOR_AGENT),
        args.operation_timeout_s,
        "initiator metadata",
    )

    waits = []
    for cycle in range(args.cycles + 1):
        tag = _completion_tag(cycle)
        agent.send_notif(INITIATOR_AGENT, serialized_descs)
        wait_start = time.perf_counter()
        _wait_until(
            lambda tag=tag: agent.check_remote_xfer_done(INITIATOR_AGENT, tag),
            args.operation_timeout_s,
            f"cycle {cycle} READ completion",
        )
        waits.append({"cycle": cycle, "wait_for_read_s": time.perf_counter() - wait_start})

    agent.deregister_memory(reg_descs)
    del tensors
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "role": "source",
        "device": str(device),
        "total_bytes": args.total_bytes,
        "region_count": args.region_count,
        "transfer_slices": args.transfer_slices,
        "min_region_bytes": min(sizes),
        "max_region_bytes": max(sizes),
        "allocate_s": allocate_s,
        "fill_s": fill_s,
        "registration": registration,
        "transfer_descriptor_build_s": xfer_desc_build_s,
        "rounds": waits,
    }


def _initiator_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from nixl._api import nixl_agent, nixl_agent_config
    from torch_memory_saver import torch_memory_saver

    device = _set_device(torch, args.initiator_device)
    sizes = _split_total_bytes(args.total_bytes, args.region_count)
    agent = nixl_agent(INITIATOR_AGENT, nixl_agent_config(True, True, 0))
    agent.fetch_remote_metadata(SOURCE_AGENT, args.host, args.port)
    agent.send_local_metadata(args.host, args.port)
    _wait_until(
        lambda: agent.check_remote_metadata(SOURCE_AGENT),
        args.operation_timeout_s,
        "source metadata",
    )

    allocate_start = time.perf_counter()
    with torch_memory_saver.region(tag=TMS_TAG, enable_cpu_backup=False):
        tensors = _allocate_tensors(torch, device, sizes)
        transfer_slices = _make_transfer_slices(tensors, args.transfer_slices)
    allocate_s = time.perf_counter() - allocate_start
    original_addresses = [tensor.data_ptr() for tensor in tensors]

    rounds = []
    for cycle in range(args.cycles + 1):
        round_result: dict[str, Any] = {"cycle": cycle, "after_tms_resume": cycle > 0}
        if cycle > 0:
            pause_start = time.perf_counter()
            torch_memory_saver.pause(TMS_TAG)
            round_result["tms_pause_s"] = time.perf_counter() - pause_start

            resume_start = time.perf_counter()
            torch_memory_saver.resume(TMS_TAG)
            round_result["tms_resume_s"] = time.perf_counter() - resume_start
            resumed_addresses = [tensor.data_ptr() for tensor in tensors]
            round_result["virtual_addresses_preserved"] = resumed_addresses == original_addresses
            if not round_result["virtual_addresses_preserved"]:
                raise RuntimeError(f"TMS changed a tensor virtual address in cycle {cycle}")

        round_result["zero_destination_s"] = _zero_destination(torch, tensors, device)
        reg_descs, registration = _register_tensors(agent, tensors, device, torch)
        round_result.update(registration)

        remote_desc_bytes = _wait_for_notification(agent, SOURCE_AGENT, args.operation_timeout_s)
        remote_descs = agent.deserialize_descs(remote_desc_bytes)
        xfer_desc_start = time.perf_counter()
        local_descs = agent.get_xfer_descs(transfer_slices)
        round_result["transfer_descriptor_build_s"] = time.perf_counter() - xfer_desc_start
        if local_descs is None or int(local_descs.descCount()) != args.transfer_slices:
            raise RuntimeError(f"failed to build {args.transfer_slices} local transfer descriptors")
        if int(remote_descs.descCount()) != args.transfer_slices:
            raise RuntimeError(
                f"expected {args.transfer_slices} remote transfer descriptors, got {remote_descs.descCount()}"
            )

        transfer_start = time.perf_counter()
        initialize_start = time.perf_counter()
        handle = agent.initialize_xfer(
            "READ",
            local_descs,
            remote_descs,
            SOURCE_AGENT,
            _completion_tag(cycle),
        )
        round_result["initialize_read_s"] = time.perf_counter() - initialize_start
        if handle is None:
            raise RuntimeError(f"failed to initialize cycle {cycle} READ")

        post_start = time.perf_counter()
        state = agent.transfer(handle)
        round_result["post_read_s"] = time.perf_counter() - post_start
        if state == "ERR":
            raise RuntimeError(f"failed to post cycle {cycle} READ")

        wait_start = time.perf_counter()
        while state != "DONE":
            if time.perf_counter() - wait_start >= args.operation_timeout_s:
                raise TimeoutError(f"timed out waiting for cycle {cycle} READ")
            state = agent.check_xfer_state(handle)
            if state == "ERR":
                raise RuntimeError(f"cycle {cycle} READ entered the error state")
            if state != "DONE":
                time.sleep(0.001)
        round_result["wait_read_s"] = time.perf_counter() - wait_start
        round_result["read_total_s"] = time.perf_counter() - transfer_start
        agent.release_xfer_handle(handle)

        verify_s, verified = _verify_destination(torch, transfer_slices, device)
        round_result["verify_s"] = verify_s
        round_result["verified"] = verified
        agent.deregister_memory(reg_descs)
        if not verified:
            raise RuntimeError(f"cycle {cycle} READ data verification failed")
        rounds.append(round_result)

    del tensors
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "role": "initiator",
        "device": str(device),
        "total_bytes": args.total_bytes,
        "region_count": args.region_count,
        "transfer_slices": args.transfer_slices,
        "min_region_bytes": min(sizes),
        "max_region_bytes": max(sizes),
        "allocate_s": allocate_s,
        "rounds": rounds,
    }


def _worker_main(args: argparse.Namespace) -> None:
    report: dict[str, Any] = {
        "role": args.worker_role,
        "region_count": args.region_count,
        "error": None,
    }
    try:
        if args.worker_role == "source":
            report.update(_source_worker(args))
        else:
            report.update(_initiator_worker(args))
    except Exception as exc:
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        args.worker_output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _find_tms_preload() -> Path:
    spec = importlib.util.find_spec("torch_memory_saver")
    if spec is None or spec.origin is None:
        raise RuntimeError("torch_memory_saver is not importable in the current Python environment")
    site_packages = Path(spec.origin).resolve().parents[1]
    exact = site_packages / "torch_memory_saver_hook_mode_preload.abi3.so"
    if exact.is_file():
        return exact
    candidates = sorted(site_packages.glob("torch_memory_saver_hook_mode_preload*.so"))
    if not candidates:
        raise FileNotFoundError(f"TMS preload library not found under {site_packages}")
    return candidates[0]


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _worker_command(
    args: argparse.Namespace,
    role: str,
    region_count: int,
    port: int,
    worker_output: Path,
    ready_file: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-role",
        role,
        "--region-count",
        str(region_count),
        "--total-bytes",
        str(args.total_bytes),
        "--cycles",
        str(args.cycles),
        "--transfer-slices",
        str(args.transfer_slices),
        "--source-device",
        args.source_device,
        "--initiator-device",
        args.initiator_device,
        "--host",
        args.host,
        "--port",
        str(port),
        "--operation-timeout-s",
        str(args.operation_timeout_s),
        "--case-timeout-s",
        str(args.case_timeout_s),
        "--worker-output",
        str(worker_output),
    ]
    if role == "source":
        command.extend(("--ready-file", str(ready_file)))
    return command


def _worker_env(preload_path: Path, enable_tms: bool) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if enable_tms:
        previous_preload = env.get("LD_PRELOAD", "").strip()
        env["LD_PRELOAD"] = f"{preload_path} {previous_preload}".strip()
        env["TMS_INIT_ENABLE"] = "1"
        env["TMS_INIT_ENABLE_CPU_BACKUP"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        env["PSRL_TMS_ENABLE"] = "1"
    return env


def _terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _wait_for_ready(process: subprocess.Popen[Any], ready_file: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not ready_file.is_file():
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"source worker exited before becoming ready (return code {return_code})")
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the source worker to become ready")
        time.sleep(0.05)


def _wait_for_process_pair(source: subprocess.Popen[Any], initiator: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while source.poll() is None or initiator.poll() is None:
        if source.poll() not in (None, 0):
            raise RuntimeError(f"source worker failed with return code {source.returncode}")
        if initiator.poll() not in (None, 0):
            raise RuntimeError(f"initiator worker failed with return code {initiator.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"case exceeded {timeout_s:.1f}s")
        time.sleep(0.1)
    if source.returncode != 0 or initiator.returncode != 0:
        raise RuntimeError(
            f"worker failure: source return code={source.returncode}, initiator return code={initiator.returncode}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"worker report was not written: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _timing_summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "mean": statistics.fmean(values), "max": max(values)}


def _combine_case(
    region_count: int,
    source_report: dict[str, Any],
    initiator_report: dict[str, Any],
    source_log: Path,
    initiator_log: Path,
) -> dict[str, Any]:
    rounds = initiator_report["rounds"]
    reregister_values = [float(item["register_memory_s"]) for item in rounds if item["after_tms_resume"]]
    return {
        "region_count": region_count,
        "total_bytes": initiator_report["total_bytes"],
        "transfer_slices": initiator_report["transfer_slices"],
        "min_region_bytes": initiator_report["min_region_bytes"],
        "max_region_bytes": initiator_report["max_region_bytes"],
        "source_registration_s": source_report["registration"]["register_memory_s"],
        "initial_registration_s": rounds[0]["register_memory_s"],
        "reregister_after_tms_s": _timing_summary(reregister_values),
        "read_s": _timing_summary([float(item["read_total_s"]) for item in rounds]),
        "all_reads_verified": all(bool(item["verified"]) for item in rounds),
        "rounds": rounds,
        "source_report": source_report,
        "logs": {"source": str(source_log), "initiator": str(initiator_log)},
    }


def _run_case(
    args: argparse.Namespace,
    region_count: int,
    case_index: int,
    preload_path: Path,
    temp_dir: Path,
) -> dict[str, Any]:
    port = args.port + case_index if args.port else _free_port(args.host)
    ready_file = temp_dir / f"regions_{region_count}.ready"
    source_output = temp_dir / f"regions_{region_count}.source.json"
    initiator_output = temp_dir / f"regions_{region_count}.initiator.json"
    source_log = args.output.parent / f"{args.output.stem}.regions_{region_count}.source.log"
    initiator_log = args.output.parent / f"{args.output.stem}.regions_{region_count}.initiator.log"

    source_process = None
    initiator_process = None
    with (
        source_log.open("w", encoding="utf-8") as source_stream,
        initiator_log.open("w", encoding="utf-8") as initiator_stream,
    ):
        try:
            source_process = subprocess.Popen(
                _worker_command(args, "source", region_count, port, source_output, ready_file),
                stdout=source_stream,
                stderr=subprocess.STDOUT,
                env=_worker_env(preload_path, enable_tms=False),
                text=True,
            )
            _wait_for_ready(source_process, ready_file, args.operation_timeout_s)
            initiator_process = subprocess.Popen(
                _worker_command(args, "initiator", region_count, port, initiator_output, ready_file),
                stdout=initiator_stream,
                stderr=subprocess.STDOUT,
                env=_worker_env(preload_path, enable_tms=True),
                text=True,
            )
            _wait_for_process_pair(source_process, initiator_process, args.case_timeout_s)
        finally:
            _terminate_process(initiator_process)
            _terminate_process(source_process)

    source_report = _read_json(source_output)
    initiator_report = _read_json(initiator_output)
    if source_report.get("error") or initiator_report.get("error"):
        raise RuntimeError(f"worker error for region_count={region_count}; see {source_log} and {initiator_log}")
    return _combine_case(region_count, source_report, initiator_report, source_log, initiator_log)


def _launcher_main(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preload_path = _find_tms_preload()
    report: dict[str, Any] = {
        "region_counts": list(args.region_counts),
        "total_bytes": args.total_bytes,
        "cycles": args.cycles,
        "transfer_slices": args.transfer_slices,
        "source_device": args.source_device,
        "initiator_device": args.initiator_device,
        "tms_preload": str(preload_path),
        "cases": [],
        "error": None,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="nixl-registration-probe-", dir=args.output.parent) as temp_name:
            temp_dir = Path(temp_name)
            for case_index, region_count in enumerate(args.region_counts):
                case = _run_case(args, region_count, case_index, preload_path, temp_dir)
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(
                    "[NIXL_REGISTRATION_PROBE] "
                    f"regions={region_count} "
                    f"initial_register_s={case['initial_registration_s']:.6f} "
                    f"reregister_mean_s={case['reregister_after_tms_s']['mean']:.6f} "
                    f"read_mean_s={case['read_s']['mean']:.6f} "
                    f"verified={case['all_reads_verified']}",
                    flush=True,
                )
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)
    if args.worker_role is not None:
        _worker_main(args)
    else:
        _launcher_main(args)


if __name__ == "__main__":
    main()
