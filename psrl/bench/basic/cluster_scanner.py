from __future__ import annotations

import json
import re
import shlex
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any

import paramiko


# --- Data Classes ---
@dataclass
class CPUInfo:
    model_name: str | None = None
    sockets: int | None = None
    cores_per_socket: int | None = None
    threads_per_core: int | None = None
    cpus: int | None = None
    cpu_mhz: float | None = None
    total_ram_bytes: int | None = None


@dataclass
class GPUInfo:
    index: int
    name: str | None
    uuid: str | None
    pci_bus_id: str | None
    memory_total_bytes: int | None
    nvlink: list[dict[str, Any]] = field(default_factory=list)  # NVLink info per GPU


@dataclass
class NetworkInterfaceInfo:
    name: str
    is_up: bool | None
    speed_gbps: float | None  # unified: store in Gbps
    link_type: str | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeInfo:
    hostname: str
    ip: str
    cpu: CPUInfo
    gpus: list[GPUInfo]
    net_interfaces: list[NetworkInterfaceInfo]
    nvlink: list[dict[str, Any]] = field(default_factory=list)  # kept for backwards-compat if needed


# --- Paramiko SSH Runner ---
class SSHRunner:
    def __init__(
        self,
        hostname: str,
        username: str | None = None,
        port: int = 22,
        key_filename: str | None = None,
        password: str | None = None,
        timeout: int = 15,
    ):
        self.hostname = hostname
        self.username = username
        self.port = port
        self.key_filename = key_filename
        self.password = password
        self.timeout = timeout
        self._client = None
        self._lock = threading.Lock()

    def _ensure_connected(self):
        with self._lock:
            if self._client:
                return
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(
                self.hostname,
                port=self.port,
                username=self.username or None,
                key_filename=self.key_filename,
                password=self.password,
                timeout=self.timeout,
                look_for_keys=True,
            )
            self._client = c

    def run(self, cmd: str, timeout: int | None = None) -> tuple[int, str, str]:
        try:
            self._ensure_connected()
        except Exception as e:
            return 255, "", f"SSH_CONNECT_ERROR: {e}"
        try:
            stdin, stdout, stderr = self._client.exec_command(cmd, timeout=timeout or self.timeout)
            out = stdout.read().decode(errors="ignore")
            err = stderr.read().decode(errors="ignore")
            rc = stdout.channel.recv_exit_status()
            return rc, out, err
        except Exception as e:
            return 254, "", f"SSH_EXEC_ERROR: {e}"

    def close(self):
        with self._lock:
            if self._client:
                self._client.close()
                self._client = None

    def __enter__(self):
        self._ensure_connected()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# --- Parsers ---
def parse_lscpu(output: str) -> dict[str, str]:
    d = {}
    for line in output.splitlines():
        if ":" in line:
            k, v = [s.strip() for s in line.split(":", 1)]
            d[k] = v
    return d


def parse_meminfo_kb(output: str) -> int | None:
    m = re.search(r"MemTotal:\s+(\d+)\s+kB", output)
    if m:
        return int(m.group(1)) * 1024
    return None


def parse_nvidia_query(output: str) -> list[dict[str, str]]:
    items = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                idx = int(parts[0])
            except ValueError:
                continue
            items.append(
                {
                    "index": idx,
                    "name": parts[1],
                    "uuid": parts[2],
                    "pci_bus_id": parts[3],
                    "memory_total_mb": parts[4],
                }
            )
    return items


# --- Microbenchmark Runner ---
class MicrobenchRunner:
    LOCAL_P2P_SCRIPT = r"""
import json, time, sys
try:
    import torch
except Exception as e:
    print(json.dumps({"error":"import_torch_failed","detail":str(e)}))
    sys.exit(0)

def measure_p2p(src, dst, size_gb=0.25):
    nbytes = int(size_gb * (1024**3))
    nelems = max(1, nbytes // 4)
    try:
        a = torch.empty(nelems, dtype=torch.float32, device=f'cuda:{src}')
        a.uniform_(0,1)
        torch.cuda.synchronize()
        t0 = time.time()
        b = a.to(device=f'cuda:{dst}', non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.time()
        bw_bytes_s = (a.element_size() * a.numel()) / (t1 - t0) if (t1 - t0) > 0 else None
        bw_gb_s = bw_bytes_s / (1024**3) if bw_bytes_s else None
        return {"src":src,"dst":dst,"bytes": a.element_size()*a.numel(),"seconds": (t1-t0), "gb_per_s": bw_gb_s}
    except Exception as e:
        return {"src":src,"dst":dst,"error": str(e)}


if __name__ == "__main__":
    ng = 0
    try:
        import torch
        ng = torch.cuda.device_count()
    except:
        pass
    info = {"cuda_available": False, "device_count": ng, "results": []}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
    except:
        print(json.dumps(info))
        sys.exit(0)

    src_dst = None
    if len(sys.argv) >= 3:
        try:
            src_dst = (int(sys.argv[1]), int(sys.argv[2]))
        except:
            src_dst = None

    pairs = []
    if src_dst:
        pairs.append(src_dst)
    else:
        for i in range(ng):
            for j in range(ng):
                if i==j: continue
                pairs.append((i,j))

    for (i,j) in pairs:
        res = measure_p2p(i,j, size_gb=0.25)
        info["results"].append(res)

    print(json.dumps(info))
"""

    NCCL_ALLREDUCE_SCRIPT_TEMPLATE = r"""
import os, json, time, sys
try:
    import torch
    import torch.distributed as dist
except Exception as e:
    print(json.dumps({"error":"import_torch_failed","detail":str(e)}))
    sys.exit(0)

rank = int(os.environ.get("RANK","0"))
world_size = int(os.environ.get("WORLD_SIZE","2"))
master_addr = os.environ.get("MASTER_ADDR")
master_port = int(os.environ.get("MASTER_PORT","{master_port}"))
try:
    local_gpu = int(sys.argv[1])
except:
    local_gpu = 0

if not torch.cuda.is_available():
    print(json.dumps({"error":"cuda_not_available","rank":rank}))
    sys.exit(0)

torch.cuda.set_device(local_gpu)
os.environ["NCCL_DEBUG"] = "WARN"
init_method = f"tcp://{master_addr}:{master_port}"
try:
    dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size, rank=rank)
except Exception as e:
    print(json.dumps({"error":"init_failed","detail":str(e),"rank":rank}))
    sys.exit(0)

tensor_mb = {tensor_mb}
nelems = (tensor_mb * 1024 * 1024) // 4
t = torch.ones(nelems, dtype=torch.float32, device=f"cuda:{local_gpu}")

for i in range({warmup}):
    try:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    except Exception as e:
        print(json.dumps({"error":"warmup_failed","detail":str(e),"rank":rank}))
        sys.exit(0)

times = []
for i in range({iters}):
    torch.cuda.synchronize()
    t0 = time.time()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    t1 = time.time()
    times.append(t1 - t0)

avg_sec = sum(times)/len(times) if times else None
bytes_per_iter = t.element_size() * t.numel()
gb_per_sec = (bytes_per_iter / (1024.0**3)) / avg_sec if avg_sec and avg_sec>0 else None
out = {
    "rank": rank,
    "local_gpu": local_gpu,
    "tensor_mb": tensor_mb,
    "iters": {iters},
    "avg_seconds": avg_sec,
    "bytes_per_iter": bytes_per_iter,
    "gb_per_sec": gb_per_sec,
    "times": times
}
print(json.dumps(out))
"""

    @staticmethod
    def write_and_run(
        sr: SSHRunner,
        content: str,
        remote_path: str,
        python_cmd: str = "python -u",
        run_args: str | None = None,
        timeout: int = 300,
    ):
        """Write remote file, chmod, then run it. run_args is a single string appended to the command (can be None).
        Returns (rc, stdout, stderr).
        """
        safe_content = content.replace("EOF", "EOFX")
        cmd = f"cat > {shlex.quote(remote_path)} << 'EOFX'\n{safe_content}\nEOFX\n"
        rc, _, err = sr.run(cmd)
        if rc != 0:
            return rc, "", f"failed_writing:{err}"
        rc, out_ch, err_ch = sr.run(f"chmod +x {shlex.quote(remote_path)} || true")
        run_cmd = f"{python_cmd} {shlex.quote(remote_path)}"
        if run_args:
            run_cmd += f" {run_args}"
        rc, out_run, err_run = sr.run(run_cmd, timeout=timeout)
        return rc, out_run, err_run


# --- Cluster Scanner ---
class ClusterScanner:
    """
    Provide node basic scan (CPU/GPU/network + NVLink if available)
    and a standalone microbench API that accepts two (ip, gpu_index) tuples to run nccl.
    """

    def __init__(
        self,
        ips: list[str],
        ssh_username: str | None = None,
        ssh_key: str | None = None,
        ssh_password: str | None = None,
        ssh_port: int = 22,
        concurrency: int = 16,
    ):
        self.ips = ips
        self.username = ssh_username
        self.key = ssh_key
        self.password = ssh_password
        self.port = ssh_port
        self.concurrency = concurrency

    # --- Network Scan ---
    def scan_network_interfaces(self, sr: SSHRunner) -> list[NetworkInterfaceInfo]:
        nlist: list[NetworkInterfaceInfo] = []

        def _parse_ibstat_rate_to_gbps(rate_raw: str | None) -> float | None:
            if not rate_raw:
                return None
            s = rate_raw.strip()
            m = re.search(
                r"([0-9]+(?:\.[0-9]+)?)\s*(G|Gbps|Gbit/s|Gbp/s|G/s|M|Mbps|Mb/s)?",
                s,
                re.I,
            )
            if not m:
                m2 = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
                if not m2:
                    return None
                val = float(m2.group(1))
                return float(val)
            val = float(m.group(1))
            unit = (m.group(2) or "").lower()
            if unit.startswith("m"):
                # Mbps -> Gbps
                return val / 1000.0
            return val

        rc, ibstat_out, ibstat_err = sr.run("which ibstat >/dev/null 2>&1 && ibstat || true")
        ibstat_raw = ((ibstat_out or "") + (ibstat_err or "")).strip()

        if rc == 0 and ibstat_raw:
            ca_matches = list(re.finditer(r"^CA\s+'(?P<name>[^']+)'\s*$", ibstat_raw, flags=re.M))
            ca_blocks = []
            if ca_matches:
                for i, m in enumerate(ca_matches):
                    name = m.group("name")
                    start = m.start()
                    end = ca_matches[i + 1].start() if i + 1 < len(ca_matches) else len(ibstat_raw)
                    block = ibstat_raw[start:end].rstrip()
                    ca_blocks.append((name, block))
            else:
                ca_blocks = [("unknown", ibstat_raw)]

            for ca_name, block in ca_blocks:
                port_matches = list(re.finditer(r"^\s*Port\s+(?P<pn>\d+):\s*$", block, flags=re.M))
                if not port_matches:
                    port_blocks = [("1", block)]
                else:
                    port_blocks = []
                    for i, pm in enumerate(port_matches):
                        pn = pm.group("pn")
                        pstart = pm.start()
                        pend = port_matches[i + 1].start() if i + 1 < len(port_matches) else len(block)
                        pblock = block[pstart:pend].rstrip()
                        port_blocks.append((pn, pblock))

                for portnum, port_block in port_blocks:
                    m_ll = re.search(r"Link layer:\s*([^\n\r]+)", port_block, re.I)
                    link_layer = m_ll.group(1).strip() if m_ll else None

                    m_state = re.search(r"State:\s*([^\n\r]+)", port_block, re.I)
                    state = m_state.group(1).strip() if m_state else None

                    m_phys = re.search(r"Physical state:\s*([^\n\r]+)", port_block, re.I)
                    physical_state = m_phys.group(1).strip() if m_phys else None

                    m_rate = re.search(r"Rate:\s*([^\n\r]+)", port_block, re.I)
                    rate_raw = m_rate.group(1).strip() if m_rate else None
                    rate_gbps = _parse_ibstat_rate_to_gbps(rate_raw) if rate_raw else None

                    link_type = "unknown"
                    if link_layer:
                        ll = link_layer.lower()
                        if ll.startswith("ether"):
                            link_type = "roce"
                        elif ll.startswith("infin"):
                            link_type = "infiniband"
                        else:
                            link_type = ll

                    is_up = False
                    if state and "active" in state.lower():
                        is_up = True
                    if physical_state and "linkup" in physical_state.replace(" ", "").lower():
                        is_up = True

                    pseudo_name = f"{ca_name}"
                    details = {
                        "rdma_port": portnum,
                        "rdma_state": state,
                        "rdma_physical_state": physical_state,
                        "rdma_link_layer": link_layer,
                        "raw_rate": rate_raw,
                    }
                    speed_gbps = float(rate_gbps) if (rate_gbps is not None) else None

                    nlist.append(
                        NetworkInterfaceInfo(
                            name=pseudo_name,
                            is_up=is_up,
                            speed_gbps=speed_gbps,
                            link_type=link_type,
                            details=details,
                        )
                    )
            return nlist
        else:
            # fallback: use /sys/class/infiniband to scan devices
            rc_ls, ls_out, ls_err = sr.run("ls /sys/class/infiniband 2>/dev/null || true")
            if rc_ls != 0 or not ls_out.strip():
                return []

            devices = [d.strip() for d in ls_out.strip().splitlines() if d.strip()]

            for device in devices:
                # Check if device has ports directory
                rc_ports, ports_out, _ = sr.run(
                    f"ls /sys/class/infiniband/{shlex.quote(device)}/ports 2>/dev/null | head -10 || true"
                )
                if rc_ports != 0 or not ports_out.strip():
                    # No ports directory, might be a bond device or device without traditional ports
                    # Try to get basic info without port
                    details = {
                        "device": device,
                        "method": "sysfs_fallback",
                        "note": "no_ports_directory",
                    }
                    nlist.append(
                        NetworkInterfaceInfo(
                            name=device,
                            is_up=None,
                            speed_gbps=None,
                            link_type="infiniband",
                            details=details,
                        )
                    )
                    continue

                # Parse port numbers
                port_nums = [p.strip() for p in ports_out.strip().splitlines() if p.strip().isdigit()]
                if not port_nums:
                    port_nums = ["1"]  # default to port 1 if parsing fails

                for port_num in port_nums:
                    port_path = f"/sys/class/infiniband/{device}/ports/{port_num}"

                    # Read rate (format: "X Gbps" or "X Gb/s")
                    rc_rate, rate_raw, _ = sr.run(f"cat {shlex.quote(port_path)}/rate 2>/dev/null || echo ''")
                    rate_gbps = _parse_ibstat_rate_to_gbps(rate_raw.strip()) if rate_raw.strip() else None

                    # Read state
                    rc_state, state_raw, _ = sr.run(f"cat {shlex.quote(port_path)}/state 2>/dev/null || echo ''")
                    state = state_raw.strip() if state_raw.strip() else None

                    # Read physical state
                    rc_phys, phys_raw, _ = sr.run(f"cat {shlex.quote(port_path)}/phys_state 2>/dev/null || echo ''")
                    physical_state = phys_raw.strip() if phys_raw.strip() else None

                    # Read link layer
                    rc_ll, ll_raw, _ = sr.run(f"cat {shlex.quote(port_path)}/link_layer 2>/dev/null || echo ''")
                    link_layer = ll_raw.strip() if ll_raw.strip() else None

                    # Determine link type
                    link_type = "unknown"
                    if link_layer:
                        ll = link_layer.lower()
                        if ll.startswith("ether") or "ethernet" in ll:
                            link_type = "roce"
                        elif ll.startswith("infin") or "infiniband" in ll:
                            link_type = "infiniband"
                        else:
                            link_type = ll
                    else:
                        # Default to infiniband if unknown
                        link_type = "infiniband"

                    # Determine if up
                    is_up = False
                    if state and ("active" in state.lower() or "armed" in state.lower()):
                        is_up = True
                    if physical_state and "linkup" in physical_state.replace(" ", "").lower():
                        is_up = True

                    # Create interface name (device name with port if multiple ports)
                    pseudo_name = f"{device}:{port_num}" if len(port_nums) > 1 else device

                    details = {
                        "device": device,
                        "rdma_port": port_num,
                        "rdma_state": state,
                        "rdma_physical_state": physical_state,
                        "rdma_link_layer": link_layer,
                        "raw_rate": rate_raw.strip() if rate_raw.strip() else None,
                        "method": "sysfs",
                    }

                    nlist.append(
                        NetworkInterfaceInfo(
                            name=pseudo_name,
                            is_up=is_up,
                            speed_gbps=(float(rate_gbps) if (rate_gbps is not None) else None),
                            link_type=link_type,
                            details=details,
                        )
                    )

            return nlist

    # --- Basic Node Scan ---
    def scan_node_basic(self, ip: str) -> NodeInfo:
        sr = SSHRunner(
            hostname=ip,
            username=self.username,
            key_filename=self.key,
            password=self.password,
            port=self.port,
        )
        try:
            with sr:
                rc, out_hn, err_hn = sr.run("hostname -f || hostname")
                hostname = out_hn.strip() or ip

                rc, out_lscpu, err_lscpu = sr.run("LC_ALL=en_US.UTF-8 lscpu || true")
                lscpu = parse_lscpu(out_lscpu)

                rc, out_mem, err_mem = sr.run("cat /proc/meminfo || true")
                total_mem = parse_meminfo_kb(out_mem)

                cpu = CPUInfo(
                    model_name=lscpu.get("Model name"),
                    sockets=(int(lscpu.get("Socket(s)", "0")) if "Socket(s)" in lscpu else None),
                    cores_per_socket=(
                        int(lscpu.get("Core(s) per socket", "0")) if "Core(s) per socket" in lscpu else None
                    ),
                    threads_per_core=(
                        int(lscpu.get("Thread(s) per core", "0")) if "Thread(s) per core" in lscpu else None
                    ),
                    cpus=int(lscpu.get("CPU(s)", "0")) if "CPU(s)" in lscpu else None,
                    cpu_mhz=(float(lscpu.get("CPU MHz", "0.0")) if "CPU MHz" in lscpu else None),
                    total_ram_bytes=total_mem,
                )

                rc, out_nq, err_nq = sr.run(
                    "which nvidia-smi >/dev/null 2>&1 && "
                    "nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,memory.total "
                    "--format=csv,noheader,nounits || true"
                )
                gpus: list[GPUInfo] = []
                if out_nq.strip():
                    entries = parse_nvidia_query(out_nq)
                    for e in entries:
                        mem_mb = None
                        try:
                            mem_mb = int(float(e.get("memory_total_mb", "0")))
                        except (ValueError, TypeError):
                            mem_mb = None
                        g = GPUInfo(
                            index=e["index"],
                            name=e["name"],
                            uuid=e["uuid"],
                            pci_bus_id=e["pci_bus_id"],
                            memory_total_bytes=((mem_mb * 1024 * 1024) if mem_mb else None),
                        )
                        gpus.append(g)
                else:
                    rc, out_lspci, err_lspci = sr.run("lspci | grep -i nvidia || true")
                    idx = 0
                    for ln in out_lspci.splitlines():
                        ln = ln.strip()
                        if ln:
                            g = GPUInfo(
                                index=idx,
                                name=ln,
                                uuid=None,
                                pci_bus_id=None,
                                memory_total_bytes=None,
                            )
                            gpus.append(g)
                            idx += 1

                # network interfaces
                nlist = self.scan_network_interfaces(sr)

                # --- Per-GPU NVLink ---
                rc_nv, out_nv, err_nv = sr.run(
                    "which nvidia-smi >/dev/null 2>&1 && nvidia-smi nvlink --status || true"
                )
                nv_status_raw = ((out_nv or "") + (err_nv or "")).strip()

                # parse the nvlink text and attach parsed entries to corresponding GPUInfo.nvlink
                if nv_status_raw:
                    lines = nv_status_raw.splitlines()
                    curr_idx = None
                    blocks: dict[int, list[str]] = {}
                    for ln in lines:
                        # match lines like: "GPU 5: NVIDIA H20 (UUID: GPU-...)" or "GPU 5:"
                        m = re.match(r"^\s*GPU\s*(\d+)\s*[:\s]", ln, re.I)
                        if m:
                            curr_idx = int(m.group(1))
                            blocks[curr_idx] = [ln]
                        else:
                            if curr_idx is not None:
                                blocks[curr_idx].append(ln)
                            else:
                                # Ignore preamble text before the first GPU block.
                                pass

                    # attach parsed link entries to each GPU object (by index)
                    for g in gpus:
                        g.nvlink = []
                        if g.index in blocks:
                            blk_lines = blocks[g.index]
                            links = []
                            for line in blk_lines:
                                # match "Link 0: 26.562 GB/s" etc.
                                ml = re.search(
                                    r"^\s*Link\s+(\d+)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*(GB/s|GBps|G/s|G/s)?",
                                    line,
                                    re.I,
                                )
                                if ml:
                                    link_id = int(ml.group(1))
                                    speed = float(ml.group(2))
                                    # Preserve the reported numeric rate under the existing `speed_gbps` field.
                                    links.append({"link_id": link_id, "speed_gbps": speed})
                            if links:
                                # sort by link_id for stable ordering
                                links.sort(key=lambda x: x["link_id"])
                                g.nvlink = links
                            else:
                                # no explicit Link lines parsed, keep whole block as raw
                                blk_text = "\n".join(blk_lines).strip()
                                g.nvlink = [
                                    {
                                        "link_id": None,
                                        "speed_gbps": None,
                                        "raw": blk_text,
                                    }
                                ]
                        else:
                            g.nvlink = []
                else:
                    # Keep each `nvlink` list empty when no status is available.
                    for g in gpus:
                        g.nvlink = []

                node = NodeInfo(
                    hostname=hostname,
                    ip=ip,
                    cpu=cpu,
                    gpus=gpus,
                    net_interfaces=nlist,
                    nvlink=[],
                )
                return node
        except Exception:
            return NodeInfo(
                hostname=f"error-{ip}",
                ip=ip,
                cpu=CPUInfo(),
                gpus=[],
                net_interfaces=[],
                nvlink=[],
            )
        finally:
            sr.close()

    # --- Concurrent Node Scan ---
    def scan_all_basic(self) -> list[dict[str, Any]]:
        nodes: list[NodeInfo] = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            futures = {ex.submit(self.scan_node_basic, ip): ip for ip in self.ips}
            for fut in as_completed(futures):
                ip = futures[fut]
                try:
                    node = fut.result()
                except Exception:
                    node = NodeInfo(
                        hostname=f"error-{ip}",
                        ip=ip,
                        cpu=CPUInfo(),
                        gpus=[],
                        net_interfaces=[],
                        nvlink=[],
                    )
                nodes.append(node)
        return [asdict(n) for n in nodes]

    # --- Summary ---
    def scan_summary(self) -> dict[str, Any]:
        """
        Run scan_all_basic() and produce a compact summary dict (JSON-able) with:
          - cpus: list of { prototype: {...}, count: N }
          - gpus: list of { prototype: {...}, count: N }
          - nvlinks: list of { prototype: {...}, count: N }
          - net_interfaces: list of { prototype: {...}, count: N }
        Prototype equality rules are implemented in helper functions below.
        """
        nodes = self.scan_all_basic()

        from collections import Counter

        cpu_counter = Counter()
        gpu_counter = Counter()
        netif_counter = Counter()

        def cpu_prototype_key(cpu: dict[str, Any]) -> str:
            # choose fields that define a CPU "type"
            proto = {
                "model_name": cpu.get("model_name"),
                "sockets": cpu.get("sockets"),
                "cores_per_socket": cpu.get("cores_per_socket"),
                "threads_per_core": cpu.get("threads_per_core"),
                "cpus": cpu.get("cpus"),
                "cpu_mhz": cpu.get("cpu_mhz"),
            }
            return json.dumps(proto, sort_keys=True)

        def gpu_prototype_key(gpu: dict[str, Any]) -> str:
            # ignore index/uuid/pci_bus_id (unique identifiers).
            # use name, memory_total_bytes, and nvlink pattern as the prototype
            nv = gpu.get("nvlink") or []
            # normalize nvlink to list of speeds
            speeds = []
            for it in nv:
                sp = it.get("speed_gbps")
                if sp is None:
                    speeds.append(None)
                else:
                    # round for stable grouping
                    speeds.append(round(float(sp), 6))
            proto = {
                "name": gpu.get("name"),
                "memory_total_bytes": gpu.get("memory_total_bytes"),
                "nvlink_speeds": speeds,
            }
            return json.dumps(proto, sort_keys=True)

        def netif_prototype_key(netif: dict[str, Any]) -> str:
            # ignore name (unique), consider link_type and speed_gbps as defining features
            speed = netif.get("speed_gbps")
            if speed is not None:
                speed = round(float(speed), 3)
            proto = {"link_type": netif.get("link_type"), "speed_gbps": speed}
            return json.dumps(proto, sort_keys=True)

        # walk nodes and aggregate
        for n in nodes:
            cpu = n.get("cpu") or {}
            cpu_key = cpu_prototype_key(cpu)
            cpu_counter[cpu_key] += 1

            for g in n.get("gpus") or []:
                g_key = gpu_prototype_key(g)
                gpu_counter[g_key] += 1

            for ni in n.get("net_interfaces") or []:
                ni_key = netif_prototype_key(ni)
                netif_counter[ni_key] += 1

        def counter_to_list(counter: Counter) -> list[dict[str, Any]]:
            out = []
            for k, v in counter.items():
                try:
                    proto = json.loads(k)
                except json.JSONDecodeError:
                    proto = {"raw": k}
                out.append({"prototype": proto, "count": int(v)})
            # sort by count desc
            out.sort(key=lambda x: x["count"], reverse=True)
            return out

        summary = {
            "nodes_scanned": len(nodes),
            "cpus": counter_to_list(cpu_counter),
            "gpus": counter_to_list(gpu_counter),
            "net_interfaces": counter_to_list(netif_counter),
        }
        return summary

    # --- Pairwise Microbenchmark ---
    def microbench_between(
        self,
        left: tuple[str, int],
        right: tuple[str, int],
        master_port: int = 29500,
        tensor_mb: int = 64,
        iters: int = 10,
        warmup: int = 3,
        timeout: int = 600,
    ) -> dict[str, Any]:
        left_ip, left_gpu = left
        right_ip, right_gpu = right
        if left_ip == right_ip:
            # local intra-node p2p: run the p2p script specifying src/dst
            sr = SSHRunner(
                hostname=left_ip,
                username=self.username,
                key_filename=self.key,
                password=self.password,
                port=self.port,
            )
            try:
                with sr:
                    bench_script = MicrobenchRunner.LOCAL_P2P_SCRIPT
                    # pass src/dst as args
                    run_args = f"{left_gpu} {right_gpu}"
                    rc, out, err = MicrobenchRunner.write_and_run(
                        sr,
                        bench_script,
                        f"/tmp/gpu_p2p_{int(time.time())}.py",
                        python_cmd="python -u",
                        run_args=run_args,
                        timeout=300,
                    )
                    if rc != 0:
                        return {
                            "ok": False,
                            "error": "remote_exec_failed",
                            "rc": rc,
                            "stdout": out,
                            "stderr": err,
                        }
                    try:
                        parsed = json.loads(out.strip().splitlines()[-1])
                        return {
                            "ok": True,
                            "method": "local_p2p",
                            "result": parsed,
                            "stdout": out,
                            "stderr": err,
                        }
                    except Exception as e:
                        return {
                            "ok": False,
                            "error": "parse_failed",
                            "exc": str(e),
                            "stdout": out,
                            "stderr": err,
                        }
            finally:
                sr.close()
        else:
            # Run one copy of the NCCL script per host with rank-specific environment variables.
            script = MicrobenchRunner.NCCL_ALLREDUCE_SCRIPT_TEMPLATE.format(
                master_port=master_port, tensor_mb=tensor_mb, iters=iters, warmup=warmup
            )
            left_sr = SSHRunner(
                hostname=left_ip,
                username=self.username,
                key_filename=self.key,
                password=self.password,
                port=self.port,
            )
            right_sr = SSHRunner(
                hostname=right_ip,
                username=self.username,
                key_filename=self.key,
                password=self.password,
                port=self.port,
            )
            results = {}
            try:
                with left_sr, right_sr:
                    # deploy scripts
                    left_path = f"/tmp/nccl_allreduce_{int(time.time())}_rank0.py"
                    right_path = f"/tmp/nccl_allreduce_{int(time.time())}_rank1.py"

                    # write left
                    rc_lw, _, err_lw = left_sr.run(f"cat > {shlex.quote(left_path)} << 'EOFX'\n{script}\nEOFX\n")
                    if rc_lw != 0:
                        return {
                            "ok": False,
                            "error": "write_left_failed",
                            "stderr": err_lw,
                        }
                    left_sr.run(f"chmod +x {shlex.quote(left_path)} || true")

                    # write right
                    rc_rw, _, err_rw = right_sr.run(f"cat > {shlex.quote(right_path)} << 'EOFX'\n{script}\nEOFX\n")
                    if rc_rw != 0:
                        return {
                            "ok": False,
                            "error": "write_right_failed",
                            "stderr": err_rw,
                        }
                    right_sr.run(f"chmod +x {shlex.quote(right_path)} || true")

                    # run both in parallel
                    def run_rank(sr: SSHRunner, path: str, rank: int, local_gpu: int):
                        cmd = f"MASTER_ADDR={left_ip} MASTER_PORT={master_port} RANK={rank} WORLD_SIZE=2 "
                        cmd += f"python -u {shlex.quote(path)} {local_gpu}"
                        rc, out, err = sr.run(cmd, timeout=timeout)
                        return {"rc": rc, "stdout": out, "stderr": err}

                    with ThreadPoolExecutor(max_workers=2) as ex:
                        future_l = ex.submit(run_rank, left_sr, left_path, 0, left_gpu)
                        future_r = ex.submit(run_rank, right_sr, right_path, 1, right_gpu)
                        out_l = future_l.result()
                        out_r = future_r.result()

                    # try to parse JSON outputs (last non-empty line)
                    parsed_l = None
                    parsed_r = None
                    try:
                        parsed_l = json.loads(out_l["stdout"].strip().splitlines()[-1])
                    except Exception:
                        parsed_l = {
                            "parse_error": True,
                            "stdout": out_l["stdout"],
                            "stderr": out_l["stderr"],
                        }
                    try:
                        parsed_r = json.loads(out_r["stdout"].strip().splitlines()[-1])
                    except Exception:
                        parsed_r = {
                            "parse_error": True,
                            "stdout": out_r["stdout"],
                            "stderr": out_r["stderr"],
                        }

                    results = {
                        "left": {
                            "host": left_ip,
                            "gpu": left_gpu,
                            "out": out_l,
                            "parsed": parsed_l,
                        },
                        "right": {
                            "host": right_ip,
                            "gpu": right_gpu,
                            "out": out_r,
                            "parsed": parsed_r,
                        },
                    }
                    return {"ok": True, "method": "nccl_allreduce", "result": results}
            except Exception as e:
                return {"ok": False, "error": str(e)}
            finally:
                left_sr.close()
                right_sr.close()


# --- Example Usage ---
if __name__ == "__main__":
    ips = ["192.168.1.1", "192.168.1.2"]
    username = "root"
    ssh_port = 36000
    ssh_key = None
    scanner = ClusterScanner(
        ips=ips,
        ssh_username=username,
        ssh_key=ssh_key,
        ssh_port=ssh_port,
        concurrency=8,
    )

    basic = scanner.scan_all_basic()
    print(json.dumps({"nodes": basic}, indent=2))

    # print summary
    summary = scanner.scan_summary()
    print(json.dumps({"summary": summary}, indent=2))

    # out = scanner.microbench_between(("127.0.0.1", 0), ("127.0.0.1", 1))
    # print(json.dumps(out, indent=2))
