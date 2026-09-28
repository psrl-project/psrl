# Ray Cluster Startup Scripts

This directory provides two scripts for starting a Ray cluster, each suited to a
different deployment topology. They are functionally equivalent (stop any stale
Ray -> start head -> start workers); the difference is **how commands are
executed on each node**.

## Which one to use?

| Script | Use case |
|--------|----------|
| `ray_start_local.sh` | The **head node is the machine running the script** (typical: single-node multi-GPU, or the local box also acts as head). Use this on the current 8x A800 dev machine. |
| `ray_start_pssh.sh`  | **Pure multi-node** deployment where every node (including head) is reachable over standard SSH:22 with passwordless login, and the script is launched from a management node that does not participate in compute. |

In one line: **starting a cluster on the local machine -> `local`; fanning out
to a set of remote nodes from a management box -> `pssh`.**

## Core differences

### `ray_start_local.sh` (local machine as head + hardened)
- Uses `is_local_host()` to decide the target: **the local machine runs commands
  directly via `bash -c`, no SSH**; only genuine remote workers go over SSH.
- SSH uses `SSH_PORT` (default **2222**, matching the local sshd's non-default
  port); override via environment variable.
- `set -uo pipefail`, checks the exit code of each step, and exits on the first
  node that fails to start.
- Runs `ray status` at the end to verify cluster health, and clears the stale
  `/tmp/ray/ray_current_cluster` address file.

### `ray_start_pssh.sh` (classic multi-node, fire-and-forget)
- All nodes (including head) are reached via `ssh <host>` (**default port 22**).
- Commands are backgrounded + `wait`; no per-node exit-code checking and no
  cluster health check.
- Relies on passwordless SSH:22 from the launcher to every node.
- Note: if the local sshd is not on port 22, this script will fail to reach even
  the local head -- use `local` in that case.

## Usage

```bash
# Option 1: pass a hostfile (one host per line, first line is head;
#           blank lines and # comments are supported)
bash ray_start_local.sh /path/to/hostfile
bash ray_start_pssh.sh  /path/to/hostfile

# Option 2: no argument -- resolve nodes from NODE_IP_LIST / NODE_NUM in the env
bash ray_start_local.sh
```

Both scripts `source /data_storage/yl_test/lhy/psrl.sh` to read environment
variables (`PSRL_WORKSPACE`, `NODE_IP_LIST`, etc.) and activate the conda
environment `psrl-new`.

Common overridable variables: `SSH_PORT` (local only, default 2222),
`PORT` (Ray communication port, 8887), `DASHBOARD_PORT` (8265).
