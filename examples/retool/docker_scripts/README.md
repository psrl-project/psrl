# Docker Image Preparation for ReTool

This directory contains the scripts needed to bake the
[SandboxFusion](https://github.com/bytedance/SandboxFusion) server image
(`code_sandbox:server`) into a shared-filesystem tar, fan it out to every
node in the cluster via `pssh`, and manage the `dockerd` daemon on those
nodes.

Once every node has the image locally, continue to
[`../sandbox_fusion/README.md`](../sandbox_fusion/README.md) to launch the
actual service.

---

## Files

| File | Description |
|------|-------------|
| [`docker_install.sh`](docker_install.sh) | Pull an image from Docker Hub (via `docker`, `skopeo`, or `crane`) and save it as a tar under `DOCKER_IMAGE_DIR`. Shared-filesystem friendly. |
| [`docker_copy.sh`](docker_copy.sh) | `pssh` fan-out: copy the tar to `/tmp/` on every node, `docker load`, retag to `DOCKER_IMAGE_TAG`, clean up. |
| [`docker_manager.sh`](docker_manager.sh) | Cluster-wide `dockerd` control: `start` / `stop` / `restart` / `status` / `logs` on one host or all hosts. |
| [`docker_common.sh`](docker_common.sh) | Shared helpers (sourced only — not executable). Parses `DOCKER_NODE_IPS` into `DOCKER_CLUSTER_HOSTS` / `DOCKER_CLUSTER_MANAGER` / `DOCKER_CLUSTER_WORKERS`. |
| [`example.sh`](example.sh) | End-to-end reference invocation chaining `docker_install.sh` + `docker_copy.sh`. |

---

## One-time flow: bake once, fan out to every node

The SandboxFusion image is ~1–2 GB. Pulling it directly on every node hits
Docker Hub `num_nodes` times and often fails on proxy-restricted clusters
(the Docker daemon usually can't read user-space `http_proxy` env vars).
The recommended flow instead is:

1. Pull **once** to a shared-FS tar on the manager node
   ([Step 1](#step-1--build-or-pull-the-image-once)).
2. Fan `docker load` out to every node over `pssh`
   ([Step 2](#step-2--fan-the-tar-out-to-every-node)).

That way Docker Hub is hit once per image.

---

### Step 1 — Build or pull the image once

#### Option A: build from source

Follow the upstream guide at
[bytedance/SandboxFusion](https://github.com/bytedance/SandboxFusion) to
build the server image, then tag it:

```bash
docker tag <built-image>  code_sandbox:server
```

Save it to the shared FS so you can fan it out in Step 2:

```bash
mkdir -p ${PSRL_WORKSPACE}/docker_images
docker save -o ${PSRL_WORKSPACE}/docker_images/code_sandbox.tar code_sandbox:server
```

#### Option B: pull from a Docker Hub mirror via `skopeo`

On proxy-restricted clusters, `docker pull` often fails because `dockerd`
doesn't see the user's `http_proxy`. `skopeo` is a user-space tool that
honours `HTTP_PROXY` / `HTTPS_PROXY` from the current shell and writes
straight to a `docker-archive` tar without needing `dockerd` at all:

```bash
DOCKERHUB_MIRROR=docker.m.daocloud.io \
DOCKER_INSTALL_METHOD=skopeo \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_install.sh
```

What each variable does:

| Variable | Effect |
|----------|--------|
| `DOCKER_IMAGE_DIR` | Output directory for the tar (must live on a shared FS for Step 2) |
| `DOCKER_IMAGE_FILE` | Tar filename (basename) |
| `DOCKER_IMAGE_TAG` | Image reference to pull (e.g. `code_sandbox:server`, `python:3.11-slim`) |
| `DOCKER_INSTALL_METHOD` | `docker` (default) \| `skopeo` \| `crane` |
| `DOCKERHUB_MIRROR` | Registry mirror host. Short refs get rewritten to `<mirror>/library/<repo>` / `<mirror>/<ns>/<repo>` automatically |
| `SKOPEO` | Optional explicit path to the `skopeo` binary |

The mirror-rewriting rule in `apply_dockerhub_mirror` is the same one used
by PSRL's install scripts: anything that already contains a `.` in the
first path component (i.e. looks like a registry host) is left alone, so
specifying `quay.io/foo/bar` won't accidentally get re-prefixed.

> **Note**: `docker.m.daocloud.io` is currently allow-listed — it serves
> `python:3.11-slim` and related common images fine, but does **not**
> serve arbitrary third-party repositories. If you need a less common
> source, fall back to `DOCKER_INSTALL_METHOD=docker` with a running
> daemon and `HTTP_PROXY` configured daemon-side.

#### Option C: pull via `crane`

Same as `skopeo` but using [`crane`](https://github.com/google/go-containerregistry):

```bash
DOCKER_INSTALL_METHOD=crane \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_install.sh
```

---

### Step 2 — Fan the tar out to every node

Once the tar exists on the shared FS, fan out with `docker_copy.sh`:

```bash
DOCKER_NODE_IPS="${NODE_IPS}" \
DOCKER_NODE_NUM=8 \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_copy.sh
```

What this does, per node, in parallel over `pssh`:

1. `cp $DOCKER_IMAGE_DIR/$DOCKER_IMAGE_FILE /tmp/` — copies from the shared
   FS to local disk. (The shared FS already sees the file, so this is
   really just a local-disk cache so that `docker load` doesn't stream from
   network storage.)
2. `docker load -i /tmp/<file>` — imports the image into the local dockerd.
3. Parses the output of `docker load` (`Loaded image: <ref>` or
   `Loaded image ID: sha256:<hash>`) to extract the loaded reference, then
   `docker tag <loaded> $DOCKER_IMAGE_TAG`. This lets you load a tar whose
   original tag doesn't match what the SandboxFusion service expects
   (`code_sandbox:server`) without having to know the original tag ahead
   of time.
4. `rm /tmp/<file>` — cleans up the local cache.

`docker_copy.sh` env vars:

| Variable | Required | Effect |
|----------|----------|--------|
| `DOCKER_NODE_IPS` | yes | Comma-separated list of `ip:gpu_count` pairs. `gpu_count` is stripped before use — it's kept for parity with other scripts. |
| `DOCKER_NODE_NUM` | no | Limit to the first `N` nodes. Default: all entries |
| `DOCKER_IMAGE_DIR` | yes | Source directory containing the tar |
| `DOCKER_IMAGE_FILE` | yes | Tar filename |
| `DOCKER_IMAGE_TAG` | no | If set, retag the loaded image to this reference on every node |

---

### Putting it together — the reference script

[`example.sh`](example.sh) chains Steps 1 and 2 in a single invocation.
The committed copy pulls `python:3.11-slim` as a smoke test, but the
pattern is the same for `code_sandbox:server`:

```bash
DOCKER_IMAGE_TAG="code_sandbox:server"
DOCKER_IMAGE_FILE="code_sandbox.tar"

DOCKERHUB_MIRROR=docker.m.daocloud.io \
DOCKER_INSTALL_METHOD=skopeo \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=$DOCKER_IMAGE_FILE \
DOCKER_IMAGE_TAG=$DOCKER_IMAGE_TAG \
  bash examples/retool/docker_scripts/docker_install.sh

DOCKER_NODE_IPS="${NODE_IPS}" \
DOCKER_NODE_NUM=8 \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=$DOCKER_IMAGE_FILE \
DOCKER_IMAGE_TAG=$DOCKER_IMAGE_TAG \
  bash examples/retool/docker_scripts/docker_copy.sh
```

> **Heads up**: the committed `example.sh` currently points at
> `scripts/docker/docker_install.sh`, which is a
> legacy path that no longer exists — the real scripts are the ones in
> this directory (`examples/retool/docker_scripts/`). Update the path when
> copying the example.

---

## Managing docker daemons across the cluster

[`docker_manager.sh`](docker_manager.sh) is a thin `pssh` wrapper for
starting / stopping / inspecting `dockerd` on one host or all hosts:

```bash
# Start dockerd everywhere (idempotent: pkills any existing dockerd first on restart)
DOCKER_NODE_IPS="${NODE_IPS}" \
DOCKER_NODE_NUM=8 \
  bash examples/retool/docker_scripts/docker_manager.sh start all

# Check whether dockerd is responsive on every node
DOCKER_NODE_IPS="${NODE_IPS}" \
  bash examples/retool/docker_scripts/docker_manager.sh status all

# Tail /var/log/docker.log on one specific node
DOCKER_NODE_IPS="${NODE_IPS}" \
  bash examples/retool/docker_scripts/docker_manager.sh logs "192.168.1.1"
```

Actions:

| Action | Command run on each node |
|--------|--------------------------|
| `start` | `nohup dockerd > /var/log/docker.log 2>&1 &` |
| `stop` | `pkill dockerd` |
| `restart` | `pkill dockerd; nohup dockerd > /var/log/docker.log 2>&1 &` |
| `status` | `docker version >/dev/null 2>&1` → `✓` running / `✗` not running |
| `logs` | `tail -f /var/log/docker.log` |

`launch_service.sh` (in `../sandbox_fusion/`) already does a `start`-if-not-running
pass before initialising the swarm, so in most cases you don't need to
invoke `docker_manager.sh` explicitly — it's useful mainly for debugging
stuck nodes.

> **Proxy caveat**: the `start` action does **not** set any
> `http_proxy` / `https_proxy` env var on the daemon. That's intentional
> for the SandboxFusion deploy path (the image has already been `docker
> load`-ed locally in Step 2, so the daemon never needs to hit the
> internet). If you later want this daemon to be able to `docker pull`,
> you'll need to write a systemd drop-in or `/etc/docker/daemon.json`
> manually.

---

## Env var cheat sheet

| Variable | Used by | Description |
|----------|---------|-------------|
| `DOCKER_IMAGE_DIR` | install, copy | Directory containing the tar (shared FS for fan-out) |
| `DOCKER_IMAGE_FILE` | install, copy | Tar filename (basename) |
| `DOCKER_IMAGE_TAG` | install, copy | Image reference; used as the pull ref in install and the retag target in copy |
| `DOCKER_INSTALL_METHOD` | install | `docker` (default) \| `skopeo` \| `crane` |
| `DOCKERHUB_MIRROR` | install | Registry-mirror host (e.g. `docker.m.daocloud.io`). Rewrites short refs |
| `SKOPEO` | install | Explicit path to `skopeo` binary (auto-detected if on `$PATH` or at `/usr/bin/skopeo`) |
| `DOCKER_NODE_IPS` | copy, manager | Comma-separated `ip:gpu_count`; `gpu_count` is stripped |
| `DOCKER_NODE_NUM` | copy, manager | Limit to first `N` hosts (default: all) |

---

## Verification

After Step 2 finishes, verify the image is present on every node:

```bash
DOCKER_NODE_IPS=... pssh -H "$DOCKER_NODE_IPS_SPACE_SEP" -i \
    "docker images | grep code_sandbox || echo MISSING"
```

Any node printing `MISSING` should be re-fanned:

```bash
DOCKER_NODE_IPS=<that-one-node>:8 DOCKER_NODE_NUM=1 \
DOCKER_IMAGE_DIR=${PSRL_WORKSPACE}/docker_images \
DOCKER_IMAGE_FILE=code_sandbox.tar \
DOCKER_IMAGE_TAG=code_sandbox:server \
  bash examples/retool/docker_scripts/docker_copy.sh
```

Then move on to the full smoke test (Swarm service + `curl`) documented in
[`../sandbox_fusion/README.md`](../sandbox_fusion/README.md).

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `skopeo: command not found` | Package not installed | `dnf -y install skopeo --nobest` — the `--nobest` is required when Docker CE's `containerd.io` is installed (skopeo has a conflicting containerd dep) |
| `docker: permission denied` on a node | Current user not in the `docker` group | `sudo usermod -aG docker $USER && newgrp docker` on the node, or run the scripts as root |
| `pssh: host rejected (publickey)` | SSH key missing | `ssh-copy-id root@<ip>` to every host in `DOCKER_NODE_IPS` first |
| `docker load` fails with `no space left on device` | `/var/lib/docker` is full on one or more nodes | `docker system prune -a` on the affected node, or mount a larger volume at `/var/lib/docker` |
| `Loaded image ID: sha256:...` but no tag appears | The tar was a untagged / `<none>:<none>` export | `DOCKER_IMAGE_TAG` is picked up automatically in that case — `docker_copy.sh` tags it. If `DOCKER_IMAGE_TAG` is unset, the loaded image will be `<none>` and unusable; set it |
| Mirror returns 403 / "manifest unknown" for `code_sandbox:server` | Mirror allow-list doesn't include the repo | Use a different mirror, or build from source (Option A), or fall back to direct pull (`DOCKER_INSTALL_METHOD=docker` + daemon-side proxy) |
| A few nodes randomly miss the image after fan-out | Transient `pssh` SSH timeout on those nodes | Re-run `docker_copy.sh` with just those IPs in `DOCKER_NODE_IPS`; `docker load` is idempotent (reloading the same image is a no-op besides re-tagging) |

---

## Next step

With `code_sandbox:server` present on every node, proceed to
[`../sandbox_fusion/README.md`](../sandbox_fusion/README.md) to launch the
SandboxFusion Swarm service on port 8080. After that, training launch
scripts ([`../fsdp_qwen_7b_dapo.sh`](../fsdp_qwen_7b_dapo.sh),
[`../megatron_qwen_32b_dapo.sh`](../megatron_qwen_32b_dapo.sh),
[`../megatron_qwen3_30b_dapo_small.sh`](../megatron_qwen3_30b_dapo_small.sh))
can reach `http://localhost:8080/run_code` from every rollout worker.
