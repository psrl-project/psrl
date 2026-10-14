"""
Manage labeled Docker containers and actor reaper sidecars.

The reaper uses a shell process so cleanup remains available after its actor exits.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
import subprocess
import threading
import time

# A dotted module name, not `__file__`. A path-shaped name sits outside the `psrl.*`
# hierarchy, so these records never reached the configured handlers and cleanup ran with
# no telemetry at all: zero log lines across 456 episodes.
psrl_logger = logging.getLogger("psrl.utils.common.docker_utils")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# Upper bound on reaper polling latency after parent exit.
_REAPER_POLL_INTERVAL_SECS = 5

# Serialize and throttle dangling-image pruning across an actor's episodes.
_PRUNE_LOCK = threading.Lock()
_LAST_PRUNE_MONOTONIC = 0.0

# Images removed per `docker rmi` call, to bound the argument list.
_PRUNE_BATCH_SIZE = 200

# Cleanup runs here rather than on asyncio's default executor. `asyncio.to_thread` uses
# that shared pool, capped at `min(32, cpu+4)`, and every `docker ps`/`rm` can block for
# tens of seconds on a loaded daemon. Dozens of episodes tearing down at once therefore
# saturated the pool and stalled the Harbor event loop itself, which looked like a hang.
# A dedicated small pool means cleanup can queue without ever blocking episode I/O.
CLEANUP_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="psrl-docker-cleanup",
)


async def cleanup_containers_by_label(
    label_key: str,
    label_value: str,
    stop_timeout: int = 10,
) -> list[str]:
    """
    Stop Docker containers matching a specific label.

    Uses ``docker ps -q --filter label={key}={value}`` to find containers,
    then ``docker stop -t {stop_timeout}`` to stop them. Idempotent: no-op
    if no containers match.

    Args:
        label_key: Docker label key to filter by.
        label_value: Docker label value to filter by.
        stop_timeout: Seconds to wait for graceful stop before Docker force-kills.

    Returns:
        List of container IDs that were stopped.
    """
    try:
        find_proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-q",
            "--filter",
            f"label={label_key}={label_value}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await find_proc.communicate()
        container_ids = [cid for cid in stdout.decode().strip().split() if cid]

        if not container_ids:
            psrl_logger.debug(f"No containers found with label {label_key}={label_value!r}.")
            return []

        psrl_logger.info(f"Stopping count={len(container_ids)} container(s) with label {label_key}={label_value!r}.")
        stop_proc = await asyncio.create_subprocess_exec(
            "docker",
            "stop",
            "-t",
            str(stop_timeout),
            *container_ids,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(stop_proc.communicate(), timeout=30.0)
        psrl_logger.info(f"Stopped count={len(container_ids)} container(s) with label {label_key}={label_value!r}.")
        return container_ids

    except asyncio.TimeoutError:
        psrl_logger.warning(f"Timeout stopping containers with label {label_key}={label_value!r}.")
        return []
    except Exception as e:
        psrl_logger.warning(f"Failed to cleanup containers with label {label_key}={label_value!r}: {e}.")
        return []


def force_remove_containers_by_label(
    label_key: str,
    label_value: str,
) -> list[str]:
    """
    Force-remove Docker containers matching a specific label.

    Args:
        label_key: Docker label key to filter by.
        label_value: Docker label value to filter by.

    Returns:
        List of container IDs that were force-removed.
    """
    try:
        find_out = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={label_key}={label_value}"],
            capture_output=True,
            timeout=30,
        )
        container_ids = [cid for cid in find_out.stdout.decode().strip().split() if cid]

        if not container_ids:
            psrl_logger.debug(f"No containers found with label {label_key}={label_value!r}.")
            return []

        psrl_logger.info(
            f"Force-removing count={len(container_ids)} container(s) with label {label_key}={label_value!r}."
        )
        subprocess.run(
            ["docker", "rm", "-f", *container_ids],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
        )
        psrl_logger.info(
            f"Force-removed count={len(container_ids)} container(s) with label {label_key}={label_value!r}."
        )
        return container_ids

    except subprocess.TimeoutExpired:
        psrl_logger.warning(f"Timeout force-removing containers with label {label_key}={label_value!r}.")
        return []
    except Exception as e:
        psrl_logger.warning(f"Failed to force-remove containers with label {label_key}={label_value!r}: {e}.")
        return []


def sanitize_compose_project_name(name: str) -> str:
    """
    Render a name the way Docker Compose does when it derives a project name.

    Mirrors `harbor.environments.docker.docker._sanitize_docker_compose_project_name`.
    Harbor passes `--project-name <sanitized session_id>`, so Compose stamps that value
    onto every container of the episode as `com.docker.compose.project`. Reproducing the
    rule here is what lets a caller find those containers by label. It is copied rather
    than imported because it is private to Harbor.

    Args:
        name (str): Raw name, normally a Harbor session id.

    Returns:
        str: The sanitized project name.
    """
    name = name.lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


def force_remove_compose_project(session_id: str) -> list[str]:
    """
    Force-remove every container Compose created for one Harbor episode.

    Cancelling the Python coroutine that awaits a Harbor job does not stop the
    containers it started. A verifier that outlives its own `timeout_sec` therefore
    keeps running `sleep infinity`, holding CPU and memory, and the next verifier
    queues behind it until the rollout pipeline wedges. This reclaims one episode's
    containers as soon as that episode is done with them.

    Distinct from `force_remove_containers_by_label`, which is keyed on the
    actor-scoped `psrl.actor_id` label and so can only be used once the whole actor
    exits. This is keyed on the Compose project, which is per episode.

    Args:
        session_id (str): The Harbor session id used as the Compose project name.

    Returns:
        list[str]: Container IDs that were force-removed.
    """
    if not session_id:
        return []
    return force_remove_containers_by_label(
        "com.docker.compose.project",
        sanitize_compose_project_name(session_id),
    )


def force_remove_compose_images(session_id: str) -> int:
    """
    Remove the images Compose built for one Harbor episode.

    Harbor's own teardown handles this correctly via
    `docker compose down --rmi local`, but only when that teardown runs. When the
    episode is cancelled or times out instead, the images survive **tagged**, so a
    dangling-image sweep never sees them. They are per episode rather than per task,
    because Harbor derives the Compose project from a fresh `session_id` every time
    and Compose names each built image `<project>-<service>`. Nothing ever reuses that
    tag, so every abandoned episode leaves one image behind forever.

    Measured leftovers before this existed: 193 to 547 tagged `*__env-main` and
    `*__verifier__trial-main` images per node, some six days old.

    Args:
        session_id (str): The Harbor session id used as the Compose project name.

    Returns:
        int: Number of images removed.
    """
    if not session_id:
        return 0

    project = sanitize_compose_project_name(session_id)
    try:
        # Compose derives image names as `<project>-<service>`, so the project prefix
        # selects exactly this episode's images and nothing shared.
        listed = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", f"reference={project}-*"],
            capture_output=True,
            timeout=120,
        )
        names = [n for n in listed.stdout.decode(errors="replace").split() if n]
        if not names:
            return 0
        subprocess.run(
            ["docker", "rmi", "-f", *names],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        psrl_logger.info("Removed %d image(s) for episode %s.", len(names), project)
        return len(names)
    except subprocess.TimeoutExpired:
        psrl_logger.warning(f"Timeout removing images for episode {project}.")
        return 0
    except Exception as exc:
        psrl_logger.warning(f"Failed to remove images for episode {project}: {exc}.")
        return 0


def prune_dangling_images(min_interval_secs: float = 900.0, timeout_secs: float = 600.0) -> bool:
    """
    Remove dangling Docker images, at most once per `min_interval_secs`.

    Each episode builds two images (agent environment and verifier). Harbor reuses the
    same tag per task, so each rebuild moves the tag to the new image and leaves the old
    one untagged, which is what `<none>` means. They accumulate for as long as training
    runs and cost disk, and they hold storage-driver state that on `fuse-overlayfs` is a
    live userspace process per mount.

    Uses `docker rmi -f` on explicit IDs in batches rather than `docker image prune -f`.
    Measured on a degraded node, `prune -f` ran for 25 minutes and removed **zero** of
    5816 dangling images with an empty log, while batched `rmi -f` cleared all of them in
    under 3 minutes. Batching also bounds the argument list.

    Deliberately throttled and never per episode. Listing the image store is itself the
    expensive operation being defended against, and images are shared between concurrent
    episodes rather than owned by one.

    Only untagged images are touched, so task images stay warm and the next episode does
    not pay a cold rebuild.

    Args:
        min_interval_secs (float): Minimum wall-clock gap between prunes.
        timeout_secs (float): Upper bound on the whole removal loop.

    Returns:
        bool: Whether a prune actually ran on this call.
    """
    global _LAST_PRUNE_MONOTONIC

    # Non-blocking: if another episode is already pruning, this one just moves on.
    if not _PRUNE_LOCK.acquire(blocking=False):
        return False
    try:
        now = time.monotonic()
        if _LAST_PRUNE_MONOTONIC and now - _LAST_PRUNE_MONOTONIC < min_interval_secs:
            return False
        _LAST_PRUNE_MONOTONIC = now

        deadline = now + timeout_secs
        removed = 0
        # An image still referenced by a live container cannot be deleted, only untagged.
        # Such an id keeps reappearing in the listing, so tracking what has already been
        # attempted is what stops this loop spinning on it until the deadline.
        attempted: set[str] = set()
        while time.monotonic() < deadline:
            listed = subprocess.run(
                ["docker", "images", "-f", "dangling=true", "-q"],
                capture_output=True,
                timeout=120,
            )
            ids = [i for i in listed.stdout.decode(errors="replace").split() if i and i not in attempted]
            if not ids:
                break
            batch = ids[:_PRUNE_BATCH_SIZE]
            attempted.update(batch)
            subprocess.run(
                ["docker", "rmi", "-f", *batch],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
            removed += len(batch)
        psrl_logger.info("Removed %d untagged Docker image(s).", removed)
        return True
    except subprocess.TimeoutExpired:
        # A removal slow enough to time out is itself evidence of a degraded daemon.
        psrl_logger.warning(f"Timeout removing untagged images after {timeout_secs}s.")
        return False
    except Exception as exc:
        psrl_logger.warning(f"Failed to prune dangling images: {exc}.")
        return False
    finally:
        _PRUNE_LOCK.release()


def spawn_actor_reaper(
    actor_id: str,
    log_dir: str | None = None,
    poll_interval: int = _REAPER_POLL_INTERVAL_SECS,
) -> subprocess.Popen:
    """
    Spawn the per-actor bash reaper sidecar.

    The sidecar polls the parent PID and removes matching containers after parent exit.

    Args:
        actor_id: Stable identifier for the spawning actor. Must match the
            ``psrl.actor_id`` label stamped on every container the actor spawns.
        log_dir: If given, append the reaper's stdout/stderr to
            ``<log_dir>/reaper_<actor_id>.log`` for post-mortem debugging.
            If None, output is discarded.
        poll_interval: Seconds between ``kill -0`` liveness checks. The reaper
            reaps within roughly this many seconds of parent death.

    Returns:
        The :class:`subprocess.Popen` handle. Callers should retain it (e.g.
        on ``self``) so it is not garbage-collected, and should call
        ``terminate()`` on graceful shutdown via ``atexit`` to skip a
        redundant post-mortem sweep.
    """
    parent_pid = os.getpid()
    label = f"psrl.actor_id={actor_id}"
    # NOTE(reaper): Python interpolates actor values at spawn time. Shell variables
    # such as `$$`, `$(date ...)`, and `$ids` remain for runtime expansion.
    script = f"""
set -u
echo "[reaper start] pid=$$ parent_pid={parent_pid} actor_id={actor_id} ts=$(date -Is)"
while kill -0 {parent_pid} 2>/dev/null; do
    sleep {poll_interval}
done
echo "[reaper] parent {parent_pid} gone at $(date -Is); reaping label={label}"
attempt=0
max_attempts=3
while [ $attempt -lt $max_attempts ]; do
    attempt=$((attempt + 1))
    ids=$(docker ps -aq --filter "label={label}" 2>/dev/null)
    if [ -z "$ids" ]; then
        if [ $attempt -eq 1 ]; then
            echo "[reaper] no containers to reap"
        else
            echo "[reaper] all containers reaped after $((attempt - 1)) pass(es)"
        fi
        break
    fi
    n=$(echo "$ids" | wc -l)
    echo "[reaper] pass $attempt/$max_attempts: force-removing $n container(s)"
    fail=0
    for cid in $ids; do
        out=$(docker rm -f "$cid" 2>&1)
        rc=$?
        if [ $rc -ne 0 ]; then
            fail=$((fail + 1))
            echo "[reaper] rm -f $cid FAILED rc=$rc: $out"
        fi
    done
    if [ $fail -eq 0 ]; then
        echo "[reaper] pass $attempt/$max_attempts: all $n removed"
        break
    fi
    echo "[reaper] pass $attempt/$max_attempts: $fail failure(s); will retry"
    sleep 2
done
remaining=$(docker ps -aq --filter "label={label}" 2>/dev/null)
if [ -n "$remaining" ]; then
    rn=$(echo "$remaining" | wc -l)
    echo "[reaper] STILL ALIVE after $max_attempts pass(es): $rn container(s):"
    echo "$remaining" | sed 's/^/[reaper]   /'
else
    echo "[reaper] done at $(date -Is)"
fi
"""
    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_fd: int | object = open(os.path.join(log_dir, f"reaper_{actor_id}.log"), "ab")
        except OSError as e:
            psrl_logger.warning(
                f"Could not open reaper log file under {log_dir!r}: {e}. Reaper output will be discarded."
            )
            log_fd = subprocess.DEVNULL
    else:
        log_fd = subprocess.DEVNULL

    return subprocess.Popen(
        ["nohup", "setsid", "bash", "-c", script],
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
