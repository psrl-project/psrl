#!/bin/bash
# set -v

# Manage configured Docker daemons through parallel SSH.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=docker_common.sh
. "$SCRIPT_DIR/docker_common.sh"

if [ -z "$DOCKER_NODE_IPS" ]; then
    echo "[docker_manager.sh] Required env var DOCKER_NODE_IPS not set."
    echo "Usage:"
    echo "  DOCKER_NODE_IPS=ip1:8,ip2:8 [DOCKER_NODE_NUM=N] ./docker_manager.sh ACTION [HOST]"
    echo "  Actions: start | stop | restart | status | logs"
    exit 1
fi

ACTION=${1:-status}
HOST=${2:-all}

docker_cluster_init || exit 1
MANAGER=$DOCKER_CLUSTER_MANAGER
WORKERS=("${DOCKER_CLUSTER_WORKERS[@]}")

manage_docker() {
  local host=$1
  local action=$2

  case $action in
    start)
      echo "Launching Docker on $host..."
      pssh -H "$host" -i "nohup dockerd > /var/log/docker.log 2>&1 &"
      ;;
    stop)
      echo "Stopping Docker on $host..."
      pssh -H "$host" -i "pkill dockerd"
      ;;
    restart)
      echo "Restarting Docker on $host..."
      pssh -H "$host" -i "(pkill dockerd; nohup dockerd > /var/log/docker.log 2>&1 &)"
      ;;
    status)
      echo "Checking Docker status on $host..."
      if pssh -H "$host" -i "docker version >/dev/null 2>&1"; then
        echo "✓ $host: Docker is running"
      else
        echo "✗ $host: Docker is not running"
      fi
      ;;
    logs)
      echo "Viewing Docker logs on $host..."
      pssh -H "$host" -i "tail -f /var/log/docker.log"
      ;;
    *)
      echo "[docker_manager.sh] Unknown action: $action"
      echo "Usage: DOCKER_NODE_IPS=ip1:8,ip2:8 ./docker_manager.sh {start|stop|restart|status|logs} [host]"
      ;;
  esac
}

if [ "$HOST" = "all" ]; then
  for node in "$MANAGER" "${WORKERS[@]}"; do
    manage_docker "$node" "$ACTION"
  done
else
  manage_docker "$HOST" "$ACTION"
fi
