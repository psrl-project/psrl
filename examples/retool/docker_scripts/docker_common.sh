# Provide shared Docker cluster helpers for sibling scripts.
# `docker_cluster_init` populates the host array and manager and worker split.

docker_cluster_ips_to_lines() {
  printf '%s' "$DOCKER_NODE_IPS" | sed "s/:.//g; s/,/\\n/g"
}

# Populate Docker cluster globals and return success.
docker_cluster_init() {
  if [ -z "$DOCKER_NODE_IPS" ]; then
    echo "Error: DOCKER_NODE_IPS is not set"
    return 1
  fi
  if [ -z "$DOCKER_NODE_NUM" ]; then
    DOCKER_NODE_NUM=$(docker_cluster_ips_to_lines | wc -l)
  fi
  mapfile -t DOCKER_CLUSTER_HOSTS < <(docker_cluster_ips_to_lines | head -n "$DOCKER_NODE_NUM")
  if [ ${#DOCKER_CLUSTER_HOSTS[@]} -eq 0 ]; then
    echo "Error: DOCKER_NODE_IPS is empty or invalid"
    return 1
  fi
  DOCKER_CLUSTER_MANAGER="${DOCKER_CLUSTER_HOSTS[0]}"
  DOCKER_CLUSTER_WORKERS=("${DOCKER_CLUSTER_HOSTS[@]:1}")
  return 0
}

docker_cluster_hosts_csv() {
  (IFS=,; echo "${DOCKER_CLUSTER_HOSTS[*]}")
}
