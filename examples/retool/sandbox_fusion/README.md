# SandBoxFusion Deployment Guide

This guide provides step-by-step instructions to deploy the SandBoxFusion service using Docker containers across multiple nodes.

Currently we deploy SandBoxFusion with Docker Swarm, which will deploy the service on all nodes. Please follow the steps below to set up and launch the service.

## Prerequisites

- Docker installed on all nodes.
- SSH access to all nodes.
- `SANDBOX_NODE_IPS` in environment variables: comma-separated `ip:gpu_count` pairs (e.g. `192.168.1.1:8,192.168.1.2:8`). The GPU count is stripped before use.
- `SANDBOX_NODE_NUM` (optional): number of nodes to use; defaults to the count of IPs in `SANDBOX_NODE_IPS`.
- Image name is hardcoded as `code_sandbox:server` in `launch_service.sh` — load that tag onto every node before launching (see [docker_scripts/README.md](../docker_scripts/README.md)).

## Deployment Steps

1. **Build the Docker Image**

    Ensure you have built the Docker image for the SandBoxFusion service. You can do this by following the guide in [SandBoxFusion](https://github.com/bytedance/SandboxFusion) to build the code sandbox server image, then tag it as `code_sandbox:server` and fan it out with `docker_scripts/docker_copy.sh`.

2. **Prepare the Launch Script**

    Use the provided `launch_service.sh` script to automate the deployment process. Set `SANDBOX_NODE_IPS` (and optionally `SANDBOX_NODE_NUM`) before running the script.

3. **Launch the Service**

    Run the `launch_service.sh` script on the master node:

    ```bash
    SANDBOX_NODE_IPS=ip1:8,ip2:8,...,ipN:8 \
    SANDBOX_NODE_NUM=N \
      bash launch_service.sh
    ```

    Or pass the same values as positional args: `bash launch_service.sh "$SANDBOX_NODE_IPS" "$SANDBOX_NODE_NUM"`.

    This script will:
    - Check and start docker service on all nodes.
    - Initialize docker swarm on the master node.
    - Join worker nodes to the swarm.
    - Deploy the SandBoxFusion service using the `code_sandbox:server` image.

4. **Validate the Deployment**

    After deployment, you can execute the following command in the shell to request the sandbox to run a Python code snippet in each node:

    ```bash
    curl 'http://localhost:8080/run_code' \
        -H 'Content-Type: application/json' \
        --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
    ```

    Sample output:

    ```json
    {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
            "status": "Finished",
            "execution_time": 0.016735315322875977,
            "return_code": 0,
            "stdout": "Hello, world!\n",
            "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
    }
    ```

5. **(Optional) Clean the Service**:

    If you need to remove the deployed service, you can run the following command on the master node, which will use pssh to stop and remove the service from all nodes:

    ```bash
    SANDBOX_NODE_IPS=ip1:8,ip2:8,...,ipN:8 bash clear_service.sh
    ```

## Notes

As a helper tool, you can use [`examples/retool/docker_scripts/docker_manager.sh`](../docker_scripts/docker_manager.sh) to manage Docker across multiple nodes. It provides functionalities such as starting/stopping/restarting the docker service, checking docker status, and viewing docker logs. Set `DOCKER_NODE_IPS` (same `ip:gpu_count` format) when invoking it.
