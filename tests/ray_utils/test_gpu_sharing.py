"""GPU sharing test requiring at least one GPU and Ray with GPU resources.

This test allocates GPU memory across two Ray actors sharing the same GPU
and verifies memory allocation is reported correctly.
"""

import pytest
import ray


@pytest.mark.gpu
def test_gpu_sharing(ray_cluster):
    # Initialize the cluster before checking its GPU resources.
    if ray.cluster_resources().get("GPU", 0) < 1:
        pytest.skip("Requires Ray cluster with at least 1 GPU")

    import torch

    current_node_id = ray.get_runtime_context().get_node_id()

    @ray.remote(
        num_cpus=0,
        num_gpus=0.5,
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=current_node_id,
            soft=False,
        ),
    )
    class GPUWorker:
        def __init__(self, id):
            self.id = id
            self.size = 256 * 1024 * 1024  # 256MB

        def allocate_memory(self):
            device = torch.device("cuda")
            _ = torch.zeros(self.size // 4, dtype=torch.float32, device=device)
            allocated = torch.cuda.memory_allocated(device)
            return {
                "id": self.id,
                "allocated_mb": allocated / (1024 * 1024),
                "max_memory_mb": torch.cuda.get_device_properties(device).total_memory / (1024 * 1024),
            }

    actor1 = GPUWorker.remote(id=1)
    actor2 = GPUWorker.remote(id=2)

    result_ref1 = actor1.allocate_memory.remote()
    result_ref2 = actor2.allocate_memory.remote()
    results = ray.get([result_ref1, result_ref2])

    for res in results:
        assert res["allocated_mb"] > 0
        assert res["max_memory_mb"] > 0
