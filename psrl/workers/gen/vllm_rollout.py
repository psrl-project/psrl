import ray
from verl.workers.rollout.vllm_rollout import ServerAdapter


class PSRL_ServerAdapter(ServerAdapter):
    def get_node_id(self) -> str:
        """Get the node id of the vllm worker."""
        if not hasattr(self, "node_id"):
            self.node_id = None

        if self.node_id is not None:
            return self.node_id
        self.node_id = ray.get_runtime_context().get_node_id()
        return self.node_id

    def get_runtime_gpu_ids(self) -> list[int]:
        accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
        raw_gpu_ids = accelerator_ids.get("GPU", accelerator_ids.get("NPU", []))
        return [int(gpu_id) for gpu_id in raw_gpu_ids]
