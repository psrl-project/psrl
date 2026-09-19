"""
Megatron training client Actor example
Focuses on core Megatron model loading functionality
"""

import os

import ray
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from verl.models.mcore import init_mcore_model
from verl.utils.device import get_device_name
from verl.utils.megatron_utils import get_model
from verl.utils.torch_dtypes import PrecisionType

# Import verl utilities
from verl.workers.megatron_workers import set_random_seed

QWEN_MODEL_PATH = os.environ.get("PSRL_WORKSPACE", "/tmp") + "/models/Qwen2.5-0.5B-Instruct"


def make_dual_print(log_path, prefix=None):
    with open(log_path, "w") as _:
        pass

    def dual_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        if prefix:
            msg = f"[{prefix}] {msg}"
        print(msg, **kwargs)
        with open(log_path, "a") as f:
            print(msg, file=f, **kwargs)

    return dual_print


@ray.remote(num_cpus=1)
class MegatronClient:
    def __init__(self, rank, world_size, model_path=QWEN_MODEL_PATH, log_dir=None):
        # Set environment variables
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        self.rank = rank
        self.world_size = world_size
        self.model_path = model_path

        log_path = os.path.join(log_dir, f"{self.rank}.log")
        self.print = make_dual_print(log_path, prefix=f"Rank {self.rank}")

        self.print(f"[Rank {rank}] Initializing MegatronClient begin")

        # Initialize distributed training
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        # torch.cuda.set_device(0)

        # Initialize Megatron parallel state
        self._init_megatron_parallel()

        # Set random seed
        set_random_seed(0)

        # Initialize model
        self._init_model()

        # Convert model
        self._covert_model()

        self.print(f"[Rank {rank}] Initialization completed")

    def _init_megatron_parallel(self):
        """Initialize Megatron parallel state"""
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            use_sharp=False,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            nccl_communicator_config_path=None,
        )
        self.print(f"[Rank {self.rank}] Megatron parallel initialized")

    def _init_model(self):
        """Initialize Megatron model"""
        from transformers import AutoConfig
        from verl.models.mcore import hf_to_mcore_config
        from verl.utils.fs import copy_to_local

        # Copy model to local
        local_path = copy_to_local(self.model_path)

        # Get HuggingFace config
        hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=False)

        # Convert to Megatron config
        dtype = PrecisionType.to_dtype(torch.bfloat16)
        tf_config = hf_to_mcore_config(hf_config, dtype)

        self.print(f"[Rank {self.rank}] Config loaded: {hf_config.model_type}")

        def model_provider(pre_process, post_process):
            """Model provider function"""
            model = init_mcore_model(
                tf_config,
                hf_config,
                pre_process,
                post_process,
                share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
                value=False,
            )
            model.to(get_device_name())
            return model

        # Initialize Megatron model
        self.model = get_model(
            model_provider,
            wrap_with_ddp=True,
            use_distributed_optimizer=True,
        )

        self.print(f"[Rank {self.rank}] Model initialized: {self.model}")

    def _covert_model(self):
        """Convert model"""
        from psrl.utils.converter import create_parameter_mapping
        from psrl.utils.converter.megatron_converter import convert_megatron_inplace
        from transformers import AutoConfig

        model_config = AutoConfig.from_pretrained(self.model_path)
        parameter_mapping = create_parameter_mapping("Megatron", model_config)
        unified_state_dict, sharding_dict = convert_megatron_inplace(parameter_mapping, self.model)
        self.print(f"[Rank {self.rank}] Model converted: {unified_state_dict}")
        self.print(f"[Rank {self.rank}] Sharding dict: {sharding_dict}")

    def shutdown(self):
        """Shutdown client"""
        self.print(f"[Rank {self.rank}] Shutting down")
        if dist.is_initialized():
            dist.destroy_process_group()
        return True


def test_megatron():
    """Test Megatron client"""
    log_dir = os.environ.get("PSRL_WORKSPACE") + "/psrl/tests/megatron/log"
    os.makedirs(log_dir, exist_ok=True)
    ray.init(ignore_reinit_error=True)

    num_workers = 8
    print(f"Testing MegatronClient with {num_workers} workers")

    # Create clients
    clients = [
        MegatronClient.options(num_gpus=1).remote(rank=rank, world_size=num_workers, log_dir=log_dir)
        for rank in range(num_workers)
    ]

    # Shutdown clients
    ray.get([client.shutdown.remote() for client in clients])
    ray.shutdown()

    print("Test completed successfully!")


if __name__ == "__main__":
    test_megatron()
