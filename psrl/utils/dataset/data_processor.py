import copy
import logging
import os
import sys
import threading
from dataclasses import dataclass

import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import open_dict
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn

from psrl.utils.dataset.utils import create_multi_rl_datasets, create_rl_dataset, create_rl_sampler
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@dataclass
class DatasetType:
    """DataType for the dataset."""

    unknown: str = "unknown"
    train: str = "train"
    val: str = "val"
    test: str = "test"


# NOTE(lhy): Keep `ray.remote` here to avoid NCCL hangs and intermittent crashes
# during vLLM generation.
@ray.remote
class DataProcessor:
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        ps_manager_handle,
    ):
        """
        Initialize the DataProcessor, responsible for processing data batches.
        Note that this processor runs in a separate Ray worker on a single CPU.

        Args:
            config: Configuration object containing data processing parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            processor: Optional data processor, used for multimodal data
            ps_manager_handle: Parameter Server handle for communication with the PS worker.
            collate_fn: Function to collate data samples into batches.
        """

        self.config = config

        # TransferQueue bootstrap.
        tq.init()

        # Dataset and dataloader attributes
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader_iters = None
        self.val_dataloader_iters = None
        if self.config.data.get("use_multi_dataset", True):
            self.train_datasets_ratios = self.config.data.train_datasets_ratios
        else:
            self.train_datasets_ratios = [1.0]
        self._validate_data_config()
        self._val_dataloader_idx = 0  # Index for round-robin selection of validation dataloaders

        self.global_steps = 0
        self._train_sample_idx = 0
        self._val_sample_idx = 0

        # Communication handles
        self.ps_manager_handle = ps_manager_handle
        self.reward_manager_handle = None  # Will be set by the ray trainer

        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n={self.rollout_n} must be greater than or equal to alg_rollout_n={self.alg_rollout_n}."
        )
        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n

        # Reserve disjoint train and validation ID ranges without overflowing `sys.maxsize`.
        self.MAX_TRAIN_ID = sys.maxsize // (2 * self.rollout_n)
        self.MAX_VAL_ID = sys.maxsize // (2 * self.val_rollout_n)

        # Threads for data processing
        self.data_process_thread = None
        self.stop_data_process = False
        self.dataloader_lock = threading.Lock()

        # Preserve partial batches so retries do not consume extra dataloader entries.
        self.retry_buffer: dict | None = None

        # Build logger
        self.log_prefix = "DataProcessor"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

        # Create the initial datasets and dataloaders
        self.total_training_steps = None
        self._create_dataloader()

    def _validate_data_config(self) -> None:
        """Validate the data configuration for consistency.

        In multi-dataset mode (use_multi_dataset=True), train_datas and
        train_datasets_ratios must both be set.

        In legacy mode (use_multi_dataset=False), train_files must be set.
        If train_datas is also set, a warning is emitted (it will be ignored).
        """
        if self.config.data.get("use_multi_dataset", True):
            if not self.config.data.get("train_datas"):
                raise ValueError("use_multi_dataset=True requires 'train_datas' to be set in data config.")
            if not self.config.data.get("train_datasets_ratios"):
                raise ValueError("use_multi_dataset=True requires 'train_datasets_ratios' to be set in data config.")
        else:
            if not self.config.data.get("train_files"):
                raise ValueError("use_multi_dataset=False requires 'train_files' to be set in data config.")
            if self.config.data.get("train_datas"):
                psrl_logger.warning(
                    "use_multi_dataset=False but 'train_datas' is set in data config. It will be ignored."
                )

    def set_reward_manager(self, reward_manager_handle: ray.actor.ActorHandle):
        self.reward_manager_handle = reward_manager_handle

    def set_agent_loop_manager(self, agent_loop_manager_handle: ray.actor.ActorHandle):
        self.agent_loop_manager_handle = agent_loop_manager_handle

    # ------- Dataset and Dataloader Building Methods -------
    def with_rollout_tool_paths(self, data_config):
        """Attach rollout tool paths to a data config copy for prompt filtering."""
        cfg = copy.deepcopy(data_config)
        multi_turn = self.config.gen_actor_rollout_ref.rollout.multi_turn
        with open_dict(cfg):
            cfg.tool_config_path = multi_turn.get("tool_config_path", None)
            cfg.function_tool_path = multi_turn.get("function_tool_path", None)
        return cfg

    def build_train_and_val_dataset(self) -> None:
        """Build the training and validation datasets.

        Behavior depends on ``config.data.use_multi_dataset``:

        * ``True`` (default): uses the PSRL multi-dataset path. Each entry in
          ``config.data.train_datas`` / ``config.data.val_datas`` is turned into
          a separate ``RLHFDataset``. Oversampling ensures that all datasets
          contribute the same number of batches per epoch.

        * ``False``: falls back to the veRL-native single-dataset path.
          ``config.data.train_files`` / ``config.data.val_files`` are used. No
          oversampling is performed, and ``train_datasets_ratios`` is forced to
          ``[1.0]`` so that the rest of the dataloader pipeline sees a
          single-element list transparently.
        """
        if self.config.data.get("use_multi_dataset", True):
            self.train_datasets = create_multi_rl_datasets(
                [self.with_rollout_tool_paths(data_config) for data_config in self.config.data.train_datas],
                self.tokenizer,
                self.processor,
            )

            dataset_sizes = [len(dataset) for dataset in self.train_datasets]
            rough_batch_sizes = [
                int(self.config.data.train_batch_size * ratio) for ratio in self.train_datasets_ratios
            ]
            rough_num_batches = [
                dataset_size // rough_batch_size
                for dataset_size, rough_batch_size in zip(dataset_sizes, rough_batch_sizes)
            ]
            max_num_batches = max(rough_num_batches) if rough_num_batches else 0
            oversample_ratios = [
                max_num_batches / rough_num_batches[i] if rough_num_batches[i] > 0 else 1.0
                for i in range(len(self.train_datasets))
            ]
            for dataset, oversample_ratio in zip(self.train_datasets, oversample_ratios):
                dataset.over_sample_dataset(oversample_ratio)

            self.val_datasets = create_multi_rl_datasets(
                [self.with_rollout_tool_paths(data_config) for data_config in self.config.data.val_datas],
                self.tokenizer,
                self.processor,
            )
        else:
            # Keep downstream dataloader operations list-based.
            self.train_datasets = [
                create_rl_dataset(
                    data_paths=self.config.data.train_files,
                    data_config=self.with_rollout_tool_paths(self.config.data),
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    max_samples=self.config.data.get("train_max_samples", -1),
                )
            ]
            self.val_datasets = [
                create_rl_dataset(
                    data_paths=self.config.data.val_files,
                    data_config=self.with_rollout_tool_paths(self.config.data),
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    max_samples=self.config.data.get("val_max_samples", -1),
                )
            ]
            self.train_datasets_ratios = [1.0]

    def build_train_sampler(self) -> None:
        """Build the training sampler."""
        assert self.train_datasets is not None, (
            "Train datasets are not built yet. Call build_train_and_val_dataset() first."
        )

        self.train_samplers = [create_rl_sampler(self.config.data, dataset) for dataset in self.train_datasets]

    def build_train_dataloader(self) -> None:
        """Build the training dataloader.

        This method creates a StatefulDataLoader for the training dataset.
        Note that the batch size is determined by `gen_batch_size` in the configuration,
        use `train_batch_size` as fallback.
        """
        assert self.train_datasets is not None, (
            "Train datasets are not built yet. Call build_train_and_val_dataset() first."
        )
        assert self.train_samplers is not None, "Train samplers are not built yet. Call build_train_sampler() first."
        assert len(self.train_datasets) == len(self.train_samplers), (
            "The number of train datasets and train samplers must be the same."
        )
        assert len(self.train_datasets_ratios) == len(self.train_datasets), (
            "The number of train datasets and train datasets ratios must be the same."
        )

        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            total_batch_size = self.config.psrl.rollout_coordination.redundant_rollout.redundant_global_batch_size
        else:
            total_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)

        batch_sizes = [int(total_batch_size * ratio) for ratio in self.train_datasets_ratios]
        batch_sizes[-1] = total_batch_size - sum(batch_sizes[:-1])
        psrl_logger.info(f"---Train Datasets--- batch sizes: {batch_sizes}, total batch size: {total_batch_size}")

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloaders = [
            StatefulDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                drop_last=True,
                collate_fn=collate_fn,
                sampler=sampler,
            )
            for dataset, batch_size, sampler in zip(self.train_datasets, batch_sizes, self.train_samplers)
        ]

        self.train_dataloader_sizes = [len(dataloader) for dataloader in self.train_dataloaders]
        psrl_logger.info(f"Size of train dataloader: {self.train_dataloader_sizes}")

    def build_val_dataloader(self) -> None:
        """Build the validation dataloader.

        This method creates a StatefulDataLoader for the validation dataset.
        The batch size is determined by `val_batch_size` in the configuration,
        use the length of the validation dataset as fallback.
        """
        assert self.val_datasets is not None, (
            "Validation datasets are not built yet. Call build_train_and_val_dataset() first."
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_sizes = [len(dataset) for dataset in self.val_datasets]
        else:
            # Use the same batch size for all validation datasets
            val_batch_sizes = [val_batch_size] * len(self.val_datasets)
        psrl_logger.info(f"---Val Datasets--- batch sizes: {val_batch_sizes}")

        num_workers = self.config.data["dataloader_num_workers"]

        self.val_dataloaders = [
            StatefulDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )
            for dataset, batch_size in zip(self.val_datasets, val_batch_sizes)
        ]

        self.val_dataloader_sizes = [len(dataloader) for dataloader in self.val_dataloaders]
        psrl_logger.info(f"Size of validation dataloaders: {self.val_dataloader_sizes}")

    def _create_dataloader(self):
        """Create the train and validation dataloaders.

        This method initializes the train and validation datasets, samplers, and dataloaders.
        It also calculates the total number of training steps
        based on the length of the train dataloader and the total epochs.
        If `total_training_steps` is specified in the configuration, it overrides the calculated value.
        """

        # AGENT(VERL): this logic is mapped to `_create_dataloader(...)` in `verl/trainer/ppo/ray_trainer.py`

        self.build_train_and_val_dataset()
        self.build_train_sampler()
        self.build_train_dataloader()
        self.build_val_dataloader()

        total_training_steps = min(self.train_dataloader_sizes) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = min(total_training_steps, self.config.trainer.total_training_steps)
        self.total_training_steps = total_training_steps

    def get_val_data_size(self):
        """Get the size of the validation dataset."""
        assert self.val_dataloaders is not None, (
            "Validation dataloaders are not built yet. Call build_val_dataloader() first."
        )
        return sum([len(dataloader.dataset) for dataloader in self.val_dataloaders])

    def get_val_batch_num(self):
        """Get the number of batches in the validation dataloader."""
        assert self.val_dataloaders is not None, (
            "Validation dataloaders are not built yet. Call build_val_dataloader() first."
        )
        return sum([len(dataloader) for dataloader in self.val_dataloaders])

    def get_total_training_steps(self):
        """Get the total number of training steps."""
        assert self.total_training_steps is not None, (
            "Total training steps are not set. Call _create_dataloader() first."
        )

        return self.total_training_steps

    # ------- Dataloader Management Methods -------
    def save_train_dataloader(self, dataloader_local_paths: list[str] | str) -> None:
        """Save the dataloader to a local path for future resume."""
        assert self.train_dataloaders is not None, (
            "Train dataloader is not built yet. Call build_train_dataloader() first."
        )
        if not isinstance(dataloader_local_paths, list):
            dataloader_local_paths = [dataloader_local_paths] * len(self.train_dataloaders)
        assert len(dataloader_local_paths) == len(self.train_dataloaders), (
            "The number of dataloader paths must match the number of train dataloaders."
        )

        # AGENT(VERL): this logic is mapped to `_save_checkpoint(...)` in `verl/trainer/ppo/ray_trainer.py`
        # you can find it by searching for `# save dataloader` in that file.

        for i, dataloader in enumerate(self.train_dataloaders):
            os.makedirs(os.path.dirname(dataloader_local_paths[i]), exist_ok=True)
            torch.save(dataloader.state_dict(), dataloader_local_paths[i])
        psrl_logger.info(f"Train dataloader saved to {dataloader_local_paths}")

    def load_train_dataloader(self, dataloader_local_paths: list[str] | str) -> None:
        """Load the dataloader from a local path."""
        assert self.train_dataloaders is not None, (
            "Train dataloader is not built yet. Call build_train_dataloader() first."
        )
        if not isinstance(dataloader_local_paths, list):
            dataloader_local_paths = [dataloader_local_paths] * len(self.train_dataloaders)
        assert len(dataloader_local_paths) == len(self.train_dataloaders), (
            "The number of dataloader paths must match the number of train dataloaders."
        )

        # AGENT(VERL): this logic is mapped to `_load_checkpoint(...)` in `verl/trainer/ppo/ray_trainer.py`
        # you can find it by searching for `# load dataloader` in that file
        steps_per_epoch = min(self.train_dataloader_sizes)
        at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
        if at_epoch_boundary:
            psrl_logger.info(
                f"Skipping dataloader state restore: global_steps={self.global_steps} "
                f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                f"The saved state marks the dataloader as exhausted. "
                f"Next epoch will iterate from scratch."
            )
        else:
            for dataloader, dataloader_local_path in zip(self.train_dataloaders, dataloader_local_paths):
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                dataloader.load_state_dict(dataloader_state_dict)
            psrl_logger.info(f"Train dataloader loaded from {dataloader_local_paths}")

    # ------- Data Retrieval Methods -------
    def get_train_next(self) -> dict:
        """Get the next merged batch from all training dataloaders.

        Fetches one batch from each training dataloader, concatenates them with
        segment-level shuffling (preserving per-dataset row order), and returns
        the combined dict.

        Lazy-initializes the dataloader iterators on first call.

        Returns:
            dict: A merged batch dict from all training datasets.

        Raises:
            StopIteration: When any training dataloader is exhausted (epoch boundary).
                All iterators are reset before re-raising so the next call starts
                a fresh epoch.
        """
        with self.dataloader_lock:
            return self._get_train_next()

    def _get_train_next(self) -> dict:
        """Body of `get_train_next` without acquiring `dataloader_lock`."""
        if self.train_dataloader_iters is None:
            self.train_dataloader_iters = [iter(dl) for dl in self.train_dataloaders]

        try:
            batch_dicts = [next(it) for it in self.train_dataloader_iters]
        except StopIteration:
            # Reset all iterators so the caller can start the next epoch cleanly.
            self.train_dataloader_iters = [iter(dl) for dl in self.train_dataloaders]
            raise

        return self._concat_and_shuffle_batch_dicts(batch_dicts)

    def get_val_next(self):
        """Get the next batch of validation data.

        This method handles the case where the validation dataloader iterator is exhausted.

        It will raise a StopIteration exception if the validation dataloader is exhausted,
        which can be caught by the main controller to handle the end of validation data.

        Returns:
            The next batch of validation data.

        Raises:
            StopIteration: If the validation dataloader iterator is exhausted.
        """
        if self.val_dataloader_iters is None:
            self.val_dataloader_iters = [iter(dataloader) for dataloader in self.val_dataloaders]
            self._val_dataloader_idx = 0

        # Sequentially get data from each dataloader, starting from the first one
        # Continue with the current dataloader until it's exhausted, then move to the next
        num_dataloaders = len(self.val_dataloader_iters)

        while self._val_dataloader_idx < num_dataloaders:
            i = self._val_dataloader_idx
            dataloader_iter = self.val_dataloader_iters[i]
            try:
                data = next(dataloader_iter)
                return data
            except StopIteration:
                psrl_logger.info(f"Validation dataloader index={i} exhausted.")
                # Reset the exhausted iterator and move to next dataloader
                self.val_dataloader_iters[i] = iter(self.val_dataloaders[i])
                self._val_dataloader_idx = (self._val_dataloader_idx + 1) % num_dataloaders
                continue

    def get_val_sample_ids(self, batch_size: int) -> list:
        """
        Generate sample IDs for validation data with cyclic wrapping.

        This method generates unique sample IDs for validation data that are guaranteed
        not to conflict with training sample IDs by using a separate ID namespace.
        The IDs cycle within [MAX_TRAIN_ID, MAX_TRAIN_ID + MAX_VAL_ID) to avoid overflow.

        Args:
            batch_size (int): The number of sample IDs to generate.

        Returns:
            list: A list of sample IDs for the validation batch.
        """
        sample_ids = []
        for _ in range(batch_size):
            # Cycle within the validation namespace: [MAX_TRAIN_ID, MAX_TRAIN_ID + MAX_VAL_ID)
            val_offset = self._val_sample_idx % self.MAX_VAL_ID
            sample_id = self.MAX_TRAIN_ID + val_offset
            sample_ids.append(sample_id)
            self._val_sample_idx += 1
        return sample_ids

    def get_train_sample_ids(self, batch_size: int) -> list:
        """
        Generate sample IDs for training data with cyclic wrapping.

        This method generates unique sample IDs for training data that cycle within
        [0, MAX_TRAIN_ID) to avoid overflow. The IDs are used to track samples across
        the training pipeline and must not conflict with validation sample IDs.

        Args:
            batch_size (int): The number of sample IDs to generate.

        Returns:
            list: A list of sample IDs for the training batch.
        """
        with self.dataloader_lock:
            sample_ids = []
            for i in range(batch_size):
                sample_id = (self._train_sample_idx + i) % self.MAX_TRAIN_ID
                sample_ids.append(sample_id)
            self._train_sample_idx = (self._train_sample_idx + batch_size) % self.MAX_TRAIN_ID
        return sample_ids

    def get_single_controller_batch(
        self,
        dataset_type: DatasetType,
        group_repeat: bool = True,
        return_meta: bool = True,
    ) -> TensorDict | KVBatchMeta:
        """
        Get a single batch from the dataset for single controller training.

        Args:
            dataset_type (DatasetType): The type of dataset to get the batch from (train, val, test).
            group_repeat (bool): Whether to repeat the batch for each group.
            return_meta (bool): Whether to return KVBatchMeta and put the batch into TransferQueue.

        Returns:
            dict | KVBatchMeta: A single batch dict from the dataset.

        NOTE(linsh): If group_repeat is True, the batch will be repeated `rollout_n` times for each group
        and put into the TransferQueue, and a KVBatchMeta will be returned for the repeated batch.

        Raises:
            StopIteration: If there are no more batches in the dataset.
        """
        if dataset_type == DatasetType.train:
            batch_dict = self.get_train_next()
            rollout_n = self.rollout_n
        elif dataset_type == DatasetType.val:
            batch_dict = self.get_val_next()
            rollout_n = self.val_rollout_n
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        batch_size = len(batch_dict[list(batch_dict.keys())[0]])
        if dataset_type == DatasetType.train:
            sample_ids = self.get_train_sample_ids(batch_size)
        else:
            sample_ids = self.get_val_sample_ids(batch_size)

        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_stack(
            batch,
            "global_steps",
            [self.global_steps] * batch_size,
        )
        tu.assign_non_tensor_stack(
            batch,
            "validate",
            [dataset_type == DatasetType.val] * batch_size,
        )
        tu.assign_non_tensor_stack(
            batch,
            "parent_id" if rollout_n > 1 else "uid",
            sample_ids,
        )
        if group_repeat and rollout_n > 1:
            request_ids: list[int] = []
            for i in range(batch_size):
                for j in range(rollout_n):
                    request_ids.append(sample_ids[i] * rollout_n + j)
            batch = batch.repeat_interleave(repeats=rollout_n)
            tu.assign_non_tensor_stack(batch, "uid", request_ids)
            keys = [str(uid) for uid in request_ids]
            tags = [{"uid": request_id, "status": "running"} for request_id in request_ids]
        else:
            keys = [str(uid) for uid in sample_ids]
            tags = [{"uid": sample_id} for sample_id in sample_ids]

        if rollout_n > 1:
            for i in range(batch_size):
                for j in range(rollout_n):
                    tags[i * rollout_n + j]["parent_id"] = sample_ids[i]

        if not return_meta:
            return batch

        batch_meta = tq.kv_batch_put(
            keys=keys,
            partition_id="train" if dataset_type == DatasetType.train else "val",
            fields=batch,
            tags=tags,
        )

        return batch_meta

    def sample_train_prompts(self, n_prompts: int) -> TensorDict | None:
        """Sample exactly `n_prompts` fresh training prompts on demand.

        Args:
            n_prompts (int): Number of fresh prompts to sample. Must be > 0.

        Returns:
            TensorDict | None:
                Ready-to-dispatch TensorDict, or `None` if the dataloader
                yielded no data even after starting a fresh epoch.
        """
        assert n_prompts > 0, f"n_prompts must be positive, got {n_prompts!r}."
        rollout_n = self.rollout_n

        chunks: list[dict] = []
        remaining = n_prompts
        # `_get_train_next` re-raises `StopIteration` only after rebuilding its iterators, so a
        # second consecutive raise means the dataloader is exhausted and looping would spin.
        rollover_used = False
        with self.dataloader_lock:
            while remaining > 0:
                if self.retry_buffer is None or self._retry_buffer_size() == 0:
                    try:
                        self.retry_buffer = self._get_train_next()
                        rollover_used = False
                    except StopIteration:
                        if rollover_used:
                            psrl_logger.warning(
                                f"sample_train_prompts: dataloader yields no data across an epoch restart, "
                                f"sampled={n_prompts - remaining}, requested={n_prompts}."
                            )
                            break
                        rollover_used = True
                        psrl_logger.info(
                            f"sample_train_prompts: epoch boundary reached with sampled={n_prompts - remaining}, "
                            f"requested={n_prompts}. Starting the next epoch."
                        )
                        continue

                leftover_size = self._retry_buffer_size()
                take = min(remaining, leftover_size)
                chunks.append({k: v[:take] for k, v in self.retry_buffer.items()})
                if take < leftover_size:
                    self.retry_buffer = {k: v[take:] for k, v in self.retry_buffer.items()}
                else:
                    self.retry_buffer = None
                remaining -= take

        if not chunks:
            return None

        actual_n_prompts = n_prompts - remaining

        batch_dict = chunks[0] if len(chunks) == 1 else self._concat_dict_chunks(chunks)
        sample_ids = self.get_train_sample_ids(actual_n_prompts)

        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_stack(batch, "global_steps", [self.global_steps] * actual_n_prompts)
        tu.assign_non_tensor_stack(batch, "validate", [False] * actual_n_prompts)
        tu.assign_non_tensor_stack(
            batch,
            "parent_id" if rollout_n > 1 else "uid",
            sample_ids,
        )

        if rollout_n > 1:
            request_ids = [sample_ids[i] * rollout_n + j for i in range(actual_n_prompts) for j in range(rollout_n)]
            batch = batch.repeat_interleave(repeats=rollout_n)
            tu.assign_non_tensor_stack(batch, "uid", request_ids)
        else:
            request_ids = sample_ids

        ray.get(self.ps_manager_handle.add_request.remote(request_ids))

        psrl_logger.debug(
            f"sample_train_prompts: produced prompts={actual_n_prompts}, child requests={len(request_ids)} for retry."
        )
        return batch

    def _retry_buffer_size(self) -> int:
        """Return the number of prompt rows currently held in `retry_buffer`."""
        if self.retry_buffer is None:
            return 0
        first_key = next(iter(self.retry_buffer))
        return len(self.retry_buffer[first_key])

    @staticmethod
    def _concat_dict_chunks(chunks: list[dict]) -> dict:
        """Concatenate a list of dataloader-shaped dicts along dim 0."""
        if len(chunks) == 1:
            return chunks[0]
        merged: dict = {}
        for key in chunks[0].keys():
            values = [chunk[key] for chunk in chunks]
            sample = values[0]
            if isinstance(sample, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(sample, np.ndarray):
                merged[key] = np.concatenate(values, axis=0)
            else:
                raise TypeError(f"Unsupported value type {type(sample)!r} for key {key!r} in retry chunks.")
        return merged

    # ------- Streaming Data Processing Methods -------
    def start_busy_loop(self):
        """Initialize the thread for data processing."""
        if self.data_process_thread is not None:
            return

        self.data_process_thread = threading.Thread(
            target=self._process_data,
            name="data_process_thread",
            daemon=True,
        )
        self.data_process_thread.start()

    def stop_busy_loop(self):
        """Stop the data processing thread."""
        if self.data_process_thread is not None:
            self.stop_data_process = True
            self.data_process_thread.join()
            self.data_process_thread = None

    @staticmethod
    def _concat_and_shuffle_batch_dicts(batch_dicts: list[dict]) -> dict:
        """Concatenate batch dicts and shuffle at the dataset-segment level.

        Each dataset's rows are concatenated in a random segment order, but the
        relative row order *within* each dataset's segment is preserved.

        Args:
            batch_dicts: One dict per dataset, each mapping feature names to
                torch.Tensor or np.ndarray of shape (batch_size, *dims).

        Returns:
            A single dict with all features concatenated along dim 0, with
            segments arranged in a random order.

        Raises:
            ValueError: If batch_dicts is empty or any individual dict is empty.
            TypeError: If a value is neither torch.Tensor nor np.ndarray.
        """
        if not batch_dicts:
            raise ValueError("batch_dicts must not be empty.")

        num_datasets = len(batch_dicts)

        # Validate and collect per-dataset sizes
        for batch_dict in batch_dicts:
            if not batch_dict:
                raise ValueError("Batch dict must not be empty.")

        # Shuffle segment order (dataset-level permutation)
        segment_order = torch.randperm(num_datasets).tolist() if num_datasets > 1 else [0]
        shuffled_dicts = [batch_dicts[i] for i in segment_order]

        # Collect the union of all keys, using first occurrence as the type sample
        key_sample: dict[str, torch.Tensor | np.ndarray] = {}
        key_order: list[str] = []
        batch_sizes: list[int] = []

        for batch_dict in shuffled_dicts:
            first_value = next(iter(batch_dict.values()))
            batch_sizes.append(len(first_value))
            for key, value in batch_dict.items():
                if key not in key_sample:
                    key_sample[key] = value
                    key_order.append(key)

        # Concatenate each key across shuffled segments, filling missing keys with zeros/None
        concatenated: dict = {}
        for key in key_order:
            sample_value = key_sample[key]
            values = [
                batch_dict[key]
                if key in batch_dict
                else DataProcessor._create_placeholder_like(sample_value, batch_size)
                for batch_dict, batch_size in zip(shuffled_dicts, batch_sizes)
            ]
            if isinstance(sample_value, torch.Tensor):
                concatenated[key] = torch.cat(values, dim=0)
            elif isinstance(sample_value, np.ndarray):
                concatenated[key] = np.concatenate(values, axis=0)
            else:
                raise TypeError(f"Unsupported value type {type(sample_value)} for key '{key}' in batch concatenation.")

        return concatenated

    @staticmethod
    def _create_placeholder_like(sample_value: torch.Tensor | np.ndarray, batch_size: int):
        """Create a placeholder tensor/array matching the sample value's dtype and trailing shape."""
        if isinstance(sample_value, torch.Tensor):
            shape = (batch_size, *sample_value.shape[1:])
            return torch.zeros(shape, dtype=sample_value.dtype)
        if isinstance(sample_value, np.ndarray):
            shape = (batch_size, *sample_value.shape[1:])
            if sample_value.dtype == object:
                placeholder = np.empty(shape, dtype=object)
                placeholder[:] = None
                return placeholder
            return np.zeros(shape, dtype=sample_value.dtype)
        raise TypeError(f"Unsupported value type {type(sample_value)} for placeholder creation.")

    def _process_data(self):
        """
        Process the training data in a busy loop.

        This method continuously fetches batches from the training dataloader,
        processes them, and puts them into the data queue for further processing.

        The method will run until all epochs are processed or a StopIteration is raised.
        After processing, it signals the end of data processing by putting None into the data queue.

        Note: This method is designed to run in a separate thread and is intended to be called
        after initializing the DataProcessor and its dataloaders.

        The data queue will hold the processed batches, which can be consumed by the rollout server.
        """
        self.train_dataloader_iters = [iter(dataloader) for dataloader in self.train_dataloaders]
        total_epochs = self.config.trainer.total_epochs

        # loop until all epochs are processed
        while not self.stop_data_process:
            try:
                batch_meta: dict = self.get_single_controller_batch(DatasetType.train, return_meta=False)
                batch = batch_meta
                batch_size = len(batch) // self.rollout_n

                # Register the complete batch before one dispatch to preserve batching efficiency.
                psrl_logger.debug(f"Generating prompts={batch_size}, rollout_n={self.rollout_n} children.")
                if isinstance(batch, KVBatchMeta):
                    all_request_ids = [tag["uid"] for tag in batch.tags]
                else:
                    all_request_ids = tu.get(batch, "uid")
                ray.get(self.ps_manager_handle.add_request.remote(all_request_ids))
                ray.get(self.agent_loop_manager_handle.put_data.remote(batch))

                self.global_steps += 1
                if self.total_training_steps is not None and self.global_steps >= self.total_training_steps:
                    self.stop_data_process = True

            except StopIteration:
                curr_epoch = self.global_steps // min(self.train_dataloader_sizes)
                psrl_logger.debug(f"StopIteration encountered, current epoch: {curr_epoch}/{total_epochs}")
                if curr_epoch >= total_epochs:
                    psrl_logger.info("All training epochs completed, stopping data processing.")
                    self.stop_data_process = True
                else:
                    psrl_logger.debug("Reinitializing train_dataloader_iters for next epoch")
                    self.train_dataloader_iters = [iter(dl) for dl in self.train_dataloaders]
            except Exception as e:
                psrl_logger.error(f"Exception in data processing thread: {e}", exc_info=True)

        psrl_logger.info("Data processing stopped, sending shutdown signal.")
        self.agent_loop_manager_handle.put_data.remote(None)
