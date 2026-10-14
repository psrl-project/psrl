import copy
import logging
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _is_index_key(key: Any) -> bool:
    return str(key).isdigit()


def as_object_sample(value: Any) -> np.ndarray:
    """Wrap a Python value as a length-1 object array without unpacking sequences."""
    sample = np.empty((1,), dtype=object)
    sample[0] = value
    return sample


def _unwrap_reward_model_dicts(raw: Any) -> Any:
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            return _unwrap_reward_model_dicts(raw.item())
        raw = raw.tolist()
    if isinstance(raw, (list, tuple)) and len(raw) == 1 and not isinstance(raw[0], dict):
        return _unwrap_reward_model_dicts(raw[0])
    return raw


def _spec_dict(item: Any) -> dict:
    if isinstance(item, np.ndarray):
        item = item.item() if item.ndim == 0 else item.tolist()
    if isinstance(item, (DictConfig, ListConfig)):
        item = OmegaConf.to_container(item, resolve=True)
    if not isinstance(item, dict):
        raise TypeError(f"Each reward_model_dicts entry must be a dict, got {type(item)!r}.")
    return item


def normalize_reward_model_dicts(raw: Any) -> list[dict]:
    """Coerce `reward_model_dicts` into a list of spec dicts.

    Hydra index keys and NumPy object arrays unpack into the wrong container.
    """
    raw = _unwrap_reward_model_dicts(raw)
    if raw is None:
        return []
    if isinstance(raw, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, dict):
        if not raw:
            return []
        if all(_is_index_key(key) for key in raw):
            ordered = [raw[key] for key in sorted(raw.keys(), key=lambda key: int(str(key)))]
            return [_spec_dict(item) for item in ordered]
        return [_spec_dict(raw)]
    if isinstance(raw, (list, tuple)):
        return [_spec_dict(item) for item in raw]
    raise TypeError(f"reward_model_dicts must be a list of dicts, got {type(raw)!r}.")


def _has_files(value: Any) -> bool:
    """Return True when a `train_files` / `val_files` override is actually set."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, ListConfig)):
        return len(value) > 0
    return True


def _as_spec_list(raw: Any, name: str) -> list[DictConfig]:
    """Coerce `train_datas` / `val_datas` into a list of spec configs."""
    if raw is None:
        return []
    if isinstance(raw, (list, ListConfig)):
        items = list(raw)
    elif isinstance(raw, (dict, DictConfig)) and raw and all(_is_index_key(key) for key in raw):
        items = [raw[key] for key in sorted(raw.keys(), key=lambda key: int(str(key)))]
    else:
        raise TypeError(f"{name} must be a list of dataset specs, got {type(raw)!r}.")
    specs: list[DictConfig] = []
    for item in items:
        if isinstance(item, DictConfig):
            specs.append(item)
        else:
            specs.append(OmegaConf.create(item))
    return specs


def resolve_split_specs(data_cfg: DictConfig, split: str) -> list[DictConfig]:
    """Build the dataset spec list for one split.

    `train_files` / `val_files` is a single-dataset shorthand and wins over
    `train_datas` / `val_datas` so a CLI file override can replace the default mix.
    Either form becomes a list of specs. One dataset is a list of length one.
    """
    if split not in ("train", "val"):
        raise ValueError(f"split must be train or val, got {split!r}.")

    files = data_cfg.get(f"{split}_files")
    if _has_files(files):
        spec = OmegaConf.create({})
        with open_dict(spec):
            spec.file = files
            spec.max_samples = data_cfg.get(f"{split}_max_samples", -1)
        return [spec]

    specs = _as_spec_list(data_cfg.get(f"{split}_datas"), f"{split}_datas")
    if not specs:
        raise ValueError(f"data.{split}_datas or data.{split}_files must be set to load the {split} split.")
    for spec in specs:
        if not _has_files(spec.get("file")):
            raise ValueError(f"Each {split}_datas entry must set file, got {spec!r}.")
    return specs


def resolve_train_ratios(data_cfg: DictConfig, n_datasets: int) -> list[float]:
    """Return per-dataset sampling ratios that match `n_datasets`.

    A single dataset always takes the full batch. Multiple datasets use
    `train_datasets_ratios` when set, otherwise an equal split.
    """
    if n_datasets <= 0:
        raise ValueError(f"Training requires at least one dataset, got {n_datasets!r}.")
    if n_datasets == 1:
        return [1.0]
    raw_ratios = data_cfg.get("train_datasets_ratios")
    if not raw_ratios:
        equal = [1.0 / n_datasets] * (n_datasets - 1)
        equal.append(1.0 - sum(equal))
        return equal
    ratios = [float(ratio) for ratio in raw_ratios]
    if len(ratios) != n_datasets:
        raise ValueError(f"train_datasets_ratios has {len(ratios)} entries but there are {n_datasets} train datasets.")
    return ratios


def merge_dataset_config(data_cfg: DictConfig, spec: DictConfig) -> DictConfig:
    """Copy global data knobs and overlay one split spec. Spec keys win."""
    merged = copy.deepcopy(data_cfg)
    with open_dict(merged):
        merged.merge_with(spec)
    return merged


# AGENT(VERL): copy-in from verl. It should align with verl's implementation.
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """

    from verl.utils.dataset.rl_dataset import get_dataset_class

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        dataset_cls = get_dataset_class(data_config)
    else:
        from psrl.utils.dataset.rl_dataset import PSRLRLHFDataset

        dataset_cls = PSRLRLHFDataset
        print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )

    return dataset


# AGENT(VERL): copy-in from verl. It should align with verl's implementation.
def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import SequentialSampler

    # torch.utils.data.RandomSampler could not recover properly
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


def create_multi_rl_datasets(
    data_configs,
    tokenizer,
    processor,
):
    """Create multiple datasets for training and validation.

    Arguments:
        data_configs: List of data configs.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.
    Returns:
        datasets (list[Dataset]): A list of datasets for training and validation.
    """
    datasets = []
    for data_config in data_configs:
        dataset = create_rl_dataset(
            data_paths=data_config.file,
            data_config=data_config,
            tokenizer=tokenizer,
            processor=processor,
            max_samples=data_config.get("max_samples", -1),
        )
        datasets.append(dataset)
    return datasets


def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids
