# tests/dataset/test_resolve_dataset_specs.py
import pytest
from omegaconf import OmegaConf
from psrl.utils.dataset.utils import (
    merge_dataset_config,
    resolve_split_specs,
    resolve_train_ratios,
)

pytestmark = pytest.mark.cpu_test


def test_files_shorthand_is_one_spec_and_replaces_train_datas():
    cfg = OmegaConf.create(
        {
            "train_files": "/data/gsm8k/train.parquet",
            "train_max_samples": 128,
            "train_datas": [
                {"file": "/data/ignored.parquet", "prompt_key": "prompt"},
                {"file": "/data/also_ignored.parquet", "prompt_key": "prompt"},
            ],
            "train_datasets_ratios": [0.5, 0.5],
        }
    )
    specs = resolve_split_specs(cfg, "train")
    assert len(specs) == 1
    assert specs[0].file == "/data/gsm8k/train.parquet"
    assert specs[0].max_samples == 128
    assert resolve_train_ratios(cfg, n_datasets=len(specs)) == [1.0]


def test_train_datas_list_is_used_when_files_are_absent():
    cfg = OmegaConf.create(
        {
            "train_datas": [
                {"file": "/data/a.parquet", "prompt_key": "prompt"},
                {"file": "/data/b.parquet", "prompt_key": "question"},
            ],
            "train_datasets_ratios": [0.3, 0.7],
        }
    )
    specs = resolve_split_specs(cfg, "train")
    assert [spec.file for spec in specs] == ["/data/a.parquet", "/data/b.parquet"]
    assert resolve_train_ratios(cfg, n_datasets=len(specs)) == [0.3, 0.7]


def test_null_or_empty_files_fall_through_to_datas():
    cfg = OmegaConf.create(
        {
            "train_files": None,
            "val_files": [],
            "train_datas": [{"file": "/data/train.parquet"}],
            "val_datas": [{"file": "/data/val.parquet"}],
        }
    )
    assert resolve_split_specs(cfg, "train")[0].file == "/data/train.parquet"
    assert resolve_split_specs(cfg, "val")[0].file == "/data/val.parquet"


def test_missing_split_raises():
    cfg = OmegaConf.create({})
    with pytest.raises(ValueError, match="train_datas or data.train_files"):
        resolve_split_specs(cfg, "train")


def test_spec_without_file_raises():
    cfg = OmegaConf.create({"train_datas": [{"prompt_key": "prompt"}]})
    with pytest.raises(ValueError, match="must set file"):
        resolve_split_specs(cfg, "train")


def test_equal_ratios_when_missing():
    cfg = OmegaConf.create({})
    ratios = resolve_train_ratios(cfg, n_datasets=3)
    assert len(ratios) == 3
    assert sum(ratios) == pytest.approx(1.0)
    assert ratios[0] == ratios[1]


def test_ratio_length_mismatch_raises():
    cfg = OmegaConf.create({"train_datasets_ratios": [0.5, 0.5]})
    with pytest.raises(ValueError, match="2 entries but there are 3"):
        resolve_train_ratios(cfg, n_datasets=3)


def test_merge_dataset_config_inherits_global_knobs():
    data_cfg = OmegaConf.create(
        {
            "max_prompt_length": 2048,
            "truncation": "left",
            "prompt_key": "prompt",
            "train_datas": [{"file": "/data/a.parquet"}],
        }
    )
    spec = OmegaConf.create({"file": "/data/b.parquet", "prompt_key": "question"})
    merged = merge_dataset_config(data_cfg, spec)
    assert merged.file == "/data/b.parquet"
    assert merged.prompt_key == "question"
    assert merged.max_prompt_length == 2048
    assert merged.truncation == "left"
