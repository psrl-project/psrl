"""
Build the rideshare HuggingFace dataset from the raw TSF file.

This script replicates what the Monash TSF loading script does, bypassing
the pandas frequency alias issue (`H` removed in pandas 3.x).
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, "/apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data/hf_compat_env")

import numpy as np  # noqa: E402
from datasets import Dataset, DatasetDict  # noqa: E402, I001

# Load the convert_tsf_to_dataframe function from the HF-cached monash utils.py
CACHE_SNAP = Path(
    "/root/.cache/huggingface/hub/datasets--Monash-University--monash_tsf/snapshots"
    "/58aafbe2712ff481c014f562e42723f2820fd5d4"
)
spec = importlib.util.spec_from_file_location("utils_monash", CACHE_SNAP / "utils.py")
utils_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_mod)
convert_tsf_to_dataframe = utils_mod.convert_tsf_to_dataframe

TSF_FILE = Path(
    "/apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data/airs_raw"
    "/Monash-University/monash_tsf/rideshare_tmp"
    "/rideshare_dataset_with_missing_values.tsf"
)
OUTPUT_DIR = Path(
    "/apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data/airs_raw/Monash-University/monash_tsf/rideshare"
)

# rideshare config params from monash_tsf.py
ITEM_ID_COLUMNS = ["source_location", "provider_name", "provider_service"]
DATA_COLUMN = "type"
TARGET_FIELDS = [
    "price_min",
    "price_mean",
    "price_max",
    "distance_min",
    "distance_mean",
    "distance_max",
    "surge_min",
    "surge_mean",
    "surge_max",
    "api_calls",
    "temp",
    "rain",
    "humidity",
    "clouds",
    "wind",
]
PREDICTION_LENGTH = 48  # For hourly: was prediction_length_map["H"]
ROLLING_EVALUATIONS = 1


def build_examples(loaded_data, split):
    loaded_data = loaded_data.copy()
    loaded_data.set_index(ITEM_ID_COLUMNS, inplace=True)
    loaded_data.sort_index(inplace=True)

    examples = []
    for cat, item_id in enumerate(loaded_data.index.unique()):
        ts = loaded_data.loc[item_id]
        start = ts["start_timestamp"].iloc[0]

        target_fields = ts[ts[DATA_COLUMN].isin(TARGET_FIELDS)]
        target = np.vstack(target_fields["target"].tolist())

        if split in ["train", "validation"]:
            offset = PREDICTION_LENGTH * ROLLING_EVALUATIONS + PREDICTION_LENGTH * (split == "train")
            target = target[..., :-offset]

        examples.append(
            {
                "start": start,
                "target": target.tolist(),
                "feat_dynamic_real": [],
                "feat_static_cat": [cat],
                "item_id": str(item_id),
            }
        )
    return examples


def main():
    print(f"Loading TSF from {TSF_FILE}...")
    loaded_data, frequency, forecast_horizon, _, _ = convert_tsf_to_dataframe(
        str(TSF_FILE), value_column_name="target"
    )
    print(f"Loaded: frequency={frequency}, forecast_horizon={forecast_horizon}, shape={loaded_data.shape}")

    splits = {}
    for split_name in ("train", "validation", "test"):
        examples = build_examples(loaded_data, split_name)
        splits[split_name] = Dataset.from_list(examples)
        print(f"Split {split_name}: {len(examples)} examples")

    ds = DatasetDict(splits)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(OUTPUT_DIR))
    print(f"Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
