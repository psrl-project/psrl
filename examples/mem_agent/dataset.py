from psrl.utils.dataset.rl_dataset import PSRLRLHFDataset


class MemAgentDataset(PSRLRLHFDataset):
    """Accept native HotpotQA parquet and raw RULER-HQA JSON rows."""

    def _build_messages(self, example: dict, key: str) -> list[dict]:
        if key in example and example[key] is not None:
            return super()._build_messages(example, key)
        if "input" not in example:
            raise KeyError(f"Dataset row contains neither {key!r} nor 'input'.")
        return [{"role": "user", "content": str(example["input"])}]

    def __getitem__(self, item: int) -> dict:
        row = super().__getitem__(item)
        if "input" not in row:
            return row

        answers = row.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        row["reward_model"] = {
            "style": "rule",
            "ground_truth": list(answers),
        }
        row["data_source"] = "ruler_hqa"
        row["ability"] = "memory"
        row["index"] = item
        row["extra_info"] = {
            **(row.get("extra_info") or {}),
            "index": item,
            "num_docs": int(row.get("num_docs", 0) or 0),
        }
        return row
