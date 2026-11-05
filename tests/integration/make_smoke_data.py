"""Write a two-prompt parquet dataset for the subagent trainer smoke."""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    rows = []
    for index in range(2):
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"Name one color, then one more. Task {index}."}],
                "data_source": "subagent_smoke",
                "reward_model": {"style": "rule", "ground_truth": "blue"},
                "extra_info": {"smoke_index": index},
            }
        )
    pd.DataFrame(rows).to_parquet(args.output, index=False)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
