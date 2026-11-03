# Open Problem Pool data audit

This file records what the source data looks like and what the projection does to it.
The numbers come from the combined snapshot with SHA-256
`6b491fa0dcbd510bccd9c60eec121015614f6a63dfb31028e987cefd4ac17381`. The projection
manifest records the same hash, so a different snapshot invalidates these counts.

## Source shapes

| Input | Records | Field set |
|---|---:|---|
| `combined/datasets_1016.jsonl` | 8451 | The 13 fields documented in `schema.json` |
| The 7 per-source JSONL files together | 8492 | 6 fields: `uid`, `problem`, `solution`, `msc_tag`, `references`, `extra` |

The per-source files and `schema.json` disagree. The combined file is the one the
projection uses by default.

## Findings

- **No solutions.** All 8451 combined records and all 8492 per-source records have an
  empty `solution`. A rule-based correctness reward is impossible. The recipe grades the
  final proof with the external verifier.
- **Null category.** All 8451 combined records have `"category": null`, which violates
  `schema.json` (`type: string`, `minLength: 1`). The projection derives a category from
  `extra.category`, `extra.direction`, or `extra.original_field`, then falls back to
  `Mathematics`.
- **Trailing newlines.** 25346 label values carry a trailing newline. The schema allows
  one trailing newline, and the projection strips it.
- **Duplicates.** The combined file has one duplicate problem statement. The per-source
  files have 50 duplicate problem statements across the seven files.
- **Labels.** Difficulty is 1/2/3 = 369/472/7610. Status is unresolved/unknown/resolved
  = 6349/1779/323. Publicity is 1 through 5.
- **Prompt length.** Combined problem statements are p50 702, p90 1591, p99 3058, and
  max 7531 tokens under the Qwen3.8-27B tokenizer.

## Projection result

Input: `datasets/omnibus_stable/omnibus_stable_1022.jsonl`.
Filter: difficulty 1 and 2, every status. Status is metadata rather than a filter,
because the verifier is the reward authority.

| Quantity | Value |
|---|---:|
| Input records | 118 |
| Kept | 102 (79 low, 23 medium) |
| Dropped for difficulty | 16 (15 high, 1 invalid label) |
| Train rows | 81 |
| Validation rows | 21 |
| Statuses kept | resolved 24, unknown 34, unresolved 44 |
| Prompt tokens p50 / p90 / p99 / max | 678 / 1090 / 1578 / 1644 |

The split is stratified by difficulty with a fixed seed, so the validation slice keeps
both levels. The manifest records the split and the input hash.

## Reward source

The recipe grades the final proof with the external verifier. Verdicts 7 and 6 map to
1.0 and 0.8, verdicts 1 and 0 map to 0.3 and 0.0. A transport, service, or payload
failure is ungraded. An ungraded episode is masked from the loss and excluded from the
group baseline.

## Reproduce

```bash
export OPP_DATASET_ROOT=<checkout of the Open Problem Pool>

python -m examples.math_rl.audit_opp \
    --input "$OPP_DATASET_ROOT/datasets/omnibus_stable/omnibus_stable_1022.jsonl"

python -m examples.math_rl.project_dataset
```

The audit is read-only. The projection writes `train.parquet`, `val.parquet`, and
`manifest.json` under `examples/math_rl/data`.
