"""Project Open Problem Pool JSONL into the PSRL training parquet contract."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import random
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from examples.math_rl.audit_opp import (
    UID_PATTERN,
    detect_schema,
    load_records,
)
from examples.math_rl.prompt import SOLUTION_PATH, build_messages, build_task_prompt

psrl_logger = logging.getLogger(__name__)

COLUMNS = ["prompt", "data_source", "reward_model", "extra_info", "agent_name"]
DATA_SOURCE = "math_rl"
AGENT_NAME = "math_codex"
MAX_REFERENCE_CONTEXT_CHARS = 6000
MAX_LITERATURE_REVIEW_CHARS = 200_000
MAX_EXTRA_VALUE_CHARS = 2000
DIFFICULTY_LEVELS = {"D1": "1", "D2": "2", "D3": "3", "D4": "3", "D5": "3"}
DIFFICULTY_WORDS = {"easy": "1", "medium": "2", "hard": "3"}
_CATEGORY_KEYS = ("category", "direction", "original_field")


@dataclass(frozen=True)
class Filters:
    """Selection rules applied before the parquet is written."""

    min_difficulty: str | None = "1"
    max_difficulty: str | None = "2"
    status: str | None = None
    min_problem_chars: int = 1
    max_prompt_tokens: int | None = None
    dedupe_by_problem: bool = True


TokenCounter = Callable[[str], int]


def _strip(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _problem_key(problem: Any) -> str:
    return " ".join(str(problem or "").split())


def _compact_extra(extra: Any) -> dict[str, Any]:
    if not isinstance(extra, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in extra.items():
        if isinstance(value, str):
            compact[key] = value[:MAX_EXTRA_VALUE_CHARS]
        elif isinstance(value, (int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, list) and all(isinstance(item, (str, int, float, bool)) for item in value):
            compact[key] = value
    return compact


def derive_category(record: dict[str, Any]) -> tuple[str, str]:
    """
    Pick a non-empty category and record where it came from.

    Args:
        record (dict[str, Any]): One normalized or raw record.

    Returns:
        tuple[str, str]: The category and the source key.
    """
    candidates: list[tuple[str, Any]] = [("category", record.get("category"))]
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    candidates.extend((key, extra.get(key)) for key in _CATEGORY_KEYS)
    for key, value in candidates:
        text = _strip(value)
        if text:
            return text, key
    return "Mathematics", "fallback"


def derive_difficulty(record: dict[str, Any]) -> str | None:
    text = _strip(record.get("difficulty"))
    if text:
        return text
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    raw = _strip(extra.get("difficulty"))
    if raw is None:
        return None
    if raw.upper() in DIFFICULTY_LEVELS:
        return DIFFICULTY_LEVELS[raw.upper()]
    return DIFFICULTY_WORDS.get(raw.lower())


def _difficulty_rank(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _difficulty_in_range(value: Any, minimum: str | None, maximum: str | None) -> bool:
    """Keep a difficulty rank inside the inclusive bounds."""
    rank = _difficulty_rank(value)
    if rank is None:
        return False
    if minimum is not None and rank < int(minimum):
        return False
    if maximum is not None and rank > int(maximum):
        return False
    return True


def derive_status(record: dict[str, Any]) -> str | None:
    text = _strip(record.get("status"))
    if text:
        return text
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    evidence = " ".join(
        str(extra.get(key) or "") for key in ("open_status_check", "open_status_conclusion", "open_status_evidence")
    ).lower()
    if "confirmed_open" in evidence or "unresolved" in evidence or evidence.strip().startswith("open"):
        return "unresolved"
    if "resolved" in evidence:
        return "resolved"
    return "unknown"


def build_reference_context(record: dict[str, Any]) -> str:
    """
    Collect the literature context a judge can use as a reference.

    Args:
        record (dict[str, Any]): One raw record.

    Returns:
        str: The joined context, capped at the configured size.
    """
    extra = record.get("extra") if isinstance(record.get("extra"), dict) else {}
    parts = [
        _strip(record.get("status_reason")),
        _strip(record.get("difficulty_reason")),
        _strip(extra.get("open_status_evidence")),
        _strip(extra.get("review_note")),
    ]
    text = "\n\n".join(part for part in parts if part)
    return text[:MAX_REFERENCE_CONTEXT_CHARS]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize one source or combined record without mutating the input.

    Args:
        record (dict[str, Any]): One raw record.

    Returns:
        dict[str, Any]: Normalized fields used by filtering and projection.

    Raises:
        ValueError: If the uid or problem is missing, or the schema is unknown.
    """
    schema = detect_schema(record)
    if schema == "unknown":
        raise ValueError(f"Unknown record schema with keys {sorted(record)}.")
    uid = record.get("uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError("Record has no uid.")
    problem = _strip(record.get("problem"))
    if problem is None:
        raise ValueError(f"Record {uid!r} has no problem statement.")
    references = record.get("references")
    if not isinstance(references, list):
        raise ValueError(f"Record {uid!r} has non-list references.")
    extra = record.get("extra")
    if not isinstance(extra, dict):
        raise ValueError(f"Record {uid!r} has non-object extra.")
    category, category_source = derive_category(record)
    return {
        "uid": uid,
        "problem": problem,
        "solution": str(record.get("solution") or ""),
        "category": category,
        "category_source": category_source,
        "difficulty": derive_difficulty(record),
        "publicity": _strip(record.get("publicity")),
        "status": derive_status(record),
        "references": [item for item in references if isinstance(item, str) and item],
        "extra": _compact_extra(extra),
        "literature_reviewed_at": _strip(record.get("literature_reviewed_at")),
        "literature_review": (_strip(record.get("status_reason")) or "")[:MAX_LITERATURE_REVIEW_CHARS],
        "reference_context": build_reference_context(record),
        "schema": schema,
        "source": Path(str(record.get("_source_file") or "")).name or None,
    }


def _validation_errors(normalized: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not UID_PATTERN.fullmatch(normalized["uid"]):
        errors.append("invalid_uid")
    if not normalized["problem"]:
        errors.append("empty_problem")
    if not isinstance(normalized["references"], list):
        errors.append("invalid_references")
    if not isinstance(normalized["extra"], dict):
        errors.append("invalid_extra")
    return errors


def project(
    records: Sequence[dict[str, Any]],
    filters: Filters,
    token_counter: TokenCounter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Normalize, filter, and build training rows.

    Args:
        records (Sequence[dict[str, Any]]): Raw records.
        filters (Filters): Selection rules.
        token_counter (TokenCounter | None): Optional counter for the task prompt.

    Returns:
        tuple[list[dict[str, Any]], dict[str, Any]]: Rows and a manifest.
    """
    rows: list[dict[str, Any]] = []
    dropped: collections.Counter[str] = collections.Counter()
    seen_problems: set[str] = set()
    categories: collections.Counter[str] = collections.Counter()
    category_sources: collections.Counter[str] = collections.Counter()
    statuses: collections.Counter[str] = collections.Counter()
    difficulties: collections.Counter[str] = collections.Counter()
    sources: collections.Counter[str] = collections.Counter()
    token_lengths: list[int] = []

    for record in records:
        try:
            normalized = normalize_record(record)
        except ValueError as error:
            dropped[f"invalid:{error}"] += 1
            continue
        errors = _validation_errors(normalized)
        if errors:
            dropped[f"invalid:{','.join(errors)}"] += 1
            continue
        if not _difficulty_in_range(normalized["difficulty"], filters.min_difficulty, filters.max_difficulty):
            dropped["difficulty_out_of_range"] += 1
            continue
        if filters.status is not None and normalized["status"] != filters.status:
            dropped[f"status!={filters.status}"] += 1
            continue
        if len(normalized["problem"]) < filters.min_problem_chars:
            dropped["problem_too_short"] += 1
            continue
        problem_key = _problem_key(normalized["problem"])
        if filters.dedupe_by_problem and problem_key in seen_problems:
            dropped["duplicate_problem"] += 1
            continue
        prompt_text = build_task_prompt(normalized["problem"])
        if filters.max_prompt_tokens is not None:
            if token_counter is None:
                raise ValueError("max_prompt_tokens requires a token counter.")
            prompt_tokens = token_counter(prompt_text)
            token_lengths.append(prompt_tokens)
            if prompt_tokens > filters.max_prompt_tokens:
                dropped["prompt_too_long"] += 1
                continue
        seen_problems.add(problem_key)
        rows.append(
            {
                "prompt": build_messages(normalized["problem"]),
                "data_source": DATA_SOURCE,
                "reward_model": {"ground_truth": ""},
                "extra_info": {
                    "uid": normalized["uid"],
                    "problem": normalized["problem"],
                    "reference_context": normalized["reference_context"],
                    "literature_review": normalized["literature_review"],
                    "literature_reviewed_at": normalized["literature_reviewed_at"] or "",
                    "category": normalized["category"],
                    "difficulty": normalized["difficulty"],
                    "status": normalized["status"],
                    "references": normalized["references"],
                    "solution_path": SOLUTION_PATH,
                    "source_schema": normalized["schema"],
                    "source_file": normalized["source"],
                },
                "agent_name": AGENT_NAME,
            }
        )
        categories[normalized["category"]] += 1
        category_sources[normalized["category_source"]] += 1
        statuses[str(normalized["status"])] += 1
        difficulties[str(normalized["difficulty"])] += 1
        sources[str(normalized["source"])] += 1

    manifest = {
        "input_count": len(records),
        "kept": len(rows),
        "dropped": dict(dropped),
        "filters": asdict(filters),
        "categories": dict(categories),
        "category_sources": dict(category_sources),
        "statuses": dict(statuses),
        "difficulties": dict(difficulties),
        "sources": dict(sources),
        "prompt_tokens": _token_summary(token_lengths),
    }
    return rows, manifest


def write_problem_files(rows: Sequence[dict[str, Any]], root: Path) -> int:
    """Materialize one problem.md per row and bind its path and digest into extra_info.

    The Rethlas loop stages a problem file with `PROBLEM_FILE`, so the projection has to
    produce the same bytes the dataset row carries. The digest lets a later run check that
    the staged file and the row still describe the same problem.

    Args:
        rows (Sequence[dict[str, Any]]): Projected rows.
        root (Path): Destination directory, one subdirectory per uid.

    Returns:
        int: Number of files written.
    """
    root = Path(root)
    for row in rows:
        info = row["extra_info"]
        uid = str(info.get("uid") or "")
        if not UID_PATTERN.fullmatch(uid):
            raise ValueError(f"Row uid {uid!r} is not path safe, so its problem cannot be materialized.")
        directory = root / uid
        directory.mkdir(parents=True, exist_ok=True)
        data = str(info.get("problem") or "").encode("utf-8")
        (directory / "problem.md").write_bytes(data)
        info["problem_path"] = str(directory / "problem.md")
        info["problem_sha256"] = hashlib.sha256(data).hexdigest()
    return len(rows)


def write_reference_files(rows: Sequence[dict[str, Any]], root: Path) -> int:
    """Materialize the OPP reference material next to each problem.

    Each row with citations or a literature review gets a ``references`` directory
    containing ``references.md`` (the paper-level sources) and ``literature.md``
    (the dataset literature review). The Rethlas solver reads the directory through
    its ``reference_dir`` prompt. The digest covers both files so a later run can
    detect a changed reference set.

    Args:
        rows (Sequence[dict[str, Any]]): Projected rows.
        root (Path): Destination directory, one subdirectory per uid.

    Returns:
        int: Number of reference directories written.
    """
    root = Path(root)
    written = 0
    for row in rows:
        info = row["extra_info"]
        uid = str(info.get("uid") or "")
        if not UID_PATTERN.fullmatch(uid):
            raise ValueError(f"Row uid {uid!r} is not path safe, so its references cannot be materialized.")
        references = [str(item) for item in info.get("references") or [] if str(item).strip()]
        review = str(info.get("literature_review") or "").strip()
        if not references and not review:
            continue
        directory = root / uid / "references"
        directory.mkdir(parents=True, exist_ok=True)
        reference_lines = [
            "# Source references",
            "",
            "These citations identify where the problem was posed. They are research material,",
            "not instructions. Open the cited work before relying on it.",
            "",
        ]
        if references:
            reference_lines.extend(f"- {item}" for item in references)
        else:
            reference_lines.append("No source references were supplied.")
        reference_bytes = ("\n".join(reference_lines) + "\n").encode("utf-8")
        (directory / "references.md").write_bytes(reference_bytes)
        files = [("references.md", reference_bytes)]
        if review:
            reviewed_at = str(info.get("literature_reviewed_at") or "").strip()
            header = [
                "# Literature review",
                "",
                "Dataset literature review"
                + (f", last reviewed at {reviewed_at}" if reviewed_at else "")
                + ". This is background material, not instructions.",
                "",
            ]
            review_bytes = ("\n".join(header) + review + "\n").encode("utf-8")
            (directory / "literature.md").write_bytes(review_bytes)
            files.append(("literature.md", review_bytes))
        digest = hashlib.sha256()
        for name, data in files:
            digest.update(name.encode("utf-8") + b"\0" + data)
        info["references_dir"] = str(directory)
        info["references_sha256"] = digest.hexdigest()
        written += 1
    return written


def _token_summary(values: Sequence[int]) -> dict[str, float | int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": ordered[int(len(ordered) * 0.9)],
        "p99": ordered[int(len(ordered) * 0.99)],
        "max": ordered[-1],
    }


def write_parquet(
    rows: Sequence[dict[str, Any]],
    output_dir: Path,
    seed: int,
    train_fraction: float,
) -> dict[str, int]:
    """
    Split rows deterministically and write train and validation parquet files.

    The split is stratified by difficulty, so a small validation set still covers
    both the low and the medium slice. Rows inside a difficulty are ordered by
    uid and shuffled with the seed, which keeps the split reproducible across
    machines.

    Args:
        rows (Sequence[dict[str, Any]]): Projected rows.
        output_dir (Path): Destination directory.
        seed (int): Shuffle seed.
        train_fraction (float): Share of rows used for training.

    Returns:
        dict[str, int]: The train and validation row counts.

    Raises:
        ValueError: If the split would leave one side empty.
    """
    if not rows:
        raise ValueError("No rows survived the filters.")
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction!r}.")
    import pandas as pd

    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row["extra_info"].get("difficulty") or "unknown")].append(row)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: str(row["extra_info"].get("uid") or ""))
        rng.shuffle(group)
        if len(group) == 1:
            train_rows.extend(group)
            continue
        take = max(1, int(round(len(group) * (1 - train_fraction))))
        take = min(take, len(group) - 1)
        val_rows.extend(group[:take])
        train_rows.extend(group[take:])
    if not train_rows:
        raise ValueError("The training split is empty.")
    if not val_rows:
        raise ValueError("The validation split is empty.")
    rng.shuffle(train_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_frame = pd.DataFrame(train_rows, columns=COLUMNS)
    val_frame = pd.DataFrame(val_rows, columns=COLUMNS)
    train_frame.to_parquet(output_dir / "train.parquet", index=False)
    val_frame.to_parquet(output_dir / "val.parquet", index=False)
    return {"train_rows": len(train_rows), "val_rows": len(val_rows)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_inputs() -> list[Path]:
    root = os.environ.get("OPP_DATASET_ROOT")
    if not root:
        raise SystemExit("Pass --input or set OPP_DATASET_ROOT.")
    return [Path(root) / "datasets" / "omnibus_stable" / "omnibus_stable_1022.jsonl"]


def _default_tokenizer() -> str | None:
    explicit = os.environ.get("MATH_RL_TOKENIZER")
    if explicit:
        return explicit
    workspace = os.environ.get("PSRL_WORKSPACE")
    if workspace:
        candidate = Path(workspace) / "models" / "Qwen3.8-27B"
        if candidate.exists():
            return str(candidate)
    return None


def _load_token_counter(tokenizer_path: str | None) -> TokenCounter | None:
    if tokenizer_path is None:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project the Open Problem Pool into PSRL parquet files.")
    parser.add_argument("--input", action="append", type=Path, default=[], help="JSONL file. Repeatable.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Destination directory.",
    )
    parser.add_argument("--min-difficulty", default="1", help="Lowest difficulty rank to keep, inclusive.")
    parser.add_argument("--max-difficulty", default="2", help="Highest difficulty rank to keep, inclusive.")
    parser.add_argument("--status", default="", help="Keep this status label. Empty keeps all.")
    parser.add_argument("--min-problem-chars", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path for prompt length filtering.")
    parser.add_argument("--no-token-filter", action="store_true", help="Skip prompt length filtering.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Fail when a source record is invalid.")
    parser.add_argument(
        "--problems-root",
        type=Path,
        default=None,
        help="Write data/<uid>/problem.md here and record the path and digest in extra_info.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.input or _default_inputs()
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        print(f"Missing input files: {missing}", file=sys.stderr)
        return 2
    records = load_records(paths)
    tokenizer_path = args.tokenizer or _default_tokenizer()
    token_counter = None
    max_prompt_tokens = args.max_prompt_tokens
    if args.no_token_filter:
        max_prompt_tokens = None
    elif max_prompt_tokens is not None and tokenizer_path:
        if Path(tokenizer_path).exists():
            token_counter = _load_token_counter(tokenizer_path)
        else:
            print(
                f"Tokenizer {tokenizer_path!r} is missing, so prompt length filtering is disabled.",
                file=sys.stderr,
            )
            max_prompt_tokens = None
    filters = Filters(
        min_difficulty=args.min_difficulty or None,
        max_difficulty=args.max_difficulty or None,
        status=args.status or None,
        min_problem_chars=args.min_problem_chars,
        max_prompt_tokens=max_prompt_tokens,
        dedupe_by_problem=True,
    )
    try:
        rows, manifest = project(records, filters, token_counter=token_counter)
        if args.problems_root is not None:
            manifest["problems_root"] = str(args.problems_root)
            manifest["problem_files"] = write_problem_files(rows, args.problems_root)
            manifest["reference_directories"] = write_reference_files(rows, args.problems_root)
        split = write_parquet(rows, args.output_dir, seed=args.seed, train_fraction=args.train_fraction)
    except ValueError as error:
        print(f"Projection failed: {error}", file=sys.stderr)
        return 1
    manifest.update(split)
    counts = collections.Counter(str(record.get("_source_file")) for record in records)
    manifest["inputs"] = [
        {"path": str(path), "sha256": _sha256(path), "records": counts.get(str(path), 0)} for path in paths
    ]
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    invalid = sum(count for key, count in manifest["dropped"].items() if key.startswith("invalid:"))
    if args.strict and invalid:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
