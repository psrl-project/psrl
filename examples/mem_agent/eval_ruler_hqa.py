import argparse
import asyncio
import copy
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import aiohttp
from examples.mem_agent.config import MemAgentRuntimeConfig
from examples.mem_agent.reward import extract_boxed_answer
from examples.mem_agent.runner import MemAgent
from transformers import AutoTokenizer


def _normalize_answer(value: str) -> str:
    value = "".join(character for character in value.lower() if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _metrics(prediction: str, answer: str) -> tuple[float, float, float]:
    """Match VIME's first-answer F1, exact-match and substring exact-match."""
    normalized_prediction = _normalize_answer(prediction)
    normalized_answer = _normalize_answer(answer)
    prediction_tokens = normalized_prediction.split()
    answer_tokens = normalized_answer.split()
    overlap = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
    if not prediction_tokens or not answer_tokens or overlap == 0:
        f1 = 0.0
    else:
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    exact_match = float(normalized_prediction == normalized_answer)
    sub_exact_match = float(normalized_answer in normalized_prediction or normalized_prediction in normalized_answer)
    return f1, exact_match, sub_exact_match


def load_data(data_root: str, length: int) -> list[dict]:
    candidates = [Path(data_root) / f"eval_{length}.json", Path(data_root) / f"eval_{length}.jsonl"]
    data_path = next((path for path in candidates if path.is_file()), None)
    if data_path is None:
        raise FileNotFoundError(f"Cannot find eval data for length={length}; tried: {candidates}")
    if data_path.suffix == ".jsonl":
        with data_path.open(encoding="utf-8") as file:
            raw = [json.loads(line) for line in file if line.strip()]
    else:
        with data_path.open(encoding="utf-8") as file:
            raw = json.load(file)
        if isinstance(raw, dict):
            raw = list(raw.values())

    data = []
    for index, original in enumerate(raw):
        if "input" in original:
            item = dict(original)
            item.setdefault("_id", index)
        elif "prompt" in original:
            metadata = original.get("metadata") or {}
            item = {
                "_id": index,
                "input": original["prompt"],
                "answers": metadata.get("ground_truth", [original.get("label", "")]),
                "context": metadata.get("context", ""),
                "num_docs": metadata.get("num_docs", 0),
            }
        else:
            print(f"[warn] skipping row {index}: unrecognized format")
            continue
        answers = item.get("answers", [])
        item["answers"] = [answers] if isinstance(answers, str) else list(answers)
        data.append(item)
    return data


def _aggregate(records: list[dict]) -> dict[str, float | int]:
    count = len(records)
    return {
        "f1": sum(record["judge_f1"] for record in records) / count if count else 0.0,
        "em": sum(record["judge_em"] for record in records) / count if count else 0.0,
        "sub_em": sum(record["judge_sub_em"] for record in records) / count if count else 0.0,
        "total": count,
    }


def _load_cache(path: Path) -> tuple[list[dict], set]:
    records = []
    cached_ids = set()
    if not path.is_file():
        return records, cached_ids
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
                records.append(record)
                cached_ids.add(record["_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return records, cached_ids


def _repeat_samples(data: list[dict], sampling: int) -> list[dict]:
    if sampling == 1:
        return data
    repeated = []
    for sample_index in range(sampling):
        for item in data:
            sample = copy.deepcopy(item)
            sample["_id"] = item["_id"] * sampling + sample_index
            repeated.append(sample)
    return repeated


async def evaluate(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    data = _repeat_samples(load_data(args.data_root, args.length), args.sampling)
    output = Path(args.save_dir) / f"{args.save_file}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    cached_records, cached_ids = ([], set()) if args.force else _load_cache(output)
    todo = [item for item in data if item["_id"] not in cached_ids]
    print(f"Data: {len(data)}  Cached: {len(cached_ids)}  Todo: {len(todo)}  Concurrency: {args.n_proc}")

    config = MemAgentRuntimeConfig(
        chunk_tokens=int(os.getenv("MEM_CHUNK_TOKENS", "2048")),
        max_memory_tokens=int(os.getenv("MEM_MAX_MEMORY", "1024")),
        max_final_tokens=int(os.getenv("MEM_MAX_FINAL", "256")),
        max_chunks=int(os.getenv("MEM_MAX_CHUNKS", "64")),
        allow_context_truncation=True,
    )
    semaphore = asyncio.Semaphore(args.n_proc)
    completed = []
    errors = 0

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=86400)) as session:
        agent = MemAgent(
            tokenizer=tokenizer,
            base_url=args.base_url,
            model=args.model,
            config=config,
            api_key=args.api_key,
            http_session=session,
        )

        async def process(item: dict) -> dict | None:
            try:
                async with semaphore:
                    result = await agent.run(
                        str(item["input"]),
                        str(item["context"]),
                        {"temperature": args.temperature, "top_p": args.top_p, "stream": False},
                    )
            except Exception as error:
                print(f"[error] sample {item['_id']}: {error}")
                return None
            answer = str(item["answers"][0]) if item["answers"] else ""
            prediction = extract_boxed_answer(result.final_response[-300:])
            f1, exact_match, sub_exact_match = _metrics(prediction, answer)
            record = {
                "_id": item["_id"],
                "answer": answer,
                "pred": prediction,
                "judge_f1": f1,
                "judge_em": exact_match,
                "judge_sub_em": sub_exact_match,
                "response": result.final_response,
            }
            for key, value in item.items():
                if key not in {"context", "response"} and key not in record:
                    record[key] = value
            return record

        mode = "w" if args.force else "a"
        with output.open(mode, encoding="utf-8") as file:
            tasks = [asyncio.create_task(process(item)) for item in todo]
            for done, task in enumerate(asyncio.as_completed(tasks), start=1):
                record = await task
                if record is None:
                    errors += 1
                    continue
                completed.append(record)
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()
                if done == 1 or done % 10 == 0:
                    stats = _aggregate(completed)
                    print(f"[{done}/{len(tasks)}] F1={stats['f1'] * 100:.2f} errors={errors}")

    stats = _aggregate(completed if args.force else [*cached_records, *completed])
    print(f"\n=== ruler_hqa [n_docs={args.length}] total={stats['total']} errors={errors} ===")
    for metric in ("f1", "em", "sub_em"):
        print(f"  {metric}: {stats[metric] * 100:.2f}")


def main() -> None:
    host = os.getenv("VLLM_SERVE_HOST", os.getenv("SERVE_HOST", "127.0.0.1"))
    port = os.getenv("VLLM_SERVE_PORT", os.getenv("SERVE_PORT", "8000"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=200)
    parser.add_argument("--data-root", default=os.getenv("DATA_ROOT", "/data/hotpotqa"))
    parser.add_argument("--save-dir", "-s", default="results/ruler_hqa")
    parser.add_argument("--save-file", "-f", default="model")
    parser.add_argument("--model", "-m", required=True)
    parser.add_argument("--tokenizer", "-t", required=True)
    parser.add_argument("--base-url", default=f"http://{host}:{port}/v1")
    parser.add_argument("--api-key", default=os.getenv("SERVE_API_KEY", "EMPTY"))
    parser.add_argument("--api", choices=["recurrent"], default="recurrent")
    parser.add_argument("--n-proc", "-n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--sampling", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.n_proc < 1 or args.sampling < 1:
        parser.error("--n-proc and --sampling must be positive")
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
