import re
import string
from collections import Counter
from typing import Any

_PUNCTUATION = set(string.punctuation)


def _last_boxed_only_string(text: str) -> str | None:
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    index = text.rfind("\\boxed")
    if index < 0:
        index = text.rfind("\\fbox")
        if index < 0:
            return None
    open_braces = 0
    for cursor in range(index, len(text)):
        if text[cursor] == "{":
            open_braces += 1
        elif text[cursor] == "}":
            open_braces -= 1
            if open_braces == 0:
                return text[index : cursor + 1]
    return None


def _remove_boxed(value: str) -> str:
    if value.startswith("\\boxed "):
        return value[len("\\boxed ") :]
    for prefix in ("\\boxed{", "\\fbox{"):
        if value.startswith(prefix) and value.endswith("}"):
            return value[len(prefix) : -1]
    return ""


def extract_boxed_answer(text: str) -> str:
    boxed = _last_boxed_only_string(text)
    return _remove_boxed(boxed).strip() if boxed is not None else ""


def _normalize(value: str) -> str:
    value = value.lower().replace("\n", "")
    for source, target in (
        ("\\!", ""),
        ("\\\\", "\\"),
        ("tfrac", "frac"),
        ("dfrac", "frac"),
        ("\\left", ""),
        ("\\right", ""),
        ("^{\\circ}", ""),
        ("^\\circ", ""),
        ("\\$", ""),
        ("\\%", ""),
    ):
        value = value.replace(source, target)
    return value.replace(" ", "")


def _answers(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("answers", ground_truth.get("ground_truth", []))
    if isinstance(ground_truth, str):
        return [ground_truth]
    if isinstance(ground_truth, (list, tuple)):
        return [str(value) for value in ground_truth]
    return [] if ground_truth is None else [str(ground_truth)]


def _qa_normalize(value: str) -> str:
    value = "".join(character for character in value.lower() if character not in _PUNCTUATION)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _qa_metrics(prediction: str, answer: str) -> tuple[float, float, float]:
    prediction_tokens = _qa_normalize(prediction).split()
    answer_tokens = _qa_normalize(answer).split()
    exact_match = float(prediction_tokens == answer_tokens)
    sub_exact_match = float(
        bool(prediction_tokens)
        and bool(answer_tokens)
        and (
            " ".join(answer_tokens) in " ".join(prediction_tokens)
            or " ".join(prediction_tokens) in " ".join(answer_tokens)
        )
    )
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if not prediction_tokens or not answer_tokens or overlap == 0:
        return 0.0, exact_match, sub_exact_match
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall), exact_match, sub_exact_match


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    **kwargs,
) -> dict[str, float]:
    """Return VIME-compatible binary accuracy for the final boxed answer."""
    prediction = extract_boxed_answer((solution_str or "")[-300:])
    num_turns = int(extra_info.get("num_turns", 0) or 0)
    answers = _answers(ground_truth)
    accuracy = float(bool(prediction) and any(_normalize(prediction) == _normalize(answer) for answer in answers))
    score = accuracy / num_turns if num_turns > 0 else accuracy
    qa_metrics = [_qa_metrics(prediction, answer) for answer in answers]
    return {
        "score": score,
        "acc": score,
        "f1": max((metrics[0] for metrics in qa_metrics), default=0.0),
        "em": max((metrics[1] for metrics in qa_metrics), default=0.0),
        "sub_em": max((metrics[2] for metrics in qa_metrics), default=0.0),
    }
