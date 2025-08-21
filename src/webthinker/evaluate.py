"""Evaluate."""

from collections import Counter
import re
import string
from typing import Dict, List

from src.webthinker.config import GROUP_KEYS


def identify_group(
    task: Dict[str, str],
    default: str = "default",
    group_keys: List[str] = None,
) -> str:
    """Identify group of the task."""
    group_keys = group_keys or GROUP_KEYS
    for key in task:
        if key.lower() in group_keys:
            return task[key]
    return default


###################
# Evaluate QA tasks
###################
def normalize_qa_answer(answer: str) -> str:
    """Normalize the answer for QA tasks."""
    # Remove article (a, an, the)
    answer = re.sub(r"\b(a|an|the)\b", " ", answer)
    # Remove extra spaces
    answer = " ".join(answer.strip().split())
    # Remove punctuation
    answer = "".join(c for c in answer if c not in string.punctuation)
    # Lowercase
    answer = answer.lower()
    return answer


def token_f1(label: str, pred: str) -> float:
    """Calculate token-level F1."""
    pred_tokens = pred.split()
    label_tokens = label.split()
    common = Counter(pred_tokens) & Counter(label_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_qa(
    results: List[Dict[str, str]],
) -> Dict[str, float]:
    """Evaluate."""
    label = [
        normalize_qa_answer(result["label"])
        for result in sorted(results, key=lambda x: x["id"])
    ]
    pred = [
        normalize_qa_answer(result["pred"])
        for result in sorted(results, key=lambda x: x["id"])
    ]
    exact_match = sum(
        1 for l, p in zip(label, pred) if l == p
    ) / len(label)
    accuracy = sum(
        1 for l, p in zip(label, pred) if l in p
    ) / len(label)
    f1 = sum(
        token_f1(l, p) for l, p in zip(label, pred)
    ) / len(label)
    return {
        "exact_match": exact_match,
        "accuracy": accuracy,
        "f1": f1,
    }


def evaluate_qa_group(
    results: List[Dict[str, str]],
):
    """Evaluate QA task by group."""
    label = [
        (normalize_qa_answer(result["label"]), result["group"])
        for result in sorted(results, key=lambda x: x["id"])
    ]
    pred = [
        (normalize_qa_answer(result["pred"]), result["group"])
        for result in sorted(results, key=lambda x: x["id"])
    ]
    exact_match = {}
    for (l, g), (p, _) in zip(label, pred):
        exact_match.setdefault(g, []).append(1 if l == p else 0)
    accuracy = {}
    for (l, g), (p, _) in zip(label, pred):
        accuracy.setdefault(g, []).append(1 if l in p else 0)
    f1 = {}
    for (l, g), (p, _) in zip(label, pred):
        f1.setdefault(g, []).append(token_f1(l, p))
    return {
        "exact_match": {k: sum(v) / len(v) for k, v in exact_match.items()},
        "accuracy": {k: sum(v) / len(v) for k, v in accuracy.items()},
        "f1": {k: sum(v) / len(v) for k, v in f1.items()},
    }
