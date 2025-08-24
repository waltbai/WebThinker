"""Evaluate."""

import re
import string
from collections import Counter
from typing import Dict, List

from langchain_core.messages import SystemMessage
from openai import APIError

from src.webthinker.config import GROUP_KEYS, MAX_OUTPUT_RETRY
from src.webthinker.model import get_evaluation_model
from src.webthinker.prompts import EVALUATE_PROMPT
from src.webthinker.schema import EvaluationOutput


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
    llm_eval: bool = False,
) -> Dict[str, float]:
    """Evaluate."""
    # Prepare lists
    questions = [
        result["question"]
        for result in sorted(results, key=lambda x: x["id"])
    ]
    label = [
        normalize_qa_answer(result["label"])
        for result in sorted(results, key=lambda x: x["id"])
    ]
    pred = [
        normalize_qa_answer(result["pred"])
        for result in sorted(results, key=lambda x: x["id"])
    ]
    group = [
        result["group"]
        for result in sorted(results, key=lambda x: x["id"])
    ]

    # Handle basic evaluation
    results = {
        "exact_match": exact_match(label, pred, group),
        "accuracy": accuracy(label, pred, group),
        "f1": f1_score(label, pred, group),
    }

    # Handle llm evaluation
    if llm_eval:
        results["llm_score"] = llm_score(questions, label, pred, group)

    return results


def exact_match(
    label: List[str],
    pred: List[str],
    group: List[str] = None,
):
    """Calculate exact match."""
    # Overall
    scores = [
        1 if l == p else 0 for l, p in zip(label, pred)
    ]

    # Group
    group_scores = {}
    for g, s in zip(group, scores):
        group_scores.setdefault(g, []).append(s)

    return {
        "overall": sum(scores) / len(scores) if scores else 0,
        "group": {
            g: sum(s) / len(s) if s else 0
            for g, s in group_scores.items()
        }
    }


def accuracy(
    label: List[str],
    pred: List[str],
    group: List[str] = None,
):
    """Calculate accuracy."""
    # Overall
    scores = [
        1 if l in p else 0 for l, p in zip(label, pred)
    ]

    # Group
    group_scores = {}
    for g, s in zip(group, scores):
        group_scores.setdefault(g, []).append(s)

    return {
        "overall": sum(scores) / len(scores) if scores else 0,
        "group": {
            g: sum(s) / len(s) if s else 0
            for g, s in group_scores.items()
        }
    }


def f1_score(
    label: List[str],
    pred: List[str],
    group: List[str] = None,
):
    """Calculate F1 score."""
    # Overall
    scores = [
        token_f1(l, p) for l, p in zip(label, pred)
    ]

    # Group
    group_scores = {}
    for g, s in zip(group, scores):
        group_scores.setdefault(g, []).append(s)

    return {
        "overall": sum(scores) / len(scores) if scores else 0,
        "group": {
            g: sum(s) / len(s) if s else 0
            for g, s in group_scores.items()
        }
    }


def llm_score(
    questions: List[str],
    label: List[str],
    pred: List[str],
    group: List[str] = None,
):
    """Calculate LLM equivalence score."""
    # Overall
    model = get_evaluation_model().with_structured_output(
        EvaluationOutput,
    ).with_retry(
        stop_after_attempt=MAX_OUTPUT_RETRY,
    )
    scores = []
    for q, l, p in zip(questions, label, pred):
        try:
            content = EVALUATE_PROMPT.format(
                research_question=q,
                labeled_answer=l,
                predicted_answer=p,
            )
            response = model.invoke(
                [SystemMessage(content=content)],
            )
            result = response["justification"]
        except (APIError, TypeError):
            result = "Incorrect"

        if result == "Correct":
            scores.append(1)
        else:
            scores.append(0)

    # Overall
    group_scores = {}
    for g, s in zip(group, scores):
        group_scores.setdefault(g, []).append(s)

    return {
        "overall": sum(scores) / len(scores) if scores else 0,
        "group": {
            g: sum(s) / len(s) if s else 0
            for g, s in group_scores.items()
        }
    }
