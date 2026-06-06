import sys
from pathlib import Path

root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = _remove_citations(text)
    text = _remove_articles(text)
    text = _remove_punctuation(text)
    text = _white_space_fix(text)
    return text


def _remove_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", " ", text)


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def _remove_punctuation(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def _white_space_fix(text: str) -> str:
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0

    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def citation_hit(
    answer: str, supporting_doc_ids: list[str], cited_doc_ids: list[str]
) -> float:
    if not cited_doc_ids:
        return 0.0

    supporting = set(supporting_doc_ids)
    cited = set(cited_doc_ids)

    return float(bool(supporting & cited))
