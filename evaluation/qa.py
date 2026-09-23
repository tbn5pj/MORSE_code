"""Official-style normalized answer F1 and exact match."""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(value):
    value = str(value).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_metrics(prediction, answer):
    predicted, gold = normalize_answer(prediction), normalize_answer(answer)
    exact = float(predicted == gold)
    left, right = predicted.split(), gold.split()
    overlap = sum((Counter(left) & Counter(right)).values())
    f1 = 0.0 if not overlap else 2 * overlap / (len(left) + len(right))
    return {"f1": f1, "em": exact}

