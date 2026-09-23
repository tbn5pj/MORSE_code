"""Paper-matched one-pass sequential sentence-level chain-QMI compressor."""
from __future__ import annotations

from .common import build_units, chain_scores
from ..objective import j_b
from ..presentation import canonical


def select_by_score(units, scores, budget):
    ordered = sorted(units, key=lambda row: float(scores[str(row["uid"])]), reverse=True)
    selected, used = [], 0
    for row in ordered:
        tokens = max(1, int(row["num_tokens"]))
        if used + tokens <= int(budget):
            selected.append(row)
            used += tokens
    return selected, used


def compress_one_candidate(example, order, budget, tokenizer, nll):
    units = build_units(example, tokenizer)
    scores = chain_scores(example, units, order, nll)
    selected, used = select_by_score(units, scores, budget)
    selected = canonical(selected)
    return {"order": list(order), "retained_units": selected,
            "retained_unit_ids": [str(row["uid"]) for row in selected], "used_tokens": used,
            "j_b": j_b(example["question"], example, selected, nll), "chain_scores": scores}


def compress_one_candidate_many(example, order, budgets, tokenizer, nll):
    """Evaluate several strict budgets while sharing the frozen chain scores."""
    units = build_units(example, tokenizer)
    scores = chain_scores(example, units, order, nll)
    return {str(name): _materialize(example, order, units, scores, budget, nll)
            for name, budget in budgets.items()}


def _materialize(example, order, units, scores, budget, nll):
    selected, used = select_by_score(units, scores, budget)
    selected = canonical(selected)
    return {"order": list(order), "retained_units": selected,
            "retained_unit_ids": [str(row["uid"]) for row in selected], "used_tokens": used,
            "j_b": j_b(example["question"], example, selected, nll)}
