"""Canonical compression-aware objective J_B."""
from __future__ import annotations


def retained_text(example, selected):
    grouped = {}
    for row in selected:
        grouped.setdefault(str(row["context_id"]), []).append(row)
    blocks = []
    for context in example["contexts"]:
        rows = sorted(grouped.get(str(context["id"]), []), key=lambda row: int(row["sent_order"]))
        if rows:
            blocks.append(f"Title: {context['title']}\n" + " ".join(str(row["text"]) for row in rows))
    return "\n\n".join(blocks)


def retained_question_prefix(text):
    return f"Context:\n{text}\n\nQuestion:\n" if text.strip() else "Question:\n"


def j_b(query, example, selected, nll, base=None):
    if base is None:
        base = float(nll("Question:\n", str(query))["total_nll"])
    conditional = float(nll(retained_question_prefix(retained_text(example, selected)), str(query))["total_nll"])
    return base - conditional

