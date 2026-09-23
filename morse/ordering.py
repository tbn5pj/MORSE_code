"""Reverse evidence-first context ordering."""
from __future__ import annotations


def reverse_scores(query, contexts, nll):
    base = float(nll("Question:\n", str(query))["total_nll"])
    return {str(row["id"]): base - float(nll(f"Context:\n{row['text']}\n\nQuestion:\n", str(query))["total_nll"])
            for row in contexts}


def reverse_order(query, contexts, nll):
    scores = reverse_scores(query, contexts, nll)
    return sorted((str(row["id"]) for row in contexts),
                  key=lambda cid: (-scores[cid], next(int(row["doc_order"]) for row in contexts if str(row["id"]) == cid)))

