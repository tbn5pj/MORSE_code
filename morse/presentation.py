"""Paper-matched retained-unit presentation policies."""
from __future__ import annotations


def canonical(selected):
    return sorted(selected, key=lambda row: (int(row["doc_order"]), int(row["sent_order"])))


def reconstruct_canonical(example, selected):
    selected_ids = {str(row["uid"]) for row in selected}
    blocks, order = [], []
    for context in sorted(example["contexts"], key=lambda row: int(row["doc_order"])):
        sentences = [str(sentence).strip() for i, sentence in enumerate(context["sentences"])
                     if f"{context['id']}::s{i}" in selected_ids]
        if sentences:
            blocks.append(f"Title: {context['title']}\n" + " ".join(sentences))
            order.append(str(context["id"]))
    return "\n\n".join(blocks), order


def reconstruct_postcompression_reverse(example, selected, nll):
    selected_ids = {str(row["uid"]) for row in selected}
    contexts = []
    for context in sorted(example["contexts"], key=lambda row: int(row["doc_order"])):
        sentences = [str(sentence).strip() for index, sentence in enumerate(context["sentences"])
                     if f"{context['id']}::s{index}" in selected_ids]
        if sentences:
            contexts.append({"id": str(context["id"]), "doc_order": int(context["doc_order"]),
                             "text": f"Title: {context['title']}\n" + " ".join(sentences)})
    base = float(nll("Question:\n", str(example["question"]))["total_nll"])
    for row in contexts:
        conditional = float(nll(f"Context:\n{row['text']}\n\nQuestion:\n",
                                str(example["question"]))["total_nll"])
        row["score"] = base - conditional
    ranked = sorted(contexts, key=lambda row: (-row["score"], row["doc_order"]))
    return "\n\n".join(row["text"] for row in ranked), [row["id"] for row in ranked]
