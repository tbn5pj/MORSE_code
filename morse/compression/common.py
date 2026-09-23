"""Shared sentence identity and chain-QMI serialization."""
from __future__ import annotations


def build_units(example, tokenizer):
    result = []
    for doc_order, context in enumerate(example["contexts"]):
        for sent_id, sentence in enumerate(context["sentences"]):
            sentence = str(sentence).strip()
            if sentence:
                result.append({"uid": f"{context['id']}::s{sent_id}", "context_id": str(context["id"]),
                    "title": str(context["title"]), "sent_id": sent_id, "text": sentence,
                    "doc_order": doc_order, "sent_order": sent_id,
                    "prev_sents_same_doc": [str(value).strip() for value in context["sentences"][:sent_id]
                                             if str(value).strip()],
                    "num_tokens": len(tokenizer(sentence, add_special_tokens=False).input_ids)})
    return result


def previous_contexts_prefix(example, unit, order):
    by_id = {str(row["id"]): row for row in example["contexts"]}
    positions = {str(cid): index for index, cid in enumerate(order)}
    current = str(unit["context_id"])
    blocks = [by_id[cid]["text"] for cid in order[:positions[current]]]
    prior = unit["prev_sents_same_doc"]
    blocks.append(f"Title: {unit['title']}\n" + (" ".join(prior) if prior else ""))
    return "\n\n".join(blocks)


def chain_scores(example, units, order, nll):
    scores = {}
    for unit in units:
        history, target = previous_contexts_prefix(example, unit, order), str(unit["text"])
        no_question = nll(f"Previous context:\n{history}\n\nSentence:\n", target)
        with_question = nll(f"Question: {example['question']}\nPrevious context:\n{history}\n\nSentence:\n", target)
        scores[str(unit["uid"])] = float(no_question["total_nll"] - with_question["total_nll"])
    return scores

