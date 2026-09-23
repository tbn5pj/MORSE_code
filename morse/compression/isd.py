"""Exact no-cache iterative sentence deletion compressor."""
from __future__ import annotations

import math

from .common import build_units
from ..objective import j_b
from ..presentation import canonical
from ..utils import sha256_json


def _maps(units):
    value = {str(row["uid"]): row for row in units}
    if len(value) != len(units):
        raise ValueError("unit IDs must be unique")
    return value


def _flatten(example, units, order):
    ids = [str(row["id"]) for row in example["contexts"]]
    if len(order) != len(ids) or set(order) != set(ids):
        raise ValueError("candidate order is not a full permutation")
    grouped = {}
    for row in units:
        grouped.setdefault(str(row["context_id"]), []).append(row)
    return [str(row["uid"]) for cid in order
            for row in sorted(grouped.get(str(cid), []), key=lambda value: int(value["sent_order"]))]


def _history(example, unit_map, surviving, position):
    target = unit_map[str(surviving[position])]
    target_context, sequence, grouped = str(target["context_id"]), [], {}
    for uid in surviving[:position]:
        row, cid = unit_map[str(uid)], str(unit_map[str(uid)]["context_id"])
        if cid not in grouped:
            sequence.append(cid)
            grouped[cid] = []
        grouped[cid].append(row)
    titles = {str(row["id"]): str(row["title"]) for row in example["contexts"]}
    blocks = []
    for cid in sequence:
        if cid != target_context:
            rows = sorted(grouped[cid], key=lambda row: int(row["sent_order"]))
            if rows:
                blocks.append(f"Title: {titles[cid]}\n" + " ".join(str(row["text"]) for row in rows))
    current = sorted(grouped.get(target_context, []), key=lambda row: int(row["sent_order"]))
    blocks.append(f"Title: {target['title']}\n" + (" ".join(str(row["text"]) for row in current) if current else ""))
    return "\n\n".join(blocks)


def _score(example, unit_map, surviving, position, nll):
    row, history = unit_map[str(surviving[position])], _history(example, unit_map, surviving, position)
    target = str(row["text"])
    no_question = nll(f"Previous context:\n{history}\n\nSentence:\n", target)
    with_question = nll(f"Question: {example['question']}\nPrevious context:\n{history}\n\nSentence:\n", target)
    return float(no_question["total_nll"] - with_question["total_nll"])


def compress_isd_candidate(example, order, budget, tokenizer, nll):
    units, deletions = build_units(example, tokenizer), []
    unit_map = _maps(units)
    surviving = _flatten(example, units, order)
    used = sum(max(1, int(unit_map[uid]["num_tokens"])) for uid in surviving)
    scores = ([_score(example, unit_map, surviving, index, nll) for index in range(len(surviving))]
              if used > int(budget) else [])
    while used > int(budget) and surviving:
        if any(not math.isfinite(value) for value in scores):
            raise RuntimeError("nonfinite ISD score")
        position = min(range(len(scores)), key=lambda index: (scores[index], -index))
        uid = surviving[position]
        used -= max(1, int(unit_map[uid]["num_tokens"]))
        surviving.pop(position)
        scores = scores[:position] + [_score(example, unit_map, surviving, i, nll)
                                      for i in range(position, len(surviving))]
        deletions.append({"iteration": len(deletions) + 1, "deleted_uid": uid,
                          "deleted_position": position, "used_tokens_after": used})
    selected = canonical([unit_map[uid] for uid in surviving])
    trace = {"context_order": list(order), "deletions": deletions,
             "retained_uids_compression_order": list(surviving),
             "tie_break": "minimum_score_then_latest_current_sequence_position"}
    trace["trajectory_hash"] = sha256_json({"context_order": list(order),
        "deletions": [row["deleted_uid"] for row in deletions], "retained": list(surviving)})
    return {"order": list(order), "retained_units": selected,
            "retained_unit_ids": [str(row["uid"]) for row in selected], "used_tokens": used,
            "j_b": j_b(example["question"], example, selected, nll), "trajectory": trace}
