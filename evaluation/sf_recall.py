"""Supporting-fact recall from exact title/sentence identities."""


def supporting_fact_recall(retained_units, gold_pairs):
    kept = {(str(row["title"]), int(row["sent_id"])) for row in retained_units}
    gold = {(str(title), int(index)) for title, index in gold_pairs}
    return len(kept & gold) / len(gold) if gold else 0.0

