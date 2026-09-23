class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        class Value:
            input_ids = [sum(map(ord, token)) % 97 + 1 for token in str(text).split()]
        return Value()


class DeterministicNLL:
    def __call__(self, prefix, target):
        tokens = str(target).split()
        # Deterministic, request-sensitive scalar stand-in for structural tests.
        total = sum(ord(value) for value in str(prefix) + "|" + str(target)) % 1009 / 17.0
        return {"total_nll": total, "avg_nll": total / max(1, len(tokens)), "num_tokens": len(tokens)}


def tiny_example():
    contexts = [
        {"id": "c0", "title": "Zero", "sentences": ["alpha evidence", "second fact"],
         "text": "Title: Zero\nalpha evidence second fact", "doc_order": 0},
        {"id": "c1", "title": "One", "sentences": ["beta distractor", "another line"],
         "text": "Title: One\nbeta distractor another line", "doc_order": 1},
        {"id": "c2", "title": "Two", "sentences": ["gamma clue", "last sentence"],
         "text": "Title: Two\ngamma clue last sentence", "doc_order": 2},
    ]
    return {"example_id": "tiny-001", "question": "where is alpha", "contexts": contexts,
            "gold_pairs": [["Zero", 0]]}

